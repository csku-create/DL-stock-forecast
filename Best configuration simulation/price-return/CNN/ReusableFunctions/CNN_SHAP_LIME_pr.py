# Imports
import os
import sys
import time
import psutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from thop import profile
import shap
from lime.lime_tabular import LimeTabularExplainer

# Project Setup
project_root = "/content"

if project_root not in sys.path:
    sys.path.append(project_root)

from ReusableFunctions.DataPreprocessing import DataPreprocessing
from ReusableFunctions.EvaluationMetrics import EvaluationMetrics as EM
from reproducibility_settings import set_global_seed

# Reproducibility
set_global_seed( seed=42, framework="torch" )

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("GPU Name:",torch.cuda.get_device_name(0))
else:
    print("Running on CPU")

# Utility Functions
def sync_device():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def get_parameter_count(model):
    total_params = sum( p.numel() for p in model.parameters() )
    trainable_params = sum( p.numel() for p in model.parameters() if p.requires_grad )
    return total_params, trainable_params

def get_computational_requirement( model, input_shape, device ):
    dummy_input = torch.randn( 1, input_shape[0], input_shape[1], device=device)

    # Preserve model training/evaluation state
    previous_training_state = model.training   

    model.eval()
    with torch.no_grad():
        macs, _ = profile( model, inputs=(dummy_input,), verbose=False )

    if previous_training_state:
        model.train()

    flops =  2 * macs
    return macs, flops

# CNN Model
class CNNModel(nn.Module):
    def __init__( self, input_shape, filters, kernel_size, pooling_size, dropout_rate, activation ):
        super().__init__()

        # Activation
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "leaky_relu":
            self.activation = nn.LeakyReLU( negative_slope=0.01 )
        else:
            raise ValueError( "activation must be 'relu' or 'leaky_relu'" )

        # Padding
        padding = ( (kernel_size - 1) // 2 )

        # CNN layers
        self.conv1 = nn.Conv1d(
            in_channels=input_shape[0],
            out_channels=filters,
            kernel_size=kernel_size,
            padding=padding
        )
        self.pool = nn.MaxPool1d(kernel_size=pooling_size)
        self.dropout = nn.Dropout(dropout_rate)

        # Determine flattened dimension automatically
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_shape[0], input_shape[1] )
            dummy_output = self.conv1(dummy_input)
            dummy_output = self.activation(dummy_output)
            dummy_output = self.pool(dummy_output)
            flattened_size = ( dummy_output.flatten(start_dim=1).shape[1])
      
        # Regression output
        self.fc = nn.Linear( flattened_size, 1 )

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.flatten( start_dim=1 )
        x = self.fc(x)
        return x



# Explainable AI (SHAP + LIME)
class OriginalScaleCNN(nn.Module):
    """
    Wraps the trained CNN so SHAP sees predictions in the original target scale.
    MinMaxScaler inverse transform is affine:
        y_original = (y_scaled - min_) / scale_
    """
    def __init__( self, base_model, scaler_y ):
        super().__init__()
        self.base_model = base_model
        self.register_buffer(
            "inverse_scale",
            torch.tensor( 1.0 / scaler_y.scale_[0], dtype=torch.float32 )
        )
        self.register_buffer(
            "inverse_offset",
            torch.tensor( -scaler_y.min_[0] / scaler_y.scale_[0], dtype=torch.float32 )
        )

    def forward( self, x ):
        return self.base_model( x ) * self.inverse_scale + self.inverse_offset


def _safe_spearman( a, b ):
    """Spearman correlation without requiring scipy."""
    a = pd.Series( np.asarray(a) ).rank( method="average" ).to_numpy()
    b = pd.Series( np.asarray(b) ).rank( method="average" ).to_numpy()

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return float( np.corrcoef( a, b )[0, 1] )


def _normalize_importance( values ):
    values = np.asarray( values, dtype=float )
    total = np.sum( values )
    if total <= 0 or not np.isfinite(total):
        return np.zeros_like( values )
    return values / total


def _extract_shap_array( shap_values_raw ):
    """
    Normalize SHAP output to:
        (samples, features, time)
    for a single-output regression model.
    """
    if isinstance( shap_values_raw, list ):
        shap_values_raw = shap_values_raw[0]

    values = np.asarray( shap_values_raw )

    # Some SHAP versions append an output dimension of size 1.
    if values.ndim == 4 and values.shape[-1] == 1:
        values = values[..., 0]

    if values.ndim != 3:
        raise ValueError(
            "Unexpected SHAP shape. Expected (samples, features, time), "
            f"received {values.shape}."
        )

    return values


def run_shap_lime_analysis(
    model,
    X_train,
    X_test,
    scaler_y,
    selected_features,
    window_size,
    ticker,
    device,
    shap_background_size=100,
    shap_explain_size=20,
    lime_explain_size=20,
    lime_num_samples=2000,
    random_state=42
):
    """
    Explain a CNN forecasting model using comparable SHAP and LIME analyses.

    Input arrays:
        X_train/X_test: (samples, time, features)

    CNN input:
        (samples, features, time)

    Key methodology:
      * SHAP and LIME explain the SAME test observations.
      * SHAP uses GradientExplainer on the native CNN tensor.
      * LIME uses all flattened lag-feature inputs in each local surrogate;
        no omitted variables are silently treated as zero.
      * Both explainers return predictions on the original target scale.
      * Global importance is aggregated across the same samples and lags.
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    if X_train.ndim != 3 or X_test.ndim != 3:
        raise ValueError(
            "XAI expects X_train and X_test with shape "
            "(samples, time, features). "
            f"Received {X_train.shape} and {X_test.shape}."
        )

    n_train, train_window, n_features_from_x = X_train.shape
    n_test, test_window, n_test_features = X_test.shape
    n_features = len(selected_features)

    if n_train == 0 or n_test == 0:
        raise ValueError("XAI requires non-empty training and test datasets.")

    if train_window != window_size or test_window != window_size:
        raise ValueError(
            f"window_size={window_size} does not match X shapes "
            f"{X_train.shape} and {X_test.shape}."
        )

    if n_features_from_x != n_features or n_test_features != n_features:
        raise ValueError(
            f"selected_features contains {n_features} names, but "
            f"X_train/X_test contain {n_features_from_x}/{n_test_features} features."
        )

    xai_dir = f"{ticker}_CNN_XAI"
    os.makedirs(xai_dir, exist_ok=True)

    model.eval()

    # ============================================================
    # Representative samples
    # ============================================================
    background_count = min(shap_background_size, n_train)

    # Use exactly the same test observations for SHAP and LIME.
    common_count = min(
        shap_explain_size,
        lime_explain_size,
        n_test
    )

    if common_count < 1:
        raise ValueError("At least one test sample must be explained.")

    background_indices = np.linspace(
        0, n_train - 1, background_count, dtype=int
    )

    if common_count == 1:
        explain_indices = np.array([0], dtype=int)
    else:
        explain_indices = np.linspace(
            0, n_test - 1, common_count, dtype=int
        )

    # Defensive uniqueness for very small datasets.
    explain_indices = np.unique(explain_indices)
    common_count = len(explain_indices)

    background_tensor = torch.tensor(
        X_train[background_indices],
        dtype=torch.float32,
        device=device
    ).permute(0, 2, 1)

    shap_test_tensor = torch.tensor(
        X_test[explain_indices],
        dtype=torch.float32,
        device=device
    ).permute(0, 2, 1)

    # Differentiable inverse-target-scaling wrapper.
    explainable_model = OriginalScaleCNN(
        base_model=model,
        scaler_y=scaler_y
    ).to(device)
    explainable_model.eval()

    # ============================================================
    # SHAP
    # ============================================================
    print("\n=== SHAP Analysis ===")

    with torch.enable_grad():
        shap_explainer = shap.GradientExplainer(
            explainable_model,
            background_tensor
        )
        shap_values_raw = shap_explainer.shap_values(
            shap_test_tensor,
            rseed=random_state
        )

    shap_values = _extract_shap_array(shap_values_raw)

    expected_shap_shape = (
        common_count,
        n_features,
        window_size
    )
    if shap_values.shape != expected_shap_shape:
        raise ValueError(
            "Unexpected SHAP shape after normalization. "
            f"Expected {expected_shap_shape}, got {shap_values.shape}."
        )

    shap_abs = np.abs(shap_values)

    # Global feature importance: mean |SHAP| over samples and lags.
    shap_feature_importance = shap_abs.mean(axis=(0, 2))

    shap_feature_df = pd.DataFrame({
        "Feature": selected_features,
        "SHAP Mean Abs Importance": shap_feature_importance
    }).sort_values(
        "SHAP Mean Abs Importance",
        ascending=False
    ).reset_index(drop=True)

    shap_feature_df["SHAP Normalized Importance"] = _normalize_importance(
        shap_feature_df["SHAP Mean Abs Importance"].to_numpy()
    )
    shap_feature_df["SHAP Rank"] = np.arange(
        1, len(shap_feature_df) + 1
    )

    shap_feature_df.to_csv(
        os.path.join(xai_dir, "shap_global_feature_importance.csv"),
        index=False
    )

    print("\nSHAP Global Feature Importance:")
    print(shap_feature_df.to_string(index=False))

    # Lag importance: mean |SHAP| over samples and features.
    shap_lag_importance = shap_abs.mean(axis=(0, 1))

    lag_labels = [
        f"t-{window_size - 1 - i}"
        for i in range(window_size)
    ]

    shap_lag_df = pd.DataFrame({
        "Lag": lag_labels,
        "SHAP Mean Abs Importance": shap_lag_importance
    }).sort_values(
        "SHAP Mean Abs Importance",
        ascending=False
    ).reset_index(drop=True)

    shap_lag_df["SHAP Normalized Importance"] = _normalize_importance(
        shap_lag_df["SHAP Mean Abs Importance"].to_numpy()
    )
    shap_lag_df["SHAP Rank"] = np.arange(
        1, len(shap_lag_df) + 1
    )

    shap_lag_df.to_csv(
        os.path.join(xai_dir, "shap_lag_importance.csv"),
        index=False
    )

    # Mean over samples -> (features, time), transpose -> (time, features).
    shap_time_feature = shap_abs.mean(axis=0).T

    shap_time_feature_df = pd.DataFrame(
        shap_time_feature,
        index=lag_labels,
        columns=selected_features
    )
    shap_time_feature_df.index.name = "Lag"
    shap_time_feature_df.to_csv(
        os.path.join(xai_dir, "shap_time_feature_importance.csv")
    )

    shap_long_rows = []
    for time_idx, lag in enumerate(lag_labels):
        for feature_idx, feature in enumerate(selected_features):
            shap_long_rows.append({
                "Lag": lag,
                "Feature": feature,
                "SHAP Mean Abs Importance":
                    float(shap_time_feature[time_idx, feature_idx])
            })

    shap_long_df = pd.DataFrame(shap_long_rows).sort_values(
        "SHAP Mean Abs Importance",
        ascending=False
    )
    shap_long_df.to_csv(
        os.path.join(xai_dir, "shap_time_feature_heatmap.csv"),
        index=False
    )

    # Local SHAP explanation for first shared test sample.
    local_test_index = int(explain_indices[0])
    local_shap = shap_values[0]          # (features, time)
    local_input = X_test[local_test_index]  # (time, features)

    with torch.inference_mode():
        local_prediction = explainable_model(
            torch.tensor(
                local_input,
                dtype=torch.float32,
                device=device
            ).T.unsqueeze(0)
        ).item()

    local_rows = []
    for time_idx, lag in enumerate(lag_labels):
        for feature_idx, feature in enumerate(selected_features):
            local_rows.append({
                "Test Sample Index": local_test_index,
                "Lag": lag,
                "Feature": feature,
                "Scaled Input Value":
                    float(local_input[time_idx, feature_idx]),
                "SHAP Value":
                    float(local_shap[feature_idx, time_idx]),
                "Abs SHAP Value":
                    float(abs(local_shap[feature_idx, time_idx])),
                "Prediction": float(local_prediction)
            })

    shap_local_df = pd.DataFrame(local_rows).sort_values(
        "Abs SHAP Value",
        ascending=False
    )
    shap_local_df.to_csv(
        os.path.join(xai_dir, "shap_local_explanation.csv"),
        index=False
    )

    print(
        f"\nSHAP local explanation generated for test sample "
        f"{local_test_index}; prediction={local_prediction:.6f}"
    )
    print(
        shap_local_df[
            ["Lag", "Feature", "SHAP Value", "Abs SHAP Value"]
        ].head(10).to_string(index=False)
    )

    # SHAP plots.
    plt.figure(figsize=(10, 5))
    plot_df = shap_feature_df.sort_values(
        "SHAP Mean Abs Importance",
        ascending=True
    )
    plt.barh(
        plot_df["Feature"],
        plot_df["SHAP Mean Abs Importance"]
    )
    plt.xlabel("Mean |SHAP value| (target units)")
    plt.ylabel("Feature")
    plt.title("CNN SHAP Global Feature Importance")
    plt.tight_layout()
    plt.savefig(
        os.path.join(xai_dir, "shap_global_feature_importance.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(10, 5))
    plot_lag_df = shap_lag_df.sort_values(
        "SHAP Mean Abs Importance",
        ascending=True
    )
    plt.barh(
        plot_lag_df["Lag"],
        plot_lag_df["SHAP Mean Abs Importance"]
    )
    plt.xlabel("Mean |SHAP value| (target units)")
    plt.ylabel("Lag")
    plt.title("CNN SHAP Lag Importance")
    plt.tight_layout()
    plt.savefig(
        os.path.join(xai_dir, "shap_lag_importance.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(10, 6))
    heat = plt.imshow(
        shap_time_feature,
        aspect="auto"
    )
    plt.colorbar(
        heat,
        label="Mean |SHAP value| (target units)"
    )
    plt.xticks(
        np.arange(n_features),
        selected_features,
        rotation=45,
        ha="right"
    )
    plt.yticks(
        np.arange(window_size),
        lag_labels
    )
    plt.xlabel("Feature")
    plt.ylabel("Lag")
    plt.title("CNN SHAP Time × Feature Importance")
    plt.tight_layout()
    plt.savefig(
        os.path.join(xai_dir, "shap_time_feature_heatmap.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    local_plot_df = shap_local_df.head(15).copy()
    local_plot_df["Term"] = (
        local_plot_df["Feature"]
        + "["
        + local_plot_df["Lag"]
        + "]"
    )
    local_plot_df = local_plot_df.sort_values(
        "SHAP Value",
        ascending=True
    )

    plt.figure(figsize=(10, 7))
    plt.barh(
        local_plot_df["Term"],
        local_plot_df["SHAP Value"]
    )
    plt.axvline(0.0, linewidth=1)
    plt.xlabel("SHAP value (target units)")
    plt.ylabel("Feature × Lag")
    plt.title(
        f"CNN Local SHAP Explanation — Test Sample {local_test_index}"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(xai_dir, "shap_local_explanation.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # ============================================================
    # LIME
    # ============================================================
    print("\n=== LIME Analysis ===")

    X_train_lime = X_train.reshape(n_train, -1)
    X_test_lime = X_test.reshape(n_test, -1)

    lime_feature_names = []
    lime_feature_meta = []

    # Flatten order for (time, feature) is time-major.
    for time_idx, lag in enumerate(lag_labels):
        for feature_idx, feature in enumerate(selected_features):
            lime_feature_names.append(f"{feature}[{lag}]")
            lime_feature_meta.append(
                (time_idx, feature_idx, lag, feature)
            )

    n_flat_features = len(lime_feature_names)

    def lime_predict(flat_array):
        flat_array = np.asarray(flat_array, dtype=np.float32)

        if flat_array.ndim == 1:
            flat_array = flat_array.reshape(1, -1)

        if flat_array.shape[1] != n_flat_features:
            raise ValueError(
                f"LIME predictor expected {n_flat_features} flattened inputs, "
                f"received {flat_array.shape[1]}."
            )

        sequence_array = flat_array.reshape(
            -1, window_size, n_features
        )
        sequence_tensor = torch.tensor(
            sequence_array,
            dtype=torch.float32,
            device=device
        ).permute(0, 2, 1)

        model.eval()
        with torch.inference_mode():
            prediction_scaled = model(
                sequence_tensor
            ).detach().cpu().numpy().reshape(-1, 1)

        return scaler_y.inverse_transform(
            prediction_scaled
        ).reshape(-1)

    # feature_selection="none" is essential for the global comparison:
    # every flattened lag-feature variable receives a local coefficient.
    lime_explainer = LimeTabularExplainer(
        training_data=X_train_lime,
        feature_names=lime_feature_names,
        mode="regression",
        discretize_continuous=False,
        feature_selection="none",
        random_state=random_state
    )

    lime_local_rows = []
    lime_fidelity_rows = []

    # NaN initialization makes incomplete coefficient maps detectable.
    lime_weight_matrix = np.full(
        (common_count, n_flat_features),
        np.nan,
        dtype=float
    )

    for local_i, test_index in enumerate(explain_indices):
        explanation = lime_explainer.explain_instance(
            X_test_lime[test_index],
            lime_predict,
            num_features=n_flat_features,
            num_samples=lime_num_samples
        )

        if not explanation.local_exp:
            raise RuntimeError(
                f"LIME returned no local coefficients for "
                f"test sample {test_index}."
            )

        # Regression versions can expose different keys; use the map with
        # the greatest coefficient coverage.
        map_key = max(
            explanation.local_exp,
            key=lambda key: len(explanation.local_exp[key])
        )
        feature_weights = explanation.local_exp[map_key]

        if len(feature_weights) != n_flat_features:
            raise RuntimeError(
                "LIME returned an incomplete local coefficient vector: "
                f"{len(feature_weights)} of {n_flat_features} terms."
            )

        for flat_index, weight in feature_weights:
            flat_index = int(flat_index)
            if flat_index < 0 or flat_index >= n_flat_features:
                raise RuntimeError(
                    f"LIME returned invalid flattened feature index "
                    f"{flat_index}."
                )
            lime_weight_matrix[
                local_i, flat_index
            ] = float(weight)

        if np.isnan(lime_weight_matrix[local_i]).any():
            missing = int(
                np.isnan(lime_weight_matrix[local_i]).sum()
            )
            raise RuntimeError(
                f"LIME coefficient map is incomplete for test sample "
                f"{test_index}: {missing} terms are missing."
            )

        prediction = float(
            lime_predict(
                X_test_lime[test_index:test_index + 1]
            )[0]
        )
        local_prediction_lime = float(
            np.ravel(explanation.local_pred)[0]
        )

        lime_fidelity_rows.append({
            "Test Sample Index": int(test_index),
            "Prediction": prediction,
            "LIME Local Prediction": local_prediction_lime,
            "LIME Fidelity R2": float(explanation.score)
        })

        for flat_index, weight in enumerate(
            lime_weight_matrix[local_i]
        ):
            time_idx, feature_idx, lag, feature = (
                lime_feature_meta[flat_index]
            )
            lime_local_rows.append({
                "Test Sample Index": int(test_index),
                "Lag": lag,
                "Feature": feature,
                "Term": lime_feature_names[flat_index],
                "Scaled Input Value":
                    float(X_test[
                        test_index,
                        time_idx,
                        feature_idx
                    ]),
                "LIME Weight": float(weight),
                "Abs LIME Weight": float(abs(weight)),
                "Prediction": prediction,
                "LIME Local Prediction": local_prediction_lime,
                "LIME Fidelity R2":
                    float(explanation.score)
            })

    lime_local_df = pd.DataFrame(lime_local_rows)
    lime_fidelity_df = pd.DataFrame(lime_fidelity_rows)

    lime_local_df.to_csv(
        os.path.join(xai_dir, "lime_local_explanations.csv"),
        index=False
    )
    lime_fidelity_df.to_csv(
        os.path.join(xai_dir, "lime_fidelity.csv"),
        index=False
    )

    # Mean local coefficient over the SAME explained test samples.
    lime_abs_mean = np.mean(
        np.abs(lime_weight_matrix),
        axis=0
    )
    lime_signed_mean = np.mean(
        lime_weight_matrix,
        axis=0
    )

    # Aggregate by original feature.
    lime_feature_importance = np.zeros(
        n_features,
        dtype=float
    )
    lime_feature_signed = np.zeros(
        n_features,
        dtype=float
    )

    for flat_index, (
        time_idx,
        feature_idx,
        lag,
        feature
    ) in enumerate(lime_feature_meta):
        lime_feature_importance[feature_idx] += (
            lime_abs_mean[flat_index]
        )
        lime_feature_signed[feature_idx] += (
            lime_signed_mean[flat_index]
        )

    # Mean across lags.
    lime_feature_importance /= window_size
    lime_feature_signed /= window_size

    lime_feature_df = pd.DataFrame({
        "Feature": selected_features,
        "LIME Mean Abs Importance":
            lime_feature_importance,
        "LIME Mean Signed Weight":
            lime_feature_signed
    }).sort_values(
        "LIME Mean Abs Importance",
        ascending=False
    ).reset_index(drop=True)

    lime_feature_df["LIME Normalized Importance"] = _normalize_importance(
        lime_feature_df[
            "LIME Mean Abs Importance"
        ].to_numpy()
    )
    lime_feature_df["LIME Rank"] = np.arange(
        1, len(lime_feature_df) + 1
    )

    lime_feature_df.to_csv(
        os.path.join(xai_dir, "lime_global_feature_importance.csv"),
        index=False
    )

    print("\nLIME Global Feature Importance:")
    print(lime_feature_df.to_string(index=False))

    # Aggregate by lag.
    lime_lag_importance = np.zeros(
        window_size,
        dtype=float
    )

    for flat_index, (
        time_idx,
        feature_idx,
        lag,
        feature
    ) in enumerate(lime_feature_meta):
        lime_lag_importance[time_idx] += (
            lime_abs_mean[flat_index]
        )

    lime_lag_importance /= n_features

    lime_lag_df = pd.DataFrame({
        "Lag": lag_labels,
        "LIME Mean Abs Importance":
            lime_lag_importance
    }).sort_values(
        "LIME Mean Abs Importance",
        ascending=False
    ).reset_index(drop=True)

    lime_lag_df["LIME Normalized Importance"] = _normalize_importance(
        lime_lag_df[
            "LIME Mean Abs Importance"
        ].to_numpy()
    )
    lime_lag_df["LIME Rank"] = np.arange(
        1, len(lime_lag_df) + 1
    )

    lime_lag_df.to_csv(
        os.path.join(xai_dir, "lime_lag_importance.csv"),
        index=False
    )

    # Time x feature table.
    lime_time_feature = lime_abs_mean.reshape(
        window_size,
        n_features
    )
    lime_time_feature_df = pd.DataFrame(
        lime_time_feature,
        index=lag_labels,
        columns=selected_features
    )
    lime_time_feature_df.index.name = "Lag"
    lime_time_feature_df.to_csv(
        os.path.join(xai_dir, "lime_time_feature_importance.csv")
    )

    plt.figure(figsize=(10, 5))
    lime_plot_df = lime_feature_df.sort_values(
        "LIME Mean Abs Importance",
        ascending=True
    )
    plt.barh(
        lime_plot_df["Feature"],
        lime_plot_df["LIME Mean Abs Importance"]
    )
    plt.xlabel(
        "Mean |LIME coefficient| "
        "(target units / scaled-input unit)"
    )
    plt.ylabel("Feature")
    plt.title("CNN LIME Global Feature Importance")
    plt.tight_layout()
    plt.savefig(
        os.path.join(xai_dir, "lime_global_feature_importance.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(10, 5))
    lime_lag_plot_df = lime_lag_df.sort_values(
        "LIME Mean Abs Importance",
        ascending=True
    )
    plt.barh(
        lime_lag_plot_df["Lag"],
        lime_lag_plot_df["LIME Mean Abs Importance"]
    )
    plt.xlabel(
        "Mean |LIME coefficient| "
        "(target units / scaled-input unit)"
    )
    plt.ylabel("Lag")
    plt.title("CNN LIME Lag Importance")
    plt.tight_layout()
    plt.savefig(
        os.path.join(xai_dir, "lime_lag_importance.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(10, 6))
    lime_heat = plt.imshow(
        lime_time_feature,
        aspect="auto"
    )
    plt.colorbar(
        lime_heat,
        label="Mean |LIME coefficient|"
    )
    plt.xticks(
        np.arange(n_features),
        selected_features,
        rotation=45,
        ha="right"
    )
    plt.yticks(
        np.arange(window_size),
        lag_labels
    )
    plt.xlabel("Feature")
    plt.ylabel("Lag")
    plt.title("CNN LIME Time × Feature Importance")
    plt.tight_layout()
    plt.savefig(
        os.path.join(xai_dir, "lime_time_feature_heatmap.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # ============================================================
    # SHAP vs LIME comparison
    # ============================================================
    print("\n=== SHAP vs LIME Comparison ===")

    comparison_df = pd.DataFrame({
        "Feature": selected_features,
        "SHAP Importance": shap_feature_importance,
        "LIME Importance": lime_feature_importance
    })

    comparison_df["SHAP Normalized"] = _normalize_importance(
        comparison_df["SHAP Importance"].to_numpy()
    )
    comparison_df["LIME Normalized"] = _normalize_importance(
        comparison_df["LIME Importance"].to_numpy()
    )

    # Average ranks preserve ties rather than forcing arbitrary integer ranks.
    comparison_df["SHAP Rank"] = comparison_df[
        "SHAP Importance"
    ].rank(
        ascending=False,
        method="average"
    )

    comparison_df["LIME Rank"] = comparison_df[
        "LIME Importance"
    ].rank(
        ascending=False,
        method="average"
    )

    comparison_df["Rank Difference"] = (
        comparison_df["SHAP Rank"]
        - comparison_df["LIME Rank"]
    ).abs()

    # Compare rankings rather than raw magnitudes. SHAP attribution values
    # and LIME surrogate coefficients are not the same mathematical quantity.
    spearman_correlation = _safe_spearman(
        comparison_df["SHAP Importance"],
        comparison_df["LIME Importance"]
    )

    comparison_df["SHAP-LIME Spearman"] = (
        spearman_correlation
    )
    comparison_df = comparison_df.sort_values(
        "SHAP Rank"
    ).reset_index(drop=True)

    comparison_df.to_csv(
        os.path.join(
            xai_dir,
            "shap_lime_feature_comparison.csv"
        ),
        index=False
    )

    print(comparison_df.to_string(index=False))
    print(
        f"\nSHAP-LIME Spearman rank correlation: "
        f"{spearman_correlation:.6f}"
    )

    mean_lime_fidelity = (
        float(
            lime_fidelity_df[
                "LIME Fidelity R2"
            ].mean()
        )
        if not lime_fidelity_df.empty
        else np.nan
    )

    top_shap_feature = str(
        shap_feature_df.iloc[0]["Feature"]
    )
    top_lime_feature = str(
        lime_feature_df.iloc[0]["Feature"]
    )
    top_shap_lag = str(
        shap_lag_df.iloc[0]["Lag"]
    )
    top_lime_lag = str(
        lime_lag_df.iloc[0]["Lag"]
    )

    xai_summary = {
        "SHAP Top Feature": top_shap_feature,
        "SHAP Top Feature Importance": float(
            shap_feature_df.iloc[0][
                "SHAP Mean Abs Importance"
            ]
        ),
        "SHAP Top Lag": top_shap_lag,
        "LIME Top Feature": top_lime_feature,
        "LIME Top Feature Importance": float(
            lime_feature_df.iloc[0][
                "LIME Mean Abs Importance"
            ]
        ),
        "LIME Top Lag": top_lime_lag,
        "LIME Mean Fidelity R2": mean_lime_fidelity,
        "SHAP-LIME Spearman": spearman_correlation,
        "SHAP Explained Samples": int(common_count),
        "LIME Explained Samples": int(common_count)
    }

    pd.DataFrame(
        [xai_summary]
    ).to_csv(
        os.path.join(xai_dir, "xai_summary.csv"),
        index=False
    )

    print("\n=== XAI Summary ===")
    for key, value in xai_summary.items():
        print(f"{key}: {value}")

    print(
        f"\nCommon SHAP/LIME test samples explained: "
        f"{common_count}"
    )
    print(
        f"SHAP/LIME outputs saved to: {xai_dir}"
    )

    return {
        "summary": xai_summary,
        "shap_feature_importance": shap_feature_df,
        "shap_lag_importance": shap_lag_df,
        "lime_feature_importance": lime_feature_df,
        "lime_lag_importance": lime_lag_df,
        "lime_fidelity": lime_fidelity_df,
        "comparison": comparison_df
    }


# Metric Function
#y_true: actual next day value; y_pred: predicted next day value
def compute_metrics( eval_types, y_true, y_pred, current_close, forecast_window, market, initial_capital, position_fraction ):
    y_true = np.asarray( y_true ).reshape(-1)
    y_pred = np.asarray( y_pred ).reshape(-1)
    current_close = np.asarray( current_close ).reshape(-1)
    
    r2 = EM.r2( y_true, y_pred )
   
    # Invalid R2 protection
    if np.isnan(r2) or np.isinf(r2):
        return tuple([np.nan] * 124)
        
    rmse = EM.rmse( y_true, y_pred )
    mae = EM.mae( y_true, y_pred )
    if eval_types==1:
        accuracy = EM.accuracy( y_true, y_pred )
    else:
        accuracy = EM.accuracy_change( y_true, y_pred )
    profit_index,cop0,cos0 = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=forecast_window, market=market, PI_types=1, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.0 ) #100share per trade, sell at T+N+bearish
    profit_index1,cop1,cos1 = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=forecast_window, market=market, PI_types=2, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.0 ) #100share per trade, sell at T+N
    profit_index2,cop2,cos2 = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=forecast_window, market=market, PI_types=3, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.0 ) #%captical fraction per trade per trade, sell at T+N+bearish
    profit_index3,cop3,cos3 = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=forecast_window, market=market, PI_types=4, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.0 ) #%captical fraction per trade per trade, sell at T+N
    profit_index4,cop4,cos4 = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=forecast_window, market=market, PI_types=5, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.0 ) #%captical fraction per trade per trade, sell at T+N+bearish, include costs
    profit_index5,cop5,cos5 = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=forecast_window, market=market, PI_types=6, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.01 ) #%captical fraction per trade per trade, sell at T+N+bearish+threshold=0.01

    profit_index0a,cop0a,cos0a = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=5, market=market, PI_types=1, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.0 ) #100share per trade, sell at T+N+bearish
    profit_index0b,cop0b,cos0b = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=10, market=market, PI_types=1, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.0 ) #100share per trade, sell at T+N+bearish
    profit_index0c,cop0c,cos0c = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=20, market=market, PI_types=1, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.0 ) #100share per trade, sell at T+N+bearish
    profit_index0d,cop0d,cos0d = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=30, market=market, PI_types=1, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.0 ) #100share per trade, sell at T+N+bearish
    
    profit_index1a,cop1a,cos1a = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=5, market=market, PI_types=2, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.0 ) #100share per trade, sell at T+N
    profit_index1b,cop1b,cos1b = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=10, market=market, PI_types=2, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.0 ) #100share per trade, sell at T+N
    profit_index1c,cop1c,cos1c = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=20, market=market, PI_types=2, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.0 ) #100share per trade, sell at T+N
    profit_index1d,cop1d,cos1d = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=30, market=market, PI_types=2, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.0 ) #100share per trade, sell at T+N

    profit_index2a,cop2a,cos2a = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=5, market=market, PI_types=3, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.0 ) #%captical fraction per trade per trade, sell at T+N+bearish
    profit_index2b,cop2b,cos2b = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=10, market=market, PI_types=3, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.0 ) #%captical fraction per trade per trade, sell at T+N+bearish
    profit_index2c,cop2c,cos2c = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=20, market=market, PI_types=3, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.0 ) #%captical fraction per trade per trade, sell at T+N+bearish
    profit_index2d,cop2d,cos2d = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=30, market=market, PI_types=3, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.0 ) #%captical fraction per trade per trade, sell at T+N+bearish
    
    profit_index3a,cop3a,cos3a = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=5, market=market, PI_types=4, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.0 ) #%captical fraction per trade per trade, sell at T+N
    profit_index3b,cop3b,cos3b = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=10, market=market, PI_types=4, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.0 ) #%captical fraction per trade per trade, sell at T+N
    profit_index3c,cop3c,cos3c = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=20, market=market, PI_types=4, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.0 ) #%captical fraction per trade per trade, sell at T+N
    profit_index3d,cop3d,cos3d = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=30, market=market, PI_types=4, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.0 ) #%captical fraction per trade per trade, sell at T+N

    profit_index4a,cop4a,cos4a = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=5, market=market, PI_types=5, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.0 ) #%captical fraction per trade per trade, sell at T+N+bearish, include costs
    profit_index4b,cop4b,cos4b = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=10, market=market, PI_types=5, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.0 ) #%captical fraction per trade per trade, sell at T+N+bearish, include costs
    profit_index4c,cop4c,cos4c = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=20, market=market, PI_types=5, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.0 ) #%captical fraction per trade per trade, sell at T+N+bearish, include costs
    profit_index4d,cop4d,cos4d = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=30, market=market, PI_types=5, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.0 ) #%captical fraction per trade per trade, sell at T+N+bearish, include costs
        
    profit_index5a,cop5a,cos5a = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=5, market=market, PI_types=6, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.01 ) #%captical fraction per trade per trade, sell at T+N+bearish+threshold
    profit_index5b,cop5b,cos5b = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=10, market=market, PI_types=6, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.01 ) #%captical fraction per trade per trade, sell at T+N+bearish+threshold
    profit_index5c,cop5c,cos5c = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=20, market=market, PI_types=6, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.01 ) #%captical fraction per trade per trade, sell at T+N+bearish+threshold
    profit_index5d,cop5d,cos5d = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=30, market=market, PI_types=6, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.01 ) #%captical fraction per trade per trade, sell at T+N+bearish+threshold

    
    profit_index5e,cop5e,cos5e = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=1, market=market, PI_types=6, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.03 ) #%captical fraction per trade per trade, sell at T+N+bearish+threshold
    profit_index5f,cop5f,cos5f = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=5, market=market, PI_types=6, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.03 ) #%captical fraction per trade per trade, sell at T+N+bearish+threshold
    profit_index5g,cop5g,cos5g = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=10, market=market, PI_types=6, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.03 ) #%captical fraction per trade per trade, sell at T+N+bearish+threshold
    profit_index5h,cop5h,cos5h = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=20, market=market, PI_types=6, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.03 ) #%captical fraction per trade per trade, sell at T+N+bearish+threshold
    profit_index5i,cop5i,cos5i = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=30, market=market, PI_types=6, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.03 ) #%captical fraction per trade per trade, sell at T+N+bearish+threshold

    profit_index5j,cop5j,cos5j = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=1, market=market, PI_types=6, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.05 ) #%captical fraction per trade per trade, sell at T+N+bearish+threshold
    profit_index5k,cop5k,cos5k = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=5, market=market, PI_types=6, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.05 ) #%captical fraction per trade per trade, sell at T+N+bearish+threshold
    profit_index5l,cop5l,cos5l = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=10, market=market, PI_types=6, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.05 ) #%captical fraction per trade per trade, sell at T+N+bearish+threshold
    profit_index5m,cop5m,cos5m = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=20, market=market, PI_types=6, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.05 ) #%captical fraction per trade per trade, sell at T+N+bearish+threshold
    profit_index5n,cop5n,cos5n = EM.profitability_index( eval_types, y_pred, current_close, forecast_window=30, market=market, PI_types=6, initial_capital=initial_capital, position_fraction=position_fraction, buy_threshold=0.05 ) #%captical fraction per trade per trade, sell at T+N+bearish+threshold
    return ( rmse, mae, r2, accuracy, profit_index, profit_index1, profit_index2, profit_index3, profit_index4, profit_index5,
           profit_index0a, profit_index0b, profit_index0c, profit_index0d,
           profit_index1a, profit_index1b, profit_index1c, profit_index1d,
             profit_index2a, profit_index2b, profit_index2c, profit_index2d,
             profit_index3a, profit_index3b, profit_index3c, profit_index3d,
             profit_index4a, profit_index4b, profit_index4c, profit_index4d,
           profit_index5a, profit_index5b, profit_index5c, profit_index5d,
           profit_index5e, profit_index5f, profit_index5g, profit_index5h, profit_index5i,
           profit_index5j, profit_index5k, profit_index5l, profit_index5m, profit_index5n,
             cop0, cos0, cop1, cos1, cop2, cos2, cop3, cos3, cop4, cos4, cop5, cos5,
           cop0a, cos0a,cop0b, cos0b, cop0c, cos0c, cop0d, cos0d,
           cop1a, cos1a,cop1b, cos1b, cop1c, cos1c, cop1d, cos1d,
             cop2a, cos2a,cop2b, cos2b, cop2c, cos2c, cop2d, cos2d,
             cop3a, cos3a,cop3b, cos3b, cop3c, cos3c, cop3d, cos3d,
           cop4a, cos4a,cop4b, cos4b, cop4c, cos4c, cop4d, cos4d,
           cop5a, cos5a,cop5b, cos5b, cop5c, cos5c, cop5d, cos5d,
           cop5e, cos5e,cop5f, cos5f, cop5g, cos5g, cop5h, cos5h,
           cop5i, cos5i,cop5j, cos5j, cop5k, cos5k, cop5l, cos5l,
           cop5m, cos5m,cop5n, cos5n)

# Train and Evaluate CNN
def train_and_evaluate_model(
    model,
    train_loader,
    val_data,
    test_data,
    optimizer,
    criterion,
    scaler_y,
    forecast_window,
    epochs,
    market,
    initial_capital,
    position_fraction,
    eval_types,
    patience=0.2
):   
    # Early Stopping Configuration
    if isinstance(patience,float):
        actual_patience = max(1,int(patience * epochs))
    else:
        actual_patience = int(patience)
    
    # Prepare Validation Data
    X_val_tensor = torch.tensor(val_data[0], dtype=torch.float32).permute(0,2,1).to(device)
    y_val_tensor = torch.tensor(val_data[1],dtype=torch.float32).to(device)
    z_val_tensor = torch.tensor( val_data[2], dtype=torch.float32, device=device )
    # Prepare Test Data
    X_test_tensor = torch.tensor(test_data[0], dtype=torch.float32).permute(0,2,1).to(device)
    y_test_tensor = torch.tensor(test_data[1],dtype=torch.float32).to(device)
    z_test_tensor = torch.tensor( test_data[2], dtype=torch.float32, device=device )

    # Reconstruct Full Training Dataset
    full_train_x_list = []
    full_train_y_list = []
    full_train_z_list = []

    for xb, yb, zb in train_loader:
        full_train_x_list.append(xb)
        full_train_y_list.append(yb)
        full_train_z_list.append(zb)

    full_train_x = torch.cat(full_train_x_list,dim=0).to(device)
    full_train_y = torch.cat(full_train_y_list,dim=0).to(device)
    full_train_z = torch.cat( full_train_z_list, dim=0 ).to(device)

    # Model Complexity
    total_params, trainable_params = get_parameter_count( model )
    macs, flops = (
        get_computational_requirement(
            model=model,
            input_shape=( full_train_x.shape[1], full_train_x.shape[2] ),
            device=device ) 
        )
    
    # Memory Measurement Setup
    process = psutil.Process(os.getpid())
    cpu_memory_before =  process.memory_info().rss / (1024 ** 2)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    
    # Training Variables
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None
    optimization_time = 0.0
    epochs_completed = 0
    
    # Total Training Timing   
    sync_device()
    training_start = time.perf_counter()
    
    # Training Loop
    for epoch in range( epochs ):
        # Training Phase
        model.train()
        for xb, yb, zb in train_loader:
            xb = xb.to( device )
            yb = yb.to( device )
            optimizer.zero_grad( set_to_none=True )

            predictions = model( xb ).squeeze( -1 )
            loss = criterion( predictions, yb.squeeze(-1) )
            sync_device()
            optimization_start = time.perf_counter()

            loss.backward()
            optimizer.step()

            sync_device()
            optimization_time += time.perf_counter() - optimization_start

        # Validation Phase
        model.eval()
        with torch.inference_mode():
            val_predictions = model( X_val_tensor ).squeeze( -1 )
            val_loss = criterion( val_predictions, y_val_tensor.squeeze(-1) ).item()

        epochs_completed = epoch + 1

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            epochs_no_improve += 1

            if ( epochs_no_improve >= actual_patience ):
                print( f"Early stopping at " f"epoch {epoch + 1}" )
                break
    
    # End Total Training Timing
    sync_device()
    training_time = time.perf_counter() - training_start
    
    # Restore Best CNN
    if best_state is not None:
        model.load_state_dict( best_state )
        model.to( device )
    
    # Memory Usage
    cpu_memory_after =  process.memory_info().rss / (1024 ** 2)
    cpu_memory_consumption =  cpu_memory_after - cpu_memory_before

    if torch.cuda.is_available():
        peak_gpu_memory =  torch.cuda.max_memory_allocated( device ) / (1024 ** 2)
    else:
        peak_gpu_memory = 0.0
    
    # Inference   
    model.eval()
    
    # Train Inference
    sync_device()
    train_inference_start = time.perf_counter()

    with torch.inference_mode():
        train_output = model( full_train_x )

    sync_device()
    train_inference_time = time.perf_counter() - train_inference_start

    train_preds_scaled =  train_output.squeeze(-1).cpu().numpy()
    train_preds_unscaled =  scaler_y.inverse_transform( train_preds_scaled.reshape( -1, 1 ) )
    y_train_unscaled = scaler_y.inverse_transform( full_train_y.cpu().numpy().reshape( -1, 1 ) )
    
    # Validation Inference
    sync_device()
    val_inference_start = time.perf_counter()   

    with torch.inference_mode():
        val_output = model( X_val_tensor )

    sync_device()
    val_inference_time =  time.perf_counter() - val_inference_start

    val_preds_scaled = ( val_output.squeeze(-1).cpu().numpy())
    val_preds_unscaled =  scaler_y.inverse_transform( val_preds_scaled.reshape( -1, 1 ) )
    y_val_unscaled = scaler_y.inverse_transform( np.asarray( val_data[1] ).reshape( -1, 1 ) )
    
    # Test Inference
    sync_device()
    test_inference_start = time.perf_counter()

    with torch.inference_mode():
        test_output = model( X_test_tensor )

    sync_device()
    test_inference_time = time.perf_counter() - test_inference_start

    test_preds_scaled = test_output.squeeze(-1).cpu().numpy()
    test_preds_unscaled = scaler_y.inverse_transform(test_preds_scaled.reshape( -1, 1 ) )
    y_test_unscaled = scaler_y.inverse_transform( np.asarray( test_data[1] ).reshape( -1, 1 ) )

    if len( X_test_tensor ) > 0:
        test_latency_per_sample =  test_inference_time / len( X_test_tensor )
    else:
        test_latency_per_sample = np.nan
    
    # Predictive Metrics
    train_metrics = compute_metrics(eval_types, y_train_unscaled, train_preds_unscaled, full_train_z.cpu().numpy(), forecast_window, market, initial_capital, position_fraction )
    val_metrics = compute_metrics( eval_types, y_val_unscaled, val_preds_unscaled, z_val_tensor.cpu().numpy(), forecast_window, market, initial_capital, position_fraction  )
    test_metrics = compute_metrics( eval_types, y_test_unscaled, test_preds_unscaled, z_test_tensor.cpu().numpy(), forecast_window, market, initial_capital, position_fraction  )
    
    # Print Predictive Metrics
    print("\n=== Train Metrics ===")
    print(
        f"RMSE: {train_metrics[0]:.4f}, "
        f"MAE: {train_metrics[1]:.4f}, "
        f"R2: {train_metrics[2]:.4f}, "
        f"Acc: {train_metrics[3]:.4f}, "
        f"ProfitIdx: {train_metrics[4]:.4f}, "
        f"ProfitIdx1: {train_metrics[5]:.4f}, "
        f"ProfitIdx2: {train_metrics[6]:.4f}, "
        f"ProfitIdx3: {train_metrics[7]:.4f}, "
        f"ProfitIdx4: {train_metrics[8]:.4f}, "
        f"ProfitIdx5: {train_metrics[9]:.4f}, "
        f"ProfitIdx0a: {train_metrics[10]:.4f}, "
        f"ProfitIdx0b: {train_metrics[11]:.4f}, "
        f"ProfitIdx0c: {train_metrics[12]:.4f}, "
        f"ProfitIdx0d: {train_metrics[13]:.4f}, "
        f"ProfitIdx1a: {train_metrics[14]:.4f}, "
        f"ProfitIdx1b: {train_metrics[15]:.4f}, "
        f"ProfitIdx1c: {train_metrics[16]:.4f}, "
        f"ProfitIdx1d: {train_metrics[17]:.4f}, "
        f"ProfitIdx2a: {train_metrics[18]:.4f}, "
        f"ProfitIdx2b: {train_metrics[19]:.4f}, "
        f"ProfitIdx2c: {train_metrics[20]:.4f}, "
        f"ProfitIdx2d: {train_metrics[21]:.4f}, "
        f"ProfitIdx3a: {train_metrics[22]:.4f}, "
        f"ProfitIdx3b: {train_metrics[23]:.4f}, "
        f"ProfitIdx3c: {train_metrics[24]:.4f}, "
        f"ProfitIdx3d: {train_metrics[25]:.4f}, "
        f"ProfitIdx4a: {train_metrics[26]:.4f}, "
        f"ProfitIdx4b: {train_metrics[27]:.4f}, "
        f"ProfitIdx4c: {train_metrics[28]:.4f}, "
        f"ProfitIdx4d: {train_metrics[29]:.4f}, "
        f"ProfitIdx5a: {train_metrics[30]:.4f}, "
        f"ProfitIdx5b: {train_metrics[31]:.4f}, "
        f"ProfitIdx5c: {train_metrics[32]:.4f}, "
        f"ProfitIdx5d: {train_metrics[33]:.4f}, "
        f"ProfitIdx5e: {train_metrics[34]:.4f}, "
        f"ProfitIdx5f: {train_metrics[35]:.4f}, "
        f"ProfitIdx5g: {train_metrics[36]:.4f}, "
        f"ProfitIdx5h: {train_metrics[37]:.4f}, "
        f"ProfitIdx5i: {train_metrics[38]:.4f}, "
        f"ProfitIdx5j: {train_metrics[39]:.4f}, "
        f"ProfitIdx5k: {train_metrics[40]:.4f}, "
        f"ProfitIdx5l: {train_metrics[41]:.4f}, "
        f"ProfitIdx5m: {train_metrics[42]:.4f}, "
        f"ProfitIdx5n: {train_metrics[43]:.4f}, "  
        f"countp0: {train_metrics[44]:.4f}, "
        f"counts0: {train_metrics[45]:.4f}, "
        f"countp1: {train_metrics[46]:.4f}, "
        f"counts1: {train_metrics[47]:.4f}, "
        f"countp2: {train_metrics[48]:.4f}, "
        f"counts2: {train_metrics[49]:.4f}, "
        f"countp3: {train_metrics[50]:.4f}, "
        f"counts3: {train_metrics[51]:.4f}, "
        f"countp4: {train_metrics[52]:.4f}, "
        f"counts4: {train_metrics[53]:.4f}, "
        f"countp5: {train_metrics[54]:.4f}, "
        f"counts5: {train_metrics[55]:.4f}, "
        f"countp0a: {train_metrics[56]:.4f}, "
        f"counts0a: {train_metrics[57]:.4f}, "
        f"countp0b: {train_metrics[58]:.4f}, "
        f"counts0b: {train_metrics[59]:.4f}, "
        f"countp0c: {train_metrics[60]:.4f}, "
        f"counts0c: {train_metrics[61]:.4f}, "
        f"countp0d: {train_metrics[62]:.4f}, "
        f"counts0d: {train_metrics[63]:.4f}, "
        f"countp1a: {train_metrics[64]:.4f}, "
        f"counts1a: {train_metrics[65]:.4f}, "
        f"countp1b: {train_metrics[66]:.4f}, "
        f"counts1b: {train_metrics[67]:.4f}, "
        f"countp1c: {train_metrics[68]:.4f}, "
        f"counts1c: {train_metrics[69]:.4f}, "
        f"countp1d: {train_metrics[70]:.4f}, "
        f"counts1d: {train_metrics[71]:.4f}, "
        f"countp2a: {train_metrics[72]:.4f}, "
        f"counts2a: {train_metrics[73]:.4f}, "
        f"countp2b: {train_metrics[74]:.4f}, "
        f"counts2b: {train_metrics[75]:.4f}, "
        f"countp2c: {train_metrics[76]:.4f}, "
        f"counts2c: {train_metrics[77]:.4f}, "
        f"countp2d: {train_metrics[78]:.4f}, "
        f"counts2d: {train_metrics[79]:.4f}, "
        f"countp3a: {train_metrics[80]:.4f}, "
        f"counts3a: {train_metrics[81]:.4f}, "
        f"countp3b: {train_metrics[82]:.4f}, "
        f"counts3b: {train_metrics[83]:.4f}, "
        f"countp3c: {train_metrics[84]:.4f}, "
        f"counts3c: {train_metrics[85]:.4f}, "
        f"countp3d: {train_metrics[86]:.4f}, "
        f"counts3d: {train_metrics[87]:.4f}, "
        f"countp4a: {train_metrics[88]:.4f}, "
        f"counts4a: {train_metrics[89]:.4f}, "
        f"countp4b: {train_metrics[90]:.4f}, "
        f"counts4b: {train_metrics[91]:.4f}, "
        f"countp4c: {train_metrics[92]:.4f}, "
        f"counts4c: {train_metrics[93]:.4f}, "
        f"countp4d: {train_metrics[94]:.4f}, "
        f"counts4d: {train_metrics[95]:.4f}, "
        f"countp5a: {train_metrics[96]:.4f}, "
        f"counts5a: {train_metrics[97]:.4f}, "
        f"countp5b: {train_metrics[98]:.4f}, "
        f"counts5b: {train_metrics[99]:.4f}, "
        f"countp5c: {train_metrics[100]:.4f}, "
        f"counts5c: {train_metrics[101]:.4f}, "
        f"countp5d: {train_metrics[102]:.4f}, "
        f"counts5d: {train_metrics[103]:.4f}, "
        f"countp5e: {train_metrics[104]:.4f}, "
        f"counts5e: {train_metrics[105]:.4f}, "
        f"countp5f: {train_metrics[106]:.4f}, "
        f"counts5f: {train_metrics[107]:.4f}, "
        f"countp5g: {train_metrics[108]:.4f}, "
        f"counts5g: {train_metrics[109]:.4f}, "
        f"countp5h: {train_metrics[110]:.4f}, "
        f"counts5h: {train_metrics[111]:.4f}, "
        f"countp5i: {train_metrics[112]:.4f}, "
        f"counts5i: {train_metrics[113]:.4f}, "
        f"countp5j: {train_metrics[114]:.4f}, "
        f"counts5j: {train_metrics[115]:.4f}, "
        f"countp5k: {train_metrics[116]:.4f}, "
        f"counts5k: {train_metrics[117]:.4f}, "
        f"countp5l: {train_metrics[118]:.4f}, "
        f"counts5l: {train_metrics[119]:.4f}, "
        f"countp5m: {train_metrics[120]:.4f}, "
        f"counts5m: {train_metrics[121]:.4f}, "
        f"countp5n: {train_metrics[122]:.4f}, "
        f"counts5n: {train_metrics[123]:.4f} "       
    )
    print("\n=== Validation Metrics ===")
    print(
        f"RMSE: {val_metrics[0]:.4f}, "
        f"MAE: {val_metrics[1]:.4f}, "
        f"R2: {val_metrics[2]:.4f}, "
        f"Acc: {val_metrics[3]:.4f}, "
        f"ProfitIdx: {val_metrics[4]:.4f}, "
        f"ProfitIdx1: {val_metrics[5]:.4f}, "
        f"ProfitIdx2: {val_metrics[6]:.4f}, "
        f"ProfitIdx3: {val_metrics[7]:.4f}, "
        f"ProfitIdx4: {val_metrics[8]:.4f}, "
        f"ProfitIdx5: {val_metrics[9]:.4f}, "
        f"ProfitIdx0a: {val_metrics[10]:.4f}, "
        f"ProfitIdx0b: {val_metrics[11]:.4f}, "
        f"ProfitIdx0c: {val_metrics[12]:.4f}, "
        f"ProfitIdx0d: {val_metrics[13]:.4f}, "
        f"ProfitIdx1a: {val_metrics[14]:.4f}, "
        f"ProfitIdx1b: {val_metrics[15]:.4f}, "
        f"ProfitIdx1c: {val_metrics[16]:.4f}, "
        f"ProfitIdx1d: {val_metrics[17]:.4f}, "
        f"ProfitIdx2a: {val_metrics[18]:.4f}, "
        f"ProfitIdx2b: {val_metrics[19]:.4f}, "
        f"ProfitIdx2c: {val_metrics[20]:.4f}, "
        f"ProfitIdx2d: {val_metrics[21]:.4f}, "
        f"ProfitIdx3a: {val_metrics[22]:.4f}, "
        f"ProfitIdx3b: {val_metrics[23]:.4f}, "
        f"ProfitIdx3c: {val_metrics[24]:.4f}, "
        f"ProfitIdx3d: {val_metrics[25]:.4f}, "
        f"ProfitIdx4a: {val_metrics[26]:.4f}, "
        f"ProfitIdx4b: {val_metrics[27]:.4f}, "
        f"ProfitIdx4c: {val_metrics[28]:.4f}, "
        f"ProfitIdx4d: {val_metrics[29]:.4f}, "
        f"ProfitIdx5a: {val_metrics[30]:.4f}, "
        f"ProfitIdx5b: {val_metrics[31]:.4f}, "
        f"ProfitIdx5c: {val_metrics[32]:.4f}, "
        f"ProfitIdx5d: {val_metrics[33]:.4f}, "
        f"ProfitIdx5e: {val_metrics[34]:.4f}, "
        f"ProfitIdx5f: {val_metrics[35]:.4f}, "
        f"ProfitIdx5g: {val_metrics[36]:.4f}, "
        f"ProfitIdx5h: {val_metrics[37]:.4f}, "
        f"ProfitIdx5i: {val_metrics[38]:.4f}, "
        f"ProfitIdx5j: {val_metrics[39]:.4f}, "
        f"ProfitIdx5k: {val_metrics[40]:.4f}, "
        f"ProfitIdx5l: {val_metrics[41]:.4f}, "
        f"ProfitIdx5m: {val_metrics[42]:.4f}, "
        f"ProfitIdx5n: {val_metrics[43]:.4f}, "
        f"countp0: {val_metrics[44]:.4f}, "
        f"counts0: {val_metrics[45]:.4f}, "
        f"countp1: {val_metrics[46]:.4f}, "
        f"counts1: {val_metrics[47]:.4f}, "
        f"countp2: {val_metrics[48]:.4f}, "
        f"counts2: {val_metrics[49]:.4f}, "
        f"countp3: {val_metrics[50]:.4f}, "
        f"counts3: {val_metrics[51]:.4f}, "
        f"countp4: {val_metrics[52]:.4f}, "
        f"counts4: {val_metrics[53]:.4f}, "
        f"countp5: {val_metrics[54]:.4f}, "
        f"counts5: {val_metrics[55]:.4f}, "
        f"countp0a: {val_metrics[56]:.4f}, "
        f"counts0a: {val_metrics[57]:.4f}, "
        f"countp0b: {val_metrics[58]:.4f}, "
        f"counts0b: {val_metrics[59]:.4f}, "
        f"countp0c: {val_metrics[60]:.4f}, "
        f"counts0c: {val_metrics[61]:.4f}, "
        f"countp0d: {val_metrics[62]:.4f}, "
        f"counts0d: {val_metrics[63]:.4f}, "
        f"countp1a: {val_metrics[64]:.4f}, "
        f"counts1a: {val_metrics[65]:.4f}, "
        f"countp1b: {val_metrics[66]:.4f}, "
        f"counts1b: {val_metrics[67]:.4f}, "
        f"countp1c: {val_metrics[68]:.4f}, "
        f"counts1c: {val_metrics[69]:.4f}, "
        f"countp1d: {val_metrics[70]:.4f}, "
        f"counts1d: {val_metrics[71]:.4f}, "
        f"countp2a: {val_metrics[72]:.4f}, "
        f"counts2a: {val_metrics[73]:.4f}, "
        f"countp2b: {val_metrics[74]:.4f}, "
        f"counts2b: {val_metrics[75]:.4f}, "
        f"countp2c: {val_metrics[76]:.4f}, "
        f"counts2c: {val_metrics[77]:.4f}, "
        f"countp2d: {val_metrics[78]:.4f}, "
        f"counts2d: {val_metrics[79]:.4f}, "
        f"countp3a: {val_metrics[80]:.4f}, "
        f"counts3a: {val_metrics[81]:.4f}, "
        f"countp3b: {val_metrics[82]:.4f}, "
        f"counts3b: {val_metrics[83]:.4f}, "
        f"countp3c: {val_metrics[84]:.4f}, "
        f"counts3c: {val_metrics[85]:.4f}, "
        f"countp3d: {val_metrics[86]:.4f}, "
        f"counts3d: {val_metrics[87]:.4f}, "
        f"countp4a: {val_metrics[88]:.4f}, "
        f"counts4a: {val_metrics[89]:.4f}, "
        f"countp4b: {val_metrics[90]:.4f}, "
        f"counts4b: {val_metrics[91]:.4f}, "
        f"countp4c: {val_metrics[92]:.4f}, "
        f"counts4c: {val_metrics[93]:.4f}, "
        f"countp4d: {val_metrics[94]:.4f}, "
        f"counts4d: {val_metrics[95]:.4f}, "
        f"countp5a: {val_metrics[96]:.4f}, "
        f"counts5a: {val_metrics[97]:.4f}, "
        f"countp5b: {val_metrics[98]:.4f}, "
        f"counts5b: {val_metrics[99]:.4f}, "
        f"countp5c: {val_metrics[100]:.4f}, "
        f"counts5c: {val_metrics[101]:.4f}, "
        f"countp5d: {val_metrics[102]:.4f}, "
        f"counts5d: {val_metrics[103]:.4f}, "
        f"countp5e: {val_metrics[104]:.4f}, "
        f"counts5e: {val_metrics[105]:.4f}, "
        f"countp5f: {val_metrics[106]:.4f}, "
        f"counts5f: {val_metrics[107]:.4f}, "
        f"countp5g: {val_metrics[108]:.4f}, "
        f"counts5g: {val_metrics[109]:.4f}, "
        f"countp5h: {val_metrics[110]:.4f}, "
        f"counts5h: {val_metrics[111]:.4f}, "
        f"countp5i: {val_metrics[112]:.4f}, "
        f"counts5i: {val_metrics[113]:.4f}, "
        f"countp5j: {val_metrics[114]:.4f}, "
        f"counts5j: {val_metrics[115]:.4f}, "
        f"countp5k: {val_metrics[116]:.4f}, "
        f"counts5k: {val_metrics[117]:.4f}, "
        f"countp5l: {val_metrics[118]:.4f}, "
        f"counts5l: {val_metrics[119]:.4f}, "
        f"countp5m: {val_metrics[120]:.4f}, "
        f"counts5m: {val_metrics[121]:.4f}, "
        f"countp5n: {val_metrics[122]:.4f}, "
        f"counts5n: {val_metrics[123]:.4f} "
    )
    print("\n=== Test Metrics ===")
    print(
        f"RMSE: {test_metrics[0]:.4f}, "
        f"MAE: {test_metrics[1]:.4f}, "
        f"R2: {test_metrics[2]:.4f}, "
        f"Acc: {test_metrics[3]:.4f}, "
        f"ProfitIdx: {test_metrics[4]:.4f}, "
        f"ProfitIdx1: {test_metrics[5]:.4f}, "
        f"ProfitIdx2: {test_metrics[6]:.4f}, "
        f"ProfitIdx3: {test_metrics[7]:.4f}, "
        f"ProfitIdx4: {test_metrics[8]:.4f}, "
        f"ProfitIdx5: {test_metrics[9]:.4f}, "
        f"ProfitIdx0a: {test_metrics[10]:.4f}, "
        f"ProfitIdx0b: {test_metrics[11]:.4f}, "
        f"ProfitIdx0c: {test_metrics[12]:.4f}, "
        f"ProfitIdx0d: {test_metrics[13]:.4f}, "
        f"ProfitIdx1a: {test_metrics[14]:.4f}, "
        f"ProfitIdx1b: {test_metrics[15]:.4f}, "
        f"ProfitIdx1c: {test_metrics[16]:.4f}, "
        f"ProfitIdx1d: {test_metrics[17]:.4f}, "
        f"ProfitIdx2a: {test_metrics[18]:.4f}, "
        f"ProfitIdx2b: {test_metrics[19]:.4f}, "
        f"ProfitIdx2c: {test_metrics[20]:.4f}, "
        f"ProfitIdx2d: {test_metrics[21]:.4f}, "
        f"ProfitIdx3a: {test_metrics[22]:.4f}, "
        f"ProfitIdx3b: {test_metrics[23]:.4f}, "
        f"ProfitIdx3c: {test_metrics[24]:.4f}, "
        f"ProfitIdx3d: {test_metrics[25]:.4f}, "
        f"ProfitIdx4a: {test_metrics[26]:.4f}, "
        f"ProfitIdx4b: {test_metrics[27]:.4f}, "
        f"ProfitIdx4c: {test_metrics[28]:.4f}, "
        f"ProfitIdx4d: {test_metrics[29]:.4f}, "
        f"ProfitIdx5a: {test_metrics[30]:.4f}, "
        f"ProfitIdx5b: {test_metrics[31]:.4f}, "
        f"ProfitIdx5c: {test_metrics[32]:.4f}, "
        f"ProfitIdx5d: {test_metrics[33]:.4f}, "
        f"ProfitIdx5e: {test_metrics[34]:.4f}, "
        f"ProfitIdx5f: {test_metrics[35]:.4f}, "
        f"ProfitIdx5g: {test_metrics[36]:.4f}, "
        f"ProfitIdx5h: {test_metrics[37]:.4f}, "
        f"ProfitIdx5i: {test_metrics[38]:.4f}, "
        f"ProfitIdx5j: {test_metrics[39]:.4f}, "
        f"ProfitIdx5k: {test_metrics[40]:.4f}, "
        f"ProfitIdx5l: {test_metrics[41]:.4f}, "
        f"ProfitIdx5m: {test_metrics[42]:.4f}, "
        f"ProfitIdx5n: {test_metrics[43]:.4f}, "
        f"countp0: {test_metrics[44]:.4f}, "
        f"counts0: {test_metrics[45]:.4f}, "
        f"countp1: {test_metrics[46]:.4f}, "
        f"counts1: {test_metrics[47]:.4f}, "
        f"countp2: {test_metrics[48]:.4f}, "
        f"counts2: {test_metrics[49]:.4f}, "
        f"countp3: {test_metrics[50]:.4f}, "
        f"counts3: {test_metrics[51]:.4f}, "
        f"countp4: {test_metrics[52]:.4f}, "
        f"counts4: {test_metrics[53]:.4f}, "
        f"countp5: {test_metrics[54]:.4f}, "
        f"counts5: {test_metrics[55]:.4f}, "
        f"countp0a: {test_metrics[56]:.4f}, "
        f"counts0a: {test_metrics[57]:.4f}, "
        f"countp0b: {test_metrics[58]:.4f}, "
        f"counts0b: {test_metrics[59]:.4f}, "
        f"countp0c: {test_metrics[60]:.4f}, "
        f"counts0c: {test_metrics[61]:.4f}, "
        f"countp0d: {test_metrics[62]:.4f}, "
        f"counts0d: {test_metrics[63]:.4f}, "
        f"countp1a: {test_metrics[64]:.4f}, "
        f"counts1a: {test_metrics[65]:.4f}, "
        f"countp1b: {test_metrics[66]:.4f}, "
        f"counts1b: {test_metrics[67]:.4f}, "
        f"countp1c: {test_metrics[68]:.4f}, "
        f"counts1c: {test_metrics[69]:.4f}, "
        f"countp1d: {test_metrics[70]:.4f}, "
        f"counts1d: {test_metrics[71]:.4f}, "
        f"countp2a: {test_metrics[72]:.4f}, "
        f"counts2a: {test_metrics[73]:.4f}, "
        f"countp2b: {test_metrics[74]:.4f}, "
        f"counts2b: {test_metrics[75]:.4f}, "
        f"countp2c: {test_metrics[76]:.4f}, "
        f"counts2c: {test_metrics[77]:.4f}, "
        f"countp2d: {test_metrics[78]:.4f}, "
        f"counts2d: {test_metrics[79]:.4f}, "
        f"countp3a: {test_metrics[80]:.4f}, "
        f"counts3a: {test_metrics[81]:.4f}, "
        f"countp3b: {test_metrics[82]:.4f}, "
        f"counts3b: {test_metrics[83]:.4f}, "
        f"countp3c: {test_metrics[84]:.4f}, "
        f"counts3c: {test_metrics[85]:.4f}, "
        f"countp3d: {test_metrics[86]:.4f}, "
        f"counts3d: {test_metrics[87]:.4f}, "
        f"countp4a: {test_metrics[88]:.4f}, "
        f"counts4a: {test_metrics[89]:.4f}, "
        f"countp4b: {test_metrics[90]:.4f}, "
        f"counts4b: {test_metrics[91]:.4f}, "
        f"countp4c: {test_metrics[92]:.4f}, "
        f"counts4c: {test_metrics[93]:.4f}, "
        f"countp4d: {test_metrics[94]:.4f}, "
        f"counts4d: {test_metrics[95]:.4f}, "
        f"countp5a: {test_metrics[96]:.4f}, "
        f"counts5a: {test_metrics[97]:.4f}, "
        f"countp5b: {test_metrics[98]:.4f}, "
        f"counts5b: {test_metrics[99]:.4f}, "
        f"countp5c: {test_metrics[100]:.4f}, "
        f"counts5c: {test_metrics[101]:.4f}, "
        f"countp5d: {test_metrics[102]:.4f}, "
        f"counts5d: {test_metrics[103]:.4f}, "
        f"countp5e: {test_metrics[104]:.4f}, "
        f"counts5e: {test_metrics[105]:.4f}, "
        f"countp5f: {test_metrics[106]:.4f}, "
        f"counts5f: {test_metrics[107]:.4f}, "
        f"countp5g: {test_metrics[108]:.4f}, "
        f"counts5g: {test_metrics[109]:.4f}, "
        f"countp5h: {test_metrics[110]:.4f}, "
        f"counts5h: {test_metrics[111]:.4f}, "
        f"countp5i: {test_metrics[112]:.4f}, "
        f"counts5i: {test_metrics[113]:.4f}, "
        f"countp5j: {test_metrics[114]:.4f}, "
        f"counts5j: {test_metrics[115]:.4f}, "
        f"countp5k: {test_metrics[116]:.4f}, "
        f"counts5k: {test_metrics[117]:.4f}, "
        f"countp5l: {test_metrics[118]:.4f}, "
        f"counts5l: {test_metrics[119]:.4f}, "
        f"countp5m: {test_metrics[120]:.4f}, "
        f"counts5m: {test_metrics[121]:.4f}, "
        f"countp5n: {test_metrics[122]:.4f}, "
        f"counts5n: {test_metrics[123]:.4f} " 
    )

    # Print Computational Performance
    print(
        "\n=== Computational Performance ==="
        f"\nEpochs completed               : " f"{epochs_completed}"
        f"\nTraining time                  : " f"{training_time:.6f} seconds"
        f"\nOptimization time              : " f"{optimization_time:.6f} seconds"
        f"\nTrain inference time           : " f"{train_inference_time:.6f} seconds"
        f"\nValidation inference time      : " f"{val_inference_time:.6f} seconds"
        f"\nTest inference time            : " f"{test_inference_time:.6f} seconds"
        f"\nTest latency / sample          : " f"{test_latency_per_sample:.9f} seconds"
        f"\nTotal parameters               : " f"{total_params:,}"
        f"\nTrainable parameters           : " f"{trainable_params:,}"
        f"\nMACs / sample                  : " f"{macs:,.0f}"
        f"\nApprox. FLOPs / sample         : " f"{flops:,.0f}"
        f"\nPeak GPU memory                : " f"{peak_gpu_memory:.2f} MB"
        f"\nCPU memory change              : " f"{cpu_memory_consumption:.2f} MB"
    )
    
    # Validation Plot
    plt.figure(figsize=(12,5))
    plt.plot( y_val_unscaled.reshape(-1), label="Actual (Val)" )
    plt.plot(val_preds_unscaled.reshape(-1), label="Predicted (Val)")
    plt.title("CNN - Validation Set")
    plt.xlabel("Time steps")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Test Plot
    plt.figure( figsize=( 12, 5 ) )
    plt.plot(y_test_unscaled.reshape(-1), label="Actual (Test)")
    plt.plot(test_preds_unscaled.reshape(-1), label="Predicted (Test)" )
    plt.title("CNN - Test Set")
    plt.xlabel("Time steps")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Computational Metrics Dictionary  
    computational_metrics = {
        "training_time_sec": training_time,
        "optimization_time_sec": optimization_time,
        "train_inference_time_sec": train_inference_time,
        "validation_inference_time_sec": val_inference_time,
        "test_inference_time_sec": test_inference_time,
        "test_latency_per_sample_sec": test_latency_per_sample,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "macs_per_sample": macs,
        "flops_per_sample": flops,
        "peak_gpu_memory_mb": peak_gpu_memory,
        "cpu_memory_change_mb": cpu_memory_consumption,
        "epochs_completed": epochs_completed
    }
    return  train_metrics, val_metrics, test_metrics, computational_metrics


def start_experiment(ticker_choice, eval_types):
    model_name = "CNN"
    start_date = "2014-01-01"
    end_date = "2024-12-31"
    patience = 0.2

    experiments = [
        {
            "market": "US",
            "ticker": "AAPL",
            "fixed_indicators": ["20MA", "50MA", "MACD", "Upper_BB", "CCI", "Williams_%R"],
            "window_size": 5,
            "forecast_window": 1,
            "filters": 128,
            "kernel_size": 3,
            "pooling_size": 2,
            "dropout_rate": 0.2,
            "learning_rate": 0.0001,
            "batch_size": 16,
            "requested_epochs": 100,
            "activation": "relu",
        },
        {
            "market": "US",
            "ticker": "QCOM",
            "fixed_indicators": ["20MA", "50MA", "RSI", "Lower_BB", "CCI", "Williams_%R"],
            "window_size": 5,
            "forecast_window": 1,
            "filters": 128,
            "kernel_size": 2,
            "pooling_size": 2,
            "dropout_rate": 0.3,
            "learning_rate": 0.0005,
            "batch_size": 16,
            "requested_epochs": 50,
            "activation": "relu",
        },
        {
            "market": "US",
            "ticker": "BNY",
            "fixed_indicators": ["50MA", "MACD", "Upper_BB", "Lower_BB", "CCI", "Williams_%R"],
            "window_size": 60,
            "forecast_window": 1,
            "filters": 64,
            "kernel_size": 3,
            "pooling_size": 3,
            "dropout_rate": 0.2,
            "learning_rate": 0.0005,
            "batch_size": 16,
            "requested_epochs": 150,
            "activation": "leaky_relu",
        },
        {
            "market": "US",
            "ticker": "C",
            "fixed_indicators": ["50MA", "RSI", "MACD", "Lower_BB", "CCI", "ATR"],
            "window_size": 5,
            "forecast_window": 1,
            "filters": 128,
            "kernel_size": 2,
            "pooling_size": 2,
            "dropout_rate": 0.3,
            "learning_rate": 0.0005,
            "batch_size": 64,
            "requested_epochs": 150,
            "activation": "leaky_relu",
        },
        {
            "market": "KL",
            "ticker": "0097.KL",
            "fixed_indicators": ["20MA", "RSI", "MACD", "CCI", "ATR", "OBV"],
            "window_size": 5,
            "forecast_window": 1,
            "filters": 128,
            "kernel_size": 2,
            "pooling_size": 2,
            "dropout_rate": 0.2,
            "learning_rate": 0.0001,
            "batch_size": 16,
            "requested_epochs": 100,
            "activation": "relu",
        },
        {
            "market": "KL",
            "ticker": "0166.KL",
            "fixed_indicators": ["20MA", "MACD", "Lower_BB", "CCI", "ATR", "Williams_%R"],
            "window_size": 5,
            "forecast_window": 1,
            "filters": 64,
            "kernel_size": 3,
            "pooling_size": 2,
            "dropout_rate": 0.2,
            "learning_rate": 0.0001,
            "batch_size": 16,
            "requested_epochs": 150,
            "activation": "leaky_relu",
        },
        {
            "market": "KL",
            "ticker": "1023.KL",
            "fixed_indicators": ["MACD", "Upper_BB", "Lower_BB", "CCI", "ATR", "OBV"],
            "window_size": 5,
            "forecast_window": 1,
            "filters": 64,
            "kernel_size": 2,
            "pooling_size": 2,
            "dropout_rate": 0.2,
            "learning_rate": 0.0005,
            "batch_size": 64,
            "requested_epochs": 100,
            "activation": "leaky_relu",
        },
        {
            "market": "KL",
            "ticker": "5258.KL",
            "fixed_indicators": ["50MA", "RSI", "MACD", "Lower_BB", "CCI", "Williams_%R"],
            "window_size": 5,
            "forecast_window": 1,
            "filters": 128,
            "kernel_size": 2,
            "pooling_size": 2,
            "dropout_rate": 0.2,
            "learning_rate": 0.0005,
            "batch_size": 32,
            "requested_epochs": 100,
            "activation": "leaky_relu",
        },
    ]

    for experiment_index, config in enumerate(experiments, start=1):
        if experiment_index == ticker_choice:
            market = config["market"]
            ticker = config["ticker"]
            fixed_indicators = config["fixed_indicators"]
            window_size = config["window_size"]
            forecast_window = config["forecast_window"]
            filters = config["filters"]
            kernel_size = config["kernel_size"]
            pooling_size = config["pooling_size"]
            dropout_rate = config["dropout_rate"]
            learning_rate = config["learning_rate"]
            batch_size = config["batch_size"]
            requested_epochs = config["requested_epochs"]
            activation = config["activation"]
            # Reset RNG independently for every ticker
            set_global_seed(seed=42, framework="torch")
    
            print("\n" + "=" * 100)
            print(f"EXPERIMENT {experiment_index}/{len(experiments)}: {ticker}")
            print("=" * 100)
            # Load and Process Data
            data_processor = DataPreprocessing( ticker=ticker, start_date=start_date, end_date=end_date )
            df = data_processor.add_technical_indicators()
            selected_features =  ["Close"] + fixed_indicators 
    
            if eval_types==1:
                historical_input_windows, next_price, current_price, _, _ = data_processor.create_windowed_data( df[selected_features].values, window_size, forecast_window ) 
                X_train, X_val, X_test, y_train, y_val, y_test, z_train, z_val, z_test = data_processor.split_dataset( historical_input_windows, next_price, current_price, train_size=0.7, val_size=0.1 ) #X: historical_input_windows; y: next_price; z: current_price
            elif eval_types==2:
                historical_input_windows, _, current_price, next_price_change, _ = data_processor.create_windowed_data( df[selected_features].values, window_size, forecast_window ) 
                X_train, X_val, X_test, y_train, y_val, y_test, z_train, z_val, z_test = data_processor.split_dataset( historical_input_windows, next_price_change, current_price, train_size=0.7, val_size=0.1 ) #X: historical_input_windows; y: next_price; z: current_price
            else:
                historical_input_windows, _, current_price, _, next_price_return = data_processor.create_windowed_data( df[selected_features].values, window_size, forecast_window ) 
                X_train, X_val, X_test, y_train, y_val, y_test, z_train, z_val, z_test = data_processor.split_dataset( historical_input_windows, next_price_return, current_price, train_size=0.7, val_size=0.1 ) #X: historical_input_windows; y: next_price; z: current_price
    
            scaler = MinMaxScaler()
            train_shape = X_train.shape
            X_train_flat =  X_train.reshape( -1, train_shape[2] ) 
    
            # Fit scaler ONLY on training data
            scaler.fit( X_train_flat )
            X_train = scaler.transform( X_train_flat ).reshape( train_shape ) 
    
            X_val =  scaler.transform( X_val.reshape( -1, train_shape[2] ) ).reshape( X_val.shape ) 
            X_test =  scaler.transform( X_test.reshape( -1, train_shape[2] ) ).reshape( X_test.shape ) 
    
            scaler_y = MinMaxScaler() 
            y_train_scaled =  scaler_y.fit_transform( y_train.reshape( -1, 1 ) ) 
            y_val_scaled = scaler_y.transform( y_val.reshape( -1, 1 ) ) 
            y_test_scaled =  scaler_y.transform( y_test.reshape( -1, 1 ) ) 
            X_train_tensor =  torch.tensor( X_train, dtype=torch.float32 ).permute( 0, 2, 1 ) 
            y_train_tensor =  torch.tensor( y_train_scaled, dtype=torch.float32 )
            z_train_tensor =  torch.tensor( z_train, dtype=torch.float32 )
            
            train_dataset = TensorDataset( X_train_tensor, y_train_tensor, z_train_tensor )
            train_loader = DataLoader( train_dataset, batch_size=batch_size, shuffle=False )
    
            # CNN Model    
            model = CNNModel(
                input_shape=( X_train.shape[2], X_train.shape[1] ),
                filters=filters,
                kernel_size=kernel_size,
                pooling_size=pooling_size,
                dropout_rate=dropout_rate,
                activation=activation
            ).to( device )
    
    
    
            # Display Architecture
            print( "\n=== CNN Model Architecture ===" )
            print( model )
    
            # Optimizer and Loss
            optimizer = optim.Adam( model.parameters(), lr=learning_rate )
            criterion = ( nn.MSELoss() )
    
            # Experiment Information   
            print(
                "\nRunning CNN experiment:"
                f"\nIndicators       : " f"{fixed_indicators}"
                f"\nFilters          : " f"{filters}"
                f"\nKernel Size      : " f"{kernel_size}"
                f"\nPooling Size     : " f"{pooling_size}"
                f"\nActivation       : " f"{activation}"
                f"\nDropout          : " f"{dropout_rate}"
                f"\nLearning Rate    : " f"{learning_rate}"
                f"\nBatch Size       : " f"{batch_size}"
                f"\nWindow Size      : " f"{window_size}"
                f"\nForecast Window  : " f"{forecast_window}"
                f"\nRequested Epochs : " f"{requested_epochs}\n"
            )
    
            # Train and Evaluate   
            train_metrics, val_metrics, test_metrics, computational_metrics = train_and_evaluate_model(
                model=model,
                train_loader=train_loader,
                val_data=( X_val, y_val_scaled, z_val ),
                test_data=( X_test, y_test_scaled, z_test ),
                optimizer=optimizer,
                criterion=criterion,
                scaler_y=scaler_y,
                forecast_window=forecast_window,
                epochs=requested_epochs,
                market=market,
                initial_capital=100000,
                position_fraction=0.1,
                eval_types=eval_types,
                patience=patience
            )
    
            # SHAP + LIME Explainability
            # Note:
            #   X_train/X_test here are still in (samples, time, features) format.
            #   run_shap_lime_analysis converts them to CNN format internally.
            xai_results = run_shap_lime_analysis(
                model=model,
                X_train=X_train,
                X_test=X_test,
                scaler_y=scaler_y,
                selected_features=selected_features,
                window_size=window_size,
                ticker=ticker,
                device=device,
                shap_background_size=100,
                shap_explain_size=20,
                lime_explain_size=20,
                lime_num_samples=2000,
                random_state=42
            )
            xai_summary = xai_results["summary"]
    
            # Final Experiment Report   
            result_row = {
                # Model Identification
                "Ticker": ticker,
                "Model":  model_name,
    
                # Shared / Control Parameters
                "Indicators": ", ".join( fixed_indicators ),
                "Dropout": dropout_rate,
                "LR": learning_rate,
                "Batch Size": batch_size,
                "Window Size": window_size,
                "Forecast Window": forecast_window,
                "Requested Epochs": requested_epochs,
                "Actual Epochs": computational_metrics[ "epochs_completed" ],
    
                # Explainability Summary
                "SHAP Top Feature": xai_summary["SHAP Top Feature"],
                "SHAP Top Feature Importance": xai_summary["SHAP Top Feature Importance"],
                "SHAP Top Lag": xai_summary["SHAP Top Lag"],
                "LIME Top Feature": xai_summary["LIME Top Feature"],
                "LIME Top Feature Importance": xai_summary["LIME Top Feature Importance"],
                "LIME Top Lag": xai_summary["LIME Top Lag"],
                "LIME Mean Fidelity R2": xai_summary["LIME Mean Fidelity R2"],
                "SHAP-LIME Spearman": xai_summary["SHAP-LIME Spearman"],
                "SHAP Explained Samples": xai_summary["SHAP Explained Samples"],
                "LIME Explained Samples": xai_summary["LIME Explained Samples"],
    
    
                # CNN-specific Architecture
                "Filters": filters,
                "Kernel Size": kernel_size,
                "Pooling Size": pooling_size,
                "Activation": activation,
    
                # Train Metrics
                "train RMSE": train_metrics[0],
                "train MAE": train_metrics[1],
                "train R2": train_metrics[2],
                "train Accuracy": train_metrics[3],
                "train Profit Index": train_metrics[4],
                "train Profit Index 1": train_metrics[5],
                "train Profit Index 2": train_metrics[6],
                "train Profit Index 3": train_metrics[7],
                "train Profit Index 4": train_metrics[8],
                "train Profit Index 5": train_metrics[9],
                "train Profit Index 0a": train_metrics[10],
                "train Profit Index 0b": train_metrics[11],
                "train Profit Index 0c": train_metrics[12],
                "train Profit Index 0d": train_metrics[13],
                "train Profit Index 1a": train_metrics[14],
                "train Profit Index 1b": train_metrics[15],
                "train Profit Index 1c": train_metrics[16],
                "train Profit Index 1d": train_metrics[17],
                "train Profit Index 2a": train_metrics[18],
                "train Profit Index 2b": train_metrics[19],
                "train Profit Index 2c": train_metrics[20],
                "train Profit Index 2d": train_metrics[21],
                "train Profit Index 3a": train_metrics[22],
                "train Profit Index 3b": train_metrics[23],
                "train Profit Index 3c": train_metrics[24],
                "train Profit Index 3d": train_metrics[25],
                "train Profit Index 4a": train_metrics[26],
                "train Profit Index 4b": train_metrics[27],
                "train Profit Index 4c": train_metrics[28],
                "train Profit Index 4d": train_metrics[29],
                "train Profit Index 5a": train_metrics[30],
                "train Profit Index 5b": train_metrics[31],
                "train Profit Index 5c": train_metrics[32],
                "train Profit Index 5d": train_metrics[33],
                "train Profit Index 5e": train_metrics[34],
                "train Profit Index 5f": train_metrics[35],
                "train Profit Index 5g": train_metrics[36],
                "train Profit Index 5h": train_metrics[37],
                "train Profit Index 5i": train_metrics[38],
                "train Profit Index 5j": train_metrics[39],
                "train Profit Index 5k": train_metrics[40],
                "train Profit Index 5l": train_metrics[41],
                "train Profit Index 5m": train_metrics[42],
                "train Profit Index 5n": train_metrics[43],
                "train countp0": train_metrics[44],
                "train counts0": train_metrics[45],
                "train countp1": train_metrics[46],
                "train counts1": train_metrics[47],
                "train countp2": train_metrics[48],
                "train counts2": train_metrics[49],
                "train countp3": train_metrics[50],
                "train counts3": train_metrics[51],
                "train countp4": train_metrics[52],
                "train counts4": train_metrics[53],
                "train countp5": train_metrics[54],
                "train counts5": train_metrics[55],
                "train countp0a": train_metrics[56],
                "train counts0a": train_metrics[57],
                "train countp0b": train_metrics[58],
                "train counts0b": train_metrics[59],
                "train countp0c": train_metrics[60],
                "train counts0c": train_metrics[61],
                "train countp0d": train_metrics[62],
                "train counts0d": train_metrics[63],
                "train countp1a": train_metrics[64],
                "train counts1a": train_metrics[65],
                "train countp1b": train_metrics[66],
                "train counts1b": train_metrics[67],
                "train countp1c": train_metrics[68],
                "train counts1c": train_metrics[69],
                "train countp1d": train_metrics[70],
                "train counts1d": train_metrics[71],
                "train countp2a": train_metrics[72],
                "train counts2a": train_metrics[73],
                "train countp2b": train_metrics[74],
                "train counts2b": train_metrics[75],
                "train countp2c": train_metrics[76],
                "train counts2c": train_metrics[77],
                "train countp2d": train_metrics[78],
                "train counts2d": train_metrics[79],
                "train countp3a": train_metrics[80],
                "train counts3a": train_metrics[81],
                "train countp3b": train_metrics[82],
                "train counts3b": train_metrics[83],
                "train countp3c": train_metrics[84],
                "train counts3c": train_metrics[85],
                "train countp3d": train_metrics[86],
                "train counts3d": train_metrics[87],
                "train countp4a": train_metrics[88],
                "train counts4a": train_metrics[89],
                "train countp4b": train_metrics[90],
                "train counts4b": train_metrics[91],
                "train countp4c": train_metrics[92],
                "train counts4c": train_metrics[93],
                "train countp4d": train_metrics[94],
                "train counts4d": train_metrics[95],
                "train countp5a": train_metrics[96],
                "train counts5a": train_metrics[97],
                "train countp5b": train_metrics[98],
                "train counts5b": train_metrics[99],
                "train countp5c": train_metrics[100],
                "train counts5c": train_metrics[101],
                "train countp5d": train_metrics[102],
                "train counts5d": train_metrics[103],
                "train countp5e": train_metrics[104],
                "train counts5e": train_metrics[105],
                "train countp5f": train_metrics[106],
                "train counts5f": train_metrics[107],
                "train countp5g": train_metrics[108],
                "train counts5g": train_metrics[109],
                "train countp5h": train_metrics[110],
                "train counts5h": train_metrics[111],
                "train countp5i": train_metrics[112],
                "train counts5i": train_metrics[113],
                "train countp5j": train_metrics[114],
                "train counts5j": train_metrics[115],
                "train countp5k": train_metrics[116],
                "train counts5k": train_metrics[117],
                "train countp5l": train_metrics[118],
                "train counts5l": train_metrics[119],
                "train countp5m": train_metrics[120],
                "train counts5m": train_metrics[121],
                "train countp5n": train_metrics[122],
                "train counts5n": train_metrics[123],
    
                # Validation Metrics
                "validate RMSE": val_metrics[0],
                "validate MAE": val_metrics[1],
                "validate R2": val_metrics[2],
                "validate Accuracy": val_metrics[3],
                "validate Profit Index": val_metrics[4],
                "validate Profit Index 1": val_metrics[5],
                "validate Profit Index 2": val_metrics[6],
                "validate Profit Index 3": val_metrics[7],
                "validate Profit Index 4": val_metrics[8],
                "validate Profit Index 5": val_metrics[9],
                "validate Profit Index 0a": val_metrics[10],
                "validate Profit Index 0b": val_metrics[11],
                "validate Profit Index 0c": val_metrics[12],
                "validate Profit Index 0d": val_metrics[13],
                "validate Profit Index 1a": val_metrics[14],
                "validate Profit Index 1b": val_metrics[15],
                "validate Profit Index 1c": val_metrics[16],
                "validate Profit Index 1d": val_metrics[17],
                "validate Profit Index 2a": val_metrics[18],
                "validate Profit Index 2b": val_metrics[19],
                "validate Profit Index 2c": val_metrics[20],
                "validate Profit Index 2d": val_metrics[21],
                "validate Profit Index 3a": val_metrics[22],
                "validate Profit Index 3b": val_metrics[23],
                "validate Profit Index 3c": val_metrics[24],
                "validate Profit Index 3d": val_metrics[25],
                "validate Profit Index 4a": val_metrics[26],
                "validate Profit Index 4b": val_metrics[27],
                "validate Profit Index 4c": val_metrics[28],
                "validate Profit Index 4d": val_metrics[29],
                "validate Profit Index 5a": val_metrics[30],
                "validate Profit Index 5b": val_metrics[31],
                "validate Profit Index 5c": val_metrics[32],
                "validate Profit Index 5d": val_metrics[33],
                "validate Profit Index 5e": val_metrics[34],
                "validate Profit Index 5f": val_metrics[35],
                "validate Profit Index 5g": val_metrics[36],
                "validate Profit Index 5h": val_metrics[37],
                "validate Profit Index 5i": val_metrics[38],
                "validate Profit Index 5j": val_metrics[39],
                "validate Profit Index 5k": val_metrics[40],
                "validate Profit Index 5l": val_metrics[41],
                "validate Profit Index 5m": val_metrics[42],
                "validate Profit Index 5n": val_metrics[43],
                "validate countp0": val_metrics[44],
                "validate counts0": val_metrics[45],
                "validate countp1": val_metrics[46],
                "validate counts1": val_metrics[47],
                "validate countp2": val_metrics[48],
                "validate counts2": val_metrics[49],
                "validate countp3": val_metrics[50],
                "validate counts3": val_metrics[51],
                "validate countp4": val_metrics[52],
                "validate counts4": val_metrics[53],
                "validate countp5": val_metrics[54],
                "validate counts5": val_metrics[55],
                "validate countp0a": val_metrics[56],
                "validate counts0a": val_metrics[57],
                "validate countp0b": val_metrics[58],
                "validate counts0b": val_metrics[59],
                "validate countp0c": val_metrics[60],
                "validate counts0c": val_metrics[61],
                "validate countp0d": val_metrics[62],
                "validate counts0d": val_metrics[63],
                "validate countp1a": val_metrics[64],
                "validate counts1a": val_metrics[65],
                "validate countp1b": val_metrics[66],
                "validate counts1b": val_metrics[67],
                "validate countp1c": val_metrics[68],
                "validate counts1c": val_metrics[69],
                "validate countp1d": val_metrics[70],
                "validate counts1d": val_metrics[71],
                "validate countp2a": val_metrics[72],
                "validate counts2a": val_metrics[73],
                "validate countp2b": val_metrics[74],
                "validate counts2b": val_metrics[75],
                "validate countp2c": val_metrics[76],
                "validate counts2c": val_metrics[77],
                "validate countp2d": val_metrics[78],
                "validate counts2d": val_metrics[79],
                "validate countp3a": val_metrics[80],
                "validate counts3a": val_metrics[81],
                "validate countp3b": val_metrics[82],
                "validate counts3b": val_metrics[83],
                "validate countp3c": val_metrics[84],
                "validate counts3c": val_metrics[85],
                "validate countp3d": val_metrics[86],
                "validate counts3d": val_metrics[87],
                "validate countp4a": val_metrics[88],
                "validate counts4a": val_metrics[89],
                "validate countp4b": val_metrics[90],
                "validate counts4b": val_metrics[91],
                "validate countp4c": val_metrics[92],
                "validate counts4c": val_metrics[93],
                "validate countp4d": val_metrics[94],
                "validate counts4d": val_metrics[95],
                "validate countp5a": val_metrics[96],
                "validate counts5a": val_metrics[97],
                "validate countp5b": val_metrics[98],
                "validate counts5b": val_metrics[99],
                "validate countp5c": val_metrics[100],
                "validate counts5c": val_metrics[101],
                "validate countp5d": val_metrics[102],
                "validate counts5d": val_metrics[103],
                "validate countp5e": val_metrics[104],
                "validate counts5e": val_metrics[105],
                "validate countp5f": val_metrics[106],
                "validate counts5f": val_metrics[107],
                "validate countp5g": val_metrics[108],
                "validate counts5g": val_metrics[109],
                "validate countp5h": val_metrics[110],
                "validate counts5h": val_metrics[111],
                "validate countp5i": val_metrics[112],
                "validate counts5i": val_metrics[113],
                "validate countp5j": val_metrics[114],
                "validate counts5j": val_metrics[115],
                "validate countp5k": val_metrics[116],
                "validate counts5k": val_metrics[117],
                "validate countp5l": val_metrics[118],
                "validate counts5l": val_metrics[119],
                "validate countp5m": val_metrics[120],
                "validate counts5m": val_metrics[121],
                "validate countp5n": val_metrics[122],
                "validate counts5n": val_metrics[123],
    
                # Test Metrics
                "test RMSE": test_metrics[0],
                "test MAE": test_metrics[1],
                "test R2": test_metrics[2],
                "test Accuracy": test_metrics[3],
                "test Profit Index": test_metrics[4],
                "test Profit Index 1": test_metrics[5],
                "test Profit Index 2": test_metrics[6],
                "test Profit Index 3": test_metrics[7],
                "test Profit Index 4": test_metrics[8],
                "test Profit Index 5": test_metrics[9],
                "test Profit Index 0a": test_metrics[10],
                "test Profit Index 0b": test_metrics[11],
                "test Profit Index 0c": test_metrics[12],
                "test Profit Index 0d": test_metrics[13],
                "test Profit Index 1a": test_metrics[14],
                "test Profit Index 1b": test_metrics[15],
                "test Profit Index 1c": test_metrics[16],
                "test Profit Index 1d": test_metrics[17],
                "test Profit Index 2a": test_metrics[18],
                "test Profit Index 2b": test_metrics[19],
                "test Profit Index 2c": test_metrics[20],
                "test Profit Index 2d": test_metrics[21],
                "test Profit Index 3a": test_metrics[22],
                "test Profit Index 3b": test_metrics[23],
                "test Profit Index 3c": test_metrics[24],
                "test Profit Index 3d": test_metrics[25],
                "test Profit Index 4a": test_metrics[26],
                "test Profit Index 4b": test_metrics[27],
                "test Profit Index 4c": test_metrics[28],
                "test Profit Index 4d": test_metrics[29],
                "test Profit Index 5a": test_metrics[30],
                "test Profit Index 5b": test_metrics[31],
                "test Profit Index 5c": test_metrics[32],
                "test Profit Index 5d": test_metrics[33],
                "test Profit Index 5e": test_metrics[34],
                "test Profit Index 5f": test_metrics[35],
                "test Profit Index 5g": test_metrics[36],
                "test Profit Index 5h": test_metrics[37],
                "test Profit Index 5i": test_metrics[38],
                "test Profit Index 5j": test_metrics[39],
                "test Profit Index 5k": test_metrics[40],
                "test Profit Index 5l": test_metrics[41],
                "test Profit Index 5m": test_metrics[42],
                "test Profit Index 5n": test_metrics[43],
                "test countp0": test_metrics[44],
                "test counts0": test_metrics[45],
                "test countp1": test_metrics[46],
                "test counts1": test_metrics[47],
                "test countp2": test_metrics[48],
                "test counts2": test_metrics[49],
                "test countp3": test_metrics[50],
                "test counts3": test_metrics[51],
                "test countp4": test_metrics[52],
                "test counts4": test_metrics[53],
                "test countp5": test_metrics[54],
                "test counts5": test_metrics[55],
                "test countp0a": test_metrics[56],
                "test counts0a": test_metrics[57],
                "test countp0b": test_metrics[58],
                "test counts0b": test_metrics[59],
                "test countp0c": test_metrics[60],
                "test counts0c": test_metrics[61],
                "test countp0d": test_metrics[62],
                "test counts0d": test_metrics[63],
                "test countp1a": test_metrics[64],
                "test counts1a": test_metrics[65],
                "test countp1b": test_metrics[66],
                "test counts1b": test_metrics[67],
                "test countp1c": test_metrics[68],
                "test counts1c": test_metrics[69],
                "test countp1d": test_metrics[70],
                "test counts1d": test_metrics[71],
                "test countp2a": test_metrics[72],
                "test counts2a": test_metrics[73],
                "test countp2b": test_metrics[74],
                "test counts2b": test_metrics[75],
                "test countp2c": test_metrics[76],
                "test counts2c": test_metrics[77],
                "test countp2d": test_metrics[78],
                "test counts2d": test_metrics[79],
                "test countp3a": test_metrics[80],
                "test counts3a": test_metrics[81],
                "test countp3b": test_metrics[82],
                "test counts3b": test_metrics[83],
                "test countp3c": test_metrics[84],
                "test counts3c": test_metrics[85],
                "test countp3d": test_metrics[86],
                "test counts3d": test_metrics[87],
                "test countp4a": test_metrics[88],
                "test counts4a": test_metrics[89],
                "test countp4b": test_metrics[90],
                "test counts4b": test_metrics[91],
                "test countp4c": test_metrics[92],
                "test counts4c": test_metrics[93],
                "test countp4d": test_metrics[94],
                "test counts4d": test_metrics[95],
                "test countp5a": test_metrics[96],
                "test counts5a": test_metrics[97],
                "test countp5b": test_metrics[98],
                "test counts5b": test_metrics[99],
                "test countp5c": test_metrics[100],
                "test counts5c": test_metrics[101],
                "test countp5d": test_metrics[102],
                "test counts5d": test_metrics[103],
                "test countp5e": test_metrics[104],
                "test counts5e": test_metrics[105],
                "test countp5f": test_metrics[106],
                "test counts5f": test_metrics[107],
                "test countp5g": test_metrics[108],
                "test counts5g": test_metrics[109],
                "test countp5h": test_metrics[110],
                "test counts5h": test_metrics[111],
                "test countp5i": test_metrics[112],
                "test counts5i": test_metrics[113],
                "test countp5j": test_metrics[114],
                "test counts5j": test_metrics[115],
                "test countp5k": test_metrics[116],
                "test counts5k": test_metrics[117],
                "test countp5l": test_metrics[118],
                "test counts5l": test_metrics[119],
                "test countp5m": test_metrics[120],
                "test counts5m": test_metrics[121],
                "test countp5n": test_metrics[122],
                "test counts5n": test_metrics[123],
    
                # Computational Performance
                "Training Time (s)": computational_metrics[ "training_time_sec" ],
                "Optimization Time (s)": computational_metrics[ "optimization_time_sec" ],
                "Train Inference Time (s)": computational_metrics[ "train_inference_time_sec" ],
                "Validation Inference Time (s)": computational_metrics[ "validation_inference_time_sec" ],
                "Test Inference Time (s)": computational_metrics[ "test_inference_time_sec" ],
                "Test Latency / Sample (s)": computational_metrics[ "test_latency_per_sample_sec" ],
                "Total Parameters": computational_metrics[ "total_parameters" ],
                "Trainable Parameters": computational_metrics[ "trainable_parameters" ],
                "MACs / Sample": computational_metrics[ "macs_per_sample" ],
                "Approx FLOPs / Sample": computational_metrics[ "flops_per_sample" ],
                "Peak GPU Memory (MB)": computational_metrics[ "peak_gpu_memory_mb" ],
                "CPU Memory Change (MB)": computational_metrics[ "cpu_memory_change_mb" ]
            }
    
            # Convert Results to DataFrame   
            result_df = pd.DataFrame([result_row])
    
            # Display Final Report  
            print( "\n=== Final CNN Experiment Report ===" )
            print( result_df.to_string( index=False ) )
    
            # Save Results  
            output_file = f"{ticker}_CNN_experiment_results-{eval_types}-0.csv"
            result_df.to_csv( output_file, index=False )
            print( f"\nResults saved to: " f"{output_file}" )
    
            # Final Experiment Report
            result_row1 = {
                # Model Identification
                "Ticker": ticker,
                "Model":  model_name,
    
                # Shared / Control Parameters
                "Indicators": ", ".join( fixed_indicators ),
                "Dropout": dropout_rate,
                "LR": learning_rate,
                "Batch Size": batch_size,
                "Window Size": window_size,
                "Forecast Window": forecast_window,
                "Requested Epochs": requested_epochs,
                "Actual Epochs": computational_metrics[ "epochs_completed" ],
    
                # Explainability Summary
                "SHAP Top Feature": xai_summary["SHAP Top Feature"],
                "SHAP Top Feature Importance": xai_summary["SHAP Top Feature Importance"],
                "SHAP Top Lag": xai_summary["SHAP Top Lag"],
                "LIME Top Feature": xai_summary["LIME Top Feature"],
                "LIME Top Feature Importance": xai_summary["LIME Top Feature Importance"],
                "LIME Top Lag": xai_summary["LIME Top Lag"],
                "LIME Mean Fidelity R2": xai_summary["LIME Mean Fidelity R2"],
                "SHAP-LIME Spearman": xai_summary["SHAP-LIME Spearman"],
                "SHAP Explained Samples": xai_summary["SHAP Explained Samples"],
                "LIME Explained Samples": xai_summary["LIME Explained Samples"],
    
    
                # CNN-specific Architecture
                "Filters": filters,
                "Kernel Size": kernel_size,
                "Pooling Size": pooling_size,
                "Activation": activation,
    
                # Train Metrics
                "train RMSE": train_metrics[0],
                "train MAE": train_metrics[1],
                "train R2": train_metrics[2],
                "train Accuracy": train_metrics[3],
                "train Profit Index": train_metrics[4],
                "train Profit Index 1": train_metrics[5],
                "train Profit Index 2": train_metrics[6],
                "train Profit Index 3": train_metrics[7],
                "train Profit Index 4": train_metrics[8],
    
                "train countp0": train_metrics[44],
                "train counts0": train_metrics[45],
                "train countp1": train_metrics[46],
                "train counts1": train_metrics[47],
                "train countp2": train_metrics[48],
                "train counts2": train_metrics[49],
                "train countp3": train_metrics[50],
                "train counts3": train_metrics[51],
                "train countp4": train_metrics[52],
                "train counts4": train_metrics[53],
    
    
                # Validation Metrics
                "validate RMSE": val_metrics[0],
                "validate MAE": val_metrics[1],
                "validate R2": val_metrics[2],
                "validate Accuracy": val_metrics[3],
                "validate Profit Index": val_metrics[4],
                "validate Profit Index 1": val_metrics[5],
                "validate Profit Index 2": val_metrics[6],
                "validate Profit Index 3": val_metrics[7],
                "validate Profit Index 4": val_metrics[8],
    
                "validate countp0": val_metrics[44],
                "validate counts0": val_metrics[45],
                "validate countp1": val_metrics[46],
                "validate counts1": val_metrics[47],
                "validate countp2": val_metrics[48],
                "validate counts2": val_metrics[49],
                "validate countp3": val_metrics[50],
                "validate counts3": val_metrics[51],
                "validate countp4": val_metrics[52],
                "validate counts4": val_metrics[53],
    
    
                # Test Metrics
                "test RMSE": test_metrics[0],
                "test MAE": test_metrics[1],
                "test R2": test_metrics[2],
                "test Accuracy": test_metrics[3],
                "test Profit Index": test_metrics[4],
                "test Profit Index 1": test_metrics[5],
                "test Profit Index 2": test_metrics[6],
                "test Profit Index 3": test_metrics[7],
                "test Profit Index 4": test_metrics[8],
    
                "test countp0": test_metrics[44],
                "test counts0": test_metrics[45],
                "test countp1": test_metrics[46],
                "test counts1": test_metrics[47],
                "test countp2": test_metrics[48],
                "test counts2": test_metrics[49],
                "test countp3": test_metrics[50],
                "test counts3": test_metrics[51],
                "test countp4": test_metrics[52],
                "test counts4": test_metrics[53],
    
    
                # Computational Performance
                "Training Time (s)": computational_metrics[ "training_time_sec" ],
                "Optimization Time (s)": computational_metrics[ "optimization_time_sec" ],
                "Train Inference Time (s)": computational_metrics[ "train_inference_time_sec" ],
                "Validation Inference Time (s)": computational_metrics[ "validation_inference_time_sec" ],
                "Test Inference Time (s)": computational_metrics[ "test_inference_time_sec" ],
                "Test Latency / Sample (s)": computational_metrics[ "test_latency_per_sample_sec" ],
                "Total Parameters": computational_metrics[ "total_parameters" ],
                "Trainable Parameters": computational_metrics[ "trainable_parameters" ],
                "MACs / Sample": computational_metrics[ "macs_per_sample" ],
                "Approx FLOPs / Sample": computational_metrics[ "flops_per_sample" ],
                "Peak GPU Memory (MB)": computational_metrics[ "peak_gpu_memory_mb" ],
                "CPU Memory Change (MB)": computational_metrics[ "cpu_memory_change_mb" ]
            }
    
            # Convert Report to DataFrame
            result_df1 = pd.DataFrame( [result_row1] )
    
            # Display Final Report
            print( "\n=== Final Experiment Report ===" )
            print( result_df1.to_string( index=False ) )
    
            # Save Report
            output_file =  f"{ticker}_CNN_experiment_results-{eval_types}-1.csv"
            result_df1.to_csv( output_file, index=False )
            print( f"\nResults saved to: " f"{output_file}" )
    
            plt.close("all")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nAll 4 US CNN + SHAP + LIME experiments completed.")

# Main Execution
#if __name__ == "__main__":
#    model_name = "LSTM"
#    market = "US"
    #tikcer_choice 1: AAPL, 2:QCOM, 3:BNY, 4:C, 5:0097.KL, 6:0166.KL, 7:1023.KL, 8:5258.KL
    #eval_types 1:price level, 2: price change, 3: price return (yet implement)
#    start_experiment(ticker_choice=1, eval_types=1)

if len(sys.argv) > 2:
    arg1 = int(sys.argv[1])
    arg2 = int(sys.argv[2])

    #tikcer_choice 1: AAPL, 2:QCOM, 3:BNY, 4:C, 5:0097.KL, 6:0166.KL, 7:1023.KL, 8:5258.KL
    #eval_types 1:price level, 2: price change, 3: price return (yet implement)
    start_experiment(ticker_choice=arg1, eval_types=arg2)