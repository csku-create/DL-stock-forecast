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
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# Explainability
# Colab: !pip install lime shap
try:
    import shap
except ImportError:
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    LimeTabularExplainer = None

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
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )

if torch.cuda.is_available():
    print( "GPU Name:", torch.cuda.get_device_name(0) )
else:
    print( "Running on CPU" )

# Utility Functions
def sync_device():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def get_parameter_count(model):
    total_params = sum( p.numel() for p in model.parameters() )
    trainable_params = sum( p.numel() for p in model.parameters() if p.requires_grad )
    return total_params, trainable_params

def get_computational_requirement( model, input_shape ):
    sequence_length = input_shape[0]
    number_of_features = input_shape[1]
    input_features = sequence_length * number_of_features
    output_features = model.linear.out_features
    macs = input_features * output_features
    flops = 2 * macs
    return macs, flops


# Explainability Functions

def make_flat_feature_names(selected_features, window_size):
    """Create names matching X.reshape(n_samples, -1): oldest lag -> newest lag."""
    names = []
    for t in range(window_size):
        lag = window_size - 1 - t
        for feature in selected_features:
            names.append(f"{feature}[t-{lag}]")
    return names


def predict_unscaled_from_flat(model, X_flat, window_size, n_features, scaler_y, batch_size=2048):
    """Predict original-scale target values from flattened windows."""
    X_flat = np.asarray(X_flat, dtype=np.float32)
    preds = []
    model.eval()

    with torch.inference_mode():
        for start in range(0, len(X_flat), batch_size):
            batch = X_flat[start:start + batch_size]
            batch_3d = batch.reshape(-1, window_size, n_features)
            batch_tensor = torch.tensor(batch_3d, dtype=torch.float32, device=device)
            pred_scaled = model(batch_tensor).squeeze(-1).detach().cpu().numpy()
            preds.append(pred_scaled)

    if not preds:
        return np.empty((0,), dtype=float)

    pred_scaled = np.concatenate(preds)
    return scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)


def run_shap_lime_analysis(
    model,
    X_train,
    X_test,
    selected_features,
    scaler_y,
    ticker,
    window_size,
    lime_local_samples=20,
    lime_num_samples=3000,
    local_sample_index=0,
    random_state=42,
):
    """
    Explain a linear window model with exact closed-form SHAP values and LIME.

    SHAP:
      * exact for this affine model
      * uses the training mean as the background expectation
      * reports contributions in original target units

    LIME:
      * explains the same representative test rows used for SHAP-vs-LIME comparison
      * uses every flattened lag-feature variable (feature_selection="none")
      * reports contribution-like values = local coefficient * standardized input
        so the global comparison is closer in meaning to SHAP than raw coefficients

    Important:
      Raw LIME coefficients and SHAP values are different quantities. The comparison
      below therefore uses LIME contribution magnitudes and SHAP contribution
      magnitudes over the same test observations.
    """
    print("\n=== SHAP + LIME Explainability ===")

    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)

    n_features = len(selected_features)
    n_flat_features = window_size * n_features

    if X_train.ndim != 3 or X_test.ndim != 3:
        raise ValueError(
            "X_train and X_test must have shape "
            "(samples, window_size, features)."
        )

    if X_train.shape[1:] != (window_size, n_features):
        raise ValueError(
            f"Training data shape {X_train.shape} does not match "
            f"window_size={window_size}, n_features={n_features}."
        )

    if X_test.shape[1:] != (window_size, n_features):
        raise ValueError(
            f"Test data shape {X_test.shape} does not match "
            f"window_size={window_size}, n_features={n_features}."
        )

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            "X_train and X_test must be non-empty for SHAP/LIME."
        )

    output_dir = f"{ticker}_LR_XAI"
    os.makedirs(output_dir, exist_ok=True)

    flat_feature_names = make_flat_feature_names(
        selected_features,
        window_size
    )

    X_train_flat = X_train.reshape(
        len(X_train),
        n_flat_features
    )
    X_test_flat = X_test.reshape(
        len(X_test),
        n_flat_features
    )

    # ============================================================
    # Exact SHAP for this affine model
    # ============================================================
    model.eval()

    weights_scaled = (
        model.linear.weight
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
        .astype(np.float64)
    )
    bias_scaled = float(
        model.linear.bias
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)[0]
    )

    if len(weights_scaled) != n_flat_features:
        raise ValueError(
            f"Linear layer has {len(weights_scaled)} weights, "
            f"but flattened input has {n_flat_features} variables."
        )

    # General MinMaxScaler inverse transform:
    # y_original = (y_scaled - min_) / scale_
    # This remains correct even if feature_range is not (0, 1).
    target_scaler_scale = float(scaler_y.scale_[0])
    target_scaler_min = float(scaler_y.min_[0])

    if target_scaler_scale == 0:
        raise ValueError(
            "scaler_y.scale_[0] is zero; cannot invert target scaling."
        )

    inverse_target_scale = 1.0 / target_scaler_scale
    inverse_target_offset = -target_scaler_min / target_scaler_scale

    weights_original = weights_scaled * inverse_target_scale
    bias_original = (
        bias_scaled * inverse_target_scale
        + inverse_target_offset
    )

    # Interventional SHAP for a linear model with the training mean
    # as the background expectation.
    background_mean = X_train_flat.mean(axis=0)
    expected_value = (
        bias_original
        + float(np.dot(weights_original, background_mean))
    )

    shap_values = (
        X_test_flat - background_mean
    ) * weights_original

    predictions_unscaled = predict_unscaled_from_flat(
        model=model,
        X_flat=X_test_flat,
        window_size=window_size,
        n_features=n_features,
        scaler_y=scaler_y,
    )

    shap_reconstructed = (
        expected_value
        + shap_values.sum(axis=1)
    )

    shap_reconstruction_mae = float(
        np.mean(
            np.abs(
                predictions_unscaled
                - shap_reconstructed
            )
        )
    )

    print(
        f"SHAP expected value (original target units): "
        f"{expected_value:.6f}"
    )
    print(
        f"SHAP reconstruction MAE               : "
        f"{shap_reconstruction_mae:.12f}"
    )

    # ============================================================
    # SHAP global results over the full test set
    # ============================================================
    mean_abs_shap_flat = np.mean(
        np.abs(shap_values),
        axis=0
    )
    signed_mean_shap_flat = np.mean(
        shap_values,
        axis=0
    )

    shap_flat_df = pd.DataFrame({
        "Time-Feature": flat_feature_names,
        "Mean |SHAP|": mean_abs_shap_flat,
        "Mean SHAP": signed_mean_shap_flat,
        "Linear Weight (original target units)": weights_original,
        "Training Mean (scaled input)": background_mean,
    }).sort_values(
        "Mean |SHAP|",
        ascending=False
    )

    shap_flat_df["Rank"] = np.arange(
        1,
        len(shap_flat_df) + 1
    )

    shap_flat_df.to_csv(
        os.path.join(
            output_dir,
            "shap_time_feature_importance.csv"
        ),
        index=False
    )

    shap_cube = shap_values.reshape(
        len(X_test),
        window_size,
        n_features
    )

    shap_feature_importance = np.mean(
        np.abs(shap_cube),
        axis=(0, 1)
    )
    shap_feature_signed = np.mean(
        shap_cube,
        axis=(0, 1)
    )
    shap_lag_importance = np.mean(
        np.abs(shap_cube),
        axis=(0, 2)
    )
    shap_time_feature = np.mean(
        np.abs(shap_cube),
        axis=0
    )

    shap_feature_df = pd.DataFrame({
        "Feature": selected_features,
        "Mean |SHAP|": shap_feature_importance,
        "Mean SHAP": shap_feature_signed,
    }).sort_values(
        "Mean |SHAP|",
        ascending=False
    )

    shap_feature_df["Rank"] = (
        shap_feature_df["Mean |SHAP|"]
        .rank(method="average", ascending=False)
    )

    shap_feature_df.to_csv(
        os.path.join(
            output_dir,
            "shap_global_feature_importance.csv"
        ),
        index=False
    )

    lag_labels = [
        f"t-{window_size - 1 - t}"
        for t in range(window_size)
    ]

    shap_lag_df = pd.DataFrame({
        "Lag": lag_labels,
        "Mean |SHAP|": shap_lag_importance,
    }).sort_values(
        "Mean |SHAP|",
        ascending=False
    )

    shap_lag_df["Rank"] = (
        shap_lag_df["Mean |SHAP|"]
        .rank(method="average", ascending=False)
    )

    shap_lag_df.to_csv(
        os.path.join(
            output_dir,
            "shap_lag_importance.csv"
        ),
        index=False
    )

    shap_heatmap_df = pd.DataFrame(
        shap_time_feature,
        index=lag_labels,
        columns=selected_features,
    )
    shap_heatmap_df.index.name = "Lag"
    shap_heatmap_df.to_csv(
        os.path.join(
            output_dir,
            "shap_time_feature_heatmap.csv"
        )
    )

    # Global SHAP feature plot.
    plot_df = shap_feature_df.sort_values(
        "Mean |SHAP|",
        ascending=True
    )
    plt.figure(figsize=(9, 5))
    plt.barh(
        plot_df["Feature"],
        plot_df["Mean |SHAP|"]
    )
    plt.xlabel("Mean absolute SHAP value (target units)")
    plt.ylabel("Feature")
    plt.title(
        "Linear Regression - Global SHAP Feature Importance"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            "shap_global_feature_importance.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    # SHAP lag plot.
    lag_plot_df = pd.DataFrame({
        "Lag": lag_labels,
        "Mean |SHAP|": shap_lag_importance
    })
    plt.figure(figsize=(9, 5))
    plt.bar(
        lag_plot_df["Lag"],
        lag_plot_df["Mean |SHAP|"]
    )
    plt.xlabel("Lag")
    plt.ylabel("Mean absolute SHAP value (target units)")
    plt.title(
        "Linear Regression - SHAP Lag Importance"
    )
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            "shap_lag_importance.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    # SHAP time-feature heatmap.
    plt.figure(
        figsize=(
            max(9, n_features * 1.3),
            max(5, window_size * 0.8)
        )
    )
    image = plt.imshow(
        shap_time_feature,
        aspect="auto"
    )
    plt.colorbar(
        image,
        label="Mean absolute SHAP value (target units)"
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
    plt.title(
        "Linear Regression - SHAP Time × Feature Importance"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            "shap_time_feature_heatmap.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    # Local SHAP explanation.
    local_sample_index = int(
        np.clip(
            local_sample_index,
            0,
            len(X_test) - 1
        )
    )

    local_shap = shap_values[local_sample_index]
    local_order = np.argsort(
        np.abs(local_shap)
    )[::-1]

    local_shap_df = pd.DataFrame({
        "Time-Feature":
            np.asarray(flat_feature_names)[local_order],
        "SHAP Value":
            local_shap[local_order],
        "Input Value (scaled)":
            X_test_flat[
                local_sample_index,
                local_order
            ],
    })

    local_shap_df["Absolute SHAP"] = np.abs(
        local_shap_df["SHAP Value"]
    )
    local_shap_df["Prediction"] = (
        predictions_unscaled[local_sample_index]
    )
    local_shap_df["Expected Value"] = expected_value

    local_shap_df.to_csv(
        os.path.join(
            output_dir,
            "shap_local_explanation.csv"
        ),
        index=False
    )

    top_local = local_shap_df.head(
        min(15, len(local_shap_df))
    ).iloc[::-1]

    plt.figure(figsize=(10, 7))
    plt.barh(
        top_local["Time-Feature"],
        top_local["SHAP Value"]
    )
    plt.xlabel(
        "SHAP contribution to prediction (target units)"
    )
    plt.ylabel("Time-Feature")
    plt.title(
        "Linear Regression - Local SHAP Explanation "
        f"(test sample {local_sample_index})"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            "shap_local_explanation.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    # Optional native SHAP waterfall.
    if shap is not None:
        try:
            explanation = shap.Explanation(
                values=local_shap,
                base_values=expected_value,
                data=X_test_flat[local_sample_index],
                feature_names=flat_feature_names,
            )
            shap.plots.waterfall(
                explanation,
                max_display=15,
                show=False
            )
            plt.title(
                "Linear Regression - SHAP Waterfall "
                f"(test sample {local_sample_index})"
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    output_dir,
                    "shap_waterfall.png"
                ),
                dpi=300,
                bbox_inches="tight"
            )
            plt.show()
        except Exception as exc:
            print(
                f"Native SHAP waterfall skipped: {exc}"
            )

    # ============================================================
    # LIME
    # ============================================================
    if LimeTabularExplainer is None:
        raise ImportError(
            "LIME is not installed. Install with: pip install lime"
        )

    lime_explainer = LimeTabularExplainer(
        training_data=X_train_flat,
        feature_names=flat_feature_names,
        mode="regression",
        discretize_continuous=False,
        feature_selection="none",
        random_state=random_state,
    )

    def lime_predict_fn(X_flat):
        return predict_unscaled_from_flat(
            model=model,
            X_flat=X_flat,
            window_size=window_size,
            n_features=n_features,
            scaler_y=scaler_y,
        )

    # Use representative test rows spread across the test period.
    n_lime = min(
        max(1, int(lime_local_samples)),
        len(X_test_flat)
    )

    if n_lime == 1:
        lime_indices = np.array([0], dtype=int)
    else:
        lime_indices = np.linspace(
            0,
            len(X_test_flat) - 1,
            n_lime,
            dtype=int
        )
        lime_indices = np.unique(lime_indices)
        n_lime = len(lime_indices)

    # The LIME local surrogate is fitted in the explainer's standardized
    # representation. These values convert each selected test row to that
    # same representation so coefficient * standardized_input becomes a
    # contribution-like quantity in target units.
    lime_standardized_test = (
        X_test_flat - lime_explainer.scaler.mean_
    ) / lime_explainer.scaler.scale_

    lime_coefficient_abs_sum = np.zeros(
        n_flat_features,
        dtype=np.float64
    )
    lime_coefficient_signed_sum = np.zeros(
        n_flat_features,
        dtype=np.float64
    )
    lime_contribution_abs_sum = np.zeros(
        n_flat_features,
        dtype=np.float64
    )
    lime_contribution_signed_sum = np.zeros(
        n_flat_features,
        dtype=np.float64
    )

    lime_records = []
    fidelity_records = []

    for test_idx in lime_indices:
        explanation = lime_explainer.explain_instance(
            X_test_flat[test_idx],
            lime_predict_fn,
            num_features=n_flat_features,
            num_samples=lime_num_samples,
        )

        if not explanation.local_exp:
            raise RuntimeError(
                f"LIME returned no local explanation "
                f"for test sample {test_idx}."
            )

        # Prefer the regression output key with the most coefficients.
        label_key = max(
            explanation.local_exp,
            key=lambda key: len(
                explanation.local_exp[key]
            )
        )
        feature_weights = (
            explanation.local_exp[label_key]
        )

        if len(feature_weights) != n_flat_features:
            raise RuntimeError(
                "Incomplete LIME explanation: "
                f"received {len(feature_weights)} coefficients, "
                f"expected {n_flat_features}."
            )

        local_weights = np.full(
            n_flat_features,
            np.nan,
            dtype=np.float64
        )

        for feature_idx, coefficient in feature_weights:
            local_weights[int(feature_idx)] = float(coefficient)

        if np.isnan(local_weights).any():
            raise RuntimeError(
                f"LIME coefficient vector is incomplete "
                f"for test sample {test_idx}."
            )

        # Do NOT sign-flip LIME using the true model weights.
        # That would leak model information into the explanation.

        standardized_instance = (
            lime_standardized_test[test_idx]
        )

        local_contributions = (
            local_weights * standardized_instance
        )

        lime_coefficient_abs_sum += np.abs(
            local_weights
        )
        lime_coefficient_signed_sum += local_weights

        lime_contribution_abs_sum += np.abs(
            local_contributions
        )
        lime_contribution_signed_sum += (
            local_contributions
        )

        model_prediction = float(
            lime_predict_fn(
                X_test_flat[
                    test_idx:test_idx + 1
                ]
            )[0]
        )

        lime_local_prediction = float(
            np.ravel(
                explanation.local_pred
            )[0]
        )

        fidelity = float(
            getattr(
                explanation,
                "score",
                np.nan
            )
        )

        fidelity_records.append({
            "Test Sample Index": int(test_idx),
            "Model Prediction": model_prediction,
            "LIME Local Prediction": lime_local_prediction,
            "Absolute Prediction Error":
                abs(
                    model_prediction
                    - lime_local_prediction
                ),
            "LIME Fidelity R2": fidelity,
        })

        order = np.argsort(
            np.abs(local_contributions)
        )[::-1]

        for rank, feature_idx in enumerate(
            order,
            start=1
        ):
            lime_records.append({
                "Test Sample Index": int(test_idx),
                "Rank": rank,
                "Time-Feature":
                    flat_feature_names[feature_idx],
                "LIME Coefficient":
                    local_weights[feature_idx],
                "Absolute LIME Coefficient":
                    abs(local_weights[feature_idx]),
                "Standardized Input Value":
                    standardized_instance[feature_idx],
                "LIME Contribution":
                    local_contributions[feature_idx],
                "Absolute LIME Contribution":
                    abs(local_contributions[feature_idx]),
                "Input Value (scaled)":
                    X_test_flat[
                        test_idx,
                        feature_idx
                    ],
                "Model Prediction":
                    model_prediction,
                "LIME Local Prediction":
                    lime_local_prediction,
                "LIME Fidelity R2":
                    fidelity,
            })

    lime_mean_abs_coefficient = (
        lime_coefficient_abs_sum / n_lime
    )
    lime_mean_signed_coefficient = (
        lime_coefficient_signed_sum / n_lime
    )
    lime_mean_abs_contribution = (
        lime_contribution_abs_sum / n_lime
    )
    lime_mean_signed_contribution = (
        lime_contribution_signed_sum / n_lime
    )

    lime_flat_df = pd.DataFrame({
        "Time-Feature": flat_feature_names,
        "Mean |LIME Coefficient|":
            lime_mean_abs_coefficient,
        "Mean LIME Coefficient":
            lime_mean_signed_coefficient,
        "Mean |LIME Contribution|":
            lime_mean_abs_contribution,
        "Mean LIME Contribution":
            lime_mean_signed_contribution,
    }).sort_values(
        "Mean |LIME Contribution|",
        ascending=False
    )

    lime_flat_df["Rank"] = (
        lime_flat_df["Mean |LIME Contribution|"]
        .rank(method="average", ascending=False)
    )

    lime_flat_df.to_csv(
        os.path.join(
            output_dir,
            "lime_time_feature_importance.csv"
        ),
        index=False
    )

    # Aggregate contribution-like LIME values by original feature and lag.
    lime_contribution_cube = (
        lime_mean_abs_contribution.reshape(
            window_size,
            n_features
        )
    )

    lime_feature_importance = (
        lime_contribution_cube.mean(axis=0)
    )
    lime_lag_importance = (
        lime_contribution_cube.mean(axis=1)
    )

    lime_feature_df = pd.DataFrame({
        "Feature": selected_features,
        "Mean |LIME Contribution|":
            lime_feature_importance,
    }).sort_values(
        "Mean |LIME Contribution|",
        ascending=False
    )

    lime_feature_df["Rank"] = (
        lime_feature_df[
            "Mean |LIME Contribution|"
        ].rank(
            method="average",
            ascending=False
        )
    )

    lime_feature_df.to_csv(
        os.path.join(
            output_dir,
            "lime_global_feature_importance.csv"
        ),
        index=False
    )

    lime_lag_df = pd.DataFrame({
        "Lag": lag_labels,
        "Mean |LIME Contribution|":
            lime_lag_importance,
    }).sort_values(
        "Mean |LIME Contribution|",
        ascending=False
    )

    lime_lag_df["Rank"] = (
        lime_lag_df[
            "Mean |LIME Contribution|"
        ].rank(
            method="average",
            ascending=False
        )
    )

    lime_lag_df.to_csv(
        os.path.join(
            output_dir,
            "lime_lag_importance.csv"
        ),
        index=False
    )

    pd.DataFrame(
        lime_records
    ).to_csv(
        os.path.join(
            output_dir,
            "lime_local_explanations.csv"
        ),
        index=False
    )

    fidelity_df = pd.DataFrame(
        fidelity_records
    )
    fidelity_df.to_csv(
        os.path.join(
            output_dir,
            "lime_fidelity.csv"
        ),
        index=False
    )

    lime_plot_df = lime_feature_df.sort_values(
        "Mean |LIME Contribution|",
        ascending=True
    )
    plt.figure(figsize=(9, 5))
    plt.barh(
        lime_plot_df["Feature"],
        lime_plot_df[
            "Mean |LIME Contribution|"
        ]
    )
    plt.xlabel(
        "Mean absolute LIME contribution (target units)"
    )
    plt.ylabel("Feature")
    plt.title(
        "Linear Regression - Aggregated LIME Feature Importance"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            "lime_global_feature_importance.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    # ============================================================
    # SHAP vs LIME comparison on the SAME test rows
    # ============================================================
    shap_comparison_cube = (
        shap_values[lime_indices]
        .reshape(
            n_lime,
            window_size,
            n_features
        )
    )

    shap_comparison_feature_importance = np.mean(
        np.abs(shap_comparison_cube),
        axis=(0, 1)
    )

    shap_by_feature = pd.Series(
        shap_comparison_feature_importance,
        index=selected_features,
        dtype=float
    )

    lime_by_feature = pd.Series(
        lime_feature_importance,
        index=selected_features,
        dtype=float
    )

    shap_total = shap_by_feature.sum()
    lime_total = lime_by_feature.sum()

    shap_norm = (
        shap_by_feature / shap_total
        if shap_total > 0
        else shap_by_feature * 0.0
    )
    lime_norm = (
        lime_by_feature / lime_total
        if lime_total > 0
        else lime_by_feature * 0.0
    )

    comparison_df = pd.DataFrame({
        "Feature": selected_features,
        "SHAP Importance (common samples)":
            shap_by_feature.values,
        "LIME Contribution Importance":
            lime_by_feature.values,
        "SHAP Normalized":
            shap_norm.values,
        "LIME Normalized":
            lime_norm.values,
    })

    comparison_df["SHAP Rank"] = (
        comparison_df[
            "SHAP Importance (common samples)"
        ].rank(
            method="average",
            ascending=False
        )
    )

    comparison_df["LIME Rank"] = (
        comparison_df[
            "LIME Contribution Importance"
        ].rank(
            method="average",
            ascending=False
        )
    )

    comparison_df["Rank Difference"] = (
        comparison_df["SHAP Rank"]
        - comparison_df["LIME Rank"]
    ).abs()

    comparison_df = comparison_df.sort_values(
        "SHAP Rank"
    )

    comparison_df.to_csv(
        os.path.join(
            output_dir,
            "shap_lime_feature_comparison.csv"
        ),
        index=False
    )

    if len(selected_features) > 1:
        rank_corr, rank_corr_p = spearmanr(
            comparison_df["SHAP Rank"],
            comparison_df["LIME Rank"],
        )
        rank_corr = float(rank_corr)
        rank_corr_p = float(rank_corr_p)
    else:
        rank_corr = np.nan
        rank_corr_p = np.nan

    mean_lime_fidelity = float(
        fidelity_df["LIME Fidelity R2"].mean()
    )

    mean_lime_prediction_error = float(
        fidelity_df[
            "Absolute Prediction Error"
        ].mean()
    )

    summary = {
        "SHAP Expected Value":
            expected_value,
        "SHAP Reconstruction MAE":
            shap_reconstruction_mae,
        "SHAP Top Feature (full test)":
            str(
                shap_feature_df.iloc[0][
                    "Feature"
                ]
            ),
        "SHAP Top Feature Importance (full test)":
            float(
                shap_feature_df.iloc[0][
                    "Mean |SHAP|"
                ]
            ),
        "SHAP Top Lag":
            str(
                shap_lag_df.iloc[0]["Lag"]
            ),
        "LIME Top Feature":
            str(
                lime_feature_df.iloc[0][
                    "Feature"
                ]
            ),
        "LIME Top Feature Importance":
            float(
                lime_feature_df.iloc[0][
                    "Mean |LIME Contribution|"
                ]
            ),
        "Mean LIME Fidelity R2":
            mean_lime_fidelity,
        "Mean LIME Prediction Error":
            mean_lime_prediction_error,
        "SHAP-LIME Spearman Correlation":
            rank_corr,
        "SHAP-LIME Spearman p-value":
            rank_corr_p,
        "SHAP Full-Test Samples":
            int(len(X_test)),
        "SHAP Comparison Samples":
            int(n_lime),
        "LIME Explained Samples":
            int(n_lime),
    }

    pd.DataFrame(
        [summary]
    ).to_csv(
        os.path.join(
            output_dir,
            "xai_summary.csv"
        ),
        index=False
    )

    print(
        "\n--- SHAP Global Feature Importance "
        "(full test set) ---"
    )
    print(
        shap_feature_df.to_string(
            index=False
        )
    )

    print(
        "\n--- LIME Global Contribution Importance "
        "(common samples) ---"
    )
    print(
        lime_feature_df.to_string(
            index=False
        )
    )

    print(
        "\n--- SHAP vs LIME "
        "(same test samples) ---"
    )
    print(
        comparison_df.to_string(
            index=False
        )
    )

    print(
        f"\nMean LIME fidelity R2          : "
        f"{mean_lime_fidelity:.6f}"
    )
    print(
        f"Mean LIME prediction error     : "
        f"{mean_lime_prediction_error:.12f}"
    )
    print(
        f"SHAP-LIME Spearman correlation : "
        f"{rank_corr:.6f}"
    )
    print(
        f"XAI outputs saved to           : "
        f"{output_dir}"
    )

    return (
        summary,
        shap_feature_df,
        lime_feature_df,
        comparison_df
    )


# Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__( self, input_size ):
        super().__init__()
        self.linear = nn.Linear( input_size, 1 )

    def forward(self, x):
        x = x.reshape( x.size(0), -1 )
        return self.linear( x )



# Metric Function
#y_true: actual next day value; y_pred: predicted next day value
def compute_metrics( eval_types, y_true, y_pred, current_close, forecast_window, market, initial_capital, position_fraction ):
    y_true = np.asarray( y_true ).reshape(-1)
    y_pred = np.asarray( y_pred ).reshape(-1)
    current_close = np.asarray( current_close ).reshape(-1)

    r2 = EM.r2( y_true, y_pred )

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

# Train and Evaluate Model
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
    if isinstance( patience, float ):
        actual_patience = max( 1, int( patience * epochs ) )
    else:
        actual_patience = int( patience )
    
    # Validation / Test Tensors   
    X_val_tensor = torch.tensor( val_data[0], dtype=torch.float32, device=device )
    y_val_tensor = torch.tensor( val_data[1], dtype=torch.float32, device=device )
    z_val_tensor = torch.tensor( val_data[2], dtype=torch.float32, device=device )
    X_test_tensor = torch.tensor( test_data[0], dtype=torch.float32, device=device )
    y_test_tensor = torch.tensor( test_data[1], dtype=torch.float32, device=device )
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
    macs, flops = get_computational_requirement( model=model, input_shape=( full_train_x.shape[1], full_train_x.shape[2] ) )
   
    # Memory Measurement
    process = psutil.Process( os.getpid() )
    cpu_memory_before = process.memory_info().rss / (1024 ** 2)    

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats( device )
    
    # Training Setup
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None
    optimization_time = 0.0
    epochs_completed = 0
    
    # Total Training Timer
    sync_device()
    training_start = time.perf_counter()
 

    # Training Loop
    for epoch in range( epochs ):
        # Training
        model.train()
        for xb, yb, zb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad( set_to_none=True )

            # Forward pass
            predictions = model( xb ).squeeze( -1 )
            loss = criterion( predictions, yb.squeeze(-1) )

            sync_device()
            optimization_start = time.perf_counter()           

            loss.backward()
            optimizer.step()

            sync_device()
            optimization_time += time.perf_counter() - optimization_start

        # Validation
        model.eval()
        with torch.inference_mode():
            val_predictions = model( X_val_tensor ).squeeze( -1 )
            val_loss = criterion( val_predictions, y_val_tensor.squeeze(-1) ).item()

        epochs_completed = epoch + 1

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            # SAME BEST-MODEL RESTORATION AS LSTM
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            epochs_no_improve += 1

            if ( epochs_no_improve >= actual_patience ):
                print( f"Early stopping at epoch " f"{epoch + 1}" )
                break


    
    # End Training Timer
    sync_device()
    training_time = time.perf_counter() - training_start
    
    # Restore Best Model   
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    
    # Memory
    cpu_memory_after = process.memory_info().rss / (1024 ** 2)  
    cpu_memory_consumption =  cpu_memory_after - cpu_memory_before

    if torch.cuda.is_available():
        peak_gpu_memory = torch.cuda.max_memory_allocated(device)/ (1024 ** 2)        
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

    train_preds_scaled = train_output.squeeze(-1).cpu().numpy()
    train_preds_unscaled =  scaler_y.inverse_transform( train_preds_scaled.reshape(-1,1))
    y_train_unscaled = scaler_y.inverse_transform(full_train_y.cpu().numpy().reshape(-1,1))
    
    # Validation Inference   
    sync_device()
    val_inference_start =  time.perf_counter()

    with torch.inference_mode():
        val_output = model( X_val_tensor )

    sync_device()
    val_inference_time = time.perf_counter() - val_inference_start

    val_preds_scaled = val_output.squeeze(-1).cpu().numpy()
    val_preds_unscaled =  scaler_y.inverse_transform( val_preds_scaled.reshape(-1,1))
    y_val_unscaled =  scaler_y.inverse_transform( np.asarray( val_data[1] ).reshape(-1,1))


    
    # Test Inference   
    sync_device()
    test_inference_start = time.perf_counter()

    with torch.inference_mode():
        test_output = model( X_test_tensor )

    sync_device()
    test_inference_time = time.perf_counter() - test_inference_start

    test_preds_scaled = test_output.squeeze(-1).cpu().numpy()
    test_preds_unscaled = scaler_y.inverse_transform( test_preds_scaled.reshape(-1,1))
    y_test_unscaled =  scaler_y.inverse_transform( np.asarray( test_data[1] ).reshape(-1,1))

    
    # Test Latency Per Sample   
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

    # Computational Metrics   
    print("\n=== Computational Performance ===")
    print(
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
    plt.figure(figsize=(12, 5))
    plt.plot(y_val_unscaled.reshape(-1),label="Actual (Val)")
    plt.plot(val_preds_unscaled.reshape(-1),label="Predicted (Val)")
    plt.title("Linear Regression - Validation Set")
    plt.xlabel("Time steps")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Test Plot
    plt.figure(figsize=(12, 5))
    plt.plot(y_test_unscaled.reshape(-1),label="Actual (Test)")
    plt.plot(test_preds_unscaled.reshape(-1),label="Predicted (Test)")
    plt.title("Linear Regression - Test Set")
    plt.xlabel("Time steps")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Computational Dictionary
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
    return train_metrics, val_metrics, test_metrics, computational_metrics

        # Main Execution

def start_experiment(ticker_choice, eval_types):
    model_name = "Linear Regression"
    start_date = "2014-01-01"
    end_date = "2024-12-31"
    patience = 0.2

    experiments = [
        {
            "market": "US",
            "ticker": "AAPL",
            "fixed_indicators": ["20MA", "MACD", "Upper_BB", "CCI", "Williams_%R", "OBV"],
            "window_size": 5,
            "forecast_window": 1,
            "learning_rate": 0.001,
            "batch_size": 16,
            "requested_epochs": 150,
        },
        {
            "market": "US",
            "ticker": "QCOM",
            "fixed_indicators": ["20MA", "RSI", "MACD", "Lower_BB", "ATR", "Williams_%R"],
            "window_size": 5,
            "forecast_window": 1,
            "learning_rate": 0.001,
            "batch_size": 16,
            "requested_epochs": 150,
        },
        {
            "market": "US",
            "ticker": "BNY",
            "fixed_indicators": ["20MA", "50MA", "RSI", "MACD", "Lower_BB", "ATR"],
            "window_size": 5,
            "forecast_window": 1,
            "learning_rate": 0.001,
            "batch_size": 16,
            "requested_epochs": 150,
        },
        {
            "market": "US",
            "ticker": "C",
            "fixed_indicators": ["20MA", "50MA", "Upper_BB", "Lower_BB", "CCI", "Williams_%R"],
            "window_size": 5,
            "forecast_window": 1,
            "learning_rate": 0.001,
            "batch_size": 64,
            "requested_epochs": 150,
        },
        {
            "market": "KL",
            "ticker": "0097.KL",
            "fixed_indicators": ["20MA", "MACD", "CCI", "ATR", "Williams_%R", "OBV"],
            "window_size": 5,
            "forecast_window": 1,
            "learning_rate": 0.001,
            "batch_size": 16,
            "requested_epochs": 150,
        },
        {
            "market": "KL",
            "ticker": "0166.KL",
            "fixed_indicators": ["20MA", "RSI", "MACD", "Upper_BB", "ATR", "Williams_%R"],
            "window_size": 5,
            "forecast_window": 1,
            "learning_rate": 0.0005,
            "batch_size": 16,
            "requested_epochs": 150,
        },
        {
            "market": "KL",
            "ticker": "1023.KL",
            "fixed_indicators": ["50MA", "MACD", "Upper_BB", "Lower_BB", "Williams_%R", "OBV"],
            "window_size": 5,
            "forecast_window": 1,
            "learning_rate": 0.0005,
            "batch_size": 32,
            "requested_epochs": 150,
        },
        {
            "market": "KL",
            "ticker": "5258.KL",
            "fixed_indicators": ["RSI", "MACD", "Upper_BB", "Lower_BB", "CCI", "Williams_%R"],
            "window_size": 5,
            "forecast_window": 1,
            "learning_rate": 0.0005,
            "batch_size": 32,
            "requested_epochs": 150,
        },
    ]

    for experiment_index, config in enumerate(experiments, start=1):
        if experiment_index == ticker_choice:
            market = config["market"]
            ticker = config["ticker"]
            fixed_indicators = config["fixed_indicators"]
            window_size = config["window_size"]
            forecast_window = config["forecast_window"]
            learning_rate = config["learning_rate"]
            batch_size = config["batch_size"]
            requested_epochs = config["requested_epochs"]
            # Reset RNG independently for every ticker
            set_global_seed(seed=42, framework="torch")
    
            print("\n" + "=" * 100)
            print(f"EXPERIMENT {experiment_index}/{len(experiments)}: {ticker}")
            print("=" * 100)
            # Load Data
            data_processor = DataPreprocessing( ticker=ticker, start_date=start_date, end_date=end_date )
            df = ( data_processor.add_technical_indicators() )
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
    
            scaler.fit( X_train_flat )
            X_train = scaler.transform( X_train_flat).reshape( train_shape )    
            X_val = scaler.transform( X_val.reshape( -1, train_shape[2] ) ).reshape( X_val.shape )  
            X_test = scaler.transform( X_test.reshape( -1, train_shape[2] ) ).reshape( X_test.shape )
    
            # Target Scaling
            scaler_y = MinMaxScaler()
            y_train_scaled =  scaler_y.fit_transform( y_train.reshape( -1, 1 ) )
            y_val_scaled =  scaler_y.transform( y_val.reshape( -1, 1 ) )
            y_test_scaled = scaler_y.transform( y_test.reshape( -1, 1 ) )    
    
            train_dataset = TensorDataset(
                torch.tensor( X_train, dtype=torch.float32 ),
                torch.tensor( y_train_scaled, dtype=torch.float32 ),
                torch.tensor( z_train, dtype=torch.float32 )
            )
            train_loader = DataLoader( train_dataset, batch_size=batch_size, shuffle=False )
    
            # Linear Regression Model
            input_size = (X_train.shape[1]* X_train.shape[2])
            model = LinearRegressionModel(input_size=input_size).to(device)
    
            print("\n=== Linear Regression Model Architecture ===")
            print(model)
    
            optimizer = optim.Adam( model.parameters(), lr=learning_rate )
            criterion = nn.MSELoss()
    
            # Experiment Information   
            print(
                "\nRunning Linear Regression experiment:"
                f"\nIndicators       : " f"{fixed_indicators}"
                f"\nInput Size       : " f"{input_size}"
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
            # X_train/X_test are already scaled inputs; explanations are reported in
            # original target (price) units through scaler_y.
            xai_summary, shap_feature_df, lime_feature_df, xai_comparison_df = run_shap_lime_analysis(
                model=model,
                X_train=X_train,
                X_test=X_test,
                selected_features=selected_features,
                scaler_y=scaler_y,
                ticker=ticker,
                window_size=window_size,
                lime_local_samples=20,
                lime_num_samples=3000,
                local_sample_index=0,
                random_state=42,
            )
    
            # Final Experiment Report
            result_row = {   
                # Model
                "Model": model_name,
    
                # Parameters
                "Indicators": ", ".join( fixed_indicators ),
                "LR": learning_rate,
                "Batch Size": batch_size,
                "Window Size": window_size,
                "Forecast Window": forecast_window,
                "Requested Epochs": requested_epochs,
                "Actual Epochs": computational_metrics[ "epochs_completed" ],
    
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
    
                # Explainability Summary
                "SHAP Top Feature": xai_summary["SHAP Top Feature (full test)"],
                "SHAP Top Feature Importance": xai_summary["SHAP Top Feature Importance (full test)" ],
                "SHAP Top Lag": xai_summary["SHAP Top Lag"],
                "SHAP Reconstruction MAE": xai_summary["SHAP Reconstruction MAE"],
                "LIME Top Feature": xai_summary["LIME Top Feature"],
                "LIME Top Feature Importance": xai_summary["LIME Top Feature Importance"],
                "Mean LIME Fidelity R2": xai_summary["Mean LIME Fidelity R2"],
                "SHAP-LIME Spearman Correlation": xai_summary[ "SHAP-LIME Spearman Correlation" ],
    
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
    
            # DataFrame
            result_df = pd.DataFrame( [ result_row ] )
    
            print("\n=== Final Linear Regression Experiment Report ===")
            print( result_df.to_string( index=False ) )
    
            # Save
            output_file = f"{ticker}_LR_experiment_results-{eval_types}-0.csv"
            result_df.to_csv( output_file, index=False )
            print( f"\nResults saved to: " f"{output_file}" )
    
            # Final Experiment Report
            result_row1 = {
                # Model
                "Model": model_name,
    
                # Parameters
                "Indicators": ", ".join( fixed_indicators ),
                "LR": learning_rate,
                "Batch Size": batch_size,
                "Window Size": window_size,
                "Forecast Window": forecast_window,
                "Requested Epochs": requested_epochs,
                "Actual Epochs": computational_metrics[ "epochs_completed" ],
    
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
    
    
                # Explainability Summary
                "SHAP Top Feature": xai_summary["SHAP Top Feature (full test)"],
                "SHAP Top Feature Importance": xai_summary["SHAP Top Feature Importance (full test)" ],
                "SHAP Top Lag": xai_summary["SHAP Top Lag"],
                "SHAP Reconstruction MAE": xai_summary["SHAP Reconstruction MAE"],
                "LIME Top Feature": xai_summary["LIME Top Feature"],
                "LIME Top Feature Importance": xai_summary["LIME Top Feature Importance"],
                "Mean LIME Fidelity R2": xai_summary["Mean LIME Fidelity R2"],
                "SHAP-LIME Spearman Correlation": xai_summary[ "SHAP-LIME Spearman Correlation" ],
    
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
            output_file =  f"{ticker}_LR_experiment_results-{eval_types}-1.csv"
            result_df1.to_csv( output_file, index=False )
            print( f"\nResults saved to: " f"{output_file}" )
    
            # Release resources before the next ticker.
            plt.close("all")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nAll 4 US Linear Regression + SHAP + LIME experiments completed.")

# Main Execution
#if __name__ == "__main__":
#    model_name = "LR"
    
    #tikcer_choice 1: AAPL, 2:QCOM, 3:BNY, 4:C, 5:0097.KL, 6:0166.KL, 7:1023.KL, 8:5258.KL
    #eval_types 1:price level, 2: price change, 3: price return (yet implement)
#    start_experiment(ticker_choice=1, eval_types=1)

if len(sys.argv) > 2:
    arg1 = int(sys.argv[1])
    arg2 = int(sys.argv[2])

    #tikcer_choice 1: AAPL, 2:QCOM, 3:BNY, 4:C, 5:0097.KL, 6:0166.KL, 7:1023.KL, 8:5258.KL
    #eval_types 1:price level, 2: price change, 3: price return (yet implement)
    start_experiment(ticker_choice=arg1, eval_types=arg2)
