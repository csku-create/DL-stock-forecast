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

# Explainable AI (optional dependencies)
# In Colab, install once if needed: !pip install shap lime
try:
    import shap
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    shap = None
    LimeTabularExplainer = None

# Project setup
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
    print("GPU Name:", torch.cuda.get_device_name(0))
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
    dummy_input = torch.randn( 1, input_shape[0], input_shape[1], device=device )

    # Ensure model is in evaluation mode during profiling
    previous_training_state = model.training
    model.eval()
    with torch.no_grad():
        macs, _ = profile( model, inputs=(dummy_input,), verbose=False )

    if previous_training_state:
        model.train()

    flops = 2 * macs
    return macs, flops

# Explainability Utilities
class UnscaledPredictionWrapper(nn.Module):
    """Wrap a scaled-output model so SHAP explanations are expressed in original target units."""
    def __init__( self, base_model, scaler_y ):
        super().__init__()
        self.base_model = base_model
        self.register_buffer(
            "target_scale",
            torch.tensor( scaler_y.scale_[0], dtype=torch.float32 )
        )
        self.register_buffer(
            "target_min",
            torch.tensor( scaler_y.min_[0], dtype=torch.float32 )
        )

    def forward( self, x ):
        y_scaled = self.base_model( x )
        return ( y_scaled - self.target_min ) / self.target_scale


def _normalize_shap_values( shap_values, expected_shape ):
    """Normalize SHAP outputs across SHAP versions to (samples, time, features)."""
    if isinstance( shap_values, list ):
        shap_array = np.asarray( shap_values[0] )
    else:
        shap_array = np.asarray( shap_values )

    # Recent SHAP versions may append an output dimension of size 1.
    if shap_array.ndim == 4 and shap_array.shape[-1] == 1:
        shap_array = np.squeeze( shap_array, axis=-1 )

    if shap_array.shape != expected_shape:
        raise ValueError(
            f"Unexpected SHAP shape {shap_array.shape}; expected {expected_shape}."
        )
    return shap_array


def explain_model_shap_lime(
    model,
    X_train,
    X_test,
    selected_features,
    scaler_y,
    ticker,
    device,
    shap_background_size=100,
    shap_test_size=30,
    lime_training_size=1000,
    lime_test_size=30,
    lime_num_samples=2000,
    lime_num_features=None,
    random_state=42
):
    """
    Generate comparable SHAP and LIME explanations for a trained LSTM.

    Design choices:
      * SHAP and LIME explain exactly the same test samples.
      * SHAP uses GradientExplainer on the native 3-D LSTM input.
      * LIME flattens (time, feature), explains all flattened inputs for the
        global comparison, then reshapes coefficients back to (time, feature).
      * Both explainers operate on the original target scale.
      * Global feature importance is aggregated across test samples and lags.
    """
    if shap is None or LimeTabularExplainer is None:
        print(
            "\n[Explainability skipped] SHAP and/or LIME is not installed. "
            "Install with: pip install shap lime"
        )
        return {
            "shap_top_feature": None,
            "shap_top_importance": np.nan,
            "lime_top_feature": None,
            "lime_top_importance": np.nan,
            "shap_lime_rank_correlation": np.nan,
            "lime_mean_fidelity": np.nan,
            "shap_samples_explained": 0,
            "lime_samples_explained": 0
        }

    rng = np.random.default_rng(random_state)
    model.eval()

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    if X_train.ndim != 3 or X_test.ndim != 3:
        raise ValueError(
            f"Expected 3-D LSTM inputs; got X_train={X_train.shape}, "
            f"X_test={X_test.shape}."
        )

    n_train, sequence_length, n_features = X_train.shape
    n_test = X_test.shape[0]

    if n_train == 0 or n_test == 0:
        raise ValueError("SHAP/LIME requires non-empty training and test sets.")

    if len(selected_features) != n_features:
        raise ValueError(
            f"selected_features has {len(selected_features)} names but "
            f"X has {n_features} features."
        )

    output_dir = f"{ticker}_LSTM_XAI"
    os.makedirs(output_dir, exist_ok=True)

    lag_labels = [
        f"t-{sequence_length - 1 - i}"
        for i in range(sequence_length)
    ]

    # --------------------------------------------------------------
    # Use the SAME test observations for SHAP and LIME.
    # This makes feature-rank agreement directly interpretable.
    # --------------------------------------------------------------
    requested_common_size = min(shap_test_size, lime_test_size)
    n_explain = min(requested_common_size, n_test)

    if n_explain < 1:
        raise ValueError("At least one test sample must be explained.")

    # Evenly cover the test period rather than clustering explanations
    # near one end of the time series.
    if n_explain == 1:
        explain_test_idx = np.array([0], dtype=int)
    else:
        explain_test_idx = np.linspace(
            0, n_test - 1, n_explain, dtype=int
        )

    # Defensive uniqueness in case a very small test set is used.
    explain_test_idx = np.unique(explain_test_idx)
    n_explain = len(explain_test_idx)

    # --------------------------------------------------------------
    # SHAP
    # --------------------------------------------------------------
    n_background = min(shap_background_size, n_train)
    background_idx = np.sort(
        rng.choice(n_train, size=n_background, replace=False)
    )

    background_tensor = torch.tensor(
        X_train[background_idx],
        dtype=torch.float32,
        device=device
    )
    shap_test_tensor = torch.tensor(
        X_test[explain_test_idx],
        dtype=torch.float32,
        device=device
    )

    explainable_model = UnscaledPredictionWrapper(
        model, scaler_y
    ).to(device)

    print("\n=== SHAP Explainability ===")

    original_model_training_state = model.training

    try:
        model.eval()
        explainable_model.eval()

        # GradientExplainer needs autograd. On CUDA, disable cuDNN
        # only for the explanation pass so LSTM backward works in eval mode.
        with torch.enable_grad():
            if torch.cuda.is_available():
                with torch.backends.cudnn.flags(enabled=False):
                    shap_explainer = shap.GradientExplainer(
                        explainable_model,
                        background_tensor
                    )
                    shap_values_raw = shap_explainer.shap_values(
                        shap_test_tensor,
                        rseed=random_state
                    )
            else:
                shap_explainer = shap.GradientExplainer(
                    explainable_model,
                    background_tensor
                )
                shap_values_raw = shap_explainer.shap_values(
                    shap_test_tensor,
                    rseed=random_state
                )
    finally:
        if original_model_training_state:
            model.train()
            explainable_model.train()
        else:
            model.eval()
            explainable_model.eval()

    shap_values = _normalize_shap_values(
        shap_values_raw,
        expected_shape=(n_explain, sequence_length, n_features)
    )

    # Global feature importance: mean absolute attribution over
    # explained samples and all lags.
    shap_feature_importance = np.mean(
        np.abs(shap_values), axis=(0, 1)
    )
    shap_feature_df = pd.DataFrame({
        "Feature": selected_features,
        "SHAP MeanAbs": shap_feature_importance
    }).sort_values(
        "SHAP MeanAbs", ascending=False
    ).reset_index(drop=True)
    shap_feature_df["SHAP Rank"] = np.arange(
        1, len(shap_feature_df) + 1
    )
    shap_feature_df.to_csv(
        os.path.join(output_dir, "shap_global_feature_importance.csv"),
        index=False
    )

    # Lag importance.
    shap_lag_importance = np.mean(
        np.abs(shap_values), axis=(0, 2)
    )
    shap_lag_df = pd.DataFrame({
        "Lag": lag_labels,
        "SHAP MeanAbs": shap_lag_importance
    }).sort_values(
        "SHAP MeanAbs", ascending=False
    ).reset_index(drop=True)
    shap_lag_df["SHAP Lag Rank"] = np.arange(
        1, len(shap_lag_df) + 1
    )
    shap_lag_df.to_csv(
        os.path.join(output_dir, "shap_lag_importance.csv"),
        index=False
    )

    # Time x feature matrix.
    shap_time_feature = np.mean(np.abs(shap_values), axis=0)
    pd.DataFrame(
        shap_time_feature,
        index=lag_labels,
        columns=selected_features
    ).to_csv(
        os.path.join(output_dir, "shap_time_feature_importance.csv")
    )

    # Global SHAP plot.
    plot_df = shap_feature_df.sort_values(
        "SHAP MeanAbs", ascending=True
    )
    plt.figure(figsize=(9, 5))
    plt.barh(plot_df["Feature"], plot_df["SHAP MeanAbs"])
    plt.title("SHAP Global Feature Importance")
    plt.xlabel("Mean |SHAP value| (target units)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "shap_global_feature_importance.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    # SHAP lag plot.
    plt.figure(figsize=(12, 5))
    plt.plot(lag_labels, shap_lag_importance)
    plt.title("SHAP Importance by Input Lag")
    plt.xlabel("Lag")
    plt.ylabel("Mean |SHAP value| (target units)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "shap_lag_importance.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    # SHAP time-feature heatmap.
    plt.figure(figsize=(10, 8))
    image = plt.imshow(
        shap_time_feature,
        aspect="auto",
        origin="lower"
    )
    plt.colorbar(image, label="Mean |SHAP value|")
    plt.yticks(np.arange(sequence_length), lag_labels)
    plt.xticks(
        np.arange(n_features),
        selected_features,
        rotation=45,
        ha="right"
    )
    plt.title("SHAP Time x Feature Importance")
    plt.xlabel("Feature")
    plt.ylabel("Lag")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "shap_time_feature_heatmap.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    # Representative local SHAP explanation.
    local_shap_index = 0
    local_shap_by_feature = np.sum(
        shap_values[local_shap_index], axis=0
    )
    local_shap_df = pd.DataFrame({
        "Feature": selected_features,
        "SHAP Contribution": local_shap_by_feature
    }).sort_values(
        "SHAP Contribution",
        key=np.abs,
        ascending=False
    )
    local_shap_df.insert(
        0,
        "Test Sample Index",
        int(explain_test_idx[local_shap_index])
    )
    local_shap_df.to_csv(
        os.path.join(output_dir, "shap_local_explanation.csv"),
        index=False
    )

    local_plot_df = local_shap_df.sort_values(
        "SHAP Contribution"
    )
    plt.figure(figsize=(9, 5))
    plt.barh(
        local_plot_df["Feature"],
        local_plot_df["SHAP Contribution"]
    )
    plt.axvline(0.0, linewidth=1)
    plt.title(
        "SHAP Local Explanation - Test Sample "
        f"{int(explain_test_idx[local_shap_index])}"
    )
    plt.xlabel("Signed SHAP contribution (target units)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "shap_local_explanation.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    # --------------------------------------------------------------
    # LIME
    # --------------------------------------------------------------
    print("\n=== LIME Explainability ===")

    flat_feature_names = [
        f"{lag}:{feature}"
        for lag in lag_labels
        for feature in selected_features
    ]
    n_flat_features = sequence_length * n_features

    n_lime_training = min(lime_training_size, n_train)
    lime_training_idx = np.sort(
        rng.choice(
            n_train,
            size=n_lime_training,
            replace=False
        )
    )
    lime_training_flat = X_train[
        lime_training_idx
    ].reshape(n_lime_training, -1)

    def lime_predict_fn(flat_x):
        flat_x = np.asarray(flat_x, dtype=np.float32)
        sequence_x = flat_x.reshape(
            -1, sequence_length, n_features
        )
        tensor_x = torch.tensor(
            sequence_x,
            dtype=torch.float32,
            device=device
        )
        model.eval()
        with torch.inference_mode():
            pred_scaled = model(
                tensor_x
            ).detach().cpu().numpy().reshape(-1, 1)
        return scaler_y.inverse_transform(
            pred_scaled
        ).reshape(-1)

    # IMPORTANT:
    # feature_selection="none" fits the local model with every flattened
    # lag-feature input. This avoids the previous bias where only a top-k
    # subset received coefficients and every omitted input was later treated
    # as a genuine zero during global aggregation.
    lime_explainer = LimeTabularExplainer(
        training_data=lime_training_flat,
        feature_names=flat_feature_names,
        mode="regression",
        discretize_continuous=False,
        feature_selection="none",
        random_state=random_state
    )

    # For valid global aggregation, use all flattened features.
    # A user-supplied smaller value is deliberately not used for the
    # SHAP-vs-LIME global comparison because that would reintroduce
    # sparse-selection zero bias.
    effective_lime_num_features = n_flat_features

    lime_weight_cube = np.full(
        (n_explain, sequence_length, n_features),
        np.nan,
        dtype=float
    )
    lime_local_rows = []
    lime_fidelity_rows = []

    for local_i, test_i in enumerate(explain_test_idx):
        instance_flat = X_test[test_i].reshape(-1)

        explanation = lime_explainer.explain_instance(
            instance_flat,
            lime_predict_fn,
            num_features=effective_lime_num_features,
            num_samples=lime_num_samples
        )

        # Regression explanations may expose more than one key depending
        # on LIME version. Select the key with the most coefficients.
        if not explanation.local_exp:
            raise RuntimeError(
                f"LIME returned no local coefficients for test sample {test_i}."
            )

        local_key = max(
            explanation.local_exp,
            key=lambda key: len(explanation.local_exp[key])
        )
        local_pairs = explanation.local_exp[local_key]

        # With feature_selection='none' and num_features=n_flat_features,
        # every flattened input should have one coefficient.
        if len(local_pairs) != n_flat_features:
            raise RuntimeError(
                "LIME did not return a complete coefficient vector: "
                f"got {len(local_pairs)} coefficients, "
                f"expected {n_flat_features}."
            )

        for flat_index, weight in local_pairs:
            time_index = int(flat_index // n_features)
            feature_index = int(flat_index % n_features)

            lime_weight_cube[
                local_i, time_index, feature_index
            ] = float(weight)

            lime_local_rows.append({
                "Test Sample Index": int(test_i),
                "Lag": lag_labels[time_index],
                "Feature": selected_features[feature_index],
                "Flat Feature": flat_feature_names[flat_index],
                "LIME Weight": float(weight),
                "Abs LIME Weight": abs(float(weight))
            })

        if np.isnan(lime_weight_cube[local_i]).any():
            raise RuntimeError(
                f"Incomplete LIME coefficient map for test sample {test_i}."
            )

        predicted_value = float(
            lime_predict_fn(
                instance_flat.reshape(1, -1)
            )[0]
        )
        lime_fidelity_rows.append({
            "Test Sample Index": int(test_i),
            "Model Prediction": predicted_value,
            "LIME Fidelity R2": float(explanation.score),
            "LIME Local Prediction": float(
                np.ravel(explanation.local_pred)[0]
            )
        })

    lime_local_df = pd.DataFrame(lime_local_rows)
    lime_local_df.to_csv(
        os.path.join(output_dir, "lime_local_explanations.csv"),
        index=False
    )

    lime_fidelity_df = pd.DataFrame(lime_fidelity_rows)
    lime_fidelity_df.to_csv(
        os.path.join(output_dir, "lime_fidelity.csv"),
        index=False
    )

    # Global feature importance: mean absolute local coefficient over
    # the SAME test samples and all lags.
    lime_feature_importance = np.mean(
        np.abs(lime_weight_cube), axis=(0, 1)
    )
    lime_feature_df = pd.DataFrame({
        "Feature": selected_features,
        "LIME MeanAbs": lime_feature_importance
    }).sort_values(
        "LIME MeanAbs", ascending=False
    ).reset_index(drop=True)
    lime_feature_df["LIME Rank"] = np.arange(
        1, len(lime_feature_df) + 1
    )
    lime_feature_df.to_csv(
        os.path.join(output_dir, "lime_global_feature_importance.csv"),
        index=False
    )

    lime_lag_importance = np.mean(
        np.abs(lime_weight_cube), axis=(0, 2)
    )
    lime_lag_df = pd.DataFrame({
        "Lag": lag_labels,
        "LIME MeanAbs": lime_lag_importance
    }).sort_values(
        "LIME MeanAbs", ascending=False
    ).reset_index(drop=True)
    lime_lag_df["LIME Lag Rank"] = np.arange(
        1, len(lime_lag_df) + 1
    )
    lime_lag_df.to_csv(
        os.path.join(output_dir, "lime_lag_importance.csv"),
        index=False
    )

    lime_time_feature = np.mean(
        np.abs(lime_weight_cube), axis=0
    )
    pd.DataFrame(
        lime_time_feature,
        index=lag_labels,
        columns=selected_features
    ).to_csv(
        os.path.join(output_dir, "lime_time_feature_importance.csv")
    )

    lime_plot_df = lime_feature_df.sort_values(
        "LIME MeanAbs", ascending=True
    )
    plt.figure(figsize=(9, 5))
    plt.barh(
        lime_plot_df["Feature"],
        lime_plot_df["LIME MeanAbs"]
    )
    plt.title("LIME Aggregated Global Feature Importance")
    plt.xlabel("Mean |local LIME coefficient| (target units / scaled-input unit)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "lime_global_feature_importance.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    # --------------------------------------------------------------
    # SHAP vs LIME comparison
    # --------------------------------------------------------------
    comparison_df = pd.DataFrame({
        "Feature": selected_features,
        "SHAP MeanAbs": shap_feature_importance,
        "LIME MeanAbs": lime_feature_importance
    })

    shap_total = comparison_df["SHAP MeanAbs"].sum()
    lime_total = comparison_df["LIME MeanAbs"].sum()

    comparison_df["SHAP Normalized"] = (
        comparison_df["SHAP MeanAbs"] / shap_total
        if shap_total > 0 else 0.0
    )
    comparison_df["LIME Normalized"] = (
        comparison_df["LIME MeanAbs"] / lime_total
        if lime_total > 0 else 0.0
    )

    comparison_df["SHAP Rank"] = comparison_df[
        "SHAP MeanAbs"
    ].rank(
        method="average",
        ascending=False
    )
    comparison_df["LIME Rank"] = comparison_df[
        "LIME MeanAbs"
    ].rank(
        method="average",
        ascending=False
    )

    comparison_df["Rank Difference"] = (
        comparison_df["SHAP Rank"]
        - comparison_df["LIME Rank"]
    ).abs()

    comparison_df = comparison_df.sort_values(
        "SHAP Rank"
    ).reset_index(drop=True)

    comparison_df.to_csv(
        os.path.join(output_dir, "shap_lime_feature_comparison.csv"),
        index=False
    )

    # Correlate importance ranks directly. Average ranks make ties valid.
    rank_correlation = comparison_df[
        "SHAP Rank"
    ].corr(
        comparison_df["LIME Rank"],
        method="spearman"
    )

    lime_mean_fidelity = lime_fidelity_df[
        "LIME Fidelity R2"
    ].mean()

    summary = {
        "shap_top_feature": str(
            shap_feature_df.iloc[0]["Feature"]
        ),
        "shap_top_importance": float(
            shap_feature_df.iloc[0]["SHAP MeanAbs"]
        ),
        "lime_top_feature": str(
            lime_feature_df.iloc[0]["Feature"]
        ),
        "lime_top_importance": float(
            lime_feature_df.iloc[0]["LIME MeanAbs"]
        ),
        "shap_lime_rank_correlation": float(rank_correlation),
        "lime_mean_fidelity": float(lime_mean_fidelity),
        "shap_samples_explained": int(n_explain),
        "lime_samples_explained": int(n_explain)
    }

    pd.DataFrame([summary]).to_csv(
        os.path.join(output_dir, "xai_summary.csv"),
        index=False
    )

    print("\n=== XAI Summary ===")
    print(
        f"SHAP top feature                : "
        f"{summary['shap_top_feature']}"
    )
    print(
        f"LIME top feature                : "
        f"{summary['lime_top_feature']}"
    )
    print(
        f"SHAP-LIME rank correlation      : "
        f"{summary['shap_lime_rank_correlation']:.4f}"
    )
    print(
        f"Mean LIME local fidelity (R2)   : "
        f"{summary['lime_mean_fidelity']:.4f}"
    )
    print(
        f"Common test samples explained   : {n_explain}"
    )
    print(
        f"Explainability files saved to   : {output_dir}/"
    )

    return summary


# LSTM Model
class LSTMModel(nn.Module):
    def __init__( self, input_size, hidden_size, dropout_rate ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.dropout = nn.Dropout( dropout_rate )
        self.fc = nn.Linear( hidden_size, 1 )

    def forward(self, x):
        # x:
        # (batch_size, sequence_length, features)
        out, _ = self.lstm(x)

        # out:
        # (batch_size, sequence_length, hidden_size)
        out = out[:, -1, :]

        # out:
        # (batch_size, hidden_size)
        out = self.dropout(out)
        out = self.fc(out)

        # out:
        # (batch_size, 1)
        return out

# Metric Function
#y_true: actual next day value; y_pred: predicted next day value
def compute_metrics( eval_types, y_true, y_pred, current_close, forecast_window, market, initial_capital, position_fraction ):
    y_true = np.asarray( y_true ).reshape(-1)
    y_pred = np.asarray( y_pred ).reshape(-1)
    current_close = np.asarray( current_close ).reshape(-1)
    
    r2 = EM.r2( y_true, y_pred )

    # Protect against invalid R2
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

    # Early stopping configuration
    if isinstance(patience, float):
        actual_patience = max( 1, int(patience * epochs) )
    else:
        actual_patience = int( patience )

    # Convert validation/test sets to tensors
    X_val_tensor = torch.tensor( val_data[0], dtype=torch.float32, device=device )
    y_val_tensor = torch.tensor( val_data[1], dtype=torch.float32, device=device )
    z_val_tensor = torch.tensor( val_data[2], dtype=torch.float32, device=device )
    X_test_tensor = torch.tensor( test_data[0], dtype=torch.float32, device=device )
    y_test_tensor = torch.tensor( test_data[1], dtype=torch.float32, device=device )
    z_test_tensor = torch.tensor( test_data[2], dtype=torch.float32, device=device )

    # Reconstruct full training dataset
    full_train_x_list = []
    full_train_y_list = []
    full_train_z_list = []

    for xb, yb, zb in train_loader:
        full_train_x_list.append(xb)
        full_train_y_list.append(yb)
        full_train_z_list.append(zb)

    full_train_x = torch.cat( full_train_x_list, dim=0 ).to(device)
    full_train_y = torch.cat( full_train_y_list, dim=0 ).to(device)
    full_train_z = torch.cat( full_train_z_list, dim=0 ).to(device)

    # Model Complexity
    total_params, trainable_params = get_parameter_count(model)
    macs, flops = get_computational_requirement(
        model=model,
        input_shape=( full_train_x.shape[1], full_train_x.shape[2] ),
        device=device
    )

    # Memory Measurement Setup
    process = psutil.Process( os.getpid() )
    cpu_memory_before =  process.memory_info().rss / (1024 ** 2)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats( device )

    # Training Setup
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None
    optimization_time = 0.0
    epochs_completed = 0

    # Total Training Timing
    sync_device()
    training_start = time.perf_counter()

    # Training Loop
    for epoch in range(epochs):

        # Training phase

        model.train()
        for xb, yb, zb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad( set_to_none=True )

            # Forward pass is part of total training time, but intentionally excluded from optimization time.
            predictions = model( xb ).squeeze(-1)
            loss = criterion( predictions, yb.squeeze(-1) )

            # Optimization timing
            sync_device()
            optimization_start = time.perf_counter()
            loss.backward()
            optimizer.step()
            sync_device()
            optimization_time += time.perf_counter() - optimization_start

        # Validation phase
        model.eval()
        with torch.inference_mode():
            val_predictions = model( X_val_tensor ).squeeze(-1)
            val_loss = criterion( val_predictions, y_val_tensor.squeeze(-1) ).item()
        epochs_completed = epoch + 1

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

        else:
            epochs_no_improve += 1
            if epochs_no_improve >= actual_patience:
                print( f"Early stopping at epoch " f"{epoch + 1}" )
                break

    # End Total Training Timing
    sync_device()
    training_time = time.perf_counter() - training_start

    # Restore Best Model
    if best_state is not None:
        model.load_state_dict( best_state )
        model.to(device)

    # Memory Usage
    cpu_memory_after = process.memory_info().rss / (1024 ** 2)
    cpu_memory_consumption =  cpu_memory_after - cpu_memory_before

    if torch.cuda.is_available():
        peak_gpu_memory = torch.cuda.max_memory_allocated( device ) / (1024 ** 2)
    else:
        peak_gpu_memory = 0.0

    # Inference
    model.eval()

    # Train inference
    sync_device()
    train_inference_start = time.perf_counter()

    with torch.inference_mode():
        train_output = model( full_train_x )

    sync_device()
    train_inference_time = time.perf_counter() - train_inference_start
    train_preds_scaled = train_output.squeeze(-1).cpu().numpy()
    train_preds_unscaled = scaler_y.inverse_transform( train_preds_scaled.reshape(-1, 1) )
    y_train_unscaled = scaler_y.inverse_transform( full_train_y.cpu().numpy().reshape(-1, 1) )

    # Validation inference
    sync_device()
    val_inference_start = time.perf_counter()

    with torch.inference_mode():
        val_output = model( X_val_tensor )

    sync_device()
    val_inference_time = time.perf_counter() - val_inference_start
    val_preds_scaled = val_output.squeeze(-1).cpu().numpy()
    val_preds_unscaled = scaler_y.inverse_transform(val_preds_scaled.reshape(-1, 1) )
    y_val_unscaled = scaler_y.inverse_transform( np.asarray( val_data[1] ).reshape(-1, 1) )

    # Test inference
    sync_device()
    test_inference_start = time.perf_counter()

    with torch.inference_mode():
        test_output = model( X_test_tensor )

    sync_device()
    test_inference_time = time.perf_counter() - test_inference_start
    test_preds_scaled = test_output.squeeze(-1).cpu().numpy()
    test_preds_unscaled = scaler_y.inverse_transform(test_preds_scaled.reshape(-1, 1))
    y_test_unscaled = scaler_y.inverse_transform(np.asarray(test_data[1]).reshape(-1, 1))

    # Test latency per sample
    if len(X_test_tensor) > 0:
        test_latency_per_sample = test_inference_time/ len(X_test_tensor)
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

    # Print Computational Metrics
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
    plt.figure( figsize=(12, 5) )
    plt.plot( y_val_unscaled.reshape(-1), label="Actual (Val)" )
    plt.plot( val_preds_unscaled.reshape(-1), label="Predicted (Val)" )
    plt.title( "Validation Set" )
    plt.xlabel( "Time steps" )
    plt.ylabel( "Price" )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Test Plot
    plt.figure( figsize=(12, 5) )
    plt.plot( y_test_unscaled.reshape(-1), label="Actual (Test)" )
    plt.plot( test_preds_unscaled.reshape(-1), label="Predicted (Test)" )
    plt.title( "Test Set" )
    plt.xlabel( "Time steps" )
    plt.ylabel( "Price" )
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

    return train_metrics, val_metrics, test_metrics, computational_metrics

# Main Execution

def start_experiment(ticker_choice, eval_types):
    model_name = "LSTM"
    market = "US"
    #ticker_choice=1;
    #eval_types=1
    start_date = "2014-01-01"
    end_date = "2024-12-31"
    patience = 0.2

    experiments = [
        {
            "market": "US",
            "ticker": "AAPL",
            "fixed_indicators": ["20MA", "RSI", "MACD", "Upper_BB", "CCI", "Williams_%R"],
            "window_size": 60,
            "forecast_window": 1,
            "hidden_size": 64,
            "dropout_rate": 0.2,
            "learning_rate": 0.0005,
            "batch_size": 16,
            "requested_epochs": 150,
        },
        {
            "market": "US",
            "ticker": "QCOM",
            "fixed_indicators": ["20MA", "50MA", "MACD", "Upper_BB", "ATR", "OBV"],
            "window_size": 60,
            "forecast_window": 1,
            "hidden_size": 128,
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "requested_epochs": 150,
        },
        {
            "market": "US",
            "ticker": "BNY",
            "fixed_indicators": ["50MA", "MACD", "Upper_BB", "Lower_BB", "ATR", "Williams_%R"],
            "window_size": 60,
            "forecast_window": 1,
            "hidden_size": 64,
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "requested_epochs": 150,
        },
        {
            "market": "US",
            "ticker": "C",
            "fixed_indicators": ["50MA", "MACD", "Upper_BB", "CCI", "ATR", "Williams_%R"],
            "window_size": 60,
            "forecast_window": 1,
            "hidden_size": 128,
            "dropout_rate": 0.4,
            "learning_rate": 0.001,
            "batch_size": 32,
            "requested_epochs": 150,
        },
        {
            "market": "KL",
            "ticker": "0097.KL",
            "fixed_indicators": ["20MA", "RSI", "MACD", "CCI", "ATR", "OBV"],
            "window_size": 60,
            "forecast_window": 1,
            "hidden_size": 128,
            "dropout_rate": 0.4,
            "learning_rate": 0.001,
            "batch_size": 32,
            "requested_epochs": 150,
        },
        {
            "market": "KL",
            "ticker": "0166.KL",
            "fixed_indicators": ["RSI", "MACD", "Upper_BB", "Lower_BB", "ATR", "Williams_%R"],
            "window_size": 5,
            "forecast_window": 1,
            "hidden_size": 128,
            "dropout_rate": 0.5,
            "learning_rate": 0.001,
            "batch_size": 32,
            "requested_epochs": 150,
        },
        {
            "market": "KL",
            "ticker": "1023.KL",
            "fixed_indicators": ["RSI", "Upper_BB", "CCI", "ATR", "Williams_%R", "OBV"],
            "window_size": 60,
            "forecast_window": 1,
            "hidden_size": 64,
            "dropout_rate": 0.3,
            "learning_rate": 0.001,
            "batch_size": 16,
            "requested_epochs": 150,
        },
        {
            "market": "KL",
            "ticker": "5258.KL",
            "fixed_indicators": ["50MA", "RSI", "CCI", "ATR", "Williams_%R", "OBV"],
            "window_size": 5,
            "forecast_window": 1,
            "hidden_size": 32,
            "dropout_rate": 0.3,
            "learning_rate": 0.001,
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
            hidden_size = config["hidden_size"]
            dropout_rate = config["dropout_rate"]
            learning_rate = config["learning_rate"]
            batch_size = config["batch_size"]
            requested_epochs = config["requested_epochs"]
            # Reset RNG independently for every ticker
            set_global_seed(seed=42, framework="torch")
    
            print("\n" + "=" * 100)
            print(f"EXPERIMENT {experiment_index}/{len(experiments)}: {ticker}")
            print("=" * 100)
            # Load and Process Data
            data_processor = DataPreprocessing( ticker=ticker, start_date=start_date, end_date=end_date )
            df = data_processor.add_technical_indicators()
            selected_features = ( ["Close"] + fixed_indicators )
    
            if eval_types==1:
                historical_input_windows, next_price, current_price, _, _ = data_processor.create_windowed_data( df[selected_features].values, window_size, forecast_window ) 
                X_train, X_val, X_test, y_train, y_val, y_test, z_train, z_val, z_test = data_processor.split_dataset( historical_input_windows, next_price, current_price, train_size=0.7, val_size=0.1 ) #X: historical_input_windows; y: next_price; z: current_price
            elif eval_types==2:
                historical_input_windows, _, current_price, next_price_change, _ = data_processor.create_windowed_data( df[selected_features].values, window_size, forecast_window ) 
                X_train, X_val, X_test, y_train, y_val, y_test, z_train, z_val, z_test = data_processor.split_dataset( historical_input_windows, next_price_change, current_price, train_size=0.7, val_size=0.1 ) #X: historical_input_windows; y: next_price; z: current_price
            else:
                historical_input_windows, _, current_price, _, next_price_return = data_processor.create_windowed_data( df[selected_features].values, window_size, forecast_window ) 
                X_train, X_val, X_test, y_train, y_val, y_test, z_train, z_val, z_test = data_processor.split_dataset( historical_input_windows, next_price_return, current_price, train_size=0.7, val_size=0.1 ) #X: historical_input_windows; y: next_price; z: current_price
    
            # Input Feature Scaling
            scaler = MinMaxScaler()
            train_shape = ( X_train.shape )
            X_train_flat = X_train.reshape( -1, train_shape[2] )
    
            # Fit ONLY on training data
            scaler.fit( X_train_flat )
            X_train = scaler.transform( X_train_flat ).reshape( train_shape )
            X_val = scaler.transform( X_val.reshape( -1, train_shape[2] ) ).reshape( X_val.shape )
            X_test = scaler.transform( X_test.reshape( -1, train_shape[2] ) ).reshape( X_test.shape )
    
            # Target Scaling y
            scaler_y = MinMaxScaler()
            y_train_scaled = scaler_y.fit_transform( y_train.reshape(-1, 1) )
            y_val_scaled = scaler_y.transform( y_val.reshape(-1, 1) )
            y_test_scaled = scaler_y.transform( y_test.reshape(-1, 1) )
           
    
            # DataLoader
            train_dataset = TensorDataset(
                torch.tensor( X_train, dtype=torch.float32 ),
                torch.tensor( y_train_scaled, dtype=torch.float32 ),
                torch.tensor( z_train, dtype=torch.float32 )
            )
            train_loader = DataLoader( train_dataset, batch_size=batch_size, shuffle=False )
    
            # Model
            model = LSTMModel( input_size=X_train.shape[2], hidden_size=hidden_size, dropout_rate=dropout_rate ).to(device)
    
            print( "\n=== LSTM Model Architecture ===" )
            print( model )
    
            # Optimizer and Loss
            optimizer = optim.Adam( model.parameters(), lr=learning_rate )
            criterion = nn.MSELoss()
    
            # Experiment Information
            print(
                "\nRunning experiment:"
                f"\nIndicators       : {fixed_indicators}"
                f"\nHidden Size      : {hidden_size}"
                f"\nDropout          : {dropout_rate}"
                f"\nLearning Rate    : {learning_rate}"
                f"\nBatch Size       : {batch_size}"
                f"\nWindow Size      : {window_size}"
                f"\nForecast Window  : {forecast_window}"
                f"\nRequested Epochs : {requested_epochs}\n"
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
    
            # Explainability: SHAP + LIME
            xai_summary = explain_model_shap_lime(
                model=model,
                X_train=X_train,
                X_test=X_test,
                selected_features=selected_features,
                scaler_y=scaler_y,
                ticker=ticker,
                device=device,
                shap_background_size=100,
                shap_test_size=30,
                lime_training_size=1000,
                lime_test_size=30,
                lime_num_samples=2000,
                lime_num_features=None,
                random_state=42
            )
    
            # Final Experiment Report
            result_row = {
                # Experiment Parameters
                # Model
                "Ticker": ticker,
                "Model": model_name,
    
                "Indicators": ", ".join(fixed_indicators),
                "Hidden Size": hidden_size,
                "Dropout": dropout_rate,
                "LR": learning_rate,
                "Batch Size": batch_size,
                "Window Size": window_size,
                "Forecast Window": forecast_window,
                "Requested Epochs": requested_epochs,
                "Actual Epochs": computational_metrics[ "epochs_completed" ],
    
                # Explainability Summary
                "SHAP Top Feature": xai_summary[ "shap_top_feature" ],
                "SHAP Top Importance": xai_summary[ "shap_top_importance" ],
                "LIME Top Feature": xai_summary[ "lime_top_feature" ],
                "LIME Top Importance": xai_summary[ "lime_top_importance" ],
                "SHAP-LIME Rank Correlation": xai_summary[ "shap_lime_rank_correlation" ],
                "LIME Mean Fidelity R2": xai_summary[ "lime_mean_fidelity" ],
                "SHAP Samples Explained": xai_summary[ "shap_samples_explained" ],
                "LIME Samples Explained": xai_summary[ "lime_samples_explained" ],
    
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
    
            # Convert Report to DataFrame
            result_df = pd.DataFrame( [result_row] )
    
            # Display Final Report
            print( "\n=== Final Experiment Report ===" )
            print( result_df.to_string( index=False ) )
    
            # Save Report
            output_file =  f"{ticker}_LSTM_experiment_results-{eval_types}-0.csv"
            result_df.to_csv( output_file, index=False )
            print( f"\nResults saved to: " f"{output_file}" )
    
            # Final Experiment Report
            result_row1 = {
                # Experiment Parameters
                # Model
                "Ticker": ticker,
                "Model": model_name,
    
                "Indicators": ", ".join(fixed_indicators),
                "Hidden Size": hidden_size,
                "Dropout": dropout_rate,
                "LR": learning_rate,
                "Batch Size": batch_size,
                "Window Size": window_size,
                "Forecast Window": forecast_window,
                "Requested Epochs": requested_epochs,
                "Actual Epochs": computational_metrics[ "epochs_completed" ],
    
                # Explainability Summary
                "SHAP Top Feature": xai_summary[ "shap_top_feature" ],
                "SHAP Top Importance": xai_summary[ "shap_top_importance" ],
                "LIME Top Feature": xai_summary[ "lime_top_feature" ],
                "LIME Top Importance": xai_summary[ "lime_top_importance" ],
                "SHAP-LIME Rank Correlation": xai_summary[ "shap_lime_rank_correlation" ],
                "LIME Mean Fidelity R2": xai_summary[ "lime_mean_fidelity" ],
                "SHAP Samples Explained": xai_summary[ "shap_samples_explained" ],
                "LIME Samples Explained": xai_summary[ "lime_samples_explained" ],
    
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
            output_file =  f"{ticker}_LSTM_experiment_results-{eval_types}-1.csv"
            result_df1.to_csv( output_file, index=False )
            print( f"\nResults saved to: " f"{output_file}" )
    
            # Release resources before the next ticker.
            plt.close("all")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nAll 4 US LSTM + SHAP + LIME experiments completed.")

    
# Main Execution

#if __name__ == "__main__":
#    model_name = "LSTM"
    
    #tikcer_choice 1: AAPL, 2:QCOM, 3:BNY, 4:C, 5:0097.KL, 6:0166.KL, 7:1023.KL, 8:5258.KL
    #eval_types 1:price level, 2: price change, 3: price return (yet implement)
#    start_experiment(ticker_choice=1, eval_types=1)

if len(sys.argv) > 2:
    arg1 = int(sys.argv[1])
    arg2 = int(sys.argv[2])

    #tikcer_choice 1: AAPL, 2:QCOM, 3:BNY, 4:C, 5:0097.KL, 6:0166.KL, 7:1023.KL, 8:5258.KL
    #eval_types 1:price level, 2: price change, 3: price return (yet implement)
    start_experiment(ticker_choice=arg1, eval_types=arg2)