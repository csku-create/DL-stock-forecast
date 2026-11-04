import os
import pandas as pd


def _resolve_metric_columns(df):
    """Prefer validation metrics for model selection; support legacy CSV schemas."""
    candidates = [
        ("validate R2", "validate Accuracy", "validate Profit Index"),
        ("val R2", "val Accuracy", "val Profit Index"),
        ("R2", "Accuracy", "Profit Index"),
    ]

    for r2_col, acc_col, profit_col in candidates:
        if all(col in df.columns for col in (r2_col, acc_col, profit_col)):
            return r2_col, acc_col, profit_col

    raise KeyError(
        "Could not find model-selection columns. Expected either "
        "['validate R2', 'validate Accuracy', 'validate Profit Index'] "
        "or legacy ['R2', 'Accuracy', 'Profit Index']. "
        f"Available columns include: {df.columns.tolist()}"
    )


def record_best_models(
    input_csv_path,
    output_dir="best_results",
    forecast_horizons=(1, 30),
    top_k=5,
):
    """
    Save the top models for each requested forecast horizon.

    Selection policy:
      1. Exclude rows with negative validation R2.
      2. Rank by validation Accuracy descending.
      3. Break ties with validation Profit Index descending.

    Validation metrics are deliberately used instead of test metrics to avoid
    selecting models on the held-out test set.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_csv_path):
        print(f"File does not exist: {input_csv_path}")
        return None

    df = pd.read_csv(input_csv_path)

    if df.empty:
        print("Input CSV is empty.")
        return None

    if "Forecast Window" not in df.columns:
        raise KeyError(
            "Missing required column 'Forecast Window'. "
            f"Available columns: {df.columns.tolist()}"
        )

    r2_col, accuracy_col, profit_col = _resolve_metric_columns(df)

    # Ensure ranking columns are numeric; invalid entries become NaN and are removed.
    for col in ("Forecast Window", r2_col, accuracy_col, profit_col):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["Forecast Window", r2_col, accuracy_col, profit_col]
    )

    # Exclude negative validation R2.
    df = df[df[r2_col] >= 0]

    if df.empty:
        print(f"No valid records found (all {r2_col} values are negative or invalid).")
        return None

    original_columns = df.columns.tolist()
    best_models = []

    for forecast_horizon in forecast_horizons:
        df_fh = df[df["Forecast Window"] == forecast_horizon]

        if df_fh.empty:
            print(f"No valid rows found for Forecast Window = {forecast_horizon}.")
            continue

        df_sorted = df_fh.sort_values(
            by=[accuracy_col, profit_col, r2_col],
            ascending=[False, False, False],
            kind="mergesort",
        )

        best_models.append(df_sorted.head(top_k))

    if not best_models:
        print(
            "No best models found for the requested forecast horizons: "
            f"{list(forecast_horizons)}"
        )
        return None

    result_df = pd.concat(best_models, ignore_index=True)
    result_df = result_df[original_columns]

    basename = os.path.basename(input_csv_path)
    ticker_symbol = basename.split("_")[0]
    filename_upper = basename.upper()

    if "CNN" in filename_upper:
        model_type = "CNN"
    elif "LSTM" in filename_upper:
        model_type = "LSTM"
    elif "LR" in filename_upper:
        model_type = "LR"
    else:
        # Fall back to the Model column when possible.
        if "Model" in result_df.columns and not result_df["Model"].dropna().empty:
            model_value = str(result_df["Model"].dropna().iloc[0]).upper()
            model_type = "LR" if "LINEAR" in model_value else model_value.replace(" ", "_")
        else:
            model_type = "UNKNOWN"

    output_path = os.path.join(
        output_dir,
        f"{ticker_symbol}_{model_type}_best.csv",
    )

    result_df.to_csv(output_path, index=False)

    print(
        f"Top {top_k} best models per available forecast window saved to: "
        f"{output_path}"
    )
    print(
        f"Selection columns: {accuracy_col} (primary), "
        f"{profit_col} (tie-break), {r2_col} (final tie-break)."
    )

    return result_df


if __name__ == "__main__":
    # Example:
    # record_best_models("AAPL_LR_experiment_results.csv")
    pass
