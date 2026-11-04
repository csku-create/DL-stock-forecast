import pandas as pd
import os

def record_best_models(input_csv_path, output_dir="best_results"):
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(input_csv_path):
        df = pd.read_csv(input_csv_path)

   
        # Exclude negative R2
        df = df[df["R2"] >= 0]

        if df.empty:
            print("No valid records found (all R² < 0).")
            return

        # Get original column order to preserve header structure
        columns = df.columns.tolist()
        best_models = []

        for forecast_horizon in [1, 30]:
            df_fh = df[df["Forecast Window"] == forecast_horizon]
            if df_fh.empty:
                continue

            # Sort: first by Accuracy (descending), then by Profit Index (descending)
            df_sorted = df_fh.sort_values(
                by=["Accuracy", "Profit Index"],
                ascending=[False, False]
            )

            # Take top 5
            top5 = df_sorted.head(5)
            best_models.append(top5)

        if best_models:
            result_df = pd.concat(best_models, ignore_index=True)

            # Ensure consistent column ordering
            result_df = result_df[columns]

            ticker_symbol = os.path.basename(input_csv_path).split("_")[0]

            # Detect model type from filename
            filename_upper = input_csv_path.upper()
            if "CNN" in filename_upper:
                model_type = "CNN"
            elif "LSTM" in filename_upper:
                model_type = "LSTM"
            elif "LR" in filename_upper:
                model_type = "LR"
            else:
                model_type = "UNKNOWN"

            output_path = os.path.join(output_dir, f"{ticker_symbol}_{model_type}_best.csv")
            result_df.to_csv(output_path, index=False)
            print(f"Top 5 best models (per forecast window) saved to: {output_path}")
        else:
            print("No best models found (check if forecast horizon 1/30 exists in the file).")
            
    else:
        print(f"File does not exist: {input_csv_path}")
        return

