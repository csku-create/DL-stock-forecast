# Naive Persistence Baseline
# Converted from the original Linear Regression + SHAP/LIME experiment.
#
# Persistence rule:
#   eval_types == 1 (price level):  y_hat(t+h) = current Close(t)
#   eval_types == 2 (price change):  y_hat(t+h) = 0
#   eval_types == 3 (price return):  y_hat(t+h) = 0
#
# There is no model fitting, optimizer, target scaling, SHAP, or LIME.

import os
import sys
import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project Setup
project_root = "/content"

if project_root not in sys.path:
    sys.path.append(project_root)

from ReusableFunctions.DataPreprocessing import DataPreprocessing
from ReusableFunctions.EvaluationMetrics import EvaluationMetrics as EM


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


def naive_persistence_predict(eval_types, y_true, current_close):
    """
    Generate a naive persistence forecast in the same target representation
    used by compute_metrics().

    Price level:
        next price is assumed to equal the latest/current close.

    Price change:
        no change is assumed, so prediction = 0.

    Price return:
        no return is assumed, so prediction = 0.
    """
    y_true = np.asarray(y_true).reshape(-1)
    current_close = np.asarray(current_close).reshape(-1)

    if eval_types == 1:
        return current_close.astype(float, copy=True)

    if eval_types in (2, 3):
        return np.zeros_like(y_true, dtype=float)

    raise ValueError(
        "eval_types must be 1 (price level), "
        "2 (price change), or 3 (price return)."
    )


def evaluate_naive_persistence(
    train_data,
    val_data,
    test_data,
    forecast_window,
    market,
    initial_capital,
    position_fraction,
    eval_types,
):
    """
    Evaluate the persistence baseline.

    Each data tuple is:
        (y_true, current_close)

    No training is performed, therefore:
        epochs = 0
        parameters = 0
        MACs = 0
        FLOPs = 0
        optimization time = 0
    """
    process = psutil.Process(os.getpid())
    cpu_memory_before = process.memory_info().rss / (1024 ** 2)

    y_train, z_train = train_data
    y_val, z_val = val_data
    y_test, z_test = test_data

    # ------------------------------------------------------------
    # "Training" phase
    # ------------------------------------------------------------
    # A persistence model has nothing to fit.
    training_start = time.perf_counter()
    training_time = time.perf_counter() - training_start
    optimization_time = 0.0
    epochs_completed = 0

    # ------------------------------------------------------------
    # Train inference
    # ------------------------------------------------------------
    start = time.perf_counter()
    train_preds = naive_persistence_predict(
        eval_types=eval_types,
        y_true=y_train,
        current_close=z_train,
    )
    train_inference_time = time.perf_counter() - start

    # ------------------------------------------------------------
    # Validation inference
    # ------------------------------------------------------------
    start = time.perf_counter()
    val_preds = naive_persistence_predict(
        eval_types=eval_types,
        y_true=y_val,
        current_close=z_val,
    )
    val_inference_time = time.perf_counter() - start

    # ------------------------------------------------------------
    # Test inference
    # ------------------------------------------------------------
    start = time.perf_counter()
    test_preds = naive_persistence_predict(
        eval_types=eval_types,
        y_true=y_test,
        current_close=z_test,
    )
    test_inference_time = time.perf_counter() - start

    if len(test_preds) > 0:
        test_latency_per_sample = test_inference_time / len(test_preds)
    else:
        test_latency_per_sample = np.nan

    cpu_memory_after = process.memory_info().rss / (1024 ** 2)
    cpu_memory_consumption = cpu_memory_after - cpu_memory_before

    # No trainable model exists.
    total_params = 0
    trainable_params = 0
    macs = 0
    flops = 0
    peak_gpu_memory = 0.0

    # Existing predictive + profitability metrics are preserved.
    train_metrics = compute_metrics(
        eval_types,
        y_train,
        train_preds,
        z_train,
        forecast_window,
        market,
        initial_capital,
        position_fraction,
    )

    val_metrics = compute_metrics(
        eval_types,
        y_val,
        val_preds,
        z_val,
        forecast_window,
        market,
        initial_capital,
        position_fraction,
    )

    test_metrics = compute_metrics(
        eval_types,
        y_test,
        test_preds,
        z_test,
        forecast_window,
        market,
        initial_capital,
        position_fraction,
    )

    print("\n=== Train Metrics ===")
    print(
        f"RMSE: {train_metrics[0]:.4f}, "
        f"MAE: {train_metrics[1]:.4f}, "
        f"R2: {train_metrics[2]:.4f}, "
        f"Acc: {train_metrics[3]:.4f}"
    )

    print("\n=== Validation Metrics ===")
    print(
        f"RMSE: {val_metrics[0]:.4f}, "
        f"MAE: {val_metrics[1]:.4f}, "
        f"R2: {val_metrics[2]:.4f}, "
        f"Acc: {val_metrics[3]:.4f}"
    )

    print("\n=== Test Metrics ===")
    print(
        f"RMSE: {test_metrics[0]:.4f}, "
        f"MAE: {test_metrics[1]:.4f}, "
        f"R2: {test_metrics[2]:.4f}, "
        f"Acc: {test_metrics[3]:.4f}"
    )

    print("\n=== Computational Performance ===")
    print(
        f"\nEpochs completed               : {epochs_completed}"
        f"\nTraining time                  : {training_time:.9f} seconds"
        f"\nOptimization time              : {optimization_time:.9f} seconds"
        f"\nTrain inference time           : {train_inference_time:.9f} seconds"
        f"\nValidation inference time      : {val_inference_time:.9f} seconds"
        f"\nTest inference time            : {test_inference_time:.9f} seconds"
        f"\nTest latency / sample          : {test_latency_per_sample:.12f} seconds"
        f"\nTotal parameters               : {total_params}"
        f"\nTrainable parameters           : {trainable_params}"
        f"\nMACs / sample                  : {macs}"
        f"\nApprox. FLOPs / sample         : {flops}"
        f"\nPeak GPU memory                : {peak_gpu_memory:.2f} MB"
        f"\nCPU memory change              : {cpu_memory_consumption:.2f} MB"
    )

    target_name = {
        1: "Price",
        2: "Price Change",
        3: "Price Return",
    }[eval_types]

    plt.figure(figsize=(12, 5))
    plt.plot(np.asarray(y_val).reshape(-1), label="Actual (Val)")
    plt.plot(val_preds.reshape(-1), label="Naive Persistence (Val)")
    plt.title("Naive Persistence - Validation Set")
    plt.xlabel("Time steps")
    plt.ylabel(target_name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(np.asarray(y_test).reshape(-1), label="Actual (Test)")
    plt.plot(test_preds.reshape(-1), label="Naive Persistence (Test)")
    plt.title("Naive Persistence - Test Set")
    plt.xlabel("Time steps")
    plt.ylabel(target_name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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
        "epochs_completed": epochs_completed,
    }

    predictions = {
        "train": train_preds,
        "validation": val_preds,
        "test": test_preds,
    }

    return (
        train_metrics,
        val_metrics,
        test_metrics,
        computational_metrics,
        predictions,
    )


def metric_names():
    """
    Names corresponding to the 124 values returned by compute_metrics().
    """
    names = [
        "RMSE",
        "MAE",
        "R2",
        "Accuracy",
        "Profit Index",
        "Profit Index 1",
        "Profit Index 2",
        "Profit Index 3",
        "Profit Index 4",
        "Profit Index 5",
        "Profit Index 0a",
        "Profit Index 0b",
        "Profit Index 0c",
        "Profit Index 0d",
        "Profit Index 1a",
        "Profit Index 1b",
        "Profit Index 1c",
        "Profit Index 1d",
        "Profit Index 2a",
        "Profit Index 2b",
        "Profit Index 2c",
        "Profit Index 2d",
        "Profit Index 3a",
        "Profit Index 3b",
        "Profit Index 3c",
        "Profit Index 3d",
        "Profit Index 4a",
        "Profit Index 4b",
        "Profit Index 4c",
        "Profit Index 4d",
        "Profit Index 5a",
        "Profit Index 5b",
        "Profit Index 5c",
        "Profit Index 5d",
        "Profit Index 5e",
        "Profit Index 5f",
        "Profit Index 5g",
        "Profit Index 5h",
        "Profit Index 5i",
        "Profit Index 5j",
        "Profit Index 5k",
        "Profit Index 5l",
        "Profit Index 5m",
        "Profit Index 5n",
    ]

    count_labels = [
        "0", "1", "2", "3", "4", "5",
        "0a", "0b", "0c", "0d",
        "1a", "1b", "1c", "1d",
        "2a", "2b", "2c", "2d",
        "3a", "3b", "3c", "3d",
        "4a", "4b", "4c", "4d",
        "5a", "5b", "5c", "5d",
        "5e", "5f", "5g", "5h", "5i",
        "5j", "5k", "5l", "5m", "5n",
    ]

    for label in count_labels:
        names.extend([f"countp{label}", f"counts{label}"])

    if len(names) != 124:
        raise RuntimeError(
            f"Metric-name mapping has {len(names)} entries; expected 124."
        )

    return names


def add_metric_group(result_row, prefix, metrics):
    names = metric_names()

    if len(metrics) != len(names):
        raise ValueError(
            f"Expected {len(names)} metrics, received {len(metrics)}."
        )

    for name, value in zip(names, metrics):
        result_row[f"{prefix} {name}"] = value


def start_experiment(ticker_choice, eval_types):
    model_name = "Naive Persistence"
    start_date = "2014-01-01"
    end_date = "2024-12-31"

    # Keep the same ticker-specific feature sets/window sizes as the
    # original experiment so the generated samples remain aligned with
    # your other model experiments. The persistence predictor itself does
    # NOT use these indicators.
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

    if ticker_choice < 1 or ticker_choice > len(experiments):
        raise ValueError(
            f"ticker_choice must be between 1 and {len(experiments)}."
        )

    if eval_types not in (1, 2, 3):
        raise ValueError(
            "eval_types must be 1 (price level), "
            "2 (price change), or 3 (price return)."
        )

    config = experiments[ticker_choice - 1]

    market = config["market"]
    ticker = config["ticker"]
    fixed_indicators = config["fixed_indicators"]
    window_size = config["window_size"]
    forecast_window = config["forecast_window"]

    print("\n" + "=" * 100)
    print(
        f"NAIVE PERSISTENCE EXPERIMENT "
        f"{ticker_choice}/{len(experiments)}: {ticker}"
    )
    print("=" * 100)

    # ------------------------------------------------------------
    # Load and prepare exactly the same kind of windowed data
    # ------------------------------------------------------------
    data_processor = DataPreprocessing(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )

    df = data_processor.add_technical_indicators()
    selected_features = ["Close"] + fixed_indicators

    (
        historical_input_windows,
        next_price,
        current_price,
        next_price_change,
        next_price_return,
    ) = data_processor.create_windowed_data(
        df[selected_features].values,
        window_size,
        forecast_window,
    )

    if eval_types == 1:
        target = next_price
        persistence_rule = "Predicted price = current Close"
    elif eval_types == 2:
        target = next_price_change
        persistence_rule = "Predicted price change = 0"
    else:
        target = next_price_return
        persistence_rule = "Predicted price return = 0"

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        z_train,
        z_val,
        z_test,
    ) = data_processor.split_dataset(
        historical_input_windows,
        target,
        current_price,
        train_size=0.7,
        val_size=0.1,
    )

    # X_* are intentionally retained only to keep the exact same
    # sample construction/split as the other experiments.
    del X_train, X_val, X_test

    print(
        "\nRunning Naive Persistence experiment:"
        f"\nTicker           : {ticker}"
        f"\nMarket           : {market}"
        f"\nEvaluation type  : {eval_types}"
        f"\nPersistence rule : {persistence_rule}"
        f"\nWindow size      : {window_size}"
        f"\nForecast window  : {forecast_window}"
        f"\nTrain samples    : {len(y_train)}"
        f"\nValidation       : {len(y_val)}"
        f"\nTest samples     : {len(y_test)}"
    )

    (
        train_metrics,
        val_metrics,
        test_metrics,
        computational_metrics,
        predictions,
    ) = evaluate_naive_persistence(
        train_data=(y_train, z_train),
        val_data=(y_val, z_val),
        test_data=(y_test, z_test),
        forecast_window=forecast_window,
        market=market,
        initial_capital=100000,
        position_fraction=0.1,
        eval_types=eval_types,
    )

    # ------------------------------------------------------------
    # Final experiment report
    # ------------------------------------------------------------
    result_row = {
        "Model": model_name,
        "Ticker": ticker,
        "Market": market,
        "Evaluation Type": eval_types,
        "Persistence Rule": persistence_rule,
        "Indicators Used For Prediction": "None",
        "Dataset Alignment Indicators": ", ".join(fixed_indicators),
        "Window Size": window_size,
        "Forecast Window": forecast_window,
        "Requested Epochs": 0,
        "Actual Epochs": computational_metrics["epochs_completed"],
        "Train Sample": len(y_train),
        "Validation Sample": len(y_val),
        "Test Sample": len(y_test),
    }

    add_metric_group(result_row, "train", train_metrics)
    add_metric_group(result_row, "validate", val_metrics)
    add_metric_group(result_row, "test", test_metrics)

    result_row.update({
        "Training Time (s)":
            computational_metrics["training_time_sec"],
        "Optimization Time (s)":
            computational_metrics["optimization_time_sec"],
        "Train Inference Time (s)":
            computational_metrics["train_inference_time_sec"],
        "Validation Inference Time (s)":
            computational_metrics["validation_inference_time_sec"],
        "Test Inference Time (s)":
            computational_metrics["test_inference_time_sec"],
        "Test Latency / Sample (s)":
            computational_metrics["test_latency_per_sample_sec"],
        "Total Parameters":
            computational_metrics["total_parameters"],
        "Trainable Parameters":
            computational_metrics["trainable_parameters"],
        "MACs / Sample":
            computational_metrics["macs_per_sample"],
        "Approx FLOPs / Sample":
            computational_metrics["flops_per_sample"],
        "Peak GPU Memory (MB)":
            computational_metrics["peak_gpu_memory_mb"],
        "CPU Memory Change (MB)":
            computational_metrics["cpu_memory_change_mb"],
    })

    result_df = pd.DataFrame([result_row])

    print("\n=== Final Naive Persistence Experiment Report ===")
    print(result_df.to_string(index=False))

    output_file = (
        f"{ticker}_NaivePersistence_experiment_results-"
        f"{eval_types}.csv"
    )
    result_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Optional prediction-level output for checking the baseline.
    prediction_df = pd.DataFrame({
        "Actual": np.asarray(y_test).reshape(-1),
        "Current Close": np.asarray(z_test).reshape(-1),
        "Naive Prediction": predictions["test"],
    })

    prediction_file = (
        f"{ticker}_NaivePersistence_test_predictions-"
        f"{eval_types}.csv"
    )
    prediction_df.to_csv(prediction_file, index=False)
    print(f"Test predictions saved to: {prediction_file}")

    plt.close("all")

    return result_df, prediction_df


# Main Execution
# ticker_choice:
#   1=AAPL, 2=QCOM, 3=BNY, 4=C,
#   5=0097.KL, 6=0166.KL, 7=1023.KL, 8=5258.KL
#
# eval_types:
#   1=price level
#   2=price change
#   3=price return
#
# Example:
#   python naive_persistence.py 1 1

if __name__ == "__main__":
    if len(sys.argv) > 2:
        arg1 = int(sys.argv[1])
        arg2 = int(sys.argv[2])
        start_experiment(
            ticker_choice=arg1,
            eval_types=arg2,
        )
    else:
        print(
            "Usage: python naive_persistence.py "
            "<ticker_choice> <eval_types>"
        )
