#!/usr/bin/env python3
"""LightGBM benchmark for the CPU fallback flow in Lab 16."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LightGBM benchmark on creditcard.csv")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("creditcard.csv"),
        help="Path to creditcard.csv dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_result.json"),
        help="Path to output benchmark JSON",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    t0 = time.perf_counter()
    df = pd.read_csv(args.data)
    load_data_seconds = time.perf_counter() - t0

    if "Class" not in df.columns:
        raise ValueError("Dataset must contain 'Class' label column")

    x = df.drop(columns=["Class"])
    y = df["Class"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1500,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=args.random_state,
        n_jobs=-1,
    )

    t1 = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_test, y_test)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    training_seconds = time.perf_counter() - t1

    y_prob = model.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    t2 = time.perf_counter()
    model.predict(x_test.iloc[[0]])
    inference_latency_1row_ms = (time.perf_counter() - t2) * 1000

    batch = x_test.iloc[:1000] if len(x_test) >= 1000 else x_test
    t3 = time.perf_counter()
    model.predict(batch)
    batch_seconds = time.perf_counter() - t3
    inference_throughput_rows_per_second = (len(batch) / batch_seconds) if batch_seconds > 0 else float("inf")

    result = {
        "dataset": str(args.data),
        "rows": int(len(df)),
        "features": int(x.shape[1]),
        "positive_ratio": float(y.mean()),
        "load_data_seconds": round(load_data_seconds, 4),
        "training_seconds": round(training_seconds, 4),
        "best_iteration": int(getattr(model, "best_iteration_", model.n_estimators)),
        "auc_roc": round(float(roc_auc_score(y_test, y_prob)), 6),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 6),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 6),
        "inference_latency_1row_ms": round(inference_latency_1row_ms, 4),
        "inference_throughput_rows_per_second": round(inference_throughput_rows_per_second, 2),
    }

    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("Benchmark completed")
    for key, value in result.items():
        print(f"- {key}: {value}")
    print(f"Saved result to {args.output}")


if __name__ == "__main__":
    main()
