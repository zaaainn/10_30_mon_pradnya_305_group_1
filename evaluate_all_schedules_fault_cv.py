"""Scenario-grouped cross-validation across all schedules, scored on fault rows.

All original dataset records are retained. Every fault Scenario_ID is held out
in exactly one fold. Normal readings remain available to train Random Forest
and calibrate the schedule-aware threshold baseline; therefore this evaluation
does not measure an independent false-warning rate.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

from diagnosis import FEATURE_COLUMNS, predict_records
from schedule_aware_threshold import predict_thresholds
from train_model import make_classifier

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "outputs" / "labeled_fault_scenarios.csv"
DEFAULT_PREDICTIONS = ROOT / "outputs" / "all_schedules_fault_cv_predictions.csv"
DEFAULT_METRICS = ROOT / "outputs" / "all_schedules_fault_cv_metrics.json"
CONDITION_CLASSES = ["leak", "clog"]
SEVERITY_CLASSES = ["mild", "moderate", "severe"]


def summarize_condition(actual: pd.Series, predicted: pd.Series) -> dict:
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1_leak_clog": float(f1_score(
            actual, predicted, labels=CONDITION_CLASSES, average="macro", zero_division=0
        )),
        "per_condition": {
            label: {
                "precision": float(precision_score(actual, predicted, labels=[label], average="micro", zero_division=0)),
                "recall": float(recall_score(actual, predicted, labels=[label], average="micro", zero_division=0)),
                "support": int((actual == label).sum()),
            }
            for label in CONDITION_CLASSES
        },
        "missed_fault_count": int((predicted == "normal").sum()),
        "missed_fault_rate": float((predicted == "normal").mean()),
    }


def evaluate(dataset_path: Path) -> tuple[pd.DataFrame, dict]:
    data = pd.read_csv(dataset_path)
    normals = data.loc[data["Condition"] == "normal"].copy()
    faults = data.loc[data["Condition"] != "normal"].copy()
    if len(faults) != 324:
        raise ValueError(f"Expected 324 fault records, found {len(faults)}.")

    splitter = GroupKFold(n_splits=5)
    rf_predictions = []
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(faults, faults["Condition"], groups=faults["Scenario_ID"]), start=1
    ):
        train_faults = faults.iloc[train_idx].copy()
        test_faults = faults.iloc[test_idx].copy()
        condition_train = pd.concat([normals, train_faults], ignore_index=True)
        models = {
            "condition_model": make_classifier(),
            "location_model": make_classifier(),
            "severity_model": make_classifier(),
        }
        models["condition_model"].fit(condition_train[FEATURE_COLUMNS], condition_train["Condition"])
        models["location_model"].fit(train_faults[FEATURE_COLUMNS], train_faults["Fault_Location"])
        models["severity_model"].fit(train_faults[FEATURE_COLUMNS], train_faults["Severity"])
        fold_predictions = predict_records(test_faults, models)
        fold_predictions["CV_Fold"] = fold
        rf_predictions.append(fold_predictions)

    rf = pd.concat(rf_predictions).sort_index()
    # Use known normal readings only as calibration references for the rules.
    threshold = predict_thresholds(faults, baseline_data=normals)
    result = faults.copy()
    result["CV_Fold"] = rf["CV_Fold"].to_numpy()
    result["RF_Predicted_Condition"] = rf["Predicted_Condition"].to_numpy()
    result["RF_Predicted_Location"] = rf["Predicted_Fault_Location"].to_numpy()
    result["RF_Predicted_Severity"] = rf["Predicted_Severity"].to_numpy()
    for col in ["Threshold_Alert", "Threshold_Condition", "Threshold_Location"]:
        result[col] = threshold[col].to_numpy()

    actual_condition = result["Condition"]
    rf_condition = result["RF_Predicted_Condition"]
    threshold_condition = result["Threshold_Condition"]
    metrics = {
        "evaluation": "5-fold GroupKFold by Scenario_ID, fault records only",
        "record_count": int(len(result)),
        "fault_scenario_count": int(result["Scenario_ID"].nunique()),
        "schedules": sorted(result["Schedule"].unique().tolist()),
        "normal_record_count_used_for_training_and_calibration": int(len(normals)),
        "normal_records_in_test": 0,
        "false_warning_rate": None,
        "false_warning_note": "Not measured: all normal records were retained for Random Forest training and threshold baseline calibration.",
        "scenario_grouping_note": "All hourly records for a Scenario_ID stay in the same fold. Each fault record is evaluated out of fold once.",
        "random_forest": summarize_condition(actual_condition, rf_condition),
        "schedule_aware_threshold": summarize_condition(actual_condition, threshold_condition),
        "fault_location_accuracy": {
            "random_forest": float((result["Fault_Location"] == result["RF_Predicted_Location"]).mean()),
            "schedule_aware_threshold": float((result["Fault_Location"] == result["Threshold_Location"]).mean()),
        },
        "random_forest_severity": {
            "exact_accuracy": float((result["Severity"] == result["RF_Predicted_Severity"]).mean()),
            "macro_f1": float(f1_score(
                result["Severity"], result["RF_Predicted_Severity"],
                labels=SEVERITY_CLASSES, average="macro", zero_division=0,
            )),
        },
    }
    return result.reset_index(drop=True), metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    args = parser.parse_args()
    predictions, metrics = evaluate(args.dataset)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.predictions, index=False, float_format="%.6f")
    args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Scenario-grouped evaluation: {metrics['record_count']} fault records across {metrics['fault_scenario_count']} scenarios")
    for key, name in [("random_forest", "Random Forest"), ("schedule_aware_threshold", "Schedule-aware threshold")]:
        m = metrics[key]
        print(f"{name}: accuracy={m['accuracy']:.3%}, macro F1 (leak/clog)={m['macro_f1_leak_clog']:.3%}, missed={m['missed_fault_count']}/{metrics['record_count']}")
    print("Fault-location accuracy:", metrics["fault_location_accuracy"])
    print(metrics["false_warning_note"])


if __name__ == "__main__":
    main()
