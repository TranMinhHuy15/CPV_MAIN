"""Never merge validation proxies with held-out test metrics or declare a paper win automatically."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from repro import PAPER_VALUES, read_csv, write_json


def partial_auc_raw_normalized(labels, scores, max_fpr=0.1):
    fpr, tpr, _ = roc_curve(labels, scores)
    stop = np.searchsorted(fpr, max_fpr, side="right")
    x = np.append(fpr[:stop], max_fpr)
    y = np.append(tpr[:stop], np.interp(max_fpr, fpr, tpr))
    return float(np.trapz(y, x) / max_fpr)


def validate_submission(solution, predictions):
    expected = {r["id"] for r in solution}
    actual = [r["id"] for r in predictions]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Test submission IDs must match ground truth exactly, without duplicates.")
    if not all(np.isfinite(float(r["score"])) and 0 <= float(r["score"]) <= 1 for r in predictions):
        raise ValueError("Submission must contain finite probabilities in [0,1].")


def grouped_test_metrics(solution, predictions):
    validate_submission(solution, predictions)
    lookup = {r["id"]: float(r["score"]) for r in predictions}
    output = {}
    for usage in ("Public", "Private"):
        per_group = []
        for group in ("0", "1", "2"):
            records = [r for r in solution if r["Usage"] == usage and r["group"] == group]
            labels = [int(r["target"]) for r in records]
            scores = [lookup[r["id"]] for r in records]
            if len(set(labels)) != 2:
                raise ValueError("Each evaluation group needs both classes.")
            per_group.append({
                "group": int(group), "lead_time_s": [0.5, 1.0, 1.5][int(group)], "n": len(records),
                "AP": float(average_precision_score(labels, scores)),
                "AUC": float(roc_auc_score(labels, scores)),
                "pAUC_sklearn_standardized": float(roc_auc_score(labels, scores, max_fpr=0.1)),
                "pAUC_raw_divided_by_0.1": partial_auc_raw_normalized(labels, scores),
            })
        output[usage] = {
            "mAP": float(np.mean([r["AP"] for r in per_group])),
            "mAUC_grouped": float(np.mean([r["AUC"] for r in per_group])),
            "mPAUC_sklearn_standardized": float(np.mean([r["pAUC_sklearn_standardized"] for r in per_group])),
            "mPAUC_raw_divided_by_0.1": float(np.mean([r["pAUC_raw_divided_by_0.1"] for r in per_group])),
            "groups": per_group,
            "mTTA": None,
            "mTTA_reason": "Official clipped-test scalar evaluator does not define/reconstruct continuous-curve mTTA.",
        }
    return output


def create_reports(directory, snapshot):
    directory, snapshot = Path(directory), Path(snapshot)
    status = json.loads((directory / "training_status.json").read_text())
    if not status["complete"]:
        raise RuntimeError("Held-out test metrics require completion of the declared training schedule.")
    values = json.loads((directory / "validation_metrics.json").read_text())
    predictions = json.loads((directory / "test_predictions.json").read_text())
    solution = read_csv(snapshot / "solution.csv")  # First test-label access, after checkpoint choice is frozen.
    metrics = grouped_test_metrics(solution, predictions)
    submission = directory / "submission.csv"
    with submission.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "score"])
        writer.writeheader()
        writer.writerows(predictions)
    # Execute the exact Nexar evaluator, rather than claiming our reimplementation is official.
    result = subprocess.run(
        [sys.executable, str(snapshot / "evaluate_submission.py"), str(submission), str(snapshot / "solution.csv")],
        check=True, text=True, capture_output=True,
    )
    (directory / "official_evaluator_stdout.txt").write_text(result.stdout, encoding="utf-8")
    print(result.stdout, flush=True)
    for line in result.stdout.splitlines():
        if line.startswith("mAP ("):
            usage = line.split("(")[1].split(")")[0]
            if abs(float(line.rsplit(":", 1)[1]) - metrics[usage]["mAP"]) > 1e-6:
                raise RuntimeError("Official evaluator and grouped metric implementation disagree.")
    write_json(directory / "official_test_metrics.json", metrics)
    rows = []
    mapping = {"mAP": "mAP@", "mAUC": "mAUC@", "mAUC_0.1": "mAUC@0.1", "mTTA_0.1_s": "mtta@0.1"}
    for name, key in mapping.items():
        measured = float(values[key])
        rows.append({
            "scope": "author_300_video_validation_proxy", "metric": name,
            "paper_table1_nexar": PAPER_VALUES[name], "reproduced": measured,
            "numerical_delta_only": measured - PAPER_VALUES[name], "directly_comparable": False,
            "reason": "Validation scope; paper-vs-code metric definitions unresolved. Not a claimed paper gap/win.",
        })
    for usage in ("Public", "Private"):
        measured = metrics[usage]["mAP"]
        rows.append({
            "scope": f"official_nexar_test_{usage.lower()}", "metric": "mAP",
            "paper_table1_nexar": PAPER_VALUES["mAP"], "reproduced": measured,
            "numerical_delta_only": measured - PAPER_VALUES["mAP"], "directly_comparable": False,
            "reason": "Official held-out mAP; paper does not resolve Public/Private versus validation-window scoring.",
        })
    with (directory / "paper_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(directory / "evaluation_notes.json", {
        "selected_checkpoint": status["best_checkpoint"], "completed_epochs": status["completed_epochs"],
        "test_was_not_used_for_selection": True,
        "partial_auc": "Both normalizations reported on exactly the same predictions; do not pick the closest one to the paper.",
        "code_mtta": "Original code averages over negative-frame thresholds, counts misses as zero, clips after alert onset.",
        "paper_mtta": "Paper wording refers to true-positive alarms; equivalence to source implementation is unverified.",
        "test_mtta": "Not fabricated from a scalar clipped-test score; use validation risk-curve proxy, labeled separately.",
        "paper_score_0.472": "Code standardized partial AUC and paper value cannot be assumed to use the same normalization.",
        "single_seed": "One baseline run, not a statistically supported replication across seeds.",
    })
    print("Reports saved. Numerical comparisons are labeled provisional, not automatic paper reproduction claims.")
