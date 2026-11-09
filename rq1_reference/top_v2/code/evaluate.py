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

from repro import CONTEXT_VALUES, PAPER_VALUES, read_csv, write_json


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
    for usage in ("Combined", "Public", "Private"):
        per_group = []
        for group in ("0", "1", "2"):
            records = [r for r in solution if (usage == 'Combined' or r['Usage'] == usage) and r['group'] == group]
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
    # Same validation metric names/definitions as V6, including full AUC for selection.
    mapping = {"mAP": "mAP@", "mAUC": "mAUC@", "mAUC_0.1": "mAUC@0.1", "mTTA_0.1_s": "mtta@0.1"}
    for reference, published in CONTEXT_VALUES.items():
        for name, key in mapping.items():
            measured = float(values[key])
            rows.append({
                "scope": "author_300_video_validation_proxy", "metric": name, "context_reference": reference,
                "published_value": published[name], "measured": measured,
                "numerical_delta_only": measured - published[name], "directly_comparable": False,
                "reason": "No published TOP-on-Nexar result exists; validation scope; definitions unresolved. Context only.",
            })
        for usage in ("Combined", "Public", "Private"):
            measured = metrics[usage]["mAP"]
            rows.append({
                "scope": f"official_nexar_test_{usage.lower()}", "metric": "mAP", "context_reference": reference,
                "published_value": published["mAP"], "measured": measured,
                "numerical_delta_only": measured - published["mAP"], "directly_comparable": False,
                "reason": "Official held-out mAP; no published TOP-on-Nexar result; scoring window of the reference is unresolved. Context only.",
            })
    with (directory / "paper_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(directory / "evaluation_notes.json", {
        "selected_checkpoint": status["best_checkpoint"], "completed_epochs": status["completed_epochs"],
        "test_was_not_used_for_selection": True,
        "partial_auc": "Both normalizations reported on exactly the same predictions; do not pick the closest one to a reference.",
        "code_mtta": "Released code averages over negative-frame thresholds, counts misses as zero, clips after alert onset.",
        "test_mtta": "Not fabricated from a scalar clipped-test score; use validation risk-curve proxy, labeled separately.",
        "no_published_target": "TOP publishes no Nexar numbers; the context rows are other methods from the RiskProp paper, not a reproduction target.",
        "test_score_rule": "max over the 20 horizon scores of the last snippet (released AnticipationMetric rule).",
        "single_seed": "One run, not a statistically supported estimate across seeds.",
    })
    print("Reports saved. Numerical comparisons are labeled provisional, not automatic paper reproduction claims.")
