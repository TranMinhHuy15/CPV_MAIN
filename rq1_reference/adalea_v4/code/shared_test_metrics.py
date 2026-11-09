"""Exact TOP_V2/V6 matched scalar test evaluator bodies; see provenance.json."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

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
