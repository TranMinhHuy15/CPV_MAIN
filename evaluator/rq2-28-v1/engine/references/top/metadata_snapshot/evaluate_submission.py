import sys
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score


def score(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    merged = solution.merge(submission, on="id", how="left").fillna(0)

    return np.mean([
        average_precision_score(
            merged.loc[merged["group"] == g, "target"],
            merged.loc[merged["group"] == g, "score"]
        )
        for g in merged["group"].unique()
    ])


def evaluate(submission_csv: str, solution_csv: str):
    submission = pd.read_csv(submission_csv)
    solution = pd.read_csv(solution_csv)

    for usage in ["Public", "Private"]:
        subset = solution[solution["Usage"] == usage]
        mAP = score(subset[["id", "target", "group"]], submission[["id", "score"]])
        print(f"mAP ({usage}): {mAP:.6f}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python evaluate_submission.py submission.csv [solution.csv]")
        sys.exit(1)

    submission_csv = sys.argv[1]
    solution_csv = sys.argv[2] if len(sys.argv) == 3 else "solution.csv"

    evaluate(submission_csv, solution_csv)
