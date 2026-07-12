"""Evaluation for fuel-card fraud, judged the way a fraud team would judge it.

Precision-recall, not accuracy: at a low fraud rate a model that flags nothing is almost
perfectly accurate and completely useless. Value recovered at a review budget, because a
team can work only so many alerts a day and what matters is how much of the fraud dollars
those alerts recover. Per-typology recall, so a strong average never hides a fraud type the
model is blind to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(y_true, scores))


def base_rate(y_true: np.ndarray) -> float:
    return float(np.mean(y_true))


def recall_by_typology(df: pd.DataFrame, flag_col: str,
                       type_col: str = "fraud_type") -> pd.DataFrame:
    """Share of each fraud typology that is flagged."""
    fraud = df[df[type_col].astype(str) != ""]
    rows = []
    for ft, g in fraud.groupby(type_col):
        rows.append({"typology": ft, "n": len(g), "recall": float(g[flag_col].mean())})
    return pd.DataFrame(rows).sort_values("recall", ascending=False).reset_index(drop=True)


def value_at_budget(df: pd.DataFrame, score_col: str, label_col: str = "is_fraud",
                    amount_col: str = "amount", fractions=(0.01, 0.02, 0.05, 0.10)) -> pd.DataFrame:
    """Review the top fraction of transactions by score; report what fraud it recovers.

    Ranked by the score column, which for the queue is expected loss (risk times amount),
    so the figures answer: if the team works the top X% of alerts, how much of the fraud
    count and fraud value do they catch?
    """
    d = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    total_value = float(d.loc[d[label_col] == 1, amount_col].sum())
    total_count = int((d[label_col] == 1).sum())
    n = len(d)
    rows = []
    for fr in fractions:
        k = max(1, int(round(n * fr)))
        head = d.iloc[:k]
        caught_value = float(head.loc[head[label_col] == 1, amount_col].sum())
        caught_count = int((head[label_col] == 1).sum())
        rows.append({
            "budget_frac": fr,
            "alerts": k,
            "fraud_count_recall": caught_count / max(total_count, 1),
            "fraud_value_recall": caught_value / max(total_value, 1.0),
            "precision": caught_count / k,
        })
    return pd.DataFrame(rows)


def decision_tradeoff(df: pd.DataFrame, decision_col: str,
                      label_col: str = "is_fraud") -> dict:
    """Fraud caught and legitimate friction at the chosen decision thresholds."""
    declined = df[decision_col] == "decline"
    stepped = df[decision_col].isin(["decline", "step_up"])
    fraud = df[label_col] == 1
    legit = df[label_col] == 0
    return {
        "declined": int(declined.sum()),
        "decline_fraud_recall": float((declined & fraud).sum() / max(fraud.sum(), 1)),
        "false_decline_rate": float((declined & legit).sum() / max(legit.sum(), 1)),
        "stepped_or_declined": int(stepped.sum()),
        "stepped_fraud_recall": float((stepped & fraud).sum() / max(fraud.sum(), 1)),
        "stepped_friction_rate": float((stepped & legit).sum() / max(legit.sum(), 1)),
    }
