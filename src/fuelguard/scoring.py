"""Combine the two layers into a decision and a queue.

The rules give near-certain calls with reasons; the model gives a probability across the
softer cases. They are combined with a noisy-or, so a swipe that trips a hard rule is high
whatever the model thinks, and a swipe several mild signals agree on rises without any rule
firing at all. The real-time decision reads that combined risk: approve, step up to a
prompt, or decline. The review queue ranks by expected loss, risk times the amount exposed,
because a team with a fixed number of reviews should see the biggest exposures first, and
every row carries the reason an investigator needs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HARD_CONF = 0.90   # a hard-rule hit is treated as near-certain
DECLINE_P = 0.80   # model probability at or above this is declined
STEP_UP_P = 0.40   # and at or above this is stepped up to verification

# interpretable features and how they read when they drive a model alert
_CONT = {
    "gallons_vs_tank": ("high", "near-tank fill"),
    "off_route_ratio": ("high", "off its usual area"),
    "speed_from_prev_mph": ("high", "fast reuse"),
    "implied_mpg": ("low", "low fuel economy"),
    "txns_prior_24h": ("high", "repeated use"),
    "amount_vs_card_mean_prior": ("high", "above usual spend"),
}
_BIN = {"is_night": "overnight", "is_manual_entry": "hand-keyed"}


def decide(model_prob: np.ndarray, rules_df: pd.DataFrame) -> np.ndarray:
    """Real-time call. A hard rule declines deterministically, whatever the model thinks;
    otherwise the model probability sets approve, step up, or decline."""
    hard = rules_df["rules_hard_flag"].to_numpy() == 1
    p = np.asarray(model_prob, float)
    return np.where(hard | (p >= DECLINE_P), "decline",
                    np.where(p >= STEP_UP_P, "step_up", "approve"))


def legit_stats(train_feat: pd.DataFrame, label_col: str = "is_fraud") -> dict:
    lg = train_feat[train_feat[label_col] == 0]
    return {c: (float(lg[c].mean()), float(lg[c].std()) + 1e-9) for c in _CONT}


def _model_reason(row, stats: dict, max_terms: int = 3) -> str:
    scored = []
    for c, (direction, phrase) in _CONT.items():
        m, s = stats[c]
        z = (getattr(row, c) - m) / s
        if direction == "low":
            z = -z
        if z > 1.5:
            scored.append((z, phrase))
    terms = [p for _, p in sorted(scored, reverse=True)[:max_terms]]
    for c, phrase in _BIN.items():
        if getattr(row, c) == 1 and phrase not in terms:
            terms.append(phrase)
    return ", ".join(terms)


def build_queue(df: pd.DataFrame, model_prob: np.ndarray, rules_df: pd.DataFrame,
                stats: dict, amount_col: str = "amount") -> pd.DataFrame:
    """Return transactions ranked by expected loss, each with a reason and a decision.

    Ranking uses the model probability, the stronger ranker; hard rules override the
    decision and contribute their reasons, but are not blended into the ranking score,
    which would import their false positives.
    """
    p = np.asarray(model_prob, float)
    q = df.reset_index(drop=True).copy()
    q["model_prob"] = p
    q["risk"] = p
    q["expected_loss"] = p * q[amount_col].to_numpy(float)
    q["decision"] = decide(p, rules_df.reset_index(drop=True))
    rule_reasons = rules_df["reasons"].reset_index(drop=True).to_numpy()

    reasons = []
    for i, row in enumerate(q.itertuples(index=False)):
        parts = []
        rr = rule_reasons[i]
        if rr:
            parts.append(rr)
        if row.model_prob >= 0.40:
            mr = _model_reason(row, stats)
            parts.append(f"model risk {row.model_prob:.2f}: {mr}" if mr
                         else f"model risk {row.model_prob:.2f}")
        reasons.append("; ".join(parts))
    q["reasons"] = reasons
    return q.sort_values("expected_loss", ascending=False).reset_index(drop=True)
