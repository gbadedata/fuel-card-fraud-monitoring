"""A transparent rules and velocity engine for fuel-card fraud.

This is the layer a risk team deploys first: fast, explainable checks that map one to one
onto the fraud a fuel card sees, each carrying a plain reason an investigator or a
declined driver can read. Hard rules encode near-certain misuse (a card in two places at
once, gallons beyond the tank, the wrong fuel, fuel that was never burned). Soft rules
encode suspicion that needs corroboration (an off-route fill, a night-time manual entry).
The model layer handles what no single rule catches; this layer handles what should never
need a model to see.

Rules read the leakage-safe features from `features.build_features`, so they never use a
card's future either.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# thresholds, kept together so they are easy to tune and audit
SPEED_MPH = 90.0          # a truck cannot average this between fuel stops
TANK_RATIO = 1.15         # gallons over tank, with slack for tank-size variance
RAPID_HOURS = 0.34        # ~20 minutes
RAPID_MILES = 15.0        # and essentially the same location
MPG_MIN = 2.5             # below this, fuel was bought but not burned
MPG_MIN_GALLONS = 20.0    # only judge economy on a real fill
OFF_ROUTE_MILES = 600.0   # absolute floor, paired with the ratio below
OFF_ROUTE_RATIO = 3.0     # and this far beyond the card's own roaming radius

HARD, SOFT = 1.0, 0.4


def _rule_masks(f: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "IMPOSSIBLE_TRAVEL": (f["speed_from_prev_mph"].to_numpy() > SPEED_MPH),
        "TANK_OVERFLOW": (f["gallons_vs_tank"].to_numpy() > TANK_RATIO),
        "FUEL_TYPE_MISMATCH": (f["is_gasoline"].to_numpy() == 1),
        "MERCHANDISE_ON_FUEL_CARD": (f["is_merchandise"].to_numpy() == 1),
        "RAPID_REPEAT": ((f["hours_since_prev"].to_numpy() < RAPID_HOURS)
                         & (f["miles_from_prev"].to_numpy() < RAPID_MILES)
                         & (f["txns_prior_24h"].to_numpy() >= 1)),
        "IMPLAUSIBLE_MPG": ((f["implied_mpg"].to_numpy() < MPG_MIN)
                            & (f["gallons"].to_numpy() > MPG_MIN_GALLONS)),
        "OFF_ROUTE": ((f["off_route_miles"].to_numpy() > OFF_ROUTE_MILES)
                      & (f["off_route_ratio"].to_numpy() > OFF_ROUTE_RATIO)),
    }


_SEVERITY = {
    "IMPOSSIBLE_TRAVEL": HARD, "TANK_OVERFLOW": HARD, "FUEL_TYPE_MISMATCH": HARD,
    "MERCHANDISE_ON_FUEL_CARD": HARD, "RAPID_REPEAT": HARD, "IMPLAUSIBLE_MPG": HARD,
    "OFF_ROUTE": SOFT,
}


def _reason(rule: str, row) -> str:
    if rule == "IMPOSSIBLE_TRAVEL":
        return (f"card used {row.miles_from_prev:.0f} mi away "
                f"{row.hours_since_prev:.1f} h earlier (implied {row.speed_from_prev_mph:.0f} mph)")
    if rule == "TANK_OVERFLOW":
        return f"{row.gallons:.0f} gal exceeds the {row.tank_capacity:.0f} gal tank"
    if rule == "FUEL_TYPE_MISMATCH":
        return f"{row.product} bought on a diesel card"
    if rule == "MERCHANDISE_ON_FUEL_CARD":
        return "non-fuel purchase on a fuel-restricted card"
    if rule == "RAPID_REPEAT":
        return f"repeat swipe {row.hours_since_prev * 60:.0f} min after the last, same site"
    if rule == "IMPLAUSIBLE_MPG":
        return f"implied {max(row.implied_mpg, 0.0):.1f} mpg on a {row.gallons:.0f} gal fill"
    if rule == "OFF_ROUTE":
        return (f"{row.off_route_miles:.0f} mi from this card's usual area, "
                f"{row.off_route_ratio:.1f}x its normal range")
    return rule


def apply_rules(f: pd.DataFrame) -> pd.DataFrame:
    """Score each transaction against the rules; return hits, reasons, and scores."""
    masks = _rule_masks(f)
    n = len(f)
    score = np.zeros(n)
    hard = np.zeros(n, dtype=bool)
    for rule, m in masks.items():
        score += m * _SEVERITY[rule]
        if _SEVERITY[rule] == HARD:
            hard |= m

    hits: list[str] = [""] * n
    reasons: list[str] = [""] * n
    triggered = np.where(score > 0)[0]
    rows = f.itertuples(index=False)
    row_list = list(rows)
    for i in triggered:
        row = row_list[i]
        fired = [r for r, m in masks.items() if m[i]]
        hits[i] = "; ".join(fired)
        reasons[i] = "; ".join(_reason(r, row) for r in fired)

    out = pd.DataFrame({
        "rule_hits": hits,
        "reasons": reasons,
        "rules_score": score,
        "rules_flag": (score > 0).astype(int),
        "rules_hard_flag": hard.astype(int),
    }, index=f.index)
    return out
