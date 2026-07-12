"""Fuel-card monitoring demo: features, the rules engine, and where it works.

Runs on a real transaction CSV at data/transactions.csv if present, else on the mock.

    python run_demo.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fuelguard import features, fuel_data, rules


def load() -> pd.DataFrame:
    real = Path("data/transactions.csv")
    if real.exists():
        print("Loading transactions from data/transactions.csv ...")
        return fuel_data.load_fuel(real)
    print("No data/transactions.csv found; using the schema-faithful mock.")
    return fuel_data.mock_fuel_frame(seed=7)


def main() -> None:
    df = load()
    fraud_val = df.loc[df["is_fraud"] == 1, "amount"].sum()
    print(f"  {len(df):,} transactions | {df['card_id'].nunique()} cards | "
          f"{df['fleet_id'].nunique()} fleets | fraud rate {df['is_fraud'].mean():.2%} | "
          f"fraud value ${fraud_val:,.0f}\n")

    f, cols = features.build_features(df)
    print(f"Leakage-safe features: {len(cols)} (no card's future is used).")

    r = rules.apply_rules(f)
    out = pd.concat([f[["is_fraud", "fraud_type", "amount"]], r], axis=1)
    tp = int(((out["rules_flag"] == 1) & (out["is_fraud"] == 1)).sum())
    flagged = int(out["rules_flag"].sum())
    hard = out["rules_hard_flag"] == 1
    hp = int((hard & (out["is_fraud"] == 1)).sum())
    tot = int(out["is_fraud"].sum())
    print(f"  hard rules: {int(hard.sum())} alerts, precision {hp / max(int(hard.sum()), 1):.2f}, "
          f"recall {hp / max(tot, 1):.2f}")
    print(f"  all rules (incl. soft off-route): {flagged} alerts, "
          f"precision {tp / max(flagged, 1):.2f}, recall {tp / max(tot, 1):.2f}\n")

    if "fraud_type" in df.columns:
        print("Per-typology recall from rules alone:")
        for ft in ["impossible_travel", "tank_overflow", "fuel_type_mismatch",
                   "implausible_mpg", "rapid_repeat", "merchandise", "off_route"]:
            sub = out[out["fraud_type"] == ft]
            if len(sub):
                print(f"  {ft:20s} {sub['rules_flag'].mean():.0%}")
        print()

    print("Sample alerts, each with the reason a driver or investigator would see:")
    shown = 0
    for row in out[out["rules_flag"] == 1].itertuples():
        if row.fraud_type:
            print(f"  [{row.fraud_type}] {row.reasons}")
            shown += 1
        if shown >= 5:
            break
    print("\nThe soft off-route rule trades precision for reach; combining these signals "
          "in a\nsupervised model, and pricing alerts by dollars at risk, is the next step.")


if __name__ == "__main__":
    main()
