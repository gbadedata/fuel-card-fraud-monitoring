"""Fuel-card monitoring, end to end: rules, model, decision, queue, and evaluation.

Trains on the earlier part of the data and evaluates on the later part, so every number is
out of time. Runs on a real CSV at data/transactions.csv if present, else on the mock.

    python run_model.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fuelguard import features, fuel_data, metrics, model, rules, scoring


def load() -> pd.DataFrame:
    real = Path("data/transactions.csv")
    if real.exists():
        print("Loading transactions from data/transactions.csv ...")
        return fuel_data.load_fuel(real)
    print("No data/transactions.csv found; using the schema-faithful mock.\n")
    return fuel_data.mock_fuel_frame(seed=7)


def main() -> None:
    df = load()
    feat, cols = features.build_features(df)

    train, test = model.time_split(feat, frac=0.6)
    clf = model.train_model(train, cols)
    test = test.copy()
    test["model_prob"] = model.score_model(clf, test, cols)

    rules_test = rules.apply_rules(test)
    test["decision"] = scoring.decide(test["model_prob"].to_numpy(), rules_test)

    y = test["is_fraud"].to_numpy()
    print(f"Out-of-time test: {len(test):,} swipes | fraud rate {y.mean():.2%} "
          f"({int(y.sum())} fraud)\n")

    print("Ranking quality (PR-AUC, where the no-skill line is the fraud rate):")
    ra = metrics.pr_auc(y, rules_test["rules_score"].to_numpy())
    print(f"  rules score as a ranker   {ra:.3f}")
    print(f"  model                     {metrics.pr_auc(y, test['model_prob'].to_numpy()):.3f}")
    print(f"  no-skill baseline         {metrics.base_rate(y):.3f}\n")

    test["exp_loss"] = test["model_prob"].to_numpy(float) * test["amount"].to_numpy(float)
    vb = metrics.value_at_budget(test, "exp_loss")
    print("Review queue, ranked by expected loss (risk x amount):")
    print("  budget   alerts   fraud value recovered   fraud count caught")
    for r in vb.itertuples():
        print(f"  top {r.budget_frac:>4.0%}  {r.alerts:>6d}         {r.fraud_value_recall:>6.0%}"
              f"                {r.fraud_count_recall:>6.0%}")
    print()

    stats = scoring.legit_stats(train)
    flagcol = (rules_test["rules_flag"].to_numpy()
               | (test["model_prob"].to_numpy() >= 0.5)).astype(int)
    test["_flag"] = flagcol
    rec = metrics.recall_by_typology(test, "_flag")
    print("Per-typology recall (rule hit or model probability at least 0.5):")
    for r in rec.itertuples():
        print(f"  {r.typology:20s} {r.recall:>5.0%}  (n={r.n})")
    print()

    td = metrics.decision_tradeoff(test, "decision")
    print("Real-time decisions on the test stream:")
    print(f"  declined:  {td['declined']:>4d}  catching {td['decline_fraud_recall']:.0%} of fraud, "
          f"declining {td['false_decline_rate']:.2%} of legitimate swipes")
    print(f"  declined or stepped up: catching {td['stepped_fraud_recall']:.0%} of fraud, "
          f"touching {td['stepped_friction_rate']:.1%} of legitimate swipes\n")

    q = scoring.build_queue(test, test["model_prob"].to_numpy(), rules_test, stats)
    print("Top of the review queue, each with its reason:")
    for r in q.head(6).itertuples():
        tag = r.fraud_type if getattr(r, "fraud_type", "") else "legit"
        print(f"  ${r.expected_loss:>8.0f}  risk {r.risk:.2f}  [{tag}]  {r.reasons}")


if __name__ == "__main__":
    main()
