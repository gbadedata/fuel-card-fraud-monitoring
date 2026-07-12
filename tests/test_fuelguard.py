"""Tests for the fuel-card feature layer and rules engine.

The load-bearing test is leakage-safety: features computed on a time-prefix must match
the features on the full data for those same early rows, so no swipe is scored with
knowledge of a card's later activity.
"""

import numpy as np
import pandas as pd

from fuelguard import (
    features,
    fuel_data,
    investigation,
    metrics,
    model,
    rules,
    scoring,
)


def _frame():
    base = pd.Timestamp("2024-01-01")
    # one card, two fills 520 miles apart one hour apart (physically impossible)
    rows = [
        # ts_offset_h, hub, state, lat, lon, product, gallons, unit_price, odo, entry
        (0.0, "Dallas", "TX", 32.78, -96.80, "diesel", 140.0, 3.70, 500000, "chip"),
        (1.0, "Houston", "TX", 29.76, -95.37, "diesel", 130.0, 3.70, 500050, "manual"),
        (30.0, "Dallas", "TX", 32.78, -96.80, "diesel", 120.0, 3.70, 500900, "chip"),
    ]
    return pd.DataFrame({
        "ts": [base + pd.Timedelta(hours=h) for h, *_ in rows],
        "card_id": ["C0"] * 3,
        "driver_id": ["D0"] * 3,
        "fleet_id": ["F0"] * 3,
        "hub": [r[1] for r in rows],
        "state": [r[2] for r in rows],
        "lat": [r[3] for r in rows],
        "lon": [r[4] for r in rows],
        "product": [r[5] for r in rows],
        "gallons": [r[6] for r in rows],
        "unit_price": [r[7] for r in rows],
        "amount": [round(r[6] * r[7], 2) for r in rows],
        "odometer": [r[8] for r in rows],
        "entry_mode": [r[9] for r in rows],
        "tank_capacity": [200.0] * 3,
        "is_fraud": [0, 1, 0],
    })


def test_features_use_no_future_info():
    df = fuel_data.mock_fuel_frame(n_fleets=8, days=40, seed=3)
    full, cols = features.build_features(df)
    for k in (200, 600, 1200):
        if k >= len(df):
            continue
        prefix, _ = features.build_features(df.iloc[:k])
        a = full[cols].iloc[:k].reset_index(drop=True).to_numpy()
        b = prefix[cols].reset_index(drop=True).to_numpy()
        assert np.allclose(a, b, equal_nan=True), f"future info leaked at k={k}"


def test_impossible_travel_speed_and_mpg():
    f, _ = features.build_features(_frame())
    f = f.sort_values("ts").reset_index(drop=True)
    # second fill: ~500 mi in 1 h, so hundreds of mph
    assert f["speed_from_prev_mph"].iloc[1] > 90
    # and 50 odometer miles on a 130 gal fill is an impossible economy
    assert f["implied_mpg"].iloc[1] < 2.5


def test_physics_features_are_exact():
    f, _ = features.build_features(_frame())
    assert abs(f["gallons_vs_tank"].iloc[0] - 140.0 / 200.0) < 1e-9
    assert f["is_gasoline"].sum() == 0
    assert f["is_manual_entry"].iloc[1] == 1


def test_gasoline_and_merchandise_flags():
    df = _frame()
    df.loc[1, "product"] = "premium"
    df.loc[2, "product"] = "merchandise"
    f, _ = features.build_features(df)
    assert f["is_gasoline"].iloc[1] == 1
    assert f["is_merchandise"].iloc[2] == 1


def test_rules_catch_every_hard_typology():
    df = fuel_data.mock_fuel_frame(seed=7)
    f, _ = features.build_features(df)
    r = rules.apply_rules(f)
    flag = dict(zip(f.index, r["rules_flag"], strict=False))
    hard = {"impossible_travel", "tank_overflow", "fuel_type_mismatch",
            "rapid_repeat", "merchandise", "implausible_mpg"}
    for ft in hard:
        idx = f.index[f["fraud_type"] == ft]
        caught = sum(flag[i] for i in idx)
        assert caught >= 0.95 * len(idx), f"{ft}: only {caught}/{len(idx)} flagged"


def test_mock_contains_every_typology():
    df = fuel_data.mock_fuel_frame(seed=7)
    types = set(df.loc[df["is_fraud"] == 1, "fraud_type"])
    assert {"impossible_travel", "tank_overflow", "fuel_type_mismatch", "off_route",
            "rapid_repeat", "merchandise", "implausible_mpg"} <= types
    # legitimate fills never exceed the tank
    legit = df[df["is_fraud"] == 0]
    assert (legit["gallons"] <= legit["tank_capacity"]).all()


def test_time_split_orders_by_time():
    df = fuel_data.mock_fuel_frame(n_fleets=10, days=40, seed=1)
    feat, _ = features.build_features(df)
    train, test = model.time_split(feat, 0.6)
    assert train["ts"].max() <= test["ts"].min()


def test_model_ranks_and_adds_value_over_rules():
    df = fuel_data.mock_fuel_frame(seed=7)
    feat, cols = features.build_features(df)
    train, test = model.time_split(feat, 0.6)
    clf = model.train_model(train, cols)
    p = model.score_model(clf, test, cols)
    y = test["is_fraud"].to_numpy()
    # well clear of the low-single-percent base rate
    assert metrics.pr_auc(y, p) > 0.5
    # the model catches evasive fraud that stays under the rule thresholds
    rt = rules.apply_rules(test)
    ev = (test["fraud_type"] == "evasive").to_numpy()
    rules_ev = rt["rules_flag"].to_numpy()[ev].mean()
    model_ev = (p[ev] >= 0.5).mean()
    assert model_ev > rules_ev


def test_hard_rule_forces_decline():
    df = fuel_data.mock_fuel_frame(n_fleets=10, days=40, seed=2)
    feat, _ = features.build_features(df)
    rt = rules.apply_rules(feat)
    dec = scoring.decide(np.zeros(len(feat)), rt)  # model says zero risk everywhere
    hard = rt["rules_hard_flag"].to_numpy() == 1
    assert (dec[hard] == "decline").all()
    clean = (~hard) & (rt["rules_flag"].to_numpy() == 0)
    assert (dec[clean] == "approve").all()


def test_value_at_budget_is_monotone():
    df = fuel_data.mock_fuel_frame(n_fleets=12, days=40, seed=3)
    feat, cols = features.build_features(df)
    train, test = model.time_split(feat, 0.6)
    clf = model.train_model(train, cols)
    test = test.copy()
    test["p"] = model.score_model(clf, test, cols)
    test["exp_loss"] = test["p"] * test["amount"]
    vb = metrics.value_at_budget(test, "exp_loss", fractions=(0.01, 0.05, 0.1, 0.2))
    r = vb["fraud_value_recall"].to_numpy()
    assert all(r[i] <= r[i + 1] + 1e-9 for i in range(len(r) - 1))


def test_investigation_queries_run(tmp_path):
    df = fuel_data.mock_fuel_frame(n_fleets=12, days=45, seed=5)
    csv = tmp_path / "feed.csv"
    df.to_csv(csv, index=False)
    res = investigation.run(csv)
    assert set(res) == {"impossible_travel", "tank_overflow", "wrong_or_non_fuel",
                        "off_route_cards", "rapid_repeat", "overnight_manual_high"}
    # queries mapping to unambiguous injected fraud should surface something
    assert len(res["impossible_travel"][1]) > 0
    assert len(res["tank_overflow"][1]) > 0
