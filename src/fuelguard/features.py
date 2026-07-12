"""Leakage-safe features for fuel-card fraud.

Every feature that depends on history uses only a card's transactions *strictly before*
the current swipe: the previous location and time (for travel speed), the previous
odometer (for fuel economy), the running centroid of prior fuelling sites (for
off-route distance), and counts over a trailing window (for velocity). A feature that
looked at a card's later swipes would be information the authorisation decision cannot
have, so it is not allowed here, and the guard is tested by prefix invariance.

Per-transaction physics features (tank fill ratio, fuel type, merchandise flag) need no
history and are exact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "gallons_vs_tank",
    "is_gasoline",
    "is_merchandise",
    "is_manual_entry",
    "speed_from_prev_mph",
    "hours_since_prev",
    "miles_from_prev",
    "implied_mpg",
    "off_route_miles",
    "off_route_ratio",
    "txns_prior_24h",
    "gallons_prior_24h",
    "amount_vs_card_mean_prior",
    "unit_price_z",
    "hour",
    "is_night",
]

_REGION_PRICE = {
    "CA": 5.20, "OR": 4.60, "WA": 4.70, "NV": 4.60, "AZ": 4.30, "NM": 4.10,
    "TX": 3.70, "OK": 3.75, "CO": 4.00, "UT": 4.10, "MO": 3.70, "KS": 3.70,
    "AR": 3.65, "TN": 3.70, "GA": 3.75, "FL": 3.95, "NC": 3.85, "IL": 4.05,
    "IN": 3.90, "OH": 3.90, "KY": 3.75, "MN": 3.95, "NE": 3.80, "PA": 4.60,
    "NY": 4.70,
}


def _haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    r = 3958.8
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float))
                              for x in (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _prior_window_count(df: pd.DataFrame, hours: float,
                        value: np.ndarray | None = None) -> np.ndarray:
    """Per-card count (or sum of `value`) over the trailing window, strictly before."""
    w = pd.Timedelta(hours=hours)
    out = np.zeros(len(df))
    vals = None if value is None else np.asarray(value, dtype=float)
    for _, idx in df.groupby("card_id", sort=False).groups.items():
        sub = df.loc[idx].sort_values("ts")
        ts = sub["ts"].to_numpy("datetime64[ns]")
        pos = np.arange(len(ts))
        left = np.searchsorted(ts, ts - w, side="left")
        orig = sub.index.to_numpy()
        if vals is None:
            out[orig] = pos - left
        else:
            pref = np.r_[0.0, np.cumsum(vals[orig])]
            out[orig] = pref[pos] - pref[left]
    return out


def build_features(df: pd.DataFrame):
    """Return (df_with_features, feature_cols): leakage-safe fuel-card features."""
    df = df.sort_values("ts").reset_index(drop=True)
    g = df.groupby("card_id", sort=False)

    prev_lat = g["lat"].shift(1)
    prev_lon = g["lon"].shift(1)
    prev_ts = g["ts"].shift(1)
    prev_odo = g["odometer"].shift(1)
    prior_n = g.cumcount()

    miles_prev = _haversine(df["lat"], df["lon"], prev_lat, prev_lon)
    hours_prev = (df["ts"] - prev_ts).dt.total_seconds() / 3600.0
    gallons = df["gallons"].to_numpy(float)

    # running centroid of prior fuelling sites for this card (strictly before)
    mean_lat_prior = (g["lat"].cumsum() - df["lat"]) / prior_n.replace(0, np.nan)
    mean_lon_prior = (g["lon"].cumsum() - df["lon"]) / prior_n.replace(0, np.nan)
    off_route = _haversine(df["lat"], df["lon"], mean_lat_prior, mean_lon_prior)
    # roaming radius: the card's own prior mean distance from its centre. Dividing by it
    # tells a regional card straying from a long-haul card doing its normal spread.
    off_series = pd.Series(off_route, index=df.index)
    prior_mean_off = ((off_series.groupby(df["card_id"]).cumsum() - off_series)
                      / prior_n.replace(0, np.nan))
    off_route_ratio = off_route / (prior_mean_off.to_numpy() + 100.0)

    amt = df["amount"].to_numpy(float)
    mean_amt_prior = (g["amount"].cumsum() - df["amount"]) / prior_n.replace(0, np.nan)

    region_price = df["state"].map(_REGION_PRICE).fillna(3.90).to_numpy(float)
    is_fuel = (df["product"] != "merchandise").to_numpy()
    unit_price = df["unit_price"].to_numpy(float)

    feats = {
        "gallons_vs_tank": gallons / df["tank_capacity"].to_numpy(float),
        "is_gasoline": df["product"].isin(["unleaded", "premium"]).astype(int).to_numpy(),
        "is_merchandise": df["product"].isin(["merchandise", "cash"]).astype(int).to_numpy(),
        "is_manual_entry": (df["entry_mode"] == "manual").astype(int).to_numpy(),
        "speed_from_prev_mph": (miles_prev / hours_prev.replace(0, np.nan)).to_numpy(),
        "hours_since_prev": hours_prev.to_numpy(),
        "miles_from_prev": miles_prev,
        "implied_mpg": (df["odometer"].to_numpy(float) - prev_odo)
                       / np.where(gallons > 1, gallons, np.nan),
        "off_route_miles": off_route,
        "off_route_ratio": off_route_ratio,
        "txns_prior_24h": _prior_window_count(df, 24),
        "gallons_prior_24h": _prior_window_count(df, 24, gallons),
        "amount_vs_card_mean_prior": amt / (mean_amt_prior.to_numpy() + 1.0),
        "unit_price_z": np.where(is_fuel, (unit_price - region_price) / 0.5, 0.0),
        "hour": df["ts"].dt.hour.to_numpy(),
        "is_night": df["ts"].dt.hour.isin([0, 1, 2, 3, 4, 5]).astype(int).to_numpy(),
    }
    feat_df = pd.DataFrame(feats, index=df.index)
    # first-seen cards have no history: fill with neutral values, not fraud-like ones
    feat_df["speed_from_prev_mph"] = feat_df["speed_from_prev_mph"].fillna(0.0)
    feat_df["hours_since_prev"] = feat_df["hours_since_prev"].fillna(72.0)
    feat_df["miles_from_prev"] = feat_df["miles_from_prev"].fillna(0.0)
    feat_df["implied_mpg"] = feat_df["implied_mpg"].fillna(6.5)
    feat_df["off_route_miles"] = feat_df["off_route_miles"].fillna(0.0)
    feat_df["off_route_ratio"] = feat_df["off_route_ratio"].fillna(1.0)
    feat_df["amount_vs_card_mean_prior"] = feat_df["amount_vs_card_mean_prior"].fillna(1.0)
    feat_df = feat_df.fillna(0.0)

    out = pd.concat([df, feat_df], axis=1)
    return out, list(feat_df.columns)
