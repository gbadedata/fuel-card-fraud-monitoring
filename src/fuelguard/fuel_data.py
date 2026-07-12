"""Fuel-card transaction data: a schema-faithful mock with injected fraud, and a loader.

No public labelled fuel-card fraud dataset exists, because card issuers do not release
one, so the mock is the primary data source and the write-ups say so plainly. It models
fleets of trucks running realistic corridors between fuelling hubs, filling along their
routes at a fuel economy a Class 8 truck actually gets, and injects the fraud a fuel-card
risk team sees in practice:

  impossible_travel   the card is used two places too far apart in the time between
  tank_overflow       gallons far exceed the truck's tank, i.e. siphoning or resale
  fuel_type_mismatch  a diesel truck's card buys gasoline, i.e. a personal vehicle
  off_route           the card is used far outside the driver's usual operating region
  rapid_repeat        several swipes at one site within minutes
  merchandise         non-fuel goods or cash on a fuel-restricted card
  implausible_mpg     fuel bought but not burned, so the implied fuel economy is impossible

Legitimate traffic is made deliberately non-trivial: long-haul drivers really do roam,
fills really do approach tank size, so no single rule separates fraud cleanly.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

REF_DATE = pd.Timestamp("2024-01-01 00:00:00")

# Major US fuelling hubs: (city, state, latitude, longitude). Trucks fuel at hubs along
# their corridor; distances between hubs drive the travel and fuel-economy checks.
HUBS = [
    ("Atlanta", "GA", 33.75, -84.39), ("Dallas", "TX", 32.78, -96.80),
    ("Houston", "TX", 29.76, -95.37), ("Chicago", "IL", 41.88, -87.63),
    ("Indianapolis", "IN", 39.77, -86.16), ("Columbus", "OH", 39.96, -83.00),
    ("Kansas City", "MO", 39.10, -94.58), ("Denver", "CO", 39.74, -104.99),
    ("Salt Lake City", "UT", 40.76, -111.89), ("Phoenix", "AZ", 33.45, -112.07),
    ("Los Angeles", "CA", 34.05, -118.24), ("Oakland", "CA", 37.80, -122.27),
    ("Portland", "OR", 45.52, -122.68), ("Seattle", "WA", 47.61, -122.33),
    ("Memphis", "TN", 35.15, -90.05), ("Nashville", "TN", 36.16, -86.78),
    ("Charlotte", "NC", 35.23, -80.84), ("Jacksonville", "FL", 30.33, -81.66),
    ("Orlando", "FL", 28.54, -81.38), ("Little Rock", "AR", 34.75, -92.29),
    ("Oklahoma City", "OK", 35.47, -97.52), ("Albuquerque", "NM", 35.08, -106.65),
    ("Amarillo", "TX", 35.22, -101.83), ("St Louis", "MO", 38.63, -90.20),
    ("Louisville", "KY", 38.25, -85.76), ("Minneapolis", "MN", 44.98, -93.27),
    ("Omaha", "NE", 41.26, -95.93), ("Las Vegas", "NV", 36.17, -115.14),
    ("El Paso", "TX", 31.76, -106.49), ("San Antonio", "TX", 29.42, -98.49),
    ("Laredo", "TX", 27.53, -99.49), ("Harrisburg", "PA", 40.27, -76.88),
    ("Knoxville", "TN", 35.96, -83.92), ("Buffalo", "NY", 42.89, -78.88),
]

FUEL_PRICE = {  # rough regional diesel price per gallon, USD
    "CA": 5.20, "OR": 4.60, "WA": 4.70, "NV": 4.60, "AZ": 4.30, "NM": 4.10,
    "TX": 3.70, "OK": 3.75, "CO": 4.00, "UT": 4.10, "MO": 3.70, "KS": 3.70,
    "AR": 3.65, "TN": 3.70, "GA": 3.75, "FL": 3.95, "NC": 3.85, "IL": 4.05,
    "IN": 3.90, "OH": 3.90, "KY": 3.75, "MN": 3.95, "NE": 3.80, "PA": 4.60,
    "NY": 4.70,
}


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r * 2 * math.asin(math.sqrt(a))


_HUB_LAT = np.array([h[2] for h in HUBS])
_HUB_LON = np.array([h[3] for h in HUBS])
_DIST = np.array([[haversine_miles(a2, o2, a1, o1)
                   for a1, o1 in zip(_HUB_LAT, _HUB_LON, strict=False)]
                  for a2, o2 in zip(_HUB_LAT, _HUB_LON, strict=False)])


def _price(state: str, rng) -> float:
    return round(FUEL_PRICE.get(state, 3.90) + rng.normal(0, 0.12), 2)


def _next_hub(cur: int, target_miles: float, allowed: np.ndarray, rng) -> int:
    """The allowed hub whose distance from cur is closest to a target leg length.

    `allowed` keeps a driver inside their operating region so their fuelling clusters,
    which is what makes an off-route fill detectable rather than just more roaming.
    """
    d = _DIST[cur].copy()
    band = allowed[(d[allowed] >= 300) & (d[allowed] <= 1400) & (allowed != cur)]
    if len(band) == 0:
        cand = allowed[allowed != cur]
        if len(cand) == 0:
            return cur
        return int(cand[np.argmin(d[cand])])
    return int(band[np.argmin(np.abs(d[band] - target_miles))])


def mock_fuel_frame(n_fleets: int = 40, days: int = 60, seed: int = 7) -> pd.DataFrame:
    """Build a schema-faithful fuel-card transaction frame with injected fraud."""
    rng = np.random.default_rng(seed)
    window_s = days * 86_400
    rows = []
    drivers = []

    # driver profiles: home hub, operating radius (most run regional lanes, some are
    # long-haul), tank size, starting odometer, base fuel economy
    did = 0
    for fleet in range(n_fleets):
        for _ in range(int(rng.integers(4, 13))):
            long_haul = rng.random() < 0.25
            radius = float(rng.choice([2600, 3000]) if long_haul else rng.choice([600, 800, 1000]))
            home = int(rng.integers(0, len(HUBS)))
            allowed = np.where(_DIST[home] <= radius)[0]
            if len(allowed) < 3:
                allowed = np.argsort(_DIST[home])[:6]
            drivers.append({
                "driver_id": f"D{did:05d}",
                "card_id": f"C{did:05d}",
                "fleet_id": f"F{fleet:03d}",
                "home": home,
                "allowed": allowed,
                "tank": float(rng.choice([120, 200, 240, 300])),
                "odo": float(rng.integers(80_000, 800_000)),
                "mpg": float(rng.uniform(5.8, 7.2)),
            })
            did += 1

    def emit(d, ts, hub_i, product, gallons, odo, entry, fraud, ftype=""):
        city, state, lat, lon = HUBS[hub_i]
        unit = _price(state, rng) if product != "merchandise" else 0.0
        amount = (round(gallons * unit, 2) if product != "merchandise"
                  else round(rng.uniform(20, 400), 2))
        rows.append((REF_DATE + pd.Timedelta(seconds=float(ts)), d["card_id"], d["driver_id"],
                     d["fleet_id"], city, state, round(lat, 4), round(lon, 4), product,
                     round(float(gallons), 1), unit, amount, round(float(odo)), entry,
                     d["tank"], int(fraud), ftype))

    # --- legitimate corridor traffic ---
    for d in drivers:
        t = rng.uniform(0, 2 * 86_400)
        cur = d["home"]
        odo = d["odo"]
        while t < window_s:
            miles = float(np.clip(rng.normal(950, 160), 480, 1300))
            nxt = _next_hub(cur, miles, d["allowed"], rng)
            leg = haversine_miles(_HUB_LAT[cur], _HUB_LON[cur], _HUB_LAT[nxt], _HUB_LON[nxt])
            drive_h = leg / rng.uniform(46, 56)
            rest_h = rng.uniform(6, 12)
            t += (drive_h + rest_h) * 3600
            if t >= window_s:
                break
            odo += leg * rng.uniform(1.0, 1.15)
            gallons = min(leg / d["mpg"] * rng.uniform(0.95, 1.05), d["tank"])
            entry = "manual" if rng.random() < 0.08 else "chip"
            emit(d, t, nxt, "diesel", gallons, odo, entry, 0)
            cur = nxt
    n_legit = len(rows)

    # index legit rows by card for attaching fraud near real activity
    legit = pd.DataFrame(rows, columns=_RAW_COLS)
    by_card = {c: g for c, g in legit.groupby("card_id")}

    def random_context(exclude_home=False):
        d = drivers[int(rng.integers(0, len(drivers)))]
        g = by_card.get(d["card_id"])
        if g is None or len(g) == 0:
            return None
        anchor = g.iloc[int(rng.integers(0, len(g)))]
        return d, anchor

    def secs(ts):
        return (ts - REF_DATE).total_seconds()

    n_each = max(30, n_legit // 350)

    # impossible_travel: far hub, an hour or two after a real fill
    for _ in range(n_each):
        ctx = random_context()
        if not ctx:
            continue
        d, anchor = ctx
        far = int(np.argmax([haversine_miles(anchor["lat"], anchor["lon"], la, lo)
                             for la, lo in zip(_HUB_LAT, _HUB_LON, strict=False)]))
        t = secs(anchor["ts"]) + rng.uniform(0.5, 2.5) * 3600
        emit(d, t, far, "diesel", min(d["tank"] * rng.uniform(0.5, 0.9), d["tank"]),
             anchor["odometer"] + rng.uniform(0, 40), "manual", 1, "impossible_travel")

    # tank_overflow: gallons far exceed the tank
    for _ in range(n_each):
        ctx = random_context()
        if not ctx:
            continue
        d, anchor = ctx
        t = secs(anchor["ts"]) + rng.uniform(3, 20) * 3600
        hub_i = HUBS.index(next(h for h in HUBS if h[0] == anchor["hub"]))
        emit(d, t, hub_i, "diesel", d["tank"] * rng.uniform(2.0, 4.0),
             anchor["odometer"] + rng.uniform(400, 900), "chip", 1, "tank_overflow")

    # fuel_type_mismatch: a diesel card buys gasoline (a car being filled)
    for _ in range(n_each):
        ctx = random_context()
        if not ctx:
            continue
        d, anchor = ctx
        t = secs(anchor["ts"]) + rng.uniform(3, 20) * 3600
        hub_i = HUBS.index(next(h for h in HUBS if h[0] == anchor["hub"]))
        emit(d, t, hub_i, rng.choice(["unleaded", "premium"]), rng.uniform(10, 28),
             anchor["odometer"] + rng.uniform(0, 60), "chip", 1, "fuel_type_mismatch")

    # off_route: a fill far from the driver's usual region, normal timing
    for _ in range(n_each):
        ctx = random_context()
        if not ctx:
            continue
        d, anchor = ctx
        home_lat, home_lon = _HUB_LAT[d["home"]], _HUB_LON[d["home"]]
        far = int(np.argmax([haversine_miles(home_lat, home_lon, la, lo)
                             for la, lo in zip(_HUB_LAT, _HUB_LON, strict=False)]))
        t = secs(anchor["ts"]) + rng.uniform(30, 60) * 3600
        emit(d, t, far, "diesel", min(d["tank"] * rng.uniform(0.5, 0.9), d["tank"]),
             anchor["odometer"] + rng.uniform(200, 500), "chip", 1, "off_route")

    # rapid_repeat: several swipes at one site within minutes
    for _ in range(n_each):
        ctx = random_context()
        if not ctx:
            continue
        d, anchor = ctx
        hub_i = HUBS.index(next(h for h in HUBS if h[0] == anchor["hub"]))
        base = secs(anchor["ts"]) + rng.uniform(3, 20) * 3600
        for k in range(int(rng.integers(2, 4))):
            emit(d, base + k * rng.uniform(60, 400), hub_i, "diesel",
                 min(d["tank"] * rng.uniform(0.4, 0.8), d["tank"]),
                 anchor["odometer"] + rng.uniform(0, 20), "manual", 1, "rapid_repeat")

    # merchandise: non-fuel goods or cash on a fuel-restricted card
    for _ in range(n_each):
        ctx = random_context()
        if not ctx:
            continue
        d, anchor = ctx
        t = secs(anchor["ts"]) + rng.uniform(3, 20) * 3600
        hub_i = HUBS.index(next(h for h in HUBS if h[0] == anchor["hub"]))
        emit(d, t, hub_i, "merchandise", 0.0, anchor["odometer"] + rng.uniform(0, 20),
             "chip", 1, "merchandise")

    # implausible_mpg: fuel bought but not burned (tiny odometer move, big fill)
    for _ in range(n_each):
        ctx = random_context()
        if not ctx:
            continue
        d, anchor = ctx
        t = secs(anchor["ts"]) + rng.uniform(3, 20) * 3600
        hub_i = HUBS.index(next(h for h in HUBS if h[0] == anchor["hub"]))
        emit(d, t, hub_i, "diesel", min(d["tank"] * rng.uniform(0.8, 1.0), d["tank"]),
             anchor["odometer"] + rng.uniform(10, 70), "chip", 1, "implausible_mpg")

    # evasive: every signal kept under its rule threshold, anomalous only together.
    # A near-tank fill, at night, keyed in by hand, well above the card's usual spend,
    # with an economy that is low but not impossible. No single rule fires; the pattern
    # does. This is what the model is for.
    for _ in range(n_each):
        ctx = random_context()
        if not ctx:
            continue
        d, anchor = ctx
        base_s = secs(anchor["ts"]) + rng.uniform(4, 20) * 3600
        day = int(base_s // 86_400)
        night_ts = REF_DATE + pd.Timedelta(days=day, hours=int(rng.integers(0, 5)),
                                            minutes=int(rng.integers(0, 60)))
        hub_i = HUBS.index(next(h for h in HUBS if h[0] == anchor["hub"]))
        gal = d["tank"] * rng.uniform(0.92, 1.12)  # spans just under to just over the tank
        odo = anchor["odometer"] + gal * rng.uniform(3.0, 4.3)  # low mpg, above the floor
        emit(d, secs(night_ts), hub_i, "diesel", gal, odo, "manual", 1, "evasive")

    df = pd.DataFrame(rows, columns=_RAW_COLS)
    return df.sort_values("ts").reset_index(drop=True)


_RAW_COLS = ["ts", "card_id", "driver_id", "fleet_id", "hub", "state", "lat", "lon",
             "product", "gallons", "unit_price", "amount", "odometer", "entry_mode",
             "tank_capacity", "is_fraud", "fraud_type"]


def load_fuel(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load a fuel-card transaction CSV with the pipeline's columns."""
    df = pd.read_csv(path, nrows=nrows, parse_dates=["ts"])
    return df.sort_values("ts").reset_index(drop=True)


def write_mock_fuel(out_dir: str | Path, **kwargs) -> Path:
    """Write a mock CSV (drops the fraud_type label, as a live feed would not have it)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "mock_transactions.csv"
    mock_fuel_frame(**kwargs).drop(columns=["fraud_type"]).to_csv(path, index=False)
    return path
