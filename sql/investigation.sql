-- Fuel-card investigation queries, run over the raw transaction feed with DuckDB.
-- Each is a pattern an analyst runs behind an alert. None of them uses the fraud label;
-- they surface the behaviour, and the analyst judges it. The view `t` is the feed
-- (see scripts/run_investigation.py). Distances are great-circle miles.

-- name: impossible_travel
-- Consecutive fills on one card too far apart for the time between them.
WITH seq AS (
  SELECT card_id, ts, hub, lat, lon,
         lag(ts)  OVER w AS prev_ts,
         lag(hub) OVER w AS prev_hub,
         lag(lat) OVER w AS prev_lat,
         lag(lon) OVER w AS prev_lon
  FROM t
  WINDOW w AS (PARTITION BY card_id ORDER BY ts)
),
legs AS (
  SELECT card_id, prev_hub AS from_hub, hub AS to_hub, prev_ts, ts,
         2 * 3958.8 * asin(sqrt(
           pow(sin(radians(lat - prev_lat) / 2), 2) +
           cos(radians(prev_lat)) * cos(radians(lat)) *
           pow(sin(radians(lon - prev_lon) / 2), 2))) AS miles,
         date_diff('second', prev_ts, ts) / 3600.0 AS hours
  FROM seq
  WHERE prev_ts IS NOT NULL AND ts > prev_ts
)
SELECT card_id, from_hub, to_hub, ts,
       round(miles, 0) AS miles, round(hours, 2) AS hours,
       round(miles / hours, 0) AS implied_mph
FROM legs
WHERE miles / hours > 90
ORDER BY implied_mph DESC
LIMIT 15;

-- name: tank_overflow
-- Fills whose gallons exceed the truck's tank, so fuel is going somewhere else.
SELECT card_id, driver_id, ts, hub, state,
       gallons, tank_capacity,
       round(gallons / tank_capacity, 2) AS tank_ratio, amount
FROM t
WHERE gallons > tank_capacity * 1.15
ORDER BY tank_ratio DESC
LIMIT 15;

-- name: wrong_or_non_fuel
-- Gasoline or non-fuel purchases on a diesel card, summarised by card.
SELECT card_id, driver_id,
       count(*) FILTER (WHERE product IN ('unleaded', 'premium')) AS gasoline_txns,
       count(*) FILTER (WHERE product IN ('merchandise', 'cash'))  AS nonfuel_txns,
       round(sum(amount) FILTER (
         WHERE product IN ('unleaded', 'premium', 'merchandise', 'cash')), 0) AS suspect_amount
FROM t
GROUP BY card_id, driver_id
HAVING gasoline_txns + nonfuel_txns > 0
ORDER BY suspect_amount DESC
LIMIT 15;

-- name: off_route_cards
-- Cards whose fills stray far from their own centre relative to their normal spread.
WITH centre AS (
  SELECT card_id, avg(lat) AS clat, avg(lon) AS clon FROM t GROUP BY card_id
),
dist AS (
  SELECT t.card_id,
         2 * 3958.8 * asin(sqrt(
           pow(sin(radians(t.lat - c.clat) / 2), 2) +
           cos(radians(c.clat)) * cos(radians(t.lat)) *
           pow(sin(radians(t.lon - c.clon) / 2), 2))) AS miles_from_centre
  FROM t JOIN centre c USING (card_id)
)
SELECT card_id,
       round(max(miles_from_centre), 0) AS max_from_centre,
       round(avg(miles_from_centre), 0) AS avg_from_centre,
       round(max(miles_from_centre) / (avg(miles_from_centre) + 50), 2) AS stray_ratio,
       count(*) AS fills
FROM dist
GROUP BY card_id
HAVING count(*) >= 4
ORDER BY stray_ratio DESC
LIMIT 15;

-- name: rapid_repeat
-- Swipes on one card only minutes apart at essentially the same place.
WITH seq AS (
  SELECT card_id, ts, hub,
         lag(ts)  OVER w AS prev_ts,
         lag(hub) OVER w AS prev_hub
  FROM t
  WINDOW w AS (PARTITION BY card_id ORDER BY ts)
)
SELECT card_id, hub, prev_ts, ts,
       round(date_diff('second', prev_ts, ts) / 60.0, 1) AS minutes_apart
FROM seq
WHERE prev_ts IS NOT NULL
  AND hub = prev_hub
  AND date_diff('second', prev_ts, ts) BETWEEN 0 AND 1800
ORDER BY minutes_apart ASC
LIMIT 15;

-- name: overnight_manual_high
-- Individual swipes that are overnight, hand-keyed, near a full tank, and well above the
-- card's own average spend: the evasive profile that stays under the single-swipe rules
-- and is only suspicious as a combination.
WITH card_avg AS (
  SELECT card_id, avg(amount) AS avg_amt FROM t GROUP BY card_id
)
SELECT t.card_id, t.driver_id, t.ts, t.hub,
       t.gallons, t.tank_capacity,
       round(t.gallons / t.tank_capacity, 2) AS tank_ratio,
       t.entry_mode, t.amount, round(a.avg_amt, 0) AS card_avg_amount
FROM t JOIN card_avg a USING (card_id)
WHERE t.entry_mode = 'manual'
  AND extract('hour' FROM t.ts) < 6
  AND t.gallons > t.tank_capacity * 0.9
  AND t.amount > a.avg_amt * 1.3
ORDER BY t.gallons / t.tank_capacity DESC
LIMIT 15;
