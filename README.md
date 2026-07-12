# Fuel-Card Fraud Monitoring

Catching the fraud a fuel card actually sees, at the moment of the swipe.

A general card-fraud model treats a fuel purchase as a generic transaction and misses the
fraud that is specific to fuel: gallons that exceed the truck's tank, a diesel rig's card
buying gasoline, a card used two states apart within the hour, fuel bought but never
burned, non-fuel merchandise on a fuel-restricted card. This tool is built around those
fuel semantics. It pairs a transparent rules-and-velocity engine, the layer a risk team
deploys and can explain, with a leakage-safe model for the combinations no single rule
catches, and it ranks alerts by dollars at risk under a fixed review budget.

There is no public labelled fuel-card fraud dataset, because issuers do not release one,
so the project runs on a schema-faithful generator that models fleets running realistic
corridors and injects the fraud a fuel-card team fights. Every metric here is from that
synthetic data, and the write-up says so plainly.

## The two layers

**Rules and velocity** (`fuelguard.rules`). Fast, explainable checks that map one to one
onto fuel fraud, each carrying a plain reason an investigator or a declined driver can
read. Hard rules encode near-certain misuse: a card in two places at once, gallons beyond
the tank, the wrong fuel, fuel that was never burned, a burst of repeat swipes. A soft
rule flags off-route use, which needs corroboration because long-haul drivers legitimately
roam. This is what runs in authorisation and what stops the obvious cases without waiting
for a model.

**Model** (next layer). A gradient-boosted classifier over the same leakage-safe features,
to catch the patterns that only emerge from several weak signals together, scored into the
same review queue and priced by expected loss.

## The signals that make it fuel-aware

The feature layer (`fuelguard.features`) turns raw swipes into fuel-domain signals:

- **Travel speed** between consecutive fills on a card. A truck cannot average 90 mph
  between fuel stops, so a higher implied speed means the card is in two places at once.
- **Tank fill ratio**, gallons over the truck's tank size. Above one means fuel is going
  somewhere other than the tank.
- **Fuel economy**, miles per gallon implied by the odometer move between fills. Below a
  couple of mpg on a real fill means fuel was bought but not burned, the signature of
  siphoning or resale.
- **Off-route distance**, how far a fill is from the card's own usual operating area,
  divided by that card's normal roaming, so a regional card straying is caught while a
  long-haul card doing its normal spread is not.
- **Fuel type and product**, gasoline or non-fuel goods on a diesel card.
- **Velocity**, swipes and gallons over a trailing window, and the amount against the
  card's own baseline.

## Leakage discipline

Every history-based feature uses only a card's transactions strictly before the current
swipe, because an authorisation decision cannot see a card's later activity. A feature
that did would score well in a notebook and fail in production. The guard is a test, not a
comment: features computed on a time-prefix of the data must equal the features on the
full data for those same early rows (`tests/test_fuelguard.py`).

## Where it stands

On the mock, the hard rules alone catch every hard typology (impossible travel, tank
overflow, wrong fuel, implausible economy, rapid repeats, merchandise) at full recall with
precision around 0.8, and the soft off-route rule adds reach at lower precision. Tightening
that precision, and turning the score into a cost-ranked review queue, is what the model
layer adds next.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make test            # includes the no-lookahead guard
python run_demo.py   # features, the rules engine, and per-typology recall
```

Data-format notes are in [`data/README.md`](data/README.md). With no feed present,
everything runs on the mock.
