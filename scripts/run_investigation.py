"""Run the fuel-card investigation queries and print each result.

Uses data/transactions.csv if present, else writes the mock to a temporary CSV.

    python scripts/run_investigation.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fuelguard import fuel_data, investigation


def feed_path() -> tuple[str, bool]:
    real = Path("data/transactions.csv")
    if real.exists():
        return str(real), False
    tmp = Path(tempfile.gettempdir()) / "fuelguard_feed.csv"
    fuel_data.mock_fuel_frame(seed=7).to_csv(tmp, index=False)
    return str(tmp), True


def main() -> None:
    path, is_mock = feed_path()
    print(f"Investigation over the {'mock feed' if is_mock else 'feed'}: {path}\n")
    results = investigation.run(path)
    for name, (desc, df) in results.items():
        print("=" * 80)
        print(f"{name}  --  {desc}")
        print("=" * 80)
        if df.empty:
            print("(no rows)\n")
            continue
        print(df.head(15).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
