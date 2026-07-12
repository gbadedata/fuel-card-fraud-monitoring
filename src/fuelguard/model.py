"""The model layer: a gradient-boosted classifier over the leakage-safe features.

The split is by time, never at random, because a fraud model is asked to score swipes that
happen after the ones it learned from; a random split would let it train on the future of
the very cards it is tested on. The classifier is weighted for the minority class, because
fraud is a small fraction of traffic and unweighted training would simply predict "legit".

`feature_cols` is a parameter, so the same code trains on any feature set; the fuel-card
features here and the synthetic set share it unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


def time_split(df: pd.DataFrame, frac: float = 0.6, ts_col: str = "ts"):
    """Split into an earlier training frame and a later test frame."""
    ordered = df.sort_values(ts_col).reset_index(drop=True)
    cut = int(len(ordered) * frac)
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def train_model(train_df: pd.DataFrame, feature_cols: list[str],
                label_col: str = "is_fraud", seed: int = 0) -> HistGradientBoostingClassifier:
    """Fit a class-weighted gradient-boosted classifier on the given features."""
    model = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.06,
        max_iter=350,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(train_df[feature_cols].to_numpy(float), train_df[label_col].to_numpy(int))
    return model


def score_model(model: HistGradientBoostingClassifier, df: pd.DataFrame,
                feature_cols: list[str]) -> np.ndarray:
    """Return the model's fraud probability for each row."""
    return model.predict_proba(df[feature_cols].to_numpy(float))[:, 1]
