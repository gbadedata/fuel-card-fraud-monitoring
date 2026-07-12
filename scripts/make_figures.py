"""Generate the figures used in the README, from the current rules-layer pipeline.

    python scripts/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fuelguard import features, fuel_data, model, rules

BLUE, RED, GREY = "#4C72B0", "#C44E52", "#B0B0B0"
OUT = Path("docs/img")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "axes.axisbelow": True,
})


def signal_panels(feat):
    legit = feat[feat["is_fraud"] == 0]
    panels = [
        ("impossible_travel", "speed_from_prev_mph", "implied speed (mph)", True),
        ("tank_overflow", "gallons_vs_tank", "gallons / tank", False),
        ("implausible_mpg", "implied_mpg", "implied mpg", False),
        ("off_route", "off_route_ratio", "off-route / usual range", False),
        ("rapid_repeat", "txns_prior_24h", "swipes in prior 24h", False),
        ("fuel_type_mismatch", "is_gasoline", "gasoline on diesel card", None),
        ("merchandise", "is_merchandise", "non-fuel on fuel card", None),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2))
    axes = axes.ravel()
    for ax, (ftype, col, label, logy) in zip(axes, panels, strict=False):
        fr = feat[feat["fraud_type"] == ftype]
        if logy is None:  # binary signal
            ax.bar(["legit", "fraud"], [legit[col].mean(), fr[col].mean()],
                   color=[BLUE, RED], width=0.6)
            ax.set_ylim(0, 1.05)
        else:
            data = [legit[col].clip(lower=0.01 if logy else None).to_numpy(),
                    fr[col].clip(lower=0.01 if logy else None).to_numpy()]
            bp = ax.boxplot(data, tick_labels=["legit", "fraud"], patch_artist=True,
                            widths=0.55, showfliers=False)
            for patch, c in zip(bp["boxes"], [BLUE, RED], strict=False):
                patch.set_facecolor(c)
                patch.set_alpha(0.75)
            for med in bp["medians"]:
                med.set_color("black")
            if logy:
                ax.set_yscale("log")
        ax.set_title(ftype.replace("_", " "), fontsize=10.5, pad=6)
        ax.set_ylabel(label, fontsize=9)
    axes[-1].axis("off")
    axes[-1].text(0.02, 0.82, "Each fuel-fraud type\nseparates on the signal\nbuilt for it.",
                  fontsize=11, va="top", weight="bold")
    axes[-1].text(0.02, 0.34, "Boxes: legit vs fraud\non the leakage-safe\nfeature for that type.",
                  fontsize=9, va="top", color="#444")
    fig.suptitle("Fuel-aware signals separate fuel fraud from ordinary fuelling",
                 fontsize=13.5, weight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "typology_signals.png", bbox_inches="tight")
    plt.close(fig)


def corridor_map(df):
    # a regional card with an off-route fraud, to show what off-route means
    off_cards = df.loc[df["fraud_type"] == "off_route", "card_id"].value_counts().index
    card = None
    for c in off_cards:
        if (df["card_id"] == c).sum() >= 12:
            card = c
            break
    card = card or off_cards[0]
    g = df[df["card_id"] == card].sort_values("ts")
    legit = g[g["is_fraud"] == 0]
    fraud = g[g["is_fraud"] == 1]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(fuel_data._HUB_LON, fuel_data._HUB_LAT, s=14, color=GREY,
               label="fuelling hubs", zorder=1)
    ax.plot(legit["lon"], legit["lat"], "-", color=BLUE, alpha=0.5, lw=1, zorder=2)
    ax.scatter(legit["lon"], legit["lat"], s=42, color=BLUE, zorder=3,
               label="this card's normal fuelling")
    marks = {"off_route": ("*", 320, "off-route fill"),
             "impossible_travel": ("X", 160, "impossible travel"),
             "tank_overflow": ("P", 150, "tank overflow")}
    seen = set()
    for row in fraud.itertuples():
        m, s, lab = marks.get(row.fraud_type, ("D", 120, row.fraud_type))
        ax.scatter(row.lon, row.lat, marker=m, s=s, color=RED, zorder=4,
                   edgecolor="black", linewidth=0.5,
                   label=lab if lab not in seen else None)
        seen.add(lab)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"One card's fuelling: a regional lane, and the swipes that fall outside it\n"
                 f"(card {card})", fontsize=12, weight="bold")
    ax.legend(loc="lower left", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "card_corridor.png", bbox_inches="tight")
    plt.close(fig)


def _scored_test():
    df = fuel_data.mock_fuel_frame(seed=7)
    feat, cols = features.build_features(df)
    train, test = model.time_split(feat, frac=0.6)
    clf = model.train_model(train, cols)
    test = test.copy()
    test["model_prob"] = model.score_model(clf, test, cols)
    rules_test = rules.apply_rules(test)
    test["rules_score"] = rules_test["rules_score"].to_numpy()
    test["rules_flag"] = rules_test["rules_flag"].to_numpy()
    return test


def value_curve(test):
    amt = test["amount"].to_numpy(float)
    fraud = test["is_fraud"].to_numpy()
    total_val = float(amt[fraud == 1].sum())
    n = len(test)
    x = np.arange(1, n + 1) / n

    def recovered(rank_key):
        order = np.argsort(-rank_key)
        cum = np.cumsum((amt * fraud)[order])
        return cum / total_val

    model_rec = recovered(test["model_prob"].to_numpy() * amt)
    rules_rec = recovered(test["rules_score"].to_numpy() + 1e-6 * amt)
    amt_rec = recovered(amt)

    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.plot(x * 100, model_rec * 100, color=BLUE, lw=2.4,
            label="model, ranked by expected loss")
    ax.plot(x * 100, rules_rec * 100, color="#8172B3", lw=2.0, ls="--",
            label="rules score, ranked")
    ax.plot(x * 100, amt_rec * 100, color=GREY, lw=1.6, ls=":",
            label="largest amounts first")
    ax.axvline(2, color=RED, lw=1, alpha=0.7)
    at2 = np.interp(2, x * 100, model_rec * 100)
    ax.annotate(f"top 2% of swipes\nrecover {at2:.0f}% of fraud value",
                xy=(2, at2), xytext=(6, at2 - 22), fontsize=10,
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 102)
    ax.set_xlabel("share of swipes sent to review (%)")
    ax.set_ylabel("fraud value recovered (%)")
    ax.set_title("A small review budget recovers most of the fraud value",
                 fontsize=13, weight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "value_at_budget.png", bbox_inches="tight")
    plt.close(fig)


def recall_comparison(test):
    order = ["impossible_travel", "tank_overflow", "fuel_type_mismatch", "implausible_mpg",
             "rapid_repeat", "merchandise", "off_route", "evasive"]
    rules_r, model_r, labels = [], [], []
    for ft in order:
        sub = test[test["fraud_type"] == ft]
        if not len(sub):
            continue
        labels.append(ft.replace("_", " "))
        rules_r.append(sub["rules_flag"].mean())
        model_r.append((sub["model_prob"] >= 0.5).mean())

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.barh(y - 0.2, np.array(rules_r) * 100, height=0.4, color="#8172B3",
            label="rules only")
    ax.barh(y + 0.2, np.array(model_r) * 100, height=0.4, color=BLUE,
            label="model (prob >= 0.5)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("recall (%)")
    ax.set_title("Where the model earns its place: the fraud the rules let pass",
                 fontsize=13, weight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.text(2, len(labels) - 0.5, "evasive fraud stays under every rule threshold;\n"
            "the model catches it from the joint pattern",
            fontsize=9, color="#444", va="top")
    fig.tight_layout()
    fig.savefig(OUT / "recall_rules_vs_model.png", bbox_inches="tight")
    plt.close(fig)


def main():
    df = fuel_data.mock_fuel_frame(seed=7)
    feat, _ = features.build_features(df)
    signal_panels(feat)
    corridor_map(df)
    test = _scored_test()
    value_curve(test)
    recall_comparison(test)
    print("wrote 4 figures to docs/img/")


if __name__ == "__main__":
    main()
