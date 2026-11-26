"""P1: Statcast data pipeline, chronological split, and baselines.

Builds one shared snapshot so every stage trains and evaluates on the same pitches:
    data/statcast_2026.csv.gz   (gitignored; share it through Google Drive)
Each row is one 2026 regular-season pitch with its 4-category label (`pitch_cat`), the
features from P3 (count, same_hand, prev1-prev5), every raw Statcast column, and a `split`
column: train / dev / test = earliest 70% / next 15% / latest 15% of games by date.

Run from the repository root:
    python P1/pipeline.py               # baselines on dev
    python P1/pipeline.py --split test  # held-out final evaluation: run once, at the end
Other stages load the identical split with:
    from P1.pipeline import load_snapshot
    df = load_snapshot(); train = df[df["split"] == "train"]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
# Gzipped CSV rather than Parquet: it needs no compiled libraries, which some PCs block.
SNAPSHOT = DATA / "statcast_2026.csv.gz"
SEASON = ("2026-03-25", "2026-09-27")

CATEGORY = {**dict.fromkeys(["FF", "SI", "FC", "FA"], "fastball"),
            **dict.fromkeys(["SL", "ST", "SV"], "slider"),
            **dict.fromkeys(["CU", "KC", "CS"], "breaking"),
            **dict.fromkeys(["CH", "FS", "FO", "SC"], "offspeed")}
CLASSES = ("fastball", "offspeed", "breaking", "slider")  # column order of P4/evaluate.py
PRIOR = 10  # pseudo-pitches that pull small samples toward the broader mix; fixed, not tuned


def pull_season(start=SEASON[0], end=SEASON[1]):
    """Download regular-season Statcast pitches (about 20+ minutes for a full season)."""
    from pybaseball import cache, statcast
    cache.enable()
    raw = statcast(start_dt=start, end_dt=end)
    return raw[raw["game_type"] == "R"].copy()


def build(raw):
    """Label pitches, add P3's features, and assign whole games to splits by date."""
    df = raw.copy()
    df["pitch_cat"] = df["pitch_type"].map(CATEGORY)
    df = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)

    # Pitches outside the 4 categories still count as history ("other"), but are not targets.
    history = df["pitch_cat"].fillna("other").groupby([df["game_pk"], df["at_bat_number"]], sort=False)
    for k in range(1, 6):
        df[f"prev{k}"] = history.shift(k).fillna("none")
    df["same_hand"] = (df["stand"] == df["p_throws"]).astype(int)
    df["count"] = df["balls"].astype(str) + "-" + df["strikes"].astype(str)
    df = df[df["pitch_cat"].notna()].reset_index(drop=True)

    # Whole games in date order, so no game is split and the model never trains on the future.
    games = df.drop_duplicates("game_pk").sort_values(["game_date", "game_pk"])["game_pk"].to_numpy()
    n = len(games)
    split = pd.Series("test", index=games)
    split.iloc[: int(0.70 * n)] = "train"
    split.iloc[int(0.70 * n): int(0.85 * n)] = "dev"
    df["split"] = df["game_pk"].map(split)
    return df


def load_snapshot(path=SNAPSHOT):
    """The shared dataset; pulls and saves the season first if the file is missing."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        build(pull_season()).to_csv(path, index=False)
    return pd.read_csv(path, parse_dates=["game_date"], low_memory=False)


def baseline_probs(train, target):
    """Category probabilities for each target pitch, using training-set frequencies only.

    overall_mix:       league-wide mix (its top pick is always fastball)
    pitcher_mix:       the pitcher's own mix -> top pick is their most common pitch (proposal baseline)
    pitcher_count_mix: the pitcher's mix in the current count
    Each level is smoothed toward the one above, so unseen pitchers get the league mix.
    """
    overall = train["pitch_cat"].value_counts(normalize=True).reindex(CLASSES, fill_value=0)
    league = pd.DataFrame(np.tile(overall.to_numpy(), (len(target), 1)), index=target.index, columns=CLASSES)

    def shrink(keys, prior):
        counts = (train.groupby(keys)["pitch_cat"].value_counts().unstack(fill_value=0)
                  .reindex(columns=CLASSES, fill_value=0))
        rows = pd.MultiIndex.from_frame(target[keys]) if len(keys) > 1 else pd.Index(target[keys[0]])
        c = counts.reindex(rows).fillna(0).to_numpy()
        probs = (c + PRIOR * prior.to_numpy()) / (c.sum(axis=1, keepdims=True) + PRIOR)
        return pd.DataFrame(probs, index=target.index, columns=CLASSES)

    pitcher = shrink(["pitcher"], league)
    return {"overall_mix": league, "pitcher_mix": pitcher,
            "pitcher_count_mix": shrink(["pitcher", "count"], pitcher)}


def write_predictions(target, probs, path):
    """Write probabilities in the CSV format P4/evaluate.py reads, plus pitch ids for joining."""
    out = pd.DataFrame({"true_label": target["pitch_cat"].to_numpy()}, index=target.index)
    for c in CLASSES:
        out[f"p_{c}"] = probs[c].round(8)
    for col in ["game_pk", "at_bat_number", "pitch_number", "pitcher"]:
        out[col] = target[col]
    out.to_csv(path, index=False)


def describe(df):
    for name in ["train", "dev", "test"]:
        part = df[df["split"] == name]
        print(f"{name:5s}: {len(part):>8,} pitches, {part['game_pk'].nunique():>5,} games, "
              f"{part['game_date'].min():%Y-%m-%d} to {part['game_date'].max():%Y-%m-%d}")
    train = df[df["split"] == "train"]
    print("Class share (train):", train["pitch_cat"].value_counts(normalize=True).round(3).to_dict())
    for name in ["dev", "test"]:
        part = df[df["split"] == name]
        print(f"{name} pitches from pitchers unseen in train: {(~part['pitcher'].isin(train['pitcher'])).mean():.1%}")


def main():
    parser = argparse.ArgumentParser(description="P1 data pipeline and baselines")
    parser.add_argument("--split", choices=["dev", "test"], default="dev",
                        help="dev for development; test only once, for the final results")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "P4"))
    from evaluate import CLASSES as EVAL_CLASSES, evaluate, load_predictions, print_results
    assert tuple(EVAL_CLASSES) == CLASSES, "P1 and P4 disagree on the category order"

    df = load_snapshot()
    describe(df)
    train, target = df[df["split"] == "train"], df[df["split"] == args.split]
    for name, probs in baseline_probs(train, target).items():
        path = DATA / f"{args.split}_{name}.csv"
        write_predictions(target, probs, path)
        print(f"\n== {name} on {args.split}  ({path.relative_to(ROOT).as_posix()})")
        print_results(evaluate(load_predictions(path)))


if __name__ == "__main__":
    main()
