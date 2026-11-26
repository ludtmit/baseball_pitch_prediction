"""Self-check for P1/pipeline.py on a tiny made-up season. Run: python P1/test_pipeline.py"""

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "P4"))
from pipeline import CLASSES, baseline_probs, build, load_snapshot, write_predictions  # noqa: E402
from evaluate import evaluate, load_predictions  # noqa: E402

# 20 games on 20 dates, listed out of order; each game has two at-bats.
rows = []
for g in range(20):
    pitcher = 1 if g < 18 else 99  # pitcher 99 only appears in the last games
    for ab, types in [(1, ["FF", "SL", "EP", "CH"]), (2, ["CU", "FF"])]:
        for i, t in enumerate(types, start=1):
            rows.append({"game_pk": 1000 + g, "game_date": pd.Timestamp("2026-04-01") + pd.Timedelta(days=g),
                         "at_bat_number": ab, "pitch_number": i, "pitch_type": t, "pitcher": pitcher,
                         "stand": "R", "p_throws": "R" if ab == 1 else "L", "balls": 0, "strikes": i - 1})
raw = pd.DataFrame(rows).sample(frac=1, random_state=0)

df = build(raw)
first = df[df["game_pk"] == 1000]

# Labels and history: EP is not a target but shows up as "other" in the next pitch's history.
assert "EP" not in set(df["pitch_type"])
assert first["pitch_cat"].tolist() == ["fastball", "slider", "offspeed", "breaking", "fastball"]
assert first["prev1"].tolist() == ["none", "fastball", "other", "none", "breaking"]
assert first["prev2"].tolist() == ["none", "none", "slider", "none", "none"]  # resets each at-bat
assert first["same_hand"].tolist() == [1, 1, 1, 0, 0]

# Split: whole games, 14/3/3, strictly in date order.
per_game = df.groupby("game_pk")["split"].nunique()
assert (per_game == 1).all()
order = df.drop_duplicates("game_pk").sort_values("game_date")["split"].tolist()
assert order == ["train"] * 14 + ["dev"] * 3 + ["test"] * 3

# Baselines: rows sum to 1, unseen pitcher falls back to the league mix.
train, test = df[df["split"] == "train"], df[df["split"] == "test"]
probs = baseline_probs(train, test)
league = train["pitch_cat"].value_counts(normalize=True).reindex(CLASSES)
for p in probs.values():
    assert ((p.sum(axis=1) - 1).abs() < 1e-9).all()
unseen = test["pitcher"] == 99
assert (probs["pitcher_mix"][unseen].iloc[0] - league).abs().max() < 1e-9
assert probs["pitcher_mix"][~unseen].idxmax(axis=1).eq("fastball").all()

# Snapshot survives a save/reload, and output passes P4's evaluator.
with tempfile.TemporaryDirectory() as tmp:
    snap = Path(tmp) / "snap.csv.gz"
    df.to_csv(snap, index=False)
    back = load_snapshot(snap)
    assert back["prev1"].tolist() == df["prev1"].tolist()
    assert back["game_date"].tolist() == df["game_date"].tolist()
    path = Path(tmp) / "pred.csv"
    write_predictions(test, probs["pitcher_count_mix"], path)
    result = evaluate(load_predictions(path))
    assert result["count"] == len(test)

print("P1 self-check passed")
