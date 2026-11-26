# P1: Data pipeline, split, and baselines

`pipeline.py` builds the one dataset every stage uses, assigns the train/dev/test split,
and scores the baselines with P4's `evaluate.py`. The loading, labeling, feature and
split logic started as Vijayanidhi's cells B–D in `P3/baseball_predict.ipynb`; it is
moved here unchanged in behavior so all stages share one copy.

```
python P1/test_pipeline.py              # self-check on made-up data (seconds)
python P1/pipeline.py                   # build/load snapshot, baselines on dev
python P1/pipeline.py --split test      # final results only: run once, at the end
```

## Data

- **Source:** Statcast via pybaseball, 2026 regular season (Mar 25 – Sep 27), `game_type == "R"`:
  716,792 pitches.
- **Labels:** `pitch_cat`, 4 categories from Statcast `pitch_type`:
  fastball (FF, SI, FC, FA), slider (SL, ST, SV), breaking (CU, KC, CS),
  offspeed (CH, FS, FO, SC). Other types (eephus, knuckleball, pitchouts, missing)
  are not targets but appear as `"other"` in the pitch history. 712,632 labeled pitches.
- **Features:** `count` ("1-2"), `same_hand` (batter and pitcher same side),
  `prev1`–`prev5` (categories of the previous pitches in the same at-bat, `"none"` before
  the first). All raw Statcast columns are kept for the pitcher/hitter MLPs.
- **Snapshot:** `data/statcast_2026.csv.gz` (134 MB, gitignored). Download it from the team
  Drive into `data/`, or let `pipeline.py` rebuild it (pybaseball caches the pull).

## Split

Whole games in date order: earliest 70% train, next 15% dev, latest 15% test. No game is split,
and no model trains on games later than the ones it is scored on.

| Split | Pitches | Games | Dates |
|---|---:|---:|---|
| train | 498,004 | 1,700 | Mar 25 – Aug 4 |
| dev | 107,233 | 364 | Aug 4 – Aug 31 |
| test | 107,395 | 365 | Aug 31 – Sep 27 |

Train class share: fastball 55.3%, slider 22.2%, offspeed 14.5%, breaking 8.1%.
Pitchers unseen in train: 4.9% of dev pitches, 9.1% of test pitches (September call-ups),
so any model keyed on pitcher identity needs an unknown-pitcher fallback.

## Baselines (dev)

All three use training-set frequencies only. Each level is smoothed toward the one above it
with 10 pseudo-pitches (fixed, not tuned), so unseen pitchers fall back to the league mix.

| Baseline | Top pick | Dev accuracy | Dev log loss |
|---|---|---:|---:|
| League mix | always fastball | 55.97% | 1.137 |
| **Pitcher's mix (proposal baseline)** | pitcher's most common category | **56.59%** | **1.002** |
| Pitcher's mix in the current count | pitcher's most common category for the count | 57.07% | 0.978 |

**What this means for the model:**
- **Accuracy hardly separates anything.** Fastballs are 56% of pitches, so every baseline mostly predicts fastball.
  The pitcher baseline gets breaking balls right only 59 times out of 8,355.
- **Log loss shows real differences.** The P3 general MLP (no player IDs) reached a
  dev log loss of 1.043. That is worse than the pitcher baseline's 1.002, even though their accuracy is the same.
  The model needs pitcher identity to beat this baseline.
- **Count matters.** Adding the count to the pitcher's mix lowers log loss further (0.978).
  This is the strongest simple reference so far.

Prediction files (`data/dev_*.csv`, gitignored) use the columns
`true_label, p_fastball, p_offspeed, p_breaking, p_slider, game_pk, at_bat_number,
pitch_number, pitcher` and can be scored with `python P4/evaluate.py data/dev_pitcher_mix.csv`.
