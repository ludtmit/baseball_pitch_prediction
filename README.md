# Predicting the Next Baseball Pitch (ECE 539, Team G19)

Given the pitcher, the batter, the ball-strike count and the previous pitches of the at-bat,
predict the next pitch's category with a probability for each:
fastball, offspeed, breaking or slider. Data: 2026 MLB regular-season Statcast pitches via
[pybaseball](https://github.com/jldbc/pybaseball).

## Layout

| Folder | Owner | Contents |
|---|---|---|
| `P1/` | P1 | Data pipeline, shared train/dev/test split, baselines (`baselines.ipynb`) |
| `P2/` | P2 | Pitcher MLP |
| `P3/` | P3 | Features, first general MLP (`baseball_predict.ipynb`) |
| `P4/` | P4 | Evaluation: accuracy, log loss, confusion matrix |
| `final/` | all | Integrated end-to-end model |

## Setup

Python 3.10+ (or Google Colab).

```
pip install -r requirements.txt
```

## Data

Every stage uses one shared snapshot, `data/statcast_2026.csv.gz`
(712,632 pitches; `split` column = train / dev / test, the earliest 70% / next 15% / latest 15%
of games by date). It is too large for GitHub:

- **Fast:** download it from the team Google Drive folder into `data/`.
- **From scratch:** `python P1/pipeline.py` pulls the season from Statcast (about 20+ minutes),
  saves the snapshot, and scores the baselines.

Load it from any stage (run from the repository root):

```python
from P1.pipeline import load_snapshot
df = load_snapshot()
train, dev = df[df["split"] == "train"], df[df["split"] == "dev"]
```

## Run

```
python P1/test_pipeline.py              # self-check of the pipeline on made-up data
python P1/pipeline.py                   # baselines on dev -> data/dev_*.csv
python P4/evaluate.py data/dev_pitcher_mix.csv
```

The test split is for the final results only: `python P1/pipeline.py --split test`, run once.
