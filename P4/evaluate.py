"""P4: Evaluation code next-pitch classifier.

Prediction CSV format:
    true_label,p_fastball,p_offspeed,p_breaking,p_slider

All probabilities should be between 0 and 1 and each row should sum to 1.
Run: python evaluate.py predictions.csv
"""

import argparse
import csv
import math

CLASSES = ("fastball", "offspeed", "breaking", "slider")


def evaluate(rows):
    """Return accuracy, multiclass log loss, and a confusion matrix."""
    matrix = [[0] * len(CLASSES) for _ in CLASSES]
    correct = 0
    total_loss = 0.0

    if not rows:
        raise ValueError("Prediction file is empty.")

    for row_number, row in enumerate(rows, start=2):
        true_label = row["true_label"].strip().lower()
        if true_label not in CLASSES:
            raise ValueError(f"Row {row_number}: unknown true_label {true_label!r}")
        probs = [float(row[f"p_{label}"]) for label in CLASSES]
        if any(not math.isfinite(p) or p < 0 or p > 1 for p in probs):
            raise ValueError(f"Row {row_number}: probabilities must be between 0 and 1")
        if not math.isclose(sum(probs), 1.0, abs_tol=1e-5):
            raise ValueError(f"Row {row_number}: probabilities must sum to 1")

        true_index = CLASSES.index(true_label)
        predicted_index = max(range(len(probs)), key=lambda i: probs[i])
        matrix[true_index][predicted_index] += 1
        correct += (true_index == predicted_index)
        total_loss -= math.log(max(probs[true_index], 1e-15))

    return {"count": len(rows), "accuracy": correct / len(rows),
            "log_loss": total_loss / len(rows), "confusion_matrix": matrix}


def load_predictions(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def print_results(results):
    print(f"Pitches: {results['count']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Log loss: {results['log_loss']:.4f}")
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(" " * 13 + " ".join(f"{name[:5]:>6}" for name in CLASSES))
    for label, counts in zip(CLASSES, results["confusion_matrix"]):
        print(f"{label:<13}" + " ".join(f"{value:>6}" for value in counts))


def main():
    parser = argparse.ArgumentParser(description="Evaluate next-pitch probabilities")
    parser.add_argument("predictions_csv", help="CSV with actual labels and four probabilities")
    args = parser.parse_args()
    print_results(evaluate(load_predictions(args.predictions_csv)))


if __name__ == "__main__":
    main()
