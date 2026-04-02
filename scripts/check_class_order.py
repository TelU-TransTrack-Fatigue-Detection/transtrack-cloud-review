import sys, json
from pathlib import Path
from itertools import permutations

metrics_file = sys.argv[1] if len(sys.argv) > 1 else None
if not metrics_file:
    files = sorted(Path("output").glob("*_metrics.json"))
    metrics_file = str(files[-1])

data   = json.loads(Path(metrics_file).read_text())
matrix = data["confusion_matrix"]

CURRENT_NAMES = ["awake", "drowsy", "asleep"]
DATASET_LABELS = list(matrix.keys())
total = data["total"]

print(f"File  : {metrics_file}")
print(f"Total : {total}")
print(f"\nRaw confusion matrix (rows=true dataset label, cols=model pred index):")
print(f"{'':>12}", end="")
for i in range(3): print(f"  pred[{i}]", end="")
print()
rows = []
for dl in DATASET_LABELS:
    row = list(matrix[dl].values())
    rows.append(row)
    print(f"{dl:>12}  {'  '.join(f'{v:>6}' for v in row)}")

print(f"\nTrying all 6 permutations of CLASS_NAMES ['awake','drowsy','asleep']:")
print(f"{'Permutation':<35} {'Correct':>8} {'Accuracy':>9}")
print("-" * 56)

results = []
for perm in permutations(CURRENT_NAMES):
    name_to_idx = {name: i for i, name in enumerate(perm)}
    correct = 0
    for di, dl in enumerate(DATASET_LABELS):
        model_label = {"awake": "awake", "drowsy": "drowsy", "asleep": "asleep"}[dl]
        pred_idx = name_to_idx[model_label]
        correct += rows[di][pred_idx]
    acc = correct / total
    results.append((perm, correct, acc))

results.sort(key=lambda x: -x[2])
for perm, correct, acc in results:
    marker = " ← best" if perm == results[0][0] else ""
    marker += " ← current" if perm == tuple(CURRENT_NAMES) else ""
    print(f"CLASS_NAMES={list(perm)!s:<35} {correct:>8} {acc:>9.4f}{marker}")
