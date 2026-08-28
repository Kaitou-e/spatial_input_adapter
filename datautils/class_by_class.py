import pandas as pd
import numpy as np

target_csv_dir = "/home/lxdcis/hypersl_training/outputs/indian_linear_seed0/test_confusion_matrix.csv"
res_csv_dir = "/home/lxdcis/hypersl_training/outputs/indian_linear_seed0/class_confusion_matrix.csv"

# target_csv_dir = "/home/lxdcis/hypersl_training/outputs/indian_adapter_mix70_seed0/test_confusion_matrix.csv"
# res_csv_dir = "/home/lxdcis/hypersl_training/outputs/indian_adapter_mix70_seed0/class_confusion_matrix.csv"

# Load confusion matrix
cm_df = pd.read_csv(target_csv_dir, header=None)
cm = cm_df.to_numpy()

# If your CSV does NOT have row labels / an index column, instead use:
# cm_df = pd.read_csv("test_confusion_matrix.csv", header=None)
# cm = cm_df.to_numpy()

n_classes = cm.shape[0]

# Per-class quantities
tp = np.diag(cm)
support = cm.sum(axis=1)          # actual samples per class
predicted = cm.sum(axis=0)        # predicted samples per class

fn = support - tp
fp = predicted - tp
tn = cm.sum() - tp - fn - fp

# Metrics
recall = np.divide(
    tp, support,
    out=np.zeros_like(tp, dtype=float),
    where=support != 0
)

precision = np.divide(
    tp, predicted,
    out=np.zeros_like(tp, dtype=float),
    where=predicted != 0
)

f1 = np.divide(
    2 * precision * recall,
    precision + recall,
    out=np.zeros_like(precision, dtype=float),
    where=(precision + recall) != 0
)

specificity = np.divide(
    tn, tn + fp,
    out=np.zeros_like(tn, dtype=float),
    where=(tn + fp) != 0
)

# Overall Accuracy
oa = tp.sum() / cm.sum()

# Average Accuracy = mean per-class accuracy
aa = recall.mean()

# Cohen's Kappa
total = cm.sum()
po = tp.sum() / total
pe = (support * predicted).sum() / (total ** 2)
kappa = (po - pe) / (1 - pe)

# Build table
stats = pd.DataFrame({
    "Class": [f"Class {i+1}" for i in range(n_classes)],
    "Support": support,
    "Correct": tp,
    "Accuracy/Recall": recall,
    "Precision": precision,
    "F1": f1,
    "Specificity": specificity,
})

# Convert metrics to percentages if desired
for col in ["Accuracy/Recall", "Precision", "F1", "Specificity"]:
    stats[col] *= 100

print(stats.to_string(index=False))

print(f"\nOverall Accuracy (OA): {oa * 100:.2f}%")
print(f"Average Accuracy (AA): {aa * 100:.2f}%")
print(f"Kappa: {kappa:.4f}")

stats.to_csv(res_csv_dir, index=False)