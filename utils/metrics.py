"""
metrics.py
----------
Extended classification metrics required by the framework:
  Accuracy, Precision, Recall (Sensitivity), Specificity, F1-score, ROC-AUC.
Also produces Confusion Matrix and ROC Curve plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


CLASS_NAMES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]


def compute_specificity(y_true, y_pred, num_classes=4):
    """
    Macro-average specificity across all classes.
    Specificity = TN / (TN + FP)  computed per class using OvR.
    """
    specificities = []
    for c in range(num_classes):
        binary_true = (np.array(y_true) == c).astype(int)
        binary_pred = (np.array(y_pred) == c).astype(int)
        tn = np.sum((binary_true == 0) & (binary_pred == 0))
        fp = np.sum((binary_true == 0) & (binary_pred == 1))
        spec = tn / (tn + fp + 1e-8)
        specificities.append(spec)
    return float(np.mean(specificities))


def compute_all_metrics(y_true, y_pred, y_prob, num_classes=4):
    """
    Returns a dict with all required classification metrics.
    y_prob: (N, num_classes) softmax probabilities
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    spec = compute_specificity(y_true, y_pred, num_classes)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # ROC-AUC (one-vs-rest, macro)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    roc_auc = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")

    return {
        "accuracy":    acc,
        "precision":   prec,
        "recall":      rec,
        "specificity": spec,
        "f1_score":    f1,
        "roc_auc":     roc_auc,
    }


# ── Visualisation helpers ──────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, save_path, class_names=None):
    if class_names is None:
        class_names = CLASS_NAMES
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix → {save_path}")


def plot_roc_curve(y_true, y_prob, save_path, class_names=None, num_classes=4):
    if class_names is None:
        class_names = CLASS_NAMES
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc_val:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (per class)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curve → {save_path}")


def plot_training_curves(history, model_name, save_dir="."):
    """
    Plots Accuracy vs Epoch and Loss vs Epoch for a single model.
    history keys: train_loss, val_loss, train_acc, val_acc
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="royalblue")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="tomato")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} – Loss vs Epoch")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train Acc", color="royalblue")
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc",   color="tomato")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{model_name} – Accuracy vs Epoch")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = f"{save_dir}/{model_name.lower().replace(' ', '_')}_training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves → {save_path}")


def print_metrics_table(results_dict):
    """
    Pretty-prints the performance comparison table for all models.
    results_dict: { model_name: metrics_dict }
    """
    header = f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'Specificity':>12} {'F1-Score':>10} {'ROC-AUC':>10}"
    print("\n" + "=" * len(header))
    print("PERFORMANCE COMPARISON TABLE")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for model_name, m in results_dict.items():
        print(
            f"{model_name:<22} "
            f"{m['accuracy']*100:>9.2f}% "
            f"{m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} "
            f"{m['specificity']:>12.4f} "
            f"{m['f1_score']:>10.4f} "
            f"{m['roc_auc']:>10.4f}"
        )
    print("=" * len(header))


def print_efficiency_table(efficiency_dict):
    """
    efficiency_dict: { model_name: {training_time_min, inference_time_ms, num_params_M, memory_mb} }
    """
    header = f"{'Model':<22} {'Train Time (min)':>17} {'Infer Time (ms)':>16} {'Params (M)':>11} {'Memory (MB)':>12}"
    print("\n" + "=" * len(header))
    print("EFFICIENCY COMPARISON TABLE")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for model_name, e in efficiency_dict.items():
        print(
            f"{model_name:<22} "
            f"{e['training_time_min']:>17.2f} "
            f"{e['inference_time_ms']:>16.3f} "
            f"{e['num_params_M']:>11.2f} "
            f"{e['memory_mb']:>12.1f}"
        )
    print("=" * len(header))
