"""
main_framework_firefly.py
=========================
Full pipeline:

  1. Load segmented (RA-UNetX) lung images
  2. Train all three classifiers:
       - VGG16
       - ResNet50
       - ConvNeXt-Tiny  (proposed)
  3. Evaluate each model:
       - Accuracy, Precision, Recall (Sensitivity), Specificity, F1-score, ROC-AUC
       - Confusion Matrix + ROC Curve
       - Training Curves (Acc vs Epoch, Loss vs Epoch)
  4. Efficiency Metrics (all 3 models):
       - Training time, Inference time, # Parameters, Memory usage
  5. Grad-CAM visualisation (all 3 models)
  6. 5-Fold Cross-Validation (ConvNeXt-Tiny ONLY)
  7. [NEW] Firefly Feature Selection on extracted CNN features:
       - For each trained model: extract deep features → Firefly selection →
         train MLP + SVM classifier → full metrics + CV
       - Side-by-side comparison table: baseline CNN vs Firefly-MLP vs Firefly-SVM

Usage
-----
    python main_framework_firefly.py

Edit the CONFIG section below to match your paths.
"""

import os, time, psutil, json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
)
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
import joblib

# ── Project imports ────────────────────────────────────────────────────────
from models.classifier import ConvNeXtClassifier, VGG16Classifier, ResNet50Classifier
from utils.data_create  import get_train_val_dataloaders, PreMaskedClassificationDataset
from train.train_classifier import train_model
from utils.metrics import (
    compute_all_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_curves,
    print_metrics_table,
    print_efficiency_table,
    CLASS_NAMES,
)
from utils.gradcam import GradCAM, get_target_layer, visualise_gradcam
from utils.cross_validation import run_5fold_cv
from preprocess.feature import extract_features
from train.firefly import FireflyFeatureSelectionMLP, FireflyFeatureSelectionSVM


# ==========================================================================
# CONFIG  ── edit these paths / hyper-parameters
# ==========================================================================
DATASET_PATH  = r"C:\Users\akhsh\Downloads\archive (5)\COVID-19_Radiography_Dataset"
OUTPUT_DIR    = "framework_results"

BATCH_SIZE    = 8
IMAGE_SIZE    = (256, 256)
NUM_CLASSES   = 4
NUM_EPOCHS    = 10
LR            = 1e-4
NUM_WORKERS   = 0          # set >0 on Linux with multiple CPU cores
CV_EPOCHS     = 10         # epochs per fold (lighter than full training)
GRADCAM_SAMPLES_PER_CLASS = 2

# ── Firefly hyper-parameters ───────────────────────────────────────────────
FIREFLY_N_FIREFLIES   = 15
FIREFLY_MAX_ITER_MLP  = 5    # faster because MLP is GPU-accelerated
FIREFLY_MAX_ITER_SVM  = 10
FIREFLY_CLASSIFIER    = "both"   # "MLP" | "SVM" | "both"
FIREFLY_MLP_EPOCHS    = 15       # epochs for the final MLP after feature selection
FIREFLY_MLP_MODEL     = "deep"   # "deep" | "normal"
CV_FOLDS              = 5        # folds for Firefly post-selection CV

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================================================


# ══════════════════════════════════════════════════════════════════════════
#  Helper: tiny MLP definitions (mirrors models/classifier.py)
# ══════════════════════════════════════════════════════════════════════════
class MLPClassifier(nn.Module):
    """Lightweight MLP for post-firefly classification."""
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class BetterMLP(nn.Module):
    """Deeper MLP for post-firefly classification."""
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════
#  Helper: framework utilities (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════
def get_num_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def measure_memory_mb(model, device, input_channels=6, image_size=(256, 256)):
    dummy = torch.zeros(1, input_channels, *image_size).to(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy)
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        proc   = psutil.Process(os.getpid())
        before = proc.memory_info().rss / (1024 ** 2)
        with torch.no_grad():
            _ = model(dummy)
        after = proc.memory_info().rss / (1024 ** 2)
        return max(after - before, 0)


def measure_inference_time_ms(model, val_loader, device, n_batches=50):
    model.eval()
    times = []
    with torch.no_grad():
        for i, (images, _) in enumerate(val_loader):
            if i >= n_batches:
                break
            images = images.to(device)
            start  = time.perf_counter()
            _      = model(images)
            end    = time.perf_counter()
            times.append((end - start) * 1000 / images.size(0))
    return float(np.mean(times))


def evaluate_model(model, val_loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def run_gradcam_for_model(model, model_name, val_loader, device, out_dir):
    target_layer = get_target_layer(model, model_name)
    cam = GradCAM(model, target_layer)

    saved = {c: 0 for c in range(NUM_CLASSES)}
    total_needed = GRADCAM_SAMPLES_PER_CLASS * NUM_CLASSES

    for images, labels in val_loader:
        for i in range(images.size(0)):
            lbl = labels[i].item()
            if saved[lbl] >= GRADCAM_SAMPLES_PER_CLASS:
                continue

            inp        = images[i].unsqueeze(0).to(device)
            pred_idx   = cam.predict_class(inp)
            heatmap    = cam(inp, class_idx=pred_idx)
            pred_label = CLASS_NAMES[pred_idx]
            true_label = CLASS_NAMES[lbl]

            img_t = images[i][:3].clone()
            mean  = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std   = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_t = (img_t * std + mean).clamp(0, 1)
            img_pil = transforms.ToPILImage()(img_t)

            fname = (f"{model_name.lower().replace(' ', '_')}_"
                     f"true{true_label}_pred{pred_label}_{saved[lbl]}.png")
            visualise_gradcam(img_pil, heatmap, pred_label,
                              save_path=str(out_dir / fname))
            saved[lbl] += 1

        if sum(saved.values()) >= total_needed:
            break

    print(f"  Grad-CAM saved {sum(saved.values())} images for {model_name}")


# ══════════════════════════════════════════════════════════════════════════
#  Firefly pipeline helpers
# ══════════════════════════════════════════════════════════════════════════

def extract_features_from_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract penultimate-layer features from a trained CNN.

    Works by registering a forward hook on the last non-classifier linear
    layer (or the AdaptiveAvgPool output if no hook target found).
    Falls back to calling model.extract_features() if that method exists.
    """
    model.eval()
    model.to(device)

    features_list, labels_list = [], []

    # ── Try the project's own extract_features helper first ──────────────
    # (from preprocess.feature – it already knows how to handle your models)
    try:
        feats, lbls = extract_features(
            model_path=None,          # we pass the model directly below
            dataloader=dataloader,
            device=device,
            model=model,              # assumes extract_features accepts a model kwarg
        )
        return feats, lbls
    except TypeError:
        pass  # extract_features doesn't accept a model kwarg – use hook below

    # ── Hook-based extraction ─────────────────────────────────────────────
    hook_output = []

    def _hook(module, input, output):
        hook_output.append(output.detach().cpu())

    # Identify the layer to hook: avgpool or the layer before the head
    hook_layer = None
    for name, module in model.named_modules():
        if isinstance(module, (nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool1d)):
            hook_layer = module
        # For ConvNeXt the global pool is followed by a flatten + head
        if isinstance(module, nn.Flatten):
            hook_layer = module

    if hook_layer is None:
        # Last resort: hook the second-to-last module
        children = list(model.children())
        if len(children) >= 2:
            hook_layer = children[-2]

    if hook_layer is None:
        raise RuntimeError("Could not find a layer to hook for feature extraction.")

    handle = hook_layer.register_forward_hook(_hook)

    with torch.no_grad():
        for images, labels in dataloader:
            hook_output.clear()
            images = images.to(device)
            _ = model(images)           # trigger hook
            feat = hook_output[0]       # (B, C[, H, W])
            if feat.dim() > 2:          # pool spatial dims if still present
                feat = feat.mean(dim=[2, 3])
            features_list.append(feat.numpy())
            labels_list.append(labels.numpy())

    handle.remove()

    return np.vstack(features_list), np.concatenate(labels_list)


def build_mlp_and_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    device: str,
    epochs: int = 15,
    model_type: str = "deep",
) -> nn.Module:
    """Train a small MLP on selected features and return the trained model."""
    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train,  dtype=torch.long).to(device)

    in_dim = X_train.shape[1]
    if model_type == "deep":
        mlp = BetterMLP(in_dim, num_classes).to(device)
    else:
        mlp = MLPClassifier(in_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=1e-3)

    mlp.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(mlp(X_t), y_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"      [MLP epoch {epoch+1}/{epochs}] loss={loss.item():.4f}")

    return mlp


def mlp_predict_with_probs(
    mlp: nn.Module,
    X_test: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (preds, probs) arrays from an MLP."""
    mlp.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    preds, probs, times = [], [], []

    with torch.no_grad():
        for i in range(len(X_t)):
            sample = X_t[i].unsqueeze(0)
            t0  = time.perf_counter()
            out = mlp(sample)
            times.append((time.perf_counter() - t0) * 1000)
            prob = F.softmax(out, dim=1).cpu().numpy()[0]
            probs.append(prob)
            preds.append(int(out.argmax(dim=1).item()))

    avg_t = float(np.mean(times))
    return np.array(preds), np.array(probs), avg_t


def svm_predict_with_probs(
    clf: SVC,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (preds, probs, avg_ms) from an SVM (probability=True needed for probs)."""
    preds, times = [], []
    for i in range(len(X_test)):
        sample = X_test[i].reshape(1, -1)
        t0     = time.perf_counter()
        pred   = clf.predict(sample)
        times.append((time.perf_counter() - t0) * 1000)
        preds.append(pred[0])

    avg_t = float(np.mean(times))
    # Probability estimates (require probability=True in SVC)
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_test)
    else:
        # one-hot fallback
        n_cls = len(np.unique(preds))
        probs = np.eye(n_cls)[np.array(preds)]

    return np.array(preds), probs, avg_t


def compute_firefly_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
) -> dict:
    """
    Compute the same metric set as utils.metrics.compute_all_metrics so the
    results slot directly into the framework comparison tables.
    """
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_auc_score

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred,    average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred,        average="weighted", zero_division=0)

    # Specificity (per-class, then macro-average)
    cm   = confusion_matrix(y_true, y_pred)
    spec_per_class = []
    for c in range(num_classes):
        tn = cm.sum() - (cm[c, :].sum() + cm[:, c].sum() - cm[c, c])
        fp = cm[:, c].sum() - cm[c, c]
        spec_per_class.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    specificity = float(np.mean(spec_per_class))

    # ROC-AUC (one-vs-rest, weighted)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    try:
        roc_auc = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="weighted")
    except ValueError:
        roc_auc = float("nan")

    return {
        "accuracy":    acc,
        "precision":   prec,
        "recall":      rec,
        "specificity": specificity,
        "f1_score":    f1,
        "roc_auc":     roc_auc,
    }


# ── Firefly CV (post feature-selection) ───────────────────────────────────

def firefly_cv_mlp(
    X_selected: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    device: str,
    n_folds: int = 5,
    epochs: int = 15,
    model_type: str = "deep",
) -> dict:
    """5-fold CV on Firefly-selected features using MLP."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_selected, y)):
        print(f"    [MLP CV] Fold {fold+1}/{n_folds}")
        X_tr, X_va = X_selected[train_idx], X_selected[val_idx]
        y_tr, y_va = y[train_idx],          y[val_idx]

        mlp = build_mlp_and_train(X_tr, y_tr, num_classes, device, epochs, model_type)
        preds, probs, _ = mlp_predict_with_probs(mlp, X_va, device)
        fold_metrics.append(compute_firefly_metrics(y_va, preds, probs, num_classes))

    # Aggregate
    summary = {}
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics]
        summary[key]          = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals))
    return summary


def firefly_cv_svm(
    X_selected: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    n_folds: int = 5,
) -> dict:
    """5-fold CV on Firefly-selected features using SVM."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_selected, y)):
        print(f"    [SVM CV] Fold {fold+1}/{n_folds}")
        X_tr, X_va = X_selected[train_idx], X_selected[val_idx]
        y_tr, y_va = y[train_idx],          y[val_idx]

        clf = SVC(kernel="linear", probability=True)
        clf.fit(X_tr, y_tr)
        preds, probs, _ = svm_predict_with_probs(clf, X_va)
        fold_metrics.append(compute_firefly_metrics(y_va, preds, probs, num_classes))

    summary = {}
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics]
        summary[key]          = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals))
    return summary


# ── Core Firefly pipeline for ONE backbone model ──────────────────────────

def run_firefly_pipeline(
    backbone_name: str,
    backbone_model: nn.Module,
    dataloader: DataLoader,
    device: str,
    save_dir: Path,
    num_classes: int,
) -> dict:
    """
    Full Firefly pipeline for one backbone:
      extract features → firefly selection → MLP/SVM train →
      full metrics + CV

    Returns a dict keyed by variant name (e.g. "ResNet50_Firefly_MLP").
    """
    print(f"\n  {'─'*55}")
    print(f"  Firefly pipeline: {backbone_name}")
    print(f"  {'─'*55}")

    # ── 1. Extract deep features ──────────────────────────────────────────
    print("  [1/5] Extracting deep features …")
    X, y = extract_features_from_model(backbone_model, dataloader, device)
    print(f"        Feature matrix: {X.shape}  |  labels: {y.shape}")

    results = {}

    # ── 2. Firefly feature selection ──────────────────────────────────────
    # We run Firefly once per classifier type if FIREFLY_CLASSIFIER == "both"
    for clf_type in (["MLP", "SVM"] if FIREFLY_CLASSIFIER == "both"
                     else [FIREFLY_CLASSIFIER]):

        variant = f"{backbone_name}_Firefly_{clf_type}"
        print(f"\n  [2/5] Firefly ({clf_type}) on {backbone_name} …")

        if clf_type == "MLP":
            fa = FireflyFeatureSelectionMLP(
                n_fireflies=FIREFLY_N_FIREFLIES,
                n_features=X.shape[1],
                max_iter=FIREFLY_MAX_ITER_MLP,
                device=device,
            )
        else:
            fa = FireflyFeatureSelectionSVM(
                n_fireflies=FIREFLY_N_FIREFLIES,
                n_features=X.shape[1],
                max_iter=FIREFLY_MAX_ITER_SVM,
            )

        best_mask, best_val_acc = fa.run(X, y)
        n_selected = int(np.sum(best_mask))
        print(f"        Selected {n_selected}/{X.shape[1]} features | "
              f"val-acc during selection: {best_val_acc:.4f}")

        # Persist mask
        mask_stem = f"{backbone_name.lower().replace(' ', '_')}_{clf_type.lower()}"
        np.save(save_dir / f"{mask_stem}_feature_mask.npy", best_mask)
        with open(save_dir / f"{mask_stem}_feature_mask.json", "w") as fh:
            json.dump(best_mask.tolist(), fh)

        X_sel = X[:, best_mask == 1]

        # ── 3. Train/test split ───────────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X_sel, y, test_size=0.2, stratify=y, random_state=42
        )

        # ── 4. Train final classifier ─────────────────────────────────────
        print(f"  [3/5] Training final {clf_type} on {n_selected} features …")

        if clf_type == "MLP":
            t0  = time.perf_counter()
            mlp = build_mlp_and_train(
                X_train, y_train, num_classes, device,
                epochs=FIREFLY_MLP_EPOCHS, model_type=FIREFLY_MLP_MODEL,
            )
            train_min = (time.perf_counter() - t0) / 60
            torch.save(mlp.state_dict(),
                       save_dir / f"{mask_stem}_mlp_model.pth")

            y_pred, y_prob, avg_infer_ms = mlp_predict_with_probs(mlp, X_test, device)

        else:  # SVM
            t0  = time.perf_counter()
            clf = SVC(kernel="linear", probability=True)
            clf.fit(X_train, y_train)
            train_min = (time.perf_counter() - t0) / 60
            joblib.dump(clf, save_dir / f"{mask_stem}_svm_model.pkl")

            y_pred, y_prob, avg_infer_ms = svm_predict_with_probs(clf, X_test)

        # ── 5. Compute all metrics ────────────────────────────────────────
        print(f"  [4/5] Computing metrics for {variant} …")
        metrics = compute_firefly_metrics(y_true=y_test, y_pred=y_pred,
                                          y_prob=y_prob, num_classes=num_classes)
        report  = classification_report(y_test, y_pred, zero_division=0,
                                         target_names=CLASS_NAMES)

        print(f"\n  ── {variant} Evaluation ──")
        print(f"  Accuracy : {metrics['accuracy']*100:.2f}%")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  Specif.  : {metrics['specificity']:.4f}")
        print(f"  F1-score : {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
        print(f"  Avg infer: {avg_infer_ms:.3f} ms/sample")
        print(f"\n{report}")

        # Confusion matrix plot
        cm_path = save_dir / f"{mask_stem}_confusion.png"
        _plot_cm(y_test, y_pred, title=variant, save_path=str(cm_path))

        # ROC plot
        roc_path = save_dir / f"{mask_stem}_roc.png"
        _plot_roc(y_test, y_prob, num_classes=num_classes,
                  title=variant, save_path=str(roc_path))

        # ── 5-fold CV ─────────────────────────────────────────────────────
        print(f"  [5/5] 5-fold CV for {variant} …")
        if clf_type == "MLP":
            cv_summary = firefly_cv_mlp(
                X_sel, y, num_classes, device,
                n_folds=CV_FOLDS,
                epochs=FIREFLY_MLP_EPOCHS,
                model_type=FIREFLY_MLP_MODEL,
            )
        else:
            cv_summary = firefly_cv_svm(X_sel, y, num_classes, n_folds=CV_FOLDS)

        print(f"  CV Summary for {variant}:")
        for k, v in cv_summary.items():
            if not k.endswith("_std"):
                std = cv_summary.get(f"{k}_std", 0.0)
                print(f"    {k:<14}: {v:.4f} ± {std:.4f}")

        results[variant] = {
            "metrics":        metrics,
            "cv_summary":     cv_summary,
            "n_selected":     n_selected,
            "n_total":        X.shape[1],
            "train_time_min": train_min,
            "infer_ms":       avg_infer_ms,
        }

    return results


# ── Lightweight plot helpers (avoid modifying utils/metrics.py) ───────────

def _plot_cm(y_true, y_pred, title: str, save_path: str):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_roc(y_true, y_prob, num_classes: int, title: str, save_path: str):
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

    y_bin   = label_binarize(y_true, classes=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in range(num_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_prob[:, c])
        ax.plot(fpr, tpr, label=f"{CLASS_NAMES[c]} (AUC={auc(fpr,tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(title); ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ── Final comparison table ────────────────────────────────────────────────

def print_firefly_comparison_table(
    baseline_metrics: dict,     # {model_name: metrics_dict}
    firefly_results:  dict,     # {variant_name: result_dict}
):
    """Side-by-side table: baseline CNN vs Firefly-MLP vs Firefly-SVM."""
    header = f"\n{'─'*100}"
    print(header)
    print(f"{'FINAL COMPARISON TABLE':^100}")
    print(header)
    fmt   = "{:<30} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}"
    names = ["Acc%", "Prec", "Recall", "Spec", "F1", "AUC",
             "Infer(ms)", "Params(M)"]
    print(fmt.format("Model / Variant", *names))
    print("─" * 100)

    # Baseline CNNs
    for name, m in baseline_metrics.items():
        acc  = m.get("accuracy", 0) * 100
        prec = m.get("precision", m.get("weighted_precision", 0))
        rec  = m.get("recall",   m.get("weighted_recall",   0))
        spec = m.get("specificity", float("nan"))
        f1   = m.get("f1_score",   m.get("weighted_f1",    0))
        auc_ = m.get("roc_auc",    float("nan"))
        # These come from all_efficiency set upstream via the closure
        infer = float("nan")
        params= float("nan")
        print(fmt.format(
            f"[CNN] {name}",
            f"{acc:.1f}", f"{prec:.3f}", f"{rec:.3f}",
            f"{spec:.3f}", f"{f1:.3f}", f"{auc_:.3f}",
            f"{infer:.2f}", f"{params:.1f}",
        ))

    print("─" * 100)

    # Firefly variants
    for variant, res in firefly_results.items():
        m    = res["metrics"]
        acc  = m["accuracy"] * 100
        prec = m["precision"]
        rec  = m["recall"]
        spec = m["specificity"]
        f1   = m["f1_score"]
        auc_ = m["roc_auc"]
        infer= res["infer_ms"]
        feat = f"{res['n_selected']}/{res['n_total']}"
        print(fmt.format(
            f"[Firefly] {variant}",
            f"{acc:.1f}", f"{prec:.3f}", f"{rec:.3f}",
            f"{spec:.3f}", f"{f1:.3f}", f"{auc_:.3f}",
            f"{infer:.3f}", f"feats:{feat}",
        ))

    print("─" * 100)

    # Firefly CV summary
    print(f"\n{'─'*100}")
    print(f"{'FIREFLY 5-FOLD CV SUMMARY':^100}")
    print("─" * 100)
    cv_fmt = "{:<30} {:>10} {:>10} {:>10} {:>10} {:>10}"
    print(cv_fmt.format("Variant", "Acc", "Prec", "Recall", "F1", "AUC"))
    print("─" * 100)
    for variant, res in firefly_results.items():
        cv = res["cv_summary"]
        print(cv_fmt.format(
            variant,
            f"{cv['accuracy']:.3f}±{cv['accuracy_std']:.3f}",
            f"{cv['precision']:.3f}±{cv['precision_std']:.3f}",
            f"{cv['recall']:.3f}±{cv['recall_std']:.3f}",
            f"{cv['f1_score']:.3f}±{cv['f1_score_std']:.3f}",
            f"{cv['roc_auc']:.3f}±{cv['roc_auc_std']:.3f}",
        ))
    print("─" * 100)


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = Path(OUTPUT_DIR)
    firefly_out = out / "firefly_results"
    firefly_out.mkdir(exist_ok=True)

    print(f"Device : {DEVICE}")
    print(f"Output : {OUTPUT_DIR}\n")

    # ── Data loaders ──────────────────────────────────────────────────────
    print("Loading data loaders …")
    train_loader, val_loader = get_train_val_dataloaders(
        root_dir=DATASET_PATH,
        batch_size=BATCH_SIZE,
        val_split=0.2,
        image_size=IMAGE_SIZE,
        num_workers=NUM_WORKERS,
    )
    print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}\n")

    # Full dataset (used for Firefly feature extraction + CV)
    full_dataset = PreMaskedClassificationDataset(
        root_dir=DATASET_PATH,
        image_size=IMAGE_SIZE,
    )
    full_loader = DataLoader(
        full_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # ── Model definitions ─────────────────────────────────────────────────
    model_registry = {
        "ResNet50":      ResNet50Classifier(num_classes=NUM_CLASSES),
        "ConvNeXt-Tiny": ConvNeXtClassifier(num_classes=NUM_CLASSES),
        "VGG16":         VGG16Classifier(num_classes=NUM_CLASSES),
    }

    all_histories   = {}
    all_metrics     = {}
    all_efficiency  = {}
    all_firefly     = {}        # accumulated Firefly results across all backbones

    # ── Train + Evaluate each backbone model ──────────────────────────────
    for model_name, model in model_registry.items():
        print("=" * 65)
        print(f"  MODEL: {model_name}")
        print("=" * 65)
        save_path = str(out / f"{model_name.lower().replace(' ','_')}_best.pth")

        # ── (A) Standard training ──────────────────────────────────────────
        history = train_model(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=NUM_EPOCHS,
            lr=LR,
            device=DEVICE,
            save_path=save_path,
        )
        all_histories[model_name] = history

        # Reload best weights
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        model.to(DEVICE).eval()

        plot_training_curves(history, model_name, save_dir=OUTPUT_DIR)

        # ── (B) Standard evaluation ────────────────────────────────────────
        y_true, y_pred, y_prob = evaluate_model(model, val_loader, DEVICE)
        metrics = compute_all_metrics(y_true, y_pred, y_prob, NUM_CLASSES)
        all_metrics[model_name] = metrics

        plot_confusion_matrix(
            y_true, y_pred,
            save_path=str(out / f"{model_name.lower().replace(' ','_')}_confusion.png"),
        )
        plot_roc_curve(
            y_true, y_prob,
            save_path=str(out / f"{model_name.lower().replace(' ','_')}_roc.png"),
        )

        # ── (C) Efficiency ─────────────────────────────────────────────────
        num_params = get_num_params(model)
        mem_mb     = measure_memory_mb(model, DEVICE)
        infer_ms   = measure_inference_time_ms(model, val_loader, DEVICE)
        train_min  = history.get("training_time_min", 0.0)

        all_efficiency[model_name] = {
            "training_time_min": train_min,
            "inference_time_ms": infer_ms,
            "num_params_M":      num_params,
            "memory_mb":         mem_mb,
        }

        print(f"\n  {model_name}: Acc={metrics['accuracy']*100:.2f}%  "
              f"F1={metrics['f1_score']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}")
        print(f"  Efficiency: {train_min:.1f} min | {infer_ms:.2f} ms/img | "
              f"{num_params:.1f}M params | {mem_mb:.0f} MB\n")

        # ── (D) Grad-CAM ───────────────────────────────────────────────────
        grad_dir = out / f"gradcam_{model_name.lower().replace(' ','_')}"
        grad_dir.mkdir(exist_ok=True)
        run_gradcam_for_model(model, model_name, val_loader, DEVICE, grad_dir)

        # ── (E) Firefly pipeline (uses full dataset for richer features) ───
        firefly_res = run_firefly_pipeline(
            backbone_name=model_name,
            backbone_model=model,
            dataloader=full_loader,
            device=DEVICE,
            save_dir=firefly_out,
            num_classes=NUM_CLASSES,
        )
        all_firefly.update(firefly_res)

    # ── Summary tables (standard) ─────────────────────────────────────────
    print_metrics_table(all_metrics)
    print_efficiency_table(all_efficiency)

    # ── 5-Fold CV  (ConvNeXt-Tiny ONLY, original framework) ───────────────
    print("\n\nRunning 5-Fold Cross-Validation for ConvNeXt-Tiny (full image) …")
    cv_summary = run_5fold_cv(
        full_dataset=full_dataset,
        num_classes=NUM_CLASSES,
        num_epochs=CV_EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        num_workers=NUM_WORKERS,
    )

    # ── Firefly comparison table ───────────────────────────────────────────
    print_firefly_comparison_table(all_metrics, all_firefly)

    # ── Persist full Firefly result dict ──────────────────────────────────
    def _serialise(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serialisable: {type(obj)}")

    with open(firefly_out / "all_firefly_results.json", "w") as fh:
        json.dump(all_firefly, fh, default=_serialise, indent=2)

    print(f"\n✅ Framework complete.  All outputs saved to: {OUTPUT_DIR}")
    print(f"   Firefly artefacts in: {firefly_out}")


if __name__ == "__main__":
    main()