"""
main_framework_firefly.py
=========================
Loads pre-trained CNN weights (NO re-training) then runs:

  1. Feature extraction from each frozen backbone
  2. Firefly feature selection  (MLP / SVM / both)
  3. Final MLP + SVM classifier on selected features
  4. Full metrics: Accuracy, Precision, Recall, Specificity, F1, ROC-AUC
  5. Confusion matrix + ROC curve plots
  6. 5-Fold Cross-Validation on selected features
  7. Side-by-side comparison + feature-reduction table

Usage
-----
    python main_framework_firefly.py

Edit the CONFIG section below to match your saved .pth file paths.
"""

import os, time, json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path
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
from utils.data_create  import PreMaskedClassificationDataset
from utils.metrics      import CLASS_NAMES
from train.firefly      import FireflyFeatureSelectionMLP, FireflyFeatureSelectionSVM


# ==========================================================================
# CONFIG  ── edit these to match your environment
# ==========================================================================

DATASET_PATH = r"C:\Users\akhsh\Downloads\archive (5)\COVID-19_Radiography_Dataset"
OUTPUT_DIR   = "firefly_results"

# Paths to .pth files produced by main_framework.py
# Set any value to None to skip that backbone
WEIGHT_PATHS = {
    # "ResNet50":      r"framework_results\resnet50_best.pth",
    "ConvNeXt-Tiny": r"framework_results\convnext-tiny_best.pth",
    # "VGG16":         r"framework_results\vgg16_best.pth",
}

BATCH_SIZE   = 8
IMAGE_SIZE   = (256, 256)
NUM_CLASSES  = 4
NUM_WORKERS  = 0

# ── Firefly hyper-parameters ───────────────────────────────────────────────
FIREFLY_CLASSIFIER   = "both"   # "MLP" | "SVM" | "both"
FIREFLY_N_FIREFLIES  = 15
FIREFLY_MAX_ITER_MLP = 5        # lighter because MLP eval is GPU-fast
FIREFLY_MAX_ITER_SVM = 10

# ── Post-selection MLP training ────────────────────────────────────────────
FIREFLY_MLP_EPOCHS = 15
FIREFLY_MLP_MODEL  = "deep"     # "deep" (BetterMLP) | "normal" (MLPClassifier)

# ── Cross-validation ───────────────────────────────────────────────────────
CV_FOLDS = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================================================


# ══════════════════════════════════════════════════════════════════════════
#  Tiny MLP definitions
# ══════════════════════════════════════════════════════════════════════════
class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),   nn.ReLU(),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.net(x)


class BetterMLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256),   nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),   nn.ReLU(),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════
#  Load backbone weights — NO training
# ══════════════════════════════════════════════════════════════════════════
def load_backbone(model_name: str, weight_path: str, device: str) -> nn.Module:
    """Instantiate architecture, load saved state_dict, return frozen eval model."""
    registry = {
        "ConvNeXt-Tiny": lambda: ConvNeXtClassifier(num_classes=NUM_CLASSES),
        "ResNet50":      lambda: ResNet50Classifier(num_classes=NUM_CLASSES),
        "VGG16":         lambda: VGG16Classifier(num_classes=NUM_CLASSES),
    }
    if model_name not in registry:
        raise ValueError(f"Unknown model name: {model_name}")

    model = registry[model_name]()
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    # Freeze all parameters — we never back-prop through the backbone
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"  ✓ Loaded {model_name}  ←  {weight_path}")
    return model


def extract_features(model: nn.Module, dataloader: DataLoader, device: str):
    """
    ConvNeXt-aware extraction: runs backbone.features + global avg pool only,
    skipping the classifier head to get raw 768-dim feature vectors.
    """
    model.eval()

    X_list, y_list = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            x = model.backbone.features(images)   # (B, 768, H, W)
            x = x.mean([-2, -1])                  # Global avg pool → (B, 768)
            X_list.append(x.cpu().numpy())
            y_list.append(labels.numpy())

    return np.vstack(X_list), np.concatenate(y_list)

# ══════════════════════════════════════════════════════════════════════════
#  Feature extraction via forward hook (backbone stays frozen)
# ══════════════════════════════════════════════════════════════════════════
# def extract_features(model: nn.Module, dataloader: DataLoader, device: str):
#     """
#     Hook the last pooling/flatten layer to collect penultimate features.
#     Returns (X: ndarray [N, D], y: ndarray [N]).
#     """
#     model.eval()

#     # Find best hook target: prefer Flatten > AdaptiveAvgPool
#     hook_layer = None
#     for module in model.modules():
#         if isinstance(module, (nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool1d)):
#             hook_layer = module
#         if isinstance(module, nn.Flatten):
#             hook_layer = module          # already 1-D, prefer this

#     if hook_layer is None:
#         children = list(model.children())
#         hook_layer = children[-2] if len(children) >= 2 else children[-1]

#     captured = []

#     def _hook(_, __, output):
#         captured.append(output.detach().cpu())

#     handle = hook_layer.register_forward_hook(_hook)

#     X_list, y_list = [], []
#     with torch.no_grad():
#         for images, labels in dataloader:
#             captured.clear()
#             model(images.to(device))
#             feat = captured[0]
#             if feat.dim() > 2:                          # (B, C, H, W) → (B, C)
#                 feat = feat.mean(dim=list(range(2, feat.dim())))
#             X_list.append(feat.numpy())
#             y_list.append(labels.numpy())

#     handle.remove()
#     return np.vstack(X_list), np.concatenate(y_list)


# ══════════════════════════════════════════════════════════════════════════
#  MLP / SVM train + predict helpers
# ══════════════════════════════════════════════════════════════════════════
def train_mlp(X_tr, y_tr, num_classes, device, epochs, model_type):
    X_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_tr,  dtype=torch.long).to(device)

    mlp = (BetterMLP if model_type == "deep" else MLPClassifier)(
        X_tr.shape[1], num_classes
    ).to(device)

    opt  = optim.Adam(mlp.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    mlp.train()
    for ep in range(epochs):
        opt.zero_grad()
        loss = crit(mlp(X_t), y_t)
        loss.backward()
        opt.step()
        if (ep + 1) % 5 == 0:
            print(f"      [MLP] epoch {ep+1}/{epochs}  loss={loss.item():.4f}")
    return mlp


def predict_mlp(mlp, X_te, device):
    mlp.eval()
    X_t = torch.tensor(X_te, dtype=torch.float32).to(device)
    preds, probs, times = [], [], []
    with torch.no_grad():
        for i in range(len(X_t)):
            s  = X_t[i].unsqueeze(0)
            t0 = time.perf_counter()
            o  = mlp(s)
            times.append((time.perf_counter() - t0) * 1000)
            probs.append(F.softmax(o, dim=1).cpu().numpy()[0])
            preds.append(int(o.argmax(1).item()))
    return np.array(preds), np.array(probs), float(np.mean(times))


def predict_svm(clf, X_te):
    preds, times = [], []
    for i in range(len(X_te)):
        t0 = time.perf_counter()
        preds.append(clf.predict(X_te[i].reshape(1, -1))[0])
        times.append((time.perf_counter() - t0) * 1000)
    probs = (clf.predict_proba(X_te) if hasattr(clf, "predict_proba")
             else np.eye(NUM_CLASSES)[np.array(preds)])
    return np.array(preds), probs, float(np.mean(times))


# ══════════════════════════════════════════════════════════════════════════
#  Full metric set
# ══════════════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred, y_prob, num_classes):
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_auc_score

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score   (y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score       (y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    spec_vals = []
    for c in range(num_classes):
        tn = cm.sum() - (cm[c].sum() + cm[:, c].sum() - cm[c, c])
        fp = cm[:, c].sum() - cm[c, c]
        spec_vals.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    spec = float(np.mean(spec_vals))

    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    try:
        auc = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="weighted")
    except ValueError:
        auc = float("nan")

    return dict(accuracy=acc, precision=prec, recall=rec,
                specificity=spec, f1_score=f1, roc_auc=auc)


# ══════════════════════════════════════════════════════════════════════════
#  Plot helpers
# ══════════════════════════════════════════════════════════════════════════
def save_confusion_matrix(y_true, y_pred, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred),
                           display_labels=CLASS_NAMES).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"      Saved: {path}")


def save_roc_curve(y_true, y_prob, num_classes, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in range(num_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_prob[:, c])
        ax.plot(fpr, tpr, label=f"{CLASS_NAMES[c]} (AUC={auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(title); ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"      Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════
#  Cross-validation on Firefly-selected features
# ══════════════════════════════════════════════════════════════════════════
def _aggregate(fold_metrics):
    out = {}
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics]
        out[key]          = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals))
    return out


def cv_mlp(X_sel, y, num_classes, device, n_folds, epochs, model_type):
    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_m = []
    for fold, (tr, va) in enumerate(skf.split(X_sel, y)):
        print(f"    [MLP-CV] fold {fold+1}/{n_folds}")
        mlp = train_mlp(X_sel[tr], y[tr], num_classes, device, epochs, model_type)
        preds, probs, _ = predict_mlp(mlp, X_sel[va], device)
        fold_m.append(compute_metrics(y[va], preds, probs, num_classes))
    return _aggregate(fold_m)


def cv_svm(X_sel, y, num_classes, n_folds):
    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_m = []
    for fold, (tr, va) in enumerate(skf.split(X_sel, y)):
        print(f"    [SVM-CV] fold {fold+1}/{n_folds}")
        clf = SVC(kernel="linear", probability=True)
        clf.fit(X_sel[tr], y[tr])
        preds, probs, _ = predict_svm(clf, X_sel[va])
        fold_m.append(compute_metrics(y[va], preds, probs, num_classes))
    return _aggregate(fold_m)


# ══════════════════════════════════════════════════════════════════════════
#  Firefly pipeline for ONE backbone
# ══════════════════════════════════════════════════════════════════════════
def run_firefly_for_backbone(
    backbone_name: str,
    model: nn.Module,
    full_loader: DataLoader,
    device: str,
    out_dir: Path,
) -> dict:

    print(f"\n{'═'*62}")
    print(f"  Firefly pipeline  ←  {backbone_name}")
    print(f"{'═'*62}")

    # Step 1 — extract features (backbone is frozen / eval)
    print("  [1/5] Extracting features from frozen backbone …")
    X, y = extract_features(model, full_loader, device)
    print(f"        Feature matrix: {X.shape}   unique labels: {np.unique(y)}")

    results     = {}
    clf_types   = (["MLP", "SVM"] if FIREFLY_CLASSIFIER == "both"
                   else [FIREFLY_CLASSIFIER])

    for clf_type in clf_types:
        tag  = f"{backbone_name}_{clf_type}"
        stem = tag.lower().replace(" ", "_").replace("-", "_")

        # Step 2 — Firefly feature selection
        print(f"\n  [2/5] Firefly ({clf_type}) — searching optimal feature subset …")
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
        n_sel = int(np.sum(best_mask))
        print(f"        → Selected {n_sel}/{X.shape[1]} features  "
              f"(firefly best val-acc = {best_val_acc:.4f})")

        # Persist mask
        np.save(out_dir / f"{stem}_mask.npy", best_mask)
        with open(out_dir / f"{stem}_mask.json", "w") as fh:
            json.dump(best_mask.tolist(), fh)

        X_sel = X[:, best_mask == 1]

        # Step 3 — train/test split on selected features
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_sel, y, test_size=0.2, stratify=y, random_state=42
        )

        # Step 4 — train final classifier
        print(f"  [3/5] Training final {clf_type} on {n_sel} selected features …")
        t0 = time.perf_counter()

        if clf_type == "MLP":
            clf = train_mlp(X_tr, y_tr, NUM_CLASSES, device,
                            FIREFLY_MLP_EPOCHS, FIREFLY_MLP_MODEL)
            torch.save(clf.state_dict(), out_dir / f"{stem}_mlp.pth")
            y_pred, y_prob, avg_ms = predict_mlp(clf, X_te, device)
        else:
            clf = SVC(kernel="linear", probability=True)
            clf.fit(X_tr, y_tr)
            joblib.dump(clf, out_dir / f"{stem}_svm.pkl")
            y_pred, y_prob, avg_ms = predict_svm(clf, X_te)

        train_min = (time.perf_counter() - t0) / 60

        # Step 5 — compute all metrics
        print(f"  [4/5] Computing metrics …")
        metrics = compute_metrics(y_te, y_pred, y_prob, NUM_CLASSES)

        print(f"\n  ┌── {tag} {'─'*(50 - len(tag))}")
        print(f"  │  Accuracy    : {metrics['accuracy']*100:.2f}%")
        print(f"  │  Precision   : {metrics['precision']:.4f}")
        print(f"  │  Recall      : {metrics['recall']:.4f}")
        print(f"  │  Specificity : {metrics['specificity']:.4f}")
        print(f"  │  F1-score    : {metrics['f1_score']:.4f}")
        print(f"  │  ROC-AUC     : {metrics['roc_auc']:.4f}")
        print(f"  │  Infer time  : {avg_ms:.3f} ms/sample")
        print(f"  └{'─'*52}")
        print(classification_report(y_te, y_pred,
                                    target_names=CLASS_NAMES, zero_division=0))

        save_confusion_matrix(y_te, y_pred, tag,
                              str(out_dir / f"{stem}_confusion.png"))
        save_roc_curve(y_te, y_prob, NUM_CLASSES, tag,
                       str(out_dir / f"{stem}_roc.png"))

        # Step 6 — 5-fold CV on full selected features
        print(f"  [5/5] {CV_FOLDS}-fold cross-validation …")
        if clf_type == "MLP":
            cv = cv_mlp(X_sel, y, NUM_CLASSES, device,
                        CV_FOLDS, FIREFLY_MLP_EPOCHS, FIREFLY_MLP_MODEL)
        else:
            cv = cv_svm(X_sel, y, NUM_CLASSES, CV_FOLDS)

        print(f"\n  CV results ({tag}):")
        for k, v in cv.items():
            if not k.endswith("_std"):
                print(f"    {k:<14}: {v:.4f} ± {cv[k+'_std']:.4f}")

        results[tag] = {
            "metrics":        metrics,
            "cv_summary":     cv,
            "n_selected":     n_sel,
            "n_total":        int(X.shape[1]),
            "train_time_min": round(train_min, 4),
            "infer_ms":       round(avg_ms, 4),
        }

    return results


# ══════════════════════════════════════════════════════════════════════════
#  Final comparison tables
# ══════════════════════════════════════════════════════════════════════════
def print_comparison_tables(all_results: dict):
    W    = 104
    sep  = "─" * W
    dbl  = "═" * W

    # ── Holdout test metrics ───────────────────────────────────────────────
    print(f"\n{dbl}")
    print(f"{'FIREFLY — HOLDOUT TEST SET METRICS':^{W}}")
    print(dbl)
    hfmt = "{:<34} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}"
    rfmt = "{:<34} {:>7.2f} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>10.3f}"
    print(hfmt.format("Variant", "Acc%", "Prec", "Recall",
                       "Spec", "F1", "AUC", "ms/sample"))
    print(sep)
    for variant, res in all_results.items():
        m = res["metrics"]
        print(rfmt.format(
            variant,
            m["accuracy"] * 100, m["precision"], m["recall"],
            m["specificity"],    m["f1_score"],  m["roc_auc"],
            res["infer_ms"],
        ))
    print(sep)

    # ── CV table ───────────────────────────────────────────────────────────
    print(f"\n{dbl}")
    print(f"{'FIREFLY — 5-FOLD CV  (mean ± std)':^{W}}")
    print(dbl)
    cvfmt = "{:<34} {:>18} {:>18} {:>18} {:>14}"
    print(cvfmt.format("Variant", "Accuracy", "F1", "Recall", "AUC"))
    print(sep)
    for variant, res in all_results.items():
        cv = res["cv_summary"]
        print(cvfmt.format(
            variant,
            f"{cv['accuracy']:.4f} ± {cv['accuracy_std']:.4f}",
            f"{cv['f1_score']:.4f} ± {cv['f1_score_std']:.4f}",
            f"{cv['recall']:.4f}  ± {cv['recall_std']:.4f}",
            f"{cv['roc_auc']:.4f} ± {cv['roc_auc_std']:.4f}",
        ))
    print(sep)

    # ── Feature reduction summary ──────────────────────────────────────────
    print(f"\n{dbl}")
    print(f"{'FEATURE SELECTION SUMMARY':^{W}}")
    print(dbl)
    sfmt = "{:<34} {:>12} {:>12} {:>14} {:>14}"
    print(sfmt.format("Variant", "Selected", "Total", "Reduction%", "Train(min)"))
    print(sep)
    for variant, res in all_results.items():
        n_sel, n_tot = res["n_selected"], res["n_total"]
        red = (1 - n_sel / n_tot) * 100
        print(sfmt.format(
            variant, n_sel, n_tot,
            f"{red:.1f}%", f"{res['train_time_min']:.2f}",
        ))
    print(sep)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN — load weights → extract → firefly → evaluate  (no CNN re-training)
# ══════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = Path(OUTPUT_DIR)
    print(f"Device : {DEVICE}")
    print(f"Output : {OUTPUT_DIR}\n")

    # ── Dataset loader (no augmentation, shuffle=False for stable labels) ─
    print("Building dataset …")
    full_dataset = PreMaskedClassificationDataset(
        root_dir=DATASET_PATH,
        image_size=IMAGE_SIZE,
    )
    full_loader = DataLoader(
        full_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    print(f"  Total samples : {len(full_dataset)}\n")

    all_results = {}

    for model_name, weight_path in WEIGHT_PATHS.items():
        # ── Skip if path not provided or file missing ──────────────────────
        if weight_path is None:
            print(f"  Skipping {model_name}  (weight_path = None)\n")
            continue
        if not os.path.exists(weight_path):
            print(f"  WARNING: file not found, skipping {model_name}\n"
                  f"           path = {weight_path}\n")
            continue

        print(f"\n{'━'*62}")
        print(f"  Backbone : {model_name}")
        print(f"{'━'*62}")

        # Load weights — backbone is never trained here
        model = load_backbone(model_name, weight_path, DEVICE)

        # Run full Firefly pipeline
        res = run_firefly_for_backbone(
            backbone_name=model_name,
            model=model,
            full_loader=full_loader,
            device=DEVICE,
            out_dir=out,
        )
        all_results.update(res)

        # Release GPU memory before loading the next backbone
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    if not all_results:
        print("\nNo results — check that WEIGHT_PATHS point to existing .pth files.")
        return

    # ── Print all comparison tables ────────────────────────────────────────
    print_comparison_tables(all_results)

    # ── Save JSON ──────────────────────────────────────────────────────────
    def _serial(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(type(obj))

    result_path = out / "all_firefly_results.json"
    with open(result_path, "w") as fh:
        json.dump(all_results, fh, default=_serial, indent=2)
    print(f"\n  Results JSON : {result_path}")
    print(f"\n✅  Done.  All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
