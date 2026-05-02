"""
cross_validation.py
-------------------
5-Fold Cross-Validation for ConvNeXt-Tiny only (as per requirements).

Returns per-fold and mean ± std for: Accuracy, F1-score, ROC-AUC.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import time

from models.classifier import ConvNeXtClassifier
from train.loss import get_loss


def run_5fold_cv(
    full_dataset,
    num_classes: int = 4,
    num_epochs: int = 10,
    lr: float = 1e-4,
    batch_size: int = 8,
    device: str = "cuda",
    num_workers: int = 0,
):
    """
    Runs 5-fold CV on ConvNeXt-Tiny.
    full_dataset must expose integer labels at full_dataset.samples[i][2].
    Returns a dict with per-fold results and mean ± std.
    """
    # Collect labels for stratification
    labels = np.array([s[2] for s in full_dataset.samples])
    indices = np.arange(len(labels))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []
    print("\n" + "=" * 60)
    print("5-FOLD CROSS-VALIDATION  (ConvNeXt-Tiny)")
    print("=" * 60)

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        print(f"\n─── Fold {fold+1}/5 ───────────────────────────────────")

        train_loader = DataLoader(
            Subset(full_dataset, train_idx),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
        )
        val_loader = DataLoader(
            Subset(full_dataset, val_idx),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
        )

        model = ConvNeXtClassifier(num_classes=num_classes).to(device)
        criterion = get_loss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        for epoch in range(num_epochs):
            model.train()
            for images, lbls in tqdm(train_loader, desc=f"Fold {fold+1} Ep {epoch+1}", leave=False):
                images, lbls = images.to(device), lbls.to(device)
                loss = criterion(model(images), lbls)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # ── Evaluate ───────────────────────────────────────────────────────
        model.eval()
        all_labels, all_preds, all_probs = [], [], []
        with torch.no_grad():
            for images, lbls in val_loader:
                images = images.to(device)
                logits = model(images)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()
                preds  = logits.argmax(1).cpu().numpy()
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(lbls.numpy())

        all_labels = np.array(all_labels)
        all_preds  = np.array(all_preds)
        all_probs  = np.array(all_probs)

        acc = accuracy_score(all_labels, all_preds)
        f1  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        y_bin = label_binarize(all_labels, classes=list(range(num_classes)))
        roc   = roc_auc_score(y_bin, all_probs, average="macro", multi_class="ovr")

        fold_results.append({"accuracy": acc, "f1_score": f1, "roc_auc": roc})
        print(f"  Acc={acc*100:.2f}%  F1={f1:.4f}  ROC-AUC={roc:.4f}")

    # ── Summary ────────────────────────────────────────────────────────────
    accs = [r["accuracy"] * 100 for r in fold_results]
    f1s  = [r["f1_score"]       for r in fold_results]
    aucs = [r["roc_auc"]        for r in fold_results]

    summary = {
        "fold_results": fold_results,
        "accuracy_mean": np.mean(accs),
        "accuracy_std":  np.std(accs),
        "f1_mean":       np.mean(f1s),
        "f1_std":        np.std(f1s),
        "roc_auc_mean":  np.mean(aucs),
        "roc_auc_std":   np.std(aucs),
    }

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS (ConvNeXt-Tiny Only)")
    print("=" * 60)
    header = f"{'Fold':<8} {'Accuracy (%)':>13} {'F1-Score':>10} {'ROC-AUC':>10}"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(fold_results):
        print(f"Fold {i+1:<3} {r['accuracy']*100:>13.2f} {r['f1_score']:>10.4f} {r['roc_auc']:>10.4f}")
    print("-" * len(header))
    print(
        f"{'Mean±Std':<8} "
        f"{np.mean(accs):>7.2f}±{np.std(accs):.2f}  "
        f"{np.mean(f1s):>5.4f}±{np.std(f1s):.4f}  "
        f"{np.mean(aucs):>5.4f}±{np.std(aucs):.4f}"
    )
    print("=" * 60)

    return summary
