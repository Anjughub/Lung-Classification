"""
valid_classifier.py
-------------------
Evaluate one or all three trained classifiers on the validation split.
Produces all required metrics + visualisations.

Usage
-----
    python valid_classifier.py                # evaluates all 3 models
    python valid_classifier.py --model convnext
"""

import argparse, os, time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from utils.data_create import get_train_val_dataloaders
from models.classifier import ConvNeXtClassifier, VGG16Classifier, ResNet50Classifier
from utils.metrics import (
    compute_all_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    print_metrics_table,
    CLASS_NAMES,
)

# ── CONFIG ─────────────────────────────────────────────────────────────────
DATASET_PATH = r"C:\Users\akhsh\Downloads\lung_data\COVID-19_Radiography_Dataset"
MODEL_PATHS  = {
    "VGG16":         r"framework_results\vgg16_best.pth",
    "ResNet50":      r"framework_results\resnet50_best.pth",
    "ConvNeXt-Tiny": r"framework_results\convnext-tiny_best.pth",
}
OUTPUT_DIR   = "validation_results"
BATCH_SIZE   = 8
IMAGE_SIZE   = (256, 256)
NUM_CLASSES  = 4
NUM_WORKERS  = 0
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# ───────────────────────────────────────────────────────────────────────────


def build_model(name):
    if "vgg" in name.lower():
        return VGG16Classifier(num_classes=NUM_CLASSES)
    elif "resnet" in name.lower():
        return ResNet50Classifier(num_classes=NUM_CLASSES)
    else:
        return ConvNeXtClassifier(num_classes=NUM_CLASSES)


def evaluate_one(model_name, val_loader, out_dir):
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    model = build_model(model_name)
    ckpt  = MODEL_PATHS[model_name]
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.to(DEVICE).eval()

    all_labels, all_preds, all_probs = [], [], []
    per_img_times = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            t0     = time.perf_counter()
            logits = model(images)
            t1     = time.perf_counter()
            per_img_times.append((t1 - t0) * 1000 / images.size(0))
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = compute_all_metrics(y_true, y_pred, y_prob, NUM_CLASSES)
    avg_infer = float(np.mean(per_img_times))

    slug = model_name.lower().replace(" ", "_").replace("-", "")
    plot_confusion_matrix(y_true, y_pred,
                          save_path=str(out_dir / f"{slug}_confusion.png"),
                          class_names=CLASS_NAMES)
    plot_roc_curve(y_true, y_prob,
                   save_path=str(out_dir / f"{slug}_roc.png"),
                   class_names=CLASS_NAMES)

    print(f"  Accuracy:    {metrics['accuracy']*100:.2f}%")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  F1-Score:    {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
    print(f"  Infer time:  {avg_infer:.3f} ms/image")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
                        choices=["all", "convnext", "vgg16", "resnet50"])
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_dir = Path(OUTPUT_DIR)

    _, val_loader = get_train_val_dataloaders(
        root_dir=DATASET_PATH,
        batch_size=BATCH_SIZE,
        val_split=0.2,
        image_size=IMAGE_SIZE,
        num_workers=NUM_WORKERS,
    )

    if args.model == "all":
        models_to_eval = list(MODEL_PATHS.keys())
    elif args.model == "convnext":
        models_to_eval = ["ConvNeXt-Tiny"]
    elif args.model == "vgg16":
        models_to_eval = ["VGG16"]
    else:
        models_to_eval = ["ResNet50"]

    results = {}
    for name in models_to_eval:
        results[name] = evaluate_one(name, val_loader, out_dir)

    if len(results) > 1:
        print_metrics_table(results)

    print(f"\n✅ Validation results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
