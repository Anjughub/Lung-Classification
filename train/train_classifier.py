"""
train_classifier.py
-------------------
Trains a single classification model and returns full history including
per-epoch accuracy and loss (for training curve plots).
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import time
from pathlib import Path
from train.loss import get_loss


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=20,
    lr=1e-4,
    device="cuda",
    save_path="best_model.pth",
):
    model.to(device)
    criterion = get_loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    best_val_acc = 0.0
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }

    train_start = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print("-" * 30)

        # ── TRAIN ──────────────────────────────────────────────────────────
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc="Training", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss /= total
        train_acc  = correct / total

        # ── VALIDATE ───────────────────────────────────────────────────────
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

        val_loss /= total
        val_acc  = correct / total
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train  Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val    Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            torch.save(model.state_dict(), save_path)
            best_val_acc = val_acc
            print("✅ Saved best model")

    history["training_time_min"] = (time.time() - train_start) / 60.0
    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.4f}")
    return history
