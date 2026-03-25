import json
import os
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for inputs, labels in tqdm(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        labels_idx = labels.argmax(dim=1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += (preds == labels_idx).sum().item()
        total += labels.size(0)

    return running_loss / total, running_corrects / total


def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            labels_idx = labels.argmax(dim=1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels_idx).sum().item()
            total += labels.size(0)

    return running_loss / total, running_corrects / total


def train_and_record(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs: int,
    model_name: str,
    device,
    save_history_json: bool = True,
    history_dir: str = "artifacts",
) -> Dict[str, List[float]]:
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_model_{model_name}.pth")

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "best_val_acc": best_val_acc,
    }

    if save_history_json:
        save_history(history, model_name=model_name, history_dir=history_dir)

    return history


def save_history(history: Dict[str, List[float]], model_name: str, history_dir: str = "artifacts") -> str:
    os.makedirs(history_dir, exist_ok=True)
    out_path = os.path.join(history_dir, f"history_{model_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return out_path


def load_history(model_name: str, history_dir: str = "artifacts") -> Dict[str, List[float]]:
    history_path = os.path.join(history_dir, f"history_{model_name}.json")
    with open(history_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_train_curves(history: Dict[str, List[float]], model_name: str) -> None:
    epochs = list(range(1, len(history["train_losses"]) + 1))
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_losses"], "-o", label="Train Loss")
    plt.plot(epochs, history["val_losses"], "-o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss vs Epoch ({model_name})")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_accs"], "-o", label="Train Acc")
    plt.plot(epochs, history["val_accs"], "-o", label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs Epoch ({model_name})")
    plt.legend()

    plt.tight_layout()
    plt.show()


def visualize_predictions(model, dataloader, classes, device, num_images: int = 8) -> None:
    model.eval()
    inputs, labels = next(iter(dataloader))
    num_images = min(num_images, len(inputs))
    inputs = inputs[:num_images].to(device)
    labels = labels[:num_images]

    with torch.no_grad():
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu()

    rows = (num_images + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows))
    axes = np.array(axes).reshape(rows, 4)

    for i in range(num_images):
        r, c = divmod(i, 4)
        img = inputs[i].cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img * std + mean).permute(1, 2, 0).numpy()

        axes[r, c].imshow(np.clip(img, 0, 1))
        axes[r, c].axis("off")
        true_idx = labels[i].argmax().item()
        axes[r, c].set_title(
            f"P:{classes[preds[i]]}\\nT:{classes[true_idx]}",
            color=("green" if preds[i] == true_idx else "red"),
        )

    for i in range(num_images, rows * 4):
        r, c = divmod(i, 4)
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.show()


def train_multiple_models(
    model_names: Sequence[str],
    get_model_fn,
    num_classes: int,
    crop_size: int,
    train_loader,
    val_loader,
    num_epochs: int,
    device,
) -> Dict[str, Dict[str, List[float]]]:
    results: Dict[str, Dict[str, List[float]]] = {}

    for model_name in model_names:
        print(f"Training model: {model_name}")
        model = get_model_fn(model_name, num_classes=num_classes, crop_size=crop_size).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        history = train_and_record(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            model_name=model_name,
            device=device,
        )
        results[model_name] = history
        plot_train_curves(history, model_name)

    return results


def plot_model_val_compare(results: Dict[str, Dict[str, List[float]]]) -> None:
    if not results:
        return

    plt.figure(figsize=(10, 6))
    for model_name, history in results.items():
        if "val_accs" in history:
            plt.plot(history["val_accs"], label=model_name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy Comparison")
    plt.legend()
    plt.show()
