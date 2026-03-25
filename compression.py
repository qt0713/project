import zlib
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.cluster import KMeans


def apply_conv_pruning(model: nn.Module, amount: float = 0.2) -> None:
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")


def quantize_weights(weights: np.ndarray, n_clusters: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w_flat = weights.flatten().reshape(-1, 1)
    min_w, max_w = w_flat.min(), w_flat.max()
    init_centers = np.linspace(min_w, max_w, n_clusters).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, init=init_centers, n_init=1)
    labels = kmeans.fit_predict(w_flat)
    centers = kmeans.cluster_centers_.flatten()
    quantized = centers[labels].reshape(weights.shape)
    return quantized, centers, labels


def build_quantized_dict(model: nn.Module, n_clusters: int = 256) -> Dict[str, Dict[str, np.ndarray]]:
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            w = module.weight.data.cpu().numpy()
            quantized_w, centers, labels = quantize_weights(w, n_clusters=n_clusters)
            result[name] = {
                "quantized_w": quantized_w,
                "centers": centers,
                "labels": labels,
            }
    return result


def load_quantized_weights(model: nn.Module, quantized_dict: Dict[str, Dict[str, np.ndarray]]) -> None:
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name in quantized_dict:
            module.weight.data = torch.tensor(
                quantized_dict[name]["quantized_w"],
                dtype=module.weight.data.dtype,
                device=module.weight.data.device,
            )


def print_huffman_like_ratio(quantized_dict: Dict[str, Dict[str, np.ndarray]]) -> None:
    for name, qinfo in quantized_dict.items():
        indices = np.array(qinfo["labels"], dtype=np.uint8)
        compressed = zlib.compress(indices.tobytes())
        ratio = indices.nbytes / len(compressed)
        print(f"{name} compression ratio: {ratio:.2f}")


def collect_conv_sparsity(model: nn.Module) -> List[Dict[str, float]]:
    stats: List[Dict[str, float]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            w = module.weight.data.cpu().numpy()
            total = float(w.size)
            nonzero = float(np.count_nonzero(w))
            sparse = 1.0 - (nonzero / total)
            stats.append(
                {
                    "layer": name,
                    "nonzero": nonzero,
                    "total": total,
                    "sparsity": sparse,
                }
            )
    return stats


def finetune_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs: int,
    train_one_epoch_fn,
    eval_model_fn,
    device,
    stage_name: str,
) -> List[float]:
    val_accs: List[float] = []
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch_fn(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_model_fn(model, val_loader, criterion, device)
        val_accs.append(val_acc)
        print(
            f"[{stage_name}] Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )
        scheduler.step()
    return val_accs


def compute_storage_breakdown(model: nn.Module, quantized_dict: Dict[str, Dict[str, np.ndarray]]) -> pd.DataFrame:
    orig_total = 0.0
    pruned_total = 0.0

    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            w = module.weight.data.cpu().numpy()
            orig_total += w.size * 4
            nonzero = np.count_nonzero(w)
            pruned_total += nonzero * 4 + nonzero * np.ceil(np.log2(max(w.size, 1)) / 8)

    quant_total = 0.0
    huffman_total = 0.0
    for qinfo in quantized_dict.values():
        labels = np.array(qinfo["labels"], dtype=np.uint8)
        centers = np.array(qinfo["centers"], dtype=np.float32)
        quant_total += labels.nbytes + centers.nbytes
        compressed = zlib.compress(labels.tobytes())
        huffman_total += len(compressed) + centers.nbytes

    storage_kb = [orig_total / 1024, pruned_total / 1024, quant_total / 1024, huffman_total / 1024]
    labels = ["original", "prune", "quantization", "huffman"]
    return pd.DataFrame({"storage_kb": storage_kb}, index=labels)


def plot_storage_breakdown(df: pd.DataFrame) -> None:
    labels = df.index.tolist()
    values = df["storage_kb"].tolist()
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=["skyblue", "orange", "green", "slateblue"])
    plt.ylabel("Storage (KB)")
    plt.title("Model Storage Comparison")
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=11)
    plt.show()


def plot_confusion(model, loader, classes, device) -> None:
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels_idx = labels.argmax(dim=1).cpu().numpy()
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_idx)
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()


def plot_quant_centers(quantized_dict: Dict[str, Dict[str, np.ndarray]]) -> None:
    centers_all = []
    for qinfo in quantized_dict.values():
        centers_all.extend(qinfo["centers"])
    plt.figure(figsize=(8, 4))
    plt.hist(centers_all, bins=50, color="slateblue")
    plt.title("Distribution of Quantization Centers")
    plt.xlabel("Center Value")
    plt.ylabel("Frequency")
    plt.show()


def plot_finetune_compare(prune_accs: List[float], quant_accs: List[float]) -> None:
    if not prune_accs or not quant_accs:
        return
    epochs = list(range(1, len(prune_accs) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, prune_accs, "-o", label="Prune Finetune")
    plt.plot(epochs, quant_accs, "-o", label="Quantization Finetune")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Pruning vs Quantization Finetune")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
