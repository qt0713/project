import tempfile
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_fscore_support,
)


def collect_predictions(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels_idx = labels.argmax(dim=1).cpu().numpy()
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_true.extend(labels_idx.tolist())
            y_pred.extend(preds.tolist())

    return np.array(y_true), np.array(y_pred)


def compute_main_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = float((y_true == y_pred).mean())
    p, r, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "Top1_Accuracy": acc,
        "Macro_Precision": float(p),
        "Macro_Recall": float(r),
        "Macro_F1": float(f1),
    }


def per_class_metrics_df(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str]) -> pd.DataFrame:
    p, r, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        average=None,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "class": classes,
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": support,
        }
    )


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str], title: str = "Confusion Matrix") -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig_size = 10 if len(classes) <= 20 else 14
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Blues", ax=ax, xticks_rotation=90, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def top_confused_pairs(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str], k: int = 10) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred)
    records = []
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if cm[i, j] > 0:
                records.append((classes[i], classes[j], int(cm[i, j])))
    if not records:
        return pd.DataFrame(columns=["true_class", "pred_class", "count"])
    df = pd.DataFrame(records, columns=["true_class", "pred_class", "count"])
    return df.sort_values("count", ascending=False).head(k).reset_index(drop=True)


def model_param_count(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def model_size_mb(model) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:
        torch.save(model.state_dict(), tmp.name)
        size_bytes = tmp.tell()
    return float(size_bytes / (1024 * 1024))


def huffman_estimated_size_mb(quantized_dict) -> float:
    import zlib

    total_bytes = 0
    for qinfo in quantized_dict.values():
        labels = np.array(qinfo["labels"], dtype=np.uint8)
        centers = np.array(qinfo["centers"], dtype=np.float32)
        compressed = zlib.compress(labels.tobytes())
        total_bytes += len(compressed) + centers.nbytes
    return float(total_bytes / (1024 * 1024))


def inference_latency_ms_per_img(model, loader, device, warmup_batches: int = 3, measure_batches: int = 10) -> float:
    model.eval()

    with torch.no_grad():
        for i, (inputs, _) in enumerate(loader):
            if i >= warmup_batches:
                break
            inputs = inputs.to(device)
            _ = model(inputs)

    total_time = 0.0
    total_imgs = 0
    measured = 0
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            total_time += end - start
            total_imgs += inputs.size(0)
            measured += 1
            if measured >= measure_batches:
                break

    if total_imgs == 0:
        return 0.0
    return float(total_time * 1000.0 / total_imgs)
