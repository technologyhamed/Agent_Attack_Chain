import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


def evaluate_model(model, test_samples, device="cpu"):
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for sample in test_samples:
            data = sample["graph"].to(device)
            target_id = sample["target_id"]

            logits = model(data)
            probs = torch.softmax(logits, dim=-1)

            pred_id = torch.argmax(probs).item()

            y_true.append(target_id)
            y_pred.append(pred_id)
            y_prob.append(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    auc = None
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm,
    }
    return metrics
