import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_scenarios(scenarios):
    if scenarios is None:
        return []
    if isinstance(scenarios, (list, tuple)):
        return list(scenarios)
    return [item.strip() for item in scenarios.split(",") if item.strip()]


def compute_metrics(logits, labels, num_classes=3):
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = float((preds == labels).mean())

    f1_scores = []
    for cls_id in range(num_classes):
        tp = np.sum((preds == cls_id) & (labels == cls_id))
        fp = np.sum((preds == cls_id) & (labels != cls_id))
        fn = np.sum((preds != cls_id) & (labels == cls_id))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1)

    macro_f1 = float(np.mean(f1_scores))
    return {"accuracy": accuracy, "macro_f1": macro_f1}
