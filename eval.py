import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from task1_baseline.data import Task1Dataset, build_samples, load_index
from task1_baseline.model import MultiModalBlockageNet
from task1_baseline.utils import compute_metrics, parse_scenarios


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Task 1 baseline")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--index-csv", default=None)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.scenarios = parse_scenarios(args.scenarios)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    e2_columns = checkpoint["e2_columns"]
    e2_mean = checkpoint["e2_mean"]
    e2_std = checkpoint["e2_std"]
    video_mode = checkpoint.get("video_mode", "rgbd")
    image_size = checkpoint.get("image_size", 128)
    backbone = checkpoint.get(
        "backbone", checkpoint.get("config", {}).get("backbone", "simple")
    )

    index_csv = args.index_csv
    if index_csv is None:
        index_csv = Path(args.dataset_root) / "index.csv"

    index_df = load_index(index_csv, task="task1")
    samples, _ = build_samples(
        index_df,
        args.dataset_root,
        args.scenarios,
        dt=checkpoint["config"].get("dt", 0.142),
        label_tolerance=checkpoint["config"].get("label_tolerance", 0.2),
        e2_tolerance=checkpoint["config"].get("e2_tolerance", 0.05),
        e2_columns=e2_columns,
    )

    dataset = Task1Dataset(
        samples,
        image_size=image_size,
        video_mode=video_mode,
        e2_mean=e2_mean,
        e2_std=e2_std,
    )

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MultiModalBlockageNet(
        e2_dim=len(e2_columns),
        video_mode=video_mode,
        num_classes=3,
        backbone=backbone,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    all_logits = []
    all_labels = []
    confusion = np.zeros((3, 3), dtype=np.int64)

    with torch.no_grad():
        for batch in loader:
            video = batch["video"].to(device)
            e2 = batch["e2"].to(device)
            labels = batch["label"].to(device)
            logits = model(video, e2)
            preds = torch.argmax(logits, dim=1)

            for label, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                confusion[label, pred] += 1

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(logits, labels, num_classes=3)

    print("Accuracy: {:.3f}".format(metrics["accuracy"]))
    print("Macro F1: {:.3f}".format(metrics["macro_f1"]))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion)


if __name__ == "__main__":
    main()
