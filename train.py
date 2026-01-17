import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from task1_baseline.data import (
    Task1Dataset,
    build_samples,
    compute_e2_stats,
    load_index,
)
from task1_baseline.model import MultiModalBlockageNet
from task1_baseline.utils import compute_metrics, parse_scenarios, set_seed


def _resolve_scenarios(args, index_df):
    train_scenarios = parse_scenarios(args.train_scenarios)
    val_scenarios = parse_scenarios(args.val_scenarios)

    if args.preset == "loso":
        holdout = args.holdout_scenario
        if holdout is None:
            if len(val_scenarios) == 1:
                holdout = val_scenarios[0]
            else:
                raise ValueError(
                    "LOSO preset requires --holdout-scenario or a single --val-scenarios entry."
                )

        all_scenarios = index_df["scenario_id"].tolist()
        if holdout not in all_scenarios:
            raise ValueError(
                "Holdout scenario {} not found in dataset index.".format(holdout)
            )

        train_scenarios = [sid for sid in all_scenarios if sid != holdout]
        val_scenarios = [holdout]

    return train_scenarios, val_scenarios


def _build_loaders(args, index_df, train_scenarios, val_scenarios):
    train_samples, e2_columns = build_samples(
        index_df,
        args.dataset_root,
        train_scenarios,
        dt=args.dt,
        label_tolerance=args.label_tolerance,
        e2_tolerance=args.e2_tolerance,
    )

    val_samples, _ = build_samples(
        index_df,
        args.dataset_root,
        val_scenarios,
        dt=args.dt,
        label_tolerance=args.label_tolerance,
        e2_tolerance=args.e2_tolerance,
        e2_columns=e2_columns,
    )

    e2_mean, e2_std = compute_e2_stats(train_samples)

    train_dataset = Task1Dataset(
        train_samples,
        image_size=args.image_size,
        video_mode=args.video_mode,
        e2_mean=e2_mean,
        e2_std=e2_std,
    )

    val_dataset = Task1Dataset(
        val_samples,
        image_size=args.image_size,
        video_mode=args.video_mode,
        e2_mean=e2_mean,
        e2_std=e2_std,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, e2_columns, e2_mean, e2_std


def _run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_logits = []
    all_labels = []

    for batch in tqdm(loader, leave=False):
        video = batch["video"].to(device)
        e2 = batch["e2"].to(device)
        labels = batch["label"].to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(video, e2)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    if not all_labels:
        return {"loss": float("nan"), "accuracy": 0.0, "macro_f1": 0.0}

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(logits, labels, num_classes=3)
    metrics["loss"] = total_loss / labels.size(0)
    return metrics


def _compute_class_weights(train_loader, num_classes=3):
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for batch in train_loader:
        labels = batch["label"]
        for cls_id in range(num_classes):
            counts[cls_id] += torch.sum(labels == cls_id).item()
    weights = counts.sum() / (counts + 1e-6)
    weights = weights / weights.mean()
    return weights.float()


def parse_args():
    parser = argparse.ArgumentParser(description="Task 1 PyTorch baseline")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--index-csv", default=None)
    parser.add_argument(
        "--preset",
        choices=["none", "loso"],
        default="none",
        help="Preset data splits (loso = leave-one-scenario-out)",
    )
    parser.add_argument(
        "--holdout-scenario",
        default=None,
        help="Scenario ID to hold out when using --preset loso",
    )
    parser.add_argument(
        "--train-scenarios",
        default="exp1,exp2,exp3,exp4",
        help="Comma-separated list of scenario IDs",
    )
    parser.add_argument(
        "--val-scenarios",
        default="exp5",
        help="Comma-separated list of scenario IDs",
    )
    parser.add_argument("--video-mode", choices=["rgbd", "rgb", "disparity", "none"], default="rgbd")
    parser.add_argument(
        "--backbone",
        choices=["simple", "resnet18"],
        default="simple",
        help="Visual backbone for the video branch",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained weights for the visual backbone",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the visual backbone parameters",
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--dt", type=float, default=0.142)
    parser.add_argument("--label-tolerance", type=float, default=0.2)
    parser.add_argument("--e2-tolerance", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="runs/task1_baseline")
    parser.add_argument(
        "--class-weights",
        choices=["auto", "none"],
        default="auto",
        help="Use automatic class weights for loss",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device (cpu or cuda). Defaults to auto",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    if args.pretrained and args.backbone == "simple":
        raise ValueError("--pretrained requires --backbone resnet18")

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    index_csv = args.index_csv
    if index_csv is None:
        index_csv = Path(args.dataset_root) / "index.csv"
    index_df = load_index(index_csv, task="task1")

    train_scenarios, val_scenarios = _resolve_scenarios(args, index_df)
    train_loader, val_loader, e2_columns, e2_mean, e2_std = _build_loaders(
        args, index_df, train_scenarios, val_scenarios
    )

    model = MultiModalBlockageNet(
        e2_dim=len(e2_columns),
        video_mode=args.video_mode,
        num_classes=3,
        backbone=args.backbone,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    if args.class_weights == "auto":
        class_weights = _compute_class_weights(train_loader).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            model, train_loader, criterion, optimizer=optimizer, device=device
        )
        val_metrics = _run_epoch(model, val_loader, criterion, device=device)

        print(
            "Epoch {:02d} | Train loss {:.4f} acc {:.3f} f1 {:.3f} | "
            "Val loss {:.4f} acc {:.3f} f1 {:.3f}".format(
                epoch,
                train_metrics["loss"],
                train_metrics["accuracy"],
                train_metrics["macro_f1"],
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["macro_f1"],
            )
        )

        checkpoint = {
            "model_state": model.state_dict(),
            "e2_mean": e2_mean,
            "e2_std": e2_std,
            "e2_columns": e2_columns,
            "video_mode": args.video_mode,
            "image_size": args.image_size,
            "backbone": args.backbone,
            "pretrained": args.pretrained,
            "freeze_backbone": args.freeze_backbone,
            "config": vars(args),
            "train_scenarios": train_scenarios,
            "val_scenarios": val_scenarios,
        }

        torch.save(checkpoint, out_dir / "last.pt")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(checkpoint, out_dir / "best.pt")

    print("Best val loss: {:.4f}".format(best_val_loss))


if __name__ == "__main__":
    main()
