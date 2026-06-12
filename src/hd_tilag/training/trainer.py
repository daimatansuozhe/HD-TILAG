from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from hd_tilag.training.dataset import WindowDataset, load_processed
from hd_tilag.training.metrics import backtest_metrics, classification_metrics
from hd_tilag.utils.seed import set_seed


def train_model(config: dict[str, Any], processed_path: str | Path, project_root: Path | None = None) -> Path:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise RuntimeError("Training requires PyTorch. Install dependencies with `pip install -e .`.") from exc

    from hd_tilag.models import HDTILAG

    project_root = project_root or Path.cwd()
    set_seed(int(config.get("seed", 42)))
    data = load_processed(processed_path)
    training_cfg = config["training"]
    model_cfg = config["model"]
    lookback = int(training_cfg["lookback_window"])

    device = _device(config.get("device", "auto"))
    train_ds = WindowDataset(data, lookback=lookback, split="train")
    val_ds = WindowDataset(data, lookback=lookback, split="validation")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(training_cfg.get("num_workers", 0)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(training_cfg.get("num_workers", 0)),
    )

    model = HDTILAG(
        time_feature_dim=data.time_feature_dim,
        sentiment_dim=data.sentiment_features.shape[-1],
        graph_dim=int(model_cfg.get("graph_dim", 32)),
        hidden_size=int(model_cfg.get("hidden_size", 256)),
        num_layers=int(model_cfg.get("num_layers", 3)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        per_stock_output=bool(model_cfg.get("per_stock_output", True)),
        num_stocks=data.num_stocks,
    ).to(device)

    industry_adj = torch.as_tensor(data.industry_adj, dtype=torch.float32, device=device)
    style_adj = torch.as_tensor(data.style_adj, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.0001)),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(training_cfg.get("step_size", 15)),
        gamma=float(training_cfg.get("gamma", 0.5)),
    )

    output_dir = _resolve(project_root, training_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.pt"
    history: list[dict[str, float]] = []
    best_f1 = -1.0
    patience = int(training_cfg.get("early_stopping_patience", 20))
    stale_epochs = 0

    for epoch in range(1, int(training_cfg["max_epochs"]) + 1):
        train_loss = _run_epoch(
            model,
            train_loader,
            industry_adj,
            style_adj,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=float(training_cfg.get("gradient_clip_norm", 1.0)),
        )
        val_result = evaluate_loader(model, val_loader, industry_adj, style_adj, device)
        scheduler.step()
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_result["loss"],
            **{f"val_{k}": v for k, v in val_result["classification"].items()},
        }
        history.append(row)
        if val_result["classification"]["f1"] > best_f1:
            best_f1 = val_result["classification"]["f1"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "time_feature_dim": data.time_feature_dim,
                    "sentiment_dim": data.sentiment_features.shape[-1],
                    "num_stocks": data.num_stocks,
                    "selected_feature_names": data.selected_feature_names.tolist(),
                },
                best_path,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return best_path


def evaluate_checkpoint(
    config: dict[str, Any],
    processed_path: str | Path,
    checkpoint_path: str | Path,
    split: str = "test",
) -> dict[str, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise RuntimeError("Evaluation requires PyTorch. Install dependencies with `pip install -e .`.") from exc

    from hd_tilag.models import HDTILAG

    data = load_processed(processed_path)
    device = _device(config.get("device", "auto"))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = config["model"]
    model = HDTILAG(
        time_feature_dim=int(checkpoint.get("time_feature_dim", data.time_feature_dim)),
        sentiment_dim=int(checkpoint.get("sentiment_dim", data.sentiment_features.shape[-1])),
        graph_dim=int(model_cfg.get("graph_dim", 32)),
        hidden_size=int(model_cfg.get("hidden_size", 256)),
        num_layers=int(model_cfg.get("num_layers", 3)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        per_stock_output=bool(model_cfg.get("per_stock_output", True)),
        num_stocks=data.num_stocks,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    dataset = WindowDataset(data, lookback=int(config["training"]["lookback_window"]), split=split)
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False)
    industry_adj = torch.as_tensor(data.industry_adj, dtype=torch.float32, device=device)
    style_adj = torch.as_tensor(data.style_adj, dtype=torch.float32, device=device)
    return evaluate_loader(model, loader, industry_adj, style_adj, device)


def evaluate_loader(model, loader, industry_adj, style_adj, device) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    model.eval()
    losses: list[float] = []
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    all_returns: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            time_features = batch["time_features"].to(device=device, dtype=torch.float32)
            sentiment = batch["sentiment_features"].to(device=device, dtype=torch.float32)
            labels = batch["labels"].to(device=device, dtype=torch.float32)
            mask = batch["mask"].to(device=device, dtype=torch.float32)
            logits = model(time_features, sentiment, industry_adj, style_adj)
            raw_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            loss = (raw_loss * mask).sum() / mask.sum().clamp_min(1.0)
            losses.append(float(loss.item()))
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            all_masks.append(mask.detach().cpu().numpy())
            all_returns.append(batch["returns"].detach().cpu().numpy())

    probs = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, 0))
    labels = np.concatenate(all_labels, axis=0) if all_labels else np.empty((0, 0))
    masks = np.concatenate(all_masks, axis=0) if all_masks else np.empty((0, 0))
    returns = np.concatenate(all_returns, axis=0) if all_returns else np.empty((0, 0))
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "classification": classification_metrics(probs, labels, masks),
        "backtest": backtest_metrics(probs, returns, masks),
    }


def _run_epoch(
    model,
    loader,
    industry_adj,
    style_adj,
    device,
    optimizer,
    gradient_clip_norm: float,
) -> float:
    import torch
    import torch.nn.functional as F

    model.train()
    losses: list[float] = []
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        time_features = batch["time_features"].to(device=device, dtype=torch.float32)
        sentiment = batch["sentiment_features"].to(device=device, dtype=torch.float32)
        labels = batch["labels"].to(device=device, dtype=torch.float32)
        mask = batch["mask"].to(device=device, dtype=torch.float32)
        logits = model(time_features, sentiment, industry_adj, style_adj)
        raw_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        loss = (raw_loss * mask).sum() / mask.sum().clamp_min(1.0)
        loss.backward()
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def _device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _resolve(project_root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_root / path
