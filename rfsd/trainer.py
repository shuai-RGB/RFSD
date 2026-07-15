"""Training orchestration for RFSD."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.optim import Adam

from .main import RFSDConfig
from .data import RFSDData
from .evaluation import adaptive_bpr_loss, ranking_metrics
from .model import RFSD
from .utils import prepare_ranking_data


@dataclass
class TrainingResult:
    """Summary returned after a complete RFSD training run."""

    best_metrics: dict[str, float]
    final_metrics: dict[str, float]
    best_recall_epoch: int
    elapsed_seconds: float


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested_device: str) -> torch.device:
    """Use CPU when a CUDA device was requested but CUDA is unavailable."""

    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested_device)


def _evaluate(model: RFSD, ranking_data: dict) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        user_embeddings, item_embeddings, _ = model()
        scores = user_embeddings @ item_embeddings.T
    return ranking_metrics(scores, ranking_data)


def train_rfsd(config: RFSDConfig) -> TrainingResult:
    """Load data, train RFSD, and return the best and final metrics."""

    if not isinstance(config, RFSDConfig):
        config = RFSDConfig.from_namespace(config)
    set_random_seed(config.seed)
    device = resolve_device(config.device)
    print(f"RFSD dataset: {config.dataset}")
    print(f"RFSD device: {device}")

    data = RFSDData(
        data_dir=config.data_dir,
        user_top_k=config.user_top_k,
        item_top_k=config.item_top_k,
        noise_rate=config.noise_rate,
        noise_top_k=config.spectral_dim,
    )
    model = RFSD(config, data).to(device)
    print(model)
    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    ranking_data = prepare_ranking_data(
        test_interactions=data.test_interactions,
        train_interactions=data.train_interactions,
        validation_interactions=data.validation_interactions,
        num_users=data.num_users,
        num_items=data.num_items,
        device=device,
        k=config.eval_k,
    )

    best_metrics = {"ndcg": 0.0, "recall": 0.0, "hit_rate": 0.0, "map": 0.0}
    best_recall_epoch = -1
    best_recall = -1.0
    best_model_state = None
    epochs_without_improvement = 0
    started_at = time.time()
    print(f"Early stopping: {'enabled' if config.early_stop else 'disabled'}")

    for epoch in range(config.max_epochs):
        model.train()
        user_embeddings, item_embeddings, contrastive_loss = model()
        ranking_loss = adaptive_bpr_loss(
            model.interaction_graph,
            user_embeddings,
            item_embeddings,
            num_negatives=config.num_negatives,
            regularization_weight=config.regularization_weight,
            margin_threshold=config.hard_negative_margin,
            hardness_scale=config.hard_negative_scale,
        )
        total_loss = ranking_loss + config.contrastive_weight * contrastive_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        metrics = _evaluate(model, ranking_data)
        for metric_name, metric_value in metrics.items():
            best_metrics[metric_name] = max(best_metrics[metric_name], metric_value)
        print(
            f"epoch {epoch}: loss={total_loss.item():.4f}, "
            f"contrastive={contrastive_loss.item() * config.contrastive_weight:.4f}, "
            f"bpr={ranking_loss.item():.4f}, ndcg={metrics['ndcg']:.4f}, "
            f"recall={metrics['recall']:.4f}, hit_rate={metrics['hit_rate']:.4f}, "
            f"map={metrics['map']:.4f}"
        )

        if not config.early_stop:
            if metrics["recall"] > best_recall:
                best_recall = metrics["recall"]
                best_recall_epoch = epoch
            continue
        if metrics["recall"] > best_recall + config.min_delta:
            best_recall = metrics["recall"]
            best_recall_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            print(f"Early stopping metric improved to {best_recall:.6f}")
        else:
            epochs_without_improvement += 1
            print(
                "Early stopping: no improvement for "
                f"{epochs_without_improvement}/{config.patience} epochs"
            )
            if epochs_without_improvement >= config.patience:
                print(
                    f"Stopped at epoch {epoch}; best recall was "
                    f"{best_recall:.6f} at epoch {best_recall_epoch}."
                )
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored RFSD parameters from epoch {best_recall_epoch}.")

    final_metrics = _evaluate(model, ranking_data)
    elapsed_seconds = time.time() - started_at
    print("-" * 60)
    print(
        "best metrics: "
        f"ndcg={best_metrics['ndcg']:.4f}, recall={best_metrics['recall']:.4f}, "
        f"hit_rate={best_metrics['hit_rate']:.4f}, map={best_metrics['map']:.4f}, "
        f"recall_epoch={best_recall_epoch}"
    )
    print(f"running time: {elapsed_seconds:.2f} seconds")
    return TrainingResult(
        best_metrics=best_metrics,
        final_metrics=final_metrics,
        best_recall_epoch=best_recall_epoch,
        elapsed_seconds=elapsed_seconds,
    )


def train(args) -> TrainingResult:
    """Compatibility entry point accepting the original argparse namespace."""

    return train_rfsd(RFSDConfig.from_namespace(args))
