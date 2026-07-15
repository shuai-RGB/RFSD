"""RFSD training objectives and top-k recommendation metrics."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def modality_contrastive_loss(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Symmetric contrastive alignment for concatenated ID/text features."""

    def alignment_loss(embeddings: torch.Tensor) -> torch.Tensor:
        id_embeddings, text_embeddings = torch.chunk(embeddings, 2, dim=1)
        id_embeddings = F.normalize(id_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        logits = id_embeddings @ text_embeddings.T / temperature
        labels = torch.arange(embeddings.size(0), device=embeddings.device)
        return 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.T, labels)
        )

    return 0.5 * (
        alignment_loss(user_embeddings) + alignment_loss(item_embeddings)
    )


def adaptive_bpr_loss(
    interaction_graph: torch.Tensor,
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    num_negatives: int = 5,
    regularization_weight: float = 1e-4,
    margin_threshold: float = 0.3,
    hardness_scale: float = 5.0,
) -> torch.Tensor:
    """Compute BPR loss with dynamic hard-negative sampling and weighting."""

    positive_users, positive_items = interaction_graph.coalesce().indices()
    num_positives = positive_users.numel()
    if num_positives == 0:
        return user_embeddings.new_tensor(0.0)

    device = user_embeddings.device
    num_items = item_embeddings.shape[0]
    user_batch = user_embeddings[positive_users]
    positive_item_batch = item_embeddings[positive_items]
    negative_candidates = torch.randint(
        0, num_items, (num_positives, num_negatives), device=device
    )
    positive_keys = positive_users * num_items + positive_items
    for _ in range(2):
        negative_keys = positive_users.unsqueeze(1) * num_items + negative_candidates
        collisions = torch.isin(negative_keys, positive_keys)
        if not collisions.any():
            break
        negative_candidates[collisions] = torch.randint(
            0, num_items, (collisions.sum().item(),), device=device
        )

    with torch.no_grad():
        candidate_embeddings = item_embeddings[negative_candidates]
        candidate_scores = (user_batch.unsqueeze(1) * candidate_embeddings).sum(dim=2)
        hardest_indices = candidate_scores.argmax(dim=1)
    row_indices = torch.arange(num_positives, device=device)
    negative_items = negative_candidates[row_indices, hardest_indices]
    negative_item_batch = item_embeddings[negative_items]

    positive_scores = (user_batch * positive_item_batch).sum(dim=1)
    negative_scores = (user_batch * negative_item_batch).sum(dim=1)
    score_margin = positive_scores - negative_scores
    adaptive_weight = torch.sigmoid(
        hardness_scale * (margin_threshold - score_margin)
    ).square()
    ranking_loss = (adaptive_weight * F.softplus(-score_margin)).mean()
    regularization = regularization_weight * (
        user_batch.square().sum(dim=1).mean()
        + positive_item_batch.square().sum(dim=1).mean()
        + negative_item_batch.square().sum(dim=1).mean()
    )
    return ranking_loss + regularization


def ranking_metrics(
    score_matrix: torch.Tensor,
    ranking_data: dict,
) -> dict[str, float]:
    """Calculate NDCG, recall, hit rate, and MAP at k."""

    device = score_matrix.device
    num_users, num_items = score_matrix.shape
    k = min(ranking_data["k"], num_items)
    train_rows = ranking_data["train_rows"]
    train_columns = ranking_data.get("train_columns", ranking_data.get("train_cols"))
    validation_rows = ranking_data.get("validation_rows", ranking_data.get("val_rows"))
    validation_columns = ranking_data.get("validation_columns", ranking_data.get("val_cols"))
    test_rows = ranking_data["test_rows"]
    test_columns = ranking_data.get("test_columns", ranking_data.get("test_cols"))
    positives_per_user = ranking_data.get(
        "positives_per_user", ranking_data.get("pos_per_user")
    )
    discounts = ranking_data["discounts"][:k]
    ideal_gain = ranking_data.get(
        "ideal_cumulative_gain", ranking_data.get("ideal_cumsum_padded")
    )

    if train_rows is not None or validation_rows is not None:
        score_matrix = score_matrix.clone()
    if train_rows is not None:
        score_matrix[train_rows, train_columns] = -float("inf")
    if validation_rows is not None:
        score_matrix[validation_rows, validation_columns] = -float("inf")
    top_k_items = torch.topk(score_matrix, k, dim=1).indices
    hits = torch.zeros(num_users, k, dtype=torch.float32, device=device)

    batch_size = 10_240
    for start in range(0, len(test_rows), batch_size):
        rows = test_rows[start : start + batch_size]
        columns = test_columns[start : start + batch_size]
        batch_hits = (top_k_items[rows] == columns.unsqueeze(1)).float()
        hits.scatter_add_(0, rows.unsqueeze(1).expand(-1, k), batch_hits)

    valid_users = positives_per_user > 0
    if not valid_users.any():
        return {"ndcg": 0.0, "recall": 0.0, "hit_rate": 0.0, "map": 0.0}
    hits = hits[valid_users]
    positive_counts = positives_per_user[valid_users].to(device)
    hits_per_user = hits.sum(dim=1)
    hit_rate = (hits_per_user > 0).float().mean()
    recall = (hits_per_user / positive_counts).mean()

    dcg = (hits * discounts).sum(dim=1)
    ideal_k = torch.minimum(positive_counts, torch.tensor(k, device=device)).long()
    idcg = ideal_gain[ideal_k].clamp_min(1e-12)
    ndcg = (dcg / idcg).mean()

    precision = hits.cumsum(dim=1) / torch.arange(
        1, k + 1, device=device, dtype=torch.float32
    ).unsqueeze(0)
    average_precision = (precision * hits).sum(dim=1) / torch.minimum(
        positive_counts, torch.tensor(float(k), device=device)
    )
    return {
        "ndcg": ndcg.item(),
        "recall": recall.item(),
        "hit_rate": hit_rate.item(),
        "map": average_precision.mean().item(),
    }


# Compatibility names used by the original project.
cl_loss = modality_contrastive_loss


def bpr_loss(
    trn_mat,
    f_u,
    f_i,
    num_neg=5,
    reg_weight=1e-4,
    gamma=0.3,
    beta=5.0,
):
    return adaptive_bpr_loss(
        trn_mat,
        f_u,
        f_i,
        num_negatives=num_neg,
        regularization_weight=reg_weight,
        margin_threshold=gamma,
        hardness_scale=beta,
    )


def eva(predR, eval_data):
    metrics = ranking_metrics(predR, eval_data)
    return metrics["ndcg"], metrics["recall"], metrics["hit_rate"], metrics["map"]
