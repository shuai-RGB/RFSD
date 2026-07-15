"""Data loading and graph preparation for RFSD."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from .utils import build_locality_graph, build_mutual_jaccard_graph, scipy_to_torch_sparse


class RFSDData:
    """Load interaction splits and construct the graphs consumed by RFSD."""

    def __init__(
        self,
        data_dir: str | Path,
        user_top_k: int,
        item_top_k: int,
        noise_rate: float = 0.0,
        noise_top_k: int = 20,
    ) -> None:
        self.data_dir = self._resolve_data_dir(data_dir)
        self.user_top_k = user_top_k
        self.item_top_k = item_top_k
        self.noise_rate = noise_rate
        self.noise_top_k = noise_top_k
        self._load()

    @staticmethod
    def _resolve_data_dir(data_dir: str | Path) -> Path:
        """Resolve relative data paths from either project or current directory."""

        path = Path(data_dir).expanduser()
        if path.is_absolute():
            return path
        working_directory_path = Path.cwd() / path
        if working_directory_path.exists():
            return working_directory_path
        project_path = Path(__file__).resolve().parent.parent / path
        return project_path if project_path.exists() else working_directory_path

    def _load_pickle(self, filename: str):
        path = self.data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"RFSD data file not found: {path}")
        with path.open("rb") as file:
            return pickle.load(file)

    def _load(self) -> None:
        self.user_text_embeddings = torch.from_numpy(
            self._load_pickle("usr_emb_np.pkl")
        ).float()
        self.item_text_embeddings = torch.from_numpy(
            self._load_pickle("itm_emb_np.pkl")
        ).float()
        self.train_interactions = self._load_pickle("trn_mat.pkl").tocoo()
        if self.noise_rate > 0:
            self.train_interactions = add_high_order_noise(
                self.train_interactions,
                noise_rate=self.noise_rate,
                top_k=self.noise_top_k,
            )
        self.test_interactions = self._load_pickle("tst_mat.pkl").tocoo()
        self.validation_interactions = self._load_pickle("val_mat.pkl").tocoo()

        self.interaction_graph = scipy_to_torch_sparse(self.train_interactions)
        self.user_cooccurrence_graph = build_mutual_jaccard_graph(
            self.train_interactions, self.user_top_k
        )
        self.item_cooccurrence_graph = build_mutual_jaccard_graph(
            self.train_interactions.T, self.item_top_k
        )
        self.user_similarity_graph = build_locality_graph(
            self.user_text_embeddings, self.user_top_k
        )
        self.item_similarity_graph = build_locality_graph(
            self.item_text_embeddings, self.item_top_k
        )
        self.num_users, self.num_items = self.train_interactions.shape
        self.user_degrees = torch.from_numpy(
            self.train_interactions.tocsr().sum(axis=1).A1
        ).float()
        self.item_degrees = torch.from_numpy(
            self.train_interactions.tocsr().sum(axis=0).A1
        ).float()

    def load_pickle(self, filename: str):
        """Compatibility alias for the original public loader method."""

        return self._load_pickle(filename)

    def _get_file_path(self, filename: str) -> str:
        return str(self.data_dir / filename)

    def create_cooccurrence_matrices(
        self,
        matrix: sparse.spmatrix,
        user_top_k: int,
        item_top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            build_mutual_jaccard_graph(matrix, user_top_k),
            build_mutual_jaccard_graph(matrix.T, item_top_k),
        )

def add_high_order_noise(
    train_interactions: sparse.spmatrix,
    noise_rate: float = 0.01,
    top_k: int = 20,
) -> sparse.coo_matrix:
    """Add likely high-order user-item edges as controlled training noise."""

    interactions = train_interactions.tocsr().astype(np.float32)
    num_users, _ = interactions.shape
    high_order = (interactions @ interactions.T @ interactions).tolil()
    high_order[interactions.nonzero()] = 0
    high_order = high_order.tocsr()
    num_noise_edges = int(interactions.nnz * noise_rate)
    per_user = max(1, num_noise_edges // num_users)
    noise_rows: list[int] = []
    noise_columns: list[int] = []

    for user_index in range(num_users):
        row = high_order.getrow(user_index)
        if row.nnz == 0:
            continue
        candidate_items = row.indices
        candidate_scores = row.data
        if len(candidate_scores) > top_k:
            top_indices = np.argpartition(candidate_scores, -top_k)[-top_k:]
            candidate_items = candidate_items[top_indices]
            candidate_scores = candidate_scores[top_indices]
        probabilities = candidate_scores / (candidate_scores.sum() + 1e-8)
        sample_size = min(per_user, len(candidate_items))
        chosen = np.random.choice(
            len(candidate_items), size=sample_size, replace=False, p=probabilities
        )
        noise_rows.extend([user_index] * sample_size)
        noise_columns.extend(candidate_items[chosen].tolist())

    if not noise_rows:
        return interactions.tocoo()
    noise = sparse.coo_matrix(
        (np.ones(len(noise_rows), dtype=np.float32), (noise_rows, noise_columns)),
        shape=interactions.shape,
    )
    return ((interactions + noise) > 0).astype(np.float32).tocoo()
