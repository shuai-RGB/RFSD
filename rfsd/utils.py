"""Sparse graph and evaluation-data utilities used by RFSD."""

from __future__ import annotations

import numpy as np
import torch
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg


def resolve_torch_dtype(dtype) -> torch.dtype:
    """Resolve common NumPy and string dtype values to a PyTorch dtype."""

    if dtype is None:
        return torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        aliases = {
            "float": torch.float32,
            "float32": torch.float32,
            "fp32": torch.float32,
            "double": torch.float64,
            "float64": torch.float64,
            "fp64": torch.float64,
            "half": torch.float16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "long": torch.int64,
            "int64": torch.int64,
            "int": torch.int32,
            "int32": torch.int32,
            "int16": torch.int16,
            "int8": torch.int8,
            "uint8": torch.uint8,
            "bool": torch.bool,
        }
        try:
            return aliases[dtype.lower()]
        except KeyError as error:
            raise TypeError(f"Unsupported dtype string: {dtype}") from error
    numpy_to_torch = {
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.float16: torch.float16,
        np.int64: torch.int64,
        np.int32: torch.int32,
        np.int16: torch.int16,
        np.int8: torch.int8,
        np.uint8: torch.uint8,
        np.bool_: torch.bool,
    }
    try:
        return numpy_to_torch[np.dtype(dtype).type]
    except (KeyError, TypeError) as error:
        raise TypeError(f"Unsupported dtype: {dtype}") from error


def scipy_to_torch_sparse(
    matrix: sparse.spmatrix,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert a SciPy sparse matrix to a coalesced PyTorch COO tensor."""

    dtype = resolve_torch_dtype(dtype)
    matrix = matrix.tocoo()
    shape = tuple(map(int, matrix.shape))
    if matrix.nnz == 0:
        indices = torch.empty((2, 0), dtype=torch.long)
        values = torch.empty((0,), dtype=dtype)
    else:
        indices_array = np.vstack(
            (matrix.row.astype(np.int64, copy=False), matrix.col.astype(np.int64, copy=False))
        )
        indices = torch.from_numpy(indices_array)
        values = torch.from_numpy(matrix.data).to(dtype=dtype)
    tensor = torch.sparse_coo_tensor(indices, values, size=shape).coalesce()
    return tensor.to(device) if device is not None else tensor


def normalize_symmetric_graph(graph: torch.Tensor) -> torch.Tensor:
    """Apply symmetric degree normalization to a square sparse graph."""

    graph = graph.coalesce()
    row, column = graph.indices()
    values = graph.values()
    degree = torch.zeros(graph.size(0), device=graph.device).scatter_add(0, row, values.abs())
    inverse_sqrt_degree = degree.pow(-0.5)
    inverse_sqrt_degree[torch.isinf(inverse_sqrt_degree)] = 0.0
    normalized_values = inverse_sqrt_degree[row] * values * inverse_sqrt_degree[column]
    return torch.sparse_coo_tensor(
        graph.indices(), normalized_values, graph.size(), device=graph.device
    ).coalesce()


def build_mutual_jaccard_graph(
    adjacency: sparse.spmatrix,
    top_k: int = 10,
    min_support: int = 2,
) -> torch.Tensor:
    """Build a normalized mutual-top-k Jaccard graph."""

    binary = (adjacency > 0).astype(float)
    degrees = np.asarray(binary.sum(axis=1)).ravel()
    intersections = binary @ binary.T
    intersections.data[intersections.data < min_support] = 0
    intersections.eliminate_zeros()

    rows, columns = intersections.nonzero()
    denominator = degrees[rows] + degrees[columns] - intersections.data
    similarities = intersections.data / denominator
    similarity_matrix = sparse.csr_matrix(
        (similarities, (rows, columns)), shape=intersections.shape
    )

    selected_rows: list[int] = []
    selected_columns: list[int] = []
    selected_values: list[float] = []
    for row_index in range(similarity_matrix.shape[0]):
        start, end = similarity_matrix.indptr[row_index : row_index + 2]
        columns_in_row = similarity_matrix.indices[start:end]
        values_in_row = similarity_matrix.data[start:end]
        non_diagonal = columns_in_row != row_index
        columns_in_row = columns_in_row[non_diagonal]
        values_in_row = values_in_row[non_diagonal]
        if len(values_in_row) > top_k:
            top_indices = np.argpartition(values_in_row, -top_k)[-top_k:]
            columns_in_row = columns_in_row[top_indices]
            values_in_row = values_in_row[top_indices]
        selected_rows.extend([row_index] * len(values_in_row))
        selected_columns.extend(columns_in_row.tolist())
        selected_values.extend(values_in_row.tolist())

    top_k_graph = sparse.csr_matrix(
        (selected_values, (selected_rows, selected_columns)), shape=similarity_matrix.shape
    )
    mutual_mask = top_k_graph.multiply(top_k_graph.T)
    mutual_mask.data[:] = 1.0
    mutual_graph = top_k_graph.multiply(mutual_mask)
    return normalize_symmetric_graph(scipy_to_torch_sparse(mutual_graph))


def mutual_top_k_graph(top_k_graph: sparse.spmatrix) -> torch.Tensor:
    """Keep reciprocal edges from an already-selected top-k graph."""

    top_k_graph = top_k_graph.tocsr()
    mutual_mask = top_k_graph.multiply(top_k_graph.T)
    mutual_mask.data[:] = 1.0
    return normalize_symmetric_graph(
        scipy_to_torch_sparse(top_k_graph.multiply(mutual_mask))
    )


def pairwise_squared_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return all pairwise squared Euclidean distances."""

    left_norm = left.square().sum(dim=1, keepdim=True)
    right_norm = right.square().sum(dim=1, keepdim=True).T
    return (left_norm + right_norm - 2 * left @ right.T).relu()


def build_locality_graph(features: torch.Tensor, num_neighbors: int) -> torch.Tensor:
    """Build the sparse locality-preserving graph used by the original model."""

    num_nodes = features.shape[0]
    if num_neighbors < 1 or num_neighbors >= num_nodes:
        raise ValueError(
            f"num_neighbors must be in [1, {num_nodes - 1}], got {num_neighbors}"
        )

    distances = pairwise_squared_distance(features, features)
    sorted_values, sorted_indices = torch.topk(
        distances, k=num_neighbors + 1, dim=1, largest=False
    )
    boundary = sorted_values[:, num_neighbors]
    neighbor_distances = sorted_values[:, :num_neighbors]
    denominator = num_neighbors * boundary - neighbor_distances.sum(dim=1) + 1e-10
    weights = ((boundary.unsqueeze(1) - neighbor_distances) / denominator.unsqueeze(1)).relu()

    rows = torch.arange(num_nodes, device=features.device).unsqueeze(1)
    rows = rows.expand(-1, num_neighbors).reshape(-1)
    columns = sorted_indices[:, :num_neighbors].reshape(-1)
    indices = torch.stack((rows, columns))
    return torch.sparse_coo_tensor(
        indices, weights.reshape(-1), (num_nodes, num_nodes), device=features.device
    ).coalesce()


def spectral_embeddings(
    adjacency: torch.Tensor,
    embedding_dim: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Extract leading adjacency eigenvectors (legacy preprocessing helper)."""

    adjacency = adjacency.coalesce()
    indices = adjacency.indices().cpu().numpy()
    values = adjacency.values().cpu().numpy()
    scipy_adjacency = sparse.csr_matrix(
        (values, (indices[0], indices[1])), shape=adjacency.shape
    )
    _, eigenvectors = sparse_linalg.eigs(
        scipy_adjacency, k=embedding_dim, which="LM"
    )
    return torch.from_numpy(eigenvectors.real).float().to(device)


def prepare_ranking_data(
    test_interactions: sparse.spmatrix | torch.Tensor,
    num_users: int,
    num_items: int,
    device: torch.device | str,
    train_interactions: sparse.spmatrix | torch.Tensor | None = None,
    validation_interactions: sparse.spmatrix | torch.Tensor | None = None,
    k: int = 20,
) -> dict[str, torch.Tensor | int | None]:
    """Precompute indices and discounts needed for top-k evaluation."""

    def indices_of(
        matrix: sparse.spmatrix | torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if matrix is None:
            return None, None
        if isinstance(matrix, torch.Tensor):
            if matrix.is_sparse:
                indices = matrix.coalesce().indices()
                rows, columns = indices[0], indices[1]
            else:
                rows, columns = torch.nonzero(matrix > 0, as_tuple=True)
        elif hasattr(matrix, "tocoo"):
            coo_matrix = matrix.tocoo()
            rows = torch.from_numpy(coo_matrix.row)
            columns = torch.from_numpy(coo_matrix.col)
        else:
            raise TypeError(f"Unsupported interaction type: {type(matrix)}")
        return rows.long().to(device), columns.long().to(device)

    train_rows, train_columns = indices_of(train_interactions)
    validation_rows, validation_columns = indices_of(validation_interactions)
    test_rows, test_columns = indices_of(test_interactions)
    assert test_rows is not None and test_columns is not None

    positives_per_user = torch.bincount(test_rows, minlength=num_users).float()
    effective_k = min(k, num_items)
    ranks = torch.arange(2, effective_k + 2, device=device, dtype=torch.float32)
    discounts = 1.0 / torch.log2(ranks)
    ideal_cumulative_gain = torch.cat(
        (torch.zeros(1, device=device), discounts.cumsum(dim=0))
    )
    return {
        "train_rows": train_rows,
        "train_columns": train_columns,
        "validation_rows": validation_rows,
        "validation_columns": validation_columns,
        "test_rows": test_rows,
        "test_columns": test_columns,
        "positives_per_user": positives_per_user,
        "discounts": discounts,
        "ideal_cumulative_gain": ideal_cumulative_gain,
        "k": effective_k,
    }


# Compatibility aliases for the original public functions.
normalize_square_tensor = normalize_symmetric_graph
scipy_coo_to_torch_sparse = scipy_to_torch_sparse
get_jaccard_sparse_tensor = build_mutual_jaccard_graph
build_LPG_sparse = build_locality_graph
distance = pairwise_squared_distance
recipe_topk = mutual_top_k_graph
spectral_decomposition = spectral_embeddings
_ensure_torch_dtype = resolve_torch_dtype


def sparsity(matrix: sparse.spmatrix) -> float:
    return 1.0 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])


def prepare_eval_data(
    R_test,
    n_users,
    n_items,
    device,
    R_train=None,
    R_val=None,
    k=20,
):
    """Compatibility wrapper returning both new and original dictionary keys."""

    data = prepare_ranking_data(
        test_interactions=R_test,
        num_users=n_users,
        num_items=n_items,
        device=device,
        train_interactions=R_train,
        validation_interactions=R_val,
        k=k,
    )
    data.update(
        {
            "train_cols": data["train_columns"],
            "val_rows": data["validation_rows"],
            "val_cols": data["validation_columns"],
            "test_cols": data["test_columns"],
            "pos_per_user": data["positives_per_user"],
            "ideal_cumsum_padded": data["ideal_cumulative_gain"],
        }
    )
    return data
