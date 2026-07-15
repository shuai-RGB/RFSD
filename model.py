"""Neural network components for the RFSD recommender."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .main import RFSDConfig


class ProjectionHead(nn.Module):
    """Two-layer projection used to produce cluster logits."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.hidden_layer = nn.Linear(input_dim, input_dim)
        self.output_layer = nn.Linear(input_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output_layer(F.leaky_relu(self.hidden_layer(features)))


class HomogeneousGraphEncoder(nn.Module):
    """Fuse similarity and co-occurrence graphs and propagate two modalities."""

    def __init__(self, text_dim: int, id_dim: int, num_layers: int) -> None:
        super().__init__()
        self.text_projection = nn.Linear(text_dim, id_dim)
        self.graph_mix_logits = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.num_layers = num_layers

    def forward(
        self,
        similarity_graph: torch.Tensor,
        cooccurrence_graph: torch.Tensor,
        id_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        normalized_text = F.normalize(text_embeddings, p=2, dim=-1)
        normalized_ids = F.normalize(id_embeddings, p=2, dim=-1)
        projected_text = self.text_projection(normalized_text)
        features = torch.cat((normalized_ids, projected_text), dim=-1)

        similarity_graph = similarity_graph.coalesce()
        cooccurrence_graph = cooccurrence_graph.coalesce()
        graph_weights = F.softmax(self.graph_mix_logits, dim=0)
        indices = torch.cat(
            (similarity_graph.indices(), cooccurrence_graph.indices()), dim=1
        )
        values = torch.cat(
            (
                similarity_graph.values() * graph_weights[0],
                cooccurrence_graph.values() * graph_weights[1],
            )
        )
        fused_graph = torch.sparse_coo_tensor(
            indices, values, similarity_graph.size(), device=features.device
        ).coalesce()
        for _ in range(self.num_layers):
            features = torch.sparse.mm(fused_graph, features)
        return features


class GatedModalityFusion(nn.Module):
    """Fuse ID/text features and align them with an InfoNCE objective."""

    def __init__(
        self,
        embedding_dim: int,
        temperature: float,
        contrastive_batch_size: int = 2048,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.contrastive_batch_size = contrastive_batch_size
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid(),
        )
        self.text_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, mixed_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        id_embeddings, text_embeddings = torch.split(
            mixed_embeddings, self.embedding_dim, dim=1
        )
        contrastive_loss = (
            self._contrastive_loss(id_embeddings, text_embeddings)
            if self.training
            else mixed_embeddings.new_tensor(0.0)
        )
        text_gate = self.text_scale * self.gate(mixed_embeddings)
        return id_embeddings + text_embeddings * text_gate, contrastive_loss

    def _contrastive_loss(
        self, id_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        num_nodes = id_features.size(0)
        if num_nodes < 2:
            return id_features.new_tensor(0.0)
        shuffled_indices = torch.randperm(num_nodes, device=id_features.device)
        batch_losses = []
        for start in range(0, num_nodes, self.contrastive_batch_size):
            batch_indices = shuffled_indices[start : start + self.contrastive_batch_size]
            if batch_indices.size(0) < 2:
                continue
            id_batch = F.normalize(id_features[batch_indices], dim=1)
            text_batch = F.normalize(text_features[batch_indices], dim=1)
            logits = id_batch @ text_batch.T / self.temperature
            labels = torch.arange(logits.size(0), device=logits.device)
            batch_losses.append(F.cross_entropy(logits, labels))
        if not batch_losses:
            return id_features.new_tensor(0.0)
        return torch.stack(batch_losses).mean()


class RFSD(nn.Module):
    """RFSD: graph-based recommendation with soft cluster-aware propagation."""

    def __init__(self, config: RFSDConfig, data) -> None:
        super().__init__()
        if not isinstance(config, RFSDConfig):
            config = RFSDConfig.from_namespace(config)
        self.config = config
        self.num_users, self.num_items = data.train_interactions.shape

        user_text = data.user_text_embeddings
        item_text = data.item_text_embeddings
        if user_text.shape[1] != config.text_embedding_dim:
            raise ValueError(
                "Configured text_embedding_dim does not match user embeddings: "
                f"{config.text_embedding_dim} != {user_text.shape[1]}"
            )
        if item_text.shape[1] != config.text_embedding_dim:
            raise ValueError(
                "Configured text_embedding_dim does not match item embeddings: "
                f"{config.text_embedding_dim} != {item_text.shape[1]}"
            )

        self.user_text_embeddings = nn.Parameter(user_text.clone())
        self.item_text_embeddings = nn.Parameter(item_text.clone())
        self.user_id_embeddings = nn.Parameter(
            torch.empty(self.num_users, config.id_embedding_dim)
        )
        self.item_id_embeddings = nn.Parameter(
            torch.empty(self.num_items, config.id_embedding_dim)
        )
        self.user_cluster_features = nn.Parameter(
            torch.empty(self.num_users, config.spectral_dim)
        )
        self.item_cluster_features = nn.Parameter(
            torch.empty(self.num_items, config.spectral_dim)
        )
        nn.init.xavier_uniform_(self.user_id_embeddings)
        nn.init.xavier_uniform_(self.item_id_embeddings)
        nn.init.xavier_uniform_(self.user_cluster_features)
        nn.init.xavier_uniform_(self.item_cluster_features)

        self.user_graph_encoder = HomogeneousGraphEncoder(
            config.text_embedding_dim,
            config.id_embedding_dim,
            config.user_graph_layers,
        )
        self.item_graph_encoder = HomogeneousGraphEncoder(
            config.text_embedding_dim,
            config.id_embedding_dim,
            config.item_graph_layers,
        )
        self.cluster_projection = ProjectionHead(
            config.spectral_dim, config.num_clusters
        )
        self.user_fusion = GatedModalityFusion(
            config.id_embedding_dim, config.contrastive_temperature
        )
        self.item_fusion = GatedModalityFusion(
            config.id_embedding_dim, config.contrastive_temperature
        )

        self.register_buffer("interaction_graph", data.interaction_graph, persistent=False)
        self.register_buffer(
            "user_similarity_graph", data.user_similarity_graph, persistent=False
        )
        self.register_buffer(
            "user_cooccurrence_graph", data.user_cooccurrence_graph, persistent=False
        )
        self.register_buffer(
            "item_similarity_graph", data.item_similarity_graph, persistent=False
        )
        self.register_buffer(
            "item_cooccurrence_graph", data.item_cooccurrence_graph, persistent=False
        )

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user_embeddings = self.user_graph_encoder(
            self.user_similarity_graph,
            self.user_cooccurrence_graph,
            self.user_id_embeddings,
            self.user_text_embeddings,
        )
        item_embeddings = self.item_graph_encoder(
            self.item_similarity_graph,
            self.item_cooccurrence_graph,
            self.item_id_embeddings,
            self.item_text_embeddings,
        )

        cluster_features = torch.cat(
            (self.user_cluster_features, self.item_cluster_features), dim=0
        )
        cluster_logits = self.cluster_projection(cluster_features)
        user_logits, item_logits = torch.split(
            cluster_logits, (self.num_users, self.num_items), dim=0
        )
        user_memberships = self._sample_cluster_memberships(user_logits)
        item_memberships = self._sample_cluster_memberships(item_logits)

        user_layers = [user_embeddings]
        item_layers = [item_embeddings]
        num_clusters = self.config.num_clusters
        embedding_width = user_embeddings.shape[1]
        for _ in range(self.config.interaction_layers):
            weighted_items = (
                item_embeddings.unsqueeze(1) * item_memberships.unsqueeze(-1)
            ).reshape(self.num_items, num_clusters * embedding_width)
            messages_to_users = torch.sparse.mm(
                self.interaction_graph, weighted_items
            ).reshape(self.num_users, num_clusters, embedding_width)
            next_user_embeddings = (
                messages_to_users * user_memberships.unsqueeze(-1)
            ).sum(dim=1)

            weighted_users = (
                user_embeddings.unsqueeze(1) * user_memberships.unsqueeze(-1)
            ).reshape(self.num_users, num_clusters * embedding_width)
            messages_to_items = torch.sparse.mm(
                self.interaction_graph.transpose(0, 1), weighted_users
            ).reshape(self.num_items, num_clusters, embedding_width)
            next_item_embeddings = (
                messages_to_items * item_memberships.unsqueeze(-1)
            ).sum(dim=1)

            user_embeddings = next_user_embeddings
            item_embeddings = next_item_embeddings
            user_layers.append(user_embeddings)
            item_layers.append(item_embeddings)

        user_embeddings = torch.stack(user_layers).mean(dim=0)
        item_embeddings = torch.stack(item_layers).mean(dim=0)
        user_embeddings, user_contrastive_loss = self.user_fusion(user_embeddings)
        item_embeddings, item_contrastive_loss = self.item_fusion(item_embeddings)
        contrastive_loss = (user_contrastive_loss + item_contrastive_loss) * 0.5
        return user_embeddings, item_embeddings, contrastive_loss

    def _sample_cluster_memberships(self, logits: torch.Tensor) -> torch.Tensor:
        uniform_noise = torch.rand_like(logits).clamp_(min=1e-10, max=1 - 1e-10)
        gumbel_noise = -torch.log(-torch.log(uniform_noise))
        return F.softmax(
            (logits + gumbel_noise) / self.config.cluster_temperature, dim=-1
        )


# Import compatibility for code that still uses the original component names.
DCGL = RFSD
MLP = ProjectionHead
MLP1 = ProjectionHead


class HomoGraphLearning(HomogeneousGraphEncoder):
    def __init__(self, txt_emb_dim, id_emb_dim, n_layers, degree=None):
        super().__init__(txt_emb_dim, id_emb_dim, n_layers)
        self.degree = degree


class SimpleGatedFusion(GatedModalityFusion):
    def __init__(self, dim, tau, cl_batch_size=2048):
        super().__init__(dim, tau, cl_batch_size)

    def contrastive_loss(self, id_feat, text_feat):
        return self._contrastive_loss(id_feat, text_feat)
