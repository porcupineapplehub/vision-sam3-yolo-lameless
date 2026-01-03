"""
CowLamenessGraphormer: True Graph Transformer for Cow Lameness Detection.

This is a complete Graphormer-style model specifically designed for
cow lameness prediction from video features.

Key differences from GraphGPS:
1. Pure attention (no local message passing like GatedGCN)
2. Graph structure encoded in attention biases (centrality, SPD, edge)
3. Virtual node for graph-level representation
4. More interpretable attention patterns
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Dict, Optional, Tuple

from .encodings import GraphormerEncodings
from .layers import GraphormerEncoder, GraphLevelReadout


class CowLamenessGraphormer(nn.Module):
    """
    Graphormer for Cow Lameness Detection.

    Architecture:
    1. Input projection
    2. Graphormer encodings (centrality, spatial, temporal, edge)
    3. Graphormer encoder (transformer layers with graph biases)
    4. Graph-level readout
    5. Prediction head

    Node Features (50-dim):
    - Pose metrics (10): back_arch, head_bob, stride, asymmetry
    - Silhouette (5): area, circularity, aspect_ratio
    - DINOv3 embedding (32): PCA-reduced from 768
    - Metadata (3): timestamp, quality, prior_estimate
    """

    def __init__(
        self,
        input_dim: int = 50,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 512,
        edge_dim: int = 3,
        dropout: float = 0.1,
        max_degree: int = 50,
        max_spd: int = 10,
        use_virtual_node: bool = True,
        use_temporal: bool = True
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Model hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            ffn_dim: Feed-forward network dimension
            edge_dim: Edge feature dimension
            dropout: Dropout rate
            max_degree: Maximum node degree for centrality encoding
            max_spd: Maximum shortest path distance
            use_virtual_node: Whether to use virtual node
            use_temporal: Whether to use temporal encoding
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        # Graphormer encodings
        self.encodings = GraphormerEncodings(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            max_degree=max_degree,
            max_spd=max_spd,
            edge_dim=edge_dim,
            use_temporal=use_temporal
        )

        # Graphormer encoder
        self.encoder = GraphormerEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            use_virtual_node=use_virtual_node
        )

        # Graph readout
        self.readout = GraphLevelReadout(
            hidden_dim=hidden_dim,
            use_virtual_node=use_virtual_node,
            use_attention_pool=True
        )

        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )

        # Node-level prediction (optional)
        self.node_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        data: Data,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            data: PyG Data object with x, edge_index, edge_attr
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with:
            - graph_pred: (1, 1) graph-level prediction
            - node_pred: (N, 1) node-level predictions
            - attention_weights: Optional attention weights
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        timestamps = data.timestamps if hasattr(data, 'timestamps') else None

        num_nodes = x.size(0)

        # Input projection
        h = self.input_proj(x)

        # Compute Graphormer encodings
        node_encoding, attention_bias = self.encodings(
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes,
            timestamps=timestamps
        )

        # Add node encoding to features
        h = h + node_encoding

        # Encode with Graphormer
        h, vn, all_attention = self.encoder(h, attention_bias, return_attention)

        # Graph-level readout
        graph_repr = self.readout(h, vn)

        # Predictions
        graph_pred = torch.sigmoid(self.pred_head(graph_repr))
        node_pred = torch.sigmoid(self.node_pred(h))

        result = {
            'graph_pred': graph_pred,
            'node_pred': node_pred
        }

        if return_attention and all_attention:
            result['attention_weights'] = all_attention

        return result

    def predict_with_uncertainty(
        self,
        data: Data,
        n_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MC Dropout for uncertainty estimation.

        Args:
            data: PyG Data object
            n_samples: Number of forward passes

        Returns:
            mean_pred: Mean prediction
            std_pred: Standard deviation (uncertainty)
        """
        self.train()  # Enable dropout

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                result = self.forward(data)
                predictions.append(result['graph_pred'])

        predictions = torch.stack(predictions, dim=0)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)

        self.eval()
        return mean_pred, std_pred

    def get_attention_maps(self, data: Data) -> Dict[str, torch.Tensor]:
        """
        Get attention maps for interpretability.

        Returns attention weights from all layers.
        """
        result = self.forward(data, return_attention=True)

        if 'attention_weights' in result:
            return {
                f'layer_{i}': attn
                for i, attn in enumerate(result['attention_weights'])
            }
        return {}

    def get_node_embeddings(self, data: Data) -> torch.Tensor:
        """
        Get intermediate node embeddings.
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        timestamps = data.timestamps if hasattr(data, 'timestamps') else None

        num_nodes = x.size(0)

        h = self.input_proj(x)
        node_encoding, attention_bias = self.encodings(
            edge_index, edge_attr, num_nodes, timestamps
        )
        h = h + node_encoding
        h, _, _ = self.encoder(h, attention_bias, return_attention=False)

        return h


class GraphormerGraphBuilder:
    """
    Graph builder specifically for Graphormer.

    Similar to GNN pipeline's GraphBuilder but with additional fields
    for Graphormer's encodings.
    
    Features:
    - kNN edges based on embedding similarity
    - Temporal edges connecting videos chronologically (when timestamps provided)
    """

    def __init__(self, k_neighbors: int = 5):
        self.k_neighbors = k_neighbors

    def build_graph(
        self,
        node_features: torch.Tensor,
        embeddings: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Data:
        """
        Build graph for Graphormer with kNN and temporal edges.

        Args:
            node_features: (N, D) node features
            embeddings: (N, E) embeddings for kNN
            timestamps: (N,) optional timestamps for temporal edge construction
            labels: (N,) optional labels

        Returns:
            PyG Data object with:
            - x: node features
            - edge_index: combined kNN and temporal edges
            - edge_attr: [weight, is_knn, is_temporal]
            - timestamps: if provided
        """
        import numpy as np

        N = node_features.size(0)

        # Compute kNN edges
        knn_edge_index, knn_edge_weights = self._compute_knn_edges(embeddings.numpy())
        
        # Compute temporal edges if timestamps provided
        if timestamps is not None and len(timestamps) > 1:
            temp_edge_index, temp_edge_weights = self._compute_temporal_edges(timestamps.numpy())
        else:
            temp_edge_index = np.array([[], []], dtype=np.int64)
            temp_edge_weights = np.array([], dtype=np.float32)
        
        # Combine edges
        if temp_edge_index.shape[1] > 0:
            edge_index = np.concatenate([knn_edge_index, temp_edge_index], axis=1)
            
            # Create edge attributes [weight, is_knn, is_temporal]
            num_knn = knn_edge_index.shape[1]
            num_temp = temp_edge_index.shape[1]
            
            edge_attr = torch.zeros(num_knn + num_temp, 3)
            # kNN edges
            edge_attr[:num_knn, 0] = torch.tensor(knn_edge_weights)  # Similarity weight
            edge_attr[:num_knn, 1] = 1.0  # is_knn = True
            edge_attr[:num_knn, 2] = 0.0  # is_temporal = False
            # Temporal edges
            edge_attr[num_knn:, 0] = torch.tensor(temp_edge_weights)  # Temporal weight
            edge_attr[num_knn:, 1] = 0.0  # is_knn = False
            edge_attr[num_knn:, 2] = 1.0  # is_temporal = True
        else:
            edge_index = knn_edge_index
            num_edges = edge_index.shape[1]
            edge_attr = torch.zeros(num_edges, 3)
            edge_attr[:, 0] = torch.tensor(knn_edge_weights)
            edge_attr[:, 1] = 1.0  # All are kNN edges

        # Create Data object
        data = Data(
            x=node_features,
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=edge_attr
        )

        if timestamps is not None:
            data.timestamps = timestamps

        if labels is not None:
            data.y = labels

        return data

    def _compute_knn_edges(self, embeddings: 'np.ndarray'):
        """Compute kNN edges from embeddings"""
        import numpy as np

        N = len(embeddings)
        k = min(self.k_neighbors, N - 1)

        if k <= 0:
            return np.array([[], []], dtype=np.int64), np.array([])

        # Normalize
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        similarity = embeddings_norm @ embeddings_norm.T

        edges_src = []
        edges_dst = []
        edge_weights = []

        for i in range(N):
            sim_i = similarity[i].copy()
            sim_i[i] = -np.inf

            top_k_idx = np.argsort(sim_i)[-k:]

            for j in top_k_idx:
                if sim_i[j] > -np.inf:
                    edges_src.append(i)
                    edges_dst.append(j)
                    edge_weights.append(max(0, sim_i[j]))

        edge_index = np.array([edges_src, edges_dst], dtype=np.int64)
        edge_weights = np.array(edge_weights, dtype=np.float32)

        return edge_index, edge_weights
    
    def _compute_temporal_edges(self, timestamps: 'np.ndarray'):
        """
        Compute temporal edges connecting videos chronologically.
        
        Creates bidirectional edges between consecutive videos (sorted by time).
        Edge weight is based on time proximity (closer in time = higher weight).
        
        Args:
            timestamps: (N,) array of timestamps
            
        Returns:
            edge_index: (2, E) temporal edge indices
            edge_weights: (E,) temporal proximity weights
        """
        import numpy as np
        
        N = len(timestamps)
        if N < 2:
            return np.array([[], []], dtype=np.int64), np.array([], dtype=np.float32)
        
        # Sort indices by timestamp
        sorted_indices = np.argsort(timestamps)
        
        edges_src = []
        edges_dst = []
        edge_weights = []
        
        # Connect consecutive videos in time
        for i in range(len(sorted_indices) - 1):
            src_idx = sorted_indices[i]
            dst_idx = sorted_indices[i + 1]
            
            # Compute time difference (in seconds)
            time_diff = abs(timestamps[dst_idx] - timestamps[src_idx])
            
            # Weight: higher for closer videos, using exponential decay
            # Weight = exp(-time_diff / tau) where tau = 1 day in seconds
            tau = 86400.0  # 1 day
            weight = np.exp(-time_diff / tau)
            
            # Bidirectional edges
            edges_src.extend([src_idx, dst_idx])
            edges_dst.extend([dst_idx, src_idx])
            edge_weights.extend([weight, weight])
        
        edge_index = np.array([edges_src, edges_dst], dtype=np.int64)
        edge_weights = np.array(edge_weights, dtype=np.float32)
        
        return edge_index, edge_weights
