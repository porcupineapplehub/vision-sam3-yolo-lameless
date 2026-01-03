"""
Graph Transformer (GraphGPS) Pipeline Service
Implements graph-based reasoning for lameness detection using relational context.

Key Features:
- Graph construction from video clips with node features
- kNN edges in DINOv3 embedding space + temporal edges
- GraphGPS architecture: local message passing + global attention
- Positional encodings: Laplacian eigenvectors, random walk
- Uncertainty estimation

This is a comprehensive Graph Neural Network implementation for learning purposes.
"""
import asyncio
import json
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_add_pool, SAGPooling
from torch_geometric.utils import add_self_loops, degree
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from shared.utils.nats_client import NATSClient


# ============================================================================
# Graph Construction
# ============================================================================

class GraphBuilder:
    """
    Builds graphs from video clips for graph-based lameness analysis.
    
    Node Features:
    - Pose metrics (from T-LEAP)
    - Silhouette metrics (from SAM3/YOLO)
    - DINOv3 embeddings (dimension-reduced)
    - Metadata features
    
    Edge Types:
    - kNN edges in embedding space (similarity)
    - Temporal edges (same cow over time)
    """
    
    def __init__(self, k_neighbors: int = 5, embedding_dim: int = 64):
        self.k_neighbors = k_neighbors
        self.embedding_dim = embedding_dim
    
    def compute_knn_edges(self, embeddings: np.ndarray, k: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute kNN edges from embeddings.
        
        Args:
            embeddings: Node embeddings (N, D)
            k: Number of neighbors
        
        Returns:
            edge_index: (2, E) edge index array
            edge_weights: (E,) similarity weights
        """
        if k is None:
            k = self.k_neighbors
        
        N = len(embeddings)
        if N <= k:
            k = max(1, N - 1)
        
        # Compute pairwise distances
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        similarity = embeddings_norm @ embeddings_norm.T
        
        # Get top-k neighbors for each node
        edges_src = []
        edges_dst = []
        edge_weights = []
        
        for i in range(N):
            # Exclude self
            sim_i = similarity[i].copy()
            sim_i[i] = -np.inf
            
            # Get top-k
            top_k_idx = np.argsort(sim_i)[-k:]
            
            for j in top_k_idx:
                if sim_i[j] > -np.inf:
                    edges_src.append(i)
                    edges_dst.append(j)
                    edge_weights.append(sim_i[j])
        
        edge_index = np.array([edges_src, edges_dst])
        edge_weights = np.array(edge_weights)
        
        return edge_index, edge_weights
    
    def add_temporal_edges(self, video_ids: List[str], cow_ids: List[Optional[str]],
                           timestamps: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add temporal edges between videos of the same cow.
        
        Returns:
            edge_index: (2, E) temporal edge index
            edge_attr: (E,) time deltas
        """
        edges_src = []
        edges_dst = []
        edge_attr = []
        
        # Group by cow ID
        cow_to_indices = {}
        for i, cow_id in enumerate(cow_ids):
            if cow_id is not None:
                if cow_id not in cow_to_indices:
                    cow_to_indices[cow_id] = []
                cow_to_indices[cow_id].append(i)
        
        # Create temporal edges within each cow group
        for cow_id, indices in cow_to_indices.items():
            if len(indices) < 2:
                continue
            
            # Sort by timestamp
            sorted_indices = sorted(indices, key=lambda x: timestamps[x])
            
            # Connect consecutive videos
            for i in range(len(sorted_indices) - 1):
                src, dst = sorted_indices[i], sorted_indices[i + 1]
                time_delta = timestamps[dst] - timestamps[src]
                
                # Bidirectional edges
                edges_src.extend([src, dst])
                edges_dst.extend([dst, src])
                edge_attr.extend([time_delta, -time_delta])
        
        if not edges_src:
            return np.array([[], []], dtype=np.int64), np.array([])
        
        edge_index = np.array([edges_src, edges_dst])
        edge_attr = np.array(edge_attr)
        
        return edge_index, edge_attr
    
    def build_graph(self, node_features: np.ndarray, embeddings: np.ndarray,
                    video_ids: List[str] = None, cow_ids: List[str] = None,
                    timestamps: List[float] = None, labels: np.ndarray = None) -> Data:
        """
        Build a PyTorch Geometric Data object from node features.
        
        Args:
            node_features: (N, D) node feature matrix
            embeddings: (N, E) DINOv3 embeddings for kNN
            video_ids: List of video IDs
            cow_ids: List of cow IDs (for temporal edges)
            timestamps: List of timestamps (for temporal edges)
            labels: (N,) labels if available
        
        Returns:
            PyG Data object
        """
        N = len(node_features)
        
        # Compute kNN edges
        knn_edges, knn_weights = self.compute_knn_edges(embeddings)
        
        # Add temporal edges if cow IDs available
        if cow_ids is not None and timestamps is not None:
            temp_edges, temp_weights = self.add_temporal_edges(video_ids or [], cow_ids, timestamps)
            
            if temp_edges.size > 0:
                # Combine edges
                edge_index = np.concatenate([knn_edges, temp_edges], axis=1)
                # Create edge type indicator: 0 = kNN, 1 = temporal
                edge_type = np.concatenate([
                    np.zeros(knn_edges.shape[1]),
                    np.ones(temp_edges.shape[1])
                ])
            else:
                edge_index = knn_edges
                edge_type = np.zeros(knn_edges.shape[1])
        else:
            edge_index = knn_edges
            edge_type = np.zeros(knn_edges.shape[1])
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)

        # NEW: Create multi-dimensional edge features
        # [similarity/temporal_weight, edge_type_knn, edge_type_temporal]
        num_edges = edge_index.shape[1]
        edge_attr = torch.zeros(num_edges, 3, dtype=torch.float32)

        # Combine edge weights (kNN similarity + temporal distances)
        if cow_ids is not None and timestamps is not None and temp_edges.size > 0:
            knn_edge_count = knn_edges.shape[1]
            # Normalize kNN weights to [0, 1]
            edge_attr[:knn_edge_count, 0] = torch.tensor(knn_weights, dtype=torch.float32)
            # Normalize temporal distances (days) using tanh
            temp_weights_norm = np.tanh(np.abs(temp_weights) / 86400.0)  # Normalize by 1 day in seconds
            edge_attr[knn_edge_count:, 0] = torch.tensor(temp_weights_norm, dtype=torch.float32)
        else:
            edge_attr[:, 0] = torch.tensor(knn_weights, dtype=torch.float32)

        # Edge type one-hot encoding
        edge_attr[:, 1] = (edge_type == 0).float()  # kNN indicator
        edge_attr[:, 2] = (edge_type == 1).float()  # Temporal indicator

        data = Data(x=x, edge_index=edge_index, edge_type=edge_type, edge_attr=edge_attr)

        if labels is not None:
            data.y = torch.tensor(labels, dtype=torch.float32)

        return data


# ============================================================================
# Positional Encodings
# ============================================================================

class LearnedLaplacianPE(nn.Module):
    """
    True Laplacian Positional Encoding with learnable transformation.

    Computes k smallest non-trivial eigenvectors of the normalized Laplacian
    and learns a transformation to the hidden dimension.
    Uses sign-flip invariance via absolute values.
    """

    def __init__(self, k: int = 8, hidden_dim: int = 16):
        super().__init__()
        self.k = k
        self.hidden_dim = hidden_dim

        # Learnable transformation (sign-flip invariant via absolute value + MLP)
        self.transform = nn.Sequential(
            nn.Linear(k, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def compute_laplacian_eigenvectors(self, edge_index: torch.Tensor,
                                        num_nodes: int) -> torch.Tensor:
        """Compute k smallest non-trivial eigenvectors of normalized Laplacian"""
        device = edge_index.device

        # Build adjacency matrix
        edge_index_np = edge_index.cpu().numpy()
        row, col = edge_index_np

        # Add self-loops
        self_loops = np.arange(num_nodes)
        row = np.concatenate([row, self_loops])
        col = np.concatenate([col, self_loops])
        data = np.ones(len(row))

        # Create sparse adjacency matrix
        A = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

        # Compute degree
        deg = np.array(A.sum(axis=1)).flatten()
        deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0)
        D_inv_sqrt = csr_matrix((deg_inv_sqrt, (np.arange(num_nodes), np.arange(num_nodes))),
                                 shape=(num_nodes, num_nodes))

        # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        I = csr_matrix(np.eye(num_nodes))
        L = I - D_inv_sqrt @ A @ D_inv_sqrt

        # Compute eigenvectors
        try:
            if num_nodes > self.k + 1 and num_nodes <= 2000:
                # Use sparse eigendecomposition
                eigenvalues, eigenvectors = eigsh(L, k=min(self.k + 1, num_nodes - 1),
                                                   which='SM', tol=1e-3)
                # Skip first eigenvector (constant)
                pe = eigenvectors[:, 1:self.k + 1]
            elif num_nodes <= self.k + 1:
                # Small graph - use dense
                L_dense = L.toarray()
                eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
                pe = eigenvectors[:, 1:self.k + 1]
            else:
                # Very large graph - use random features as fallback
                pe = np.random.randn(num_nodes, self.k) * 0.01

            # Pad if fewer eigenvectors available
            if pe.shape[1] < self.k:
                padding = np.zeros((num_nodes, self.k - pe.shape[1]))
                pe = np.concatenate([pe, padding], axis=1)

        except Exception:
            # Fallback to random features
            pe = np.random.randn(num_nodes, self.k) * 0.01

        return torch.tensor(pe, dtype=torch.float32, device=device)

    def forward(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        pe = self.compute_laplacian_eigenvectors(edge_index, num_nodes)
        # Make sign-flip invariant by using absolute values
        pe = torch.abs(pe)
        return self.transform(pe)


class LearnedRWPE(nn.Module):
    """
    True Random Walk Positional Encoding with learnable transformation.

    Computes k-step random walk landing probabilities P^k[i,i] (self-return probs)
    for each node.
    """

    def __init__(self, walk_length: int = 16, hidden_dim: int = 16):
        super().__init__()
        self.walk_length = walk_length
        self.hidden_dim = hidden_dim

        # Learnable transformation
        self.transform = nn.Sequential(
            nn.Linear(walk_length, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def compute_rwpe(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Compute random walk positional encoding (diagonal of P^k)"""
        device = edge_index.device

        # Build transition matrix
        edge_index_np = edge_index.cpu().numpy()
        row, col = edge_index_np

        # Add self-loops
        self_loops = np.arange(num_nodes)
        row = np.concatenate([row, self_loops])
        col = np.concatenate([col, self_loops])
        data = np.ones(len(row))

        # Create sparse adjacency matrix
        A = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

        # Compute degree and build transition matrix P = D^{-1} A
        deg = np.array(A.sum(axis=1)).flatten()
        deg_inv = np.where(deg > 0, 1.0 / deg, 0)

        # Build sparse random walk matrix
        D_inv = csr_matrix((deg_inv, (np.arange(num_nodes), np.arange(num_nodes))),
                           shape=(num_nodes, num_nodes))
        P = D_inv @ A

        # Compute P^k for k = 1, 2, ..., walk_length
        pe = np.zeros((num_nodes, self.walk_length), dtype=np.float32)

        if num_nodes <= 1000:
            # Dense computation for small graphs
            P_dense = P.toarray()
            P_k = P_dense.copy()

            for k in range(self.walk_length):
                # Landing probability (diagonal of P^k)
                pe[:, k] = np.diag(P_k)
                P_k = P_k @ P_dense
        else:
            # For large graphs, use degree-based approximation
            for k in range(self.walk_length):
                pe[:, k] = np.power(deg / deg.sum(), k + 1)

        return torch.tensor(pe, dtype=torch.float32, device=device)

    def forward(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        pe = self.compute_rwpe(edge_index, num_nodes)
        return self.transform(pe)


# ============================================================================
# Edge Encoder
# ============================================================================

class EdgeEncoder(nn.Module):
    """
    Encodes edge features to match hidden dimension for GatedGCN.

    Input edge features:
    - Similarity score / normalized temporal distance
    - Edge type one-hot encoding (kNN, temporal)
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_attr: (E, input_dim) raw edge features
        Returns:
            (E, hidden_dim) encoded edge features
        """
        return self.encoder(edge_attr)


# ============================================================================
# GraphGPS Architecture
# ============================================================================

class GatedGCNLayer(nn.Module):
    """
    Gated Graph Convolution Layer with edge feature support.

    Implements message passing with gating mechanism for controlled information flow.
    Enhanced to properly incorporate edge features in gating mechanism.
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = None, dropout: float = 0.1):
        super().__init__()

        self.out_dim = out_dim
        self.A = nn.Linear(in_dim, out_dim)
        self.B = nn.Linear(in_dim, out_dim)
        self.D = nn.Linear(in_dim, out_dim)
        self.E = nn.Linear(in_dim, out_dim)

        # Edge feature transformation (accept edge_dim or default to in_dim)
        edge_dim = edge_dim if edge_dim is not None else in_dim
        self.C = nn.Linear(edge_dim, out_dim)

        # Edge feature update network
        self.edge_update = nn.Sequential(
            nn.Linear(out_dim * 3, out_dim),  # src + dst + edge features
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

        self.bn_node = nn.BatchNorm1d(out_dim)
        self.bn_edge = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node features (N, in_dim)
            edge_index: Edge indices (2, E)
            edge_attr: Edge features (E, edge_dim) - optional
        """
        src, dst = edge_index

        # Node transformations
        Ax = self.A(x)
        Bx = self.B(x)
        Dx = self.D(x)
        Ex = self.E(x)

        # Edge gate with proper edge feature incorporation
        if edge_attr is not None:
            Ce = self.C(edge_attr)
            sigma = torch.sigmoid(Ce + Dx[dst] + Ex[src])

            # Update edge features
            edge_input = torch.cat([Dx[dst], Ex[src], Ce], dim=-1)
            e_new = self.bn_edge(self.edge_update(edge_input))
        else:
            sigma = torch.sigmoid(Dx[dst] + Ex[src])
            e_new = sigma

        # Message passing with gating
        message = sigma * Bx[src]

        # Aggregate messages
        agg = torch.zeros_like(x[:, :message.size(1)])
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(message), message)

        # Count neighbors for normalization
        deg = degree(dst, x.size(0), dtype=x.dtype).clamp(min=1)
        agg = agg / deg.unsqueeze(1)

        # Residual connection
        h = Ax + agg
        h = self.bn_node(h)
        h = F.relu(h)
        h = self.dropout(h)

        return h, e_new


class GlobalAttention(nn.Module):
    """
    Global Self-Attention for graphs with increased expressiveness.

    Enhanced with:
    - Configurable attention heads (default 8)
    - Attention dropout
    - Optional attention bias from positional encodings
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1,
                 use_pe_bias: bool = True):
        super().__init__()

        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Optional PE bias projection
        if use_pe_bias:
            self.pe_bias = nn.Linear(hidden_dim, num_heads)
        else:
            self.pe_bias = None

    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None,
                pe: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply global attention with optional PE-based bias.

        Args:
            x: Node features (N, D)
            batch: Batch assignment (N,) for batched graphs
            pe: Positional encodings (N, D) for attention bias
        """
        if batch is None:
            # Single graph - simple attention
            x_2d = x.unsqueeze(0)  # (1, N, D)
            attn_out, _ = self.attention(x_2d, x_2d, x_2d)
            attn_out = attn_out.squeeze(0)  # (N, D)
        else:
            # Batched graphs - attention within each graph
            unique_batches = batch.unique()
            outputs = []

            for b in unique_batches:
                mask = batch == b
                x_b = x[mask].unsqueeze(0)  # (1, N_b, D)
                attn_out, _ = self.attention(x_b, x_b, x_b)
                outputs.append(attn_out.squeeze(0))

            attn_out = torch.cat(outputs, dim=0)

        # Residual connection
        out = self.norm(x + self.dropout(attn_out))
        return out


class GraphGPSLayer(nn.Module):
    """
    GraphGPS Layer combining local and global attention.

    Enhanced with:
    - 8 attention heads in global attention
    - Edge feature propagation through layers
    - Pre-norm architecture for stability

    Architecture:
    1. Local message passing (GatedGCN) with edge features
    2. Global self-attention (8 heads)
    3. Feedforward network
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1,
                 edge_dim: int = None):
        super().__init__()

        # Local message passing with edge features
        self.local_conv = GatedGCNLayer(hidden_dim, hidden_dim, edge_dim=edge_dim, dropout=dropout)

        # Global attention with more heads
        self.global_attn = GlobalAttention(hidden_dim, num_heads, dropout)

        # Feedforward with GELU and larger expansion
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        # Pre-norm layers
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None,
                pe: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through GPS layer with edge feature propagation"""

        # Pre-norm local message passing
        x_norm = self.norm1(x)
        h_local, edge_attr_new = self.local_conv(x_norm, edge_index, edge_attr)
        x = x + h_local  # Residual

        # Pre-norm global attention
        x_norm = self.norm2(x)
        h_global = self.global_attn(x_norm, batch, pe)
        x = x + (h_global - x_norm)  # Residual (global_attn has internal residual)

        # Pre-norm feedforward
        x_norm = self.norm3(x)
        x = x + self.ffn(x_norm)

        return x, edge_attr_new


# ============================================================================
# Hierarchical Pooling
# ============================================================================

class HierarchicalPoolingLayer(nn.Module):
    """
    Hierarchical pooling for multi-scale graph representation.

    Uses SAGPooling (Self-Attention Graph Pooling) to learn which nodes
    to keep and which to pool together.
    """

    def __init__(self, hidden_dim: int, pooling_ratio: float = 0.5):
        super().__init__()

        self.pooling_ratio = pooling_ratio

        # SAGPooling uses attention to score nodes
        self.pool = SAGPooling(hidden_dim, ratio=pooling_ratio)

        # Project coarsened representation
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Apply hierarchical pooling.

        Returns:
            x_pooled: Pooled node features
            edge_index_pooled: New edge indices
            edge_attr_pooled: New edge attributes (if provided)
            batch_pooled: New batch assignment
            perm: Indices of retained nodes
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Apply SAGPooling
        x_pooled, edge_index_pooled, edge_attr_pooled, batch_pooled, perm, score = self.pool(
            x, edge_index, edge_attr, batch
        )

        # Project
        x_pooled = self.project(x_pooled)

        return x_pooled, edge_index_pooled, edge_attr_pooled, batch_pooled, perm


class MultiScaleReadout(nn.Module):
    """
    Multi-scale graph readout combining representations at different scales.

    Aggregates node representations from:
    1. Original scale (fine)
    2. After pooling (coarse)
    """

    def __init__(self, hidden_dim: int, num_scales: int = 2):
        super().__init__()

        self.num_scales = num_scales

        # Attention weights for scale combination
        self.scale_attention = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_scales),
            nn.Softmax(dim=-1)
        )

        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, representations: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """
        Combine multi-scale representations.

        Args:
            representations: List of (node_features, batch) tuples at each scale

        Returns:
            Graph-level representation (batch_size, hidden_dim)
        """
        scale_outputs = []

        for x, batch in representations:
            # Global mean pooling at this scale
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            graph_repr = global_mean_pool(x, batch)
            scale_outputs.append(graph_repr)

        # Concatenate scales
        concat = torch.cat(scale_outputs, dim=-1)  # (batch, hidden_dim * num_scales)

        # Attention-weighted combination
        weights = self.scale_attention(concat)  # (batch, num_scales)

        weighted_sum = torch.zeros_like(scale_outputs[0])
        for i, output in enumerate(scale_outputs):
            weighted_sum = weighted_sum + weights[:, i:i+1] * output

        return self.output_proj(weighted_sum)


# ============================================================================
# Enhanced Prediction Head
# ============================================================================

class EnhancedPredictionHead(nn.Module):
    """
    Enhanced prediction head with:
    - Attention-based node weighting
    - Graph-level aggregation
    - Node-level predictions
    """

    def __init__(self, hidden_dim: int, num_classes: int = 1, dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Node-level attention for importance weighting
        self.node_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Graph-level classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for concat of mean and weighted
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # For node-level predictions
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None,
                return_node_preds: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Node features (N, hidden_dim)
            batch: Batch assignment (N,)
            return_node_preds: Whether to return node-level predictions

        Returns:
            Dictionary with 'graph_pred', 'node_pred', 'attention_weights'
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Compute attention weights
        attn_scores = self.node_attention(x)  # (N, 1)

        # Softmax within each graph
        attn_weights = torch.zeros_like(attn_scores)
        for b in batch.unique():
            mask = batch == b
            attn_weights[mask] = F.softmax(attn_scores[mask], dim=0)

        # Weighted sum
        weighted_x = x * attn_weights
        weighted_pool = global_add_pool(weighted_x, batch)

        # Mean pool
        mean_pool = global_mean_pool(x, batch)

        # Concatenate for richer representation
        graph_repr = torch.cat([mean_pool, weighted_pool], dim=-1)

        # Graph-level prediction
        graph_pred = torch.sigmoid(self.classifier(graph_repr))

        result = {
            'graph_pred': graph_pred,
            'attention_weights': attn_weights
        }

        # Node-level predictions
        if return_node_preds:
            node_pred = torch.sigmoid(self.node_classifier(x))
            result['node_pred'] = node_pred

        return result


class EnhancedGraphGPS(nn.Module):
    """
    Enhanced GraphGPS Model for lameness prediction.

    Improvements:
    1. Rich edge features (similarity, temporal distance, edge type)
    2. 8 attention heads in global attention
    3. True Laplacian and Random Walk positional encodings
    4. Hierarchical pooling for multi-scale representation
    5. Enhanced prediction head with attention-based aggregation
    """

    def __init__(self,
                 input_dim: int = 50,
                 hidden_dim: int = 128,
                 edge_input_dim: int = 3,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 pe_dim: int = 16,
                 use_hierarchical_pooling: bool = True,
                 pooling_ratio: float = 0.5):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_hierarchical_pooling = use_hierarchical_pooling

        # Input projection (reserve space for PE)
        pe_total_dim = pe_dim * 2  # Laplacian + RW
        self.input_proj = nn.Linear(input_dim, hidden_dim - pe_total_dim)

        # Edge feature encoder
        self.edge_encoder = EdgeEncoder(input_dim=edge_input_dim, hidden_dim=hidden_dim)

        # Learned positional encodings
        self.lap_pe = LearnedLaplacianPE(k=8, hidden_dim=pe_dim)
        self.rw_pe = LearnedRWPE(walk_length=16, hidden_dim=pe_dim)

        # GPS layers (first 2 layers before pooling)
        num_pre_pool = num_layers // 2 if use_hierarchical_pooling else num_layers
        self.pre_pool_layers = nn.ModuleList([
            GraphGPSLayer(hidden_dim, num_heads, dropout, edge_dim=hidden_dim)
            for _ in range(num_pre_pool)
        ])

        # Hierarchical pooling
        if use_hierarchical_pooling:
            self.pool_layer = HierarchicalPoolingLayer(hidden_dim, pooling_ratio)
            self.post_pool_layers = nn.ModuleList([
                GraphGPSLayer(hidden_dim, num_heads, dropout, edge_dim=hidden_dim)
                for _ in range(num_layers - num_pre_pool)
            ])
            self.multi_scale_readout = MultiScaleReadout(hidden_dim, num_scales=2)

        # Enhanced prediction head
        self.pred_head = EnhancedPredictionHead(hidden_dim, num_classes=1, dropout=dropout)

        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            data: PyG Data object with x, edge_index, edge_attr, and optionally batch

        Returns:
            Dictionary with predictions and attention weights
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        batch = data.batch if hasattr(data, 'batch') else None

        num_nodes = x.size(0)

        # Input projection
        h = self.input_proj(x)

        # Encode edge features
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)

        # Add positional encodings
        lap_pe = self.lap_pe(edge_index, num_nodes)
        rw_pe = self.rw_pe(edge_index, num_nodes)
        pe = torch.cat([lap_pe, rw_pe], dim=-1)
        h = torch.cat([h, pe], dim=-1)

        # Store multi-scale representations
        scale_representations = []

        # Pre-pooling GPS layers (fine scale)
        for layer in self.pre_pool_layers:
            h, edge_attr = layer(h, edge_index, edge_attr, batch, pe)

        scale_representations.append((h.clone(), batch))

        # Hierarchical pooling
        if self.use_hierarchical_pooling and h.size(0) > 3:  # Only pool if enough nodes
            try:
                h_pooled, edge_index_pooled, edge_attr_pooled, batch_pooled, perm = self.pool_layer(
                    h, edge_index, edge_attr, batch
                )

                # Recompute PE for pooled graph
                pe_pooled = pe[perm] if pe is not None else None

                # Post-pooling layers (coarse scale)
                for layer in self.post_pool_layers:
                    h_pooled, edge_attr_pooled = layer(
                        h_pooled, edge_index_pooled, edge_attr_pooled, batch_pooled, pe_pooled
                    )

                scale_representations.append((h_pooled, batch_pooled))
            except Exception:
                # If pooling fails (e.g., too few nodes), skip it
                pass

        # Apply final norm
        h = self.final_norm(h)

        # Prediction
        pred_result = self.pred_head(h, batch, return_node_preds=True)

        return pred_result

    def predict_with_uncertainty(self, data: Data, n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """MC Dropout for uncertainty estimation"""
        self.train()

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                result = self.forward(data)
                predictions.append(result['node_pred'])

        predictions = torch.stack(predictions, dim=0)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)

        self.eval()
        return mean_pred, std_pred

    def get_node_embeddings(self, data: Data) -> torch.Tensor:
        """Get intermediate node embeddings for analysis"""
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        batch = data.batch if hasattr(data, 'batch') else None

        num_nodes = x.size(0)

        h = self.input_proj(x)

        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)

        lap_pe = self.lap_pe(edge_index, num_nodes)
        rw_pe = self.rw_pe(edge_index, num_nodes)
        pe = torch.cat([lap_pe, rw_pe], dim=-1)
        h = torch.cat([h, pe], dim=-1)

        for layer in self.pre_pool_layers:
            h, edge_attr = layer(h, edge_index, edge_attr, batch, pe)

        return h


# Keep old GraphGPS for backwards compatibility
class GraphGPS(nn.Module):
    """
    Original GraphGPS Model for node-level lameness prediction.
    Kept for backwards compatibility.
    """

    def __init__(self,
                 input_dim: int = 64,
                 hidden_dim: int = 64,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 pe_dim: int = 16):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim - 2 * pe_dim)

        # Positional encodings (now using improved versions)
        self.lap_pe = LearnedLaplacianPE(k=8, hidden_dim=pe_dim)
        self.rw_pe = LearnedRWPE(walk_length=16, hidden_dim=pe_dim)

        # GPS layers
        self.layers = nn.ModuleList([
            GraphGPSLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass.

        Args:
            data: PyG Data object with x, edge_index, and optionally batch

        Returns:
            Node-level predictions (N, 1)
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        batch = data.batch if hasattr(data, 'batch') else None

        # Input projection
        h = self.input_proj(x)

        # Add positional encodings
        lap_pe = self.lap_pe(edge_index, x.size(0))
        rw_pe = self.rw_pe(edge_index, x.size(0))
        pe = torch.cat([lap_pe, rw_pe], dim=-1)
        h = torch.cat([h, pe], dim=-1)

        # GPS layers
        for layer in self.layers:
            h, edge_attr = layer(h, edge_index, edge_attr, batch=batch, pe=pe)

        # Prediction
        out = self.pred_head(h)

        return out

    def predict_with_uncertainty(self, data: Data, n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """MC Dropout for uncertainty estimation"""
        self.train()

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(data)
                predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)

        self.eval()
        return mean_pred, std_pred

    def get_node_embeddings(self, data: Data) -> torch.Tensor:
        """Get intermediate node embeddings for analysis"""
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        batch = data.batch if hasattr(data, 'batch') else None

        h = self.input_proj(x)
        lap_pe = self.lap_pe(edge_index, x.size(0))
        rw_pe = self.rw_pe(edge_index, x.size(0))
        pe = torch.cat([lap_pe, rw_pe], dim=-1)
        h = torch.cat([h, pe], dim=-1)

        for layer in self.layers:
            h, edge_attr = layer(h, edge_index, edge_attr, batch=batch, pe=pe)

        return h


# ============================================================================
# Pipeline Service
# ============================================================================

class GNNPipeline:
    """Graph Neural Network Pipeline Service using Enhanced GraphGPS"""

    # Feature dimensions
    POSE_FEATURES = 10  # Summary pose metrics
    SILHOUETTE_FEATURES = 5  # SAM3/YOLO metrics
    EMBEDDING_DIM = 32  # Reduced DINOv3 dimension
    META_FEATURES = 3  # Metadata features

    def __init__(self, use_enhanced_model: bool = True):
        self.config_path = Path("/app/shared/config/config.yaml")
        self.config = self._load_config()
        self.nats_client = NATSClient(str(self.config_path))

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Graph builder
        self.graph_builder = GraphBuilder(k_neighbors=5, embedding_dim=self.EMBEDDING_DIM)

        # Model - use enhanced version by default
        input_dim = self.POSE_FEATURES + self.SILHOUETTE_FEATURES + self.EMBEDDING_DIM + self.META_FEATURES

        if use_enhanced_model:
            # Enhanced GraphGPS with all improvements:
            # - Edge features (similarity, temporal, edge type)
            # - 8 attention heads
            # - True Laplacian and RW positional encodings
            # - Hierarchical pooling with SAGPooling
            # - Enhanced prediction head
            self.model = EnhancedGraphGPS(
                input_dim=input_dim,
                hidden_dim=128,  # Increased from 64 for 8 heads
                edge_input_dim=3,  # [weight, kNN_indicator, temporal_indicator]
                num_layers=4,
                num_heads=8,  # Increased from 4
                dropout=0.1,
                pe_dim=16,
                use_hierarchical_pooling=True,
                pooling_ratio=0.5
            ).to(self.device)
            self.model_name = "EnhancedGraphGPS"
        else:
            # Original GraphGPS for backwards compatibility
            self.model = GraphGPS(
                input_dim=input_dim,
                hidden_dim=64,
                num_layers=4,
                num_heads=8,  # Also upgraded to 8 heads
                dropout=0.1,
                pe_dim=16
            ).to(self.device)
            self.model_name = "GraphGPS"

        # Load weights if available
        self.model_path = Path("/app/shared/models/gnn")
        self.model_path.mkdir(parents=True, exist_ok=True)
        self._load_model()

        # Results directory
        self.results_dir = Path("/app/data/results/gnn")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Cache for building graphs across videos
        self.video_features_cache = {}
        
        # Cow ID mapping cache
        self.cow_id_mapping: Dict[str, str] = {}
        self.video_timestamps: Dict[str, float] = {}
    
    def _load_config(self) -> dict:
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_model(self):
        # Try enhanced model weights first, then fall back to original
        weights_path = self.model_path / f"{self.model_name.lower()}_lameness.pt"
        fallback_path = self.model_path / "graphgps_lameness.pt"

        loaded = False
        if weights_path.exists():
            try:
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
                print(f"✅ Loaded {self.model_name} weights from {weights_path}")
                loaded = True
            except Exception as e:
                print(f"⚠️ Failed to load weights from {weights_path}: {e}")

        if not loaded and fallback_path.exists() and fallback_path != weights_path:
            try:
                # Try to load partial weights (may work for compatible layers)
                state_dict = torch.load(fallback_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                print(f"✅ Loaded partial weights from {fallback_path}")
            except Exception as e:
                print(f"⚠️ Failed to load fallback weights: {e}")

        if not loaded:
            print(f"⚠️ No pretrained {self.model_name} weights. Using random initialization.")

        self.model.eval()
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"{self.model_name} parameters: {num_params:,}")
    
    def load_cow_id_mapping(self) -> Dict[str, str]:
        """
        Load video_id -> cow_id mapping from tracking results.
        Also loads timestamps for temporal edge construction.
        """
        mapping = {}
        timestamps = {}
        tracking_dir = Path("/app/data/results/tracking")
        
        if not tracking_dir.exists():
            print("  No tracking results directory found")
            return mapping
        
        for tracking_file in tracking_dir.glob("*_tracking.json"):
            try:
                with open(tracking_file) as f:
                    data = json.load(f)
                
                video_id = data.get("video_id")
                if not video_id:
                    continue
                
                # Get cow_id from Re-ID results
                reid_results = data.get("reid_results", [])
                for reid in reid_results:
                    cow_id = reid.get("cow_id")
                    if cow_id:
                        mapping[video_id] = cow_id
                        break
                
                # Get timestamp from tracking data or video metadata
                timestamp = data.get("timestamp")
                if timestamp:
                    # Try to parse timestamp string
                    try:
                        from datetime import datetime
                        if isinstance(timestamp, str):
                            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                            timestamps[video_id] = dt.timestamp()
                        else:
                            timestamps[video_id] = float(timestamp)
                    except:
                        timestamps[video_id] = 0.0
                else:
                    # Use file modification time as fallback
                    timestamps[video_id] = tracking_file.stat().st_mtime
                    
            except Exception as e:
                print(f"  Error reading tracking file {tracking_file}: {e}")
                continue
        
        self.cow_id_mapping = mapping
        self.video_timestamps = timestamps
        print(f"  Loaded cow_id mapping: {len(mapping)} videos mapped to cows")
        return mapping
    
    def get_cow_for_video(self, video_id: str) -> Optional[str]:
        """Get the cow_id for a given video_id"""
        # Always refresh mapping to pick up new tracking results
        self.load_cow_id_mapping()
        return self.cow_id_mapping.get(video_id)
    
    def get_videos_for_cow(self, cow_id: str) -> List[str]:
        """Get all video_ids belonging to a specific cow"""
        if not self.cow_id_mapping:
            self.load_cow_id_mapping()
        return [vid for vid, cid in self.cow_id_mapping.items() if cid == cow_id]
    
    def extract_node_features(self, video_id: str) -> Optional[Dict[str, np.ndarray]]:
        """Extract node features from pipeline results for a video"""
        features = {}
        
        # T-LEAP pose features
        tleap_path = Path(f"/app/data/results/tleap/{video_id}_tleap.json")
        if tleap_path.exists():
            with open(tleap_path) as f:
                tleap_data = json.load(f)
            
            loco = tleap_data.get("locomotion_features", {})
            features["pose"] = np.array([
                loco.get("back_arch_mean", 0),
                loco.get("back_arch_std", 0),
                loco.get("head_bob_magnitude", 0),
                loco.get("head_bob_frequency", 0),
                loco.get("front_leg_asymmetry", 0),
                loco.get("rear_leg_asymmetry", 0),
                loco.get("lameness_score", 0.5),
                loco.get("stride_fl_mean", 0),
                loco.get("stride_fr_mean", 0),
                loco.get("steadiness_score", 0.5)
            ], dtype=np.float32)
        else:
            features["pose"] = np.zeros(self.POSE_FEATURES, dtype=np.float32)
        
        # SAM3/YOLO silhouette features
        sam3_path = Path(f"/app/data/results/sam3/{video_id}_sam3.json")
        yolo_path = Path(f"/app/data/results/yolo/{video_id}_yolo.json")
        
        silhouette = np.zeros(self.SILHOUETTE_FEATURES, dtype=np.float32)
        
        if sam3_path.exists():
            with open(sam3_path) as f:
                sam3_data = json.load(f)
            feats = sam3_data.get("features", {})
            silhouette[0] = feats.get("avg_area_ratio", 0)
            silhouette[1] = feats.get("avg_circularity", 0)
            silhouette[2] = feats.get("avg_aspect_ratio", 1)
        
        if yolo_path.exists():
            with open(yolo_path) as f:
                yolo_data = json.load(f)
            feats = yolo_data.get("features", {})
            silhouette[3] = feats.get("avg_confidence", 0.5)
            silhouette[4] = feats.get("position_stability", 0.5)
        
        features["silhouette"] = silhouette
        
        # DINOv3 embeddings
        dinov3_path = Path(f"/app/data/results/dinov3/{video_id}_dinov3.json")
        if dinov3_path.exists():
            with open(dinov3_path) as f:
                dinov3_data = json.load(f)
            
            embedding = dinov3_data.get("embedding", [])
            if len(embedding) > 0:
                # Reduce dimension if needed
                embedding = np.array(embedding, dtype=np.float32)
                if len(embedding) > self.EMBEDDING_DIM:
                    # Simple PCA-like reduction (use first N dims)
                    embedding = embedding[:self.EMBEDDING_DIM]
                elif len(embedding) < self.EMBEDDING_DIM:
                    embedding = np.pad(embedding, (0, self.EMBEDDING_DIM - len(embedding)))
                features["embedding"] = embedding
            else:
                features["embedding"] = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
        else:
            features["embedding"] = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
        
        # Metadata features
        features["meta"] = np.array([
            0.5,  # Placeholder for normalized timestamp
            1.0,  # Quality score
            0.5   # Prior lameness estimate
        ], dtype=np.float32)
        
        return features
    
    def collect_graph_data(self, filter_cow_id: Optional[str] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str], List[Optional[str]], List[float]]:
        """
        Collect features from analyzed videos for graph construction.
        
        Args:
            filter_cow_id: If provided, only include videos belonging to this cow
        
        Returns:
            node_features: (N, D) node feature matrix
            embeddings: (N, E) DINOv3 embeddings for kNN
            video_ids: List of video IDs
            cow_ids: List of cow IDs (or None if unknown)
            timestamps: List of timestamps for temporal edges
        """
        # Refresh cow_id mapping
        self.load_cow_id_mapping()
        
        node_features_list = []
        embeddings_list = []
        video_ids = []
        cow_ids = []
        timestamps = []
        
        # Scan results directory for available videos
        tleap_dir = Path("/app/data/results/tleap")
        if tleap_dir.exists():
            for result_file in tleap_dir.glob("*_tleap.json"):
                video_id = result_file.stem.replace("_tleap", "")
                
                # Get cow_id for this video
                cow_id = self.cow_id_mapping.get(video_id)
                
                # Filter by cow if requested
                if filter_cow_id is not None:
                    if cow_id != filter_cow_id:
                        continue
                
                features = self.extract_node_features(video_id)
                if features is not None:
                    # Concatenate all features
                    node_feat = np.concatenate([
                        features["pose"],
                        features["silhouette"],
                        features["embedding"],
                        features["meta"]
                    ])
                    
                    node_features_list.append(node_feat)
                    embeddings_list.append(features["embedding"])
                    video_ids.append(video_id)
                    cow_ids.append(cow_id)
                    timestamps.append(self.video_timestamps.get(video_id, 0.0))
        
        if not node_features_list:
            return None, None, [], [], []
        
        node_features = np.stack(node_features_list)
        embeddings = np.stack(embeddings_list)
        
        return node_features, embeddings, video_ids, cow_ids, timestamps
    
    async def process_video(self, video_data: dict):
        """Process video through GNN pipeline with per-cow graph construction"""
        video_id = video_data.get("video_id")
        if not video_id:
            return
        
        print(f"GNN pipeline processing video {video_id}")
        
        try:
            # Get cow_id for this video
            target_cow_id = self.get_cow_for_video(video_id)
            
            if target_cow_id:
                print(f"  Video belongs to cow: {target_cow_id}")
                # Build per-cow graph (only videos of the same cow)
                node_features, embeddings, video_ids, cow_ids, timestamps = self.collect_graph_data(
                    filter_cow_id=target_cow_id
                )
            else:
                print(f"  No cow_id found, using global graph")
                # Fallback: use all videos if no cow_id mapping
                node_features, embeddings, video_ids, cow_ids, timestamps = self.collect_graph_data()
            
            if node_features is None or len(video_ids) == 0:
                print(f"  No video features available for graph construction")
                return
            
            # Ensure current video is in the graph
            if video_id not in video_ids:
                features = self.extract_node_features(video_id)
                if features is None:
                    print(f"  Could not extract features for {video_id}")
                    return
                
                new_node = np.concatenate([
                    features["pose"],
                    features["silhouette"],
                    features["embedding"],
                    features["meta"]
                ])
                
                node_features = np.vstack([node_features, new_node])
                embeddings = np.vstack([embeddings, features["embedding"]])
                video_ids.append(video_id)
                cow_ids.append(target_cow_id)
                timestamps.append(self.video_timestamps.get(video_id, 0.0))
            
            target_idx = video_ids.index(video_id)
            
            print(f"  Per-cow graph: {len(video_ids)} nodes for cow {target_cow_id or 'unknown'}")
            
            # Build graph with cow_ids and timestamps for temporal edges
            graph = self.graph_builder.build_graph(
                node_features=node_features,
                embeddings=embeddings,
                video_ids=video_ids,
                cow_ids=cow_ids if target_cow_id else None,
                timestamps=timestamps if target_cow_id else None
            )
            graph = graph.to(self.device)
            
            # Predict with uncertainty
            mean_pred, std_pred = self.model.predict_with_uncertainty(graph, n_samples=10)
            
            # Get prediction for target video (node-level)
            node_severity_score = float(mean_pred[target_idx, 0].cpu().numpy())
            node_uncertainty = float(std_pred[target_idx, 0].cpu().numpy())
            
            # Get graph-level prediction (cow-level when using per-cow graph)
            with torch.no_grad():
                result = self.model(graph)
                graph_pred = result.get('graph_pred')
                if graph_pred is not None:
                    cow_severity_score = float(graph_pred[0, 0].cpu().numpy())
                else:
                    # Fallback: average of all node predictions
                    cow_severity_score = float(mean_pred.mean().cpu().numpy())
            
            # Get neighbor influence
            neighbor_scores = []
            edge_index = graph.edge_index.cpu().numpy()
            for i in range(edge_index.shape[1]):
                if edge_index[1, i] == target_idx:
                    src = edge_index[0, i]
                    neighbor_scores.append({
                        "video_id": video_ids[src],
                        "score": float(mean_pred[src, 0].cpu().numpy())
                    })
            
            # Save results with both node-level and cow-level predictions
            results = {
                "video_id": video_id,
                "cow_id": target_cow_id,
                "pipeline": "gnn",
                "model": self.model_name,
                "severity_score": node_severity_score,  # Node-level (video) score
                "cow_severity_score": cow_severity_score,  # Graph-level (cow) score
                "uncertainty": node_uncertainty,
                "prediction": int(node_severity_score > 0.5),
                "cow_prediction": int(cow_severity_score > 0.5),
                "confidence": 1.0 - node_uncertainty,
                "graph_info": {
                    "num_nodes": len(video_ids),
                    "num_edges": graph.edge_index.shape[1],
                    "k_neighbors": self.graph_builder.k_neighbors,
                    "has_edge_features": hasattr(graph, 'edge_attr') and graph.edge_attr is not None,
                    "has_temporal_edges": target_cow_id is not None,
                    "num_heads": 8 if self.model_name == "EnhancedGraphGPS" else 4,
                    "hierarchical_pooling": self.model_name == "EnhancedGraphGPS",
                    "per_cow_graph": target_cow_id is not None
                },
                "neighbor_influence": neighbor_scores[:5],
                "videos_in_graph": video_ids
            }

            results_file = self.results_dir / f"{video_id}_gnn.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

            # Publish results
            await self.nats_client.publish(
                self.config.get("nats", {}).get("subjects", {}).get("pipeline_gnn", "pipeline.gnn"),
                {
                    "video_id": video_id,
                    "cow_id": target_cow_id,
                    "pipeline": "gnn",
                    "results_path": str(results_file),
                    "severity_score": node_severity_score,
                    "cow_severity_score": cow_severity_score,
                    "uncertainty": node_uncertainty,
                    "model": self.model_name
                }
            )

            print(f"  ✅ {self.model_name} completed:")
            print(f"     Video score={node_severity_score:.3f}, Cow score={cow_severity_score:.3f}")
            print(f"     Graph: {len(video_ids)} nodes, {len(neighbor_scores)} neighbors")
            if target_cow_id:
                print(f"     Cow ID: {target_cow_id}")
            
        except Exception as e:
            print(f"  ❌ Error in GNN pipeline: {e}")
            import traceback
            traceback.print_exc()
    
    async def start(self):
        """Start the GNN pipeline service"""
        await self.nats_client.connect()
        
        # Subscribe to DINOv3 results (after embeddings are computed)
        subject = self.config.get("nats", {}).get("subjects", {}).get(
            "pipeline_dinov3", "pipeline.dinov3"
        )
        print(f"GNN pipeline subscribing to: {subject}")
        
        await self.nats_client.subscribe(subject, self.process_video)
        
        print("=" * 60)
        print("GNN Pipeline Service Started")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Model: {self.model_name}")
        if self.model_name == "EnhancedGraphGPS":
            num_layers = len(self.model.pre_pool_layers) + len(self.model.post_pool_layers)
            print(f"  - Layers: {num_layers} (pre-pool: {len(self.model.pre_pool_layers)}, post-pool: {len(self.model.post_pool_layers)})")
            print(f"  - Attention heads: 8")
            print(f"  - Edge features: enabled (3-dim)")
            print(f"  - Hierarchical pooling: enabled (ratio=0.5)")
            print(f"  - Positional encodings: True Laplacian + RW")
        else:
            print(f"  - Layers: {len(self.model.layers)}")
            print(f"  - Attention heads: 4")
        print(f"Hidden dim: {self.model.hidden_dim}")
        print(f"k-neighbors: {self.graph_builder.k_neighbors}")
        print("=" * 60)
        
        await asyncio.Event().wait()


async def main():
    """Main entry point"""
    pipeline = GNNPipeline()
    await pipeline.start()


if __name__ == "__main__":
    asyncio.run(main())

