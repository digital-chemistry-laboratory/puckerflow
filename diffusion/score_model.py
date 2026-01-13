"""Equivariant score model used throughout PuckerFlow."""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from e3nn import o3
from e3nn.nn import BatchNorm
from torch import Tensor, nn
from torch_cluster import radius, radius_graph
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
from torch_scatter import scatter

from utils.coordinate_transforms import z_to_ab_batch

# --- Model Components ---


class TensorProductConvLayer(nn.Module):
    """E(3) equivariant convolution using tensor products and scatter ops."""

    def __init__(
        self,
        in_irreps: str,
        sh_irreps: str,
        out_irreps: str,
        n_edge_features: int,
        residual: bool = True,
        batch_norm: bool = True,
    ):
        """
        Args:
            in_irreps: String representation of the input irreps.
            sh_irreps: String representation of the spherical harmonics irreps.
            out_irreps: String representation of the output irreps.
            n_edge_features: Dimensionality of the edge scalar feature vector.
            residual: Whether to add a residual connection.
            batch_norm: Whether to use e3nn BatchNorm.
        """
        super().__init__()
        self.in_irreps = o3.Irreps(in_irreps)
        self.out_irreps = o3.Irreps(out_irreps)
        self.sh_irreps = o3.Irreps(sh_irreps)
        self.residual = residual

        self.tp = o3.FullyConnectedTensorProduct(
            self.in_irreps, self.sh_irreps, self.out_irreps, shared_weights=False
        )

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, n_edge_features),
            nn.ReLU(),
            nn.Linear(n_edge_features, self.tp.weight_numel),
        )
        self.batch_norm = BatchNorm(self.out_irreps) if batch_norm else None

    def forward(
        self,
        node_attr: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_sh: Tensor,
        out_nodes: Optional[int] = None,
        reduce: str = "mean",
    ) -> Tensor:
        """Apply the equivariant convolution.

        Args:
            node_attr: Tensor ``[N, in_irreps.dim]`` with source node features.
            edge_index: Long tensor of shape ``[2, E]`` describing message
                directions.
            edge_attr: Scalar edge attributes ``[E, n_edge_features]``.
            edge_sh: Edge spherical harmonics ``[E, sh_irreps.dim]``.
            out_nodes: Optional explicit number of nodes after scatter.
            reduce: Reduction operator for :func:`torch_scatter.scatter`.

        Returns:
            Tensor ``[out_nodes, out_irreps.dim]`` containing updated features.
        """
        out_nodes = out_nodes or node_attr.shape[0]
        edge_src, edge_dst = edge_index

        # Compute tensor product
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        # Aggregate messages
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)

        if self.residual:
            # Pad node_attr to match out_irreps dimension if needed
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        if self.batch_norm:
            out = self.batch_norm(out)

        return out


class TensorProductScoreModel(nn.Module):
    """
    The main E(3) Equivariant Score Model.

    This model predicts the 'ab' puckering coordinates vector given the
    molecular graph and a time step 't'.
    """

    def __init__(
        self,
        in_node_features: int,
        sigma_embed_dim: int,
        ns: int,
        nv: int,
        num_conv_layers: int,
        max_radius: float,
        radius_embed_dim: int,
        use_second_order_repr: bool,
        batch_norm: bool,
        residual: bool,
        ft_embedding: bool = False,
        sh_lmax: int = 2,
        in_edge_features: int = 4,
    ):
        """Configure the equivariant score network.

        Args:
            in_node_features: Dimension of raw per-atom features.
            sigma_embed_dim: Dimension of the sinusoidal time embedding.
            ns: Number of scalar channels (``l = 0``) per layer.
            nv: Number of vector/tensor channels (``l >= 1``).
            num_conv_layers: Depth of the tensor product stack.
            max_radius: Cutoff distance (Å) for the neighbor graph.
            radius_embed_dim: Width of the Gaussian distance basis.
            use_second_order_repr: Whether to include ``l = 2`` irreps.
            batch_norm: Enable equivariant batch normalization.
            residual: Use residual additions between convolution layers.
            ft_embedding: Legacy boolean (kept for backwards compatibility).
            sh_lmax: Maximum ``l`` for spherical harmonics evaluation.
            in_edge_features: Number of scalar bond-edge features.
        """
        super().__init__()
        self.in_node_features = in_node_features
        self.in_edge_features = in_edge_features
        self.sigma_embed_dim = sigma_embed_dim
        self.max_radius = max_radius
        self.radius_embed_dim = radius_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.ft_embedding = ft_embedding

        # --- Embedding Layers ---
        self.node_embedding = self._make_mlp(in_node_features + sigma_embed_dim, ns)
        self.edge_embedding = self._make_mlp(
            in_edge_features + sigma_embed_dim + radius_embed_dim, ns
        )
        self.distance_expansion = GaussianSmearing(0.0, max_radius, radius_embed_dim)

        # --- Interaction Layers ---
        self.conv_layers = self._build_conv_layers(
            num_conv_layers, use_second_order_repr, ns, nv, batch_norm, residual
        )

        # --- Output Layers ---
        self.center_edge_embedding = self._make_mlp(radius_embed_dim, ns, dropout=0.1)
        self.final_linear = self._make_final_mlp(ns)

        self.mean_conv = TensorProductConvLayer(
            in_irreps=self.conv_layers[-1].out_irreps.simplify(),
            sh_irreps=self.sh_irreps.simplify(),
            out_irreps=o3.Irreps(f"{ns}x0o").simplify(),
            n_edge_features=2 * ns,  # (center_edge_emb + node_attr[dst])
            residual=False,
            batch_norm=batch_norm,
        )

    def _make_mlp(
        self, input_dim: int, output_dim: int, dropout: float = 0.0
    ) -> nn.Sequential:
        """Return a two-layer MLP block with optional dropout."""
        layers = [nn.Linear(input_dim, output_dim), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(output_dim, output_dim))
        return nn.Sequential(*layers)

    def _make_final_mlp(self, ns: int) -> nn.Sequential:
        """Create the scalar head converting hidden states into z-values."""
        return nn.Sequential(
            nn.Linear(ns, ns * 3, bias=False),
            nn.Tanh(),
            nn.Linear(ns * 3, ns, bias=False),
            nn.Tanh(),
            nn.Linear(ns, 1, bias=False),
        )

    def _build_conv_layers(
        self,
        num_conv_layers: int,
        use_second_order_repr: bool,
        ns: int,
        nv: int,
        batch_norm: bool,
        residual: bool,
    ) -> nn.ModuleList:
        """Build the tensor-product convolution stack."""
        irrep_seq = self._get_irreps_sequence(use_second_order_repr, ns, nv)
        conv_layers = []

        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            conv_layers.append(
                TensorProductConvLayer(
                    in_irreps=in_irreps,
                    sh_irreps=self.sh_irreps.simplify(),
                    out_irreps=out_irreps,
                    n_edge_features=3
                    * ns,  # (edge_emb + node_attr[src] + node_attr[dst])
                    residual=residual,
                    batch_norm=batch_norm,
                )
            )

        return nn.ModuleList(conv_layers)

    def _get_irreps_sequence(
        self, use_second_order_repr: bool, ns: int, nv: int
    ) -> List[str]:
        """Return a list of irreps strings for each convolution block."""
        if use_second_order_repr:
            # l=0, l=1, l=2
            return [
                f"{ns}x0e",
                f"{ns}x0e + {nv}x1o + {nv}x2e",
                f"{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o",
                f"{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o",
            ]
        else:
            # l=0, l=1 only
            return [
                f"{ns}x0e",
                f"{ns}x0e + {nv}x1o",
                f"{ns}x0e + {nv}x1o + {nv}x1e",
                f"{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o",
            ]

    def forward(self, data: Data) -> Tensor:
        """Predict ``ab`` Fourier coordinates for every molecule in ``data``.

        The ``Data``/``Batch`` object is expected to include at least ``x``,
        ``edge_index``, ``edge_attr``, ``pos``, ``t``, ``bl``, ``batch`` and
        ``ptr``. Single-graph inputs are automatically promoted to a batch of
        size one.
        """
        is_single_graph = data.batch is None
        if is_single_graph:
            # Create synthetic batch info for a single graph
            data.batch = torch.zeros(
                data.x.shape[0], dtype=torch.int64, device=data.x.device
            )
            data.ptr = torch.tensor([0, data.x.shape[0]], device=data.x.device)

        # Build standard graph
        node_attr, edge_index, edge_attr, edge_sh = self.build_conv_graph(data)
        src, dst = edge_index

        # Embeddings
        node_attr = self.node_embedding(node_attr)
        edge_attr = self.edge_embedding(edge_attr)

        # Interaction Layers
        for li, layer in enumerate(self.conv_layers):
            # Augment edge features with node features
            edge_attr_ = torch.cat(
                [edge_attr, node_attr[src, : self.ns], node_attr[dst, : self.ns]], -1
            )
            node_attr = layer(node_attr, edge_index, edge_attr_, edge_sh, reduce="mean")

        # Mean Plane Graph (for final prediction)
        (
            mean_edge_index,
            mean_edge_attr,
            mean_edge_sh,
        ) = self.build_mean_plane_conv_graph(data)

        # Embed mean plane edge attributes
        mean_edge_attr = self.center_edge_embedding(mean_edge_attr)

        # Augment mean edge features with destination node features
        mean_edge_attr = torch.cat(
            [mean_edge_attr, node_attr[mean_edge_index[1], : self.ns]], -1
        )

        # Final Convolution 
        out = self.mean_conv(
            node_attr,
            mean_edge_index,
            mean_edge_attr,
            mean_edge_sh,
            out_nodes=len(data.bl),  # This is N_atoms
            reduce="mean",
        )

        # Prediction Head
        z_pred = self.final_linear(out)

        # Convert z-predictions to ab-predictions
        pred = z_to_ab_batch(data, z_pred)

        return pred

    def build_mean_plane_conv_graph(
        self, data: Data
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Builds the graph for the mean-plane convolution.
        This graph connects all nodes to a virtual "mean plane" point.

        Args:
            data: The PyG Data (or Batch) object.

        Returns:
            A tuple (edge_index, edge_attr, edge_sh).
        """
        edge_index = radius(
            data.pos, data.pos, 100, batch_x=data.batch, batch_y=data.batch
        ).to(data.x.device)
        edge_index, _ = remove_self_loops(edge_index)
        src, dst = edge_index  

        projected_src = torch.cat(
            (data.pos[src, :2], torch.zeros(len(src), 1, device=data.pos.device)), 1
        )
        edge_vec = data.pos[dst] - projected_src

        # Mask out zero-length edges (to avoid NaNs in spherical harmonics)
        edge_len = edge_vec.norm(dim=-1)
        mask = edge_len > 1e-8
        edge_index = edge_index[:, mask]
        edge_vec = edge_vec[mask]
        edge_len = edge_len[mask]

        # Expand distances
        edge_attr = self.distance_expansion(edge_len)

        # Compute spherical harmonics
        edge_sh = o3.spherical_harmonics(
            self.sh_irreps, edge_vec, normalize=True, normalization="component"
        ).to(data.x.device)

        return edge_index, edge_attr, edge_sh

    def build_conv_graph(
        self, data: Data
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Builds the main interaction graph, including radius edges.
        Embeds node, edge, and time features.

        Args:
            data: The PyG Data (or Batch) object.

        Returns:
            A tuple (node_attr, edge_index, edge_attr, edge_sh).
        """
        radius_edges = radius_graph(data.pos, self.max_radius, data.batch).to(
            data.x.device
        )
        edge_index = torch.cat([data.edge_index, radius_edges], 1).long()

        radius_edge_attr = torch.zeros(
            radius_edges.shape[-1], self.in_edge_features, device=data.x.device
        )
        edge_attr = torch.cat([data.edge_attr, radius_edge_attr], 0)

        t_per_node = torch.cat(
            [
                data.t[i].repeat(data.ptr[i + 1] - data.ptr[i])
                for i in range(len(data.ptr) - 1)
            ]
        )
        node_sigma_emb = get_timestep_embedding(t_per_node, self.sigma_embed_dim).to(
            data.x.device
        )
        data.node_sigma_emb = node_sigma_emb

        edge_sigma_emb = node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)

        node_attr = torch.cat([data.x, node_sigma_emb], 1).float()

        src, dst = edge_index
        edge_vec = data.pos[dst.long()] - data.pos[src.long()]

        edge_length_emb = self.distance_expansion(edge_vec.norm(dim=-1)).to(
            data.x.device
        )
        edge_attr = torch.cat([edge_attr, edge_length_emb], 1).float()

        edge_sh = o3.spherical_harmonics(
            self.sh_irreps, edge_vec, normalize=True, normalization="component"
        ).to(data.x.device)

        return node_attr, edge_index, edge_attr, edge_sh


class GaussianSmearing(nn.Module):
    """
    Expands a distance scalar into a vector of Gaussian basis functions.
    """

    def __init__(self, start: float = 0.0, stop: float = 5.0, num_gaussians: int = 50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian smearing to a tensor of distances.

        Args:
            dist: A 1D tensor of distances.

        Returns:
            A 2D tensor of shape (len(dist), num_gaussians).
        """
        dist = dist.view(-1, 1) - self.offset.view(1, -1).to(dist.device)
        return torch.exp(self.coeff * torch.pow(dist, 2))


# Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
def get_timestep_embedding(
    timesteps: torch.Tensor, embedding_dim: int, max_positions: int = 10000
) -> torch.Tensor:
    """
    Creates sinusoidal timestep embeddings.

    Args:
        timesteps: A 1D tensor of time steps.
        embedding_dim: The dimension of the embedding.
        max_positions: Maximum number of positions.

    Returns:
        A 2D tensor of shape (len(timesteps), embedding_dim).
    """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2

    # Calculate embedding frequencies
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )

    # Calculate embedding values
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    # Zero pad if embedding_dim is odd
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode="constant")

    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb
