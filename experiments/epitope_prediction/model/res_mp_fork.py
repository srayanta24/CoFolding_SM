#!/usr/bin/env python3
"""A faithful fork of EpiFormer's actual antigen-branch encoder (src/epiformer/model/res_mp.py's
`MultiRelationalEGNNLayer`/`ResMP` -- the class `EpiformerBlock` instantiates by default,
`ag_resmp_type="egnn"`), for Model D -- see PLAN.md's Phase 4 write-up. This is a *port*, not
a reimplementation from the paper text (that's Model C, `egnn_encoder.py`): the math below
(message MLPs, RBF distance encoding, coordinate-update rule, per-relation weight sharing,
layer norm, dropout, residual) is copied as closely as practical from their source, changed
only where needed to interface with our data instead of theirs. What's kept identical:
hidden_dim=128, num_layers=4, the exact message/coord MLP shapes, RBF(0..20, 16 centers),
`coords_agg='mean'` degree-normalized aggregation.

Two real interface changes from their code, both necessary, neither touching the math:
1. `forward()` takes (h, x, edges_by_relation: dict[int, Tensor[2,E]]) on plain tensors,
   not a PyG HeteroData mutated in place -- their forward() hard-depends on HeteroData
   indexing/mutation throughout, but the layer's own math has no other PyG dependency once
   that's removed (verified by reading the source directly before forking).
2. `torch_scatter.scatter_add` -> manual `torch.zeros(...).index_add_(...)`, so this encoder
   runs in `.venvs/epitope-prediction` (which deliberately has no torch-scatter, same
   aarch64-fragility-avoidance convention as `features.py`'s `build_knn_edge_index`) instead
   of requiring the compiled-extension venv Phase 3 needed a from-source build for.

Their relation index (edge_type=(node_type, f'r{rel}', node_type), rel 0-3) has no semantic
meaning inside res_mp.py itself -- the sequential/short-range/knn/spatial assignment happens
in their own dataset construction, not here. Mapped 1:1 to our features.py's 1-indexed
relations in the same paper order (rho1 sequential->r0, rho2 short-range->r1, rho3 knn->r2,
rho4 medium-spatial->r3) for a semantically faithful port, not just a shape-compatible one.

Also changed from their default: `edge_dim=0` (we have no precomputed per-edge feature
vector -- their RBF distance term is computed inline regardless, and edge_attr concatenated
at width 0 contributes nothing, verified this concatenates cleanly rather than assumed).
`node_dim=28` (Model C's node_scalars width: 7 geometric + 21-dim residue one-hot from
`features.py`) instead of their AsEP-specific 105; no PLM/ESM2 branch (matches Model C's
antigen-only convention).
"""

import torch
import torch.nn as nn

NODE_DIM = 28  # Model C's node_scalars width -- res_mp_fork.py's fork target this exactly matches
EDGE_DIM = 0  # no precomputed edge features; see module docstring
RESMP_HIDDEN_DIM = 128  # their value, kept as-is (vs our own Model C/A/B's HIDDEN_DIM=64) -- a faithful port, not a re-tuned one
RESMP_NUM_LAYERS = 4
N_RELATIONS = 4


def _scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """torch_scatter.scatter_add's exact semantics via plain index_add_ -- see module
    docstring point 2."""
    shape = (dim_size,) + src.shape[1:]
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    return out


class MultiRelationalEGNNLayer(nn.Module):
    """Ported from res_mp.py's class of the same name. Math unchanged; forward() signature
    and aggregation mechanism changed per the module docstring."""

    def __init__(self, node_dim, edge_dim, hidden_dim, num_relations=4,
                 act_fn=nn.SiLU(), residual=True, normalize=False,
                 coords_agg="mean", tanh=False, update_coords=True):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.residual = residual
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.update_coords = update_coords
        self.epsilon = 1e-8

        self.message_mlps = nn.ModuleDict()
        self.coord_mlps = nn.ModuleDict()
        for r in range(num_relations):
            self.message_mlps[str(r)] = nn.Sequential(
                nn.Linear(2 * node_dim + edge_dim + 16, hidden_dim),  # +16 for RBF distance
                act_fn,
                nn.Linear(hidden_dim, hidden_dim),
                act_fn,
            )
            coord_layers = [
                nn.Linear(hidden_dim, hidden_dim),
                act_fn,
                nn.Linear(hidden_dim, 1, bias=False),
            ]
            if tanh:
                coord_layers.append(nn.Tanh())
            self.coord_mlps[str(r)] = nn.Sequential(*coord_layers)

        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, node_dim),
        )

        self.rbf_centers = nn.Parameter(torch.linspace(0, 20, 16), requires_grad=False)
        self.rbf_width = 1.0

    def compute_rbf(self, distances: torch.Tensor) -> torch.Tensor:
        rbf = torch.exp(-0.5 * ((distances.unsqueeze(-1) - self.rbf_centers) / self.rbf_width) ** 2)
        return rbf

    def forward(self, h: torch.Tensor, x: torch.Tensor, edges_by_relation: dict[int, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        num_nodes = h.shape[0]
        total_messages = torch.zeros_like(h)
        coord_updates = torch.zeros_like(x)

        for rel in range(4):
            edge_index = edges_by_relation.get(rel + 1)  # our features.py is 1-indexed, see module docstring
            if edge_index is None or edge_index.size(1) == 0:
                continue

            row, col = edge_index
            edge_attr = torch.zeros(edge_index.size(1), self.edge_dim, dtype=h.dtype, device=h.device)

            coord_diff = x[row] - x[col]
            distances = torch.norm(coord_diff, dim=1, keepdim=True)
            if self.normalize:
                coord_diff = coord_diff / (distances + self.epsilon)

            rbf_dist = self.compute_rbf(distances.squeeze(-1))
            if rbf_dist.dim() == 1:
                rbf_dist = rbf_dist.unsqueeze(0)

            message_input = torch.cat([h[row], h[col], edge_attr, rbf_dist], dim=1)

            mlp_key = str(rel % self.num_relations)
            messages = self.message_mlps[mlp_key](message_input)
            coord_weights = self.coord_mlps[mlp_key](messages)

            rel_messages = _scatter_add(messages, col, num_nodes)
            total_messages += rel_messages

            coord_update = coord_diff * coord_weights
            if self.coords_agg == "mean":
                degree = _scatter_add(torch.ones_like(col, dtype=torch.float), col, num_nodes).unsqueeze(1)
                coord_update = coord_update / (degree[col] + self.epsilon)
            rel_coord_updates = _scatter_add(coord_update, col, num_nodes)
            coord_updates += rel_coord_updates

        node_input = torch.cat([h, total_messages], dim=1)
        h_new = self.node_mlp(node_input)
        if self.residual:
            h_new = h + h_new
        x_new = x + coord_updates if self.update_coords else x
        return h_new, x_new


class ResMPFork(nn.Module):
    """Ported from res_mp.py's `ResMP`. forward() takes plain tensors (see module
    docstring) instead of a HeteroData; everything else -- input/output projections,
    per-layer LayerNorm+dropout, the layer stack itself -- matches their code."""

    def __init__(self, node_dim: int = NODE_DIM, edge_dim: int = EDGE_DIM, hidden_dim: int = RESMP_HIDDEN_DIM,
                 num_layers: int = RESMP_NUM_LAYERS, num_relations: int = N_RELATIONS,
                 act_fn=nn.SiLU(), residual: bool = True, normalize: bool = False,
                 dropout: float = 0.1, layer_norm: bool = True, update_coords: bool = True):
        super().__init__()
        self.layer_norm = layer_norm
        self.node_proj_in = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList([
            MultiRelationalEGNNLayer(
                node_dim=hidden_dim, edge_dim=edge_dim, hidden_dim=hidden_dim,
                num_relations=num_relations, act_fn=act_fn, residual=residual,
                normalize=normalize, update_coords=update_coords,
            )
            for _ in range(num_layers)
        ])
        if layer_norm:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.node_proj_out = nn.Linear(hidden_dim, node_dim)

    def forward(self, node_scalars: torch.Tensor, ca_coords: torch.Tensor, edges_by_relation: dict[int, torch.Tensor]) -> torch.Tensor:
        h = self.node_proj_in(node_scalars)
        x = ca_coords.clone()
        for i, layer in enumerate(self.layers):
            h, x = layer(h, x, edges_by_relation)
            if self.layer_norm:
                h = self.layer_norms[i](h)
            if self.dropout:
                h = self.dropout(h)
        return self.node_proj_out(h)


if __name__ == "__main__":
    torch.manual_seed(0)
    n = 20
    node_scalars = torch.randn(n, NODE_DIM)
    ca_coords = torch.randn(n, 3)
    edges_by_relation = {
        1: torch.randint(0, n, (2, 30)),
        2: torch.randint(0, n, (2, 40)),
        3: torch.randint(0, n, (2, 60)),
        4: torch.zeros((2, 0), dtype=torch.long),
    }
    model = ResMPFork()
    out = model(node_scalars, ca_coords, edges_by_relation)
    print("output shape:", tuple(out.shape))
    loss = out.sum()
    loss.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for _ in model.parameters())
    print(f"gradients flowed to {n_grad}/{n_total} parameter tensors")
