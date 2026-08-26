#!/usr/bin/env python3
"""A from-scratch, pure-PyTorch E(3)-equivariant multi-relational GNN encoder (EGNN-R),
borrowing the antigen-side encoder design from EpiFormer (arXiv 2606.04154) for Model C
-- see PLAN.md's write-up. Antigen-only: no antibody input, no cross-attention (we have
no candidate antibody at prediction time, unlike EpiFormer's own antibody-aware setup).

No torch-scatter/e3nn dependency, matching this project's existing aarch64-fragility
discipline (see features.py's build_knn_edge_index) -- per-relation aggregation uses
plain torch.Tensor.index_add_, which needs no compiled extension.

Simplification worth being explicit about: EpiFormer's own EGNN-R updates a per-residue
4-atom coordinate matrix (N, CA, C-beta, O) equivariantly. Here the equivariant channel
operates on CA position only (features.py's backbone_coords[:, 1, :]) for tractability
-- the other 3 backbone atoms' *relative* geometry (bond lengths/angles to CA) is
instead folded into the invariant scalar features via build_backbone_coords' matrix
before this encoder runs, rather than kept as a second equivariant channel. A real
architectural simplification, not a bug -- flagged so it isn't mistaken for a faithful
reimplementation of the paper's full multi-atom update.
"""

import torch
import torch.nn as nn


class EGNNRLayer(nn.Module):
    """One multi-relational EGNN-R layer: per-relation message MLPs (sequential edges
    and spatial edges represent different physics, per EpiFormer's own stated
    rationale for not sharing one message function across relations), an invariant
    scalar update, and an equivariant coordinate update."""

    def __init__(self, hidden_dim: int, n_relations: int):
        super().__init__()
        self.n_relations = n_relations
        self.msg_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim * 2 + 1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(n_relations)
        ])
        self.coord_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
            for _ in range(n_relations)
        ])
        self.update_mlp = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, h: torch.Tensor, x: torch.Tensor, edges_by_relation: dict[int, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """h: [N, hidden_dim] invariant scalars. x: [N, 3] equivariant coordinates
        (CA position). edges_by_relation: {1..n_relations: edge_index [2, E]}.

        Real bug found and fixed here: the coordinate update summed over every
        neighbor (up to ~120 with KNN_K=30 across 4 relations) with no degree
        normalization and no bound on the learned scalar magnitude -- a classic EGNN
        runaway-feedback failure (bigger coordinates -> bigger distances in the next
        layer -> bigger messages -> bigger coordinates...). Verified concretely:
        training loss diverged to ~1e20-1e27 from epoch 1 across all 5 ensemble
        members before this fix, val AUC stuck near chance (~0.5). The real
        EpiFormer code (src/epiformer/model/res_mp.py) avoids exactly this with
        degree-normalized ('mean') aggregation -- applied the same fix here, plus a
        tanh bound on the coordinate scalar (an available but off-by-default option
        in their own code) since mean-aggregation alone wasn't sufficient in testing."""
        agg_msg = torch.zeros_like(h)
        agg_coord = torch.zeros_like(x)
        degree = torch.zeros(h.shape[0], 1, device=h.device)

        for rel in range(1, self.n_relations + 1):
            edge_index = edges_by_relation.get(rel)
            if edge_index is None or edge_index.numel() == 0:
                continue
            src, dst = edge_index[0], edge_index[1]
            delta = x[dst] - x[src]
            dist2 = (delta ** 2).sum(-1, keepdim=True)
            m = self.msg_mlps[rel - 1](torch.cat([h[src], h[dst], dist2], dim=-1))
            s = torch.tanh(self.coord_mlps[rel - 1](m))  # [E, 1] bounded scalar magnitude
            coord_msg = delta / torch.sqrt(dist2 + 1e-8) * s
            agg_msg.index_add_(0, dst, m)
            agg_coord.index_add_(0, dst, coord_msg)
            degree.index_add_(0, dst, torch.ones(dst.shape[0], 1, device=h.device))

        degree = degree.clamp(min=1.0)
        agg_msg = agg_msg / degree
        agg_coord = agg_coord / degree

        h_new = h + self.update_mlp(torch.cat([h, agg_msg], dim=-1))
        x_new = x + agg_coord
        return h_new, x_new


class EGNNREncoder(nn.Module):
    """Stack of EGNN-R layers. Returns only the final invariant scalar embeddings
    (the classification head only needs those, matching EpitopeGNN's output_head
    pattern in gnn.py) -- the updated coordinates are discarded after the last layer."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 4, n_relations: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([EGNNRLayer(hidden_dim, n_relations) for _ in range(num_layers)])

    def forward(self, node_scalars: torch.Tensor, ca_coords: torch.Tensor, edges_by_relation: dict[int, torch.Tensor]) -> torch.Tensor:
        h = torch.relu(self.input_proj(node_scalars))
        x = ca_coords
        for layer in self.layers:
            h, x = layer(h, x, edges_by_relation)
        return h


if __name__ == "__main__":
    # Minimal shape/gradient smoke test -- no real structure needed.
    torch.manual_seed(0)
    n, in_dim, hidden_dim = 20, 28, 64
    node_scalars = torch.randn(n, in_dim)
    ca_coords = torch.randn(n, 3)
    edges_by_relation = {
        1: torch.randint(0, n, (2, 30)),
        2: torch.randint(0, n, (2, 40)),
        3: torch.randint(0, n, (2, 60)),
        4: torch.zeros((2, 0), dtype=torch.long),
    }
    enc = EGNNREncoder(in_dim=in_dim, hidden_dim=hidden_dim)
    out = enc(node_scalars, ca_coords, edges_by_relation)
    print("output shape:", tuple(out.shape))
    loss = out.sum()
    loss.backward()
    n_grad = sum(1 for p in enc.parameters() if p.grad is not None)
    n_total = sum(1 for _ in enc.parameters())
    print(f"gradients flowed to {n_grad}/{n_total} parameter tensors")
