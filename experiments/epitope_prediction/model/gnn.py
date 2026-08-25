#!/usr/bin/env python3
"""Shared GNN architecture for Model A (geometric-only) and Model B (geometric+ESM2),
trained as 5-member bagging-PU ensembles with calibration -- per your addition to
PLAN.md: a confidence score, not just a point prediction, so the steering step can
decide *how many* predicted residues to trust, not just which ones rank highest.

Bagging-PU (PLAN.md sec "Ensemble training + confidence"): each of the 5 members
trains on the same confirmed positives but a different bootstrap resample of the
*unlabeled* residues as weak negatives. This does double duty: it's the
confidence-estimation mechanism (members disagree more on genuinely ambiguous
residues) and a principled treatment of positive-unlabeled uncertainty (a single
model trained once on "unlabeled = negative" can't express "I'm not sure this is
really a negative, it just hasn't been crystallized with an antibody yet").

Usage:
    python3 experiments/epitope_prediction/model/gnn.py train --model A
    python3 experiments/epitope_prediction/model/gnn.py train --model B
"""

import argparse
import hashlib
import sys
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import SAGEConv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import load_cached, load_cached_c  # noqa: E402
from egnn_encoder import EGNNREncoder  # noqa: E402
from esm2_features import PROJECTED_DIM, Projection  # noqa: E402
from features import GEOMETRIC_DIM, N_RELATIONS, RESIDUE_TYPE_DIM  # noqa: E402 -- features.py is the single source of truth for these
from res_mp_fork import EDGE_DIM as RESMP_EDGE_DIM  # noqa: E402
from res_mp_fork import NODE_DIM as RESMP_NODE_DIM  # noqa: E402
from res_mp_fork import ResMPFork  # noqa: E402

HIDDEN_DIM = 64
NUM_LAYERS = 4
ENSEMBLE_SIZE = 5
VAL_FRACTION = 0.1
PATIENCE = 5  # epochs without validation AUC improvement before stopping
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def compute_pos_weight(entries: list[dict]) -> float:
    """(n_negative / n_positive) from the training split, for BCEWithLogitsLoss's
    pos_weight -- previously absent despite a 6.4-9.8% positive rate (unweighted loss
    trains on the bagging-PU-bootstrapped batch composition alone, which doesn't
    correct for the base rate). Clamped to [1, 30] so a pathological bootstrap draw
    (or a very rare-positive structure) can't destabilize training with an extreme
    weight."""
    n_pos = sum(e["labels"].sum().item() for e in entries)
    n_total = sum(e["labels"].numel() for e in entries)
    n_neg = n_total - n_pos
    if n_pos == 0:
        return 1.0
    return max(1.0, min(30.0, n_neg / n_pos))


def split_train_val(entries: list[dict], val_fraction: float = VAL_FRACTION) -> tuple[list[dict], list[dict]]:
    """Deterministic (hash-of-pdb_id, not random-seeded) structure-level split --
    entirely separate from databases/splits/dev.txt and test.txt, which stay reserved
    for the Model A vs. B comparison and final reporting, not used for early stopping
    mechanics (PLAN.md sec 4). Structure-level (not residue-level) so a whole antigen's
    residues are either all-train or all-val, matching how the model will actually be
    evaluated later."""
    train, val = [], []
    for e in entries:
        h = int(hashlib.sha256(e["pdb_id"].encode()).hexdigest(), 16)
        (val if (h % 100) < int(val_fraction * 100) else train).append(e)
    return train, val


@torch.no_grad()
def _eval_auc(model: "EpitopeGNN", entries: list[dict], device: str) -> float:
    model.eval()
    all_y, all_p = [], []
    for e in entries:
        logits = _forward_entry(model, e, device)
        all_p.append(torch.sigmoid(logits).cpu())
        all_y.append(e["labels"])
    y = torch.cat(all_y).numpy()
    p = torch.cat(all_p).numpy()
    if y.sum() == 0 or y.sum() == len(y):
        return 0.5
    return roc_auc_score(y, p)


class EpitopeGNN(nn.Module):
    """SAGEConv message passing (verified working on this GPU without the optional
    pyg-lib/torch-scatter compiled extensions that knn_graph needed -- see
    model/features.py's build_knn_edge_index for that story), residual connections,
    single-logit-per-residue output head. feature_set="A" (geometric only, 4-dim
    input) or "B" (geometric + a learned projection of ESM2 embeddings).

    feature_set="C" swaps the SAGEConv stack for EGNNREncoder (EpiFormer-inspired
    multi-relational equivariant encoder, see egnn_encoder.py) -- a different
    architecture, not just different input features, so its forward() takes
    (node_scalars, ca_coords, edges_by_relation) instead of (node_features,
    edge_index). Model A/B's architecture and call signature are unchanged.

    feature_set="D" swaps in ResMPFork (res_mp_fork.py) -- a faithful port of
    EpiFormer's own antigen-branch encoder code (not a reimplementation, see
    res_mp_fork.py's module docstring). Same input convention as "C" (reuses Model
    C's exact cache), but ResMPFork keeps EpiFormer's own hidden_dim=128 (not this
    class's shared HIDDEN_DIM=64) and projects back down to NODE_DIM before the
    output head, matching their own ResMP.forward()'s node_proj_out -- so "D"'s
    output_head input width differs from "A"/"B"/"C"."""

    def __init__(self, feature_set: str = "A", hidden_dim: int = HIDDEN_DIM, num_layers: int = NUM_LAYERS):
        super().__init__()
        self.feature_set = feature_set
        if feature_set == "C":
            in_dim = GEOMETRIC_DIM + RESIDUE_TYPE_DIM
            self.encoder = EGNNREncoder(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, n_relations=N_RELATIONS)
            self.output_head = nn.Linear(hidden_dim, 1)
            return
        if feature_set == "D":
            self.encoder = ResMPFork(node_dim=RESMP_NODE_DIM, edge_dim=RESMP_EDGE_DIM)
            self.output_head = nn.Linear(RESMP_NODE_DIM, 1)
            return

        in_dim = GEOMETRIC_DIM + (PROJECTED_DIM if feature_set == "B" else 0)
        if feature_set == "B":
            self.esm2_projection = Projection()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.output_head = nn.Linear(hidden_dim, 1)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, esm2_embeddings: torch.Tensor | None = None) -> torch.Tensor:
        x = node_features
        if self.feature_set == "B":
            assert esm2_embeddings is not None, "Model B requires esm2_embeddings"
            x = torch.cat([x, self.esm2_projection(esm2_embeddings)], dim=-1)
        h = torch.relu(self.input_proj(x))
        for layer in self.layers:
            h = h + torch.relu(layer(h, edge_index))
        return self.output_head(h).squeeze(-1)  # [N] logits

    def forward_c(self, node_scalars: torch.Tensor, ca_coords: torch.Tensor, edges_by_relation: dict[int, torch.Tensor]) -> torch.Tensor:
        """Shared by feature_set "C" and "D" -- both encoders take the same
        (node_scalars, ca_coords, edges_by_relation) signature and return a tensor
        the shared output_head consumes; only self.encoder's class differs."""
        h = self.encoder(node_scalars, ca_coords, edges_by_relation)
        return self.output_head(h).squeeze(-1)  # [N] logits


def _forward_entry(model: "EpitopeGNN", e: dict, device: str) -> torch.Tensor:
    """Dispatches a cached entry through `model`, keyed off model.feature_set --
    Model A/B entries carry node_features/edge_index(/esm2_embeddings); Model C/D
    entries carry node_scalars/backbone_coords/edges_by_relation (see
    features.py's build_multirelational_features / dataset.py's build_c_entries --
    Model D reuses Model C's cache verbatim, see res_mp_fork.py)."""
    if model.feature_set in ("C", "D"):
        node_scalars = e["node_scalars"].to(device)
        ca_coords = e["backbone_coords"][:, 1, :].to(device)  # index 1 = CA, see build_backbone_coords' [N,CA,CB,O] order
        edges_by_relation = {rel: idx.to(device) for rel, idx in e["edges_by_relation"].items()}
        return model.forward_c(node_scalars, ca_coords, edges_by_relation)

    x = e["node_features"].to(device)
    edge_index = e["edge_index"].to(device)
    esm2 = e.get("esm2_embeddings")
    if esm2 is not None:
        esm2 = esm2.to(device)
    return model(x, edge_index, esm2)


def train_member(train_entries: list[dict], val_entries: list[dict], feature_set: str, seed: int, device: str,
                  bootstrap_unlabeled: bool = True, max_epochs: int = 60, patience: int = PATIENCE, lr: float = 1e-3,
                  pos_weight: float = 1.0) -> EpitopeGNN:
    """Trains one ensemble member with early stopping on a held-out validation slice
    (best-checkpoint pattern: track the best validation AUC seen, restore those weights
    at the end, rather than literally halting the loop -- avoids ensemble members
    stopping at different epoch counts, simpler to reason about). Missing from the
    first training pass (a real gap, not a design choice) -- added after Model B's
    initial run showed a textbook overfitting signature (much lower training loss than
    Model A, but *worse* held-out AUC and a failed confidence-sanity check) that a
    fixed 30-epoch schedule with no validation signal couldn't catch.

    Bagging-PU: for each entry, positives are always kept; unlabeled (label=0)
    residues are bootstrap-resampled per member (sampling with replacement from the
    unlabeled pool, same size) so different members see a different weak-negative set
    -- the source of both PU-learning robustness and ensemble disagreement/confidence."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = EpitopeGNN(feature_set=feature_set).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    best_auc = -1.0
    best_state = None
    epochs_since_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        order = torch.randperm(len(train_entries), generator=gen).tolist()
        for idx in order:
            e = train_entries[idx]
            y = e["labels"].to(device)

            if bootstrap_unlabeled:
                pos_idx = (y == 1).nonzero(as_tuple=True)[0]
                neg_idx = (y == 0).nonzero(as_tuple=True)[0]
                if len(neg_idx) > 0:
                    boot_neg = neg_idx[torch.randint(0, len(neg_idx), (len(neg_idx),), generator=gen)]
                    keep = torch.cat([pos_idx, boot_neg]).unique()
                else:
                    keep = pos_idx
                mask = torch.zeros(len(y), dtype=torch.bool)
                mask[keep] = True
            else:
                mask = torch.ones(len(y), dtype=torch.bool)

            opt.zero_grad()
            logits = _forward_entry(model, e, device)
            loss = loss_fn(logits[mask], y[mask])
            loss.backward()
            opt.step()
            total_loss += loss.item()

        val_auc = _eval_auc(model, val_entries, device)
        improved = val_auc > best_auc + 1e-4
        if improved:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if (epoch + 1) % 5 == 0 or improved:
            print(f"[gnn:{feature_set}:seed{seed}] epoch {epoch + 1}/{max_epochs} "
                  f"train_loss={total_loss / len(train_entries):.4f} val_auc={val_auc:.4f} "
                  f"{'(best)' if improved else f'(no improvement {epochs_since_improvement}/{patience})'}",
                  file=sys.stderr)

        if epochs_since_improvement >= patience:
            print(f"[gnn:{feature_set}:seed{seed}] early stopping at epoch {epoch + 1}, best val_auc={best_auc:.4f}", file=sys.stderr)
            break

    model.load_state_dict(best_state)
    return model


def train_ensemble(feature_set: str, train_entries: list[dict], device: str, n_members: int = ENSEMBLE_SIZE) -> list[EpitopeGNN]:
    train_split, val_split = split_train_val(train_entries)
    print(f"[gnn:{feature_set}] {len(train_split)} train / {len(val_split)} internal-val structures "
          f"(hash-split, distinct from databases/splits/dev.txt)", file=sys.stderr)
    pos_weight = compute_pos_weight(train_split)
    n_pos = sum(e["labels"].sum().item() for e in train_split)
    n_total = sum(e["labels"].numel() for e in train_split)
    print(f"[gnn:{feature_set}] pos_weight={pos_weight:.2f} (positive rate {n_pos/n_total:.1%})", file=sys.stderr)
    return [train_member(train_split, val_split, feature_set, seed=i, device=device, pos_weight=pos_weight) for i in range(n_members)]


@torch.no_grad()
def ensemble_predict(models: list[EpitopeGNN], entry: dict, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (propensity, confidence) per residue: propensity = mean predicted
    probability across ensemble members; confidence = 1 - normalized std (agreement)."""
    probs = []
    for model in models:
        model.eval()
        logits = _forward_entry(model, entry, device)
        probs.append(torch.sigmoid(logits).cpu())
    probs = torch.stack(probs, dim=0)  # [n_members, N]
    propensity = probs.mean(dim=0)
    # Max possible std for Bernoulli-like disagreement in [0,1] is 0.5 (half predict 0, half predict 1).
    confidence = 1.0 - (probs.std(dim=0) / 0.5).clamp(max=1.0)
    return propensity, confidence


def save_ensemble(models: list[EpitopeGNN], feature_set: str) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for i, model in enumerate(models):
        torch.save(model.state_dict(), CHECKPOINT_DIR / f"model_{feature_set}_seed{i}.pt")


def load_ensemble(feature_set: str, device: str, n_members: int = ENSEMBLE_SIZE) -> list[EpitopeGNN]:
    models = []
    for i in range(n_members):
        model = EpitopeGNN(feature_set=feature_set).to(device)
        model.load_state_dict(torch.load(CHECKPOINT_DIR / f"model_{feature_set}_seed{i}.pt", map_location=device))
        models.append(model)
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["train"])
    parser.add_argument("--model", required=True, choices=["A", "B", "C", "D"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_entries = load_cached_c("train") if args.model in ("C", "D") else load_cached("train")
    print(f"[gnn] training Model {args.model} ensemble ({ENSEMBLE_SIZE} members) on {len(train_entries)} structures", file=sys.stderr)
    models = train_ensemble(args.model, train_entries, device)
    save_ensemble(models, args.model)
    print(f"[gnn] saved ensemble to {CHECKPOINT_DIR}", file=sys.stderr)

    import calibration  # local import: calibration.py imports this module too (fit_calibrator uses ensemble_predict)
    _, val_split = split_train_val(train_entries)  # deterministic (hash-of-pdb_id) -- reproduces the exact slice used for early stopping above
    calibrator = calibration.fit_calibrator(models, val_split, device)
    calibration.save_calibrator(calibrator, args.model)
    print(f"[gnn] fit + saved isotonic calibrator to {CHECKPOINT_DIR}/calibrator_{args.model}.pkl", file=sys.stderr)


if __name__ == "__main__":
    main()
