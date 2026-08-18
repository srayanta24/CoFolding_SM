#!/usr/bin/env python3
"""Model B only: per-residue ESM2 embeddings for the antigen sequence.

Design note, a deviation from PLAN.md's original wording ("reuse the antigen sequence
extraction already built for databases/src/build_splits.py"): that extraction reads the
*full declared* sequence from the mmCIF's _entity_poly block, which can include
residues with no resolved coordinates (crystallographic gaps, common in flexible
loops). model/features.py's geometric features, by contrast, are indexed by the
*resolved* residue list (from _atom_site — only residues with real coordinates). Zipping
those two together positionally would silently misalign whenever a structure has a gap.
Instead, this module builds the ESM2 input sequence directly from the same resolved
residue list model/features.py already produces (via data/interface_labels.py's
get_chain_atoms + model/features.py's group_residues), guaranteeing the two feature
sets are aligned by construction — at the minor, well-precedented cost of ESM2 seeing a
sequence with gaps silently closed up rather than gap tokens.

Usage:
    python3 experiments/epitope_prediction/model/esm2_features.py <pdb_id>
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import group_residues  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from interface_labels import get_chain_atoms  # noqa: E402

ESM2_MODEL_NAME = "esm2_t30_150M_UR50D"
ESM2_LAYER = 30
ESM2_DIM = 640
PROJECTED_DIM = 32

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "SEC": "U", "PYL": "O", "MSE": "M",  # common modified residues
}

_model = None
_batch_converter = None
_device = None


def load_model():
    global _model, _batch_converter, _device
    if _model is not None:
        return _model, _batch_converter, _device
    import esm

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    model, alphabet = getattr(esm.pretrained, ESM2_MODEL_NAME)()
    model = model.to(_device).eval()
    _model, _batch_converter = model, alphabet.get_batch_converter()
    return _model, _batch_converter, _device


def residues_to_sequence(residues: list) -> str:
    """residues: model/features.py's group_residues() output,
    [((chain, seq_id, comp_id), [Atom, ...]), ...]."""
    return "".join(THREE_TO_ONE.get(key[2], "X") for key, _ in residues)


@torch.no_grad()
def compute_esm2_embeddings(pdb_id: str, residues: list | None = None) -> torch.Tensor | None:
    """Returns Tensor[N, ESM2_DIM] aligned 1:1 with `residues` (or with
    model/features.py's own residue ordering for this pdb_id if not provided). ESM2 has
    a max practical context length; sequences longer than 1022 residues are truncated
    (rare for a single antigen chain, but not impossible) -- truncated positions get a
    zero embedding rather than crashing, flagged via the returned mask being all-True
    only up to the truncation point."""
    if residues is None:
        chains = get_chain_atoms(pdb_id)
        if chains is None:
            return None
        _, antigen_atoms = chains
        if not antigen_atoms:
            return None
        residues = group_residues(antigen_atoms)

    seq = residues_to_sequence(residues)
    if not seq:
        return None

    model, batch_converter, device = load_model()
    max_len = 1022  # ESM2's practical limit minus BOS/EOS
    seq_used = seq[:max_len]

    _, _, tokens = batch_converter([("antigen", seq_used)])
    tokens = tokens.to(device)
    out = model(tokens, repr_layers=[ESM2_LAYER])
    # out['representations'][layer] shape: [1, seq_len+2, dim]; drop BOS (pos 0) and EOS.
    emb = out["representations"][ESM2_LAYER][0, 1:1 + len(seq_used), :].cpu()

    if len(seq_used) < len(seq):
        pad = torch.zeros(len(seq) - len(seq_used), ESM2_DIM)
        emb = torch.cat([emb, pad], dim=0)
    return emb


class Projection(torch.nn.Module):
    """Learned projection from ESM2_DIM down to PROJECTED_DIM before concatenation
    with Model A's geometric features (4-dim) -- without this, the ESM2 embedding
    would dominate the much lower-dimensional geometric features by sheer size
    (PLAN.md sec 4)."""

    def __init__(self, in_dim: int = ESM2_DIM, out_dim: int = PROJECTED_DIM):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x))


if __name__ == "__main__":
    pdb_id = sys.argv[1] if len(sys.argv) > 1 else "pdb_00001a14"
    chains = get_chain_atoms(pdb_id)
    if chains is None:
        print(f"{pdb_id}: no usable antigen")
    else:
        _, antigen_atoms = chains
        residues = group_residues(antigen_atoms)
        seq = residues_to_sequence(residues)
        print(f"{pdb_id}: {len(residues)} residues, sequence: {seq[:60]}{'...' if len(seq) > 60 else ''}")
        emb = compute_esm2_embeddings(pdb_id, residues)
        print(f"ESM2 embedding shape: {tuple(emb.shape)}, mean={emb.mean().item():.4f}")
