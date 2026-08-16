"""Curated real (experimentally known) antibody-antigen complexes, used as a
known-binder calibration baseline in experiments/score_reference.py.

Each entry must be independently fetch-verified before being added — don't extend this
list by guessing PDB IDs or chain assignments. fetch_target.select_chain() (reused by
fetch_reference.py) matches the FASTA header's *label* chain id, not the *auth* id in
brackets (e.g. RCSB reports "Chain C[auth Y]" — use C, not Y). Get this wrong and you
silently fetch the wrong chain with no error.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceComplex:
    name: str
    antigen_pdb: str
    antigen_chain: str  # label chain id, not auth
    heavy_pdb: str
    heavy_chain: str
    light_pdb: str
    light_chain: str
    notes: str


REFERENCE_COMPLEXES: dict[str, ReferenceComplex] = {
    "hel": ReferenceComplex(
        name="hel",
        antigen_pdb="1FDL", antigen_chain="C",
        heavy_pdb="1FDL", heavy_chain="B",
        light_pdb="1FDL", light_chain="A",
        notes=(
            "D1.3 anti-HEL Fab bound to hen egg white lysozyme. Chosen because "
            "hel_antibody is an existing campaign against the same antigen (PDB 1DPX). "
            "Verified 2026-08-15: `curl https://www.rcsb.org/fasta/entry/1FDL` returns "
            "exactly three chains — 'Chain C[auth Y]|HEN EGG WHITE LYSOZYME', "
            "'Chain B[auth H]|...HEAVY CHAIN', 'Chain A[auth L]|...LIGHT CHAIN' — "
            "label ids C/B/A used here, not auth ids Y/H/L."
        ),
    ),
}
