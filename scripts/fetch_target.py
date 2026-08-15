#!/usr/bin/env python3
"""Fetch a protein sequence given a UniProt accession or PDB ID.

Auto-detects which source to use from the identifier's shape:
  - UniProt accession, e.g. P69905, Q9Y2K2-1  -> rest.uniprot.org
  - PDB ID, e.g. 1CRN, 7XYZ                    -> files.rcsb.org

Usage:
    python3 scripts/fetch_target.py P69905
    python3 scripts/fetch_target.py 1CRN --chain A
    python3 scripts/fetch_target.py P69905 --out target.fasta
"""

import argparse
import re
import sys
import urllib.error
import urllib.request

UNIPROT_RE = re.compile(
    r"^([A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}|[OPQ][0-9][A-Z0-9]{3}[0-9])(-\d+)?$",
    re.IGNORECASE,
)
PDB_RE = re.compile(r"^[1-9][A-Za-z0-9]{3}$")


def identify_source(identifier: str) -> str:
    if PDB_RE.match(identifier):
        return "pdb"
    if UNIPROT_RE.match(identifier):
        return "uniprot"
    raise ValueError(
        f"'{identifier}' doesn't look like a UniProt accession (e.g. P69905) "
        f"or a PDB ID (e.g. 1CRN)"
    )


def fetch(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.read().decode()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} fetching {url}") from e


def parse_fasta(text: str) -> list[tuple[str, str]]:
    """Returns [(header, sequence), ...] — a PDB FASTA can have multiple chains."""
    entries = []
    header, seq_lines = None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                entries.append((header, "".join(seq_lines)))
            header, seq_lines = line[1:].strip(), []
        elif line.strip():
            seq_lines.append(line.strip())
    if header is not None:
        entries.append((header, "".join(seq_lines)))
    return entries


def fetch_uniprot(accession: str) -> list[tuple[str, str]]:
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    return parse_fasta(fetch(url))


def fetch_pdb(pdb_id: str) -> list[tuple[str, str]]:
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id.upper()}"
    return parse_fasta(fetch(url))


def select_chain(entries: list[tuple[str, str]], chain: str | None) -> tuple[str, str]:
    if not entries:
        raise RuntimeError("no sequences returned")
    if chain is None:
        # Default: longest chain (usually the protein of interest, not a
        # short cofactor/peptide chain also present in the entry).
        return max(entries, key=lambda e: len(e[1]))
    for header, seq in entries:
        if f"Chain {chain}" in header or f"Chains {chain}" in header or f"|{chain}|" in header:
            return header, seq
    raise ValueError(
        f"chain '{chain}' not found; available headers: {[h for h, _ in entries]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("identifier", help="UniProt accession or PDB ID")
    parser.add_argument("--chain", help="For PDB entries with multiple chains, pick one (default: longest)")
    parser.add_argument("--out", help="Write raw sequence (no header) to this file; default: stdout")
    args = parser.parse_args()

    source = identify_source(args.identifier)
    print(f"[fetch_target] '{args.identifier}' identified as {source}", file=sys.stderr)

    entries = fetch_uniprot(args.identifier) if source == "uniprot" else fetch_pdb(args.identifier)
    header, sequence = select_chain(entries, args.chain)
    print(f"[fetch_target] using: {header} ({len(sequence)} residues)", file=sys.stderr)

    if args.out:
        with open(args.out, "w") as f:
            f.write(sequence + "\n")
        print(f"[fetch_target] wrote sequence to {args.out}", file=sys.stderr)
    else:
        print(sequence)


if __name__ == "__main__":
    main()
