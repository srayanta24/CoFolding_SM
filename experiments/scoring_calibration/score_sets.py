#!/usr/bin/env python3
"""Score the calibration manifest by cofolding each entry (PLAN.md milestone 3).

Reads `sets/pilot.tsv` (or any manifest produced by build_sets.py), cofolds each
antibody-antigen pair through Boltz-2, and collects the confidence metrics into
`results/scores.tsv` — one row per entry, in the same metric space as a BoltzGen
campaign's `all_designs_metrics.csv`.

Two deliberate choices, both about comparability rather than convenience
------------------------------------------------------------------------
**Boltz-2, not OpenFold3.** BoltzGen's own folding/refold stage loads a Boltz-2
derived checkpoint (`boltz2_conf_final.ckpt`, BOLTZGEN_PIPELINE.md sec 4), so
Boltz-2 is the backend whose ipTM is actually comparable to a campaign's
`design_to_target_iptm`. OpenFold3 would be a second independent opinion, which is
a different and later question.

**No MSA** (`--no-msa`, added to run_design.py for this). BoltzGen refolds designs
with the target template and no MSAs (paper sec 3.5). Scoring real complexes *with*
an MSA would hand them an information source the designs never get, inflating the
reference and defeating the point of the calibration. It is also much faster and
avoids 300+ remote MSA-server queries.

Resumability
------------
Results are appended and flushed per entry, and a rerun skips entry_ids already
present in `results/scores.tsv`. A multi-hour run that dies partway can simply be
restarted. Per-entry wall-clock is recorded because PLAN.md's 3-5 min/complex figure
is an estimate from the BoltzGen paper's timing figure, never measured on this box —
measuring it is one of the pilot's deliverables.

Scope cut for the pilot, stated rather than left implicit
---------------------------------------------------------
This collects the *confidence* metrics only (ipTM, pTM, pLDDT, confidence_score).
It does not compute PLIP hydrogen bonds / salt bridges or delta-SASA, which are the
other half of BoltzGen's ranking inputs. Those need the analysis-stage tooling and
are only required for the filter-rule work (milestone 6, itself conditional). All
three of the pilot's primary questions — calibration percentiles, the memorisation
gap, and ipTM-vs-log(Kd) dose-response — turn on ipTM, so the pilot can answer them
without that machinery. The predicted structures are kept on disk so the contact
metrics can be added later without re-running any GPU work.

Usage
-----
    python3 experiments/scoring_calibration/score_sets.py --limit 2      # smoke test
    python3 experiments/scoring_calibration/score_sets.py                # full pilot
    python3 experiments/scoring_calibration/score_sets.py --manifest sets/manifest.tsv
"""

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_design import build_boltz_config, run_boltz2  # noqa: E402

SCORES_DIR = HERE / ".scores"
RESULTS_DIR = HERE / "results"
RESULTS_TSV = RESULTS_DIR / "scores.tsv"

# Carried through from the manifest so the results file is self-contained for
# analysis — no join back to the manifest needed.
# Superset of label columns across both benchmarks; missing ones come through blank,
# so one scorer serves the EXPERIMENT_01 manifest and the ASD within-antigen manifest
# without the two needing identical schemas.
CARRY = ["entry_id", "set", "source", "pdb", "split", "label",
         "kd_molar", "ddg_kcal", "elisa_ratio", "n_res_total",
         "target_pdb", "neg_log_kd", "affinity_decile"]
METRICS = ["confidence_score", "ptm", "iptm", "complex_plddt", "complex_iplddt",
           "complex_pde", "complex_ipde", "ligand_iptm", "protein_iptm"]
OUT_FIELDS = CARRY + METRICS + ["status", "seconds", "cif_path"]


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def already_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as fh:
        return {r["entry_id"] for r in csv.DictReader(fh, delimiter="\t")
                if r.get("entry_id")}


def collect_confidence(out_dir: Path) -> tuple[dict, str]:
    """Pull Boltz-2's confidence JSON and the predicted structure path.

    Boltz-2 writes confidence_<name>_model_0.json next to the CIF. Every key is
    kept in the raw sidecar; only the ones in METRICS become columns.
    """
    conf_files = sorted(out_dir.rglob("confidence_*.json"))
    cifs = sorted(out_dir.rglob("*.cif"))
    if not conf_files:
        return {}, (str(cifs[0]) if cifs else "")
    data = json.loads(conf_files[0].read_text())
    return data, (str(cifs[0]) if cifs else "")


def score_entry(row: dict, keep_structures: bool) -> dict:
    entry_id = row["entry_id"]
    work = SCORES_DIR / entry_id
    work.mkdir(parents=True, exist_ok=True)
    out = {k: row.get(k, "") for k in CARRY}

    t0 = time.time()
    try:
        cfg = build_boltz_config(
            target_seq=row["antigen_seq"],
            modality="antibody",
            partner_value=f"{row['heavy_seq']},{row['light_seq']}",
            out_path=work / "config.yaml",
            use_msa=False,
        )
        ok = run_boltz2(cfg, work / "boltz2", use_msa=False)
        data, cif = collect_confidence(work / "boltz2")
        for m in METRICS:
            v = data.get(m)
            out[m] = f"{v:.6f}" if isinstance(v, (int, float)) else ""
        out["status"] = "ok" if (ok and data) else ("no_metrics" if ok else "failed")
        out["cif_path"] = cif if keep_structures else ""
        if data:
            (work / "confidence_raw.json").write_text(json.dumps(data, indent=2))
    except Exception:
        for m in METRICS:
            out.setdefault(m, "")
        out["status"] = "error"
        out["cif_path"] = ""
        (work / "error.txt").write_text(traceback.format_exc())
    out["seconds"] = f"{time.time() - t0:.1f}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=str(HERE / "sets" / "pilot.tsv"))
    ap.add_argument("--out", default=None,
                    help="results TSV (default results/scores.tsv); use a separate "
                         "file for a different benchmark so the two never mix")
    ap.add_argument("--limit", type=int, default=None,
                    help="score at most N not-yet-done entries (smoke test)")
    ap.add_argument("--sets", default=None,
                    help="comma-separated set names to restrict to")
    ap.add_argument("--no-keep-structures", action="store_true",
                    help="don't record CIF paths (structures are still written)")
    args = ap.parse_args()

    rows = read_manifest(Path(args.manifest))
    if args.sets:
        want = {s.strip() for s in args.sets.split(",")}
        rows = [r for r in rows if r["set"] in want]

    results_tsv = Path(args.out) if args.out else RESULTS_TSV
    done = already_done(results_tsv)
    todo = [r for r in rows if r["entry_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]

    results_tsv.parent.mkdir(parents=True, exist_ok=True)
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not results_tsv.exists()

    # Guard against appending rows whose field list differs from the existing
    # header. This happened for real: CARRY was later extended with the ASD label
    # columns, and appending the wider rows to a file written with the narrower
    # header silently shifted every column after the 10th by three. The data was
    # intact but unparseable, so `status` read blank and every downstream scorer
    # correctly concluded there was nothing to do -- a failure that looks exactly
    # like success in the logs. Fail loudly instead.
    if not new_file:
        with results_tsv.open(encoding="utf-8") as fh:
            existing = fh.readline().rstrip("\n").split("\t")
        if existing != OUT_FIELDS:
            raise SystemExit(
                f"[score_sets] REFUSING TO APPEND: {results_tsv} has a {len(existing)}-"
                f"column header but this build writes {len(OUT_FIELDS)} columns.\n"
                f"  missing from file: {[c for c in OUT_FIELDS if c not in existing]}\n"
                f"  extra in file:     {[c for c in existing if c not in OUT_FIELDS]}\n"
                f"  Rewrite the file with the current schema, or write to a new --out.")

    print(f"[score_sets] manifest {args.manifest}: {len(rows)} entries, "
          f"{len(done)} already done, {len(todo)} to score", flush=True)

    t_start = time.time()
    with results_tsv.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_FIELDS, delimiter="\t",
                           extrasaction="ignore", lineterminator="\n")
        if new_file:
            w.writeheader()
            fh.flush()
        for i, row in enumerate(todo, 1):
            res = score_entry(row, keep_structures=not args.no_keep_structures)
            w.writerow(res)
            fh.flush()  # resumability: never lose more than the in-flight entry
            elapsed = time.time() - t_start
            rate = elapsed / i
            print(f"[{i}/{len(todo)}] {res['entry_id']:<44} {res['status']:<10} "
                  f"iptm={res.get('iptm') or '-':<10} {res['seconds']:>7}s "
                  f"| avg {rate/60:.1f} min/entry, ETA {(len(todo)-i)*rate/3600:.1f}h",
                  flush=True)

    print(f"[score_sets] done in {(time.time()-t_start)/3600:.2f}h -> {results_tsv}")


if __name__ == "__main__":
    main()
