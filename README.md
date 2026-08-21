# co-scientist: cofolding + antibody design on a single GPU box

This repo is a working AI co-scientist for structural biology: it can **predict
how a protein complex folds** (cofolding) and **generate novel antibody
candidates against a target** (de novo binder design), running entirely on a
single local GPU (an NVIDIA DGX Spark / GB10, aarch64). Everything here was
built and verified end-to-end on that hardware — nothing is aspirational.

This README covers both the *science* (what these models are actually doing,
in enough depth to reason about the outputs) and the *mechanics* (how to
actually run something). If you just want to run a design, jump to
[Quickstart](#quickstart). If you want to understand what's happening under
the hood, read [How cofolding works](#how-cofolding-works) and
[How antibody design works](#how-antibody-design-works) first.

For the full design rationale, hardware-compatibility findings, and every
workaround discovered along the way, see [DESIGN.md](DESIGN.md) (cofolding),
[UI_DESIGN.md](UI_DESIGN.md) (a not-yet-built dashboard layer),
[BOLTZGEN_PIPELINE.md](BOLTZGEN_PIPELINE.md) (exactly how BoltzGen's design
pipeline works internally — architecture, checkpoints, and config knobs per
stage, written for improving generation quality rather than just running it),
and [IMPROVE_DESIGN.md](IMPROVE_DESIGN.md) (the active plan for improving the
design step: epitope prediction, a Germinal comparison, and fine-tuning, in
that order — epitope prediction is now built and validated, see
[experiments/epitope_prediction/PLAN.md](experiments/epitope_prediction/PLAN.md)).
This README is the practical entry point; those docs are the detailed record.

---

## What's actually in here

Two distinct capabilities, backed by different models, invoked through one
Claude Code skill (`.claude/skills/design/SKILL.md`):

| Capability | Question it answers | Backend(s) | Typical runtime |
|---|---|---|---|
| **A. Cofolding** | "If protein X and Y/this small molecule/this nucleic acid are put together, what structure do they form, and how confident is that prediction?" | Boltz-2, OpenFold3 | 1–3 min |
| **B. Antibody/binder design** | "Generate novel antibody candidates that might bind target X" | BoltzGen | tens of minutes to hours |

These are genuinely different problems. (A) evaluates a *given* pair — you
supply both sides. (B) *generates* something new — you supply only the
target, and the model proposes binders. A prior session in this project
initially conflated the two ("design an antibody against lysozyme" could
mean either), so this README keeps them clearly separate throughout.

Three more Claude Code skills support the antibody-design *improvement* work
(building/using training data and a steering signal, not running Skill A/B
themselves):

| Skill | Purpose |
|---|---|
| [`.claude/skills/database/SKILL.md`](.claude/skills/database/SKILL.md) | Create/recreate the local antibody-design database (`databases/` — SAbDab, AACDB, AB-Bind, ANDD, AbDesign DB, ASD) |
| [`.claude/skills/splits/SKILL.md`](.claude/skills/splits/SKILL.md) | Build the leak-free train/dev/test splits (`databases/splits/`) used to evaluate anything trained on that data |
| [`.claude/skills/epitope_guided_design/SKILL.md`](.claude/skills/epitope_guided_design/SKILL.md) | Predict a target's epitope and steer Skill B's diffusion design toward it via BoltzGen's `binding_types` conditioning (`experiments/epitope_prediction/`) |

---

## Quickstart

```bash
# One-time setup: clones each backend's source, creates isolated venvs,
# installs CUDA torch, sets up the permanent weights/ folder.
python3 scripts/setup_env.py

# Skill A — cofold a target (fetched by ID) against a small molecule
python3 scripts/run_design.py \
  --target-id P69905 \
  --partner-modality small_molecule --partner-value "CC(=O)Oc1ccccc1C(=O)O" \
  --name my_first_design --backend both

# Skill B — generate novel antibodies against a target, starting from
# nothing but a sequence (predicts the target's structure first, then
# designs against that prediction — see "How antibody design works" below)
python3 scripts/design_binder.py --target-id P09758 --name my_target
# ^ validates and prints the launch command; add --launch to run immediately

# Skill B, alternative — if you already have a real (experimental)
# target structure, skip the prediction step and use it directly:
cd data/designs/my_target && source ../../../.venvs/boltzgen/bin/activate
boltzgen check my_target.yaml --cache ../../../weights/boltzgen
boltzgen run my_target.yaml --output run1 --protocol antibody-anything \
  --num_designs 50 --budget 5 --num_workers 0 --cache ../../../weights/boltzgen
```

`setup_env.py` takes a while the first time (source clones + ~15GB of model
weights across four backends). Everything after that is fast to re-run;
weights and source are cached permanently under `weights/` and `src/`, not
in some OS temp/cache directory that could get cleared.

---

## Hardware & environment (read this if something doesn't install)

This was built on an **NVIDIA DGX Spark (GB10, "Grace Blackwell")** —
aarch64 CPU, a single Blackwell-class GPU (**CUDA compute capability
sm_121**), ~119GB unified CPU/GPU memory, CUDA 13.0. Two things about that
matter a lot for anyone trying to reproduce this on different hardware:

1. **aarch64 is not x86_64.** Most bio-ML tooling assumes x86_64 — Docker
   images, prebuilt wheels for compiled CUDA extensions
   (`cuequivariance-ops-torch`, DGL, etc.) are frequently x86_64-only or lag
   behind on aarch64. Every backend here needed at least a smoke test before
   being trusted; two (Boltz-2, BoltzGen) needed a documented workaround for
   a missing aarch64 kernel package, and one (Protenix) is **currently
   non-functional** on this exact GPU for a different reason (below).
2. **sm_121 is very new.** The installed PyTorch build's own
   `torch.cuda.get_arch_list()` tops out at sm_120 — one generation behind
   this GPU. Most operations still work fine via PTX forward-compatibility
   JIT, but Protenix hits at least one op that doesn't have a compatible
   kernel path at all (`CUDA error: no kernel image is available for
   execution on the device`). This is why **Protenix is excluded** from
   both skills below — it installs fine, it just can't run inference here.

If you're on a different GPU (especially anything x86_64, or an older/more
common compute capability like sm_80–sm_90), some of this may just work
without the workarounds documented here — but verify with
`scripts/smoke_test.py` before trusting it, don't assume.

### Isolated environments, not one shared venv

Each backend lives in its own venv under `.venvs/` (`boltz2`, `openfold3`,
`protenix`, `boltzgen`), not a shared environment. This isn't just tidiness
— Protenix pins `torch==2.3.1` exactly, a version with **no aarch64 CUDA
wheel at all**. Installing it into a shared env with the others silently
downgrades everyone's torch and kills GPU support project-wide. Each venv
gets its own `pip install torch --index-url https://download.pytorch.org/whl/cu130`
before its backend goes in.

### Source, not opaque packages

`src/{boltz,openfold3,protenix,boltzgen}` are shallow git clones of each
project's real repository, installed **editable** (`pip install -e`) into
their venv. This means the code actually executing is sitting in this repo,
readable and patchable, not hidden inside `site-packages`. One gotcha if
you ever redo this yourself: switching an *already pip-installed* package to
editable can leave a stale non-editable copy behind in `site-packages` that
silently shadows the new editable one (imports resolve to the wrong place,
or a submodule 404s). `setup_env.py` clears this automatically now, but it's
worth knowing if you ever see an import behave strangely after a reinstall.

### Weights live permanently in `weights/`, not a cache

By default these four tools each pick their own cache location
(`~/.cache/huggingface`, `~/.openfold3`, a bare `~/checkpoint`, etc.) —
scattered, and semantically a "cache" that some cleanup tool could
legitimately delete. Everything here is redirected into `weights/<backend>/`
in this repo instead, treated as permanent data. Boltz-2 and OpenFold3 have
real CLI flags for this (`--cache`, `--inference-ckpt-path`); Protenix
doesn't, so it gets a narrow symlink (`weights/protenix_root/checkpoint -> ../protenix`)
that satisfies its `$PROTENIX_ROOT_DIR` env var without pointing that
variable at anything broader (it also controls unrelated data directories).

---

## How cofolding works

**The problem**: given two or more molecular entities — protein chains, a
small molecule, a nucleic acid strand — predict the 3D structure of the
*complex* they form together, not just each piece in isolation. This is
harder than single-protein structure prediction (what the original
AlphaFold2 did) because it requires reasoning about the *interface*: which
residues contact which, how the binding pose is constrained by both
partners simultaneously.

**How Boltz-2 and OpenFold3 actually do this** (both are open
reimplementations in the lineage of AlphaFold3's approach):

1. **Input featurization.** Each chain gets tokenized (per-residue for
   protein/RNA/DNA, per-atom for small molecules). If MSA (multiple sequence
   alignment) is enabled, each protein chain's sequence is also searched
   against a large sequence database (via a remote ColabFold/MMseqs2-style
   server in this setup, not a local database) to find evolutionarily
   related sequences — co-evolving positions across related sequences are a
   strong signal for which residues are physically close in 3D, a technique
   that predates AlphaFold3 but is still folded in as one input signal
   among several.
2. **Trunk (representation learning).** A deep transformer-like network (an
   "Evoformer"-descended architecture, called the Pairformer in the
   AlphaFold3 lineage) iteratively refines two representations: a per-token
   embedding and a per-token-*pair* embedding capturing relationships
   between every pair of tokens across *all* chains — this is precisely the
   mechanism that lets the model reason about the interface, since
   cross-chain pairs get the same treatment as within-chain pairs.
3. **Structure generation via diffusion.** Rather than directly regressing
   3D coordinates (the original AlphaFold2 approach), these models use a
   diffusion process: start from random noise in 3D space and iteratively
   denoise toward a structure consistent with the trunk's representation,
   conditioned at each step. This is the same class of generative process
   as image diffusion models, applied to atomic coordinates. Running this
   multiple times with different random seeds gives an ensemble of
   candidate structures rather than one deterministic answer.
4. **Confidence heads.** Alongside the structure, the model predicts its
   own confidence: per-token **pLDDT** (predicted Local Distance Difference
   Test — how reliable each individual residue/atom's position is), a
   **PAE** matrix (Predicted Aligned Error — for every pair of tokens, how
   much their *relative* position/orientation might be wrong, which is the
   metric that actually matters for interface quality, since two domains
   can each be individually confident (`high pLDDT`) while being
   incorrectly *positioned relative to each other*), and summary scores
   **pTM** and **ipTM** (predicted TM-score for the whole structure, and
   specifically for the interface between chains).

**Reading the output metrics** (from `run_design.py`'s printed summary):

- **ipTM** is the single most important number for "does this complex form
  a real interface." Above ~0.8 suggests a confident, likely-correct
  interface. Below ~0.5 means don't trust the binding pose at all — the
  model is essentially guessing. The 0.3–0.6 range (where most of the
  antibody design validations in this project land) is genuinely
  ambiguous and shouldn't be over-interpreted from a single seed.
- **pTM** describes confidence in the *overall fold*, not specifically the
  interface — a high pTM with a low ipTM means "each chain individually
  folds sensibly, but I'm not confident about how they're arranged
  relative to each other."
- Boltz-2 and OpenFold3 report these on **different scales/conventions**
  in places (see the confidence-summary keys `run_design.py` prints per
  backend) — **never average a Boltz-2 score with an OpenFold3 score.**
  Report them side by side as two independent opinions, which is exactly
  the point of running both.
- A caveat worth taking seriously (from the model survey that shaped this
  project, DESIGN.md's original research): published benchmark accuracy
  for these cofolding models (PoseBusters-style pass rates in the
  75–85% range) is measured on targets similar to what the models were
  trained on. Independent analysis (the "Runs N' Poses" benchmark, and a
  2025 bioRxiv paper questioning whether these methods have "moved beyond
  memorisation") found accuracy drops sharply for genuinely novel
  chemotypes or binding sites unlike anything in the training set. Treat a
  single confident-looking prediction on a truly novel target with
  appropriate skepticism — multi-seed ensembles and, where possible,
  orthogonal validation (crystallography, other computational methods)
  matter more than the model's own confidence score suggests.

**Why two backends instead of one**: OpenFold3 is the more thoroughly
validated, fully-open-stack option (data + weights + training code all
public — the only one of the three original candidates with that property).
Boltz-2 is faster and has a simpler single-file config format. Running both
gives two independently-trained opinions on the same question, which is
more informative than trusting either alone, especially given the
memorization caveat above.

---

## Skill A: cofold a target against a partner (technical)

**Script**: `scripts/run_design.py`, using `scripts/fetch_target.py` for
sequence retrieval and `scripts/_common.py` for shared paths/helpers.

### Step by step

1. **Get the target sequence.** Either supply it directly (`--target-seq`)
   or give an identifier and let the tool fetch it:
   ```bash
   python3 scripts/fetch_target.py P69905      # UniProt accession
   python3 scripts/fetch_target.py 1CRN         # PDB ID
   ```
   The identifier format is auto-detected (UniProt accessions and PDB IDs
   have distinct, non-overlapping shapes) — you don't need to say which
   kind you're giving it. For a PDB entry with multiple chains, the longest
   chain is used by default (`--chain` to override).

2. **Describe the partner.** One of:
   - `small_molecule` + a SMILES string
   - `protein` + a sequence
   - `antibody` + a sequence, **or two sequences separated by a comma**
     (`"HEAVY_SEQ,LIGHT_SEQ"`) for a real two-chain Fab paratope — a single
     chain is only a meaningful approximation for a nanobody/VHH, since a
     conventional antibody's binding site is formed by both chains
     together
   - `peptide` + a sequence
   - `rna` / `dna` + a sequence

   `antibody` and `peptide` have **no CDR-specific or other specialized
   handling** in either backend — they're modeled as plain protein chain(s).
   This is a real limitation to be upfront about: the model isn't "aware"
   it's looking at an antibody, it just sees another protein sequence.

3. **Run it.**
   ```bash
   python3 scripts/run_design.py \
     --target-id <UniProt-or-PDB-ID>   `# or --target-seq <sequence>` \
     --partner-modality <modality> \
     --partner-value <SMILES-or-sequence[,second-sequence]> \
     --name <job-name> \
     --backend <boltz2|openfold3|both>
   ```
   This builds a native config for each requested backend (Boltz-2's
   single-YAML format, OpenFold3's JSON query-set format — both written to
   `data/designs/<name>/`), runs each with the flags known to actually work
   on this hardware (documented in the script's own docstrings — a
   `--num_workers 0` deadlock workaround for Boltz-2, an explicit
   checkpoint path for OpenFold3), and reports structure file paths +
   confidence metrics.

4. **Look at the output.** Structures land in
   `data/designs/<name>/<backend>/.../*.cif`. No visualization is wired up
   in this repo yet (see UI_DESIGN.md for a planned dashboard) — open the
   `.cif` in PyMOL, ChimeraX, or [molstar.org/viewer](https://molstar.org/viewer)
   (no install needed) in the meantime.

### Worked example (from this project's own runs)

Crambin (PDB `1CRN`, 46 residues) cofolded against aspirin via Boltz-2:
`confidence_score=0.87, ptm=0.92, iptm=0.78` in about 10 seconds of actual
inference — high confidence across the board, as expected for a small,
well-behaved test case.

---

## How antibody design works

**The problem**: rather than evaluating a given antibody, *generate a new
one* likely to bind a specified target — starting from nothing but the
target's structure (and optionally which surface patch to bind).

This is a fundamentally different kind of model than cofolding. Cofolding
answers "given both molecules, what's the structure" — a discriminative /
generative-conditional-on-both-inputs problem. Antibody design answers
"given only the target, propose a molecule" — genuinely generative,
open-ended, and the reason it needs a different tool (BoltzGen) entirely.
(Boltz-2 and OpenFold3 cannot do this — they have nothing to generate,
since they always require both binding partners as input.)

### The five-stage pipeline

BoltzGen's `boltzgen run` executes five stages in sequence for every
campaign:

1. **Design (backbone generation).** A diffusion model — architecturally
   related to the cofolding trunk/diffusion approach above, but run in
   "generate" mode rather than "predict" mode — produces novel 3D
   backbones (the protein chain trace, without side chains yet) positioned
   against the target structure. For antibody design specifically, this
   isn't generating an entire antibody from scratch: it starts from a real,
   solved antibody Fab framework (see "Fab scaffold grafting" below) and
   only redesigns the CDR loops — the hypervariable regions that actually
   contact the antigen — while holding the rest of the framework fixed.
   This is the step that takes the vast majority of the wall-clock time
   (~25–40 min for 50 designs on this hardware).
2. **Inverse folding (sequence design).** Given a 3D backbone, decide
   *which amino acid* goes at each position — the inverse of the structure
   prediction problem (sequence → structure becomes structure → sequence).
   A separate model handles this, since backbone geometry alone doesn't
   fully determine the optimal sequence.
3. **Folding (self-consistency check).** The newly-designed sequence gets
   run back through a cofolding model (BoltzGen bundles a Boltz-2-derived
   checkpoint for this) *from scratch*, with the target — i.e., "if I
   didn't know this sequence was designed to fold this way, would a
   cofolding model independently agree that it does?" This is the internal
   analog of the external validation step in Skill A, but automated as
   part of the pipeline: designs whose refolded structure looks nothing
   like what was intended get penalized in the ranking that follows.
4. **Analysis.** Computes the full metrics table per design: interface
   pTM/ipTM against the target (from the refolding step), RMSD between the
   originally-designed structure and its independent refold (a
   self-consistency measure — large RMSD means the design step and the
   refold step disagree, a red flag), interface contact counts
   (hydrogen bonds, salt bridges via PLIP), amino-acid composition
   checks, and a battery of **developability liability** checks (oxidation-
   prone residues, protease cleavage motifs, deamidation sites,
   hydrophobic patches) — the kind of thing that matters for whether a
   candidate could ever actually become a manufacturable therapeutic, not
   just whether it binds in silico.
5. **Filtering.** Ranks all generated designs by a combination of quality
   metrics and sequence diversity (so the final set isn't just N near-
   duplicates of the single best design), producing the final ranked
   output.

### Fab scaffold grafting

Rather than generating an antibody from a blank slate, antibody-mode
BoltzGen designs conditioned on one of 14 real, solved antibody Fab
structures bundled in this repo at
`data/boltzgen_examples/repo/example/fab_scaffolds/` — real therapeutic
antibodies (adalimumab, belimumab, ustekinumab, and others), each a
solved crystal structure with its CDR loop positions annotated for
redesign. The model treats the framework (~90% of the antibody) as fixed
and only regenerates the CDR loops — this dramatically constrains the
design space to sequences that are more likely to actually fold as a
stable antibody (since the framework is empirically known to be
stable/expressible), at the cost of being unable to explore framework-level
novelty. A `nanobody_scaffolds/` directory exists in the same location for
single-domain (VHH) design via `--protocol nanobody-anything`.

### What "the target" actually requires

Unlike Skill A, this needs the target's **real 3D structure** — a
PDB/CIF file with actual atomic coordinates — not just a sequence, because
the design step conditions the diffusion process on the target's physical
shape (and optionally a specific epitope). Fetch one from RCSB PDB
(`curl -s https://files.rcsb.org/download/<PDB_ID>.pdb -o ...`), preferring
a well-resolved apo (unbound) structure unless you specifically want to
condition on a known epitope from an existing complex structure.

**One indexing gotcha worth knowing before you build a spec by hand**:
residue indices in BoltzGen's YAML spec format are the mmCIF
`label_seq_id`, **not** the PDB file's author-assigned residue numbers,
which frequently differ (crystallographers renumber for all sorts of
reasons — a different numbering convention, missing residues, etc.). Check
indexing in a viewer (molstar.org/viewer shows both when you hover a
residue) before assuming a residue number from a paper maps directly into
a spec file.

### Reading the output metrics

The per-design metrics table (`final_designs_metrics_<budget>.csv`)
includes, among many columns:

- **`design_to_target_iptm`** — same interface-confidence concept as
  Skill A's ipTM, computed from the internal refolding self-consistency
  check. This is the primary "does this look like it binds" signal.
- **`filter_rmsd`** — self-consistency RMSD between the original design
  and its independent refold. Low (under ~2–3 Å) is good; the two
  independent stages of the pipeline agree. High values (seen up to ~18Å
  in some of this project's own lower-ranked results) mean treat that
  design's ranking with real skepticism even if other metrics look
  passable.
- **`plip_hbonds_refolded`** — predicted hydrogen bonds at the interface,
  from the refolded structure. More isn't strictly better, but zero is a
  bad sign.
- **`liability_num_violations`** and the various `liability_*` columns —
  developability red flags (oxidation-prone tryptophan/methionine,
  protease cleavage motifs, asparagine deamidation, hydrophobic patches).
  This is the column to scrutinize first before taking any candidate
  further — a design can look great on binding metrics and still be a
  poor real-world candidate if it's riddled with liabilities.

**The most important caveat, worth repeating explicitly**: everything this
pipeline produces is a **computational prediction of a novel candidate**,
scored by the model's own (self-consistency-checked, but still
model-generated) confidence. It is not a validated binder. Real confidence
requires either external validation — chain a top design into Skill A
against a cofolding backend it wasn't scored by, which this project already
does as a cross-check — or, ultimately, wet-lab testing. Campaign hit rates
in practice are typically well under 100%; BoltzGen's own documentation
notes design campaigns commonly need 10,000–60,000 generated candidates to
find real hits, far more than the 50-design smoke-test scale used in this
project's example runs.

---

## Skill B: antibody/binder design (technical)

**Tooling**: `scripts/design_binder.py` (sequence-only entry point, wraps
everything below) or the `boltzgen` CLI directly (if you already have an
experimental target structure and want to skip the prediction step).

### Path 1 — sequence only (no experimental structure needed)

```bash
python3 scripts/design_binder.py \
  --target-id <UniProt-or-PDB-ID>   `# or --target-seq <raw sequence>` \
  --name <job-name> \
  --protocol antibody-anything      `# or nanobody-anything for a single-domain binder`
```

Under the hood this runs `scripts/predict_structure.py` (folds the target
alone via Boltz-2 — no partner, just a single-chain structure prediction),
auto-generates the design spec referencing the predicted structure and the
bundled scaffolds, and runs `boltzgen check` to validate it. By default it
stops there and prints the exact `boltzgen run` command to launch the real
campaign — pass `--launch` (with `--num_designs`/`--budget` if you want
something other than the 50/5 smoke-test default) to run it immediately
instead.

**This stacks two layers of prediction** — the target fold is itself
predicted, not experimental, and the binder is then designed against that
prediction. `design_binder.py` prints the predicted structure's average
pLDDT specifically so this is visible rather than hidden: above ~85 (e.g.
Crambin folded at 95.1 in this project's own test) is trustworthy to design
against; below ~70 triggers an explicit warning, usually meaning part of
the sequence doesn't fold into an ordered domain by itself (a signal
peptide, a transmembrane segment, a disordered region) — if you know the
folded domain's boundaries, re-run with just that subsequence. Prefer Path
2 below whenever a real structure exists; this path exists to fill the gap
when one doesn't, not as a strictly-better default.

### Path 2 — you already have (or want to fetch) a real structure

1. **Get the target structure.**
   ```bash
   mkdir -p data/designs/<name> && cd data/designs/<name>
   curl -s https://files.rcsb.org/download/<PDB_ID>.pdb -o target.pdb
   ```

2. **Write a design spec YAML.** Reference the target file and the bundled
   Fab scaffolds:
   ```yaml
   entities:
       - file:
           path: target.pdb
           include:
               - chain:
                   id: A   # pick the chain you want to target
       - file:
           path:
               - ../../boltzgen_examples/repo/example/fab_scaffolds/adalimumab.6cr1.yaml
               - ../../boltzgen_examples/repo/example/fab_scaffolds/belimumab.5y9k.yaml
               # ... (see data/designs/hel_antibody/hel.yaml or
               #      data/designs/trop2_antibody/trop2.yaml in this repo
               #      for the full 14-scaffold list used so far — or let
               #      design_binder.py generate this for you automatically)
   ```
   Omit `binding_types` to let the model bind anywhere on the exposed
   surface (fine for a first pass); add it to restrict to a specific known
   epitope once you have one.

3. **Validate before spending GPU time.**
   ```bash
   source ../../../.venvs/boltzgen/bin/activate
   boltzgen check <name>.yaml --output check_output --cache ../../../weights/boltzgen
   ```
   This is cheap (seconds) and catches spec errors — chain ID typos,
   residue ranges that don't exist — before a multi-hour run. It also
   writes an mmCIF you can open in a viewer to visually confirm the
   designed region (should render as a distinct chain from the target).

4. **Smoke test, then scale up.**
   ```bash
   boltzgen run <name>.yaml --output run1 --protocol antibody-anything \
     --num_designs 50 --budget 5 --num_workers 0 --cache ../../../weights/boltzgen
   ```
   `--num_workers 0` is **required** on this hardware — omitting it
   reproduces the same CUDA-fork/DataLoader deadlock documented for
   Boltz-2 (see [Troubleshooting](#troubleshooting)). 50 designs is
   BoltzGen's own recommended first step, not an arbitrary choice here —
   confirm the pipeline behaves before committing to the 10,000–60,000
   design scale a real campaign needs. `--protocol nanobody-anything` is
   the equivalent for single-domain binders (with `nanobody_scaffolds/`
   instead of `fab_scaffolds/`).

5. **Read the results** from `run1/final_ranked_designs/`:
   `final_<budget>_designs/` (the actual CIF structures),
   `final_designs_metrics_<budget>.csv` (everything scored), and
   `results_overview.pdf` (plotted summary) — see the metrics guide above
   for what the columns mean. (`design_binder.py --launch` produces the
   same layout under `data/designs/<name>/run1/`.)

### Worked examples (from this project's own runs)

**Target: Hen Egg White Lysozyme** (PDB `1DPX`, a classic structural
biology benchmark target). 50 designs → top candidate `hel_18`:
`design_to_target_iptm=0.93`, `design_ptm=0.91`, `filter_rmsd=1.13`,
10 predicted interface H-bonds. High confidence across every axis — a
"the pipeline works as intended" result, as expected for a well-behaved
benchmark target with no real drug-discovery difficulty.

**Target: human TROP2 / TACSTD2** (PDB `7E5N`, an active oncology antigen
— the target of two approved antibody-drug conjugates, sacituzumab
govitecan and datopotamab deruxtecan). 50 designs → top candidate
`trop2_36`: `design_to_target_iptm=0.44`, `filter_rmsd=2.60`, 7 predicted
H-bonds, but **20 liability violations flagged**. Substantially lower
confidence than the HEL run — consistent with TROP2 being a harder, more
realistic target than a benchmark protein. This is the expected pattern:
don't be surprised or alarmed when a real target scores worse than a
textbook one; that's the campaign correctly reflecting real difficulty,
not a broken pipeline.

`trop2_36`'s heavy and light chains (`full_sequence_1`/`full_sequence_2`
in the metrics CSV) were then independently cross-validated by chaining
into Skill A — cofolding the designed antibody (both chains, as a real
Fab) against TROP2 from scratch via Boltz-2 and OpenFold3, neither of
which was involved in scoring the original design:

| Source | ptm | ipTM |
|---|---|---|
| BoltzGen's own internal refold (design step) | 0.74 | 0.44 |
| Boltz-2 (independent, external) | 0.67 | 0.57 |
| OpenFold3, best of 5 seeds (independent, external) | 0.60 | 0.49 |

Three independently-trained models land in the same **moderate-confidence
band** (ipTM 0.44–0.57) rather than wildly disagreeing — that consistency
is itself informative (no backend is flagging this as either a confident
hit or obvious noise), but "moderate and consistent" is still not
"validated." This is exactly the intended use of this cross-check: it
would have been a much bigger red flag if the external backends had
contradicted BoltzGen's own self-consistency score outright.

---

## Repository structure

```
co_folding/
├── README.md                  # this file
├── DESIGN.md                   # cofolding backend research, hardware findings, architecture plan
├── UI_DESIGN.md                  # planned dashboard layer (not yet built)
├── BOLTZGEN_PIPELINE.md            # BoltzGen's design pipeline internals
├── IMPROVE_DESIGN.md                # active plan for improving the design step
├── .claude/skills/
│   ├── design/SKILL.md               # Skill A/B, cofolding + antibody/binder design
│   ├── database/SKILL.md              # create/recreate databases/
│   ├── splits/SKILL.md                 # build databases/splits/
│   └── epitope_guided_design/SKILL.md   # predict + steer via binding_types conditioning
├── src/                        # shallow clones, editable-installed — real source, not opaque pip packages
│   ├── boltz/  openfold3/  protenix/  boltzgen/
├── weights/                    # permanent model weights (not a cache — see above)
│   ├── boltz2/  openfold3/  protenix/  protenix_root/  boltzgen/
├── .venvs/                     # one isolated venv per backend (+ data-fetch, epitope-prediction, mmseqs2)
├── configs/examples/            # minimal example JobSpecs, used by the smoke test
├── data/
│   ├── designs/                 # every run_design.py / boltzgen run output lands here
│   └── boltzgen_examples/        # sparse clone of BoltzGen's example/ dir — Fab/nanobody scaffolds
├── databases/                   # local antibody-design database (see database skill)
│   ├── src/                       # fetch code for every source, reproducible on a new machine
│   ├── splits/                     # leak-free train/dev/test splits (see splits skill)
│   └── sabdab/  aacdb/  ab_bind/  andd/  abdesign_db/  asd/
├── experiments/
│   └── epitope_prediction/         # epitope model + binding_types steering (see epitope_guided_design skill)
│       ├── PLAN.md                    # methodology, model comparison, downstream steering results
│       ├── data/  model/  eval/  steering/
└── scripts/
    ├── _common.py                # shared paths/helpers (REPO_ROOT, WEIGHTS_DIR, venv_bin(), ...)
    ├── setup_env.py                # one-time environment setup for all four backends
    ├── smoke_test.py                # verifies each backend actually works on this hardware
    ├── fetch_target.py               # UniProt/PDB sequence fetch (Skill A)
    ├── run_design.py                  # cofold target + partner via Boltz-2/OpenFold3 (Skill A)
    ├── predict_structure.py            # single-chain structure prediction (used by design_binder.py)
    └── design_binder.py                 # sequence-only antibody/binder design entry point (Skill B, Path 1)
```

Skill B's Path 2 (an experimental structure you already have) is still
invoked directly via the `boltzgen` CLI, following the step-by-step process
above — there's no wrapper script for that path specifically, since
`design_binder.py`'s spec-generation logic already covers it internally
whenever a structure is available (Path 1 is really "Path 2 plus an
automatic prediction step in front of it").

`backends/`, `jobspec/`, `agent/`, `orchestration/`, and `eval/` — a more
general agent-orchestration architecture described in `DESIGN.md` — are
**designed but not yet built**. What exists today (`scripts/*.py` +
the `boltzgen` CLI directly) is the working, narrower predecessor to that
architecture: it covers exactly the two capabilities in this README, for
one modality shape each, without a general agent-callable tool wrapper.

---

## Backend comparison & known limitations

| Backend | Role | Status on this hardware | License |
|---|---|---|---|
| **OpenFold3** | Cofolding (primary) | ✅ Works cleanly, no workarounds | Apache 2.0 |
| **Boltz-2** | Cofolding (fast/secondary) | ✅ Works with 2 documented flags (`--num_workers 0`, `--no_kernels`) | MIT |
| **Protenix** | Cofolding (would-be tertiary) | ❌ **Blocked** — sm_121 has no compatible kernel for at least one op; not usable for anything until an aarch64-Blackwell torch build exists | Apache 2.0 |
| **BoltzGen** | De novo antibody/binder design | ✅ Works with 1 documented flag (`--num_workers 0`) | MIT |
| AlphaFold3 | (reference only, not used) | N/A — not installed | Weights: CC-BY-NC-SA, bars commercial use |
| RFantibody | (considered, not used) | N/A — assumes CUDA 11.8 + DGL, open aarch64 wheel bugs; BoltzGen chosen instead as lower-risk | MIT (inference); training code exclusively licensed to Xaira Therapeutics |

Don't offer Protenix as an option for anything — it installs fine but
cannot run inference on this GPU. If you're running this on different
(non-aarch64, or older-compute-capability) hardware, it may well work
there; re-run `scripts/smoke_test.py` to check before relying on it.

Neither `antibody` nor `peptide` modality in Skill A has CDR-specific
handling — worth restating here since it's easy to assume otherwise. Only
Skill B (BoltzGen's `antibody-anything`/`nanobody-anything` protocols)
actually understands antibody structure (framework vs. CDR loops).

---

## Troubleshooting

**A Boltz-2 or BoltzGen run looks stuck (high elapsed time, ~0% CPU,
unmoving GPU memory).** This is a known deadlock, not a slow step —
verified twice in this project (once in Boltz-2, once in BoltzGen's
folding step): both use PyTorch DataLoader workers that fork *after* CUDA
is already initialized in the main process, which can deadlock depending
on process/CUDA-context ordering. Signature: elapsed time climbing, CPU
time barely moving (check with `ps -o pid,etime,time,%cpu -p <pid>` — a
genuinely active process should show `time` climbing roughly in step with
`etime`), all threads parked in `futex_do_wait`
(`for t in /proc/<pid>/task/*/; do cat $t/wchan; done`). Fix: kill it,
rerun with `--num_workers 0` — for BoltzGen, add `--reuse` too, so already-
completed pipeline stages aren't redone.

**Monitoring GPU usage while a job runs**: don't just dump
`nvidia-smi --query-compute-apps` and assume every listed process belongs
to your job — this machine may have other GPU processes running (a local
LLM server, a desktop compositor, etc.) that can be mistaken for your job's
progress if you're not filtering to the specific PID(s) you launched. This
mistake actually happened during this project's own TROP2 campaign: an
unrelated `llama-server` process sitting on 32GB was mistaken for a stalled
BoltzGen run for several hours. Always filter to your own tracked PID(s)
specifically.

**After switching a backend to an editable install, `import <pkg>` gives
`__file__ == None` or a `ModuleNotFoundError` for a submodule that
definitely exists in `src/`.** A stale non-editable copy is still sitting
in that venv's `site-packages/<pkg>/`, shadowing the editable install as a
namespace package. `rm -rf .venvs/<name>/lib/python3.*/site-packages/<pkg>`
and reinstall editable. `setup_env.py` does this automatically now, but
it's worth knowing if you ever do a manual reinstall.

**A weight-download prompt hangs / fails when run non-interactively.**
OpenFold3's first-run weight download asks for interactive `yes/no`
confirmation, which fails under a plain pipe. `run_design.py` and
`smoke_test.py` already pipe `"yes\n" * 10` as stdin to handle this — if
you're invoking `run_openfold` directly, do the same or run it in a real
terminal for the first download.
