# Germinal comparison — detailed plan

Status: **paused 2026-08-23**, after milestone 1 · draft v1 · 2026-08-22

Paused deliberately, not abandoned: milestone 1 (PyRosetta, §4) is done and verified;
milestone 2 (rest of `.venvs/germinal` — JAX+cuda12, ColabDesign, ablm_model) is
deprioritized because pushing further on the JAX/ColabDesign side was judged not worth
the effort right now. JAX itself + its CUDA plugin were installed and passed a basic GPU
smoke test (`jax.devices()` → `[CudaDevice(id=0)]`), but ColabDesign — the actual AF2
autodiff engine Stage A hallucination depends on (§1) — was never installed or verified,
and that's the larger, less-derisked piece of work this pause defers. Everything already
done (cloned repo at `src/germinal`, `.venvs/germinal` with working PyRosetta, this plan
itself) is left in place to resume from later, not rolled back.

Implements Option 2 from [IMPROVE_DESIGN.md](../../IMPROVE_DESIGN.md): stand up Germinal
(`github.com/SantiagoMille/germinal`, Apache 2.0) as an independent, parallel antibody
design pipeline and cross-validate it against BoltzGen on the same targets — not a merge
into BoltzGen's codebase, per IMPROVE_DESIGN.md §2's original framing.

This plan was written after cloning Germinal's actual source (not just its README) and
running Protenix's real toy inference on this machine today to get a fresh, exact
traceback — not relying on DESIGN.md's July note alone. Several findings below reframe
the original question ("Chai-1 vs. OpenFold3 vs. recompiled Protenix as the oracle").

## 1. Architecture reality check — two decoupled stages, only one is swappable

Germinal is not one gradient loop with a pluggable oracle. It is **two independent
stages with completely different integration profiles**:

**Stage A — hallucination** (`germinal/design/design.py:130-148`). Built on
`colabdesign.mk_afdesign_model`, i.e. **AlphaFold2/Multimer via ColabDesign, in JAX**.
Gradients flow **in-process, end-to-end through JAX autodiff**
(`colabdesign/af/model.py:248`: `jax.jit(jax.value_and_grad(_model, ...))`) straight
through AF2's Haiku-transformed structure module and distogram head
(`colabdesign/af/alphafold/model/model.py:53-54`). Every custom loss (rg, helix, i_ptm,
termini-distance) is written directly against AF2-specific output tensors
(`outputs["distogram"]["logits"]`, `outputs["structure_module"]["final_atom_positions"]`).
**This is not configurable to a different structure predictor.** Swapping in OpenFold3 or
Protenix here would mean either porting a PyTorch diffusion model into JAX-differentiable
form and re-deriving every loss against its outputs, or rewriting Germinal's entire
hallucination engine to backprop through a multi-step diffusion denoiser instead of a
single AF2-style regression pass — a research project in itself, not a config change.
**Not in scope for this comparison.** (Good news buried in this: AF2/Multimer's own
params are Apache/CC-BY licensed, not the restrictive AF3 weights license — so the
hallucination engine itself carries no license blocker here.)

**Stage B — validation/filtering** (`germinal/filters/filter_utils.py:578-669`,
`run_structure_prediction()`). After hallucination *and* AbMPNN redesign, exactly one of
AF3/Chai-1/Protenix is run **once per candidate sequence, non-differentiably**, purely to
score/accept-reject the finished sequence:

```python
if run_settings["structure_model"] == "af3":       af3.run_af3(...)
elif run_settings["structure_model"] == "chai":    chai.run_chai(...)
elif run_settings["structure_model"] == "protenix": protenix.run_protenix(...)
else: raise ValueError(...)
```

No gradients ever flow through this stage. **This is the actual "oracle choice" — and
it's a much lower-stakes decision than the hallucination engine would have been.** There
is no abstraction layer (no base class/Protocol; confirmed by grep) — each adapter is a
bespoke function with even a different return arity (af3/protenix return a 3-tuple with
`ipsae`; chai returns a 2-tuple without it, special-cased inline). A new adapter plugs in
exactly the way Protenix does: a function returning a PDB path + a metrics dict shaped
like the others, plus one new `elif` branch.

## 2. The oracle question, resolved

### 2a. Root cause of Protenix's sm_121 failure — it's not Protenix's code

Reran Protenix's real toy inference today (`protenix pred -i
configs/examples/toy_protein_ligand_protenix.json --trimul_kernel torch --triatt_kernel
torch`, the exact smoke-test invocation `scripts/smoke_test.py` uses) to get a fresh
traceback rather than trusting DESIGN.md's July note at face value. Confirmed failure
point:

```
File ".../protenix/model/modules/primitives.py", line 98, in forward
    return F.linear(input, self.weight, self.bias)
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

This is a **plain `torch.nn.functional.linear` call** — bog-standard bf16 GEMM through
PyTorch's own dispatch, not a Protenix-authored CUDA kernel. (Protenix's own custom
fused kernels — triangle-multiplication/-attention — are already routed to the `torch`
fallback via `--trimul_kernel torch --triatt_kernel torch`, the existing documented
workaround; this failure is downstream of that, in code with no Protenix-specific
compiled extension at all.) So **"recompile Protenix" is the wrong frame** — there's
nothing in Protenix's own source to recompile here.

Diagnosed the actual gap: the installed stable build (`torch==2.13.0+cu130`, the newest
available stable aarch64 wheel — verified against the full `download.pytorch.org/whl/cu130`
index) reports `get_arch_list() == ['sm_80','sm_90','sm_100','sm_110','sm_120']` — SASS
for sm_120 with **no embedded PTX** for it. GB10 is `sm_121` (`get_device_capability() ==
(12, 1)`) — one step ahead, and without a PTX fallback in the fatbinary, the driver has
nothing to JIT-compile forward-compatibly for it.

**Verified fix, in isolation**: installed PyTorch's nightly build
(`2.15.0.dev20260822+cu130` — same CUDA tag already used in `.venvs/protenix`, no system
CUDA/driver change needed) into a throwaway venv. Its `get_arch_list()` includes
`compute_120` (PTX, not just SASS) — and reran the *exact* failing operation
(`F.linear` on a `bfloat16` tensor) plus the other primitive ops Protenix's forward pass
actually uses (`scaled_dot_product_attention`, `bmm`, `LayerNorm`, all bf16 on `cuda:0`
on this GB10): **all pass.** The same nightly is available under `cu132`/`cu134` tags
too, if a toolkit bump is ever wanted for other reasons — `cu130` is the minimal diff.

**Not yet proven**: only the isolated ops were tested, not a full Protenix forward pass
through the real model (that requires the venv-scoped torch swap below, which wasn't
done yet since this is still the planning phase). Milestone 3 below is exactly that
verification, sequenced as the first real implementation step, not assumed to work.

**Recommended fix**: upgrade *only* `.venvs/protenix`'s torch to this nightly build.
Contained to that one venv — consistent with this project's existing per-backend-venv
isolation (README.md's own rationale: a shared env already broke CUDA support once,
for Protenix's `torch==2.3.1` pin, elsewhere in this project). No other backend's venv
needs to change.

### 2b. Chai-1 — real risk, likely hits the same wall, deprioritize for v1

Never actually run on this hardware (only "surveyed" per IMPROVE_DESIGN.md). Two
separate concerns, not one:

- **License/packaging**: Apache 2.0, and the `chai-lab` PyPI wheel is pure Python
  (`chai_lab-0.6.1-py3-none-any.whl` — no compiled extension of its own, so no
  Protenix-style aarch64-wheel risk in Chai-1's *own* code).
- **But it inherits the same torch gap, likely without a fix available**: `chai-lab`
  pins `torch<2.7,>=2.3.1` in its own package metadata (verified via PyPI JSON), and runs
  **in-process** (`germinal/filters/chai.py:32`: `from chai_lab.chai1 import
  run_inference` — a direct Python import, not a subprocess like Protenix/AF3). The PTX
  fix found above only appears in a *nightly* build dated today, nowhere near stable
  2.13 — and every torch version in chai-lab's own permitted range (`<2.7`) predates
  even the *stable* 2.13 build that still lacks the fix. Forcing a newer torch into
  chai-lab's env (overriding its pin) would be a second, compounding source of
  unverified risk on top of "never run on this hardware at all" and "not the vendor's
  own tested config."

**Recommendation**: skip Chai-1 for v1. Protenix's fix path is already substantially
verified; Chai-1's isn't, and has a structural reason (the torch pin) to expect the same
failure without an equally clean fix.

### 2c. OpenFold3 — feasible to add natively, but redundant given this project's existing pattern

Writing a `germinal/filters/openfold3.py` with a `run_openfold3()` function (PDB path +
a metrics dict keyed like Protenix's: `plddt`, `plddt_binder`, `ptm`, `iptm`,
`chain_ptm`, `pae`, `pae_matrix`, `aggregate_score`, `binder_pae`) plus one new `elif` in
`filter_utils.py:651` is a small, contained change — OpenFold3 is this project's most
validated backend, and there's no JAX/PyTorch bridging concern since nothing in Stage B
is differentiated.

But: this project already has a standing pattern for exactly this — README.md's TROP2
worked example chains a BoltzGen design into Skill A (`scripts/run_design.py`) as an
*external*, independent cross-check, precisely because a design shouldn't be graded only
by the backend that helped produce it. IMPROVE_DESIGN.md §2 states the same intent for
Germinal specifically: "run it on the same targets/database as BoltzGen, and
cross-validate... same pattern this project already uses." That means the right place
for OpenFold3 here isn't inside Germinal's own accept/reject loop — it's applying Skill A
to **both** pipelines' final top candidates afterward, with **zero new code**, giving an
apples-to-apples check that neither pipeline's own self-reported (and, for BoltzGen,
self-consistency-checked-but-still-self-scored) metric can provide alone.

### Decision

- **In-loop validation oracle (Stage B, drives Germinal's own accept/reject filtering)**:
  **Protenix**, once its venv's torch is upgraded (§2a). It's already the shipped
  default in two of Germinal's six example configs (`configs/run/vhh.yaml:63`,
  `vhh_pdl1.yaml:63`) — using it needs a torch swap and one adapter patch (§3), not new
  Germinal code.
- **External cross-check (both pipelines' final candidates)**: OpenFold3 + Boltz-2 via
  the existing Skill A pipeline — no new adapter needed, matches the TROP2 precedent
  exactly.
- **Chai-1, AF3**: out of scope for v1 (§2b; AF3 license-excluded as already established
  project-wide).

## 3. Required patch — Germinal's Protenix adapter assumes conda, this project doesn't use it

`germinal/filters/protenix.py`'s `_run_protenix()` hardcodes:

```python
run_cmds = ["conda", "run", "-n", conda_env, "--no-capture-output", "protenix", "pred", ...]
```

This project has no conda anywhere (verified — no `conda`/`mamba`/`micromamba` on this
machine) and deliberately uses one plain venv per backend instead (README.md's stated
rationale: a shared/conda env already silently broke CUDA for one backend by pinning an
incompatible torch, elsewhere in this project). Fix: point `run_cmds` at
`.venvs/protenix/bin/protenix` directly instead of shelling into a conda env — a small,
local patch to vendored code, the same "src is real, patchable source" approach this
project already takes elsewhere (README's editable-install rationale). Must also
re-verify Germinal's assumed Protenix CLI flags (`-i -o -s -n -e -c -p --use_msa`) still
match this project's vendored Protenix CLI signature before trusting the adapter
blindly — a quick `protenix pred --help` diff, not assumed.

## 4. PyRosetta — resolved, no conda needed

`filter_utils.py` calls into `pyrosetta_utils` for FastRelax and interface scoring
**unconditionally on every candidate** (lines 106, 114, 137, 154, 249, 263, 266, 710,
717 — not behind a feature flag). This is genuinely required to run Germinal's filter
stage as designed, independent of whichever Stage-B oracle is chosen.

**Correction to the original plan draft**: that draft concluded PyRosetta's pip route
was x86_64-only, based on `pyrosetta.org/downloads`' marketing page, and proposed conda
as the fix. Reading `pyrosetta-installer`'s actual source changed the picture:
`get_pyrosetta_os()` explicitly detects `platform.uname().machine == 'aarch64'` and
builds a wheel URL for it — aarch64 *is* a first-class target of the wheel channel, the
downloads page was just incomplete. The real wheel exists:
`west.rosettacommons.org/pyrosetta/release/release/PyRosetta4.Release.python310.aarch64.wheel/`
→ `pyrosetta-2023.11+release.fe5f8333f1c-cp310-cp310-linux_aarch64.whl` (confirmed by
directly hitting the URL, not just inferring from the installer script). Two real
constraints, both resolved:

- **The aarch64 wheel only exists for `cp39`/`cp310`** (not `cp312`, which every other
  venv in this project uses) — but this is a non-issue here, since Germinal's own
  upstream `environment.yml` already pins `python=3.10` anyway (§5's "python version
  mismatch" question resolves itself: use 3.10 for this one venv, matching upstream,
  rather than forcing 3.12).
- **`install_pyrosetta()`'s own convenience wrapper has an unrelated bug**: it correctly
  detects `os_name == 'aarch64'` via `get_pyrosetta_os()`, then immediately rejects it —
  `if os_name not in ['ubuntu', 'mac', 'm1']: sys.exit(1)` — an allowlist that forgot its
  own aarch64 branch. Bypassed by installing the wheel URL directly instead of calling
  `pyrosetta_installer.install_pyrosetta()`.

**Done, verified today**: no conda/mamba anywhere on this machine, and none was
needed. Installed `uv` (user-level, no sudo — the tool DESIGN.md's own environment
survey already recommended and this project just hadn't installed yet), used
`uv python install 3.10` to get an isolated CPython 3.10.21 for aarch64 (no system
package changes, no apt/sudo — Ubuntu 24.04's own repos don't ship 3.10 at all), created
`.venvs/germinal` on it, and `uv pip install`ed the wheel URL above directly. Verified
with a real import + `pyrosetta.init()` + the specific submodules
`germinal/filters/pyrosetta_utils.py` needs (`FastRelax`, `InterfaceAnalyzerMover`, ...)
— all load and initialize cleanly. One caveat worth carrying forward: this wheel is
dated **2023.11** (the aarch64 wheel channel appears frozen there, while `linux`/`mac`
wheels go much more recent) — over two years stale relative to whatever release
`linux-64`/`osx` users get. Fine for FastRelax/interface-scoring (stable, long-standing
Rosetta functionality), but worth knowing if a future Germinal feature depends on a
newer Rosetta capability.

## 5. Other environment findings (lower risk, but real)

- **JAX aarch64+CUDA12 wheels exist** on PyPI (`jaxlib`, `jax-cuda12-plugin` up to
  `0.11.1`, `cp312`-compatible — matches this project's existing Python 3.12.3
  convention). But the JAX/XLA-level equivalent of the sm_121 kernel-gap question
  **hasn't been verified the way Protenix's was** — wheel availability isn't the same as
  a working kernel. First implementation step for the germinal venv should be a live JAX
  GPU matmul smoke test (same pattern as §2a's torch test), not an assumption that wheel
  existence implies it works.
- **IgLM's license**: Germinal's default antibody-LM (`ablm_model: "iglm"`) is JHU
  Academic Software License — non-commercial, separate from Germinal's own Apache 2.0.
  **AbLang2 is already a first-class, supported alternative**
  (`ablm_model: "ablang"`, `filter_utils.py:277-301`, no separate import risk — verified
  it's a real code path, not a stub). Recommend defaulting to AbLang2 here unless a
  specific reason favors IgLM, since it removes a license caveat for free.
- **Python version**: resolved by §4 — `.venvs/germinal` uses Python 3.10 (via
  `uv python install 3.10`), matching both Germinal's own `environment.yml` and the
  PyRosetta aarch64 wheel's `cp310` tag. A deliberate one-venv exception to this
  project's otherwise-uniform 3.12.3, not an oversight.
- **AF3's adapter hardcodes an x86_64 path** inside its Apptainer invocation
  (`germinal/filters/af3.py:601`: `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu`) — moot
  since AF3 is already excluded by license project-wide, just confirms upstream Germinal
  isn't aarch64-audited in general, consistent with everything else found here.

## 6. Data/targets — reuse, don't rebuild

Use the same held-out targets already selected for the epitope-steering downstream
comparison (`experiments/epitope_prediction/PLAN.md` §10: 3 `dev.txt` + 5 `test.txt`
single-copy-antigen targets), so Germinal can be benchmarked head-to-head against
BoltzGen on **identical antigens** — directly matches IMPROVE_DESIGN.md §2's own stated
intent. Same split discipline already established
(`databases/splits/README.md`): iterate against `dev.txt` targets only; reserve
`test.txt` targets for final reported numbers.

## 7. Evaluation / comparison harness

Germinal's own final metrics (`plddt`, `iptm`, `pae`, `ipsae`, `interface_scores`,
`sap_score`, ... from `filter_utils.py`) are shaped completely differently from
BoltzGen's `final_designs_metrics_<budget>.csv` columns (`design_to_target_iptm`,
`filter_rmsd`, `plip_hbonds_refolded`, `liability_*`). Plan:

- A small normalization layer (`germinal_comparison/compare_report.py`), following the
  same "JSON is the stable contract, markdown is the human-readable view" pattern
  `experiments/report.py` already established for BoltzGen.
- **Primary comparison axis**: recompute each pipeline's own top candidates' real
  contact residues from their own output structure against the true epitope, using the
  *same* method already built for the epitope-steering work
  (`downstream_eval.py`'s `design_contacts_by_label_seq`, 5Å heavy-atom distance) — this
  makes the comparison apples-to-apples regardless of each pipeline's internal scoring
  conventions.
- **Secondary axis — cost, not just quality**: IMPROVE_DESIGN.md already flags
  Germinal's published cost profile (2-8 min/design on an H100, ~200-400 GPU-hours for
  ~200 successful designs) as roughly two orders of magnitude more expensive per-design
  than BoltzGen's one-shot diffusion sampling. A fair comparison has to normalize on
  GPU-hours spent, not raw design count — and this machine's real per-design wall-clock
  (aarch64, unified memory, sm_121) is unverified and could differ from the published
  H100 numbers in either direction; the pilot (§8) exists partly to measure this
  directly rather than assume it.

## 8. Compute budget / pilot scope

Given the cost uncertainty above and this project's existing "smoke test then scale"
discipline (BoltzGen's own 50-design smoke tests, not the 10,000-60,000-scale real
campaigns its docs describe), recommend a first pilot of **order 10-20 designs on 1-2
`dev.txt` targets**, timed carefully on this specific GB10, before committing to
anything matching BoltzGen's existing 50-1000-design campaigns already on disk. Germinal
is also structurally different here: each design is its own gradient-descent trajectory
(iterative, three-phase per design), not a single batched sampling call like BoltzGen's
diffusion — so per-design cost doesn't amortize across a batch the same way, and needs
its own empirical measurement rather than inferring it from BoltzGen's cost profile.

## 9. Folder layout

```
experiments/germinal_comparison/
  PLAN.md                  # this file
  setup_env.py               # .venvs/germinal setup: JAX+cuda12 smoke test first, colabdesign
                               vendored, ablm_model defaulted to ablang, chai-lab excluded,
                               PyRosetta path per §4's decision
  configs/                    # per-target run configs, adapted from germinal's own
                                 configs/run/vhh.yaml — structure_model: protenix pointed
                                 at .venvs/protenix (§3's patch)
  compare_report.py             # normalizes Germinal's + BoltzGen's metrics into one
                                   comparable table + JSON sidecar (§7)
  runs/                          # per-target campaign outputs (gitignored, like data/designs/)
```

## 10. Milestones

1. ✅ Resolve the PyRosetta path (§4) — no conda needed: `uv`-managed Python 3.10,
   direct wheel install bypassing `pyrosetta-installer`'s aarch64 guard bug, verified
   with a real `pyrosetta.init()` + the exact submodules Germinal's filter stage uses.
2. ⏸️ **Paused.** Set up the rest of `.venvs/germinal` (JAX+cuda12, colabdesign,
   ablm_model=ablang) — a live JAX CUDA smoke test *first* (before installing anything
   downstream of it), to catch a possible sm_121 JAX gap early, same discipline already
   used for the torch fix in §2a. JAX+cuda12 itself is installed and passed the smoke
   test (`jax.devices()` → `[CudaDevice(id=0)]`); ColabDesign (the actual dependency
   Stage A needs) was never attempted — deprioritized here, resume by installing it and
   probing whether AF2's Haiku-transformed ops run cleanly on this hardware.
3. ⬜ Patch `germinal/filters/protenix.py` to call `.venvs/protenix`'s binary directly
   instead of `conda run` (§3); upgrade `.venvs/protenix`'s torch to the verified
   nightly build (§2a); rerun `scripts/smoke_test.py`'s real Protenix path end-to-end —
   the full model, not just the isolated `F.linear`/SDPA/bmm/LayerNorm ops already
   confirmed during planning — before trusting the fix generally.
4. ⬜ Set `ablm_model: ablang` in the working config (license).
5. ⬜ Pilot: 10-20 designs on 1-2 `dev.txt` targets; measure real per-design wall-clock
   on this GB10 rather than assuming Germinal's published H100 numbers transfer.
6. ⬜ Build `compare_report.py`; run the same 8 targets already used in the
   epitope-steering comparison (`epitope_prediction/PLAN.md` §10) through both
   pipelines.
7. ⬜ Report: does Germinal's published "fewer designs needed to find a hit" tradeoff
   actually materialize on real held-out targets here, once GPU-hours (not raw design
   count) are the comparison unit — the actual go/no-go question this whole exercise
   exists to answer.
