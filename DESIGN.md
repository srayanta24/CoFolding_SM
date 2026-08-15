# co-scientist: agentic cofolding orchestration — design doc

Status: draft v1 · 2026-07-12

## 1. Goal

Build an AI agent ("co-scientist") that helps a researcher **build, train, and
test protein cofolding models** across five modalities:

- protein–ligand (small molecule)
- protein–protein
- antibody–target (antigen)
- protein–peptide
- protein–nucleic acid (RNA/DNA)

Input to a job is either a **protein sequence** or an **existing protein
structure**, plus a **target modality** selecting what gets cofolded against
it. The agent does not reimplement cofolding architectures — it wraps and
orchestrates existing open-source models, generating configs, launching
runs, parsing results, and iterating.

This doc covers architecture, repo layout, the agent's control loop, and a
phased build plan. It assumes the model-selection survey already done
(OpenFold3 primary, Boltz-2 secondary, Protenix tertiary; AlphaFold3 and
Chai-1 as reference/triage only — see rationale below).

## 2. Hardware & environment constraints (this machine)

Confirmed by inspection, not assumption — this shapes several decisions below:

| Fact | Value | Implication |
|---|---|---|
| Arch | `aarch64` (NVIDIA GB10, DGX Spark / Grace Blackwell) | Most bio-ML Docker images and some PyPI wheels (flash-attn, DeepSpeed, cuEquivariance) are published `amd64`-only. Every backend needs an aarch64 compatibility check before it's trusted. |
| GPU | 1× GB10, CUDA 13.0, driver 580.159.03 | Single-GPU node. No multi-GPU data-parallel scratch training. |
| Memory | 119GB system RAM, unified with GPU (nvidia-smi reports "Not Supported" for VRAM — Grace-class unified memory) | Large unified pool is actually an advantage for MSA/pair-representation tensors that OOM on typical 24–48GB discrete GPUs — but memory *bandwidth*, not capacity, is the likely bottleneck for training throughput. |
| Torch | not installed | Clean slate — pick the install path deliberately (below), don't `pip install torch` blind and discover a CPU-only or missing-CUDA wheel later. |
| Package manager | no conda/mamba/uv present | Recommend `uv` for speed + clean lockfiles; confirm with you before installing anything at the OS/user level. |
| Docker | installed, daemon reachable, **current user not in `docker` group** | Needs `sudo usermod -aG docker $USER` (+ re-login) before container workflows work — a one-time setup step, will ask before running. |

**Environment strategy — revised after testing (§7 step 1 complete):** the
NGC-container plan below was the pre-verification assumption; in practice a
bare `uv pip install torch --index-url https://download.pytorch.org/whl/cu130`
resolves a real CUDA-enabled aarch64 wheel directly (torch 2.13.0+cu130,
confirmed working on GB10 via `torch.cuda.is_available()` and a live matmul)
— no container needed. What *is* required: **one venv per backend**, not a
shared one. Installing Boltz-2, OpenFold3, and Protenix into the same
environment silently downgrades torch to 2.3.1 (Protenix pins `torch==2.3.1`
exactly) and that version has **no aarch64 CUDA wheel at all** — its `nvidia-*`
dependencies are gated to `platform_machine == "x86_64"` in the wheel
metadata, so a shared venv quietly becomes CPU-only for everyone in it. Fix:
`.venvs/{boltz2,openfold3,protenix}/`, each installing the cu130 torch index
first, then the backend package. Optional perf kernels (cuEquivariance,
DeepSpeed) get probed per-backend at setup time and disabled with a fallback
if no aarch64 build exists — see §7 step 1 results below for exactly which
ones needed this.

**Training-scope implication:** scratch/full training of an AF3-class model
is out of scope for local hardware (these are trained on O(100+) GPU-days).
The realistic local scope is: **inference, evaluation, and targeted
fine-tuning** (LoRA/low-rank adapters, partial unfreezing, or fine-tuning on
small curated/task-specific datasets — e.g. adapting to a specific antibody
scaffold family or ligand chemotype). If full pretraining-scale training is
ever wanted, the agent's job-launch abstraction (§6) should support a remote
backend (cloud GPU cluster) behind the same interface, but that's a later
milestone, not v1.

## 3. Model backends

| Backend | Role | aarch64 status (verified) | Why |
|---|---|---|---|
| **OpenFold3** | primary | ✅ **PASS, no workarounds.** Toy protein-ligand prediction ran clean end-to-end (~80s inference, 5 samples, avg pLDDT 93.4, ipTM 0.78). | Only backend with full stack public (data + weights + training code + eval scripts), Apache 2.0, actively maintained, covers all 5 modalities, JSON-config driven — matches "agent edits config, doesn't touch model code." |
| **Boltz-2** | secondary / fast-iteration | ✅ **PASS, 2 workarounds needed.** (1) Default `num_workers=2` forks DataLoader workers after CUDA init in the main process → deadlock (25 threads stuck in `futex_do_wait` for 2h50m before being killed). Fix: `--num_workers 0`. (2) `cuequivariance-ops-torch` (compiled backend for the fused triangle-multiplication kernel) has **zero aarch64 distributions** — `pip index versions` returns none. Fix: `--no_kernels` (falls back to pure-PyTorch triangle-mult, slower but correct). With both fixes: confidence_score 0.93, ptm 0.96, iptm 0.89 in 42s. | MIT, single-YAML-per-job CLI, easiest to wrap, precedent for exactly this agent-orchestrates-Boltz pattern (BoltzGen). Training-code maturity still unconfirmed — not exercised by this smoke test. |
| **Protenix** | tertiary / cheap experiments | ❌ **FAILS at inference.** MSA search succeeds (with Protenix's own default MSA server — the `colabfold` mode has an unrelated file-naming bug in its result parser). Forward pass then hits `CUDA error: no kernel image is available for execution on the device`. Root cause: `torch.cuda.get_device_capability()` reports GB10 as **sm_121**; this torch build's `get_arch_list()` tops out at **sm_120** — one generation short. Most ops still work via PTX forward-compat (proven by OpenFold3/Boltz-2 running fully), but at least one op in Protenix's forward pass has no compatible kernel path. Not fixed by any documented flag; needs either a torch build with sm_121 support or an upstream Protenix fix. Also required the same `torch==2.3.1`-pin workaround as §2 to install at all (`--no-deps` + manual dependency install, then trimul/triatt kernels forced to `torch` to dodge the same missing `cuequivariance-ops-torch` gap as Boltz-2). | Apache 2.0, full training code open, smaller params (368M–464M) → cheapest fine-tuning runs — **currently blocked on this machine**, revisit if/when a newer torch aarch64 wheel adds sm_121 support. |
| **Chai-1** | triage-only reference | not tested (deprioritized — inference-only, not needed for the fine-tuning path) | Apache 2.0, MSA-free mode is convenient for low-latency sanity checks, but inference-only (no training code) — never a fine-tuning target. |
| **AlphaFold3** | benchmark reference only | not applicable | Weights are CC-BY-NC-SA, explicitly bars commercial use and using outputs to train similar models. Not a fork base under any circumstance unless that license changes. |

**Practical implication:** build the agent's first backend adapter (§7 step 2)
against **OpenFold3 and Boltz-2** — both are fully working. Treat Protenix as
blocked pending either a torch upgrade with sm_121 kernels or an upstream fix;
don't sink adapter-layer effort into it yet.

## 4. Repo layout (`~/co_folding`)

Revised 2026-07-18: backends are no longer opaque PyPI installs — each is a
shallow git clone under `src/`, installed editable into its own venv, so
the actual running code is on disk and inspectable/modifiable. Model
weights live permanently under `weights/`, not scattered across each
tool's own default cache (`~/.cache`, `~/.openfold3`, `~/checkpoint`, ...).

```
co_folding/
├── DESIGN.md                 # this file
├── UI_DESIGN.md               # UI layer design (companion doc)
├── .claude/skills/design/SKILL.md  # the only project skill; nothing lives outside this repo
├── pyproject.toml            # uv-managed, single env for the agent + shared tooling
├── src/                       # shallow git clones, installed editable — not pip/PyPI
│   ├── boltz/                 # jwohlwend/boltz
│   ├── openfold3/              # aqlaboratory/openfold-3
│   ├── protenix/                # bytedance/Protenix
│   └── boltzgen/                 # HannesStark/boltzgen
├── weights/                   # permanent, not a cache — survives cache clears
│   ├── boltz2/
│   ├── openfold3/               # of3-p2-155k.pt
│   ├── protenix/                 # protenix_base_default_v1.0.0.pt
│   ├── protenix_root/             # $PROTENIX_ROOT_DIR target: checkpoint/ symlinks to ../protenix
│   └── boltzgen/                  # pass --cache weights/boltzgen explicitly, not picked up by default
├── .venvs/                    # one venv per backend (shared venv breaks CUDA — see §2)
│   ├── boltz2/ openfold3/ protenix/ boltzgen/
├── backends/                  # NOT YET BUILT — see §7 step 2. scripts/ below is the pre-adapter interim.
│   ├── base.py                # BackendAdapter protocol: to_job_config(), submit(), parse_result()
│   ├── openfold3.py  boltz2.py  protenix.py  chai1.py
├── jobspec/                   # NOT YET BUILT — see §7 step 2
│   ├── schema.py               # unified JobSpec: input (sequence|structure), modality, partner spec, sampling params
│   └── translate.py            # JobSpec -> per-backend native config (YAML/JSON)
├── agent/                     # NOT YET BUILT — see §7 step 4
│   ├── tools/                  # Claude-callable tools: submit_job, check_status, fetch_metrics, compare_runs, launch_finetune
│   ├── loop.py                  # generate -> validate -> rank -> evolve -> report control loop
│   └── memory/                   # run history, learned preferences (which backend/config worked for which task type)
├── orchestration/              # NOT YET BUILT — see §7 step 7 (later/optional)
│   ├── local_runner.py         # subprocess/Docker execution on this GB10 node
│   ├── slurm_runner.py          # stub for future HPC burst
│   └── tracking.py               # W&B or MLflow run logging
├── eval/                       # NOT YET BUILT — see §7 step 3
│   ├── metrics.py               # PoseBusters-style checks, confidence (pLDDT/PAE/ipTM), ensemble/multi-seed scoring
│   └── datasets/                 # PDB/PLINDER-derived eval sets per modality
├── data/
│   ├── designs/                # run_design.py / smoke_test.py output (structures, configs, metrics)
│   └── boltzgen_examples/       # sparse clone of boltzgen's example/ dir — fab/nanobody scaffold templates, reused as-is
├── configs/
│   └── examples/                # example JobSpecs per modality, for smoke tests
└── scripts/                    # interim CLI layer, ahead of backends/+jobspec/ existing
    ├── _common.py               # REPO_ROOT, SRC_DIR, WEIGHTS_DIR, venv_bin(), produced_structure()
    ├── setup_env.py             # clones src/, editable-installs, relocates weights/ per backend
    ├── smoke_test.py             # toy protein-ligand prediction through each backend, reports pass/fail
    ├── fetch_target.py            # UniProt/PDB sequence fetch, auto-detected from identifier shape
    └── run_design.py               # cofold a fetched/given target against a given partner (§A of SKILL.md)
```

`data/` (except `boltzgen_examples/`, which is small reference YAML/CIF) and
any future `docker/` build artifacts are gitignored — the repo itself stays
small (code + configs), matching how Boltz/Protenix/OpenFold3 themselves are
structured. `src/` and `weights/` are also gitignored (large, and easily
reproduced via `scripts/setup_env.py`) — they're vendored on disk for
inspectability and permanence, not meant to be committed.

**Note on `backends/`, `jobspec/`, `agent/`, `orchestration/`, `eval/`**:
these are the target architecture from §7's phased plan, not yet built.
`scripts/{smoke_test,run_design,fetch_target}.py` are the working interim —
they do a narrower version of what `backends/`+`jobspec/` will eventually
generalize (currently: Boltz-2/OpenFold3 only, one modality shape, no
agent-callable tool wrapper yet).

## 5. Unified job specification

The agent's core abstraction is a backend-agnostic `JobSpec`, so the LLM
only ever reasons about *one* schema regardless of which backend ends up
running the job:

```python
JobSpec:
  target:
    type: "sequence" | "structure"
    value: str            # raw sequence, or path/PDB-ID for structure input
  partner:
    modality: "small_molecule" | "protein" | "antibody" | "peptide" | "rna" | "dna"
    value: str             # SMILES, sequence, or antibody framework spec depending on modality
  sampling:
    n_seeds: int
    use_msa: bool
    templates: list[str] | None
  backend_hint: str | None  # optional override; otherwise agent picks based on §6 routing rules
```

`jobspec/translate.py` converts this into OpenFold3's JSON schema, Boltz-2's
YAML schema, or Protenix's JSON schema. This is the layer that absorbs each
backend's quirks (e.g. Boltz-2's binding-affinity head is a separate flag;
antibody modality maps to plain protein-protein input for backends without
CDR-specific handling) so the rest of the system doesn't need to know them.

## 6. Agent control loop

Modeled on the generate → critique → rank → evolve → report pattern (the
closest validated prior art: Google's AI co-scientist, and the
AutoBinder-Agent pattern of using cofolding runs as validation gates):

1. **Plan** — given a research question ("does this antibody bind this
   antigen better with CDR-H3 mutation X vs Y?"), the agent drafts one or
   more `JobSpec`s.
2. **Route** — pick a backend per job: OpenFold3 by default, Boltz-2 for
   fast/cheap triage or affinity scoring, Protenix for cheap fine-tuning
   experiments. Routing rules live in `agent/tools/` as an explicit,
   inspectable function, not a hidden LLM judgment call, so runs are
   reproducible.
3. **Submit** — `orchestration/local_runner.py` launches the container job
   on the GB10 node (Slurm/Ray backend is a v2 stub for when/if multi-node
   is available).
4. **Evaluate** — `eval/metrics.py` computes confidence metrics (pLDDT,
   PAE, ipTM) and, where ground truth exists, PoseBusters-style physical
   plausibility checks. Given the survey finding that single-shot cofolding
   overstates real-world generalization, the default is **multi-seed
   ensemble sampling with rescoring**, not single-shot accept.
5. **Rank / critique** — agent compares runs, flags low-confidence or
   physically implausible results, decides whether to retry with different
   sampling params, escalate to a fine-tuning run, or report out.
6. **Report** — structured summary back to the researcher: what was tried,
   what won, confidence caveats, suggested next experiment.

Fine-tuning is a distinct, heavier tool (`launch_finetune`) gated behind an
explicit user confirmation — it's a multi-hour/GPU-hours-consuming action,
not something the loop triggers silently.

## 7. Phased build plan

1. ✅ **Environment validation — done, 2026-07-12.** aarch64 smoke test run
   for all three primary backends (venv setup, import, weight download, one
   toy protein-ligand prediction each). Result: OpenFold3 and Boltz-2 fully
   working (see §3 for exact numbers and required flags); Protenix blocked
   on a torch/sm_121 kernel gap. Reproducible via
   `scripts/setup_env.py` + `scripts/smoke_test.py` (§4).
2. **Backend adapters + JobSpec translation** — get one modality (protein-
   ligand, already proven above) working end-to-end through OpenFold3 and
   Boltz-2 via `jobspec/translate.py`, run manually (no agent yet).
3. **Eval harness** — confidence metrics + a small curated eval set (few
   dozen known complexes across modalities) so later agent decisions have
   something to rank against.
4. **Agent tools + control loop** — wrap submit/status/evaluate as
   Claude-callable tools, implement the plan→route→submit→evaluate→report
   loop for inference-only jobs first.
5. **Remaining modalities** — extend JobSpec/adapters to protein-protein,
   antibody-antigen, protein-peptide, protein-nucleic acid.
6. **Fine-tuning tool** — LoRA/partial fine-tune on Protenix (cheapest)
   first, gated behind confirmation, with tracking via W&B/MLflow.
7. **(Later, optional)** remote/HPC runner for full training-scale jobs if
   local single-GPU capacity becomes the bottleneck.

## 8. Open risks / decisions

Resolved by the step-1 smoke test (2026-07-12):

- ~~aarch64 compatibility unverified~~ → verified: OpenFold3 and Boltz-2 work
  fully (with the two documented Boltz-2 flags); Protenix is blocked on a
  torch/sm_121 kernel gap (see §3).
- ~~shared-venv risk~~ → confirmed real (Protenix's `torch==2.3.1` pin
  silently breaks CUDA for everyone in a shared env) and now mitigated by
  per-backend venvs.

Still open:

- **Protenix's sm_121 kernel gap** — no fix identified yet beyond avoiding
  the backend. Worth periodically re-checking whether a newer aarch64 torch
  wheel (`get_arch_list()` including `sm_121`) becomes available, or whether
  Protenix ships a fix upstream.
- **Boltz-2 training-code maturity** still unconfirmed — this smoke test only
  exercised inference, not fine-tuning.
- **Docker group permission** turned out to be moot for step 1 (native
  pip/uv install worked, no container needed) — revisit only if a future step
  actually needs Docker (e.g. nf-core/proteinfold integration).
- **MSA generation strategy**: both working backends used the remote MSA
  path successfully (ColabFold public server for Boltz-2, Protenix's own
  server for the Protenix attempt) — no local MMseqs2/database download
  needed so far. Revisit only if remote rate limits become a bottleneck at
  higher run volume.
- **License compliance for fine-tuned weights**: confirm redistribution
  terms per backend if fine-tuned models are ever shared/published.
- **`--no_kernels` / `torch`-fallback performance cost unmeasured** — both
  workarounds trade the fused CUDA kernels for pure-PyTorch fallbacks; fine
  for a toy single-chain target, but worth benchmarking before assuming this
  scales to larger complexes or higher-throughput agent workloads.

## 9. Next step

Step 1 is done — OpenFold3 and Boltz-2 are confirmed working end-to-end on
this machine (§3), reproducible with `scripts/setup_env.py` +
`scripts/smoke_test.py`. Move to step 2: build the first real
`backends/` adapter + `jobspec/translate.py` against these two backends.
