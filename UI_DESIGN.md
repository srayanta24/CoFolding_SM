# co-scientist: UI design doc

Status: draft v1 · 2026-07-12
Companion to [DESIGN.md](DESIGN.md) — read that first for backend/agent architecture,
`JobSpec` schema, and backend routing. This doc covers the human-facing UI only.

## 1. Goal

Give the researcher a way to run the cofolding workflow — submit jobs,
watch them run, inspect structures and confidence metrics, compare runs,
and launch fine-tuning — without going through the agent conversation for
every action. The UI and the agent are two clients of the same backend, not
two separate systems.

## 2. Architecture

```
┌─────────────┐      ┌──────────────┐      ┌───────────────────────┐
│  Streamlit   │      │  Claude      │      │                       │
│  dashboard   │      │  agent       │      │  orchestration/       │
│  (human)     │      │  (tools/)    │      │  backends/ (DESIGN.md)│
└──────┬───────┘      └──────┬───────┘      │                       │
       │  HTTP                │  HTTP         └───────────┬───────────┘
       └──────────────┬───────┘                            │
                       ▼                                    │
              ┌─────────────────┐                           │
              │  FastAPI service │───────────────────────────┘
              │  (co_folding.api)│  calls JobSpec/orchestration layer directly
              └─────────────────┘
                       │
                       ▼
              SQLite (run metadata, status, metric history)
```

**Single API layer, two clients.** `agent/tools/submit_job` and the
Streamlit "New Run" page both call `POST /jobs` on the same FastAPI
service. This is the whole point of the design: the agent should never have
a capability the UI doesn't expose, and vice versa — one code path for
submit/status/results, so behavior can't drift between "what the agent did"
and "what the dashboard shows." It also means the UI is optional to build
first; the API can be exercised by curl/the agent while the UI catches up.

**Why Streamlit over a React/JS frontend:** this is a single-user local
research tool on one dev machine, not a product. Streamlit is Python-native
(reuses the same venv/import path as `eval/metrics.py`, `py3Dmol` for
structure rendering), has a working prototype in hours not days, and every
widget (file upload, dataframe, plot) maps directly onto what this workflow
needs. Tradeoff: harder to make multi-user, less customizable layout/state
management than React if this ever needs to serve a team or go
production-facing — revisit only if that actually becomes a requirement.

**Why a real API layer instead of Streamlit calling the orchestration code
directly:** Streamlit reruns the whole script top-to-bottom on every
interaction, which is a bad fit for anything long-running or stateful (a
multi-minute cofolding job). Routing through FastAPI means job submission
returns immediately with a run ID, and Streamlit just polls `/jobs/{id}`
like any other client — the long-running work lives in the orchestration
layer's process, not in a Streamlit rerun.

## 3. API surface (`co_folding.api`, FastAPI)

| Endpoint | Purpose |
|---|---|
| `POST /jobs` | Submit a `JobSpec` (§5 of DESIGN.md). Returns `run_id` immediately; job runs async via `orchestration/local_runner.py`. |
| `GET /jobs` | List runs, filterable by modality/backend/status/date. |
| `GET /jobs/{run_id}` | Status, backend used, timing, current stage (queued/running/scoring/done/failed). |
| `GET /jobs/{run_id}/result` | Structure file (CIF/PDB) + confidence metrics (pLDDT, PAE, ipTM) once done. |
| `GET /jobs/{run_id}/logs` | Raw backend stdout/stderr, for debugging a failed run. |
| `POST /jobs/{run_id}/rerun` | Re-submit with modified sampling params (e.g. more seeds) — used by both the agent's retry logic and a UI "retry with more seeds" button. |
| `GET /compare?run_ids=a,b,c` | Side-by-side metrics for the agent's rank/critique step and the UI's comparison view — same aggregation logic, two renderings. |
| `POST /finetune` | Launch a fine-tuning job (LoRA/partial). Requires `confirm=true` — mirrors the agent's confirmation-gated `launch_finetune` tool; the UI shows the same explicit warning dialog. |
| `GET /finetune/{run_id}` | Fine-tune job status + tracking link (W&B/MLflow run URL). |

Backed by SQLite (`data/runs.db`) for run metadata/status — sufficient at
single-user, local-node scale; no need for Postgres unless this grows
multi-user.

## 4. Streamlit pages

```
ui/
├── app.py                 # entrypoint, sidebar nav
├── pages/
│   ├── 1_new_run.py        # JobSpec builder
│   ├── 2_run_monitor.py    # live status of in-flight + recent runs
│   ├── 3_results.py        # structure viewer + metrics for one run
│   ├── 4_compare.py        # side-by-side metrics across runs
│   └── 5_finetune.py       # fine-tune launcher + tracking links
└── components/
    ├── structure_viewer.py # py3Dmol wrapper, shared by results + compare
    └── jobspec_form.py      # shared input form, used by new_run + rerun
```

**1. New Run** — form mirroring `JobSpec`: sequence paste or structure file
upload, modality picker (small molecule / protein / antibody / peptide /
RNA / DNA) with the corresponding partner-input field (SMILES box,
sequence box, etc.), sampling params (seed count, MSA on/off, templates),
optional backend override (default: agent-style auto-routing, same rule
table as `agent/tools/route_job`). Submits to `POST /jobs`, redirects to
Run Monitor.

**2. Run Monitor** — table of runs (status, modality, backend, elapsed
time), auto-refreshing via polling `GET /jobs`. Click through to Results.
This is also where a failed run's logs are one click away (`GET
/jobs/{id}/logs`) — debugging backend failures shouldn't require dropping
to a terminal.

**3. Results** — `py3Dmol` structure render (colored by pLDDT), confidence
metric summary, PAE heatmap, PoseBusters-style plausibility flags from
`eval/metrics.py`. One "Rerun with more seeds" button hitting
`POST /jobs/{id}/rerun`.

**4. Compare** — multi-select of past runs, calls `GET /compare`, renders
a metrics table + overlaid structures (useful for "same target, different
backend" or "same target, different mutation" comparisons — the exact kind
of question the agent's rank/critique step is also answering, just
human-driven here).

**5. Fine-tune** — dataset picker (from `eval/datasets/`), backend picker
(Protenix first — cheapest per DESIGN.md §7), LoRA config, explicit confirm
checkbox before submit, link out to the W&B/MLflow run once launched.

## 5. Structure visualization

`py3Dmol` (already a transitive dependency via Protenix, confirmed
installed during the backend smoke test) embeds directly in Streamlit via
its HTML component API — no separate JS build step, consistent with the
"stay in the Python stack" bias for this UI. Confidence coloring (pLDDT
gradient, PAE heatmap as a separate matplotlib/plotly panel next to the 3D
view) reuses `eval/metrics.py` output directly — the UI does not
recompute metrics, it only renders what the eval harness already produced.

## 6. Access / auth

Single-user, local-node tool — FastAPI binds to `127.0.0.1` only, no auth
in v1. If this is ever exposed beyond localhost (e.g. accessed from another
device on the same network), that's a deliberate later decision requiring
at minimum a bearer token — not a default to fall into silently.

## 7. Phased build plan

1. **FastAPI skeleton** — `POST /jobs` + `GET /jobs/{id}` wrapping the
   already-validated backend adapters (Boltz-2 first, since it's the
   fastest end-to-end loop), backed by SQLite. Exercise it with curl before
   any UI exists.
2. **Run Monitor + Results pages** — the minimum useful UI: submit via
   curl/agent, watch it finish and see the structure, in Streamlit.
3. **New Run page** — JobSpec builder form, closes the loop so the UI is
   usable standalone without the agent or curl.
4. **Compare page** — once there are enough runs logged to make comparison
   useful (naturally follows from having multiple backends/modalities
   working per DESIGN.md §7).
5. **Fine-tune page** — after the agent's `launch_finetune` tool exists and
   has been used at least once manually, so the UI wraps a proven path
   rather than being the first place fine-tuning is ever triggered.

## 8. Open risks / decisions

- **Polling vs push**: Streamlit polling `GET /jobs` on an interval is
  simplest and sufficient at single-user scale; only worth revisiting
  (websockets/SSE) if run volume or latency expectations change.
- **Long-running job lifecycle**: FastAPI needs a background task runner
  for job execution (`orchestration/local_runner.py` launched via
  `BackgroundTasks` or a simple subprocess queue) — must not block the API
  process on a multi-minute GPU job. A single-GPU node also means only one
  job can actually run at a time regardless of how many are queued; the API
  should make queue position visible rather than pretending jobs run in
  parallel.
- **Result storage growth**: structures + logs accumulate under
  `data/weights/*_run` (per DESIGN.md layout) — no retention policy defined
  yet; fine for now given 3.5TB free, worth a cleanup command once run
  volume grows.
