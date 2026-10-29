# MAServeBench

MAServeBench builds multi-agent serving workloads, records their realized execution,
and replays frozen work against an OpenAI-compatible vLLM endpoint. Its four stage
families are **Spawn**, **Fork–Join**, **Refinement Loop**, and **Debate**. Stage
composition, task binding, deployment, and offered load are separate inputs.
The Python package retains the historical name `mas_workflow` / `app`.

## Branches and architecture

**`main` is the GPU-oriented reference branch.** Both branches share the benchmark and Semantic Trace
contracts; backend setup and the default device launcher differ. Fixes to request
preservation, generation forwarding, tokenization, and typed edges are shared.

```text
mas_workflow/
  app/specs.py, benchmark.py       declarative factors, compilation, native runs
  app/motifs/                     stage families, agent sessions, composition
  app/runtime.py, tracing.py      operation events, artifacts, request recording
  app/execution_graph.py          realized DAG and typed native dependencies
  app/semantic/                   portable bundles, validation, scenarios, replay
  app/backend_adapters.py         serving/device telemetry normalization
  app/versioned_state.py          native state versions and snapshot restoration
  app/measurements.py             scheduler/batch/operator record and join API
  app/publication.py, study.py    reproducible matrices and arrival schedules
  app/mixed_study.py, capacity.py mixed workloads and SLO/capacity analysis
  configs/                       JSON/YAML examples and versioned schemas
  prompts/                       English compatibility-workflow prompts
  tests/                         contract tests and synthetic golden bundles
scripts/                         portable construction, serving, and replay helpers
docs/                            architecture, replay contracts, reproducibility
evaluation/                      coverage schemas and an explicitly synthetic example
```

Historical workflow implementations remain for import compatibility and their
tests. The primary entry points are `app.benchmark` and `app.semantic.cli`.
Private machine launchers, old experiment dumps, ad hoc plotting/report scripts,
and the personal backend-fork submodule are excluded. The client-side core uses
an external serving endpoint; it does not require a recursive submodule clone.

## Installation

Use Python 3.10 or newer; the release checks use Python 3.12 on Linux.
From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r mas_workflow/requirements-benchmark.txt
export PYTHONPATH="$PWD/mas_workflow:$PWD"
```

PowerShell activation and module path:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD/mas_workflow;$PWD"
```

The core/mock path needs PyYAML, jsonschema, and NumPy; pytest runs the checks.
`requirements-repro.txt` pins the CPU validation dependencies. Real-model trace
collection additionally needs `transformers` and the **same tokenizer and chat
template as the serving model**. Install the serving stack in its own compatible
environment. Optional legacy ReAct integrations need LangGraph, LangChain OpenAI,
and the OpenAI SDK; they are not part of the core smoke procedure.

## Verified CPU quick start

```bash
python -m pytest mas_workflow/tests --import-mode=importlib -q
python -m app.semantic.schema_validation --repo-root .
python -m app.benchmark run --experiment mas_workflow/configs/benchmark/experiment.json --trace-dir outputs/native-smoke
python -m app.semantic.cli e2e --output outputs/semantic-e2e
```

The native example runs a mock Fork–Join workload and writes operation JSONL,
graph, summary, and visualization exports. `e2e` constructs all four synthetic
Semantic Trace bundles, resolves scenarios, replays them, and writes validation
and task metrics. Synthetic token IDs test the contract; they are not model tokens
or accelerator measurements.

## Workload construction and native execution

`configs/benchmark/experiment.json` references structure, task, and deployment
files relative to the experiment file. Structure binds a motif library to a stage
DAG. Task bindings supply instructions and information delivery; deployment
supplies the model, endpoint, generation settings, and request concurrency.
`load` counts concurrent user workflows, not agent calls.

```bash
python scripts/build_example.py --preset parallel_refine --users 4 --output outputs/experiment.json
python -m app.benchmark run --experiment outputs/experiment.json --trace-dir outputs/native
```

Available helper presets are `matched_parallel`, `matched_chain`, `parallel_refine`,
and `peer`; the Python API and publication configurations expose additional
families and compositions. See [architecture](docs/ARCHITECTURE.md).

Native workflows can also use atomic `state_write`/`state_read` stages with exact
version provenance and initial snapshots. The measurement interface records and
joins backend scheduling, batch membership, and operator observations to native
task/stage/operation identities with explicit clock alignment. See
[state and measurement APIs](docs/STATE_AND_MEASUREMENTS.md).

## GPU / Ascend serving

Use a compatible vLLM GPU environment on `main`, or a matched driver/CANN,
PyTorch/torch-npu, vLLM and **vLLM-Ascend** environment on `ascend`.
This repository does not install drivers or select a CANN/torch compatibility
matrix. Record exact installed versions with the experiment artifacts.

Backend installation sequence:

1. Create a separate serving environment with a Python version supported by the
   chosen backend release.
2. For GPU, install the vLLM wheel matching the CUDA/driver environment following
   the [official GPU installation guide](https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html).
   For Ascend, select a complete supported version tuple and install CANN,
   torch-npu, vLLM, and vLLM-Ascend following the
   [official Ascend installation guide](https://docs.vllm.ai/projects/ascend/en/main/getting_started/installation.html).
3. Install `transformers` in the benchmark environment for real-model trace
   collection. Copy or mount the exact model/tokenizer files into the environment.
4. Activate the serving environment and run the launcher below. Activate the
   benchmark environment in a second shell for collection and replay.

The backend packages and drivers are intentionally outside the CPU requirements;
the repository does not replace their hardware-specific installation steps.

```bash
export MODEL_PATH=/path/to/local/model
export SERVED_MODEL=masbench-model
export CUDA_VISIBLE_DEVICES=0
bash scripts/serve.sh
```

The launcher binds to localhost and requires an explicit device selection. It
executes the standard vLLM OpenAI server in the active **serving environment**.
Use a second shell with the benchmark environment and `PYTHONPATH` set:

```bash
python scripts/build_example.py --preset matched_parallel --backend openai_compatible --model masbench-model --users 4 --output outputs/real-experiment.json
python -m app.benchmark run --experiment outputs/real-experiment.json --trace-dir outputs/real-native
```

For telemetry use the corresponding
`configs/publication/deployment-gpu.template.json` or
`deployment-ascend.template.json`, fill in device and model identifiers, and bind
it in an experiment. Device counters depend on the backend and profiler; missing
counters remain unavailable. See [reproducibility status](docs/REPRODUCIBILITY.md)
for the distinction between CPU checks and actual device validation.

## Semantic Trace collection and fixed-work replay

Choose one native **operation event JSONL**, not a summary or model-output file.
Use the tokenizer of the served model:

```bash
python -m app.semantic.cli collect --source /path/to/native-operation-trace.jsonl --output outputs/frozen --workload-id example --tokenizer "$MODEL_PATH"
python -m app.semantic.cli validate --trace outputs/frozen
python scripts/prepare_replay.py --trace outputs/frozen --tokenizer "$MODEL_PATH" --backend vllm-gpu --backend-version SERVER_VERSION --hardware DEVICE_MODEL --output outputs/replay-config
python -m app.semantic.cli replay --trace outputs/frozen --scenario outputs/replay-config/scenario.json --system outputs/replay-config/system.json --backend vllm-openai --base-url http://127.0.0.1:8000/v1 --output outputs/replay.json
python -m app.semantic.cli measure --run outputs/replay.json --slo-sec 30 --accelerators 1 --output outputs/metrics.json
```

The scenario fixes resolved model inputs, tokenizer/chat-template identities,
output lengths, root arrivals, cache policy, and validity rules. `preserve_native`
keeps the recorded model-input content; explicit `reuse_scope_isolated` adds
deterministic cache markers and creates a different scenario. Output length and
non-streaming transport are enforced by the replay adapter. Downstream requests
use recorded content rather than newly generated text. All recorded control,
data, and state predecessors must complete on the target before release.

The vLLM adapter supports `length_locked` through `min_tokens`, `max_tokens`, and
`ignore_eos`, and invalidates completion-length mismatches. Exact output
`token_locked` additionally has an explicit, official-engine CUDA adapter;
native output IDs and a compatible vLLM Model Runner V2 are required. Existing
OpenAI/Ascend adapters reject this mode. See [token replay](docs/TOKEN_REPLAY.md)
for the backend scope, recording contract, and real-device acceptance command.
The older `app.benchmark replay` / `app.replay` path is a separate, best-effort
native-event replay interface. Use Semantic Trace replay for the stricter contract.

## Optional serialization and experiment reproduction

The release helper can add one ordering edge between independent operations of
the same frozen task. It preserves every recorded dependency and payload,
records the stage meaning, rejects cycles, and computes a new workload identity:

```bash
python -m app.semantic.cli golden --template fork_join --output outputs/control-example
python scripts/optional_serialization.py --trace outputs/control-example/fork_join --source op:w1 --target op:w2 --stage-meaning "Independent branches read their own frozen inputs; schedule branch one first." --output outputs/control-serial
python -m app.semantic.cli validate --trace outputs/control-serial
```

Resolve a new scenario for a modified bundle before replay. This is a conservative
addition operation, not permission to remove required stage barriers or data/state
dependencies. The base schema stores control dependencies as one list; added
optional edges are identified by provenance under stage `execution_control`.
See [replay contracts](docs/REPLAY_CONTRACT.md) for precise implementation scope.

Generate a structural matrix without executing device work:

```bash
python -m app.publication --manifest mas_workflow/configs/publication/structural-sweep.json --output outputs/structural-matrix
```

The matrix freezes resolved cells and fingerprints inputs. Explicit `--execute`
runs the configured study; `--resume` requires matching fingerprints. Capacity
and mixed-workload protocols retain attempted/failed tasks and exact schedules.
Hardware measurements additionally require model weights, compatible backend
versions, device availability, and the original frozen input corpus. This core
release includes runnable examples and protocols; it does not bundle private
paper result archives or claim that the example reproduces every paper figure.

## Artifact hygiene

Use the reviewer-facing mirror or source archive for double-blind review. Source
archives exclude `.git`, local environments, outputs, credentials, and machine
paths. The original upstream repository preserves its pre-existing history; that
history is not anonymized by a source cleanup. Public documentation and source
comments in this release are English.
