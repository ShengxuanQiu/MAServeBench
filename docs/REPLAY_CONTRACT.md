# Replay contract and schema

## Bundle layout

`manifest.json` stores metadata, tasks, stages, sessions, artifacts, shared-state
records, and an `operations_file` reference. `operations.jsonl` stores one typed
operation per line. `artifacts/` stores content-addressed payloads. The dataclasses
in `app/semantic/model.py` are serialized by `store.py`; JSON schemas live in
`mas_workflow/configs/semantic/`. Workload identity excludes source observations;
content identity includes the serialized trace. Artifact hashes are checked on
read and references must remain inside their bundle/store.

## Fixed work

`ScenarioManifest.resolve()` materializes each model request and records its hash,
input token IDs, and input token count. `preserve_native` leaves canonical model
inputs unchanged. Cache-scope isolation is explicit and changes scenario identity.
The CLI requires an intact hash when loading a resolved scenario; changing a
template requires explicit resolution rather than silently rehashing a run input.

`ReplayExecutor` releases operations only after their recorded control/data/state
parents finish on the target and after declared external delay. The source host's
timestamps are diagnostic. Tool and transform results come from recorded
artifacts; replay does not repeat live external tools or regenerate downstream
requests from new model outputs. State operations apply recorded versioned
transitions and value snapshots without calling a live external database. Native
state stages and initial snapshots are described in
[state and measurement APIs](STATE_AND_MEASUREMENTS.md).

Collected tools retain `recorded_duration_sec`, which replay waits inside the
tool operation before returning its result. An explicit
`controlled_external_delay` is a separate wait before readiness. Source LLM and
kernel timings remain diagnostic observations, not target service times.

The vLLM adapter deep-copies each resolved request, enforces the recorded output
length, switches to non-streaming transport, and removes transport-only streaming
options. Other recorded generation settings are retained. Reported output length
must match. Target-reported prompt token IDs or usage counts are compared with
the resolved input contract; explicit mismatches invalidate the run and affected
task. Missing target input evidence remains unavailable, not proof of an unchanged
server input. Pin and record the server/tokenizer configuration; native collection
and replay require the same model-input contract.

## Required and optional controls

The base operation schema has one control list; it has no per-edge required flag.
Existing recorded control edges, data edges, and state edges are retained as
prerequisites. A dependency being only a control edge does not imply that it is
optional: it may represent a required join or stage ordering.

`add_optional_serialization()` is a narrowly scoped release helper, derived from
the existing independent-analysis serialization experiment. It adds an edge only
between independent operations of the same task and requires an explicit stage
meaning. It preserves requests, output targets, artifacts, existing control edges,
and data/state edges, records the addition in stage provenance, and creates new
workload/content hashes. Tests check these invariants and target execution order.
It does not remove or classify arbitrary existing control edges, and graph
independence alone is not a proof of application-level semantic equivalence.

## Fidelity and metrics

`length_locked` is supported by the vLLM OpenAI adapter. `token_locked` is supported
by the opt-in CUDA vLLM 0.30.0 engine adapter in its documented dense-Qwen3/
Model-Runner-V2 profile and by the synthetic conformance backend. Ordinary HTTP
and Ascend adapters reject it. See [token replay](TOKEN_REPLAY.md). Native event
replay is a separate best-effort compatibility path. Validation failures remain
failures; task metrics retain attempted tasks in SLO denominators. Primary task
latency and deadlines use arrival to completion, including tool/external waits.
The separately named `latency_excluding_external_delay_sec` is diagnostic only.
Metrics prefer raw arrival-to-completion when reading older run files. Telemetry
labels observed, backend-reported, estimated, and unavailable values separately.

## Scenario execution scope

The Semantic Trace CLI replays one bundle per invocation. Its root-arrival modes
schedule the tasks already present in that bundle; `measurement.task_count` does
not synthesize additional tasks. A multi-ID `trace_set_ids` declaration is not a
multi-bundle corpus runner. `mixed_study.py` provides the separate native-trace
corpus/mixed-class protocol. Warmup policy metadata does not by itself execute an
additional warmup phase inside Semantic Trace replay.
