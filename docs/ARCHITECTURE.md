# Architecture and implementation map

## Workload factors and execution

`app/specs.py` defines `MotifSpec`, `StageSpec`, `StructureSpec`, `TaskBinding`,
`DeploymentSpec`, and `ExperimentConfig`. `ExperimentConfig.compile()` joins
structure with task bindings, retaining deployment separately. `stage_dependencies()`
unions explicit prerequisites, condition/participant sources, and artifact input
sources, then rejects missing IDs and cycles.

`app/motifs/families.py` implements the four stage families. A role is a reusable
slot; `AgentInstance` owns run-local history. Spawn dispatches planned or routed
work. Fork–Join runs branches and aggregates required results. Refinement Loop
executes producer/reviewer iterations until acceptance or a configured revision
bound. Debate exchanges peer outputs according to its connectivity and rounds.
`motifs/registry.py` resolves families and explicitly marked compatibility paths.

`runtime.py`, `tracing.py`, and `execution_graph.py` record operation lifecycle,
control decisions, typed edges, payloads, artifacts, stage identity, and source
observations. Native traces are run-specific. `semantic/canonicalize.py` exports
portable bundles from the realized execution. Byte tokenization is reserved for
synthetic fixtures; real collection uses the actual model tokenizer.

## Dependency semantics

Native explicit control prerequisites are enforced during execution. Information
delivery creates data edges in addition to any control edge on the same pair.
The exporter retains multiple edge kinds for that pair. Source-observed scheduling
timestamps do not become dependency edges or target service times.

Semantic Trace stores `control_dependencies`, `data_dependencies`, and
`state_dependencies`; replay waits for their union. The native ExecutionGraph
adapter accepts control/data/state edges. Atomic `state_read` and `state_write`
stages execute against a run-local versioned store; collection retains exact
versions, writers/readers, and frozen value snapshots. Explicit snapshot
restoration initializes a new run without carrying source operation IDs across
runs. Exported agent sessions use root/explicit defaults when richer lifecycle
provenance is absent.

`versioned_state.py` implements the state store. `measurements.py` records and
joins scheduling, batch and operator observations using explicit clock alignment.
See [state and measurement APIs](STATE_AND_MEASUREMENTS.md).

## Public pipeline

1. Compile a declarative experiment and execute native agent workflows.
2. Export the realized graph, resolved model inputs, and recorded artifacts.
3. Validate the Semantic Trace and resolve its scenario with the actual tokenizer.
4. Bind a separate `SystemConfig` and replay on target completion dependencies.
5. Validate all attempted tasks and compute task latency, SLO attainment, goodput,
   and resource-normalized metrics.

`semantic/coverage.py` and `coverage_audit.py` provide separate coverage analyses.
`semantic/lowering.py` invokes a user-provided external Chakra converter and checks
mapping provenance; no production converter or simulator backend is bundled.
