# Native state and target observations

## Versioned state in native workflows

`AtomicStage(kind="state_write")` and `AtomicStage(kind="state_read")` execute
against the runner's `VersionedStateStore`. Object IDs and optional values or
versions are supplied in the corresponding task binding's `parameters`:

```python
stages = (
    StageSpec("store", atomic={"kind": "state_write"},
              inputs={"context": "research.result"}),
    StageSpec("read", atomic={"kind": "state_read"}, depends_on=("store",)),
    StageSpec("respond", atomic={"kind": "llm"}, inputs={"context": "read.result"}),
)
bindings = {
    "store": {"parameters": {"state_object_id": "notes"}},
    "read": {"parameters": {"state_object_id": "notes"}},
}
```

Add these stages after the `research` stage and pass `bindings` as the task's
`stage_bindings`. Writes use an explicit `value` when supplied, otherwise the last
input artifact. Reads select the latest retained version or an explicit `version`.
Optional `expected_version` guards concurrent updates (`-1` means no prior value).
Each successful write creates a new immutable version, starting at zero. A read
records the exact version it consumed; it never infers a writer from timestamps.

Native events retain state operations and required state edges. Collection
exports those edges, `StateRecord` provenance and content-addressed value
snapshots. Frozen replay applies recorded transitions and reads the recorded
versions. It does not invoke a live storage service or rebuild downstream prompts
from new model outputs. Native-event compatibility replay preserves state
operation identity and dependencies as well.

The store belongs to one native runner. Initialize a new run from retained state
by supplying an explicit snapshot before execution:

```python
snapshot = previous_runner.state_store.snapshot()
new_runner.state_store.restore(snapshot)
```

Snapshots contain JSON values and version histories. Restoring requires an empty
store and clears source operation IDs: an initial read uses its recorded snapshot,
not an operation from another run. A persistent-store integration can supply
this snapshot at the same boundary. Mutable external database handles and
physical KV-cache state are not serialized as logical memory.

## Scheduler, batch and operator observation interface

`app.measurements.MeasurementRecorder` is a thread-safe JSONL sink for actual
backend hooks and profiler adapters. The backend supplies:

- `scheduled(request_id, time_sec=...)`: scheduling events, including later
  resumption events when available;
- `batch(batch_id, request_ids, start_sec, end_sec, phases=..., worker_id=...)`:
  the observed request set over an interval, with optional prefill/decode phases;
- `operator(operator_id, batch_id, request_ids, start_sec, end_sec,
  operator_type=..., shape=..., counters=..., worker_id=...)`: an observed
  operator/capture with explicit participation and actual dimensions/counters.

Use an explicit `clock_id` for the target clock domain. The default recorder
clock is `time.perf_counter`; integrations may supply their actual clock.
Request participation is supplied by the backend. GEMM rows are not derived from
root concurrency or batch size. Shared operators retain their request sets and
counters once, rather than assigning the full cost independently to every task.

`join_measurements` accepts native operation events and target observations. It
requires an explicit clock offset for every native run:

```text
target_time = native_relative_time + native_to_target_offsets[run_id]
```

For native and target events measured with the same host monotonic clock,
`workflow_start.clock_origin_perf` provides the native origin. Cross-host clock
alignment must be measured by the integration. Optional one-to-one request
aliases map backend IDs to native wire request IDs.

Semantic Trace replay also retains backend request IDs, operation hierarchy and
monotonic submission/completion observations. `request_events_from_run(run)`
adapts those records for the same join. The replay run's `clock_origin_perf`
provides its origin when sharing the target host clock; first scheduling still
comes from the backend observations. The CLI also accepts `--run replay.json`
instead of native JSONL files.

Joined records retain task/stage/session/operation identities, batch composition,
operator shapes and physical counters. Completion-path durations are ready to
submit, submit to first scheduling, and first scheduling to completion. The last
interval includes pauses. Missing observations remain unavailable; unmatched
requests and inconsistent clock/event order or participation are reported.

Import normalized records with the core CLI:

```bash
python -m app.measurements --native native-root-1.jsonl native-root-2.jsonl \
  --target backend-events.jsonl --alignment alignment.json --output joined.json
```

Alignment JSON contains `target_clock_id`, `native_to_target_offsets`, and optional
`request_aliases`. This interface consumes backend observations; CPU tests use
protocol fixtures. Backend-specific hooks must emit real events using the selected
serving engine/profiler. Experiment drivers and paper result archives are kept
separately from these core APIs.
