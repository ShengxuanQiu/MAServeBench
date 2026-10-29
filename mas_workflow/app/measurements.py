"""Record backend observations and join them to native operation identities.

Times and participation come from the caller's actual backend hooks/profiler.
Joining records never infers dependencies, GEMM rows, or kernel ownership from
the number of requests. Cross-process/host clocks require explicit alignment.
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from copy import deepcopy
from pathlib import Path

SCHEMA = "masbench.target_events/1.0.0"


def _number(value):
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValueError("Observation times and counters must be finite numbers")
    return value


def _ids(values):
    if (not isinstance(values, (list, tuple)) or not values or
            any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values)):
        raise ValueError("Request IDs must be a nonempty sequence of unique strings")
    return list(values)


def validate_event(event):
    if event.get("schema_version") != SCHEMA or not isinstance(event.get("clock_id"), str) or not event["clock_id"]:
        raise ValueError("A backend observation requires its schema and clock identity")
    kind = event.get("event_type")
    if kind == "request_scheduled":
        _ids([event.get("request_id")])
        _number(event["time_sec"])
    elif kind in {"batch", "operator"}:
        if not isinstance(event.get("worker_id", ""), str):
            raise ValueError("Worker identity must be text")
        if not isinstance(event.get("batch_id"), str) or not event["batch_id"]:
            raise ValueError("Batch identity must be nonempty text")
        _ids(event["request_ids"])
        if _number(event["end_sec"]) < _number(event["start_sec"]):
            raise ValueError("Observation interval ends before it starts")
        if kind == "batch":
            phases = event.get("phases", {})
            if set(phases) - set(event["request_ids"]) or any(p not in {"prefill", "decode", "unknown"} for p in phases.values()):
                raise ValueError("Batch phases must describe its actual requests")
        else:
            if not event.get("operator_id") or not event.get("operator_type"):
                raise ValueError("Operator identity and type are required")
            shape = event.get("shape", {})
            if not isinstance(shape, dict) or any(type(v) is not int or v < 1 for v in shape.values()):
                raise ValueError("Observed dimensions must be positive integers")
            for value in event.get("counters", {}).values():
                _number(value)
    else:
        raise ValueError("Unknown backend observation type")


class MeasurementRecorder:
    """Thread-safe sink usable by scheduler callbacks and operator profilers."""
    def __init__(self, clock_id: str, path: str | Path | None = None, *, clock=time.perf_counter):
        if not isinstance(clock_id, str) or not clock_id:
            raise ValueError("An explicit backend clock ID is required")
        self.clock_id, self.path, self.clock = clock_id, Path(path) if path is not None else None, clock
        self.events = []
        self._lock = threading.Lock()

    def record(self, event_type, **fields):
        if set(fields) & {"event_type", "clock_id", "schema_version"}:
            raise ValueError("Observation identity fields are controlled by the recorder")
        event = {"schema_version": SCHEMA, "clock_id": self.clock_id,
                 "event_type": event_type, **deepcopy(fields)}
        validate_event(event)
        with self._lock:
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, allow_nan=False) + "\n")
            self.events.append(event)
        return deepcopy(event)

    def scheduled(self, request_id, *, time_sec=None):
        return self.record("request_scheduled", request_id=request_id,
                           time_sec=self.clock() if time_sec is None else time_sec)

    def batch(self, batch_id, request_ids, start_sec, end_sec, *, phases=None, worker_id=""):
        return self.record("batch", batch_id=batch_id, request_ids=_ids(request_ids),
                           start_sec=start_sec, end_sec=end_sec, phases=phases or {}, worker_id=worker_id)

    def operator(self, operator_id, batch_id, request_ids, start_sec, end_sec, *,
                 operator_type, shape=None, counters=None, worker_id=""):
        return self.record("operator", operator_id=operator_id, batch_id=batch_id,
                           request_ids=_ids(request_ids), start_sec=start_sec, end_sec=end_sec,
                           operator_type=operator_type, shape=shape or {}, counters=counters or {}, worker_id=worker_id)


def request_events_from_run(run):
    """Expose completed Semantic Trace replay calls for the same join interface.

    Actual backend IDs and monotonic submission/completion observations are used
    when available. Synthetic backends without request IDs are not invented into
    target observations. First scheduling still comes from backend hooks.
    """
    events = []
    for op in run.get("operations", []):
        observations = op.get("target_observations", {})
        rid = observations.get("backend_response_id") or observations.get("client_request_id")
        if not rid:
            continue
        origin = _number(run["clock_origin_perf"])
        for kind, field, actual in (("request_ready", "ready_time", None),
                ("request_submit", "start_time", "request_submit_perf"),
                ("request_finish", "finish_time", "request_completion_perf")):
            timestamp = _number(observations[actual]) - origin if actual in observations else _number(op[field])
            events.append({"canonical_type": kind, "request_id": rid, "run_id": run["run_id"],
                "operation_id": op["operation_id"], "task_id": op["task_id"],
                "stage_instance_id": op.get("stage_instance_id"), "session_id": op.get("session_id"),
                "relative_time_sec": timestamp,
                "time_source": "backend_client_observation" if actual in observations else "client_execution_boundary"})
    return events


def join_measurements(native_events, target_events, *, native_to_target_offsets, target_clock_id,
                      request_aliases=None):
    """Align each native run to a target clock using caller-supplied offsets.

    ``target_time = native_relative_time + native_to_target_offsets[run_id]``.
    Aliases explicitly map backend request IDs to the native wire request IDs.
    Missing observations remain None/unmatched; residence includes all pauses
    after first scheduling. Shared operators retain sets, not additive per-task
    kernel costs.
    """
    aliases = request_aliases or {}
    if len(set(aliases.values())) != len(aliases):
        raise ValueError("Request aliases must be one-to-one")
    resolve = lambda rid: aliases.get(rid, rid)
    requests, issues = {}, []
    times = {"request_ready": "ready_time", "llm_request_ready": "ready_time",
             "request_submit": "submit_time", "llm_request_start": "submit_time",
             "request_finish": "completion_time", "llm_request_end": "completion_time"}
    for event in native_events:
        kind = event.get("canonical_type", event.get("event_type"))
        if kind not in times:
            continue
        rid = event.get("request_id") or event.get("request_id_for_backend") or event.get("llm_request_id")
        oid, run = event.get("operation_id") or event.get("node_id"), event["run_id"]
        if not rid or not oid:
            raise ValueError("Native request lacks a request or operation identity")
        if run not in native_to_target_offsets:
            raise ValueError("An explicit clock alignment is required for native run " + run)
        offset = _number(native_to_target_offsets[run])
        row = requests.setdefault(rid, {"request_id": rid, "run_id": run, "operation_id": oid,
            "task_id": event.get("task_id") or run, "stage_instance_id": event.get("stage_instance_id"),
            "session_id": event.get("session_id") or event.get("agent_instance_id"),
            "workflow": event.get("workflow_class") or event.get("workflow_name"),
            "ready_time": None, "submit_time": None, "completion_time": None,
            "first_scheduling_time": None, "time_sources": {}})
        if (row["operation_id"], row["run_id"]) != (oid, run):
            raise ValueError("A wire request ID maps to multiple native operations")
        field = times[kind]
        value = _number(event["relative_time_sec"]) + offset
        if row[field] is not None and row[field] != value:
            raise ValueError("Conflicting native timestamps for one request event")
        row[field] = value
        row["time_sources"][field] = event.get("time_source", "native_event")
    batches, operators, unmatched = {}, [], set()
    for raw in target_events:
        validate_event(raw)
        if raw["clock_id"] != target_clock_id:
            raise ValueError("Target observations use different clock domains")
        event = deepcopy(raw)
        if event["event_type"] == "request_scheduled":
            rid = resolve(event["request_id"])
            if rid not in requests:
                unmatched.add(rid)
            else:
                row = requests[rid]
                old = row["first_scheduling_time"]
                row["first_scheduling_time"] = event["time_sec"] if old is None else min(old, event["time_sec"])
            continue
        event["request_ids"] = [resolve(rid) for rid in event["request_ids"]]
        _ids(event["request_ids"])
        event["members"] = [{key: requests[rid][key] for key in
            ("request_id", "run_id", "task_id", "stage_instance_id", "session_id", "operation_id", "workflow")}
            for rid in event["request_ids"] if rid in requests]
        unmatched.update(set(event["request_ids"]) - requests.keys())
        if event["event_type"] == "batch":
            event["phases"] = {resolve(rid): phase for rid, phase in event.get("phases", {}).items()}
            key = (event.get("worker_id", ""), event["batch_id"])
            if key in batches:
                raise ValueError("Duplicate worker/batch identity")
            batches[key] = event
        else:
            operators.append(event)
    for event in operators:
        batch = batches.get((event.get("worker_id", ""), event["batch_id"]))
        if batch is None:
            issues.append({"code": "MISSING_BATCH", "operator_id": event["operator_id"]})
        elif not set(event["request_ids"]) <= set(batch["request_ids"]):
            issues.append({"code": "OPERATOR_PARTICIPATION_MISMATCH", "operator_id": event["operator_id"]})
    for row in requests.values():
        values = [row[k] for k in ("ready_time", "submit_time", "first_scheduling_time", "completion_time")]
        available = [v for v in values if v is not None]
        ordered = available == sorted(available)
        if not ordered:
            issues.append({"code": "CLOCK_OR_EVENT_ORDER", "request_id": row["request_id"]})
        for name, first, last in (("pre_submission_sec", 0, 1), ("backend_waiting_sec", 1, 2),
                                  ("scheduled_residence_sec", 2, 3)):
            row[name] = values[last] - values[first] if ordered and values[first] is not None and values[last] is not None else None
    return {"schema_version": "masbench.task_measurements/1.0.0", "clock_id": target_clock_id,
            "native_to_target_offsets": dict(native_to_target_offsets), "request_aliases": dict(aliases),
            "requests": sorted(requests.values(), key=lambda r: r["request_id"]),
            "batches": list(batches.values()), "operators": operators,
            "unmatched_request_ids": sorted(unmatched), "issues": issues}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", nargs="+", default=[])
    parser.add_argument("--run", nargs="+", default=[], help="Semantic Trace replay run JSON files")
    parser.add_argument("--target", required=True)
    parser.add_argument("--alignment", required=True, help="JSON with target_clock_id, native_to_target_offsets and optional request_aliases")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.native and not args.run:
        parser.error("Provide --native operation events or --run replay records")
    def read(path):
        return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    events = [event for path in args.native for event in read(path)]
    for path in args.run:
        events.extend(request_events_from_run(json.loads(Path(path).read_text(encoding="utf-8"))))
    result = join_measurements(events, read(args.target),
                              **json.loads(Path(args.alignment).read_text(encoding="utf-8")))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"requests": len(result["requests"]), "batches": len(result["batches"]),
                      "operators": len(result["operators"]), "issues": len(result["issues"])}))


if __name__ == "__main__":
    main()
