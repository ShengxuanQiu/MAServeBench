"""Conservative optional serialization of frozen, independent operations.

Recorded dependencies are immutable prerequisites. This helper only adds an
ordering edge and records its stage meaning; it is not a workflow optimizer.
"""
from __future__ import annotations

import copy

from .model import SemanticTrace
from .validators import SemanticValidator


def add_optional_serialization(trace: SemanticTrace, source: str, target: str,
                               *, stage_meaning: str) -> SemanticTrace:
    """Clone a valid trace and serialize two independent operations of one task.

    Requests, output targets, artifacts, data/state edges, and existing control
    edges are preserved. A new workload identity records the intervention.
    """
    if not stage_meaning.strip():
        raise ValueError("An explicit stage meaning is required")
    report = SemanticValidator().validate(trace)
    if not report.run_valid:
        raise ValueError(f"Invalid source trace: {report.invalid_reasons}")
    operations = {op.operation_id: op for op in trace.operations}
    if source not in operations or target not in operations or source == target:
        raise ValueError("Expected two distinct existing operations")
    if operations[source].task_id != operations[target].task_id:
        raise ValueError("Optional serialization must stay within one task")

    def ancestors(identifier: str) -> set[str]:
        found: set[str] = set()
        pending = list(operations[identifier].dependencies)
        while pending:
            parent = pending.pop()
            if parent not in found:
                found.add(parent)
                pending.extend(operations[parent].dependencies)
        return found

    if source in ancestors(target) or target in ancestors(source):
        raise ValueError("Optional serialization requires independent operations")
    result = copy.deepcopy(trace)
    changed = next(op for op in result.operations if op.operation_id == target)
    changed.control_dependencies.append(source)
    stage = next(item for item in result.stages if item.stage_instance_id == changed.stage_instance_id)
    stage.execution_control.setdefault("optional_control_additions", []).append(
        {"source": source, "target": target, "stage_meaning": stage_meaning,
         "source_workload_hash": trace.metadata.workload_hash})
    result.finalize_hashes()
    report = SemanticValidator().validate(result)
    if not report.run_valid:
        raise ValueError(f"Invalid intervention: {report.invalid_reasons}")
    return result
