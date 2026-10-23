import copy
import json

import pytest

from app.semantic.cli import load_scenario
from app.semantic.golden import build_golden, golden_scenario, golden_system
from app.semantic.interventions import add_optional_serialization
from app.semantic.replay import MockExactBackend, ReplayExecutor
from app.semantic.store import TraceBundle


def test_resolved_scenario_tampering_is_rejected(tmp_path):
    from dataclasses import asdict
    trace = build_golden("fork_join", tmp_path / "trace")
    scenario = asdict(golden_scenario(trace))
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(scenario))
    assert load_scenario(str(path)).scenario_hash == scenario["scenario_hash"]
    scenario["root_arrival"]["random_seed"] += 1
    path.write_text(json.dumps(scenario))
    with pytest.raises(ValueError, match="intact resolved scenario"):
        load_scenario(str(path))


def test_optional_serialization_preserves_all_recorded_work_and_dependencies(tmp_path):
    trace = build_golden("fork_join", tmp_path / "trace")
    original = copy.deepcopy(trace)
    changed = add_optional_serialization(trace, "op:w1", "op:w2",
        stage_meaning="The two branches read frozen independent inputs; run branch one first.")
    assert trace == original
    assert changed.metadata.workload_hash != trace.metadata.workload_hash
    for before, after in zip(trace.operations, changed.operations):
        assert before.llm == after.llm
        assert before.data_dependencies == after.data_dependencies
        assert before.state_dependencies == after.state_dependencies
        assert set(before.control_dependencies) <= set(after.control_dependencies)
        if before.operation_id == "op:w2":
            assert set(after.control_dependencies) == set(before.control_dependencies) | {"op:w1"}
        else:
            assert before == after
    assert trace.artifacts == changed.artifacts
    result = ReplayExecutor(changed, golden_scenario(changed), golden_system(),
        MockExactBackend(), TraceBundle(tmp_path / "trace").artifacts).run()
    assert result["run_valid"], result
    records = {op["operation_id"]: op for op in result["operations"]}
    assert records["op:w2"]["start_time"] >= records["op:w1"]["finish_time"]


def test_optional_serialization_rejects_cycles_redundant_edges_and_missing_meaning(tmp_path):
    trace = build_golden("fork_join", tmp_path / "trace")
    for source, target, meaning in [("op:reduce", "op:w1", "cycle"),
                                     ("op:w1", "op:reduce", "redundant"),
                                     ("op:w1", "op:w2", "")]:
        with pytest.raises(ValueError):
            add_optional_serialization(trace, source, target, stage_meaning=meaning)


def test_bundle_cannot_reference_operations_outside_its_root(tmp_path):
    root = tmp_path / "trace"
    build_golden("spawn", root)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["operations_file"] = "../outside.jsonl"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        TraceBundle(root).read()
