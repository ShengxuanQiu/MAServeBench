"""Native versioned state, snapshot restoration, and frozen state replay."""
import copy
import json

import jsonschema
import pytest

from app.benchmark import build_experiment
from app.replay import replay_trace
from app.specs import DeploymentSpec, ExperimentConfig, StageSpec, StructureSpec, TaskBinding, WorkflowSpec
from app.versioned_state import VersionedStateStore
from app.semantic.canonicalize import ByteTokenizer, canonicalize_native_trace
from app.semantic.golden import golden_scenario, golden_system
from app.semantic.replay import MockExactBackend, ReplayExecutor
from app.semantic.store import TraceBundle
from app.semantic.validators import SemanticValidator
from pathlib import Path


def collect(tmp_path, stages, bindings=None, snapshot=None):
    experiment = ExperimentConfig(StructureSpec({}, WorkflowSpec(tuple(stages))),
        TaskBinding('state test', stage_bindings=bindings or {}), DeploymentSpec())
    runner = build_experiment(experiment, trace_dir=tmp_path / 'native')
    runner.config.export_trace_views = False
    if snapshot is not None:
        runner.state_store.restore(snapshot)
    assert runner.run()['status'] == 'completed'
    path = tmp_path / 'semantic'
    trace = canonicalize_native_trace(runner.trace.trace_path, path, tokenizer=ByteTokenizer(), workload_id='state-test')
    assert SemanticValidator().validate(trace).run_valid
    return runner, trace, path


def replay(trace, path):
    scenario = golden_scenario(trace)
    scenario.model_input_contract.update(model_identity='mock-mas-model', generation_parameters={'temperature': .2})
    scenario.replay['cache_policy'] = 'preserve_native'
    scenario.resolve([], trace, ByteTokenizer().encode_request)
    system = golden_system()
    system.model_identity = 'mock-mas-model'
    return ReplayExecutor(trace, scenario, system, MockExactBackend(), TraceBundle(path).artifacts).run()


def test_native_state_write_read_export_and_replay(tmp_path):
    stages = [StageSpec('write', atomic={'kind': 'state_write'}),
        StageSpec('read', atomic={'kind': 'state_read'}, depends_on=('write',)),
        StageSpec('update', atomic={'kind': 'state_write'}, depends_on=('read',)),
        StageSpec('old', atomic={'kind': 'state_read'}, depends_on=('update',)),
        StageSpec('latest', atomic={'kind': 'state_read'}, depends_on=('update',)),
        StageSpec('respond', atomic={'kind': 'llm'}, inputs={'context': ['old.result', 'latest.result']})]
    bindings = {name: {'parameters': {'state_object_id': 'notes', **extra}} for name, extra in {
        'write': {'value': 'version zero'}, 'read': {}, 'update': {'value': 'version one', 'expected_version': 0},
        'old': {'version': 0}, 'latest': {}}.items()}
    runner, trace, path = collect(tmp_path, stages, bindings)
    assert [(s.state_object_id, s.version) for s in trace.shared_state] == [('notes', 0), ('notes', 1)]
    assert any(op.state_dependencies for op in trace.operations if op.state)
    assert runner.state_store.read('notes', 0).value == 'version zero'
    assert runner.state_store.read('notes').value == 'version one'
    native_edges = runner.trace.execution_graph.edges
    portable_edges = {(p, op.operation_id, kind) for op in trace.operations
        for kind, field in [('control', 'control_dependencies'), ('data', 'data_dependencies'), ('state', 'state_dependencies')]
        for p in getattr(op, field)}
    assert portable_edges == native_edges
    run = replay(trace, path)
    assert run['run_valid'], run
    assert sum(r.get('recorded_state_transition_used', False) for r in run['operations']) == 5
    schema = json.loads((Path(__file__).parents[1] / 'configs/canonical_trace.schema.json').read_text())
    for event in runner.trace.events:
        if event.get('canonical_schema'):
            jsonschema.validate(event, schema)
    # The existing native-event compatibility path also retains state edges.
    _, native_replay = replay_trace(runner.trace.trace_path, DeploymentSpec(), trace_dir=tmp_path / 'replayed', export_views=False)
    assert sum(op.kind == 'state' for op in native_replay.execution_graph.operations.values()) == 5
    broken = copy.deepcopy(trace)
    reader = next(op for op in broken.operations if op.state and op.state['operation'] == 'read')
    reader.state_dependencies.clear()
    broken.finalize_hashes()
    assert 'MISSING_DEPENDENCY' in SemanticValidator().validate(broken).invalid_reasons


def test_initial_snapshot_restores_values_without_source_writers(tmp_path):
    store = VersionedStateStore()
    for i in range(3):
        store.write('notes', {'value': i}, 'source:' + str(i))
    snapshot = store.snapshot()
    stages = [StageSpec('old', atomic={'kind': 'state_read'}), StageSpec('latest', atomic={'kind': 'state_read'}),
              StageSpec('respond', atomic={'kind': 'llm'}, inputs={'context': ['old.result', 'latest.result']})]
    bindings = {'old': {'parameters': {'state_object_id': 'notes', 'version': 0}},
                'latest': {'parameters': {'state_object_id': 'notes'}}}
    _, trace, path = collect(tmp_path, stages, bindings, snapshot)
    assert [s.version for s in trace.shared_state] == [0, 2]
    assert all(s.producer_operation_id is None for s in trace.shared_state)
    assert all(op.state.get('initial') for op in trace.operations if op.state)
    run = replay(trace, path)
    assert run['run_valid'], run


def test_parallel_state_writers_record_the_actual_version_order(tmp_path):
    stages = [StageSpec('a', atomic={'kind': 'state_write'}),
              StageSpec('b', atomic={'kind': 'state_write'}),
              StageSpec('read', atomic={'kind': 'state_read'}, depends_on=('a', 'b')),
              StageSpec('respond', atomic={'kind': 'llm'}, inputs={'context': 'read.result'})]
    bindings = {name: {'parameters': {'state_object_id': 'notes', 'value': name}}
                for name in ('a', 'b')}
    bindings['read'] = {'parameters': {'state_object_id': 'notes'}}
    runner, trace, path = collect(tmp_path, stages, bindings)
    assert runner.state_store.read('notes').version == 1
    writers = sorted((op for op in trace.operations if op.state and op.state['operation'] == 'write'),
                     key=lambda op: op.state['version'])
    assert writers[0].operation_id in writers[1].state_dependencies
    assert replay(trace, path)['run_valid']


def test_state_store_copies_values_and_rejects_stale_writes():
    store = VersionedStateStore()
    value = {'data': [1]}
    store.write('key', value, 'op:0', expected_version=-1)
    value['data'].append(2)
    assert store.read('key').value == {'data': [1]}
    read = store.read('key').value
    read['data'].clear()
    assert store.read('key').value == {'data': [1]}
    with pytest.raises(ValueError, match='version changed'):
        store.write('key', 2, 'op:1', expected_version=-1)
    assert store.read('key').version == 0
    restored = VersionedStateStore()
    restored.restore(store.snapshot())
    assert restored.read('key').producer_operation_id is None
    with pytest.raises(ValueError, match='empty'):
        restored.restore(store.snapshot())


@pytest.mark.parametrize('snapshot', [
    {}, {'schema_version': 'masbench.state_snapshot/1.0.0', 'objects': {'key': [{'version': 1, 'value': 0}]}},
    {'schema_version': 'masbench.state_snapshot/1.0.0', 'objects': {'key': [{'version': 0, 'value': float('nan')}]}}])
def test_invalid_state_snapshot_does_not_change_store(snapshot):
    store = VersionedStateStore()
    with pytest.raises(ValueError):
        store.restore(snapshot)
    assert store.snapshot()['objects'] == {}
