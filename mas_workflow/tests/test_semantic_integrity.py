"""Contract regressions for timing, target inputs, hierarchy and native rounds."""
import copy
import json
import time

import pytest

from app.benchmark import build_experiment
from app.specs import DeploymentSpec, ExperimentConfig, MotifSpec, RoleTask, StageSpec, StructureSpec, TaskBinding, WorkflowSpec
from app.semantic.canonicalize import ByteTokenizer, canonicalize_native_trace
from app.semantic.golden import build_golden, golden_scenario, golden_system
from app.semantic.metrics import task_metrics
from app.semantic.replay import BackendResult, MockExactBackend, ReplayExecutor
from app.semantic.store import TraceBundle
from app.semantic.validators import SemanticValidator


def test_native_tool_duration_survives_collection_and_replay(tmp_path):
    experiment = ExperimentConfig(StructureSpec({'m': MotifSpec('Fork--Join', width=1)},
        WorkflowSpec((StageSpec('work', 'm'),))),
        TaskBinding('test', roles={'Worker': RoleTask(tools=('search',))}), DeploymentSpec())
    runner = build_experiment(experiment, trace_dir=tmp_path / 'native')
    runner.config.export_trace_views = False
    search = runner.search_provider.search
    def delayed(*args, **kwargs):
        time.sleep(.02)
        return search(*args, **kwargs)
    runner.search_provider.search = delayed
    assert runner.run()['status'] == 'completed'
    path = tmp_path / 'trace'
    trace = canonicalize_native_trace(runner.trace.trace_path, path,
        tokenizer=ByteTokenizer(), workload_id='tool-test')
    scenario = golden_scenario(trace)
    scenario.model_input_contract['model_identity'] = 'mock-mas-model'
    scenario.model_input_contract['generation_parameters'] = {'temperature': .2}
    scenario.replay['cache_policy'] = 'preserve_native'
    scenario.resolve([], trace, ByteTokenizer().encode_request)
    system = golden_system()
    system.model_identity = 'mock-mas-model'
    run = ReplayExecutor(trace, scenario, system, MockExactBackend(), TraceBundle(path).artifacts).run()
    assert run['run_valid'], run
    tool = next(op for op in trace.operations if op.tool)
    assert tool.tool['recorded_duration_sec'] >= .02
    record = next(r for r in run['operations'] if r['operation_id'] == tool.operation_id)
    assert record['target_duration_sec'] >= tool.tool['recorded_duration_sec']


def test_external_delay_counts_toward_task_deadline_and_old_run_metrics(tmp_path):
    trace = build_golden('spawn', tmp_path)
    trace.operations[0].controlled_external_delay = .02
    trace.finalize_hashes()
    run = ReplayExecutor(trace, golden_scenario(trace), golden_system(), MockExactBackend()).run()
    task = run['tasks'][0]
    assert task['task_latency_sec'] == task['raw_wall_clock_latency_sec']
    assert task_metrics(run, slo_sec=.005, accelerator_count=1)['slo_attainment'] == 0
    old = copy.deepcopy(run)
    old['tasks'][0]['task_latency_sec'] = 0
    assert task_metrics(old, slo_sec=.005, accelerator_count=1)['slo_attainment'] == 0


@pytest.mark.parametrize('evidence', [{'usage': {'prompt_tokens': 0}}, {'prompt_token_ids': [0]}, 'conflicting_usage'])
def test_target_input_mismatch_invalidates_run_and_task(tmp_path, evidence):
    trace = build_golden('spawn', tmp_path)
    class Backend(MockExactBackend):
        def generate(self, request, operation, mode):
            value = super().generate(request, operation, mode)
            if evidence == 'conflicting_usage':
                ids = scenario.resolved_request_contracts[operation.operation_id]['input_token_ids']
                value.observations.update(prompt_token_ids=ids, usage={'prompt_tokens': 0})
            else:
                value.observations.update(evidence)
            return value
    scenario = golden_scenario(trace)
    run = ReplayExecutor(trace, scenario, golden_system(), Backend()).run()
    assert not run['run_valid'] and 'TOKENIZER_MISMATCH' in run['invalid_reasons']
    assert not run['tasks'][0]['successful']
    assert task_metrics(run, slo_sec=100, accelerator_count=1)['slo_attainment'] == 0


@pytest.mark.parametrize('available', [False, True])
def test_input_verification_retains_missing_and_matching_usage(tmp_path, available):
    trace = build_golden('spawn', tmp_path)
    scenario = golden_scenario(trace)
    class Backend(MockExactBackend):
        def generate(self, request, operation, mode):
            value = super().generate(request, operation, mode)
            if available:
                value.observations['usage'] = {'prompt_tokens': scenario.resolved_request_contracts[operation.operation_id]['input_token_count']}
            return value
    run = ReplayExecutor(trace, scenario, golden_system(), Backend()).run()
    assert run['run_valid']
    assert run['input_token_verification']['verified' if available else 'unavailable'] == 2


def test_native_debate_iterations_survive_export(tmp_path):
    experiment = ExperimentConfig(StructureSpec({'m': MotifSpec('Debate', width=2, rounds=2)},
        WorkflowSpec((StageSpec('debate', 'm'),))), TaskBinding('test'), DeploymentSpec())
    runner = build_experiment(experiment, trace_dir=tmp_path / 'native')
    runner.config.export_trace_views = False
    assert runner.run()['status'] == 'completed'
    trace = canonicalize_native_trace(runner.trace.trace_path, tmp_path / 'trace', tokenizer=ByteTokenizer(), workload_id='rounds')
    assert sorted(op.iteration for op in trace.operations if op.llm) == [0, 0, 1, 1, 2, 2]


@pytest.mark.parametrize('relation', ['stage_dependencies', 'parent_stage_instance_id', 'session_parent'])
def test_hierarchy_self_cycles_are_rejected(tmp_path, relation):
    trace = build_golden('spawn', tmp_path)
    stage = trace.stages[0]
    if relation == 'stage_dependencies':
        stage.stage_dependencies.append(stage.stage_instance_id)
    elif relation == 'parent_stage_instance_id':
        stage.parent_stage_instance_id = stage.stage_instance_id
    else:
        session = trace.sessions[1]
        session.parent_session_id = session.session_id
    trace.finalize_hashes()
    assert 'INVALID_HIERARCHY' in SemanticValidator().validate(trace).invalid_reasons


def test_stage_dependency_cycle_across_two_stages_is_rejected(tmp_path):
    trace = build_golden('spawn', tmp_path)
    other = copy.deepcopy(trace.stages[0])
    other.stage_instance_id = 'other'
    trace.stages.append(other)
    trace.stages[0].stage_dependencies = [other.stage_instance_id]
    other.stage_dependencies = [trace.stages[0].stage_instance_id]
    trace.finalize_hashes()
    assert 'INVALID_HIERARCHY' in SemanticValidator().validate(trace).invalid_reasons
