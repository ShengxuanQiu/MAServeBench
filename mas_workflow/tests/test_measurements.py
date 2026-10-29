"""Measured participation and explicit clock alignment, without serving hardware."""
import copy
import json
import time

import pytest

from app.measurements import MeasurementRecorder, join_measurements


def native():
    result = []
    for rid, run, stage, workflow, finish in [('r1', 'a', 'sa', 'Loop', 7), ('r2', 'b', 'sb', 'Spawn', 6.5)]:
        for kind, timestamp in [('request_ready', 0), ('request_submit', 1), ('request_finish', finish)]:
            result.append({'canonical_type': kind, 'request_id': rid, 'operation_id': 'op:' + rid,
                'run_id': run, 'stage_instance_id': stage, 'agent_instance_id': 'agent:' + rid,
                'workflow_class': workflow, 'relative_time_sec': timestamp})
    return result


def records(path=None):
    recorder = MeasurementRecorder('target-clock', path)
    recorder.scheduled('r1', time_sec=102)
    recorder.scheduled('r1', time_sec=106)  # Resume is not first scheduling.
    recorder.scheduled('r2', time_sec=103)
    recorder.batch('step:0', ['r1', 'r2'], 103, 104, phases={'r1': 'decode', 'r2': 'prefill'})
    recorder.batch('step:1', ['r1'], 106, 106.5, phases={'r1': 'decode'})
    recorder.operator('projection:0', 'step:0', ['r1', 'r2'], 103.1, 103.4,
        operator_type='matrix', shape={'M': 8, 'N': 4096, 'K': 4096}, counters={'dram_read_bytes': 8192})
    return recorder


def join(target, **kwargs):
    return join_measurements(native(), target, native_to_target_offsets={'a': 100, 'b': 100},
                             target_clock_id='target-clock', **kwargs)


def test_real_intervals_join_to_paths_batches_and_operator_sets(tmp_path):
    recorder = records(tmp_path / 'events.jsonl')
    target = [json.loads(s) for s in (tmp_path / 'events.jsonl').read_text().splitlines()]
    result = join(target)
    assert not result['issues'] and not result['unmatched_request_ids']
    first = result['requests'][0]
    assert first['pre_submission_sec'] == 1
    assert first['backend_waiting_sec'] == 1
    assert first['scheduled_residence_sec'] == 5  # Includes the unscheduled pause.
    assert {m['workflow'] for m in result['batches'][0]['members']} == {'Loop', 'Spawn'}
    op = result['operators'][0]
    assert op['shape']['M'] == 8 and len(op['request_ids']) == 2
    assert op['counters']['dram_read_bytes'] == 8192
    assert len(result['operators']) == 1  # Shared costs are retained once, not duplicated per task.
    assert target == recorder.events


def test_missing_scheduler_observations_remain_unavailable():
    result = join([])
    assert all(r['first_scheduling_time'] is None and r['scheduled_residence_sec'] is None for r in result['requests'])
    assert result['requests'][0]['pre_submission_sec'] == 1


def test_unknown_background_requests_are_retained():
    recorder = MeasurementRecorder('target-clock')
    recorder.batch('step', ['r1', 'external'], 103, 104)
    result = join(recorder.events)
    assert result['unmatched_request_ids'] == ['external']
    assert result['batches'][0]['request_ids'] == ['r1', 'external']


def test_explicit_backend_request_aliases_preserve_phase_and_hierarchy():
    events = records().events
    for e in events:
        if e['event_type'] == 'request_scheduled':
            e['request_id'] = 'server:' + e['request_id']
        else:
            e['request_ids'] = ['server:' + r for r in e['request_ids']]
            if 'phases' in e:
                e['phases'] = {'server:' + r: p for r, p in e['phases'].items()}
    result = join(events, request_aliases={'server:r1': 'r1', 'server:r2': 'r2'})
    assert not result['issues'] and not result['unmatched_request_ids']
    assert result['batches'][0]['phases'] == {'r1': 'decode', 'r2': 'prefill'}


def test_wrong_clock_alignment_is_flagged_without_negative_path_times():
    events = records().events
    events[0]['time_sec'] = 90
    result = join(events)
    assert any(e['code'] == 'CLOCK_OR_EVENT_ORDER' for e in result['issues'])
    assert result['requests'][0]['scheduled_residence_sec'] is None


def test_missing_clock_or_duplicate_identity_is_rejected():
    with pytest.raises(ValueError, match='alignment'):
        join_measurements(native(), [], native_to_target_offsets={}, target_clock_id='target-clock')
    with pytest.raises(ValueError, match='clock domains'):
        join_measurements(native(), records().events, native_to_target_offsets={'a': 100, 'b': 100}, target_clock_id='other')
    with pytest.raises(ValueError, match='one-to-one'):
        join([], request_aliases={'x': 'r1', 'y': 'r1'})
    events = native()
    events.append({**events[0], 'operation_id': 'other'})
    with pytest.raises(ValueError, match='multiple native'):
        join_measurements(events, [], native_to_target_offsets={'a': 100, 'b': 100}, target_clock_id='target-clock')


def test_operator_participation_must_agree_with_its_batch():
    recorder = MeasurementRecorder('target-clock')
    recorder.batch('step', ['r1'], 103, 104)
    recorder.operator('op', 'step', ['r2'], 103.1, 103.2, operator_type='matrix')
    assert join(recorder.events)['issues'][0]['code'] == 'OPERATOR_PARTICIPATION_MISMATCH'
    missing = join(recorder.events[1:])
    assert missing['issues'][0]['code'] == 'MISSING_BATCH'


@pytest.mark.parametrize('request_ids,start,end', [(['r1', 'r1'], 0, 1), ([], 0, 1), (['r1'], 2, 1), (['r1'], 0, float('nan'))])
def test_recorder_rejects_invalid_intervals(request_ids, start, end):
    with pytest.raises(ValueError):
        MeasurementRecorder('clock').batch('step', request_ids, start, end)


def test_recorder_copies_mutable_inputs_and_namespace_is_per_worker():
    recorder = MeasurementRecorder('target-clock')
    ids = ['r1']
    emitted = recorder.batch('step', ids, 103, 104, worker_id='0')
    ids.append('r2')
    emitted['request_ids'].append('r2')
    assert recorder.events[0]['request_ids'] == ['r1']
    recorder.batch('step', ['r2'], 103, 104, worker_id='1')
    assert len(join(recorder.events)['batches']) == 2
    with pytest.raises(ValueError, match='Duplicate worker'):
        join(recorder.events + [copy.deepcopy(recorder.events[0])])


def test_semantic_replay_observations_feed_measurement_cli(tmp_path, monkeypatch):
    from app.measurements import main
    from app.semantic.golden import build_golden, golden_scenario, golden_system
    from app.semantic.replay import MockExactBackend, ReplayExecutor
    recorder = MeasurementRecorder('same-host', tmp_path / 'backend.jsonl')
    trace = build_golden('spawn', tmp_path / 'trace')
    class Backend(MockExactBackend):
        def generate(self, request, operation, mode):
            started = time.perf_counter()
            rid = 'backend:' + operation.operation_id
            recorder.scheduled(rid)
            value = super().generate(request, operation, mode)
            value.observations.update(backend_response_id=rid, request_submit_perf=started,
                                      request_completion_perf=time.perf_counter())
            return value
    run = ReplayExecutor(trace, golden_scenario(trace), golden_system(), Backend()).run()
    assert run['run_valid']
    (tmp_path / 'run.json').write_text(json.dumps(run))
    (tmp_path / 'alignment.json').write_text(json.dumps({'target_clock_id': 'same-host',
        'native_to_target_offsets': {run['run_id']: run['clock_origin_perf']}}))
    monkeypatch.setattr('sys.argv', ['measurements', '--run', str(tmp_path / 'run.json'),
        '--target', str(tmp_path / 'backend.jsonl'), '--alignment', str(tmp_path / 'alignment.json'),
        '--output', str(tmp_path / 'joined.json')])
    main()
    joined = json.loads((tmp_path / 'joined.json').read_text())
    assert not joined['issues'] and len(joined['requests']) == 2
    for row in joined['requests']:
        assert row['task_id'] == trace.tasks[0].task_id
        assert row['stage_instance_id'] == trace.stages[0].stage_instance_id
        assert row['backend_waiting_sec'] >= 0 and row['scheduled_residence_sec'] >= 0
        assert row['time_sources']['submit_time'] == 'backend_client_observation'
