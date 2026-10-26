import copy
import json
import pytest
import asyncio
from types import SimpleNamespace

from app.semantic.token_replay import validate_token_request, VLLMEngineBackend
from app.semantic.model import OperationRecord
from app.semantic.canonicalize import ByteTokenizer, canonicalize_native_trace
from app.semantic.replay import MockExactBackend, ReplayExecutor
from app.semantic.golden import build_golden, golden_scenario, golden_system
from app.llm_client import LocalLLMClient


def valid():
    return {"recorded_output_token_ids":[1,9,2],"recorded_output_length":3,
            "recorded_output_token_source":"backend_token_ids"}


@pytest.mark.parametrize("change", [
    {"recorded_output_token_ids":None}, {"recorded_output_token_ids":[]},
    {"recorded_output_length":2}, {"recorded_output_token_source":"retokenized_output_text"},
    {"recorded_output_token_ids":[1,10,2]}, {"recorded_output_token_ids":[1,-1,2]},
    {"recorded_output_token_ids":[True,9,2]},
])
def test_reject_nonexact_or_invalid_ids(change):
    llm={**valid(),**change}
    with pytest.raises(ValueError):
        validate_token_request(llm,[1,2],vocab_size=10,max_model_len=8)


def test_never_truncates_and_preserves_special_ids():
    assert validate_token_request(valid(),[1,2],vocab_size=10,max_model_len=5)==[1,9,2]
    with pytest.raises(ValueError,match="truncation"):
        validate_token_request(valid(),[1,2],vocab_size=10,max_model_len=4)


def test_conflicting_request_features_fail_before_generation():
    backend=object.__new__(VLLMEngineBackend)
    backend.vocab_size=10;backend.max_model_len=8
    backend._sampling_params=lambda **kw:kw
    op=OperationRecord("op","llm","task","stage","session",llm=valid())
    for key,value in [("stop","x"),("n",2),("structured_outputs",{}),("bad_words",["x"]),
                      ("truncate_prompt_tokens",2),("thinking_token_budget",1)]:
        with pytest.raises(ValueError,match="Unsupported generation"):
            backend._parameters({key:value},op,"token_locked",[1,2])
    params=backend._parameters({},op,"token_locked",[1,2])
    assert params["trace_decode_token_ids"]==[1,9,2]
    assert params["max_tokens"]==params["min_tokens"]==3
    assert params["ignore_eos"] is True


def test_replay_retains_full_arrays_and_detects_cross_request_tokens(tmp_path):
    trace=build_golden("fork_join",tmp_path/"trace")
    scenario=golden_scenario(trace);scenario.replay["fidelity_mode"]="token_locked"
    scenario.scenario_hash=scenario.compute_hash()
    good=ReplayExecutor(trace,scenario,golden_system(),MockExactBackend()).run()
    assert good["run_valid"]
    assert all("output_token_ids" in x for x in good["operations"] if "output_length" in x)
    class Crossed(MockExactBackend):
        def generate(self,*args):
            r=super().generate(*args)
            r.output_token_ids=list(reversed(r.output_token_ids))
            return r
    assert not ReplayExecutor(trace,scenario,golden_system(),Crossed()).run()["run_valid"]


def test_old_trace_does_not_silently_downgrade(tmp_path):
    trace=build_golden("fork_join",tmp_path/"trace")
    scenario=golden_scenario(trace)
    for op in trace.operations:
        if op.llm:
            op.llm.pop("recorded_output_token_ids",None)
    trace.finalize_hashes()
    scenario.trace_hashes={trace.metadata.trace_id:trace.metadata.workload_hash}
    scenario.scenario_hash=scenario.compute_hash()
    assert ReplayExecutor(trace,scenario,golden_system(),MockExactBackend()).run()["run_valid"]
    scenario.replay["fidelity_mode"]="token_locked";scenario.scenario_hash=scenario.compute_hash()
    assert not ReplayExecutor(trace,scenario,golden_system(),MockExactBackend()).run()["run_valid"]


@pytest.mark.parametrize("ids,usage,valid", [([8,9],2,True),([8],2,False),(None,2,False),([True,9],2,False)])
def test_stream_capture_uses_backend_ids_including_empty_text_delta(monkeypatch,ids,usage,valid):
    chunks=[{"choices":[{"delta":{"content":"done"}}]},
            {"choices":[{"delta":{},"finish_reason":"stop",**({"token_ids":ids} if ids is not None else {})}],
             "usage":{"completion_tokens":usage}}]
    class Response:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def __iter__(self):return iter(("data: "+json.dumps(c)+"\n").encode() for c in chunks)
    class Opener:
        def open(self,*a,**kw):return Response()
    monkeypatch.setattr("app.llm_client.urllib.request.build_opener",lambda *a:Opener())
    client=LocalLLMClient()
    call=lambda:client.invoke_streaming_with_metadata("s","u",metadata={"_request_payload":{"return_token_ids":True}})
    if valid:
        text,metadata=call();assert text=="done" and metadata["output_token_ids"]==ids
    else:
        with pytest.raises(ValueError):call()


def test_requested_native_id_capture_cannot_fall_back(monkeypatch):
    client=LocalLLMClient()
    def fail(*a,**kw):raise ValueError("no native IDs")
    monkeypatch.setattr(client,"invoke_streaming_with_metadata",fail)
    with pytest.raises(RuntimeError,match="fallback"):
        client.invoke_with_metadata("s","u",metadata={"_request_payload":{"return_token_ids":True}})


def test_native_canonicalizer_retains_original_ids_instead_of_text_encoding(tmp_path):
    from app.motifs import build_workflow
    from app.runtime import WorkloadConfig
    from app.llm_backends import MockLLM
    class Captured(MockLLM):
        def invoke(self,*a,**kw):
            result=super().invoke(*a,**kw)
            result.request_metadata.update(output_token_ids=[3,7],backend_completion_tokens=2,backend_finish_reason="length")
            return result
    cfg=WorkloadConfig(topology_name="parallel_aggregate",run_id="native_capture_unit",
        task_id="task",instance_id="input",query="unit fixture",trace_dir=tmp_path/"native",
        search_provider="synthetic",export_trace_views=False,num_agents=2)
    runner=build_workflow(cfg);runner.llm=Captured();summary=runner.run()
    trace=canonicalize_native_trace(summary["trace_path"],tmp_path/"bundle",tokenizer=ByteTokenizer(),workload_id="unit")
    for op in trace.operations:
        if op.llm:
            assert op.llm["recorded_output_token_ids"]==[3,7]
            assert op.llm["recorded_output_token_source"]=="backend_token_ids"


def test_failed_engine_request_is_aborted_without_returning_partial_output():
    calls=[]
    class Engine:
        async def generate(self,prompt,params,request_id):
            calls.append(request_id)
            raise RuntimeError("interrupted request")
            yield
        async def abort(self,request_id):calls.append(request_id)
    backend=object.__new__(VLLMEngineBackend);backend.engine=Engine()
    with pytest.raises(RuntimeError,match="interrupted"):
        asyncio.run(backend._generate([1,2],None))
    assert len(calls)==2 and calls[0]==calls[1]


def test_engine_prompt_truncation_is_never_accepted():
    class Engine:
        async def generate(self,prompt,params,request_id):
            yield SimpleNamespace(finished=True,prompt_token_ids=[1],outputs=[
                SimpleNamespace(text="x",token_ids=[3],finish_reason="length",stop_reason=None)])
    backend=object.__new__(VLLMEngineBackend);backend.engine=Engine()
    with pytest.raises(ValueError,match="truncated"):
        asyncio.run(backend._generate([1,2],None))
