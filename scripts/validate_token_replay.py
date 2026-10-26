"""Finite real-device gate: native workflows, exact IDs, paired same-runner cost.

Run once with each cache setting in separate processes (cold engine). Results
and actual native event traces are written to a fresh output directory.
"""
from pathlib import Path
import argparse, copy, json, time
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor

from app.llm_backends import LLMResult
from app.motifs import build_workflow
from app.runtime import WorkloadConfig
from app.semantic.canonicalize import canonicalize_native_trace
from app.semantic.model import sha256_json
from app.semantic.replay import ReplayExecutor
from app.semantic.scenario import ScenarioManifest, SystemConfig
from app.semantic.store import TraceBundle
from app.semantic.token_replay import VLLMEngineBackend


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf8")


class NativeLLM:
    mode = "openai_compatible"

    def __init__(self, backend, model):
        self.backend, self.model = backend, model

    def invoke(self, system_prompt, user_prompt, metadata):
        request = {"model": self.model, "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}],
            "temperature": 0.2, "max_tokens": 96, "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False}}
        ids = self.backend.tokenizer.encode_request(request)
        params = self.backend._sampling_params(temperature=.2, max_tokens=96, seed=42)
        start = time.time()
        result = self.backend._submit(self.backend._generate(ids, params))
        end = time.time()
        # These IDs come directly from RequestOutput, never from decoded text.
        return LLMResult(result.output_text, result.observations["backend_response_id"],
            0., start, start, start, end, end-start, {
                "output_token_ids": result.output_token_ids,
                "backend_prompt_tokens": len(ids), "backend_completion_tokens": result.output_length,
                "backend_finish_reason": result.observations["finish_reason"],
                "backend_stop_reason": result.observations["stop_reason"],
                "native_request": request, "engine_output_updates": result.observations["engine_output_updates"]})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine-config", required=True);p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path(args.engine_config).read_text())
    save(out/"engine_config.json", config)
    backend=VLLMEngineBackend(config,args.tokenizer)
    model=config["model"]
    native=NativeLLM(backend,model);reports=[]
    try:
        for name,spec in [
            ("fork_join", {"family":"parallel_aggregate", "width":2}),
            ("fork_join_refinement", {"stages":[
                {"id":"parallel","family":"parallel_aggregate","width":2},
                {"id":"review","family":"evaluate_refine","width":1,"max_revisions":1,
                 "inputs":{"candidate":[{"source":"parallel.result","delivery":{"mode":"full"}}]}}]})]:
            for roots in [1,3]:
                def capture(index):
                    dest=out/f"native_{name}_{roots}_{index}"
                    cfg=WorkloadConfig(topology_name=spec.get("family","composition"),
                        run_id=f"native_{name}_{roots}_{index}", task_id="task", instance_id="input",
                        query="Explain two practical ways to conserve water and combine their benefits.",
                        trace_dir=dest, model=model, search_provider="synthetic", export_trace_views=False,
                        max_output_tokens=96, max_concurrent_llm_calls=16,
                        extra={"workload_spec":copy.deepcopy(spec), "generation":{
                            "temperature":.2,"max_tokens":96,"seed":42,"chat_template_kwargs":{"enable_thinking":False}}})
                    wf=build_workflow(cfg);wf.llm=native
                    summary=wf.run()
                    assert summary["status"] in {"accepted","completed"},summary
                    # The runtime records its actual model request. Verify it
                    # against the native engine invocation before collection.
                    path=Path(summary["trace_path"])
                    events=[json.loads(x) for x in path.read_text().splitlines() if x]
                    for e in events:
                        if e.get("request_metadata",{}).get("native_request"):
                            assert e["request_metadata"]["output_token_ids"]
                    trace=canonicalize_native_trace(path,out/f"trace_{name}_{roots}_{index}",
                        tokenizer=backend.tokenizer,workload_id=name)
                    for op in trace.operations:
                        if op.llm:
                            assert op.llm["recorded_output_token_source"]=="backend_token_ids"
                            end=next(e for e in events if e.get("operation_id",e.get("node_id"))==op.operation_id
                                     and e.get("request_metadata",{}).get("native_request"))
                            source=end["request_metadata"]["native_request"]
                            assert backend.tokenizer.encode_request(source)==op.llm["input_token_ids"]
                    if name=="fork_join_refinement":
                        assert any(op.data_dependencies for op in trace.operations if op.llm)
                    return trace,out/f"trace_{name}_{roots}_{index}"
                with ThreadPoolExecutor(max_workers=roots) as pool:
                    captured=list(pool.map(capture,range(roots)))
                # Real concurrently generated roots remain individually traced;
                # replay all roots concurrently, preserving each native graph.
                for block in range(3):
                    pair={}
                    for mode in (["length_locked","token_locked"] if block%2==0 else ["token_locked","length_locked"]):
                        cache_reset = backend._submit(backend.engine.reset_prefix_cache())
                        if config.get("enable_prefix_caching"):
                            assert cache_reset, "Cold prefix-cache boundary failed"
                        def replay(item):
                            trace,path=item
                            scenario=ScenarioManifest("acceptance","1",[trace.metadata.trace_id],
                                {trace.metadata.trace_id:trace.metadata.workload_hash},trace.metadata.workload_version,
                                {"model_identity":model,"tokenizer_hash":backend.tokenizer.identity["hash"],
                                 "chat_template_hash":backend.tokenizer.chat_template["hash"],"generation_parameters":{}},
                                {"fidelity_mode":mode,"tool_replay_policy":"recorded","external_delay_policy":"deterministic","cache_policy":"preserve_native"},
                                {"mode":"trace_offsets","concurrency":roots,"random_seed":42,"task_mix":{trace.metadata.trace_id:1}},
                                {"task_count":1,"task_slo_sec":600,"slo_attainment_target":.9,"warmup_policy":"none"},
                                {"error_handling":"invalidate","truncation_policy":"forbid","context_overflow_policy":"invalidate","cancellation_policy":"invalidate"})
                            scenario.resolve([],trace,backend.tokenizer.encode_request)
                            system=SystemConfig("vllm-gpu","0.30.0","RTX PRO 5000",1,model,
                                backend.tokenizer.identity["hash"],backend.tokenizer.chat_template["hash"])
                            result=ReplayExecutor(trace,scenario,system,backend,TraceBundle(path).artifacts).run()
                            assert result["run_valid"],result.get("invalid_reasons")
                            if mode=="token_locked":
                                for r in result["operations"]:
                                    op=next(x for x in trace.operations if x.operation_id==r["operation_id"])
                                    if op.llm:
                                        assert r["output_token_ids"]==op.llm["recorded_output_token_ids"]
                                        assert r["target_observations"]["engine_output_updates"]>1
                            save(out/f"runs/{name}_{roots}_b{block}_{mode}_{trace.metadata.trace_id}.json",result)
                            return result
                        start=time.perf_counter()
                        with ThreadPoolExecutor(max_workers=roots) as pool:results=list(pool.map(replay,captured))
                        pair[mode]=dict(wall_sec=time.perf_counter()-start,roots=roots,
                            cold_prefix_cache_reset=cache_reset,
                            requests=sum(sum("output_length" in x for x in r["operations"]) for r in results))
                    reports.append(dict(workflow=name,roots=roots,block=block,**pair))
                    save(out/"paired_overhead.json",reports)
        # Independent special-termination fixture: do not edit any native trace.
        op=copy.deepcopy(next(x for x in captured[0][0].operations if x.llm))
        eos=backend.tokenizer._tokenizer.eos_token_id
        fixture=[eos,13,eos]
        op.llm.update(recorded_output_token_ids=fixture,recorded_output_length=len(fixture),recorded_output_token_source="backend_token_ids")
        result=backend.generate(op.llm["canonical_request"],op,"token_locked")
        assert result.output_token_ids==fixture
        save(out/"special_termination_fixture.json",dict(expected=fixture,result=asdict(result),scope="Explicit API edge-case fixture, separate from native workflow evidence"))
        save(out/"acceptance.json",dict(passed=True,cache=config.get("enable_prefix_caching"),
            native_source="real native FamilyWorkload execution; original RequestOutput.token_ids",
            comparisons=len(reports),paired_blocks=3,engine_config_hash=sha256_json(config)))
    finally:backend.close()


if __name__=="__main__":main()
