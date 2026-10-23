"""Resolve a real-model replay scenario and its separate system declaration."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from app.semantic.canonicalize import TransformersTokenizer
from app.semantic.scenario import ScenarioManifest, SystemConfig
from app.semantic.store import TraceBundle
from app.semantic.validators import RunValidator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--backend", choices=("vllm-gpu", "vllm-ascend"), required=True)
    parser.add_argument("--backend-version", required=True, help="Actual serving-runtime version, not the client package version")
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--accelerators", type=int, default=1)
    parser.add_argument("--cache-policy", choices=("preserve_native", "reuse_scope_isolated"), default="preserve_native")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    trace = TraceBundle(args.trace).read()
    tokenizer = TransformersTokenizer(args.tokenizer)
    first = next(op.llm for op in trace.operations if op.llm)
    model = first["canonical_request"]["model"]
    scenario = ScenarioManifest("native-frozen-work", "1", [trace.metadata.trace_id],
        {trace.metadata.trace_id: trace.metadata.workload_hash}, trace.metadata.workload_version,
        {"model_identity": model, "tokenizer_hash": tokenizer.identity["hash"],
         "chat_template_hash": tokenizer.chat_template["hash"], "generation_parameters": {}},
        {"fidelity_mode": "length_locked", "tool_replay_policy": "recorded",
         "external_delay_policy": "deterministic", "cache_policy": args.cache_policy},
        {"mode": "trace_offsets", "concurrency": 1, "random_seed": 42,
         "task_mix": {trace.metadata.trace_id: 1.0}},
        {"task_count": len(trace.tasks), "task_slo_sec": 30, "slo_attainment_target": .9, "warmup_policy": "none"},
        {"error_handling": "invalidate", "truncation_policy": "forbid",
         "context_overflow_policy": "invalidate", "cancellation_policy": "invalidate"})
    scenario.resolve([s.reuse_scope_id for s in trace.sessions if s.reuse_scope_id], trace, tokenizer.encode_request)
    system = SystemConfig(args.backend, args.backend_version, args.hardware, args.accelerators,
        model_identity=model, tokenizer_hash=tokenizer.identity["hash"], chat_template_hash=tokenizer.chat_template["hash"],
        capabilities={"supports_length_lock": True, "supports_token_lock": False})
    report = RunValidator().validate(trace, scenario, system)
    if not report.run_valid:
        raise ValueError(report.to_dict())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("scenario.json", scenario), ("system.json", system)):
        (output / name).write_text(json.dumps(asdict(value), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
