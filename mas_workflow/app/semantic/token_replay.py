"""Opt-in exact-token replay through vLLM's official AsyncLLM interface.

No serving package is patched. The ordinary OpenAI length adapter is separate.
See docs/TOKEN_REPLAY.md for the measured backend/model support matrix.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import uuid
from typing import Any

from .canonicalize import TransformersTokenizer
from .model import SCHEMA_VERSION
from .replay import BackendResult


def validate_engine_profile(config: dict[str, Any]) -> None:
    if not config.get("enable_trace_replay"):
        raise ValueError("Engine config must explicitly enable_trace_replay")
    if config.get("speculative_config") or config.get("async_scheduling") is not False:
        raise ValueError("Set async_scheduling=false explicitly; speculative decoding is unsupported")
    for key in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                "decode_context_parallel_size", "prefill_context_parallel_size"):
        if config.get(key, 1) != 1:
            raise ValueError("The verified profile requires one device: " + key)


def validate_token_request(llm: dict[str, Any], input_ids: list[int], *,
                           vocab_size: int, max_model_len: int) -> list[int]:
    ids = llm.get("recorded_output_token_ids")
    if llm.get("recorded_output_token_source") != "backend_token_ids" or not isinstance(ids, list):
        raise ValueError("token_locked requires native backend_token_ids; re-tokenized text is not exact output")
    if not ids or len(ids) != llm.get("recorded_output_length"):
        raise ValueError("Empty or inconsistent recorded output token sequence")
    for sequence in (input_ids, ids):
        if any(type(x) is not int or not 0 <= x < vocab_size for x in sequence):
            raise ValueError("Token ID outside the target vocabulary")
    if len(input_ids) + len(ids) > max_model_len:
        raise ValueError("Prompt plus recorded output exceeds max_model_len; truncation is forbidden")
    return list(ids)


class VLLMEngineBackend:
    """One official async engine, shared by concurrent replay worker threads.

    Explicit construction only. Both output modes use this same engine and
    runner for paired overhead measurements. The official runner owns request
    IDs, output positions, batch reordering, cache reuse, and preemption state.
    """
    capabilities = {"supports_length_lock": True, "supports_token_lock": True}

    def __init__(self, config: dict[str, Any], tokenizer_path: str, timeout: float = 600):
        import importlib.metadata
        import os
        from vllm import SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.platforms import current_platform

        version = importlib.metadata.version("vllm")
        if version != "0.30.0" or not current_platform.is_cuda():
            raise ValueError("Validated token backend requires CUDA vLLM 0.30.0; Ascend is unsupported")
        if "trace_decode_token_ids" not in SamplingParams.__struct_fields__:
            raise ValueError("Installed vLLM lacks official trace_decode_token_ids")
        validate_engine_profile(config)
        if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "1":
            raise ValueError("Set VLLM_USE_V2_MODEL_RUNNER=1 before starting the engine")
        self.config = dict(config)
        from transformers import AutoConfig
        if AutoConfig.from_pretrained(config["model"]).model_type != "qwen3":
            raise ValueError("The verified model scope is dense Qwen3, not MoE or hybrid models")
        self.timeout = timeout
        self.tokenizer = TransformersTokenizer(tokenizer_path)
        self.vocab_size = len(self.tokenizer._tokenizer)
        self.max_model_len = int(config["max_model_len"])
        self._sampling_params = SamplingParams
        self._engine_args = AsyncEngineArgs(**config)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self.engine = self._submit(self._start())
        actual = self.engine.get_tokenizer()
        if actual.get_vocab() != self.tokenizer._tokenizer.get_vocab():
            self.close()
            raise ValueError("Engine and declared tokenizer vocabularies differ")

    async def _start(self):
        from vllm.v1.engine.async_llm import AsyncLLM
        return AsyncLLM.from_engine_args(self._engine_args)

    def _submit(self, coroutine):
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(timeout=self.timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError("vLLM request timed out; generation cancelled, replay invalid") from None

    def validate_trace(self, trace, scenario, system):
        if system.backend_name != "vllm-gpu":
            raise ValueError("The official token engine must be declared as vllm-gpu")
        if trace.metadata.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported token replay trace version")
        if system.backend_version != "0.30.0":
            raise ValueError("SystemConfig backend version differs from installed vLLM")
        if system.tokenizer_hash != self.tokenizer.identity["hash"] or system.chat_template_hash != self.tokenizer.chat_template["hash"]:
            raise ValueError("Tokenizer or chat template identity differs from the engine")
        for op in trace.operations:
            if op.llm:
                request = scenario.materialize_request(op)
                aliases = self.config.get("served_model_name") or []
                aliases = [aliases] if isinstance(aliases, str) else aliases
                if request.get("model") not in [self.config["model"], *aliases]:
                    raise ValueError("Recorded model identity differs from the configured engine")
                ids = self.tokenizer.encode_request(request)
                if ids != scenario.resolved_request_contracts[op.operation_id]["input_token_ids"]:
                    raise ValueError("Resolved prompt tokens differ from engine tokenization")
                self._parameters(request, op, scenario.replay["fidelity_mode"], ids)

    def _parameters(self, request, operation, mode, input_ids):
        llm = operation.llm
        ignored = {"model", "messages", "stream", "stream_options", "max_tokens", "max_completion_tokens",
                   "min_tokens", "ignore_eos", "chat_template_kwargs", "return_token_ids"}
        allowed = {"temperature", "top_p", "top_k", "min_p", "seed", "presence_penalty", "frequency_penalty", "repetition_penalty"}
        if set(request) - ignored - allowed:
            raise ValueError("Unsupported generation features: " + str(sorted(set(request) - ignored - allowed)))
        target = llm["recorded_output_length"]
        if type(target) is not int or target < 1 or len(input_ids) + target > self.max_model_len:
            raise ValueError("Empty output or context overflow is unsupported; no truncation")
        params = {k: request[k] for k in allowed if k in request}
        params.update(max_tokens=target, min_tokens=target, ignore_eos=True)
        if mode == "token_locked":
            params["trace_decode_token_ids"] = validate_token_request(
                llm, input_ids, vocab_size=self.vocab_size, max_model_len=self.max_model_len)
        elif mode != "length_locked":
            raise ValueError("Unknown replay fidelity mode")
        return self._sampling_params(**params)

    async def _generate(self, input_ids, params):
        import time
        request_id = "masbench-" + uuid.uuid4().hex
        started = time.perf_counter()
        updates = 0
        final = None
        try:
            async for result in self.engine.generate({"prompt_token_ids": input_ids}, params, request_id):
                updates += 1
                final = result
        except BaseException:
            await self.engine.abort(request_id)
            raise
        if final is None or not final.finished or len(final.outputs) != 1:
            raise ValueError("Incomplete or multi-output generation")
        out = final.outputs[0]
        if list(final.prompt_token_ids) != input_ids:
            raise ValueError("Backend changed or truncated prompt token IDs")
        ids = list(out.token_ids)
        return BackendResult(out.text, ids, len(ids), {
            "wall_time_sec": time.perf_counter() - started,
            "request_submit_perf": started, "request_completion_perf": time.perf_counter(),
            "backend_response_id": request_id, "engine_output_updates": updates,
            "prompt_token_ids": list(final.prompt_token_ids),
            "finish_reason": out.finish_reason, "stop_reason": out.stop_reason,
            "token_source": "backend_token_ids"})

    def generate(self, request, operation, mode):
        input_ids = self.tokenizer.encode_request(request)
        params = self._parameters(request, operation, mode, input_ids)
        result = self._submit(self._generate(input_ids, params))
        if mode == "token_locked" and result.output_token_ids != operation.llm["recorded_output_token_ids"]:
            raise ValueError("Official backend did not emit the complete recorded token sequence")
        return result

    def close(self):
        async def shutdown():
            self.engine.shutdown()
        if hasattr(self, "engine"):
            self._submit(shutdown())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=30)
