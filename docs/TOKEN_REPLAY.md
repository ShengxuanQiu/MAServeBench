# Explicit output fidelity controls

`length_locked` remains the default. The existing `vllm-openai` adapter and its
request payload are unchanged: real prefill and autoregressive generation with
`min_tokens == max_tokens`, `ignore_eos`, and exact output-length validation.
Recorded downstream inputs and typed dependencies remain authoritative.

`token_locked` uses the **official vLLM engine implementation**, with normal
model execution and sampled-token replacement before the next decode step.
This is prior-method compatibility, following
[XPerf §3.6](https://arxiv.org/html/2608.20370v1#S3.SS6) and
[AgentReplay §3.2](https://arxiv.org/html/2609.32283#S3.SS2), not a new replay
algorithm or MoE router replay. No XPerf source code was copied; no verifiable
public implementation/license was found in the linked paper or repository search.
The adapter calls vLLM's Apache-2.0 interface without modifying installed files.

## Backend scope

| Backend | Engine/runner | Model scope | Exact tokens |
|---|---|---|---|
| Existing `vllm-openai` | Existing GPU or Ascend deployment | Existing model scope | Unsupported; explicit error |
| Opt-in `vllm-engine` | CUDA vLLM 0.30.0, Model Runner V2, one device, synchronous scheduling | Dense Qwen3 | Real-device gate required; see acceptance receipt |
| vLLM-Ascend 0.21.0rc1 / vLLM 0.21.0 | Existing NPU runner | Qwen3-8B | Unsupported; no backend upgrade or patch |
| Synthetic backend | No model execution | Conformance fixtures | Synthetic checks only |

The official online chat protocol in the inspected installation does not expose
`trace_decode_token_ids`. Use its direct engine API through `vllm-engine`.
Do not send trace control data as a chat role or prompt text.

## Native recording and preparation

The native HTTP recorder retains optional backend-returned `token_ids` from SSE
choices when explicitly requested with generation `return_token_ids: true` on a
compatible server. Missing IDs or a usage-length mismatch fail the capture;
fallback clients must not fabricate IDs. Canonicalization preserves these IDs
from request metadata with source `backend_token_ids` and termination metadata.
Decoded text re-tokenization is retained for legacy length traces, labelled as
such, and is rejected by exact-token replay. Existing trace versions and fields
are unchanged.

```bash
python scripts/prepare_replay.py --trace TRACE --tokenizer MODEL_PATH --backend vllm-gpu --backend-version 0.30.0 --hardware DEVICE --fidelity-mode token_locked --output CONFIG_DIR
VLLM_USE_V2_MODEL_RUNNER=1 python -m app.semantic.cli replay --trace TRACE --scenario CONFIG_DIR/scenario.json --system CONFIG_DIR/system.json --backend vllm-engine --engine-config engine.json --tokenizer MODEL_PATH --output replay.json
```

An explicit `engine.json` supplies `model`, `max_model_len`,
`enable_trace_replay: true`, `async_scheduling: false`, and the intended ordinary
engine settings. Nothing starts this adapter implicitly. To compare overhead,
use the same engine/configuration for both fidelity modes.

Preflight checks trace version, model/tokenizer/template identity, full prompt
IDs, output-ID provenance/count/range, and context capacity **before submission**.
Context overflow fails rather than accepting vLLM's automatic trace truncation.
Empty outputs, stop strings, structured/tool-call constraints, speculative
decoding, asynchronous scheduling, multi-device execution, and unrecognized
generation options fail explicitly. Plain recorded tool-result operations are
replayed unchanged. Special token IDs inside a nonempty recorded sequence are
kept: the official trace feature overrides EOS/stop termination and ends at the
recorded length. Full returned ID arrays are saved and compared, never just text.
Request IDs are generated uniquely per invocation; operation IDs remain in the
executor's records. Request failure or timeout invalidates replay and aborts
the engine request. No sampled output is substituted into downstream artifacts.

The real-device gate is `scripts/validate_token_replay.py`. Run cache-on and
cache-off cells in fresh processes and keep every native trace, run record,
engine configuration and paired overhead block. The gate exercises native
Fork–Join and Fork–Join plus Refinement, one and three concurrent roots. It does
not rerun the paper's serving experiments or establish support for another
backend/model. Unsupported resume/feature combinations must not be inferred
from this matrix.

## Verified finite gate (2026-10-27)

Both branches pass 211 CPU tests. On CUDA vLLM 0.30.0 / Runner V2 / Qwen3-4B-Instruct-2507 / RTX PRO 5000 72GB, 16 real native roots supply 66 captured model requests. The gate replays 198 token-locked requests (16077 exact output IDs) and 198 length-locked requests across both cache settings. Every token-locked request emits its full recorded ID array, with one observed engine output update per generated token. Each paired arm starts after a confirmed prefix-cache reset. An independent EOS-containing fixture passes in both cache settings.

Paired wall-time ratios (token/length), three-block median [minimum, maximum]:

| Cache | Workflow | Roots | Ratio |
|---|---|---:|---|
| off | fork_join | 1 | 0.9923 [0.9872, 1.0025] |
| off | fork_join | 3 | 0.9943 [0.9933, 1.0032] |
| off | fork_join_refinement | 1 | 1.0000 [0.9912, 1.0022] |
| off | fork_join_refinement | 3 | 0.9934 [0.9911, 0.9950] |
| on | fork_join | 1 | 0.9954 [0.9859, 0.9966] |
| on | fork_join | 3 | 0.9944 [0.9891, 0.9970] |
| on | fork_join_refinement | 1 | 1.0004 [0.9933, 1.0007] |
| on | fork_join_refinement | 3 | 0.9993 [0.9896, 1.0041] |

These are measured finite-cohort ratios, not a statistical significance claim or a bound for other workloads. Forced KV preemption/resume and external cancellation during live model execution were not stress-tested and are outside this verified profile; interruption/abort behavior has unit coverage. No NPU exact-token support is claimed.
