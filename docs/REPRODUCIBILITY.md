# Reproducibility status

The CPU release validation passes 243 tests on each branch with Python 3.12 and
the pinned dependencies in `requirements-repro.txt`. The latest local check ran
on Windows; GitHub CI runs the same suite on Linux. It checks the entire retained
test suite, JSON schemas,
four synthetic Semantic Trace pipelines, native mock execution, example generation,
optional serialization, and structural matrix expansion. Tests include HTTP wire
protocol fixtures and generation-forwarding regressions. The source/privacy scan also checks all public Markdown and Python comments/docstrings for the English-only release requirement.

These checks execute on CPU, including when the host also has Ascend devices.
They do not constitute a fresh GPU/NPU performance rerun. Real serving launchers
require a separately installed compatible serving stack and local model weights.
Their shell syntax and CLI construction are checked; a new serving instance is
not started as part of the CPU procedure. `prepare_replay.py` requires a real
tokenizer and frozen native model trace; its collection, scenario preparation, and HTTP replay pipeline was additionally
executed with a locally generated WordLevel tokenizer and 12 CPU HTTP fixture
requests. This checks serialization and transport; model-specific tokenization
still depends on that model's files.

## Repeating a hardware experiment

Record model and tokenizer identities, chat-template hash and kwargs, generation
settings, serving/framework versions, device architecture/count, precision,
parallelism, cache settings, root arrivals, workload inputs and seeds. Freeze the
native trace before cross-hardware replay. Resolve the scenario for that exact
bundle and keep `SystemConfig` separate from workload identity. Store all attempts,
validation reports, resolved requests, trace hashes, and raw device/backend counters.

The examples use small synthetic tasks and a coverage example explicitly labeled
as illustrative. Published performance numbers require the corresponding frozen
corpus and hardware protocol. Original machine-specific experiment archives are
kept outside this cleaned core repository.

## Compatibility paths

Legacy topology and full-workflow modules remain because runtime and tests use
them. Their English prompts are examples, not byte-identical copies of older
language-specific prompts. Frozen replay uses recorded payloads, independently
of those example prompts. Optional LangGraph/ReAct, live search APIs, external
Chakra conversion, and hardware profiler integrations require their respective
dependencies and are not certified by the core CPU quick start. TPU has only an
extension interface, not a validated runnable backend.

## Anonymous distribution

The maintained upstream keeps its existing history. New release commits use an
anonymous author identity, and the public source tree excludes private paths,
results, personal backend remotes, and local credentials. Use an anonymized mirror
or the source archive for review; do not submit a clone containing upstream `.git`
metadata. Third-party package names and official dependencies are retained.
