"""Clone a Semantic Trace bundle with one explicitly justified ordering edge."""
import argparse
import shutil
from pathlib import Path

from app.semantic.interventions import add_optional_serialization
from app.semantic.store import TraceBundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--stage-meaning", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = TraceBundle(args.trace)
    trace = add_optional_serialization(source.read(), args.source, args.target,
        stage_meaning=args.stage_meaning)
    output = Path(args.output)
    if output.exists():
        raise ValueError("Output must be a new bundle directory")
    if (source.root / "artifacts").exists():
        shutil.copytree(source.root / "artifacts", output / "artifacts")
    TraceBundle(output).write(trace)
    print(trace.metadata.workload_hash)


if __name__ == "__main__":
    main()
