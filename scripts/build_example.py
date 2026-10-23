"""Write a portable, embedded experiment using the public benchmark API."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from app.publication import preset
from app.specs import DeploymentSpec, TaskBinding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("matched_parallel", "matched_chain", "parallel_refine", "peer"), default="parallel_refine")
    parser.add_argument("--backend", choices=("mock", "openai_compatible"), default="mock")
    parser.add_argument("--model", default="mock-mas-model")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--users", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    deployment = DeploymentSpec(model=args.model, backend=args.backend, endpoint=args.endpoint,
        generation={"max_tokens": 256, "temperature": 0, "seed": 42}, concurrency=32)
    experiment = preset(args.preset, TaskBinding("Compare three practical ways to reduce household energy use."), deployment)
    value = asdict(experiment)
    value["load"] = args.users
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
