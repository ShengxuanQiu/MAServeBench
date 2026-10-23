'Load agent system prompts from prompts/*.md.'

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    'Load a prompt by name, for example load_prompt("planner").'
    path = PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"缺少 prompt 文件: {path}")
    return path.read_text(encoding="utf-8")

