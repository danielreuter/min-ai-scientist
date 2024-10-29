import json

from anyio import Path

PROJECT_ROOT = Path(__file__).parent


def get_seed_ideas() -> list[dict]:
    with open(PROJECT_ROOT / "seed_ideas.json", "r") as f:
        return json.load(f)
