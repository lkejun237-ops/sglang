"""Keep private/generated release output outside this Git checkout."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def outside_checkout(path):
    resolved = Path(path).resolve()
    if resolved == REPO or resolved.is_relative_to(REPO):
        raise ValueError("Private release output must be outside the Git checkout")
    return resolved
