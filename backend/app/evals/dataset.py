"""Load a golden evaluation dataset from a YAML file."""

from pathlib import Path

import yaml

from app.evals.models import GoldenQuestion


def load_dataset(path: str | Path) -> list[GoldenQuestion]:
    """Load and validate a YAML golden dataset.

    The YAML file must have shape::

        questions:
          - id: ...
            type: ...
            question: ...
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [GoldenQuestion(**q) for q in data["questions"]]
