from pathlib import Path
import yaml
from app.evals.dataset import load_dataset
from app.evals.models import QUESTION_TYPES

DATASET = Path(__file__).resolve().parents[1] / "app" / "evals" / "golden" / "dataset.yaml"
SOURCES = Path(__file__).resolve().parents[1] / "docs" / "sources.yaml"


def _ingested_urls() -> set[str]:
    data = yaml.safe_load(SOURCES.read_text(encoding="utf-8"))
    return {s["url"] for s in data["sources"]}


def test_dataset_loads_and_is_well_formed():
    qs = load_dataset(DATASET)
    assert 100 <= len(qs) <= 120
    ids = [q.id for q in qs]
    assert len(ids) == len(set(ids)), "ids must be unique"
    types = {q.type for q in qs}
    assert types == QUESTION_TYPES, f"all 7 types must appear, got {types}"
    refusal_like = [q for q in qs if q.type in ("refusal", "adversarial")]
    assert len(refusal_like) >= 15


def test_refusal_and_adversarial_consistency():
    qs = load_dataset(DATASET)
    for q in qs:
        if q.type in ("refusal", "adversarial"):
            assert q.should_refuse is True
            assert q.required_sources == []
        else:
            assert q.should_refuse is False
            assert q.required_sources, f"{q.id} grounded must have required_sources"


def test_required_sources_are_all_ingested():
    qs = load_dataset(DATASET)
    valid = _ingested_urls()
    for q in qs:
        for src in q.required_sources:
            assert src in valid, f"{q.id} references un-ingested source {src}"
