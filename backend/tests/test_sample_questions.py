from pathlib import Path

import yaml


def test_twenty_questions_with_refusal_cases():
    path = Path(__file__).resolve().parents[1] / "scripts" / "sample_questions.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    qs = data["questions"]
    assert len(qs) == 20
    assert all(q.get("text") and q.get("type") for q in qs)
    refusal_like = [q for q in qs if q["type"] in ("refusal", "adversarial")]
    assert len(refusal_like) >= 2
