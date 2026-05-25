"""Validate citations and enforce the answer contract (refuse on violation)."""

from app.constants import REFUSAL_MESSAGE
from app.rag.generator import parse_citations


def check_and_enforce(answer: str, num_chunks: int) -> tuple[str, list[int], bool]:
    """Return (answer, citations, refused).

    Downgrades to the canonical refusal when the answer cites nothing or cites
    a source number outside the retrieved range.
    """
    if answer.strip() == REFUSAL_MESSAGE:
        return REFUSAL_MESSAGE, [], True
    citations = parse_citations(answer)
    if not citations or any(c < 1 or c > num_chunks for c in citations):
        return REFUSAL_MESSAGE, [], True
    return answer, citations, False
