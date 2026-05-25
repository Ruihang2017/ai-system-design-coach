from app.rag.citation_checker import check_and_enforce
from app.constants import REFUSAL_MESSAGE


def test_valid_answer_passes_through():
    answer, cites, refused = check_and_enforce("Use a reranker [1] and hybrid search [2].", num_chunks=3)
    assert refused is False and cites == [1, 2]


def test_no_citation_downgrades_to_refusal():
    answer, cites, refused = check_and_enforce("Just trust me, it works.", num_chunks=3)
    assert refused is True and answer == REFUSAL_MESSAGE and cites == []


def test_out_of_range_citation_downgrades_to_refusal():
    answer, cites, refused = check_and_enforce("As shown [9].", num_chunks=3)
    assert refused is True and answer == REFUSAL_MESSAGE


def test_explicit_refusal_string_is_respected():
    answer, cites, refused = check_and_enforce(REFUSAL_MESSAGE, num_chunks=3)
    assert refused is True and answer == REFUSAL_MESSAGE
