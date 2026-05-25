from app.rag.chunker import chunk_document


def test_chunk_count_and_stable_ids():
    text = "word " * 1000  # ~5000 chars
    chunks = chunk_document(text, doc_id="d1", source_url="u", title="T", chunk_size=1000, overlap=100)
    assert len(chunks) >= 4
    assert chunks[0].id == "d1::0" and chunks[1].id == "d1::1"
    assert all(c.doc_id == "d1" and c.title == "T" for c in chunks)
    assert all(len(c.text) <= 1000 for c in chunks)


def test_empty_text_yields_no_chunks():
    assert chunk_document("", "d", "u", "T", 1000, 100) == []
