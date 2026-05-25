from app.rag.fetcher import extract_main_content, fetch_url

HTML = """
<html><head><title>RAG Basics</title></head><body>
<nav>menu menu menu</nav>
<article>
<h1>Retrieval Augmented Generation</h1>
<p>RAG combines a retriever with a generator so that answers are grounded in documents.</p>
<p>A vector database stores embeddings and supports nearest neighbor search for retrieval.</p>
<p>Chunking splits documents into passages so retrieval returns focused context windows.</p>
</article>
<footer>copyright</footer></body></html>
"""


def test_extract_main_content_gets_body_text():
    md = extract_main_content(HTML)
    assert md and "Retrieval Augmented Generation" in md
    assert "grounded in documents" in md


def test_extract_fallback_on_minimal_html():
    md = extract_main_content("<html><body><p>Just one short line of text here.</p></body></html>")
    assert md and "short line of text" in md


def test_fetch_url_uses_http(httpx_mock):
    httpx_mock.add_response(url="https://example.com/doc", text=HTML)
    md = fetch_url("https://example.com/doc")
    assert md and "vector database" in md
