"""Fetch documentation pages and extract clean main-content markdown."""

import logging

import httpx
import trafilatura
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from app.constants import DEFAULT_UA

logger = logging.getLogger(__name__)


def extract_main_content(html: str) -> str | None:
    """Primary: trafilatura main-content extraction. Fallback: BeautifulSoup text."""
    md = trafilatura.extract(html, output_format="markdown", favor_recall=True)
    if md and md.strip():
        return md.strip()
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return text or None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def fetch_url(url: str, client: httpx.Client | None = None) -> str | None:
    owns_client = client is None
    client = client or httpx.Client(timeout=30, follow_redirects=True, headers={"User-Agent": DEFAULT_UA})
    try:
        resp = client.get(url)
        resp.raise_for_status()
        return extract_main_content(resp.text)
    finally:
        if owns_client:
            client.close()
