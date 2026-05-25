"""Split documents into overlapping passages."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.models import Chunk


def chunk_document(
    text: str, doc_id: str, source_url: str, title: str, chunk_size: int, overlap: int
) -> list[Chunk]:
    if not text or not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    pieces = splitter.split_text(text)
    return [
        Chunk(
            id=f"{doc_id}::{i}",
            text=piece,
            source_url=source_url,
            title=title,
            doc_id=doc_id,
            chunk_index=i,
        )
        for i, piece in enumerate(pieces)
    ]
