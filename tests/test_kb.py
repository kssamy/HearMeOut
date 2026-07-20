"""Chunker, extraction, ingestion (incl. re-ingest replacement), hybrid search."""

from __future__ import annotations

import itertools
from pathlib import Path

from conftest import FakeEmbedder
from meetnotes.db import Database
from meetnotes.kb.chunker import chunk_blocks
from meetnotes.kb.extract import Block, extract_file
from meetnotes.kb.ingest import ingest_meetings, ingest_paths


def block(text: str, heading: str = "", page: int | None = None, start: int = 0) -> Block:
    return Block(text=text, heading=heading, page=page, start_char=start)


# -- chunker ------------------------------------------------------------------


def test_chunker_prefers_heading_boundaries() -> None:
    blocks = [
        block("Intro paragraph.", heading="Intro"),
        block("Pricing details here.", heading="Pricing"),
        block("More pricing.", heading="Pricing"),
    ]
    chunks = chunk_blocks(blocks, chunk_chars=2000, overlap_chars=100)
    assert [c.heading for c in chunks] == ["Intro", "Pricing"]
    assert chunks[1].text == "Pricing details here.\n\nMore pricing."


def test_chunker_splits_long_block_with_overlap() -> None:
    sentences = " ".join(f"Sentence number {i} is right here." for i in range(100))
    chunks = chunk_blocks([block(sentences)], chunk_chars=400, overlap_chars=80)
    assert len(chunks) > 3
    for prev, nxt in itertools.pairwise(chunks):
        overlap = prev.text[-40:]
        assert overlap[:20] in prev.text
        # The next chunk starts with the tail of the previous one (the overlap).
        assert nxt.text[:20] in prev.text
    assert all(len(c.text) <= 400 + 80 for c in chunks)


def test_chunker_packs_small_blocks_until_limit() -> None:
    blocks = [block(f"Paragraph {i} " + "x" * 90) for i in range(10)]
    chunks = chunk_blocks(blocks, chunk_chars=350, overlap_chars=0)
    assert 3 <= len(chunks) <= 4
    joined = "\n\n".join(c.text for c in chunks)
    for i in range(10):
        assert f"Paragraph {i}" in joined


# -- extraction ---------------------------------------------------------------


def test_markdown_extraction_breadcrumbs(tmp_path: Path) -> None:
    md = tmp_path / "faq.md"
    md.write_text(
        "# Widget FAQ\n\n## Pricing\n\n### Enterprise\n\nCosts a lot.\n\n## Support\n\nEmail us.\n"
    )
    doc = extract_file(md)
    assert doc.title == "Widget FAQ"
    headings = [b.heading for b in doc.blocks]
    assert "Widget FAQ > Pricing > Enterprise" in headings
    assert "Widget FAQ > Support" in headings


def test_html_extraction_strips_markup(tmp_path: Path) -> None:
    html = tmp_path / "page.html"
    html.write_text(
        "<html><head><style>b{}</style><script>x()</script></head>"
        "<body><h1>Title</h1><p>Real content here.</p></body></html>"
    )
    doc = extract_file(html)
    text = " ".join(b.text for b in doc.blocks)
    assert "Real content here." in text
    assert "x()" not in text and "b{}" not in text


# -- ingestion ----------------------------------------------------------------


async def test_ingest_and_reingest_replaces_chunks(db: Database, tmp_path: Path) -> None:
    doc = tmp_path / "notes.md"
    doc.write_text("# Widget Pricing\n\nEnterprise costs $99 per seat.\n")
    embedder = FakeEmbedder() if db.vec_available else None

    result = await ingest_paths(db, embedder, [tmp_path])
    assert result.ingested == ["Widget Pricing"]
    docs = await db.list_kb_docs()
    assert len(docs) == 1
    first_chunks = docs[0]["chunks"]
    assert first_chunks >= 1

    # Unchanged file → skipped.
    result = await ingest_paths(db, embedder, [doc])
    assert result.skipped and not result.ingested

    # Changed file → replaced (still one doc, new content searchable).
    doc.write_text("# Widget Pricing\n\nEnterprise now costs $149 per seat.\n")
    result = await ingest_paths(db, embedder, [doc])
    assert result.ingested == ["Widget Pricing"]
    docs = await db.list_kb_docs()
    assert len(docs) == 1
    hits = await db.kb_keyword_search("149", k=5)
    assert hits
    old_hits = await db.kb_keyword_search("99", k=5)
    chunks = await db.get_kb_chunks(old_hits)
    assert all("$99" not in c["text"] for c in chunks)


async def test_ingest_meetings_as_docs(db: Database) -> None:
    await db.create_meeting("m1", "Q3 Kickoff", 0)
    await db.add_segment(
        "s1",
        "m1",
        speaker=0,
        channel=0,
        start_ms=0,
        end_ms=1000,
        text="we shipped the widget",
        confidence=0.9,
    )
    await db.end_meeting("m1", 1000)
    result = await ingest_meetings(db, FakeEmbedder() if db.vec_available else None)
    assert result.ingested == ["Q3 Kickoff"]
    # Second run: unchanged.
    result = await ingest_meetings(db, None)
    assert result.skipped == ["Q3 Kickoff"]


# -- hybrid search ------------------------------------------------------------


async def test_keyword_and_vector_search(db: Database, tmp_path: Path) -> None:
    (tmp_path / "pricing.md").write_text(
        "# Acme Pricing\n\nThe enterprise plan price is $99 per seat per month.\n"
    )
    (tmp_path / "recipes.md").write_text("# Cookbook\n\nBoil the pasta for nine minutes.\n")
    embedder = FakeEmbedder() if db.vec_available else None
    await ingest_paths(db, embedder, [tmp_path])

    hits = await db.kb_keyword_search("enterprise price", k=8)
    assert hits
    top = (await db.get_kb_chunks(hits[:1]))[0]
    assert top["doc_title"] == "Acme Pricing"

    if db.vec_available and embedder is not None:
        query_vec = embedder.embed(["what is the enterprise price"])[0]
        vhits = await db.kb_vector_search(query_vec, k=4)
        assert vhits
        best = (await db.get_kb_chunks([vhits[0][0]]))[0]
        assert best["doc_title"] == "Acme Pricing"
        assert -1.0 <= vhits[0][1] <= 1.0

    terms = await db.kb_entity_terms()
    assert "Acme Pricing" in terms
