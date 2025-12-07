import pytest
from src.indexing.pdf_chunker import PdfProcessor

@pytest.fixture
def pdf_processor():
    """Standard processor with a reasonable token limit."""
    return PdfProcessor(token_limit=1000)

@pytest.fixture
def small_pdf_processor():
    """
    Processor with tight limit to test overflow splitting.
    Limit 250 ensures safe_limit (225) > chunk_overlap (100).
    """
    return PdfProcessor(token_limit=250)


def test_clean_artifacts_removes_frontmatter(pdf_processor):
    """Should remove YAML frontmatter typically found in converted markdown."""
    content = """---
title: Research Paper
author: Scientist
---
# Actual Content"""
    
    cleaned = pdf_processor._clean_artifacts(content)
    assert "---" not in cleaned
    assert "author: Scientist" not in cleaned
    assert cleaned.strip() == "# Actual Content"

def test_clean_artifacts_removes_html_spans(pdf_processor):
    """Should remove HTML span tags often left by PDF parsers."""
    content = "This is <span style='color:red'>important</span> text."
    cleaned = pdf_processor._clean_artifacts(content)
    assert cleaned == "This is important text."

def test_clean_artifacts_converts_sup(pdf_processor):
    """Should convert <sup> references to bracket notation."""
    content = "Reference<sup>12</sup>"
    cleaned = pdf_processor._clean_artifacts(content)
    assert cleaned == "Reference[12]"


def test_process_file_protects_block_latex(pdf_processor):
    """Should NOT split inside a block latex formula."""
    # This relies on utils.py correctly handling tags with underscores (LATEX_BLOCK)
    latex_block = "$$\nE = mc^2 + \\int_{0}^{\\infty} x dx\n$$"
    content = f"# Math Section\nHere is a formula:\n{latex_block}"
    
    chunks = pdf_processor.process_file(content, "math.pdf")
    
    assert len(chunks) == 1
    assert latex_block in chunks[0].page_content
    # Metadata flag check
    assert chunks[0].metadata["has_latex"] is True

def test_process_file_protects_inline_latex(pdf_processor):
    """Should detect and protect inline latex."""
    content = "The value is $x = 5$ in this case."
    chunks = pdf_processor.process_file(content, "math.pdf")
    
    assert "$x = 5$" in chunks[0].page_content
    assert chunks[0].metadata["has_latex"] is True


def test_process_file_builds_context_path(pdf_processor):
    """Should build a breadcrumb-style context path from headers."""
    content = """
# Chapter 1
## Section A
### Subsection X
Content here.
    """
    chunks = pdf_processor.process_file(content, "book.pdf")
    
    assert len(chunks) > 0
    # Should capture the hierarchy
    assert chunks[0].metadata["context_path"] == "Chapter 1 > Section A > Subsection X"
    assert "Content here" in chunks[0].page_content

def test_process_file_strict_limit_enforcement(small_pdf_processor):
    """Should split content even if headers are not present, when limit is exceeded."""
    # Generate text larger than 250 tokens
    long_text = "data " * 300 
    content = f"# Big Section\n{long_text}"
    
    chunks = small_pdf_processor.process_file(content, "big.pdf")
    
    assert len(chunks) > 1
    assert chunks[0].metadata["source"] == "big.pdf"
    assert chunks[1].metadata["context_path"] == "Big Section"

def test_process_file_empty_content(pdf_processor):
    """Should handle empty strings gracefully."""
    chunks = pdf_processor.process_file("", "empty.pdf")
    assert chunks == []

def test_build_metadata_structure(pdf_processor):
    """Should create correct metadata dictionary and detect latex."""
    doc_meta = {"h1": "Title", "h2": "Subtitle"}
    filename = "test.pdf"
    content = "Some math $E=mc^2$"
    
    meta = pdf_processor._build_metadata(doc_meta, filename, content)
    
    assert meta["source"] == filename
    assert meta["context_path"] == "Title > Subtitle"
    assert meta["has_latex"] is True