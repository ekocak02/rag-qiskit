import pytest
import json

from src.indexing.notebook_chunker import NotebookProcessor

@pytest.fixture
def nb_processor():
    """Standard processor with a reasonable token limit."""
    return NotebookProcessor(token_limit=1000)

@pytest.fixture
def small_nb_processor():
    """
    Processor with tight limit to test buffering and large cell logic.
    Limit 250 ensures safe_limit (225) > chunk_overlap (100).
    """
    return NotebookProcessor(token_limit=250)


def test_format_block_code_with_output(nb_processor):
    """Should format code blocks to include their outputs."""
    block = {
        "type": "code",
        "content": "print('hello')",
        "output": "hello"
    }
    formatted = nb_processor._format_block(block)
    assert "print('hello')" in formatted
    assert "Output:" in formatted
    assert "```text\nhello\n```" in formatted

def test_format_block_text(nb_processor):
    """Should return text content as is."""
    block = {"type": "text", "content": "Just text"}
    formatted = nb_processor._format_block(block)
    assert formatted == "Just text"

def test_build_metadata(nb_processor):
    """Should extract version and flags correctly."""
    file_meta = {"filename": "test.ipynb", "qiskit_version": "0.44"}
    content = "Some code ```python``` and math $x$"
    
    meta = nb_processor._build_metadata(file_meta, content)
    
    assert meta["source"] == "test.ipynb"
    assert meta["qiskit_version"] == "0.44"
    assert meta["has_code"] is True
    assert meta["has_latex"] is True


def test_process_file_buffers_small_cells(nb_processor):
    """Should combine multiple small cells into one chunk."""
    # Create 3 small cells
    data = {
        "metadata": {"filename": "test.ipynb"},
        "content": [
            {"type": "text", "content": "Cell 1"},
            {"type": "code", "content": "x=1", "output": ""},
            {"type": "text", "content": "Cell 3"}
        ]
    }
    
    chunks = nb_processor.process_file(json.dumps(data))
    
    assert len(chunks) == 1
    assert "Cell 1" in chunks[0].page_content
    assert "x=1" in chunks[0].page_content
    assert "Cell 3" in chunks[0].page_content
    assert chunks[0].metadata["cell_range"] == "0-2"

def test_process_file_flushes_buffer_on_limit(small_nb_processor):
    """Should flush buffer when it exceeds the limit."""
    # Create cells that are small individually but large together
    # Limit is 250 tokens.
    medium_text = "word " * 60 # Approx 60 tokens
    
    data = {
        "metadata": {"filename": "test.ipynb"},
        "content": [
            {"type": "text", "content": medium_text},
            {"type": "text", "content": medium_text},
            {"type": "text", "content": medium_text},
            {"type": "text", "content": medium_text},
            {"type": "text", "content": medium_text}  
        ]
    }
    
    chunks = small_nb_processor.process_file(json.dumps(data))
    
    # Expect at least 2 chunks
    assert len(chunks) >= 2
    # Verify strategy metadata
    assert chunks[0].metadata["strategy"] == "buffer"

def test_process_large_cell_split(small_nb_processor):
    """Should split a single large cell that exceeds limit."""
    # A single cell larger than limit (250)
    large_text = "# Header\n" + ("content " * 300)
    
    data = {
        "metadata": {"filename": "large.ipynb"},
        "content": [
            {"type": "text", "content": large_text}
        ]
    }
    
    chunks = small_nb_processor.process_file(json.dumps(data))
    
    # Should be split
    assert len(chunks) > 1
    assert chunks[0].metadata["strategy"] == "markdown_header"
    assert chunks[0].metadata["cell_index"] == 0

def test_process_invalid_json(nb_processor):
    """Should handle invalid JSON gracefully."""
    chunks = nb_processor.process_file("{invalid_json")
    assert chunks == []