import json
import pytest
from src.indexing.markdown_chunker import MarkdownProcessor

@pytest.fixture
def md_processor():
    """Fixture to provide a standard MarkdownProcessor instance."""
    return MarkdownProcessor(token_limit=1000)

@pytest.fixture
def sample_json_file(tmp_path):
    """Creates a temporary JSON file for testing file processing."""
    data = {
        "title": "Quantum Intro",
        "url": "http://example.com",
        "content": "# Intro\nThis is a test.\n## Details\nMore details here.",
        "metadata": {"downloaded_py_files": ["script.py"]}
    }
    file = tmp_path / "test_data.json"
    file.write_text(json.dumps(data), encoding="utf-8")
    return str(file)

def test_convert_html_headers(md_processor):
    """Should convert HTML headers to Markdown syntax."""
    html_text = "<h1>Title</h1>\n<h2>Subtitle</h2>"
    expected = "# Title\n## Subtitle"
    result = md_processor._convert_html_headers(html_text)
    assert result == expected

def test_build_metadata_structure(md_processor):
    """Should correctly construct metadata dictionary."""
    header_meta = {"h1": "Main", "h2": "Sub"}
    original_item = {
        "url": "test_url", 
        "topic": "Physics",
        "metadata": {"downloaded_py_files": ["a.py", "b.py"]}
    }
    content = "Some text with code ```python```"
    
    meta = md_processor._build_metadata(header_meta, original_item, content)
    
    assert meta["source"] == "test_url"
    assert meta["topic"] == "Physics"
    assert meta["context_path"] == "Main > Sub"
    assert meta["related_source_file"] == "a.py, b.py"

def test_process_single_item_splitting(md_processor):
    """Should split content based on markdown headers."""
    item = {
        "content": "# Section 1\nContent 1.\n# Section 2\nContent 2."
    }
    chunks = md_processor._process_single_item(item)
    
    assert len(chunks) == 2
    assert "Section 1" in chunks[0].metadata["h1"]
    assert "Content 1" in chunks[0].page_content
    assert "Section 2" in chunks[1].metadata["h1"]

def test_masking_logic_integration(md_processor):
    """Should verify masking protects blocks during split logic."""
    content = """
    # Real Header
    Here is a bash script:
    ```bash
    # This is a comment, not a header
    echo 'hello'
    ```
    """
    item = {"content": content}
    
    chunks = md_processor._process_single_item(item)
    
    assert len(chunks) == 1
    assert "Real Header" in chunks[0].metadata.get("h1", "")
    assert "echo 'hello'" in chunks[0].page_content 
    assert "bash" in chunks[0].page_content