import os
import json
import pytest
from unittest.mock import MagicMock, patch
from bs4 import BeautifulSoup, Tag

from src.ingestion.api_docs import GitHubHandler, ContentParser, QiskitScraper, Config


@pytest.fixture
def temp_download_dir(tmp_path):
    """Creates a temporary directory for download tests."""
    d = tmp_path / "downloads"
    d.mkdir()
    return str(d)

@pytest.fixture
def github_handler(temp_download_dir):
    """Returns a GitHubHandler instance with a temp dir."""
    return GitHubHandler(download_dir=temp_download_dir)

@pytest.fixture
def content_parser(github_handler):
    """Returns a ContentParser instance linked to the handler."""
    return ContentParser(github_handler)


def test_is_github_source_link_true(github_handler):
    """Should return True for valid GitHub source links."""
    tag = Tag(name='a')
    tag['href'] = "https://github.com/qiskit/source.py"
    tag['title'] = "view source code"

    result = github_handler.is_github_source_link(tag)

    assert result is True

def test_is_github_source_link_false(github_handler):
    """Should return False for non-GitHub links."""
    tag = Tag(name='a')
    tag['href'] = "https://google.com"
    tag['title'] = "search"

    result = github_handler.is_github_source_link(tag)

    assert result is False

def test_convert_blob_to_raw(github_handler):
    """Should convert blob URLs to raw content URLs correctly."""
    url = "https://github.com/user/repo/blob/main/file.py#L10"
    expected = "https://raw.githubusercontent.com/user/repo/main/file.py"

    result = github_handler.convert_blob_to_raw(url)

    assert result == expected

@patch('requests.get')
def test_download_file_success(mock_get, github_handler):
    """Should download file and return filename on success."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"print('hello')"
    mock_get.return_value = mock_response
    
    url = "https://github.com/test/file.py"

    filename = github_handler.download_file(url)

    assert filename == "file.py"
    assert os.path.exists(os.path.join(github_handler.download_dir, "file.py"))
    mock_get.assert_called_once()

def test_download_file_already_exists(github_handler):
    """Should skip download if file exists on disk."""
    filename = "existing.py"
    file_path = os.path.join(github_handler.download_dir, filename)
    with open(file_path, 'w') as f:
        f.write("existing content")
        
    url = f"https://github.com/test/{filename}"

    with patch('requests.get') as mock_get:
        result = github_handler.download_file(url)
        
        assert result == filename
        mock_get.assert_not_called()


def test_clean_text(content_parser):
    """Should remove extra whitespace and newlines."""
    raw_text = "  This   is \n a   test.  "
    
    cleaned = content_parser.clean_text(raw_text)
    
    assert cleaned == "This is a test."

def test_process_node_python_code(content_parser):
    """Should format Python code blocks correctly."""
    html = '<div data-rehype-pretty-code-fragment>print("test")</div>'
    soup = BeautifulSoup(html, 'html.parser')
    
    result = content_parser.process_node(soup.div)
    
    assert "```python" in result
    assert 'print("test")' in result
    assert content_parser.has_code is True

def test_process_node_latex(content_parser):
    """Should format LaTeX blocks with custom markers."""
    html = '<span class="katex-display">E=mc^2</span>'
    soup = BeautifulSoup(html, 'html.parser')
    
    result = content_parser.process_node(soup.span)
    
    assert "[LATEX_START]" in result
    assert "E=mc^2" in result
    assert content_parser.has_latex is True

def test_process_node_exclude_hidden_div(content_parser):
    """Should return empty string for excluded div classes."""

    html = '<div class="lg:hidden mt-48">Hidden Content</div>'
    soup = BeautifulSoup(html, 'html.parser')
    
    result = content_parser.process_node(soup.div)
    
    assert result == ""

def test_extract_title_priority_prose(content_parser):
    """Should prefer H1 inside 'prose' div."""

    html = """
    <html>
        <h1>Generic Title</h1>
        <div class="prose">
            <h1>Specific Title</h1>
        </div>
    </html>
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    title = content_parser.extract_title(soup)
    
    assert title == "Specific Title"

def test_extract_title_fallback(content_parser):
    """Should fall back to generic H1 if prose H1 is missing."""

    html = "<html><h1>Generic Title</h1></html>"
    soup = BeautifulSoup(html, 'html.parser')
    
    title = content_parser.extract_title(soup)

    assert title == "Generic Title"


@pytest.fixture
def scraper(tmp_path):
    """Returns a QiskitScraper with mocked config."""
    # Patch Config paths to use tmp_path
    with patch.object(Config, 'OUTPUT_DIR', str(tmp_path / "processed")), \
         patch.object(Config, 'PY_FILES_DIR', str(tmp_path / "raw_py")):
        yield QiskitScraper()

def test_save_single_record(scraper):
    """Should save record to JSON file with sanitized name."""

    record = {
        "title": "My Test Title / With Slash",
        "url": "[http://example.com](http://example.com)",
        "content": "test content"
    }
    
    scraper.save_single_record(record)
    
    expected_filename = "My_Test_Title__With_Slash.json"
    expected_path = os.path.join(Config.OUTPUT_DIR, expected_filename)
    
    assert os.path.exists(expected_path)
    
    with open(expected_path, 'r') as f:
        saved_data = json.load(f)
        assert saved_data['title'] == record['title']