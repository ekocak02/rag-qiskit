import pytest
from src.ingestion.web_scraper import WebScraper

MOCK_HTML_WITH_OUTPUT = """
<html>
<body>
    <div class="prose">
        <h1 id="advanced-coding">Advanced Coding</h1>
        <p>Here is an example with output.</p>

        <!-- Input Code Block -->
        <div data-rehype-pretty-code-fragment>
            print("Hello World")
            print("Success")
        </div>

        <!-- Spacer (should be ignored) -->
        <div class="mt-32"></div>

        <!-- Output Label -->
        <p class="text-text-helper text-label-01">Output:</p>

        <!-- Output Snippets (Multi-line output simulation) -->
        <div class="snippet relative bg-[var(--shiki-color-background)] overflow-hidden">
            Hello World
        </div>
        <div class="snippet relative bg-[var(--shiki-color-background)] overflow-hidden">
            Success
        </div>

        <p>End of section.</p>
    </div>
</body>
</html>
"""


@pytest.fixture
def scraper(tmp_path):

    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    return WebScraper(raw_dir=str(raw), processed_dir=str(processed))


def test_code_output_extraction(scraper):
    """
    Test that code blocks and their corresponding output snippets
    are merged into a single text block.
    """
    data = scraper.parse_content(MOCK_HTML_WITH_OUTPUT, "http://test.com")
    content = data["content"]

    assert 'print("Hello World")' in content
    assert "Output:" in content
    assert "Hello World" in content
    assert "Success" in content

    code_index = content.find('print("Success")')
    output_index = content.find("Output:")

    assert code_index != -1
    assert output_index != -1
    assert output_index > code_index, "Output should follow the code block"


def test_metadata_extraction(scraper):
    """Test has_code and has_latex flags."""
    html = """<div class="prose"><h1>T</h1><div data-rehype-pretty-code-fragment>C</div></div>"""
    data = scraper.parse_content(html, "http://test.com")

    meta = data["metadata"]
    assert meta["has_code"] is True


def test_single_file_saving(scraper):
    """Test if data is saved to a separate JSON file named after the topic."""
    html = """<div class="prose"><h1 id="test-topic">Test Topic</h1></div>"""
    data = scraper.parse_content(html, "http://test.com")
    scraper.save_data(data)

    expected_file = scraper.processed_dir / "test-topic.json"
    assert expected_file.exists()
