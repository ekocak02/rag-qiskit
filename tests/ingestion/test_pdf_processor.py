import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.ingestion.pdf_processor import PDFProcessor


@pytest.fixture
def mock_filesystem(tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()
    return {"root": tmp_path, "raw": raw_dir, "output": output_dir}


@pytest.fixture
def mock_marker_library():
    with patch("src.ingestion.pdf_processor.create_model_dict"), patch(
        "src.ingestion.pdf_processor.PdfConverter"
    ) as mock_converter_cls, patch(
        "src.ingestion.pdf_processor.text_from_rendered"
    ) as mock_text_extractor:

        mock_instance = MagicMock()
        mock_converter_cls.return_value = mock_instance

        mock_text_extractor.return_value = ("Content", {}, {})

        yield {
            "converter_cls": mock_converter_cls,
            "converter_instance": mock_instance,
            "text_extractor": mock_text_extractor,
        }


@pytest.fixture
def clean_env():
    old_device = os.environ.get("TORCH_DEVICE")
    if "TORCH_DEVICE" in os.environ:
        del os.environ["TORCH_DEVICE"]
    yield
    if old_device:
        os.environ["TORCH_DEVICE"] = old_device


@pytest.fixture
def processor(mock_filesystem, clean_env):
    return PDFProcessor(
        raw_dir=str(mock_filesystem["raw"]), output_dir=str(mock_filesystem["output"])
    )


def test_clean_content_removes_noise(processor):
    """Verifies that regex patterns correctly strip unwanted artifacts."""

    dirty_text = (
        "# Header\n"
        "Here is a chart: ![](_page_1_Picture_1.jpeg)\n"
        "As seen in reference [1](#page-5-0).\n"
        "End of page (#page-0-1)."
    )

    expected_clean_text = (
        "# Header\n" "Here is a chart: \n" "As seen in reference 1.\n" "End of page ."
    )

    cleaned = processor._clean_content(dirty_text)

    # Normalize whitespace for comparison
    assert cleaned.replace(" ", "") == expected_clean_text.replace(" ", "")
    assert "(_page_1_Picture_1.jpeg)" not in cleaned
    assert "(#page-5-0)" not in cleaned


def test_extract_clean_metadata_finds_title(processor):
    """Verifies that the title is extracted from table_of_contents."""

    raw_meta = {
        "page_stats": {"huge": "data"},
        "polygon": [[10, 10], [20, 20]],
        "languages": ["en"],
        "table_of_contents": [
            {
                "title": "Supervised Learning with Quantum",
                "page_id": 0,
                "polygon": [...],
            },
            {"title": "Introduction", "page_id": 1},
        ],
    }

    clean_meta = processor._extract_clean_metadata(raw_meta, "default_filename")

    assert clean_meta["title"] == "Supervised Learning with Quantum"
    assert clean_meta["languages"] == ["en"]
    assert "page_stats" not in clean_meta
    assert "polygon" not in clean_meta


def test_extract_clean_metadata_fallbacks_to_filename(processor):
    """Verifies fallback to filename if TOC is missing or empty."""

    raw_meta = {"page_stats": {}}  # No TOC
    filename_stem = "research_paper_v1"

    clean_meta = processor._extract_clean_metadata(raw_meta, filename_stem)

    assert clean_meta["title"] == "research_paper_v1"


def test_process_file_integrates_cleaning(
    processor, mock_filesystem, mock_marker_library
):
    """
    Ensures process_file calls the cleaning methods and saves the result.
    """
    filename = "doc.pdf"
    (mock_filesystem["raw"] / filename).touch()

    mock_marker_library["text_extractor"].return_value = (
        "Clean me ![](_page_0_Pic.jpg)",
        {},
        {},
    )
    mock_marker_library["converter_instance"].return_value.metadata = {
        "table_of_contents": [{"title": "Real Title"}],
        "junk": "data",
    }

    output_path_str = processor.process_file(str(mock_filesystem["raw"] / filename))
    output_path = Path(output_path_str)

    content = output_path.read_text(encoding="utf-8")

    assert "title: Real Title" in content
    assert "junk: data" not in content

    assert "Clean me" in content
    assert "_page_0_Pic.jpg" not in content
