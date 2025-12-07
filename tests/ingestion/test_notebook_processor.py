import pytest
import json
from src.ingestion.notebook_processor import NotebookProcessor


@pytest.fixture
def processor():
    return NotebookProcessor()


@pytest.fixture
def sample_notebook_json():
    """Creates a mock notebook structure."""
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "source": [
                    'This is a <DefinitionTooltip definition="Explanation">Concept</DefinitionTooltip>.'
                ],
            },
            {
                "cell_type": "code",
                "source": ["import qiskit\n", "from numpy import array"],
                "outputs": [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": ["Calculation done"],
                    }
                ],
            },
            {"cell_type": "markdown", "source": ["© IBM Corp., 2017-2025"]},
        ]
    }


def test_clean_markdown_tooltip(processor):
    raw = 'This is a <DefinitionTooltip definition="Hidden info">Target Word</DefinitionTooltip>.'
    expected = "This is a Target Word."
    assert processor._clean_markdown(raw) == expected


def test_clean_markdown_admonition(processor):
    raw = '<Admonition type="note">This is important</Admonition>'
    clean = processor._clean_markdown(raw)
    assert "> **Note:** This is important" in clean


def test_clean_code_output_removes_images(processor):
    """Tests if HTML Image tags are removed from code output."""
    raw_output = 'Result: 5\n<Image src="/learning/images/test.avif" alt="Output" />'
    expected = "Result: 5"
    assert processor._clean_code_output(raw_output) == expected


def test_clean_code_output_only_image(processor):
    """Tests if output becomes empty string when only Image tag exists."""
    raw_output = '<Image src="/learning/images/test.avif" alt="Output" />'
    assert processor._clean_code_output(raw_output) == ""


def test_process_code_cell_structure(processor):
    """Tests that code cells have correct JSON structure with separate output."""
    cell = {
        "cell_type": "code",
        "source": ["print('Hello')"],
        "outputs": [{"name": "stdout", "text": ["Hello World"]}],
    }
    result = processor._process_code_cell(cell, "print('Hello')")

    assert result["type"] == "code"
    assert "```python\nprint('Hello')\n```" in result["content"]
    assert result["output"] == "Hello World"
    # Content should NOT contain the output
    assert "Output:" not in result["content"]


def test_process_code_cell_empty_output(processor):
    """Tests that output is an empty string if there is no output or it was cleaned."""
    cell = {
        "cell_type": "code",
        "source": ["x = 1"],
        "outputs": [{"data": {"text/plain": ['<Image src="plot.png" />']}}],
    }
    result = processor._process_code_cell(cell, "x = 1")

    assert result["type"] == "code"
    assert result["output"] == ""


def test_metadata_extraction_basics(processor, sample_notebook_json):
    """Tests basic metadata including has_code."""
    result = processor._extract_content(sample_notebook_json, "test.ipynb")

    assert result["metadata"]["has_code"] is True
    assert result["metadata"]["has_latex"] is False
    assert result["metadata"]["qiskit_version"] == "2.0.0+"


def test_metadata_latex_detection(processor):
    """Tests if LaTeX delimiters are detected."""
    nb_data = {"cells": [{"cell_type": "markdown", "source": ["Equation: $E=mc^2$"]}]}
    result = processor._extract_content(nb_data, "math.ipynb")
    assert result["metadata"]["has_latex"] is True


def test_qiskit_version_extraction_explicit(processor):
    """Tests extracting explicit qiskit version from pip commands."""
    nb_data = {
        "cells": [
            {"cell_type": "code", "source": ["import qiskit"]},
            {"cell_type": "code", "source": ["!pip install qiskit==0.45.1"]},
        ]
    }
    result = processor._extract_content(nb_data, "version.ipynb")
    assert result["metadata"]["qiskit_version"] == "0.45.1"


def test_qiskit_version_missing_library(processor):
    """Tests that qiskit_version is None if library is not imported."""
    nb_data = {"cells": [{"cell_type": "code", "source": ["import numpy"]}]}
    result = processor._extract_content(nb_data, "numpy.ipynb")
    assert result["metadata"]["qiskit_version"] is None
    assert result["metadata"]["has_code"] is True


def test_qiskit_version_default(processor):
    """Tests default 2.0.0+ when qiskit is imported but no version specified."""
    nb_data = {
        "cells": [
            {
                "cell_type": "code",
                "source": ["import qiskit\nqc = qiskit.QuantumCircuit()"],
            }
        ]
    }
    result = processor._extract_content(nb_data, "default.ipynb")
    assert result["metadata"]["qiskit_version"] == "2.0.0+"


def test_file_io(processor, tmp_path):
    """Integration test using temporary directories."""
    # Setup
    input_dir = tmp_path / "data/raw/ipynb_files"
    output_dir = tmp_path / "data/processed/ipynb_files"
    input_dir.mkdir(parents=True)

    # Create dummy file
    dummy_file = input_dir / "test_nb.ipynb"
    with open(dummy_file, "w") as f:
        json.dump({"cells": []}, f)

    # Run
    processor.process_directory(str(input_dir), str(output_dir))

    expected_output = output_dir / "test_nb_processed.json"
    assert expected_output.exists()
