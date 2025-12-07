import pytest
from src.indexing.utils import BaseProcessor, TokenHelper


def test_token_helper_initialization():
    """Should initialize with correct limits based on safety margin."""
    helper = TokenHelper(target_limit=1000, safety_margin=0.10)
    assert helper.raw_limit == 1000
    assert helper.safe_limit == 900


def test_token_helper_counts_correctly():
    """Should accurately count tokens for a given string."""
    helper = TokenHelper()
    text = "Hello world"
    count = helper.count_tokens(text)
    assert count > 0
    assert isinstance(count, int)


@pytest.fixture
def processor():
    return BaseProcessor(token_limit=250)


def test_mask_sensitive_blocks(processor):
    """Should replace sensitive patterns with unique ID placeholders."""
    text = "Here is code: ```print('hello')```"
    patterns = [(r"```.*?```", "CODE")]

    masked_text = processor.mask_sensitive_blocks(text, patterns)

    assert "```print('hello')```" not in masked_text
    assert "__PROTECTED_CODE_" in masked_text
    assert len(processor.mask_map) == 1


def test_clean_metadata(processor):
    """Should remove None or empty string values from metadata."""
    meta = {"valid": "value", "none": None, "empty": ""}
    cleaned = processor.clean_metadata(meta)

    assert "valid" in cleaned
    assert "none" not in cleaned
    assert "empty" not in cleaned


def test_smart_unmask_restores_content(processor):
    """Should correctly restore masked content via smart_unmask_and_split."""
    original_snippet = "```print('hello')```"
    text = f"Here is code: {original_snippet}"
    patterns = [(r"```.*?```", "STD_CODE")]

    masked_text = processor.mask_sensitive_blocks(text, patterns)

    chunks = processor.smart_unmask_and_split(masked_text, {"id": 1})

    assert len(chunks) == 1
    assert original_snippet in chunks[0].page_content
    assert chunks[0].metadata["id"] == 1


def test_smart_unmask_splits_huge_block(processor):
    """Should split a protected block if it exceeds the limit."""
    # Create a block larger than token_limit (250)
    huge_code = "print('long line')\n" * 100
    text = f"Start ```{huge_code}``` End"
    patterns = [(r"```.*?```", "CODE")]

    masked_text = processor.mask_sensitive_blocks(text, patterns)
    chunks = processor.smart_unmask_and_split(masked_text, {"id": 1})

    assert len(chunks) > 1
    assert any(c.metadata.get("split_method") == "huge_block_split" for c in chunks)


def test_smart_unmask_handles_overflow_text(processor):
    """Should force split normal text if it exceeds safe limit."""
    long_text = "word " * 300  # Exceeds 250 tokens

    chunks = processor.smart_unmask_and_split(long_text, {"id": 1})

    assert len(chunks) > 1
    assert chunks[0].metadata["id"] == 1
