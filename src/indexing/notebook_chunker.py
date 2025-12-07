import json
import uuid
from typing import List, Dict
from langchain_text_splitters import MarkdownHeaderTextSplitter
from src.indexing.utils import BaseProcessor, ProcessedChunk


class NotebookProcessor(BaseProcessor):
    def __init__(self, token_limit: int = 2000):
        super().__init__(token_limit=token_limit)
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
                ("#####", "h5"),
                ("######", "h6"),
            ],
            strip_headers=False,
        )

    def process_file(self, file_content: str) -> List[ProcessedChunk]:
        try:
            data = json.loads(file_content)
        except json.JSONDecodeError:
            return []

        file_meta = data.get("metadata", {})
        blocks = data.get("content", [])
        chunks, buffer_text, buffer_tokens, start_idx = [], [], 0, 0

        for i, block in enumerate(blocks):
            block_text = self._format_block(block)
            if not block_text.strip():
                continue

            # Count tokens of the RAW block first (approximate check)
            tokens = self.token_helper.count_tokens(block_text)

            # 1. Large Cell Logic (> Limit)
            if tokens > self.token_helper.safe_limit:
                # Flush buffer first
                if buffer_text:
                    chunks.extend(
                        self._flush_buffer(buffer_text, file_meta, start_idx, i - 1)
                    )
                    buffer_text, buffer_tokens = [], 0

                # Process this large cell individually
                chunks.extend(
                    self._process_large_block(
                        block_text, file_meta, i, block.get("type", "text")
                    )
                )
                start_idx = i + 1
                continue

            # 2. Buffer Logic (Accumulate until limit)
            if buffer_tokens + tokens > self.token_helper.safe_limit:
                chunks.extend(
                    self._flush_buffer(buffer_text, file_meta, start_idx, i - 1)
                )
                buffer_text, buffer_tokens, start_idx = [block_text], tokens, i
            else:
                buffer_text.append(block_text)
                buffer_tokens += tokens

        if buffer_text:
            chunks.extend(
                self._flush_buffer(buffer_text, file_meta, start_idx, len(blocks) - 1)
            )
        return chunks

    def _flush_buffer(
        self, text_list: List[str], meta: Dict, start: int, end: int
    ) -> List[ProcessedChunk]:
        full_text = "\n".join(text_list).strip()

        # Masking buffer content to protect small code snippets inside markdown cells
        patterns = [(r"\[LATEX_START\].*?\[LATEX_END\]", "LATEX")]
        masked_text = self.mask_sensitive_blocks(full_text, patterns)

        base_meta = self._build_metadata(meta, "")

        # Generate a unique ID for this buffer group
        split_group_id = f"buffer_{uuid.uuid4().hex[:8]}"

        base_meta.update(
            {
                "cell_range": f"{start}-{end}",
                "strategy": "buffer",
                "split_group_id": split_group_id,
            }
        )

        return self.smart_unmask_and_split(masked_text, base_meta)

    def _process_large_block(
        self, text: str, base_meta: Dict, idx: int, b_type: str
    ) -> List[ProcessedChunk]:
        """
        Handles cells that are individually larger than the token limit.
        """
        # Unique ID for this large block to link all its split parts
        split_group_id = f"large_block_{uuid.uuid4().hex[:8]}"

        if b_type == "text":
            # Mask Latex
            masked = self.mask_sensitive_blocks(
                text, [(r"\[LATEX_START\].*?\[LATEX_END\]", "LATEX")]
            )
            # Split by Headers first
            docs = self.md_splitter.split_text(masked)

            final_chunks = []
            for doc in docs:
                meta = self._build_metadata(base_meta, "")
                meta.update(
                    {
                        "cell_index": idx,
                        "strategy": "markdown_header",
                        "split_group_id": split_group_id,
                        **doc.metadata,
                    }
                )
                # Use Smart Unmasking on the header sections
                final_chunks.extend(self.smart_unmask_and_split(doc.page_content, meta))
            return final_chunks

        else:
            splits = self.code_splitter.split_text(text)
            chunks = []
            for split in splits:
                meta = self._build_metadata(base_meta, split)
                meta.update(
                    {
                        "cell_index": idx,
                        "strategy": "large_code_split",
                        "split_group_id": split_group_id,
                    }
                )
                meta["token_count"] = self.token_helper.count_tokens(split)
                chunks.append(ProcessedChunk(page_content=split, metadata=meta))
            return chunks

    def _format_block(self, block: Dict) -> str:
        content = block.get("content", "")
        if block.get("type") == "code":
            out = block.get("output", "")
            return (
                f"{content}\nOutput:\n```text\n{out}\n```" if out.strip() else content
            )
        return content

    def _build_metadata(self, file_meta: Dict, content: str) -> Dict:
        return {
            "source": file_meta.get("filename", "unknown"),
            "qiskit_version": file_meta.get("qiskit_version"),
            "has_code": "```" in content,
            "has_latex": "$" in content or "LATEX" in content,
        }
