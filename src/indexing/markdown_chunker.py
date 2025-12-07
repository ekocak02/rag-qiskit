import json
import re
import uuid
from typing import List, Dict, Any
from pathlib import Path
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from src.indexing.utils import BaseProcessor, ProcessedChunk


class MarkdownProcessor(BaseProcessor):
    def __init__(self, token_limit: int = 2000):
        super().__init__(token_limit=token_limit)

        self.header_splitter = MarkdownHeaderTextSplitter(
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
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.token_helper.safe_limit,
            chunk_overlap=int(self.token_helper.safe_limit * 0.1),
            length_function=self.token_helper.count_tokens,
        )

    def process_file(self, file_path: str) -> List[ProcessedChunk]:
        path = Path(file_path)
        if not path.exists():
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except json.JSONDecodeError:
            return []

        items = raw_data if isinstance(raw_data, list) else [raw_data]
        all_chunks = []
        for item in items:
            all_chunks.extend(self._process_single_item(item))
        return all_chunks

    def _process_single_item(self, item: Dict[str, Any]) -> List[ProcessedChunk]:
        content = item.get("content", "")
        if not content:
            return []

        # 1. Masking (Code and LaTeX)
        content = self._convert_html_headers(content)
        patterns = [
            (r"\[LATEX_START\].*?\[LATEX_END\]", "LATEX"),
            (r"```.*?```\s*Output:\s*```.*?```", "WEB_CODE"),
            (r"```.*?```", "STD_CODE"),
        ]
        masked_content = self.mask_sensitive_blocks(content, patterns)

        # 2. Structural Split (on Masked Text)
        header_docs = self.header_splitter.split_text(masked_content)

        final_chunks = []

        # 3. Smart Processing
        for doc in header_docs:
            # Check if this section needs splitting even when masked
            if (
                self.token_helper.count_tokens(doc.page_content)
                > self.token_helper.safe_limit
            ):
                sub_docs = self.recursive_splitter.split_documents([doc])
                # Generate Group ID for this large section
                section_group_id = f"md_section_{uuid.uuid4().hex[:8]}"
            else:
                sub_docs = [doc]
                section_group_id = None

            # Second pass: Smart Unmasking
            for sub_doc in sub_docs:
                base_meta = self._build_metadata(sub_doc.metadata, item, "")

                if section_group_id:
                    base_meta["split_group_id"] = section_group_id

                smart_chunks = self.smart_unmask_and_split(
                    sub_doc.page_content, base_meta
                )

                for chunk in smart_chunks:
                    chunk.metadata["has_code"] = "```" in chunk.page_content
                    chunk.metadata["has_latex"] = (
                        "$" in chunk.page_content or "LATEX" in chunk.page_content
                    )
                    final_chunks.append(chunk)

        return final_chunks

    def _build_metadata(
        self, header_meta: Dict, original_item: Dict, content: str
    ) -> Dict:
        headers = [header_meta.get(f"h{i}") for i in range(1, 4)]
        base_meta = {
            "source": original_item.get("url", "unknown"),
            "topic": original_item.get("topic") or original_item.get("title", ""),
            "content_type": "text_mixed",
            "context_path": " > ".join([h for h in headers if h]),
            **header_meta,
        }
        api_files = original_item.get("metadata", {}).get("downloaded_py_files", [])
        if api_files:
            base_meta["related_source_file"] = ", ".join(api_files)
        return base_meta

    def _convert_html_headers(self, text: str) -> str:
        def replace(m):
            return "#" * int(m.group(1)) + " " + m.group(2)

        return re.sub(r"<h([1-6])>(.*?)</h\1>", replace, text, flags=re.IGNORECASE)
