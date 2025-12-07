import re
import uuid
from typing import List, Dict, Any
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from src.indexing.utils import BaseProcessor, ProcessedChunk


class PdfProcessor(BaseProcessor):
    """
    Processor specifically designed for cleaning and chunking text extracted from PDFs
    (converted to Markdown), handling specific artifacts like HTML spans and frontmatter.
    """

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
            chunk_overlap=100,
            length_function=self.token_helper.count_tokens,
        )

    def process_file(self, content: str, filename: str) -> List[ProcessedChunk]:
        """
        Main entry point: Cleans, masks, splits by header, and then enforces token limits.
        """
        # 1. Clean PDF-to-Markdown Artifacts
        clean_content = self._clean_artifacts(content)

        # 2. Mask Sensitive Blocks (LaTeX)
        patterns = [
            (r"\$\$.*?\$\$", "LATEX_BLOCK"),
            (r"(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)", "LATEX_INLINE"),
        ]
        masked_content = self.mask_sensitive_blocks(clean_content, patterns)

        # 3. Structural Split (Headers)
        header_docs = self.header_splitter.split_text(masked_content)
        final_chunks = []

        # 4. Processing & Smart Unmasking
        for doc in header_docs:
            final_chunks.extend(self._process_header_doc(doc, filename))

        return final_chunks

    def _process_header_doc(self, doc: Any, filename: str) -> List[ProcessedChunk]:
        """Processes a single header-split document, handling recursive splitting if needed."""
        # Determine if recursive splitting is needed
        if (
            self.token_helper.count_tokens(doc.page_content)
            > self.token_helper.safe_limit
        ):
            sub_docs = self.recursive_splitter.split_documents([doc])
            section_group_id = f"pdf_section_{uuid.uuid4().hex[:8]}"
        else:
            sub_docs = [doc]
            section_group_id = None

        chunks = []
        for sub in sub_docs:
            meta = self._build_metadata(sub.metadata, filename, sub.page_content)

            if section_group_id:
                meta["split_group_id"] = section_group_id

            unmasked_chunks = self.smart_unmask_and_split(sub.page_content, meta)

            for chunk in unmasked_chunks:
                chunk.metadata["has_latex"] = (
                    "$" in chunk.page_content or "LATEX" in chunk.page_content
                )
                chunks.append(chunk)

        return chunks

    def _build_metadata(self, doc_meta: Dict, filename: str, content: str = "") -> Dict:
        """Constructs the standard metadata dictionary."""
        headers = [doc_meta.get(f"h{i}") for i in range(1, 5)]

        meta = {
            "source": filename,
            "context_path": " > ".join([h for h in headers if h]),
            **doc_meta,
        }

        # Heuristic: If content is provided, check for latex indicators
        if content:
            meta["has_latex"] = "$" in content or "LATEX" in content

        return meta

    def _clean_artifacts(self, text: str) -> str:
        """Removes common artifacts from PDF-to-Markdown conversion."""
        # Remove YAML frontmatter
        text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)
        # Remove HTML spans
        text = re.sub(r"<span[^>]*>(.*?)</span>", r"\1", text)
        # Convert superscripts to brackets
        text = re.sub(r"<sup>(.*?)</sup>", r"[\1]", text)
        return text
