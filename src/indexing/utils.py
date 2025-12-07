import tiktoken
import uuid
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from langchain_text_splitters import RecursiveCharacterTextSplitter

@dataclass
class ProcessedChunk:
    """Standard output object for processed chunks."""
    page_content: str
    metadata: Dict[str, Any]
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))

class TokenHelper:
    """Handles token counting with safety margins."""
    
    def __init__(self, model_name: str = "cl100k_base", safety_margin: float = 0.10, target_limit: int = 2000):
        try:
            self.tokenizer = tiktoken.get_encoding(model_name)
        except ValueError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            
        self.safety_margin = safety_margin
        self.raw_limit = target_limit
        self.safe_limit = int(target_limit * (1.0 - safety_margin))

    def count_tokens(self, text: Optional[str]) -> int:
        if not text:
            return 0
        return len(self.tokenizer.encode(text, disallowed_special=()))

class BaseProcessor:
    """
    Base class providing masking, cleaning, and SMART LIMIT enforcement.
    """
    def __init__(self, token_limit: int = 2000):
        self.token_helper = TokenHelper(target_limit=token_limit)
        self.mask_map: Dict[str, str] = {}
        
        # Splitter for general text overflow
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.token_helper.safe_limit,
            chunk_overlap=100,
            length_function=self.token_helper.count_tokens
        )
        
        # Splitter specifically for CODE/LATEX overflow (aggressive separators)
        self.code_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.token_helper.safe_limit,
            chunk_overlap=50,
            length_function=self.token_helper.count_tokens,
            separators=["\n\n", "\n", " ", ""]
        )

    def mask_sensitive_blocks(self, text: str, patterns: List[Tuple[str, str]]) -> str:
        """Replaces sensitive blocks with unique IDs to protect them during splitting."""
        self.mask_map.clear()
        
        for pattern, tag in patterns:
            text = self._apply_mask(text, pattern, tag)
        return text

    def _apply_mask(self, text: str, pattern: str, tag: str) -> str:
        """Internal helper to apply a single regex mask."""
        def replacer(match):
            # Generate a unique ID for the block
            uid = f"__PROTECTED_{tag}_{uuid.uuid4().hex}__"
            self.mask_map[uid] = match.group(0)
            return uid
            
        return re.sub(pattern, replacer, text, flags=re.DOTALL | re.MULTILINE)

    def clean_metadata(self, meta: Dict) -> Dict:
        """Removes None or empty string values from metadata."""
        return {k: v for k, v in meta.items() if v is not None and v != ""}

    def smart_unmask_and_split(self, masked_text: str, meta: Dict) -> List[ProcessedChunk]:
        """
        Unmasks content and handles overflows.
        Splits are performed if the unmasked content exceeds safe limits.
        """
        mask_pattern = r"__PROTECTED_[A-Z_]+_[a-f0-9]+__"
        parts = re.split(f"({mask_pattern})", masked_text)
        
        chunks: List[ProcessedChunk] = []
        buffer = ""
        
        for part in parts:
            if not part: continue
            
            if part in self.mask_map:
                chunks, buffer = self._handle_protected_block(part, buffer, meta, chunks)
            else:
                chunks, buffer = self._handle_normal_text(part, buffer, meta, chunks)
        
        # Process remaining buffer
        if buffer:
            chunks.extend(self._finalize_text_chunk(buffer, meta))
            
        return chunks

    def _handle_protected_block(self, part: str, buffer: str, meta: Dict, chunks: List[ProcessedChunk]) -> Tuple[List[ProcessedChunk], str]:
        """Handles logic when a protected block (code/latex) is encountered."""
        original_content = self.mask_map[part]
        content_tokens = self.token_helper.count_tokens(original_content)

        # Case 1: The protected block itself is too large (Huge Block)
        if content_tokens > self.token_helper.safe_limit:
            # Flush current buffer first
            if buffer:
                chunks.extend(self._finalize_text_chunk(buffer, meta))
            
            # Split and add the huge block
            chunks.extend(self._split_huge_block(original_content, meta))
            return chunks, ""

        # Case 2: Buffer + Block fits in limit -> Append
        if self.token_helper.count_tokens(buffer + original_content) <= self.token_helper.safe_limit:
            return chunks, buffer + original_content
            
        # Case 3: Overflow -> Flush buffer, start new buffer with block
        chunks.extend(self._finalize_text_chunk(buffer, meta))
        return chunks, original_content

    def _handle_normal_text(self, part: str, buffer: str, meta: Dict, chunks: List[ProcessedChunk]) -> Tuple[List[ProcessedChunk], str]:
        """Handles logic for normal text parts."""
        if self.token_helper.count_tokens(buffer + part) > self.token_helper.safe_limit:
            chunks.extend(self._finalize_text_chunk(buffer, meta))
            return chunks, part
        return chunks, buffer + part

    def _split_huge_block(self, content: str, meta: Dict) -> List[ProcessedChunk]:
        """Splits a single large code/latex block into smaller chunks."""
        sub_splits = self.code_splitter.split_text(content)
        huge_block_id = f"huge_block_{uuid.uuid4().hex[:8]}"
        chunks = []
        
        for i, sub_split in enumerate(sub_splits):
            sub_meta = meta.copy()
            sub_meta.update({
                "split_method": "huge_block_split", 
                "original_block_type": "code_or_latex",
                "chunk_index_sub": i,
                "split_group_id": huge_block_id
            })
            chunks.append(ProcessedChunk(
                page_content=sub_split, 
                metadata=self.clean_metadata(sub_meta)
            ))
        return chunks

    def _finalize_text_chunk(self, text: str, meta: Dict) -> List[ProcessedChunk]:
        """Final check: if text fits, return 1 chunk. If not, force split."""
        if not text.strip(): return []
        
        token_count = self.token_helper.count_tokens(text)
        
        # Fits perfectly
        if token_count <= self.token_helper.safe_limit:
            meta["token_count"] = token_count
            return [ProcessedChunk(page_content=text, metadata=self.clean_metadata(meta))]
        
        # Needs forced splitting
        return self._force_split_text(text, meta)

    def _force_split_text(self, text: str, meta: Dict) -> List[ProcessedChunk]:
        """Splits text that exceeds limits using the text_splitter."""
        splits = self.text_splitter.split_text(text)
        chunks = []
        forced_split_id = f"forced_split_{uuid.uuid4().hex[:8]}"
        
        for i, split in enumerate(splits):
            chunk_meta = meta.copy()
            chunk_meta.update({
                "chunk_index_sub": i,
                "token_count": self.token_helper.count_tokens(split),
                "split_group_id": forced_split_id
            })
            chunks.append(ProcessedChunk(
                page_content=split, 
                metadata=self.clean_metadata(chunk_meta)
            ))
        return chunks