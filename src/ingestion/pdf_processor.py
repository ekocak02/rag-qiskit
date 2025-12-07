import os
import torch
import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Handles the conversion of PDF documents to Markdown using the Marker library.
    Includes post-processing to clean noise and extract relevant metadata for LLM/RAG usage.
    """

    def __init__(self, raw_dir: Optional[str] = None, output_dir: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the processor with input/output configuration and device selection.
        """
        self.project_root = Path(__file__).resolve().parents[3]
        
        if raw_dir:
            self.raw_dir = Path(raw_dir)
        else:
            self.raw_dir = self.project_root / "data" / "raw" / "pdf_files"

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.project_root / "data" / "processed" / "pdf_files"

        if not self.raw_dir.exists():
            logger.warning(f"Raw directory does not exist: {self.raw_dir}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if device:
            os.environ["TORCH_DEVICE"] = device
            logger.info(f"Configured TORCH_DEVICE={device}")
        elif not os.environ.get("TORCH_DEVICE"):
            if torch.cuda.is_available():
                os.environ["TORCH_DEVICE"] = "cuda"
            elif torch.backends.mps.is_available():
                 os.environ["TORCH_DEVICE"] = "mps"
            else:
                 os.environ["TORCH_DEVICE"] = "cpu"
        
        self.converter = None 

    def run(self):
        """Sequentially processes all PDF files found in the raw_dir."""
        if not self.raw_dir.exists():
            logger.error(f"Raw directory {self.raw_dir} does not exist.")
            return

        pdf_files = list(self.raw_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.raw_dir}")
            return

        logger.info(f"Found {len(pdf_files)} PDF files to process in {self.raw_dir}")

        processed_count = 0
        for pdf_file in pdf_files:
            try:
                self.process_file(str(pdf_file))
                processed_count += 1
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
        
        logger.info(f"DONE: Processed {processed_count}/{len(pdf_files)} files.")

    def _load_converter(self):
        """Loads the Marker models and converter into memory if not already loaded."""
        if self.converter is None:
            logger.info("Loading Marker models... (this may take a moment)")
            try:
                artifact_dict = create_model_dict()
                self.converter = PdfConverter(artifact_dict=artifact_dict)
                logger.info("Models loaded successfully.")
            except Exception as e:
                logger.critical(f"Failed to load Marker models: {e}")
                raise

    def process_file(self, file_path: str) -> str:
        """
        Processes a single PDF file, cleans the output, and saves the markdown.
        """
        input_path = Path(file_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        self._load_converter()

        logger.info(f"Processing {input_path.name}...")
        
        try:
            rendered = self.converter(str(input_path))
            full_text, _, _ = text_from_rendered(rendered)
            
            # 1. Extract and Clean Metadata
            raw_metadata = rendered.metadata if hasattr(rendered, 'metadata') else {}
            clean_metadata = self._extract_clean_metadata(raw_metadata, input_path.stem)
            
            # 2. Clean Content Noise
            clean_text = self._clean_content(full_text)
            
            # 3. Save
            output_file = self._save_output(input_path.stem, clean_text, clean_metadata)
            logger.info(f"SUCCESS: Processed file saved to {output_file}")
            return str(output_file)

        except Exception as e:
            logger.error(f"Failed to process {input_path.name}. Reason: {e}")
            raise e

    def _extract_clean_metadata(self, raw_meta: Dict[str, Any], default_title: str) -> Dict[str, Any]:
        """
        Extracts only essential metadata (title, language) and discards noise (polygons, stats).
        Prioritizes extraction of the main title from the table of contents.
        """
        clean_meta = {}
        
        # Extract Title from Table of Contents (usually the first entry)
        toc = raw_meta.get('table_of_contents', [])
        title = default_title
        
        if toc and isinstance(toc, list) and len(toc) > 0:
            # Check the first item for a title
            first_entry = toc[0]
            if isinstance(first_entry, dict) and 'title' in first_entry:
                title = first_entry['title']
        
        clean_meta['title'] = title
        
        # Keep other useful but small fields if they exist
        if 'languages' in raw_meta:
            clean_meta['languages'] = raw_meta['languages']
        if 'page_count' in raw_meta:
            clean_meta['page_count'] = raw_meta['page_count']
            
        return clean_meta

    def _clean_content(self, text: str) -> str:
        """
        Removes RAG-irrelevant noise from the markdown text using Regex.
        
        Targets:
        - Internal page links: (#page-0-0)
        - Image placeholders: ![](_page_1_Picture_1.jpeg)
        """
        # 1. Remove image placeholders like ![](_page_1_Picture_1.jpeg)
        text = re.sub(r'!\[.*?\]\(_page_\d+_.*?\)', '', text)
        
        # 2. Remove internal page reference links like [1](#page-5-0) or just (#page-5-0)
        text = re.sub(r'\[(.*?)\]\(#page-\d+-\d+\)', r'\1', text)
        
        # Case B: Standalone page anchors if any remains: (#page-x-y)
        text = re.sub(r'\(#page-\d+-\d+\)', '', text)
        
        # 3. Clean up extra newlines created by removals
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()

    def _save_output(self, filename_stem: str, content: str, metadata: Dict[str, Any]) -> Path:
        """Saves the processed content with YAML frontmatter."""
        output_file = self.output_dir / f"{filename_stem}.md"
        
        # Prepare metadata header
        meta_header = "---\n"
        for key, value in metadata.items():
            # Clean newlines in values to prevent YAML breakage
            clean_val = str(value).replace("\n", " ")
            meta_header += f"{key}: {clean_val}\n"
        meta_header += "---\n\n"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(meta_header + content)
            
        return output_file

if __name__ == "__main__":
    processor = PDFProcessor()
    processor.run()