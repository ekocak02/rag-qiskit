import json
import logging
from pathlib import Path
from typing import List
from tqdm import tqdm
from dataclasses import asdict

# Import all processors
from src.indexing.python_chunker import PythonProcessor
from src.indexing.markdown_chunker import MarkdownProcessor
from src.indexing.notebook_chunker import NotebookProcessor
from src.indexing.pdf_chunker import PdfProcessor
from src.indexing.utils import ProcessedChunk



DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "merged"
OUTPUT_FILE = OUTPUT_DIR / "unified.jsonl"
TOKEN_LIMIT = 2000

INPUT_DIRS = {
    "python": DATA_DIR / "raw" / "py_files",
    "web": DATA_DIR / "processed" / "web_data",
    "api": DATA_DIR / "processed" / "qiskit_api",
    "ipynb": DATA_DIR / "processed" / "ipynb_files",
    "pdf": DATA_DIR / "processed" / "pdf_files"
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def append_to_jsonl(chunks: List[ProcessedChunk], filepath: Path):
    """Appends chunks to JSONL file immediately."""
    if not chunks: return
    with open(filepath, 'a', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

def process_directory(file_pattern: str, input_dirs: List[Path], processor, desc: str, needs_read: bool = True):
    """Generic processing function for all file types."""
    files = []
    for d in input_dirs:
        if d.exists():
            files.extend(list(d.rglob(file_pattern)))
    
    logger.info(f"Found {len(files)} files for {desc}.")
    
    for file_path in tqdm(files, desc=desc):
        try:
            if needs_read:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Processor Specific Logic
                if isinstance(processor, MarkdownProcessor):
                     chunks = processor.process_file(str(file_path))
                     
                     for chunk in chunks:
                         src = chunk.metadata.get("source", "")
                         
                         if "docs/api/qiskit/" in src:
                             chunk.metadata["source"] = src.split("api/qiskit/")[-1]
                         
                         elif src == "unknown":
                             chunk.metadata["source"] = file_path.name

                elif isinstance(processor, (PythonProcessor, PdfProcessor)):
                     # Use file_path.name -> "my_file.py" instead of full path
                     chunks = processor.process_file(content, filename=file_path.name)
                
                elif isinstance(processor, NotebookProcessor):
                     chunks = processor.process_file(content)
            else:
                 pass

            append_to_jsonl(chunks, OUTPUT_FILE)
            
        except Exception as e:
            logger.error(f"Error in {file_path.name}: {str(e)}")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_FILE.exists():
        logger.info(f"Overwriting existing file: {OUTPUT_FILE}")
        open(OUTPUT_FILE, 'w').close()

    processors = {
        "py": PythonProcessor(token_limit=TOKEN_LIMIT),
        "md": MarkdownProcessor(token_limit=TOKEN_LIMIT),
        "nb": NotebookProcessor(token_limit=TOKEN_LIMIT),
        "pdf": PdfProcessor(token_limit=TOKEN_LIMIT)
    }

    logger.info("Starting Unified Processing Pipeline...")

    # 1. Markdown & Web Data (JSON Input)
    process_directory("*.json", [INPUT_DIRS["web"], INPUT_DIRS["api"]], processors["md"], "Web/API Data")

    # 2. Python Code (Raw .py Input)
    process_directory("*.py", [INPUT_DIRS["python"]], processors["py"], "Python Code")

    # 3. Jupyter Notebooks (JSON Input)
    process_directory("*.json", [INPUT_DIRS["ipynb"]], processors["nb"], "Notebooks")

    # 4. PDF Documents (Markdown .md Input)
    process_directory("*.md", [INPUT_DIRS["pdf"]], processors["pdf"], "PDF Docs")

    logger.info(f"Processing Complete! Data saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()