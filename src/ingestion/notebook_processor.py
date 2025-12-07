import json
import re
import logging
from pathlib import Path
from typing import Dict, Any, Set, Optional, Match

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NotebookProcessor:
    """
    Parses and cleans Jupyter Notebooks for RAG ingestion.
    Handles path resolution relative to project root.
    """

    def __init__(self):
        self._compile_patterns()
        # Resolve project root (assumes this script is in src/ingestion/)
        self.project_root = Path(__file__).resolve().parents[3]

    def _compile_patterns(self):
        """Compiles regex patterns for performance."""
        self.tooltip_pattern = re.compile(
            r'<DefinitionTooltip.*?>(.*?)</DefinitionTooltip>', 
            re.DOTALL
        )
        self.admonition_pattern = re.compile(
            r'<Admonition.*?>(.*?)</Admonition>', 
            re.DOTALL
        )
        self.details_pattern = re.compile(
            r'<details>\s*<summary>(.*?)</summary>(.*?)</details>', 
            re.DOTALL
        )
        self.import_pattern = re.compile(
            r'^\s*(?:import|from)\s+(\w+)', 
            re.MULTILINE
        )
        self.noise_patterns = [
            re.compile(r'cspell:ignore'),
            re.compile(r'© IBM Corp', re.IGNORECASE)
        ]
        
        # Detection pattern for metadata
        self.latex_detection_pattern = re.compile(r'\$|\\begin\{')
        
        # Substitution pattern to wrap LaTeX in markers (Markdown/MathJax style)
        # Matches $$...$$ or \begin{...}...\end{...} or $...$
        self.latex_wrap_pattern = re.compile(
            r'(\$\$[\s\S]*?\$\$|\\begin\{.*?\}[\s\S]*?\\end\{.*?\}|\$.*?\$)'
        )

        self.qiskit_version_pattern = re.compile(
            r'qiskit(?:\[.*?\])?==([0-9]+\.[0-9]+\.?[0-9]*)', 
            re.IGNORECASE
        )
        # Regex to identify HTML Image tags in outputs
        self.output_image_pattern = re.compile(
            r'<Image\s+src=.*?>', 
            re.IGNORECASE | re.DOTALL
        )

    def process_directory(self, input_dir: Optional[str] = None, output_dir: Optional[str] = None) -> None:
        """
        Processes all .ipynb files in input_dir and saves to output_dir.
        Uses default project structure if paths are not provided.
        """
        if input_dir:
            in_path = Path(input_dir)
        else:
            in_path = self.project_root / "data" / "raw" / "ipynb_files"

        if output_dir:
            out_path = Path(output_dir)
        else:
            out_path = self.project_root / "data" / "processed" / "ipynb_files"

        if not in_path.exists():
            logger.error(f"Input directory does not exist: {in_path}")
            return

        out_path.mkdir(parents=True, exist_ok=True)

        notebook_files = list(in_path.glob("*.ipynb"))
        logger.info(f"Found {len(notebook_files)} notebooks in {in_path}")

        for file_path in notebook_files:
            try:
                processed_data = self.process_file(file_path)
                
                output_file = out_path / f"{file_path.stem}_processed.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Processed: {file_path.name} -> {output_file.name}")
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")

    def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Reads and processes a single notebook file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            notebook_data = json.load(f)

        return self._extract_content(notebook_data, file_path.name)

    def _extract_content(self, notebook_data: Dict, filename: str) -> Dict[str, Any]:
        """Core logic to extract content and metadata from notebook JSON."""
        processed_cells = []
        libraries = set()
        
        has_code = False
        has_latex = False
        raw_code_text = ""

        for cell in notebook_data.get('cells', []):
            source_list = cell.get('source', [])
            if not source_list:
                continue
            
            source_text = "".join(source_list)

            if self._is_noise(source_text):
                continue

            if cell.get('cell_type') == 'markdown':
                if self.latex_detection_pattern.search(source_text):
                    has_latex = True
                
                clean_text = self._clean_markdown(source_text)
                if clean_text:
                    processed_cells.append({
                        "type": "text", 
                        "content": clean_text
                    })

            elif cell.get('cell_type') == 'code':
                has_code = True
                raw_code_text += "\n" + source_text
                
                code_data = self._process_code_cell(cell, source_text)
                processed_cells.append(code_data)
                libraries.update(self._extract_libraries(source_text))

        qiskit_version = self._determine_qiskit_version(libraries, raw_code_text)

        return {
            "metadata": {
                "filename": filename,
                "libraries": list(libraries),
                "total_blocks": len(processed_cells),
                "has_code": has_code,
                "has_latex": has_latex,
                "qiskit_version": qiskit_version
            },
            "content": processed_cells
        }

    def _clean_markdown(self, text: str) -> str:
        """Removes HTML noise and wraps LaTeX patterns with custom markers."""
        # Clean HTML artifacts
        text = self.tooltip_pattern.sub(r'\1', text)
        text = self.admonition_pattern.sub(lambda m: f"\n> **Note:** {m.group(1).strip()}\n", text)
        text = self.details_pattern.sub(lambda m: f"\n**{m.group(1).strip()}**\n{m.group(2).strip()}\n", text)
        text = re.sub(r'{/\*.*?\*/}', '', text, flags=re.DOTALL)
        
        # Wrap LaTeX content and remove original delimiters ($ or $$)
        text = self.latex_wrap_pattern.sub(self._process_latex_match, text)
        
        return text.strip()

    def _process_latex_match(self, match: Match) -> str:
        """
        Helper function to strip delimiters ($ or $$) from LaTeX matches 
        and wrap them in specific tokens.
        
        Args:
            match: The regex match object containing the LaTeX string.
            
        Returns:
            String formatted as: [LATEX_START] cleaned_content [LATEX_END]
        """
        content = match.group(1)
        
        # Strip block delimiters $$...$$
        if content.startswith('$$') and content.endswith('$$'):
            cleaned_content = content[2:-2].strip()
        
        # Strip inline delimiters $...$ (ensure it's not a single $)
        elif content.startswith('$') and content.endswith('$') and len(content) > 1:
            cleaned_content = content[1:-1].strip()
            
        # Keep \begin{...}...\end{...} as is, they are part of the syntax
        else:
            cleaned_content = content.strip()
            
        return f" [LATEX_START] {cleaned_content} [LATEX_END] "

    def _process_code_cell(self, cell: Dict, source_text: str) -> Dict[str, str]:
        """Formats code cells and extracts relevant outputs separately."""
        code_block = f"```python\n{source_text}\n```"
        
        outputs = []
        for output in cell.get('outputs', []):
            if 'text' in output:
                outputs.append("".join(output['text']))
            elif 'data' in output and 'text/plain' in output['data']:
                outputs.append("".join(output['data']['text/plain']))
        
        clean_output = ""
        if outputs:
            raw_output = "\n".join(outputs)[:1000]
            clean_output = self._clean_code_output(raw_output)

        return {
            "type": "code",
            "content": code_block,
            "output": clean_output
        }

    def _clean_code_output(self, text: str) -> str:
        """Removes HTML image tags and cleans extra whitespace from output."""
        text = self.output_image_pattern.sub('', text)
        return text.strip()

    def _extract_libraries(self, source_code: str) -> Set[str]:
        return set(self.import_pattern.findall(source_code))

    def _is_noise(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in self.noise_patterns)

    def _determine_qiskit_version(self, libraries: Set[str], raw_code: str) -> Optional[str]:
        if "qiskit" not in libraries:
            return None
        match = self.qiskit_version_pattern.search(raw_code)
        if match:
            return match.group(1)
        return "2.0.0+"

if __name__ == "__main__":
    processor = NotebookProcessor()
    processor.process_directory()