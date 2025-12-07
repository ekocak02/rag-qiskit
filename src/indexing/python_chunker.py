import ast
import re
import uuid
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.indexing.utils import BaseProcessor, ProcessedChunk

class PythonProcessor(BaseProcessor):
    """
    Processes .py files using AST parsing logic.
    """

    def __init__(self, token_limit: int = 2000):
        super().__init__(token_limit=token_limit)
        
        # Setup recursive splitter for fallback
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.token_helper.safe_limit,
            chunk_overlap=200,
            length_function=self.token_helper.count_tokens,
            separators=["\nclass ", "\ndef ", "\n\tdef ", "\n\n", "\n", " ", ""]
        )

    def process_file(self, file_content: str, filename: str) -> List[ProcessedChunk]:
        clean_content = self._clean_code(file_content)
        source_lines = clean_content.splitlines()
        
        try:
            tree = ast.parse(clean_content)
        except SyntaxError:
            return self._create_chunk(clean_content, {"source": filename, "error": "syntax_error"})

        file_imports = self._extract_imports(tree)
        base_metadata = {
            "source": filename,
            "dependencies": file_imports,
            "content_type": "code_mixed",
            "qiskit_version": "2.0.1" 
        }

        all_chunks = []

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                all_chunks.extend(self._process_class(node, source_lines, base_metadata))
            elif isinstance(node, ast.FunctionDef):
                all_chunks.extend(self._process_function(node, source_lines, base_metadata))
            else:
                if not isinstance(node, (ast.Import, ast.ImportFrom)):
                     source = self._get_node_source_with_decorators(node, source_lines)
                     if source.strip():
                        all_chunks.extend(self._create_chunk(source, {**base_metadata, "type": "module_level"}))

        return all_chunks

    def _clean_code(self, source_code: str) -> str:
        qiskit_license_pattern = r"(?s)^\s*#\s*This code is part of Qiskit\..*?Copyright\s+IBM.*?(http://www\.apache\.org/licenses/LICENSE-2\.0|LICENSE\.txt).*?altered from the originals\.\n"
        generic_pattern = r"(?i)^\s*(#|/{2,}).*?(copyright|license|apache).*?(\n\s*(#|/{2,}).*?)*\n"
        
        cleaned_code = re.sub(qiskit_license_pattern, "", source_code, flags=re.MULTILINE)
        if len(cleaned_code) == len(source_code):
             cleaned_code = re.sub(generic_pattern, "", source_code, count=1, flags=re.MULTILINE | re.DOTALL)
        return cleaned_code.strip()

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        return imports

    def _get_node_source_with_decorators(self, node: ast.AST, full_source_lines: List[str]) -> str:
        if not hasattr(node, 'lineno') or not hasattr(node, 'end_lineno'):
            return ""
        start_line = node.lineno
        if hasattr(node, 'decorator_list') and node.decorator_list:
            start_line = min(d.lineno for d in node.decorator_list)
        return "\n".join(full_source_lines[start_line-1 : node.end_lineno])

    def _create_chunk(self, content: str, meta_template: Dict, type_override: str = None) -> List[ProcessedChunk]:
        token_count = self.token_helper.count_tokens(content)
        metadata = meta_template.copy()
        metadata["token_count"] = token_count
        if type_override:
            metadata["type"] = type_override

        if token_count <= self.token_helper.safe_limit:
            return [ProcessedChunk(page_content=content, metadata=self.clean_metadata(metadata))]

        chunks = []
        splits = self.recursive_splitter.split_text(content)
        
        # Generate Group ID for this split function/class
        split_group_id = f"py_split_{uuid.uuid4().hex[:8]}"

        for i, split in enumerate(splits):
            chunk_meta = metadata.copy()
            chunk_meta.update({
                "chunk_index": i,
                "split_method": "recursive_fallback",
                "token_count": self.token_helper.count_tokens(split),
                "split_group_id": split_group_id  # Link parts
            })
            chunks.append(ProcessedChunk(page_content=split, metadata=self.clean_metadata(chunk_meta)))
        return chunks

    def _process_class(self, node: ast.ClassDef, source_lines: List[str], base_meta: Dict) -> List[ProcessedChunk]:
        chunks = []
        class_name = node.name
        class_meta = base_meta.copy()
        class_meta.update({"class_name": class_name, "type": "class_definition"})

        header_lines = []
        if node.decorator_list:
            for decorator in node.decorator_list:
                header_lines.append(self._get_node_source_with_decorators(decorator, source_lines))
        bases = [ast.unparse(b) for b in node.bases]
        header_lines.append(f"class {class_name}({', '.join(bases)}):")
        header_block = "\n".join(header_lines)

        docstring = ast.get_docstring(node) or ""
        init_method = next((c for c in node.body if isinstance(c, ast.FunctionDef) and c.name == "__init__"), None)
        other_methods = [c for c in node.body if isinstance(c, ast.FunctionDef) and c.name != "__init__"]

        parent_parts = [header_block]
        if docstring: parent_parts.append(f'    """{docstring}"""')
        if init_method:
            parent_parts.append(self._get_node_source_with_decorators(init_method, source_lines))
        else:
            parent_parts.append("    # No __init__ method")

        parent_content = "\n".join(parent_parts)

        if self.token_helper.count_tokens(parent_content) > self.token_helper.safe_limit and docstring:
            doc_meta = class_meta.copy()
            doc_meta["type"] = "docstring_only"
            # Docstring split handles its own ID in create_chunk if needed
            chunks.extend(self._create_chunk(docstring, doc_meta))
            
            code_only = f"{header_block}\n" + (self._get_node_source_with_decorators(init_method, source_lines) if init_method else "")
            chunks.extend(self._create_chunk(code_only, class_meta))
        else:
            chunks.extend(self._create_chunk(parent_content, class_meta))

        for method in other_methods:
            method_meta = class_meta.copy()
            method_meta["parent_class"] = class_name
            chunks.extend(self._process_function(method, source_lines, method_meta))
        return chunks

    def _process_function(self, node: ast.FunctionDef, source_lines: List[str], base_meta: Dict) -> List[ProcessedChunk]:
        chunks = []
        func_name = node.name
        func_meta = base_meta.copy()
        func_meta.update({"function_name": func_name, "type": "function_definition"})

        full_source = self._get_node_source_with_decorators(node, source_lines)
        docstring = ast.get_docstring(node)

        if self.token_helper.count_tokens(full_source) > self.token_helper.safe_limit and docstring:
            doc_meta = func_meta.copy()
            doc_meta["type"] = "docstring_only"
            chunks.extend(self._create_chunk(docstring, doc_meta))
            chunks.extend(self._create_chunk(full_source, func_meta))
        else:
            chunks.extend(self._create_chunk(full_source, func_meta))
        return chunks