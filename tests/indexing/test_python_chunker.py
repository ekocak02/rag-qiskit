import pytest
import ast
from src.indexing.python_chunker import PythonProcessor

@pytest.fixture
def py_processor():
    """Standard processor with a reasonable token limit."""
    return PythonProcessor(token_limit=1000)

@pytest.fixture
def small_processor():
    """
    Processor with VERY tight limit to test splitting logic.
    Limit increased to 250 to satisfy: safe_limit (225) > chunk_overlap (200).
    """
    return PythonProcessor(token_limit=250)


def test_clean_code_removes_license(py_processor):
    """Should remove Qiskit/Apache license headers from source code."""
    license_text = """
    # This code is part of Qiskit.
    # Copyright IBM 2023.
    # This code is licensed under the Apache License 2.0.
    """
    code = "def my_func(): pass"
    full_text = f"{license_text}\n{code}"
    
    cleaned = py_processor._clean_code(full_text)
    
    assert "Copyright IBM" not in cleaned
    assert cleaned.strip() == code

def test_clean_code_keeps_content(py_processor):
    """Should strictly preserve code content outside headers."""
    code = "x = 1\n# Important comment\ny = 2"
    cleaned = py_processor._clean_code(code)
    assert cleaned == code


def test_extract_imports(py_processor):
    """Should identify direct imports and from-imports."""
    code = """
import os
import numpy as np
from qiskit import QuantumCircuit
from .local import module
    """
    tree = ast.parse(code)
    imports = py_processor._extract_imports(tree)
    
    expected_subset = ["os", "numpy", "qiskit.QuantumCircuit", "local.module"]
    for item in expected_subset:
        assert item in imports


def test_process_simple_function(py_processor):
    """Should chunk a standalone function correctly."""
    code = """
def simple_func():
    '''Docstring.'''
    return True
    """
    chunks = py_processor.process_file(code, "test.py")
    
    assert len(chunks) == 1
    assert chunks[0].metadata["function_name"] == "simple_func"
    assert chunks[0].metadata["type"] == "function_definition"

def test_process_class_structure(py_processor):
    """Should keep __init__ with class header and separate other methods."""
    code = """
class MyClass:
    '''Class Doc.'''
    def __init__(self):
        self.x = 1
    
    def method_one(self):
        return self.x
    """
    chunks = py_processor.process_file(code, "test.py")
    
    # Expectation: 
    # 1. Class definition (header + __init__)
    # 2. method_one
    assert len(chunks) >= 2
    
    class_chunk = next(c for c in chunks if c.metadata.get("type") == "class_definition")
    assert "class MyClass" in class_chunk.page_content
    assert "def __init__" in class_chunk.page_content
    
    method_chunk = next(c for c in chunks if c.metadata.get("function_name") == "method_one")
    assert "def method_one" in method_chunk.page_content
    assert method_chunk.metadata["parent_class"] == "MyClass"

def test_process_decorators(py_processor):
    """Should include decorators in the function chunk."""
    code = """
@property
@other_decorator
def decorated_func():
    pass
    """
    chunks = py_processor.process_file(code, "test.py")
    assert "@property" in chunks[0].page_content
    assert "@other_decorator" in chunks[0].page_content

def test_module_level_code(py_processor):
    """Should capture global variables or standalone code blocks."""
    code = """
import os
# Module level constant
CONSTANT = 42
def func(): pass
    """
    chunks = py_processor.process_file(code, "test.py")
    
    module_chunk = next((c for c in chunks if c.metadata.get("type") == "module_level"), None)
    
    assert module_chunk is not None
    assert "CONSTANT = 42" in module_chunk.page_content

def test_syntax_error_handling(py_processor):
    """Should return an error chunk instead of crashing on bad syntax."""
    bad_code = "def broken_func(:" # Missing )
    chunks = py_processor.process_file(bad_code, "bad.py")
    
    assert len(chunks) == 1
    assert chunks[0].metadata.get("error") == "syntax_error"
    assert chunks[0].page_content == bad_code

def test_large_docstring_split(small_processor):
    """Should split docstring from code if limit is exceeded."""
    long_doc = "word " * 300
    code = f"""
def huge_doc():
    '''{long_doc}'''
    pass
    """
    chunks = small_processor.process_file(code, "test.py")
    
    # Should result in multiple chunks due to limit (250)
    assert len(chunks) > 1
    
    types = [c.metadata.get("type") for c in chunks]
    assert "docstring_only" in types