import os
import logging
from google import genai
from typing import List, Dict, Any
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

logger = logging.getLogger(__name__)

class GeminiGenerator:
    """
    Generates answers using Google's gemini-2.5-flash-lite model via the new google-genai SDK.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.critical("GOOGLE_API_KEY not found in environment variables.")
            raise ValueError("GOOGLE_API_KEY is required.")
            
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        
        # System instruction to guide the model
        self.system_instruction = """
        You are an expert Qiskit Assistant. 
        Use the provided context to answer the user's question about Quantum Programming.
        
        Rules:
        1. Use ONLY the provided context. If the answer is not in the context, say "I don't know based on the provided documents."
        2. Pay attention to Qiskit versions in the metadata. If the context is from an old version (e.g., < 1.0), warn the user.
        3. Provide code snippets if available in the context.
        4. Be concise and technical.
        """

    @retry(
        retry=retry_if_exception_type(Exception), # Broad exception retry for 429s buried in SDK exceptions
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=20)
    )
    def _call_gemini(self, prompt: str):
        """Helper method to call Gemini with retry logic."""
        return self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Constructs a prompt and generates an answer.
        """
        if not context_chunks:
            return "I couldn't find any relevant documents to answer your question."

        context_str = ""
        prev_group_id = None
        
        for i, chunk in enumerate(context_chunks):
            meta = chunk.get('metadata', {})
            version = meta.get('qiskit_version', 'Unknown')
            source = meta.get('source', 'Unknown')
            context_path = meta.get('context_path', '')
            group_id = meta.get('split_group_id')
            chunk_idx = meta.get('chunk_index', 0)
            
            header_str = f"Document {i+1}"
            if group_id:
                header_str += f" (Part {chunk_idx + 1})"
                
            info_parts = [f"Source: {source}", f"Version: {version}"]
            if context_path:
                info_parts.append(f"Context: {context_path}")
                
            context_str += f"\n--- {header_str} | {', '.join(info_parts)} ---\n"
            context_str += chunk.get('content', '')
            context_str += "\n"

        prompt = f"""
        {self.system_instruction}
        
        User Question: {query}
        
        Context:
        {context_str}
        
        Answer:
        """
        
        try:
            response = self._call_gemini(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Generation failed after retries: {e}", exc_info=True)
            if "429" in str(e):
                 return "Sorry, I am currently receiving too many requests. Please try again in a few moments."
            return f"Sorry, I encountered an error while generating the answer. Error details: {str(e)}"
