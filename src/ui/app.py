import gradio as gr
import requests
import os

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

def query_api(message, history):
    """
    Sends the user message to the FastAPI backend and returns the answer.
    """
    try:
        payload = {"query": message}
        response = requests.post(f"{API_URL}/query", json=payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "No answer received.")
            sources = data.get("sources", [])
            
            # Format sources for display
            source_text = "\n\n**Sources:**\n"
            for i, src in enumerate(sources[:3]): # Show top 3 sources
                meta = src.get('metadata', {})
                filename = meta.get('filename') or meta.get('source') or 'Unknown'
                version = meta.get('qiskit_version', 'N/A')
                source_text += f"- **{filename}** (v{version})\n"
            
            return f"{answer}{source_text}"
        else:
            return f"Error: API returned status {response.status_code}\n{response.text}"
            
    except Exception as e:
        return f"Connection Error: {str(e)}"

# Custom CSS for a cleaner look
custom_css = """
#chatbot {min_height: 500px;}
"""

# Build the Interface
with gr.Blocks(title="Qiskit RAG Assistant") as demo:
    gr.Markdown("# ⚛️ Qiskit RAG Assistant")
    gr.Markdown("Ask questions about Quantum Programming with Qiskit. Powered by RAG & Gemini.")
    
    chatbot = gr.ChatInterface(
        fn=query_api,
        chatbot=gr.Chatbot(elem_id="chatbot"),
        textbox=gr.Textbox(placeholder="How do I create a Bell state?", container=False, scale=7),
        examples=[
            "How do I create a Bell state in Qiskit?",
            "Explain dynamic circuits.",
            "What is the difference between QuantumCircuit and DAGCircuit?"
        ]
    )

if __name__ == "__main__":
    # Get auth from env if set
    auth = None
    user = os.getenv("GRADIO_USER")
    password = os.getenv("GRADIO_PASSWORD")
    if user and password:
        auth = (user, password)
        
    demo.launch(server_name="0.0.0.0", server_port=7860, auth=auth, share=False)
