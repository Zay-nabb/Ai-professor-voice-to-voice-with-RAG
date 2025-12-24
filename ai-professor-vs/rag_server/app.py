from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.base.llms.types import CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from typing import Any

# ---------------- CONFIG ----------------

# Load configuration from environment variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

HF_API_URL = os.getenv("HF_API_URL")
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "CS_sources")

SYSTEM_PROMPT = """You are a helpful AI Professor.
Answer using the provided context only.
If the answer is not found, say you do not know.
Be concise and academic."""

# Validate required environment variables
required_vars = {
    "QDRANT_URL": QDRANT_URL,
    "QDRANT_API_KEY": QDRANT_API_KEY,
    "HF_API_URL": HF_API_URL,
    "HF_MODEL_NAME": HF_MODEL_NAME,
    "HF_API_TOKEN": HF_API_TOKEN
}

missing_vars = [var for var, value in required_vars.items() if not value]
if missing_vars:
    raise ValueError(
        f"Missing required environment variables: {', '.join(missing_vars)}. "
        f"Please create a .env file based on .env.example and set all required values."
    )

# --------------------------------------

app = FastAPI(title="AI Professor RAG Server")

class Query(BaseModel):
    question: str

# --------- Hugging Face LLM Class (extends CustomLLM) ---------
class HuggingFaceLLM(CustomLLM):
    """Custom LLM that wraps Hugging Face Inference API."""
    
    def __init__(self, api_url: str, api_token: str, system_prompt: str = "", **kwargs):
        super().__init__(**kwargs)
        self._api_url = api_url
        self._api_token = api_token
        self._system_prompt = system_prompt
    
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=4096,
            num_output=512,
            is_chat_model=False,
        )
    
    def _call_hf_api(self, prompt: str) -> str:
        print("Sending request to HF Router...")
        headers = {
            "Authorization": f"Bearer {self._api_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": HF_MODEL_NAME,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 256,
            "temperature": 0.2
        }

        try:
            response = requests.post(
                self._api_url,
                headers=headers,
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            data = response.json()
            print("HF Router responded.")
            return data["choices"][0]["message"]["content"]
        

        except Exception as e:
            print("HF Router error:", e)
            return "Model not available right now"


    
    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        """Complete the prompt."""
        text = self._call_hf_api(prompt)
        return CompletionResponse(text=text)
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        """Stream complete the prompt."""
        response = self.complete(prompt, formatted, **kwargs)
        yield response

# --------- Create LLM instance ---------
hf_llm = HuggingFaceLLM(
    api_url=HF_API_URL,
    api_token=HF_API_TOKEN,
    system_prompt=SYSTEM_PROMPT
)

# Set the LLM in Settings
Settings.llm = hf_llm

# --------- EMBEDDINGS ---------
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------- VECTOR STORE ---------
try:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    index = VectorStoreIndex.from_vector_store(vector_store)
    chat_engine = index.as_chat_engine(chat_mode="condense_plus_context",similarity_top_k=3)
    print(f"[OK] Successfully connected to Qdrant collection: {COLLECTION_NAME}")
except Exception as e:
    chat_engine = None
    index = None

# --------- API ---------
@app.post("/rag")
def rag_query(query: Query):
    if chat_engine is None:
        return {"answer": "Error: Vector store not initialized. Please check Qdrant connection and collection."}
    try:
        response = chat_engine.chat(query.question)
        # Extract response text safely
        if hasattr(response, 'response'):
            answer = response.response
        elif hasattr(response, 'message') and hasattr(response.message, 'content'):
            answer = response.message.content
        else:
            answer = str(response)
        
        # Ensure answer is a string
        if not isinstance(answer, str):
            answer = str(answer) if answer is not None else ""
        
        return {"answer": answer}
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error processing query: {error_details}")
        return {"answer": f"Error processing query: {str(e)}"}

@app.get("/health")
def health():
    return {"status": "ok"}
