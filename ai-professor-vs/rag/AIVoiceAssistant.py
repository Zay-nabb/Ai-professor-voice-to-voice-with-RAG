from qdrant_client import QdrantClient
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os # Import os for file paths

import warnings
warnings.filterwarnings("ignore")

class AIVoiceAssistant:
    def __init__(self):
        # --- QDRANT CLOUD CHANGES ---
        # 1. Put your Cluster URL here
        self._qdrant_url = "https://a21f5be5-d41b-4c39-98e2-5daf7f6148ad.us-west-2-0.aws.cloud.qdrant.io" 
        # 2. Put your API Key here
        self._api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.zC9KNvcPMOaGhLPy4xmoBlfuDnEMIba6KQgjwfxbdnM"

        # 3. Connect to Qdrant Cloud with the API Key
        self._client = QdrantClient(
            url=self._qdrant_url,
            api_key=self._api_key,
        )
        # --- END QDRANT CHANGES ---

        self._llm = Ollama(model="deepseek-r1:8b", request_timeout=120.0)
        
        # This service context is deprecated, let's use the new way
        # self._service_context = ServiceContext.from_defaults(llm=self._llm, embed_model="local")
        # New way (LlamaIndex > 0.10):
        from llama_index.core import Settings
        Settings.llm = self._llm
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        

        self._index = None
        self._create_kb()
        self._create_chat_engine()

    def _create_chat_engine(self):
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        self._chat_engine = self._index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt=self._prompt, # This will use our new professor prompt
        )

    def _create_kb(self):
        try:
            # --- PROFESSOR KB CHANGES ---
            # 1. Define the path using os.path.join for compatibility
            kb_file_path = os.path.join("rag", "professor_kb.txt")

            # Verify file exists
            if not os.path.exists(kb_file_path):
                print(f"Error: Knowledge base file not found at {kb_file_path}")
                return

            reader = SimpleDirectoryReader(
                input_files=[kb_file_path]
            )
            documents = reader.load_data()
            
            # 2. Change the collection_name to something new
            collection_name = "professor_db"
            
            # Check if collection already exists, if so, use it. If not, create it.
            # This avoids re-uploading the same docs every time.
            collections = self._client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            if collection_name in collection_names:
                # Collection exists, just load it
                vector_store = QdrantVectorStore(client=self._client, collection_name=collection_name)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                self._index = VectorStoreIndex.from_vector_store(
                    vector_store, storage_context=storage_context
                )
                print(f"Loaded existing knowledgebase: {collection_name}")
            else:
                # Collection doesn't exist, create it
                vector_store = QdrantVectorStore(client=self._client, collection_name=collection_name)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                self._index = VectorStoreIndex.from_documents(
                    documents, storage_context=storage_context
                )
                print(f"Created new knowledgebase: {collection_name}")
            # --- END PROFESSOR KB CHANGES ---

        except Exception as e:
            print(f"Error while creating knowledgebase: {e}")

    def interact_with_llm(self, customer_query):
        if not customer_query or not customer_query.strip():
            return None
        try:
            AgentChatResponse = self._chat_engine.chat(customer_query)
            answer = AgentChatResponse.response
            return answer
        except Exception as e:
            print(f"Error interacting with LLM: {e}")
            return "I'm sorry, I encountered an error processing that request."

    @property
    def _prompt(self):
        # --- PROFESSOR PROMPT CHANGE ---
        return """
            You are a helpful AI Professor. Your goal is to answer questions
            about the documents provided.
            
            Use the context from the documents to answer the user's question.
            If the answer is not in the context, clearly state that you
            do not have that information.
            
            Keep your answers concise and academic, but easy to understand.
            Do not make up information.
            """