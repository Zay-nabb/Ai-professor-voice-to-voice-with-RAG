# AI-Professor-Voice-to-Voice-with-RAG

**AI Professor** is an intelligent, voice-driven assistant designed to help students interact with university content naturally. It:

* Transcribes spoken questions.
* Retrieves relevant information using a **RAG (Retrieval-Augmented Generation)** pipeline built on university documents.
* Generates clear and accurate answers using a large language model (LLM).
* Responds back with high-quality synthesized speech.

---

## 🔐 Environment Setup

**IMPORTANT:** Before running the project, you must configure environment variables for both the client and RAG server.

### RAG Server Configuration

1. Navigate to the RAG server directory:

```bash
cd ai-professor-vs/rag_server
```

2. Copy the example environment file:

```bash
cp .env.example .env
```

3. Edit `.env` and add your API credentials:

```bash
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
HF_API_TOKEN=your_huggingface_token
```

### Client Configuration

1. Navigate to the client directory:

```bash
cd ai-professor-vs/client
```

2. Copy the example environment file:

```bash
cp .env.example .env
```

3. The default values should work, but you can customize if needed.

---

## 📥 Running the Project Locally

1. Ensure you are using **Python 3.11**.

2. **Set up environment variables** (see Environment Setup section above).

3. Install dependencies for both client and server:

**RAG Server:**
```bash
cd ai-professor-vs/rag_server
pip install -r requirements.txt
```

**Client:**
```bash
cd ai-professor-vs/client
pip install -r requirements.txt
```

4. Make sure your **Qdrant cluster** is active.

5. Start the **RAG server** first:

```bash
cd ai-professor-vs/rag_server
uvicorn app:app --host 0.0.0.0 --port 8000
```

6. In a new terminal, start the **client**:

```bash
cd ai-professor-vs/client
python app.py
```

---

## 🔒 Security Best Practices

- **Never commit `.env` files** to version control
- Keep your API keys and tokens secure
- Rotate credentials regularly
- Use `.env.example` as a template for required variables
- The `.gitignore` file is configured to exclude sensitive files

---

## ⚡ Running on GPU

If using a GPU, ensure the following are installed and configured:

* **cuDNN version 9.0**
* **cuBLASLt version 9.0** (from CUDA Toolkit)
* Add the CUDA and cuDNN **bin directories to your system PATH/environment variables**.

> Example (Windows):

```text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
C:\tools\cuda\bin
```

## 📝 Notes

* Ensure your system meets the memory requirements for the selected LLM model.
* Adjust Ollama model parameters or use quantized models if encountering memory issues.
