# AI-Professor-Voice-to-Voice-with-RAG

**AI Professor** is an intelligent, voice-driven assistant designed to help students interact with university content naturally. It:

* Transcribes spoken questions.
* Retrieves relevant information using a **RAG (Retrieval-Augmented Generation)** pipeline built on university documents.
* Generates clear and accurate answers using a large language model (LLM).
* Responds back with high-quality synthesized speech.

---

## 📥 Running the Project Locally

1. Ensure you are using **Python 3.11**.

2. Install all dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure your **Qdrant cluster** is active.

4. Start your local **Ollama** instance.

5. Run the application:

```bash
python app.py
```

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
