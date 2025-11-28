# Ai-professor-voice-to-voice-with-RAG
AI Professor is an intelligent voice-driven assistant designed to help students interact with university content naturally. It transcribes spoken questions, retrieves relevant information using a RAG pipeline built on university documents, generates clear and accurate answers with an LLM, and responds back in high-quality synthesized speech.

to run the project locally:
1- install all packages in the requirements.txt
2- make your Qdrant cluster active 
3- run your local Ollama
4- run "python app.py"

if runnning on GPU:
- ensure that you have installed cudNN ver 9.0 and from cuda toolkit cublaslt ver 9.0
