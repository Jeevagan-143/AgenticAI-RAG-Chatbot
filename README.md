\# Agentic AI RAG Chatbot



A Retrieval-Augmented Generation (RAG) chatbot built with Python, LangGraph, Pinecone, FastAPI, and a free local embedding model.



The chatbot retrieves relevant information from the provided Agentic AI eBook and returns the retrieved context with a confidence score.



\## Features



\- PDF-based knowledge retrieval

\- Pinecone vector database

\- LangGraph retrieval workflow

\- FastAPI REST API

\- Local free embeddings using Sentence Transformers

\- Confidence score

\- Out-of-scope question handling

\- Retrieved source pages included in API response



\## Tech Stack



\- Python 3.12

\- LangGraph

\- Pinecone

\- FastAPI

\- Sentence Transformers

\- PyMuPDF

\- LangChain Text Splitters



\## Project Structure



```text

AgenticAI-RAG-Chatbot/

│

├── data/

│   └── Ebook-Agentic-AI.pdf

│

├── src/

│   ├── \_\_init\_\_.py

│   ├── config.py

│   ├── ingestion.py

│   ├── graph.py

│   └── upload\_data.py

│

├── app.py

├── test\_queries.py

├── requirements.txt

├── .env.example

├── .gitignore

└── README.md

