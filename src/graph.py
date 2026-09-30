import os
from typing import TypedDict

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "agentic-ai-rag-free"
)

# Free local embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Pinecone connection
pc = Pinecone(
    api_key=PINECONE_API_KEY
)

index = pc.Index(INDEX_NAME)


class AgentState(TypedDict):
    question: str
    context: list
    answer: str
    score: float


def retrieve(state: AgentState):
    question = state["question"]

    # Convert question into embedding
    query_vector = embedding_model.encode(
        question
    ).tolist()

    # Search Pinecone
    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )

    contexts = []
    scores = []

    for match in results["matches"]:
        metadata = match.get(
            "metadata",
            {}
        )

        contexts.append({
            "text": metadata.get(
                "text",
                ""
            ),
            "page": metadata.get(
                "page",
                0
            ),
            "source": metadata.get(
                "source",
                "Ebook-Agentic-AI.pdf"
            )
        })

        scores.append(
            match.get(
                "score",
                0
            )
        )

    # Calculate average confidence score
    average_score = (
        sum(scores) / len(scores)
        if scores
        else 0
    )

    return {
        "context": contexts,
        "score": round(
            average_score,
            4
        )
    }


def generate(state: AgentState):
    contexts = state["context"]
    score = state["score"]

    # Reject low-confidence / out-of-scope questions
    if not contexts or score < 0.30:
        return {
            "answer": (
                "I could not find relevant information "
                "in the Agentic AI eBook to answer this question."
            )
        }

    # Return the most relevant retrieved context
    best_context = contexts[0]["text"]

    return {
        "answer": best_context
    }


# Create LangGraph workflow
workflow = StateGraph(
    AgentState
)

# Add nodes
workflow.add_node(
    "retrieve",
    retrieve
)

workflow.add_node(
    "generate",
    generate
)

# Define workflow
workflow.add_edge(
    START,
    "retrieve"
)

workflow.add_edge(
    "retrieve",
    "generate"
)

workflow.add_edge(
    "generate",
    END
)

# Compile graph
graph = workflow.compile()