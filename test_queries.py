from src.graph import graph


queries = [
    "What is Agentic AI?",
    "How does Agentic AI differ from traditional AI?",
    "What are the components of an Agentic AI system?",
    "What are practical applications of Agentic AI?",
    "What is the role of planning in Agentic AI?",
    "What is the capital of France?"
]


for i, query in enumerate(queries, start=1):
    print("\n" + "=" * 70)
    print(f"TEST {i}")
    print("=" * 70)
    print(f"Question: {query}")

    result = graph.invoke({
        "question": query,
        "context": [],
        "answer": "",
        "score": 0.0
    })

    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nConfidence: {result['score']}")

    print("\nRetrieved Pages:")
    for chunk in result["context"]:
        print(
            f"- Page {chunk['page']} | "
            f"{chunk['source']}"
        )