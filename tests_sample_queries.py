import requests

QUERIES = [
    "What is the core definition of Agentic AI as outlined in the eBook?",
    "What are the main architectural components required to build agentic systems?",
    "What real-world industry use cases for Agentic AI are discussed in the eBook?",
    "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
    "What key challenges or limitations of Agentic AI are mentioned in the document?",
    "What is the capital of France?",
]

for q in QUERIES:
    r = requests.post("http://localhost:8000/chat", json={"query": q}).json()
    print(f"\nQ: {r['query']}\nA: {r['final_answer']}")
    print(f"Confidence: {r['confidence_score']} | Chunks: {len(r['retrieved_context_chunks'])}")
