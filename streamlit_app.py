import streamlit as st
from src.graph import build_rag_graph

st.set_page_config(page_title="Agentic AI RAG Chatbot", layout="wide")
st.title("Agentic AI eBook Chatbot")


@st.cache_resource
def get_graph():
    return build_rag_graph()


query = st.chat_input("Ask a question about the Agentic AI eBook")
if query:
    r = get_graph().invoke({
        "query": query, "context": [], "pages": [], "retrieval_score": 0.0,
        "answer": "", "grounded_score": 0.0, "attempts": 0,
    })
    st.chat_message("user").write(query)
    st.chat_message("assistant").write(r["answer"])
    with st.sidebar:
        st.metric("Confidence score", r["grounded_score"])
        st.caption(f"Top retrieval similarity: {r['retrieval_score']}")
        st.subheader("Retrieved chunks")
        for chunk, page in zip(r["context"], r["pages"]):
            with st.expander(f"Page {page}"):
                st.write(chunk)
