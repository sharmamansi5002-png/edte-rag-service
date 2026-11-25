import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Semantic GraphRAG Engine",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Semantic GraphRAG Engine")
st.write("Hybrid vector and knowledge graph retrieval system.")

# API status
st.subheader("System Status")

try:
    response = requests.get(f"{API_URL}/health", timeout=5)
    if response.ok:
        st.success("API and Neo4j database are connected.")
    else:
        st.error("API is running, but database health check failed.")
except requests.RequestException:
    st.warning("API is not running. Start the FastAPI server first.")

st.divider()

# Document upload
st.header("📄 Upload Documents")

uploaded_file = st.file_uploader(
    "Choose a Markdown or PDF file",
    type=["md", "pdf"],
)

if st.button("Upload and Index", disabled=uploaded_file is None):
    with st.spinner("Uploading and indexing document..."):
        try:
            response = requests.post(
                f"{API_URL}/api/v1/rag/ingest",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                    )
                },
                timeout=300,
            )

            if response.ok:
                result = response.json()
                st.success(result["message"])
                st.write("Document URN:", result.get("document_urn"))
                st.write("Chunks indexed:", result.get("chunks_indexed"))
                st.json(result.get("entities", {}))
            else:
                st.error(f"Upload failed: {response.text}")

        except requests.RequestException as error:
            st.error(f"Could not connect to API: {error}")

st.divider()

# Question answering
st.header("💬 Ask a Question")

question = st.text_area(
    "Enter your question",
    placeholder="What information is available about the Finance System?",
)

col1, col2 = st.columns(2)

with col1:
    top_k = st.number_input(
        "Number of retrieved chunks",
        min_value=1,
        max_value=20,
        value=5,
    )

with col2:
    graph_hops = st.selectbox(
        "Graph expansion hops",
        options=[1, 2],
        index=0,
    )

if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching documents and knowledge graph..."):
            try:
                response = requests.post(
                    f"{API_URL}/api/v1/rag/query",
                    json={
                        "question": question,
                        "top_k": int(top_k),
                        "expand_graph_hops": graph_hops,
                    },
                    timeout=300,
                )

                if response.ok:
                    result = response.json()

                    st.subheader("Answer")
                    st.write(result.get("answer", "No answer returned."))

                    st.subheader("Retrieved Sources")
                    sources = result.get("sources", [])

                    if sources:
                        for index, source in enumerate(sources, start=1):
                            with st.expander(f"Source {index}"):
                                st.write("Chunk:", source.get("chunk_urn"))
                                st.write("Score:", source.get("score"))
                                st.write(source.get("text_snippet"))
                    else:
                        st.info("No source chunks returned.")

                    st.subheader("Knowledge Graph Context")
                    graph_context = result.get("graph_context", [])

                    if graph_context:
                        st.json(graph_context)
                    else:
                        st.info("No graph context returned.")

                else:
                    st.error(f"Query failed: {response.text}")

            except requests.RequestException as error:
                st.error(f"Could not connect to API: {error}")