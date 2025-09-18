import streamlit as st
from query_processor import QueryProcessor
from rag import RAG
import os
from run_pipeline import run_full_pipeline

st.set_page_config(page_title="Legal Document Q&A System", layout="wide")

st.title("Legal Document Q&A System")

query_processor = QueryProcessor()
rag = RAG()

def main():
    st.header("Ask a Legal Question")
    query = st.text_input("Enter your question here:")

    if st.button("Get Answer") and query.strip():
        with st.spinner("Retrieving relevant passages..."):
            results = query_processor.process_query(query)
        with st.spinner("Generating answer..."):
            response = rag.generate_answer(query, results)
            answer = response.get("answer", "No answer generated.")
            citations = response.get("citations", [])

        st.subheader("Answer")
        st.write(answer)

        if citations:
            st.subheader("Citations")
            for citation in citations:
                st.markdown(f"- {citation}")

    st.sidebar.header("Upload Additional Documents")
    uploaded_files = st.sidebar.file_uploader("Upload .txt files", accept_multiple_files=True, type=["txt"])
    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_path = os.path.join("uploaded_docs", uploaded_file.name)
            os.makedirs("uploaded_docs", exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"Uploaded {len(uploaded_files)} files successfully.")

if __name__ == "__main__":
    main()
