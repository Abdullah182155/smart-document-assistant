import streamlit as st
from rag_utils import load_documents_from_folder, split_documents, store_in_chroma
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

st.set_page_config(page_title="RAG QA App", layout="wide")
st.title("📚 RAG QA App")

# Load documents
st.sidebar.header("📂 Document Settings")
st.sidebar.info("Loading documents from static path: ./data")
documents = load_documents_from_folder("data")

if not documents:
    st.error("No documents found in the 'data' folder.")
    st.stop()

st.success(f"Loaded {len(documents)} documents.")

# Split documents
st.sidebar.subheader("🔧 Chunking Parameters")
chunk_size = st.sidebar.slider("Chunk Size", 100, 1000, 300, 50)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 300, 50, 10)

chunks = split_documents(documents, chunk_size, chunk_overlap)
st.success(f"Generated {len(chunks)} text chunks.")

# Store in ChromaDB
persist_dir = "./chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if st.sidebar.button("Store in ChromaDB"):
    store_in_chroma(chunks)
    st.success("Chunks stored successfully in ChromaDB.")

# QA Section
st.header("💬 Ask a Question")
query = st.text_input("Type your question here:")

if query:
    with st.spinner("Thinking..."):
        # Load retriever
        retriever = Chroma(persist_directory=persist_dir, embedding_function=embedding_model).as_retriever()

        # Load Phi-2 model
        model_id = "microsoft/phi-2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
        llm = HuggingFacePipeline(pipeline=pipe)

        # Retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
        result = qa_chain({"query": query})

        st.subheader("📖 Answer:")
        st.write(result["result"])

        with st.expander("📚 Source Documents"):
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**Source {i+1}:** `{doc.metadata.get('filename', '')}`")
                st.text(doc.page_content[:500] + "...")
