import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

persist_dir = "./chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_documents_from_folder(folder_path="Data"):
    documents = []

    for filename in os.listdir(folder_path):
        sanitized_filename = filename.strip()
        filepath = os.path.join(folder_path, sanitized_filename)

        file_metadata = {
            "source": filepath,
            "filename": sanitized_filename,
            "filetype": os.path.splitext(sanitized_filename)[1]
        }

        try:
            if sanitized_filename.endswith(".txt"):
                loader = TextLoader(filepath)
            elif sanitized_filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
            else:
                print(f"Skipping unsupported file: {sanitized_filename}")
                continue

            raw_docs = loader.load()
            for doc in raw_docs:
                doc.metadata.update(file_metadata)
                documents.append(doc)
        except Exception as e:
            print(f"⚠️ Error loading {sanitized_filename}: {e}")

    return documents

def split_documents(documents, chunk_size=300, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def store_in_chroma(chunks):
    chunks = [doc for doc in chunks if doc.page_content.strip()]
    if not chunks:
        print("No valid chunks to store.")
        return

    vectordb = Chroma.from_documents(
        chunks,
        embedding_model,
        persist_directory=persist_dir
    )
    vectordb.persist()
    return vectordb
