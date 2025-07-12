
# 📚 RAG QA App – Retrieval-Augmented Generation with Streamlit

This is a fully functional Retrieval-Augmented Generation (RAG) application built using:

- 🧠 A small open-source LLM (e.g., `microsoft/phi-2`)
- 🗂️ Document ingestion (PDF and TXT)
- 🧩 Intelligent chunking and vector storage (via ChromaDB)
- 🧾 Semantic search with Sentence-Transformer embeddings
- 📺 Web interface using Streamlit

Users can **upload their own documents**, ask **natural language questions**, and get **accurate answers grounded in the uploaded content**.

## 🚀 Features

- 📁 Upload one or multiple `.pdf` or `.txt` documents
- ✂️ Automatically chunk text into clean 300-character pieces (with overlap)
- 🧠 Store vectorized chunks in [ChromaDB](https://www.trychroma.com/)
- 🔍 Use [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) embeddings for semantic search
- 🤖 Generate answers using [phi-2](https://huggingface.co/microsoft/phi-2) or any compatible small LLM
- 🔄 Modular design — can plug in different models, DBs, or UIs

## 🧱 Tech Stack

| Layer       | Library            | Description                                |
|-------------|--------------------|--------------------------------------------|
| UI          | Streamlit          | Interactive web interface                  |
| RAG         | LangChain          | Document processing and orchestration      |
| Vector DB   | Chroma             | Fast local document search                 |
| Embeddings  | SentenceTransformers | Embeds text into semantic space         |
| LLM         | Transformers       | Small language models like `phi-2`         |
| Inference   | BitsAndBytes       | 8-bit model loading for memory efficiency  |

## 📁 Project Structure

```
rag-app/
├── app.py               # Main Streamlit app
├── rag_utils.py         # Core backend logic for RAG
├── requirements.txt     # Dependency versions
├── chroma_db/           # Vector database (auto-created)
└── README.md            # Documentation
```

## 📦 Installation

### 🔧 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-app.git
cd rag-app
```

### 🐍 2. (Optional) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate       # Windows
```

### 📦 3. Install dependencies

```bash
pip install -r requirements.txt
```

### ✅ 4. Run the app

```bash
streamlit run app.py
```

## ⚙️ System Requirements

- Python 3.9 or later
- 8 GB RAM (16 GB recommended)
- GPU with at least 4GB VRAM for smooth 8-bit LLM loading (or fallback to CPU)
- Internet connection (for model and embedding downloads)

## 🧪 Usage Instructions

1. Launch the app with `streamlit run app.py`
2. Upload one or more `.pdf` or `.txt` files from the sidebar
3. The app will process and chunk them into smaller segments
4. Ask any question related to the content
5. The app will retrieve relevant chunks and generate an answer

## 🤖 Example Questions

```
- What is the Nobel Prize in Literature?
- Who was the first person to win a Nobel Prize?
- What happened during World War I to the Nobel awards?
```

## 🗃️ requirements.txt (Pinned Versions)

```txt
streamlit==1.35.0
langchain==0.2.2
chromadb==0.5.0
sentence-transformers==2.7.0
transformers==4.43.0
accelerate==0.30.1
bitsandbytes==0.43.1
pypdf==4.2.0
```

## 🌐 Deployment Options

- [Streamlit Cloud](https://streamlit.io/cloud)
- [Render](https://render.com/)
- [Hugging Face Spaces](https://huggingface.co/spaces)

## ✅ Todo

- [ ] Add chunk visualization
- [ ] Support `.docx`
- [ ] UI for model switching
- [ ] File history and cleanup
- [ ] Export answer + source citations

## 📬 Contact

**Abdullah Ashraf**  
🧠 AI Engineer & Data Scientist  
📧 [Abdullah182155@gmail.com](mailto:Abdullah182155@gmail.com)  
📍 Banha, Egypt

## 🪪 License

This project is licensed under the MIT License. Feel free to use, share, or modify with credit.
