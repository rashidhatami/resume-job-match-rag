import os
from pathlib import Path
from langchain.document_loaders import (
    TextLoader, UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader,
    UnstructuredCSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import gradio as gr
import shutil

UPLOAD_DIR = "./uploaded_resumes"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Select appropriate loader based on file extension
def get_loader_for_file(filepath):
    ext = Path(filepath).suffix.lower()
    if ext == ".txt":
        return TextLoader(filepath, encoding='utf-8')
    elif ext == ".md":
        return UnstructuredMarkdownLoader(filepath)
    elif ext == ".pdf":
        return UnstructuredPDFLoader(filepath)
    elif ext == ".docx":
        return UnstructuredWordDocumentLoader(filepath)
    elif ext == ".csv":
        return UnstructuredCSVLoader(filepath)
    else:
        return None

# Text splitter configuration
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n### ", "\n## ", "\n# ", "\n- ", "\n\n", "\n", " ", ""]
)

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Local LLM
llm = Ollama(model="llama3")

# Chat memory
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# Create/load FAISS index
faiss_index_path = "faiss_index"

if os.path.exists(faiss_index_path) and not os.path.isdir(faiss_index_path):
    os.remove(faiss_index_path)

if os.path.exists(faiss_index_path):
    vector_store = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)
else:
    vector_store = FAISS.from_texts(["No document has been uploaded yet."], embedding_model)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# Upload and process resume
def upload_resume(file):
    filename = os.path.basename(file.name)
    save_path = os.path.join(UPLOAD_DIR, filename)
    shutil.copy(file.name, save_path)
    
    loader = get_loader_for_file(save_path)
    if not loader:
        return "Unsupported file format."

    try:
        documents = loader.load()
        chunks = splitter.split_documents(documents)
        vector_store.add_documents(chunks)
        vector_store.save_local(faiss_index_path)
        return f"'{filename}' uploaded and indexed successfully."
    except Exception as e:
        return f"Error processing document: {str(e)}"

# Chat function based on job description
def chat_with_resume(job_description, history):
    result = conversation_chain.invoke({"question": job_description})
    return result["answer"]

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 RAG Chatbot")

    with gr.Row():
        resume_uploader = gr.File(label="📎 Upload Your Document (PDF, DOCX, etc.)")
        upload_btn = gr.Button("⬆️ Process Document")
        upload_output = gr.Textbox(label="Upload Status")

    upload_btn.click(fn=upload_resume, inputs=resume_uploader, outputs=upload_output)

    # gr.Markdown("## 📝 ")
    chat_interface = gr.ChatInterface(fn=chat_with_resume)

demo.launch(inbrowser=True)
