# 🤖 Resume vs Job Match RAG Chatbot

A privacy-focused AI assistant to match your resume with job descriptions — or analyze any private document locally. Built with **LangChain**, **FAISS**, and **Ollama (LLaMA3)**. No data leaves your machine.

---

## 🔍 Use Cases

- Match your **resume** to a job description.
- Analyze **confidential documents**, such as:
  - Internal company reports
  - Academic papers under peer review
  - Unpublished research or grant proposals
  - Copyright-protected content

---

## 💡 Features

- 💾 Upload and process local documents (`.pdf`, `.docx`, `.txt`, etc.)
- 🔍 Ask questions or paste job descriptions to get document-aware responses
- 🔐 100% local — no OpenAI, no cloud services
- 🧠 RAG (Retrieval-Augmented Generation) pipeline using FAISS + LLM

---

## ⚙️ Requirements

Install system dependencies:

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install tesseract-ocr poppler-utils
```

Clone and install Python dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/resume-job-match-rag.git
cd resume-job-match-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Install and run `ollama`:
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama run llama3
```

---

## 🚀 Run the App

```bash
python app.py
```

The web UI will open in your browser.

---

## 📂 Supported File Formats

- `.pdf`
- `.docx`
- `.txt`
- `.md`
- `.csv`

---

## 🛡️ Why Local?

> **Don’t upload confidential content to public AI tools.**

This tool is ideal for:

- Professionals working under **NDAs**
- Researchers with **unpublished content**
- Teams analyzing **in-house documentation**

Your data stays on your computer. Period.

---

## 🙌 Built With

- [LangChain](https://www.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Ollama](https://ollama.com/)
- [LLaMA3](https://llama.meta.com/)
- [Gradio](https://www.gradio.app/)

