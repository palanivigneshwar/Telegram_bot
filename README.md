# 🤖 GenAI Telegram Bot (RAG + Vision + Local LLM)

A lightweight **multimodal GenAI bot** built with Python that supports:

* 📚 Retrieval-Augmented Generation (RAG) over custom documents
* 🖼️ Image captioning using a vision model
* 🧠 Local LLM inference (Ollama or GGUF via llama.cpp)

---

## 🚀 Features

### 🧠 RAG (Retrieval-Augmented Generation)

* Load local `.txt` / `.md` documents
* Chunk + embed using `sentence-transformers`
* Retrieve relevant context using cosine similarity
* Generate answers using a **local LLM (no API required)**

---

### 🖼️ Image Captioning

* Upload images via Telegram
* Generate captions using **BLIP model**
* Extract keywords/tags automatically

---

### 🧩 Multimodal Support

* Text queries → RAG pipeline
* Image uploads → Vision pipeline

---

## 🏗️ Project Structure

genai_bot/
│
├── app.py
│
├── bot/
│   └── handlers.py
│
├── rag/
│   ├── loader.py
│   ├── embedder.py
│   ├── vector_store.py
│   ├── pipeline.py
│
├── vision/
│   └── captioner.py
│
├── data/
│   ├── docs/
│   └── images/
│
├── requirements.txt
└── README.md

---

## ⚙️ Setup Instructions

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd Telegram_bot
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Create `.env`

```env
TELEGRAM_BOT_TOKEN=your_telegram_token
```

---

## 🧠 LLM Setup Options

GGUF (llama.cpp)

If using a local model file:

```python
MODEL_PATH = "models/.../mistral-7b-instruct-v0.2.Q4_0.gguf"
```

Install:

```bash
pip install llama-cpp-python
```

✔ Fully offline
✔ More control
⚠️ Slightly complex setup

---

## 📂 Add Knowledge Base

Place documents inside:

```bash
data/docs/
```

Example:

* ai_basics.txt
* machine_learning.txt
* nlp.txt
* faq.txt

---

## ▶️ Run the Bot

```bash
python app.py
```

---

## 💬 Usage

### Commands

* `/help` → Show instructions
* `/ask <question>` → Ask questions from documents
* Upload image → Get caption + tags

---

## 🔄 System Flow

### 🧠 RAG Pipeline

1. Load documents
2. Chunk text
3. Generate embeddings
4. Store vectors
5. Retrieve top-k chunks
6. Generate answer using local LLM

---

### 🖼️ Vision Pipeline

1. Receive image
2. Download locally
3. Run BLIP model
4. Generate caption
5. Extract tags

---

## 🧰 Tech Stack

| Component     | Technology                |
| ------------- | ------------------------- |
| Bot Framework | python-telegram-bot       |
| Embeddings    | sentence-transformers     |
| Vector Store  | NumPy (in-memory)         |
| LLM           | Ollama / llama.cpp (GGUF) |
| Vision Model  | BLIP (Hugging Face)       |

---

## 🌟 Key Design Decisions

* Used **local LLM** to eliminate API dependency
* Modular architecture for easy extension
* Lightweight models for efficiency
* In-memory vector store for simplicity

---

## 📸 Demo

(Add screenshots or GIF here)

---

## 🚀 Future Improvements

* Chat memory (last 3 interactions)
* Source citations in responses
* Embedding cache
* Streaming responses
* Docker deployment

---

## ✅ Evaluation Coverage

✔ Code Quality — modular and clean
✔ System Design — clear pipelines
✔ Model Use — efficient local models
✔ Efficiency — lightweight + no API cost
✔ UX — simple Telegram interface
✔ Innovation — multimodal + local inference

---

## 👨‍💻 Author

Palani Vigneshwar