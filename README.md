# 🧠 GenAI Training Lab

A collection of four beginner-friendly Generative AI projects built during a hands-on training session. Each project is a self-contained Jupyter/Colab notebook that loads a pre-trained model, wraps it in a simple Python function, and (in most cases) serves it through a **Gradio** web interface.

Together they cover four core AI application types: **vision-language OCR**, **machine translation**, **conversational chatbots**, and **Retrieval-Augmented Generation (RAG)**.

---

## 📦 Projects at a Glance

| # | Project | What it does | Model / Stack | Notebook |
|---|---------|--------------|---------------|----------|
| 1 | **LensScribe** | Reads text from images (receipts, signs, tables, formulas) | `zai-org/GLM-OCR` | `LensScribe_ImageCaptioning_GLM_OCR.ipynb` |
| 2 | **LinguaBridge AI** | Translation web app with 16 selectable languages | `google-t5/t5-small` | `LinguaBridge_AI__1_.ipynb` |
| 3 | **Qwen Chatbot** | Chatbot that remembers the whole conversation | `Qwen/Qwen3-0.6B` | `Qwen.ipynb` |
| 4 | **PDF Intelligence (RAG)** | Upload a PDF and ask questions about it | Groq LLM + LangChain + ChromaDB | `RAG_PDF_Reader.ipynb` |

---

## 1. 🔍 LensScribe — Image Captioning with GLM-OCR

An AI that reads and describes the text it sees in an image.

- **Model:** [`zai-org/GLM-OCR`](https://huggingface.co/zai-org/GLM-OCR), a ~0.9B-parameter vision-language model for document reading.
- **How it works:** An image and a task prompt go in; the model generates the text it reads. Here, the *caption* is the text recognised in the image, so this is not a scene captioner like BLIP.
- **Supported task prompts:** `Text Recognition:`, `Formula Recognition:`, `Table Recognition:`
- **What the notebook covers:**
  - Loading the model with `transformers` (installed from GitHub, since GLM-OCR needs a very recent version)
  - Downloading a sample image, with an auto-generated receipt image as a fallback
  - A reusable `get_caption()` function and a prompt-guided `get_caption_with_prompt()` variant
  - Experimenting with how `max_new_tokens` affects output length
  - Captioning multiple images in a loop and displaying them with `matplotlib`
- **Tip:** Use a GPU runtime (Colab → T4) for much faster inference.

## 2. 🌐 LinguaBridge AI — Translation Web App

A Gradio web app for exploring language translation with a Hugging Face model.

- **Model:** [`google-t5/t5-small`](https://huggingface.co/google-t5/t5-small)
- **Features:**
  - 16 languages: English, French, German, Spanish, Italian, Portuguese, Dutch, Romanian, Russian, Arabic, Hindi, Bengali, Tamil, Telugu, Japanese, Chinese
  - Source and target language dropdowns
  - Adjustable maximum output tokens (32–256)
  - Input validation (empty text, same source/target language) and graceful error handling
  - Beam search decoding (`num_beams=4`) for better output
  - Custom-styled Gradio UI with Translate and Clear buttons
- **⚠️ Important limitation:** `t5-small` is a general-purpose base checkpoint, **not** a translation-fine-tuned multilingual model. It was trained on a limited set of language pairs (mainly English → French/German/Romanian), so results for other pairs may be inaccurate or stay in the source language. Treat this as a demonstration, not for high-stakes translation. For reliable multilingual output, swap in a translation-fine-tuned model (such as an NLLB or Marian/Opus-MT model) and follow its prompting format.

## 3. 💬 Qwen Chatbot — AI with Memory

A conversational chatbot that remembers everything said earlier in the chat.

- **Model:** [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B), small enough to run on a laptop CPU (a GPU just makes replies faster).
- **Key idea:** Memory is simply a list of `{role, content}` messages. The full history is sent to the model on every turn using Qwen's built-in chat template.
- **What the notebook covers:**
  - Auto-detecting CPU or GPU and loading the model with the right precision
  - A `chat(user_message, history)` function using `apply_chat_template` with `enable_thinking=False` for direct answers
  - Sampling controls: `temperature=0.7`, `top_p=0.8`, `top_k=20`, `max_new_tokens=150`
  - Demos of memory ("My favourite sport is cricket" → "What sport did I just mention?") and multi-turn conversations (e.g. Smart Cities)
  - A Gradio chat app with Send and Clear buttons

## 4. 📄 PDF Intelligence — RAG PDF Reader

Upload a PDF, ask questions in plain language, and get answers grounded in the document.

- **Pipeline:**
  1. **Load**: `PyPDFLoader` reads the PDF page by page
  2. **Split**: `RecursiveCharacterTextSplitter` (chunk size 500, overlap 100)
  3. **Embed**: `sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface`
  4. **Store**: chunks go into a **ChromaDB** vector store
  5. **Retrieve**: top-4 most relevant chunks per question
  6. **Generate**: Groq-hosted `openai/gpt-oss-20b` (temperature 0) answers using only the retrieved context
- **Anti-hallucination prompt:** If the answer isn't in the PDF, the bot replies *"I couldn't find the answer in the uploaded PDF."* It also cites page numbers when useful.
- **Gradio app ("PDF Intelligence"):** Upload panel, a *Process PDF* button with a status summary (file name, pages, chunks), a chat panel, example questions, and a unique Chroma collection per uploaded PDF.
- **Requires:** A free [Groq API key](https://console.groq.com/), stored as `GROQ_API_KEY` in Colab Secrets. The notebook reads it via `google.colab.userdata`.
- **Limitation:** Works with text-based PDFs only; scanned PDFs with no text layer are not supported.

---

## 🛠️ Tech Stack

| Area | Tools |
|------|-------|
| Language | Python 3 |
| Models & NLP | Hugging Face `transformers`, `sentencepiece`, `sentence-transformers` |
| Deep learning | PyTorch, `accelerate` |
| RAG | LangChain, ChromaDB, Groq API, `pypdf` |
| UI | Gradio |
| Utilities | Pillow, Matplotlib, pandas, requests |
| Environment | Google Colab / Jupyter Notebook |

---

## 🚀 Getting Started

### Option A: Google Colab (recommended)

1. Open any notebook in [Google Colab](https://colab.research.google.com/) (**File → Upload notebook**, or open it from GitHub).
2. For **LensScribe**, switch to a GPU: **Runtime → Change runtime type → T4 GPU**.
3. For **PDF Intelligence**, add your `GROQ_API_KEY` under **🔑 Secrets** in the left sidebar and enable notebook access.
4. Run the cells from top to bottom. The first cell of each notebook installs its own dependencies.

### Option B: Run locally

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-folder>

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install Jupyter and launch it
pip install jupyter
jupyter notebook
```

Then open a notebook and run its install cell. Notes for local use:

- The **RAG notebook** imports `google.colab` for secrets and file upload. Outside Colab, replace those cells with `GROQ_API_KEY = os.environ["GROQ_API_KEY"]` and a local file path.
- **LinguaBridge** and **Qwen** run fine on CPU. **LensScribe** is best on a GPU.
- Model weights download from Hugging Face on first run, so an internet connection is required.

---

## 📁 Repository Structure

```
.
├── LensScribe_ImageCaptioning_GLM_OCR.ipynb   # Project 1: OCR / image-to-text
├── LinguaBridge_AI__1_.ipynb                  # Project 2: translation web app
├── Qwen.ipynb                                 # Project 3: chatbot with memory
├── RAG_PDF_Reader.ipynb                       # Project 4: chat with your PDF
└── README.md
```

---

## 🎓 What I Learned

- Loading and running pre-trained models from Hugging Face
- Wrapping a model in a clean, reusable Python function
- Prompting different model types (vision-language, seq2seq, causal LM)
- Giving a chatbot **memory** by passing conversation history
- Building a full **RAG pipeline**: chunking, embeddings, vector search, and grounded generation
- Turning notebooks into shareable **Gradio** web apps

---

## ⚠️ Notes & Limitations

- These are learning projects, not production systems.
- Small models (`t5-small`, `Qwen3-0.6B`) can produce inaccurate or low-quality output.
- Never commit API keys. Keep secrets in Colab Secrets or environment variables.

---

## 🙏 Acknowledgements

- [Hugging Face](https://huggingface.co/) for `transformers` and model hosting
- [Z.ai (GLM-OCR)](https://huggingface.co/zai-org/GLM-OCR), [Google (T5)](https://huggingface.co/google-t5/t5-small), and [Qwen](https://huggingface.co/Qwen) for the open models
- [Groq](https://groq.com/), [LangChain](https://www.langchain.com/), and [ChromaDB](https://www.trychroma.com/) for the RAG stack
- [Gradio](https://www.gradio.app/) for the web interfaces
- The training session that made these projects possible

---

## 📄 License

Released under the [MIT License](LICENSE). Add a `LICENSE` file if you'd like others to reuse the code.
