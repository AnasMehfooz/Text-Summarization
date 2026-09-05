# 📝 Text Summarization with Fine-Tuned T5 & Flask

An end-to-end Natural Language Processing (NLP) dialogue summarization system powered by a fine-tuned Google **T5 (Text-to-Text Transfer Transformer)** model, equipped with a modern web user interface and a production-ready REST API.

---

## 🌟 Features

- **Fine-Tuned Transformer**: Trained on the **SAMSum** dialogue dataset for coherent, abstractive summarization.
- **Interactive Web Interface**: Clean HTML/CSS/JavaScript UI for real-time text input and instant summary generation.
- **RESTful API**: Fast inference endpoint (`/summarize`) deployable anywhere with Waitress WSGI.
- **VS Code Ready**: Pre-configured with `.vscode` debug launch configurations, recommended extensions, and environment settings.
- **GPU Accelerated**: Automatic GPU acceleration when CUDA is available, with CPU fallback.

---

## 📁 Project Structure

```
Text-Summarization/
├── .vscode/
│   ├── launch.json           # VS Code one-click debugging configurations
│   ├── settings.json         # Workspace settings & exclusions
│   └── extensions.json       # Recommended extensions (Python, Pylance, Jupyter)
├── saved_summary_model/
│   ├── config.json           # Model configuration
│   ├── generation_config.json
│   ├── tokenizer.json        # Tokenizer vocabulary
│   └── tokenizer_config.json
├── static/
│   ├── style.css             # Frontend styling
│   └── script.js             # Async API client for UI
├── templates/
│   └── index.html            # Web dashboard
├── app.py                    # Flask server + Waitress WSGI runner
├── train.py                  # Fine-tuning script with HuggingFace Trainer
├── he.py                     # Training script (original)
├── text_summarizer.ipynb     # Interactive Jupyter notebook walkthrough
├── requirements.txt          # Project dependencies
├── samsum-train.csv          # Training dataset
├── samsum-validation.csv     # Validation dataset
├── samsum-test.csv           # Test dataset
└── README.md
```

---

## 🚀 Getting Started in VS Code

### 1. Clone the Repository
```bash
git clone https://github.com/AnasMehfooz/Text-Summarization.git
cd Text-Summarization
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Running the Web Application
- **In VS Code**: Press `F5` or navigate to the **Run & Debug** panel and select **`Python: Flask Web App`**.
- **Via Terminal**:
  ```bash
  python app.py
  ```
  Open your browser and navigate to `http://127.0.0.1:5000`.

---

## 🏋️ Model Training

To retrain or fine-tune the T5 model on the SAMSum dataset:
- **In VS Code**: Select **`Python: Train T5 Model`** in the Run & Debug menu.
- **Via Terminal**:
  ```bash
  python train.py
  ```
The trained weights and tokenizer will be saved to `./saved_summary_model/`.

---

## 🔌 API Reference

### Health Check
- **Endpoint**: `GET /health`
- **Response**:
  ```json
  {
    "status": "healthy",
    "device": "cuda"
  }
  ```

### Summarize Text
- **Endpoint**: `POST /summarize`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "text": "Hannah: Hey, are you free this afternoon?\nAmanda: Yes, why?\nHannah: Let's meet at the cafe around 4pm to work on our presentation.\nAmanda: Sounds great! See you then."
  }
  ```
- **Response**:
  ```json
  {
    "summary": "hannah and amanda will meet at a cafe at 4pm to prepare their presentation."
  }
  ```

---

## 📊 Dataset

Trained on the **SAMSum Corpus**, containing ~16k realistic chat dialogues with human-annotated summaries.

---

## 👤 Author
**Anas Mehfooz**
- GitHub: [@AnasMehfooz](https://github.com/AnasMehfooz)
