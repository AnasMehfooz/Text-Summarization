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
├── eval/
│   ├── rouge_scores.json     # ROUGE results on the SAMSum test set
│   └── test_predictions.csv  # Every test dialogue with reference and model summaries
├── app.py                    # Flask server + Waitress WSGI runner
├── evaluate_model.py         # ROUGE evaluation vs. the base t5-small model
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

## 📈 Results

Evaluated on the full SAMSum **test set (819 dialogues)** with the same generation settings the app uses (beam search, 4 beams). Scores are ROUGE F1 (×100) with stemming.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| t5-small, no fine-tuning (`summarize:` prefix) | 28.37 | 7.60 | 21.98 |
| **t5-small fine-tuned on SAMSum (this repo)** | **43.39** | **18.60** | **35.16** |

Fine-tuning on 4,000 training dialogues (6 epochs) improves ROUGE-1 by **+15.0** and more than doubles ROUGE-2 over the base model.

**Limitations:** the model is small and was trained on about a quarter of the training set. It sometimes attributes an action to the wrong speaker, and it tends to copy sentences when the input is not a chat. Every prediction is saved in [`eval/test_predictions.csv`](eval/test_predictions.csv) for error analysis.

To reproduce:
```bash
python evaluate_model.py
```

---

## 📊 Dataset

Trained on the **SAMSum Corpus**, containing ~16k realistic chat dialogues with human-annotated summaries.

---

## 👤 Author
**Anas Mehfooz**
- GitHub: [@AnasMehfooz](https://github.com/AnasMehfooz)
