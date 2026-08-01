# NHPT Heritage AI Prototype

Coursework 01 — Machine Learning and Related Applications (NB627BSDS), NIBM HND Data Science.

Data set - https://www.kaggle.com/datasets/wwymak/architecture-dataset

Prototype AI system for the (fictional) National Heritage Preservation Trust (NHPT), combining
computer vision, LangChain, and a locally-hosted LLM (Gemma 3 12B via Ollama) to support:
1. Automated architectural style classification from visitor/inspector photos
2. A RAG-grounded conversational assistant answering visitor questions about heritage sites
3. An interactive Streamlit app combining both into one upload-a-photo-and-chat experience

## Repository Structure

```
.
├── notebooks/
│   ├── heritage_style_classification.ipynb   # Part B — CV model (EfficientNetB0 + Grad-CAM)
│   ├── heritage_langchain_assistant.ipynb    # Part C — LangChain RAG assistant
│   └── heritage_full_pipeline.ipynb          # Combined CV + RAG demo (cell-by-cell) + Streamlit launcher
├── app/
│   └── app.py                                # Streamlit app: upload photo -> style + Grad-CAM + chat
├── models/                                    
│   ├── best_head_model.keras                 # Part B, Stage 1 checkpoint
│   └── efficientnetb0_heritage_style_final.keras  # Part B, final fine-tuned model (85.96% accuracy)
├── diagrams/
│   └── nhpt_architecture.svg                 # Part A1 — system architecture diagram
├── report/
│   ├── NHPT_Final_Report.docx         
├── presentation/
│   └── NHPT_Coursework_Presentation.pptx      # 8-slide summary deck
└── README.md
```

## ⚠️ Before you push: add your model files

Both `.keras` files are produced by `notebooks/heritage_style_classification.ipynb` when you run it
(`best_head_model.keras` mid-training, `efficientnetb0_heritage_style_final.keras` at the end). They
are **not included in this package** — copy both from wherever you ran that notebook into `models/`
before committing.

## Knowledge Base — Real Data

The assistant's knowledge base is **not** a folder of hand-written documents — it's fetched **live
from Wikipedia** at run time, one real UK heritage building per architectural style class:

| Internal slug | Real building | Style class |
|---|---|---|
| `ashcombe_abbey` | Westminster Abbey | Gothic architecture |
| `falmoor_house` | Kenwood House | Georgian architecture |
| `thornfield_palace` | Blenheim Palace | Baroque architecture |
| `st_edwins_priory` | Durham Cathedral | Romanesque architecture |
| `marlow_court` | Uppark | Queen Anne architecture |
| `chiswick_hall` | Chiswick House | Palladian architecture |

The internal slugs keep the project's original naming/citation style; each maps to a real,
independently verifiable Wikipedia article. First run fetches live and builds a local Chroma index
(`chroma_heritage_db_v2/`, gitignored); subsequent runs reuse that index without re-fetching.

**Attribution:** Wikipedia content is CC BY-SA — any public-facing use of this prototype should
display attribution to end users.

## Key Results

| Metric | Value |
|---|---|
| CV Top-1 test accuracy | 85.96% (6 architectural styles) |
| Weakest class | Palladian architecture (F1 0.60) — confused with Georgian/Baroque |
| LLM | Gemma 3 12B via Ollama (local, self-hosted — the only model used throughout) |
| Vector store | Chroma, 6 live-fetched documents, 501 chunks, cosine distance |

## Setup

```
pip install tensorflow scikit-learn matplotlib opencv-python-headless
pip install langchain langchain-community langchain-chroma langchain-huggingface langchain-ollama
pip install sentence-transformers beautifulsoup4 streamlit pillow
ollama pull gemma3:12b
```

`notebooks/heritage_style_classification.ipynb` additionally requires the `arcDataset` (25-class
Architectural Styles dataset: https://www.kaggle.com/datasets/wwymak/architecture-dataset) placed
as a sibling folder.

## Running the Interactive App

```
cd app
streamlit run app.py
```

Requires `ollama serve` running in the background and `efficientnetb0_heritage_style_final.keras`
present (copy it into `app/` or adjust `MODEL_PATH` in `app.py`).

## Known Limitations

- The RAG retriever does not fully resolve conversational pronouns ("it", "that") across turns —
  a history-aware retriever (`create_history_aware_retriever`) is the recommended fix, documented in
  the technical report's Testing & Validation and Future Improvements sections.
- CV model shows class imbalance sensitivity (Queen Anne: 425 training images vs. Romanesque: 107).
- If `app.py` fails to load the model with a `BatchNormalization`/`Functional` deserialization error,
  it is almost certainly an environment mismatch, not a code issue — make sure you run
  `streamlit run app.py` from the same activated venv (`nibm_nic_env`) used for the training
  notebooks, not a different Python/Anaconda environment. No code change should be needed.

## Author

Kalingu — HND Data Science, NIBM (Student ID: COHNDDS24.2F-011)
