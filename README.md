# TabuLab

**Generate. Validate. Trust.**

> PS 65 College Project — Synthetic Tabular Data Generator using CTGAN

TabuLab is a premium, dark-themed Streamlit application that lets you generate realistic synthetic tabular data using [CTGAN (SDV 2.0)](https://docs.sdv.dev/sdv), compare it statistically against your real data, evaluate its machine-learning utility, and assess basic privacy risks — all from a single, polished interface.

---

## Architecture

```
TabuLab/
├── app.py                        ← Premium Streamlit dashboard (single-file)
├── requirements.txt              ← Pinned Python dependencies
├── README.md
├── .gitignore
├── data/                         ← Drop your CSVs here  (git-ignored)
│   └── .gitkeep
├── outputs/                      ← Generated files land here (git-ignored)
│   └── .gitkeep
└── notebooks/
    └── ctgan_project.ipynb       ← Full 9-stage Jupyter notebook (unchanged)
```

---

## Features

| Stage | What it does |
|---|---|
| **Home** | Cinematic hero, product overview, workflow diagram |
| **Dataset** | Upload CSV or load Titanic demo · EDA · Auto-preprocessing · Train/test split |
| **Train** | Configure CTGAN epochs & batch size · Live training status |
| **Generate** | Draw synthetic rows from the trained model · CSV download |
| **Evaluate** | SDV quality score · Real vs synthetic distributions · ML utility (TSTR) · Privacy risk checks |
| **Report** | Pipeline status · Key metrics · Download synthetic CSV + report |

---

## Design System

| Token | Value |
|---|---|
| Background | `#090A0A` |
| Surface | `#171A18` |
| Primary text | `#F5F5F0` |
| Accent | `#C7FF55` |
| Border | `#292D29` |
| Font | Inter / JetBrains Mono |

---

## Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/TabuLab.git
cd TabuLab
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> SDV pulls in PyTorch (~2 GB). First install takes 5–10 minutes.

---

## Running the App

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Running the Notebook

```bash
jupyter notebook notebooks/ctgan_project.ipynb
# or
jupyter lab notebooks/ctgan_project.ipynb
```

Run cells top-to-bottom. Each section has a Markdown explanation.

---

## Evaluation Methodology

### Statistical Quality (SDV Report)
Uses `sdmetrics` — Column Shapes and Column Pair Trends — scored 0–100%.

### ML Utility: TSTR (Train on Synthetic, Test on Real)

```
Real dataset
 ├─ Train split (80%) ──▶ CTGAN trained here
 │                     ├──▶ Synthetic data generated
 │                     │       └──▶ TSTR model trained on synthetic
 │                     │                └──▶ Evaluated on real test set
 │                     └──▶ Baseline model trained on real train
 │                               └──▶ Evaluated on real test set
 └─ Test split (20%) ──▶ Evaluation only — CTGAN never sees this
```

### Privacy Risk (Heuristic)
- **Exact-match check** — synthetic rows byte-identical to any training record.
- **Nearest-neighbour distance** — normalised Euclidean distance to the closest real training row.

> These are heuristics only. They do **not** prove anonymity and do **not** provide formal differential-privacy guarantees.

---

## Deployment: Streamlit Community Cloud

### 1. Push to GitHub

```bash
git init
git add app.py requirements.txt README.md .gitignore notebooks/ data/.gitkeep outputs/.gitkeep
git commit -m "TabuLab — PS 65 CTGAN Synthetic Data Generator"
git remote add origin https://github.com/<your-username>/TabuLab.git
git branch -M main
git push -u origin main
```

### 2. Deploy

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app** → select your repo, branch `main`, file `app.py`.
3. Click **Deploy**.

> **Cloud limits (free tier):** Keep epochs ≤ 100 and row counts ≤ 50,000 to stay within the ~1 GB RAM limit.

---

## Privacy & Responsible Use

- Synthetic data is **not** inherently private.
- Do not commit real datasets or generated outputs to public repositories.
- The `.gitignore` excludes `data/*.csv`, `outputs/`, and trained model files.
- Never share synthetic data derived from sensitive personal records without a formal privacy audit.

---

## Limitations

| Limitation | Detail |
|---|---|
| Training time | 100 epochs on 1 000 rows ≈ 30–120 s |
| Dataset size | Best results with ≥ 500 rows and mixed column types |
| Rare categories | Very infrequent values may not appear in synthetic data |
| No time-series | CTGAN handles i.i.d. tabular data only |
| Cloud RAM | Free Streamlit tier has ~1 GB — keep datasets small |
| Privacy | No formal DP; heuristic checks only |

---

*Built for PS 65 · SDV 2.0.0 · CTGAN 0.12.1 · Streamlit 1.62.0*
