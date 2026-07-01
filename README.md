# LumenVox — Customer Feedback Intelligence with NLP & Generative AI

**Turning unstructured customer feedback into actionable business insights for SaaS teams.**

**Status: V1 complete** — full runtime pipeline, modeling, business reporting, and portfolio documentation are finished. See [PROJECT_STATUS.md](PROJECT_STATUS.md) for details.

---

## Key Results

| Metric | Value |
|--------|-------|
| Total feedback records | 1,000 |
| Negative feedback rate | 40.4% |
| Best sentiment model | LinearSVC |
| Negative recall | 1.0000 |
| High-risk unresolved items | 135 |
| Unresolved critical items | 71 |
| Top risk product area | billing |

## Repository Outputs

| Report | Description |
|--------|-------------|
| [reports/executive_summary.md](reports/executive_summary.md) | Leadership brief with KPI snapshot and recommended actions |
| [reports/lumenvox_project_summary.md](reports/lumenvox_project_summary.md) | Portfolio-oriented technical summary |
| [reports/model_metrics.md](reports/model_metrics.md) | Sentiment modeling evaluation report |
| [reports/topic_insights.md](reports/topic_insights.md) | Product area and operational risk insights |
| [reports/data_dictionary.md](reports/data_dictionary.md) | Column definitions for raw and processed datasets |
| [reports/interview_notes.md](reports/interview_notes.md) | Interview-ready project explanation and Q&A |

## Business Problem

Modern SaaS companies collect thousands of open-text feedback items every month — support tickets, NPS surveys, product reviews, chat transcripts, and emails. Leadership teams know the data exists, but struggle to answer basic questions:

- What are customers complaining about most?
- Which product areas drive churn risk?
- How does sentiment differ across customer segments and plan types?
- Where should engineering and product prioritize fixes?

Raw feedback is unstructured, multilingual, and scattered across channels. Without a structured analytics approach, valuable signals get lost in noise.

## Proposed Solution

**LumenVox** is a business-oriented Data Science and AI project that builds an end-to-end pipeline to ingest, analyze, and summarize customer feedback for a fictional SaaS company. The project applies NLP, machine learning, and exploratory analytics to transform open-text messages into executive-ready insights.

The workflow is designed as a real portfolio-grade DS project — not a single sentiment notebook — with clear stages from data generation through modeling, reporting, and (optionally) AI-assisted synthesis.

## Pipeline Overview

Running `python run_pipeline.py` executes six runtime steps:

1. **Dataset generation** — synthetic customer feedback with correlated metadata
2. **Text preprocessing and feature engineering** — cleaning, normalization, derived flags
3. **Exploratory data analysis** — distributions, charts, EDA report
4. **Sentiment modeling** — TF-IDF vectorization, three-model comparison, artifact persistence
5. **Product area and topic insights** — risk aggregation and escalation queue
6. **Executive reporting** — consolidated leadership and portfolio summaries

## Main Capabilities

| Capability | Description |
|------------|-------------|
| **Sentiment analysis** | Classify feedback as positive, neutral, or negative to track customer satisfaction trends |
| **Topic & product-area analysis** | Identify recurring themes across billing, performance, usability, integrations, and more |
| **Segment & plan breakdowns** | Compare feedback patterns across customer segments and subscription tiers |
| **Priority & resolution tracking** | Surface high-impact issues and operational response metrics |
| **Executive insights** | Aggregate findings into business-oriented summaries for product and leadership teams |

> **Note on Generative AI:** A generative AI layer will be added later as an **assistant for synthesis and reporting** — helping draft executive summaries from validated analytical outputs. It will **not** replace rigorous NLP and machine learning analysis. Models and metrics remain the source of truth.

## Dataset

The project uses a synthetic but realistic customer feedback dataset (`data/raw/customer_feedback_raw.csv`) with **1,000 records** spanning:

- **6 channels:** support tickets, reviews, NPS surveys, chat, email, contact forms
- **5 customer segments:** freelancer, small business, startup, enterprise, agency
- **4 plan types:** free, basic, pro, enterprise
- **10 product areas:** billing, performance, usability, support, pricing, features, onboarding, exports, integrations, account access
- **Bilingual messages:** English (~60%) and Spanish (~40%)

Each record includes metadata such as rating, sentiment label, priority, resolution status, and response time — with realistic correlations between fields (e.g., negative feedback concentrated in support channels, critical priority skewed toward billing and account access).

## Project Structure

```
lumenvox/
├── data/
│   ├── raw/              # Raw feedback CSV (generated)
│   │   └── customer_feedback_raw.csv
│   ├── processed/        # Cleaned and feature-engineered data
│   │   └── customer_feedback_clean.csv   # Block 2 output
│   └── sample/           # Sample subsets for demos
├── notebooks/            # Exploratory analysis and experiments
├── src/                  # Reusable Python modules
│   ├── __init__.py
│   ├── generate_dataset.py
│   ├── preprocess_text.py    # Block 2 — cleaning & derived features
│   ├── eda_analysis.py       # Block 2 — EDA charts & summary report
│   ├── train_sentiment_model.py  # Block 3 — sentiment classifiers
│   ├── topic_insights.py     # Block 4 — product area & topic insights
│   └── generate_executive_report.py  # Block 5 — executive reporting
├── reports/              # Generated business reports
│   ├── eda_summary.md            # Block 2 output
│   ├── model_metrics.md          # Block 3 output
│   ├── sentiment_model_comparison.csv  # Block 3 output
│   ├── topic_insights.md         # Block 4 output
│   ├── product_area_summary.csv  # Block 4 output
│   ├── high_risk_feedback.csv    # Block 4 output
│   ├── executive_summary.md      # Block 5 output
│   ├── lumenvox_project_summary.md  # Block 5 output
│   ├── data_dictionary.md        # Block 6 output
│   └── interview_notes.md        # Block 6 output
├── visuals/              # Charts and figures for stakeholders
│   ├── *.png                     # Block 2 EDA charts
│   ├── confusion_matrix_*.png    # Block 3 confusion matrices
│   ├── negative_feedback_by_segment.png      # Block 4 output
│   ├── unresolved_critical_by_area.png         # Block 4 output
│   ├── negative_rate_by_channel.png            # Block 4 output
│   └── avg_response_time_by_area.png           # Block 4 output
├── models/               # Trained ML/NLP models
│   ├── sentiment_model.pkl       # Block 3 best model
│   ├── tfidf_vectorizer.pkl      # Block 3 fitted vectorizer
│   └── model_metadata.json       # Block 3 selection metadata
├── PROJECT_STATUS.md     # Block 6 — project status and roadmap
├── README.md
├── requirements.txt
└── run_pipeline.py       # Pipeline entry point
```

## MVP Roadmap

| Phase | Focus | Deliverables |
|-------|-------|--------------|
| **1 — Foundation** | Data generation & EDA | Synthetic dataset, exploratory notebooks, data quality checks |
| **2 — NLP Baselines** | Text preprocessing & labeling | Cleaning pipeline, TF-IDF features, baseline classifiers |
| **3 — Modeling** | Sentiment & topic models | Trained classifiers, evaluation metrics, model persistence |
| **4 — Business Reporting** | Insights & visualization | Segment breakdowns, trend charts, stakeholder reports |
| **5 — GenAI Layer (optional)** | Assisted synthesis | Executive summary drafts grounded in analytical outputs |

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone or navigate to the project directory
cd lumenvox

# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

Generate the synthetic customer feedback dataset:

```bash
python run_pipeline.py
```

Running `python run_pipeline.py` executes the full workflow end to end: **dataset generation**, **text preprocessing**, **exploratory data analysis**, **sentiment modeling**, **product area and topic insights**, and **executive reporting**. Step 1 creates `data/raw/customer_feedback_raw.csv` with 1,000 records and prints a summary of distributions.

## Generated Outputs

After a successful pipeline run, the repository contains:

| Category | Artifacts |
|----------|-----------|
| **Datasets** | `data/raw/customer_feedback_raw.csv`, `data/processed/customer_feedback_clean.csv` |
| **Reports** | `reports/eda_summary.md`, `model_metrics.md`, `topic_insights.md`, `executive_summary.md`, `lumenvox_project_summary.md`, and CSV summaries |
| **Visuals** | EDA charts, confusion matrices, and risk analysis plots in `visuals/` |
| **Models** | `models/sentiment_model.pkl`, `tfidf_vectorizer.pkl`, `model_metadata.json` |

## Model Selection Criterion

Sentiment models are compared on accuracy, macro/weighted F1, and per-class metrics. The **best model** is selected by **highest negative recall** — prioritizing detection of negative feedback because missing dissatisfied customers is more costly than false alarms. **Macro F1-score** serves as the tie-breaker when models achieve comparable negative recall. LinearSVC was selected with 1.0000 negative recall on the current dataset.

## Limitations

- The dataset is **synthetic** and designed for portfolio demonstration, not production decision-making.
- Sentiment labels were **generated programmatically** alongside messages, which can inflate model performance compared to noisy real-world labels.
- Model metrics may be **higher than real-world conditions** where language is messier, sarcasm is common, and labels are inconsistent.
- There is **no production deployment** or validation against real customer feedback yet.

## Future Improvements

- Replace or supplement with a **real or semi-real dataset** with human-annotated labels
- Conduct **model error analysis** on misclassified sentiment cases
- Build a **product-area classifier** to automate routing beyond metadata
- Add a **GenAI summary assistant with fallback** grounded in validated analytical outputs
- Build a **Streamlit dashboard** for interactive stakeholder exploration
- Publish to **GitHub** and share a **LinkedIn post** highlighting portfolio outcomes

## Portfolio Value

LumenVox demonstrates end-to-end Data Science and NLP skills in a business context: structured pipeline design, text preprocessing, classical ML model comparison with business-driven selection criteria, exploratory and operational analytics, executive reporting, and portfolio-ready documentation. It shows the ability to translate unstructured customer feedback into actionable insights — a common real-world problem for SaaS and product teams — while clearly communicating limitations and next steps.

## Block 2 — Text Preprocessing & Exploratory Analysis

Block 2 turns the raw feedback into an analysis-ready dataset and produces a first round of business-oriented insights.

### Text preprocessing

`src/preprocess_text.py` loads the raw CSV, validates the expected schema, and cleans each message into a `clean_message` column. The normalization rules are:

- Lowercase the text.
- Remove URLs (`https://`, `http://`, and `www.` patterns).
- Replace non-letter/non-digit characters with spaces while **preserving Spanish and English characters** (including accented letters and `ñ`).
- Collapse repeated whitespace and strip leading/trailing spaces.
- Non-string or missing values become an empty string.

### Derived feature flags

The preprocessing step engineers 10 analytical columns on top of the original 13:

| Column | Description |
|--------|-------------|
| `clean_message` | Normalized message text |
| `message_length` | Character length of `clean_message` |
| `word_count` | Number of words (0 for empty messages) |
| `month` | Month extracted from `date` (1–12) |
| `is_negative` | `sentiment_label == "negative"` |
| `is_positive` | `sentiment_label == "positive"` |
| `is_critical` | `priority == "critical"` |
| `is_high_priority` | `priority == "high"` |
| `is_unresolved` | `resolved == False` |
| `is_unresolved_critical` | Critical **and** unresolved |

The cleaned dataset is saved to `data/processed/customer_feedback_clean.csv` (1,000 rows, 23 columns).

### Exploratory analysis & generated visuals

`src/eda_analysis.py` uses **matplotlib** to generate six charts in `visuals/`:

- `sentiment_distribution.png` — feedback counts by sentiment
- `feedback_by_channel.png` — feedback volume per channel
- `feedback_by_area.png` — feedback volume per product area
- `negative_feedback_by_area.png` — negative feedback by product area
- `avg_response_time_by_priority.png` — mean response time by priority
- `resolution_rate_by_priority.png` — resolution rate (%) by priority

### EDA summary report

The same module writes a business-oriented Markdown report to `reports/eda_summary.md`. All statistics are computed programmatically and cover the dataset overview, key distributions, main negative feedback areas, channels with the highest negative concentration, response time and resolution rate by priority, initial business insights, and suggested next analytical steps.

## Block 3 — Sentiment Modeling

`src/train_sentiment_model.py` trains and evaluates classical NLP classifiers on the cleaned feedback text to predict `sentiment_label` from `clean_message`.

### TF-IDF vectorization

Text is converted to TF-IDF features with:

- Unigrams and bigrams (`ngram_range=(1, 2)`)
- Up to 5,000 features (`max_features=5000`)
- `min_df=2` and `max_df=0.95`

The vectorizer is fitted on the training split only.

### Models compared

Three supervised classifiers are trained and evaluated on a stratified 80/20 train/test split:

| Model | Notes |
|-------|-------|
| **Logistic Regression** | `class_weight="balanced"`, `max_iter=1000` |
| **LinearSVC** | `class_weight="balanced"` |
| **Multinomial Naive Bayes** | Baseline probabilistic classifier |

### Model selection

Models are compared on accuracy, macro/weighted F1, and per-class metrics. The **best model** is selected by **highest negative recall** — prioritizing detection of negative feedback — with **macro F1-score** as the tie-breaker.

### Saved artifacts

| Artifact | Description |
|----------|-------------|
| `models/sentiment_model.pkl` | Best-performing classifier |
| `models/tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer |
| `models/model_metadata.json` | Selection criteria and vectorizer config |
| `reports/sentiment_model_comparison.csv` | Metrics for all three models |
| `reports/model_metrics.md` | Business-oriented modeling report |
| `visuals/confusion_matrix_*.png` | Confusion matrix per model |

## Block 4 — Product Area & Topic Insights

`src/topic_insights.py` analyzes customer feedback by product area, channel, customer segment, plan type, priority, and resolution status to surface business-critical issues and actionable recommendations.

### Product area risk summary

Aggregates feedback into a per-`product_area` summary with total volume, negative and critical counts, unresolved backlog, average rating, response time, and resolution rate. Output is saved to `reports/product_area_summary.csv`, sorted by unresolved critical feedback, negative volume, and response time.

### High-risk feedback extraction

Filters negative feedback with **high** or **critical** priority that remains **unresolved**, producing an escalation-ready list in `reports/high_risk_feedback.csv` sorted by priority (critical first) and response time.

### Negative feedback by segment

Ranks customer segments by negative feedback volume and share, with a bar chart at `visuals/negative_feedback_by_segment.png`.

### Channel risk

Computes negative rate per channel to identify where dissatisfaction is concentrated, visualized in `visuals/negative_rate_by_channel.png`.

### Operational risk analysis

The markdown report (`reports/topic_insights.md`) discusses unresolved critical backlog, slow response times in high-risk areas, customer trust risks, and escalation needs — all grounded in computed statistics from the cleaned dataset.

### Generated business recommendations

Five action-oriented recommendations for product, support, and operations teams are included in the report, covering escalation rules, channel workflow review, weekly tracking, and qualitative review of high-risk samples.

### Saved artifacts

| Artifact | Description |
|----------|-------------|
| `reports/product_area_summary.csv` | Per-area risk and operational metrics |
| `reports/high_risk_feedback.csv` | Negative, high/critical, unresolved feedback |
| `reports/topic_insights.md` | Business-oriented insights report |
| `visuals/negative_feedback_by_segment.png` | Negative count by customer segment |
| `visuals/unresolved_critical_by_area.png` | Unresolved critical count by product area |
| `visuals/negative_rate_by_channel.png` | Negative rate (%) by channel |
| `visuals/avg_response_time_by_area.png` | Mean response time by product area |

## Block 5 — Executive Reporting

`src/generate_executive_report.py` is a read-only consolidation layer that loads existing pipeline artifacts (CSVs, JSON, and Markdown reports from Blocks 2–4), computes executive KPIs programmatically, and writes two stakeholder-facing summaries.

### Consolidated reports

| Report | Audience | Purpose |
|--------|----------|---------|
| `reports/executive_summary.md` | Leadership & operations | Business-oriented brief with KPI snapshot, risk insights, and recommended actions |
| `reports/lumenvox_project_summary.md` | Portfolio & recruiters | Technical pipeline overview, key results, and demonstrated skills |

### How artifacts are synthesized

The module loads:

- `data/processed/customer_feedback_clean.csv` — volume, sentiment, channel, and segment statistics
- `reports/sentiment_model_comparison.csv` and `models/model_metadata.json` — best model and metrics
- `reports/product_area_summary.csv` — risk area rankings and response times
- `reports/high_risk_feedback.csv` — escalation queue counts
- `reports/eda_summary.md`, `model_metrics.md`, `topic_insights.md` — upstream narrative context

All KPIs in the executive reports are computed from these inputs — no hardcoded placeholder values.

### Business recommendations and limitations

The executive summary includes 5–7 action-oriented recommendations grounded in computed findings (escalation priorities, channel audits, segment outreach, sentiment triage). Limitations are explicitly stated: synthetic data, generated labels, and the need for human validation before production use.

### Saved artifacts

| Artifact | Description |
|----------|-------------|
| `reports/executive_summary.md` | Consolidated leadership brief |
| `reports/lumenvox_project_summary.md` | Portfolio-oriented project summary |

---

*LumenVox — A portfolio project demonstrating business-oriented Data Science, NLP, and AI for customer feedback intelligence.*
