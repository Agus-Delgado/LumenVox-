# LumenVox — Project Status

## Current Version

**V1 complete.** The full runtime pipeline, modeling workflow, business reporting, and portfolio documentation are finished.

## Completed Blocks

| Block | Focus | Status |
|-------|-------|--------|
| **Block 1** | Dataset generation | Complete |
| **Block 2** | Text preprocessing and EDA | Complete |
| **Block 3** | Sentiment modeling | Complete |
| **Block 4** | Product area and topic insights | Complete |
| **Block 5** | Executive reporting | Complete |
| **Block 6** | Repository polish and portfolio packaging | Complete |

## Current Pipeline

The runtime pipeline (`run_pipeline.py`) executes six steps in order:

1. Dataset generation
2. Text preprocessing and feature engineering
3. Exploratory data analysis
4. Sentiment modeling
5. Product area and topic insights
6. Executive reporting

Block 6 is documentation only and is not part of the runtime pipeline.

## Main Outputs

### Datasets

- `data/raw/customer_feedback_raw.csv` — 1,000 records, 13 columns
- `data/processed/customer_feedback_clean.csv` — 1,000 records, 23 columns

### Reports

- `reports/eda_summary.md`
- `reports/model_metrics.md`
- `reports/sentiment_model_comparison.csv`
- `reports/topic_insights.md`
- `reports/product_area_summary.csv`
- `reports/high_risk_feedback.csv`
- `reports/executive_summary.md`
- `reports/lumenvox_project_summary.md`
- `reports/data_dictionary.md`
- `reports/interview_notes.md`

### Visuals

- Block 2 EDA charts (`visuals/sentiment_distribution.png`, `feedback_by_channel.png`, etc.)
- Block 3 confusion matrices (`visuals/confusion_matrix_*.png`)
- Block 4 risk charts (`visuals/negative_feedback_by_segment.png`, `unresolved_critical_by_area.png`, etc.)

### Models

- `models/sentiment_model.pkl` — best classifier (LinearSVC)
- `models/tfidf_vectorizer.pkl` — fitted TF-IDF vectorizer
- `models/model_metadata.json` — selection criteria and vectorizer config

## What Works

- End-to-end pipeline runs with a single command (`python run_pipeline.py`)
- Synthetic dataset generation with realistic correlations across channels, segments, and product areas
- Text preprocessing with bilingual support and 10 derived analytical features
- EDA with programmatic statistics and stakeholder-ready charts
- Three-model sentiment comparison with negative-recall-driven selection
- Product-area risk aggregation and high-risk feedback extraction
- Executive and portfolio reports synthesized from computed pipeline artifacts
- Data dictionary, interview notes, and project status documentation for GitHub and recruiter review

## Known Limitations

- **Synthetic data** — the dataset is generated for portfolio demonstration, not production decision-making
- **Generated labels** — sentiment labels were created programmatically alongside messages, which can inflate model performance
- **Inflated model performance** — the synthetic structure makes text patterns more regular and labels more consistent than real-world feedback
- **No real deployment** — the pipeline runs locally; there is no API, batch scoring service, or production infrastructure
- **No real customer feedback validation** — findings have not been validated against actual support escalations or churn events

## Recommended Next Version

- Replace or supplement with a **real or semi-real dataset** with human-annotated labels
- Conduct **model error analysis** on misclassified sentiment cases
- Build a **product-area classifier** to automate routing beyond metadata
- Add a **GenAI summary assistant with fallback** grounded in validated analytical outputs
- Build a **Streamlit dashboard** for interactive exploration by stakeholders
- Finalize **GitHub polish and LinkedIn post** for portfolio visibility
