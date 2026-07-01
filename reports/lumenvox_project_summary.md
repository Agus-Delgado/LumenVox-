# LumenVox — Project Summary for Portfolio

## Project Objective

LumenVox is a business-oriented Data Science and AI portfolio project that builds an end-to-end pipeline to ingest, analyze, and summarize customer feedback for a fictional SaaS company. The goal is to demonstrate how unstructured text can be transformed into actionable insights for product, support, and operations teams.

## Business Context

The project simulates a feedback intelligence platform for a SaaS product serving freelancers, small businesses, startups, enterprises, and agencies across multiple subscription tiers. Customer messages arrive through support tickets, reviews, NPS surveys, chat, email, and contact forms — covering billing, performance, usability, integrations, and other product areas. Leadership needs to prioritize fixes, reduce churn risk, and improve operational response times.

## Technical Pipeline

The LumenVox pipeline consists of six completed steps:

1. **Synthetic dataset generation** — realistic bilingual feedback with business correlations
2. **Text preprocessing and feature engineering** — cleaning, normalization, derived flags
3. **Exploratory data analysis** — distributions, charts, and business-oriented EDA report
4. **Sentiment modeling** — TF-IDF features and classical classifiers with model selection
5. **Product area and topic insights** — risk summaries, high-risk extraction, recommendations
6. **Executive reporting** — consolidated executive and portfolio summaries

## Dataset Design

The project uses **1,000 synthetic feedback records** spanning 2025 with bilingual messages (English and Spanish). Each record includes channel, customer segment, plan type, product area, rating, sentiment label, priority, resolution status, and response time.

Key design elements:

- **10 product areas:** billing, performance, usability, support, pricing, features, onboarding, exports, integrations, account access
- **6 channels:** support_ticket, review, nps_survey, chat, email, contact_form
- **5 customer segments:** freelancer, small_business, startup, enterprise, agency
- **4 plan types:** free, basic, pro, enterprise
- **Business correlation rules:** negative feedback concentrated in email and support channels; critical priority skewed toward billing and account access; response times escalate with priority level

## NLP and Machine Learning Approach

Text is normalized into a `clean_message` column (lowercase, URL removal, punctuation stripping with multilingual character preservation). Features are extracted with **TF-IDF** using unigrams and bigrams (`ngram_range=(1, 2)`), up to 5,000 features.

Three supervised classifiers are compared:

- **Logistic Regression** with balanced class weights
- **LinearSVC** with balanced class weights
- **Multinomial Naive Bayes** as a probabilistic baseline

Model selection prioritizes **negative recall** with **macro F1** as tie-breaker. The best model (**LinearSVC**) achieves 100.0% negative recall.

## Business Intelligence Layer

Beyond classification, the pipeline produces operational intelligence artifacts:

- **`product_area_summary.csv`** — per-area volume, negative counts, unresolved critical backlog, response times, and resolution rates
- **`high_risk_feedback.csv`** — escalation-ready list of negative, high/critical, unresolved feedback sorted by priority and response time
- **`topic_insights.md`** — business-oriented risk analysis with segment and channel breakdowns
- **`executive_summary.md`** — consolidated leadership brief with KPIs and recommended actions

## Key Results

| Metric | Value |
|--------|-------|
| Total feedback | 1,000 |
| Negative feedback rate | 40.4% |
| High-risk feedback count | 135 |
| Unresolved critical feedback | 71 |
| Top risk product area | billing (21 unresolved critical) |
| Best model | LinearSVC (negative recall: 100.0%) |

## Tools Used

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- joblib
- Markdown reporting

## Portfolio Value

This project demonstrates end-to-end Data Science and AI capabilities:

- **Unstructured text analysis** — ingesting and normalizing multilingual feedback
- **NLP preprocessing** — cleaning pipelines with derived feature engineering
- **Supervised text classification** — TF-IDF vectorization and classical ML models
- **Model evaluation** — multi-metric comparison with business-driven selection criteria
- **Business metric design** — negative recall prioritization for operational triage
- **Customer/product analytics** — segment, channel, and product-area risk analysis
- **Executive reporting** — stakeholder-ready summaries grounded in computed KPIs
- **AI/data storytelling** — translating technical outputs into actionable recommendations

## What This Project Demonstrates

- NLP preprocessing and text normalization
- Supervised machine learning for sentiment classification
- Model selection based on business criteria (negative recall)
- Customer feedback intelligence and risk prioritization
- Product-area operational analytics
- Stakeholder-facing reporting for leadership and portfolio review

## Future Improvements

- Validate pipeline on **real customer feedback** datasets
- Experiment with **embedding-based** representations (e.g., sentence transformers)
- Add **topic modeling** (LDA, NMF) for unsupervised theme discovery
- Build a **product-area classifier** to reduce reliance on metadata labels
- Add a **GenAI synthesis layer** for draft summaries grounded in validated outputs
- Deploy an interactive **Streamlit dashboard** for stakeholder exploration

## Generated Artifacts

- `data/raw/customer_feedback_raw.csv`
- `data/processed/customer_feedback_clean.csv`
- `reports/eda_summary.md`
- `reports/model_metrics.md`
- `reports/sentiment_model_comparison.csv`
- `reports/topic_insights.md`
- `reports/product_area_summary.csv`
- `reports/high_risk_feedback.csv`
- `reports/executive_summary.md`
- `reports/lumenvox_project_summary.md`
- `models/sentiment_model.pkl`
- `models/tfidf_vectorizer.pkl`
- `models/model_metadata.json`
- `visuals/*.png`