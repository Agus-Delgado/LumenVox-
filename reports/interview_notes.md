# LumenVox — Interview Notes

## 30-Second Pitch

LumenVox is an end-to-end NLP and Data Science portfolio project that turns unstructured SaaS customer feedback into actionable business intelligence. I built a six-step pipeline — from synthetic data generation through text preprocessing, EDA, sentiment modeling, product-area risk analysis, and executive reporting — to help product and support teams identify churn risk, prioritize escalations, and surface operational bottlenecks. The best model (LinearSVC) achieves perfect negative recall on a 1,000-record bilingual dataset, and the pipeline produces stakeholder-ready reports and visuals without manual analysis.

## Business Problem

SaaS companies collect thousands of open-text feedback items every month across support tickets, NPS surveys, reviews, chat, email, and contact forms. Leadership teams need to know what customers complain about, which product areas drive churn risk, and where to prioritize fixes — but raw feedback is unstructured, multilingual, and scattered across channels. Manual review does not scale, introduces inconsistency, and makes it easy to miss high-priority negative cases buried in volume.

## Technical Solution

The pipeline runs six sequential steps via `run_pipeline.py`:

1. **Dataset generation** — creates 1,000 synthetic but realistic feedback records with correlated metadata.
2. **Text preprocessing** — cleans messages, engineers derived features, and saves an analysis-ready CSV.
3. **Exploratory data analysis** — computes distributions and generates stakeholder charts and an EDA report.
4. **Sentiment modeling** — trains three classifiers (Logistic Regression, LinearSVC, Multinomial Naive Bayes) on TF-IDF features and selects the best by negative recall.
5. **Product area and topic insights** — aggregates risk by product area, channel, and segment; extracts high-risk unresolved feedback.
6. **Executive reporting** — consolidates all artifacts into leadership and portfolio summaries with computed KPIs.

## Why NLP

Customer feedback is primarily unstructured text. Simple keyword counts miss context, negation, and multilingual phrasing. NLP methods — text normalization, TF-IDF vectorization, and supervised classification — convert raw messages into structured sentiment labels that can be aggregated, tracked over time, and linked to business metadata like product area, priority, and resolution status.

## Why Negative Recall Matters

Missing negative feedback is more costly than a false alarm. A dissatisfied customer who goes undetected represents churn risk, reputation damage, and missed escalation. False positives (flagging neutral feedback as negative) add review overhead but rarely cause harm. The model selection criterion therefore prioritizes **negative recall** — catching as many true negatives as possible — with macro F1 as the tie-breaker when models perform similarly.

## Key Results

- **1,000** feedback records across 6 channels, 5 segments, and 10 product areas
- **40.4%** negative feedback rate
- **LinearSVC** selected as best model
- **1.0000** negative recall
- **135** high-risk unresolved items (negative, high/critical priority, unresolved)
- **71** unresolved critical items
- **billing** as top risk product area (21 unresolved critical cases)

These strong model results should be interpreted carefully: the dataset is synthetic and labels were generated alongside the messages, so performance is likely inflated compared to noisy real-world feedback.

## What I Would Improve With Real Data

- **Human annotation** — replace programmatic labels with expert-reviewed sentiment and product-area tags
- **Taxonomy refinement** — validate and extend product-area categories against actual customer language
- **Error analysis** — inspect misclassified cases to understand failure modes (sarcasm, mixed sentiment, code-switching)
- **Real feedback validation** — compare pipeline findings against known support escalations and churn events
- **GenAI-assisted reporting** — draft executive summaries from validated analytical outputs with human review
- **Dashboarding** — build an interactive Streamlit dashboard for stakeholders to explore trends and drill into high-risk items

## Potential Interview Questions

**Why did you use TF-IDF instead of embeddings?**
TF-IDF is interpretable, fast, and works well on small-to-medium datasets without GPU infrastructure. For a portfolio baseline with 1,000 records, it demonstrates solid classical NLP fundamentals. With real data at scale, I would compare TF-IDF against transformer embeddings and evaluate trade-offs in accuracy, latency, and cost.

**Why did you compare three models?**
Comparing Logistic Regression, LinearSVC, and Multinomial Naive Bayes shows deliberate model selection rather than picking one algorithm arbitrarily. Each has different assumptions (linear boundaries vs. probabilistic word counts), and the comparison reveals which approach best captures negative feedback on this dataset.

**Why prioritize recall for negative feedback?**
In a triage context, the cost of missing a negative case (undetected churn risk) exceeds the cost of reviewing a false positive. Prioritizing negative recall aligns the model with the business objective of catching dissatisfied customers early.

**What are the limitations of synthetic data?**
Synthetic messages follow programmatic templates and correlations, so text patterns are more regular and labels are more consistent than in real feedback. This inflates model performance and may not reflect sarcasm, typos, or ambiguous sentiment found in production data.

**How would this change with real customer feedback?**
I would expect lower model accuracy, noisier labels, and the need for human annotation, active learning, and periodic retraining. Product-area classification would likely require a dedicated model rather than relying on metadata alone.

**How would you deploy this?**
I would containerize the preprocessing and inference steps, expose a batch or API endpoint for scoring incoming feedback, store predictions in a data warehouse, and trigger alerts when high-priority negative cases are detected. Model artifacts (`sentiment_model.pkl`, `tfidf_vectorizer.pkl`) would be versioned and loaded at serving time.

**Where would GenAI fit?**
GenAI would serve as a synthesis assistant — drafting executive summaries, explaining trends, and summarizing high-risk feedback samples — grounded in validated analytical outputs. It would not replace the ML models or computed KPIs; those remain the source of truth.

**How would you monitor the model?**
Track prediction volume and class distribution over time, monitor negative recall and precision on a held-out validation set refreshed with newly annotated data, set alerts for distribution drift (e.g., sudden spike in neutral predictions), and schedule periodic retraining when performance degrades below thresholds.

**How do you handle bilingual feedback?**
The preprocessing preserves Spanish and English characters. TF-IDF treats each language's tokens independently, which works for a bilingual dataset of this size. At scale, I would evaluate language-specific models or multilingual embeddings.

**What business actions does the pipeline enable?**
It identifies the top risk product area (billing), quantifies the high-risk escalation queue (135 items), ranks channels by negative rate, and recommends targeted actions — all computed programmatically from the cleaned dataset and saved as executive-ready reports.
