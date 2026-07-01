"""Consolidated executive and portfolio reporting for the LumenVox pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "customer_feedback_clean.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

EDA_REPORT_PATH = REPORTS_DIR / "eda_summary.md"
MODEL_REPORT_PATH = REPORTS_DIR / "model_metrics.md"
MODEL_COMPARISON_PATH = REPORTS_DIR / "sentiment_model_comparison.csv"
TOPIC_REPORT_PATH = REPORTS_DIR / "topic_insights.md"
PRODUCT_AREA_SUMMARY_PATH = REPORTS_DIR / "product_area_summary.csv"
HIGH_RISK_FEEDBACK_PATH = REPORTS_DIR / "high_risk_feedback.csv"
MODEL_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

EXECUTIVE_SUMMARY_PATH = REPORTS_DIR / "executive_summary.md"
PROJECT_SUMMARY_PATH = REPORTS_DIR / "lumenvox_project_summary.md"

BOOLEAN_COLUMNS = [
    "resolved",
    "is_negative",
    "is_critical",
    "is_unresolved_critical",
]

_TRUE_STRINGS = {"true", "1", "yes", "y", "t"}
_FALSE_STRINGS = {"false", "0", "no", "n", "f"}

REQUIRED_PATHS = (
    CLEAN_PATH,
    EDA_REPORT_PATH,
    MODEL_REPORT_PATH,
    MODEL_COMPARISON_PATH,
    TOPIC_REPORT_PATH,
    PRODUCT_AREA_SUMMARY_PATH,
    HIGH_RISK_FEEDBACK_PATH,
    MODEL_METADATA_PATH,
)


def _require_paths(*paths: Path) -> None:
    """Raise FileNotFoundError for the first missing path."""
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)


def _format_rate(value: float) -> str:
    """Format a proportion as a percentage string."""
    return f"{value * 100:.1f}%"


def _format_date(value) -> str:
    """Format a date-like value as YYYY-MM-DD."""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)[:10]


def _coerce_boolean(series: pd.Series) -> pd.Series:
    """Safely coerce a column to boolean without treating invalid values as True."""
    if series.dtype == bool:
        return series

    normalized = series.astype(str).str.strip().str.lower()
    result = pd.Series(False, index=series.index, dtype=bool)
    result[normalized.isin(_TRUE_STRINGS)] = True
    result[normalized.isin(_FALSE_STRINGS)] = False
    return result


def load_inputs() -> dict:
    """Load and validate all upstream pipeline artifacts."""
    _require_paths(*REQUIRED_PATHS)

    clean_df = pd.read_csv(CLEAN_PATH, parse_dates=["date"])
    for column in BOOLEAN_COLUMNS:
        if column in clean_df.columns:
            clean_df[column] = _coerce_boolean(clean_df[column])

    return {
        "clean_df": clean_df,
        "model_comparison": pd.read_csv(MODEL_COMPARISON_PATH),
        "product_area_summary": pd.read_csv(PRODUCT_AREA_SUMMARY_PATH),
        "high_risk_feedback": pd.read_csv(HIGH_RISK_FEEDBACK_PATH),
        "model_metadata": json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8")),
        "eda_report": EDA_REPORT_PATH.read_text(encoding="utf-8"),
        "model_report": MODEL_REPORT_PATH.read_text(encoding="utf-8"),
        "topic_report": TOPIC_REPORT_PATH.read_text(encoding="utf-8"),
    }


def get_top_findings(inputs: dict) -> dict:
    """Compute executive KPIs from loaded pipeline artifacts."""
    df = inputs["clean_df"]
    product_area_summary = inputs["product_area_summary"]
    high_risk_feedback = inputs["high_risk_feedback"]
    model_comparison = inputs["model_comparison"]
    model_metadata = inputs["model_metadata"]

    total_feedback = len(df)
    negative_count = int(df["is_negative"].sum())
    negative_rate = negative_count / total_feedback if total_feedback else 0.0

    language_counts = df["language"].value_counts()
    language_split = ", ".join(
        f"{lang} {count} ({count / total_feedback * 100:.1f}%)"
        for lang, count in language_counts.items()
    )

    total_by_channel = df["channel"].value_counts()
    negative_by_channel = df[df["is_negative"]]["channel"].value_counts().reindex(
        total_by_channel.index,
        fill_value=0,
    )
    channel_negative_rate = (negative_by_channel / total_by_channel).sort_values(
        ascending=False,
    )
    top_negative_channel = channel_negative_rate.index[0]
    top_negative_channel_rate = float(channel_negative_rate.iloc[0])

    segment_negative_counts = df[df["is_negative"]]["customer_segment"].value_counts()
    top_negative_segment = segment_negative_counts.index[0]
    top_negative_segment_count = int(segment_negative_counts.iloc[0])

    top_risk_row = product_area_summary.sort_values(
        "unresolved_critical_feedback",
        ascending=False,
    ).iloc[0]
    top_risk_area = top_risk_row["product_area"]
    top_risk_area_unresolved_critical = int(top_risk_row["unresolved_critical_feedback"])

    top_negative_row = product_area_summary.sort_values(
        "negative_feedback",
        ascending=False,
    ).iloc[0]
    top_negative_area = top_negative_row["product_area"]
    top_negative_area_count = int(top_negative_row["negative_feedback"])

    slowest_row = product_area_summary.sort_values(
        "avg_response_time_hours",
        ascending=False,
    ).iloc[0]
    slowest_response_area = slowest_row["product_area"]
    slowest_response_area_hours = float(slowest_row["avg_response_time_hours"])

    best_model_name = model_metadata["best_model_name"]
    best_model_row = model_comparison.loc[
        model_comparison["model_name"] == best_model_name
    ].iloc[0]

    return {
        "total_feedback": total_feedback,
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
        "negative_count": negative_count,
        "negative_rate": negative_rate,
        "critical_count": int(df["is_critical"].sum()),
        "unresolved_critical_count": int(df["is_unresolved_critical"].sum()),
        "language_split": language_split,
        "top_negative_channel": top_negative_channel,
        "top_negative_channel_rate": top_negative_channel_rate,
        "top_negative_segment": top_negative_segment,
        "top_negative_segment_count": top_negative_segment_count,
        "top_risk_area": top_risk_area,
        "top_risk_area_unresolved_critical": top_risk_area_unresolved_critical,
        "top_negative_area": top_negative_area,
        "top_negative_area_count": top_negative_area_count,
        "slowest_response_area": slowest_response_area,
        "slowest_response_area_hours": slowest_response_area_hours,
        "high_risk_count": len(high_risk_feedback),
        "critical_high_risk_count": int((high_risk_feedback["priority"] == "critical").sum()),
        "high_priority_high_risk_count": int((high_risk_feedback["priority"] == "high").sum()),
        "best_model_name": best_model_name,
        "best_model_negative_recall": float(best_model_row["negative_recall"]),
        "best_model_macro_f1": float(best_model_row["macro_f1"]),
        "best_model_weighted_f1": float(best_model_row["weighted_f1"]),
    }


def generate_executive_summary(inputs: dict, findings: dict) -> None:
    """Write the business-oriented executive summary report."""
    df = inputs["clean_df"]

    channels = ", ".join(df["channel"].value_counts().index.tolist())
    segments = ", ".join(df["customer_segment"].value_counts().index.tolist())
    product_areas = ", ".join(df["product_area"].value_counts().index.tolist())

    lines: list[str] = []
    lines.append("# LumenVox — Executive Summary")
    lines.append("")
    lines.append("## Executive Overview")
    lines.append("")
    lines.append(
        "LumenVox transforms unstructured customer feedback from a fictional SaaS company "
        "into actionable intelligence for product, support, and operations leaders. "
        f"Across **{findings['total_feedback']:,} feedback records**, "
        f"**{_format_rate(findings['negative_rate'])}** of messages are negative, "
        f"**{findings['high_risk_count']}** items require immediate escalation, and "
        f"**{findings['unresolved_critical_count']}** critical cases remain unresolved. "
        f"The selected sentiment model (**{findings['best_model_name']}**) achieves "
        f"**{_format_rate(findings['best_model_negative_recall'])} negative recall**, "
        "supporting reliable triage of dissatisfied customers."
    )
    lines.append("")
    lines.append("## Executive KPI Snapshot")
    lines.append("")
    lines.append("| KPI | Value |")
    lines.append("|-----|-------|")
    lines.append(f"| Total feedback records | {findings['total_feedback']:,} |")
    lines.append(f"| Negative feedback rate | {_format_rate(findings['negative_rate'])} |")
    lines.append(f"| High-risk unresolved feedback | {findings['high_risk_count']} |")
    lines.append(
        f"| Unresolved critical feedback | {findings['unresolved_critical_count']} |"
    )
    lines.append(
        f"| Top risk product area | {findings['top_risk_area']} "
        f"({findings['top_risk_area_unresolved_critical']} unresolved critical) |"
    )
    lines.append(
        f"| Highest negative-rate channel | {findings['top_negative_channel']} "
        f"({_format_rate(findings['top_negative_channel_rate'])}) |"
    )
    lines.append(f"| Best sentiment model | {findings['best_model_name']} |")
    lines.append(
        f"| Negative recall | {_format_rate(findings['best_model_negative_recall'])} |"
    )
    lines.append("")
    lines.append("## Business Problem")
    lines.append("")
    lines.append(
        "SaaS companies collect thousands of open-text feedback items every month through "
        "support tickets, product reviews, NPS surveys, live chat, email, and contact forms. "
        "Leadership teams struggle to answer basic questions: What are customers complaining "
        "about? Which product areas drive churn risk? Where should engineering prioritize fixes? "
        "Without structured analytics, valuable signals remain buried in unstructured, multilingual "
        "feedback scattered across channels."
    )
    lines.append("")
    lines.append("## Dataset Overview")
    lines.append("")
    lines.append(f"- **Total feedback records:** {findings['total_feedback']:,}")
    lines.append(
        f"- **Date range:** {_format_date(findings['date_min'])} to "
        f"{_format_date(findings['date_max'])}"
    )
    lines.append(f"- **Language split:** {findings['language_split']}")
    lines.append(f"- **Channels:** {channels}")
    lines.append(f"- **Customer segments:** {segments}")
    lines.append(f"- **Product areas:** {product_areas}")
    lines.append("")
    lines.append("## Key Feedback Patterns")
    lines.append("")
    lines.append(
        f"- **Negative feedback:** {findings['negative_count']} records "
        f"({_format_rate(findings['negative_rate'])} of total volume)"
    )
    lines.append(
        f"- **Top negative channel:** {findings['top_negative_channel']} "
        f"({_format_rate(findings['top_negative_channel_rate'])} negative rate)"
    )
    lines.append(
        f"- **Top negative segment:** {findings['top_negative_segment']} "
        f"({findings['top_negative_segment_count']} negative records)"
    )
    lines.append(
        f"- **Top negative product area:** {findings['top_negative_area']} "
        f"({findings['top_negative_area_count']} negative records)"
    )
    lines.append(
        f"- **Unresolved critical backlog:** {findings['unresolved_critical_count']} cases"
    )
    lines.append("")
    lines.append("## Sentiment Modeling Results")
    lines.append("")
    lines.append(
        f"- **Best model:** {findings['best_model_name']}"
    )
    lines.append(
        f"- **Negative recall:** {_format_rate(findings['best_model_negative_recall'])}"
    )
    lines.append(
        f"- **Macro F1:** {findings['best_model_macro_f1']:.3f}"
    )
    lines.append(
        f"- **Weighted F1:** {findings['best_model_weighted_f1']:.3f}"
    )
    lines.append(
        "- **Selection rationale:** Negative recall was prioritized because missing negative "
        "feedback is more costly than false alarms — dissatisfied customers who go undetected "
        "represent churn and reputation risk. Macro F1 serves as the tie-breaker when models "
        "achieve comparable negative recall."
    )
    lines.append("")
    lines.append("## Product Area Risk Insights")
    lines.append("")
    lines.append(
        f"- **Top risk area:** {findings['top_risk_area']} "
        f"({findings['top_risk_area_unresolved_critical']} unresolved critical cases)"
    )
    lines.append(
        f"- **Slowest response area:** {findings['slowest_response_area']} "
        f"({findings['slowest_response_area_hours']:.1f} hours average response time)"
    )
    lines.append(
        f"- **High-risk feedback count:** {findings['high_risk_count']} "
        "(negative, high/critical priority, unresolved)"
    )
    lines.append(
        f"- **Critical high-risk feedback:** {findings['critical_high_risk_count']} cases"
    )
    lines.append("")
    lines.append("## Recommended Business Actions")
    lines.append("")
    lines.append(
        f"1. **Escalate {findings['top_risk_area']} backlog** — "
        f"{findings['top_risk_area_unresolved_critical']} unresolved critical cases require "
        "immediate product and support review."
    )
    lines.append(
        f"2. **Review {findings['slowest_response_area']} response workflows** — "
        f"average response time of {findings['slowest_response_area_hours']:.1f} hours "
        "exceeds acceptable SLA thresholds for high-impact areas."
    )
    lines.append(
        f"3. **Audit the {findings['top_negative_channel']} channel** — "
        f"{_format_rate(findings['top_negative_channel_rate'])} negative rate suggests "
        "workflow or expectation gaps that drive dissatisfaction."
    )
    lines.append(
        f"4. **Target {findings['top_negative_segment']} segment outreach** — "
        f"{findings['top_negative_segment_count']} negative records indicate concentrated "
        "pain points worth proactive account management."
    )
    lines.append(
        f"5. **Deploy sentiment triage with {findings['best_model_name']}** — "
        f"use automated negative detection ({_format_rate(findings['best_model_negative_recall'])} "
        "recall) to route urgent feedback before SLA breaches."
    )
    lines.append(
        f"6. **Clear the {findings['high_risk_count']}-item high-risk queue** — "
        f"prioritize the {findings['critical_high_risk_count']} critical cases and establish "
        "weekly escalation reviews."
    )
    lines.append(
        f"7. **Address {findings['top_negative_area']} product friction** — "
        f"{findings['top_negative_area_count']} negative records make this the highest-volume "
        "dissatisfaction driver for engineering prioritization."
    )
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- The dataset is **synthetic** and designed for portfolio demonstration, not production "
        "decision-making."
    )
    lines.append(
        "- Sentiment labels were **generated programmatically** alongside the synthetic messages, "
        "which can inflate model performance compared to noisy real-world labels."
    )
    lines.append(
        "- Model metrics may be **higher than real-world conditions** where language is messier, "
        "sarcasm is common, and labels are inconsistent."
    )
    lines.append(
        "- Findings require **human validation** and ongoing **taxonomy refinement** before "
        "operational deployment."
    )
    lines.append("")
    lines.append("## Next Steps")
    lines.append("")
    lines.append("- Conduct **error analysis** on misclassified sentiment cases.")
    lines.append("- Validate findings against **real customer feedback** samples.")
    lines.append("- Build a **product-area classifier** to automate routing beyond metadata.")
    lines.append("- Add **time trend analysis** to track sentiment and resolution over months.")
    lines.append(
        "- Optionally add a **GenAI-assisted summary layer** grounded in validated outputs."
    )
    lines.append(
        "- Optionally build a **Streamlit dashboard** for interactive exploration by stakeholders."
    )
    lines.append("")
    lines.append("## Generated Artifacts")
    lines.append("")
    lines.append("- `data/processed/customer_feedback_clean.csv`")
    lines.append("- `reports/eda_summary.md`")
    lines.append("- `reports/model_metrics.md`")
    lines.append("- `reports/topic_insights.md`")
    lines.append("- `reports/product_area_summary.csv`")
    lines.append("- `reports/high_risk_feedback.csv`")
    lines.append("- `reports/executive_summary.md`")
    lines.append("- `reports/lumenvox_project_summary.md`")
    lines.append("- `reports/sentiment_model_comparison.csv`")
    lines.append("- `models/sentiment_model.pkl`")
    lines.append("- `models/tfidf_vectorizer.pkl`")
    lines.append("- `models/model_metadata.json`")
    lines.append("- `visuals/*.png`")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EXECUTIVE_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def generate_project_summary(inputs: dict, findings: dict) -> None:
    """Write the portfolio-oriented project summary report."""
    lines: list[str] = []
    lines.append("# LumenVox — Project Summary for Portfolio")
    lines.append("")
    lines.append("## Project Objective")
    lines.append("")
    lines.append(
        "LumenVox is a business-oriented Data Science and AI portfolio project that builds an "
        "end-to-end pipeline to ingest, analyze, and summarize customer feedback for a fictional "
        "SaaS company. The goal is to demonstrate how unstructured text can be transformed into "
        "actionable insights for product, support, and operations teams."
    )
    lines.append("")
    lines.append("## Business Context")
    lines.append("")
    lines.append(
        "The project simulates a feedback intelligence platform for a SaaS product serving "
        "freelancers, small businesses, startups, enterprises, and agencies across multiple "
        "subscription tiers. Customer messages arrive through support tickets, reviews, NPS "
        "surveys, chat, email, and contact forms — covering billing, performance, usability, "
        "integrations, and other product areas. Leadership needs to prioritize fixes, reduce "
        "churn risk, and improve operational response times."
    )
    lines.append("")
    lines.append("## Technical Pipeline")
    lines.append("")
    lines.append("The LumenVox pipeline consists of six completed steps:")
    lines.append("")
    lines.append("1. **Synthetic dataset generation** — realistic bilingual feedback with business correlations")
    lines.append("2. **Text preprocessing and feature engineering** — cleaning, normalization, derived flags")
    lines.append("3. **Exploratory data analysis** — distributions, charts, and business-oriented EDA report")
    lines.append("4. **Sentiment modeling** — TF-IDF features and classical classifiers with model selection")
    lines.append("5. **Product area and topic insights** — risk summaries, high-risk extraction, recommendations")
    lines.append("6. **Executive reporting** — consolidated executive and portfolio summaries")
    lines.append("")
    lines.append("## Dataset Design")
    lines.append("")
    lines.append(
        f"The project uses **{findings['total_feedback']:,} synthetic feedback records** spanning "
        "2025 with bilingual messages (English and Spanish). Each record includes channel, "
        "customer segment, plan type, product area, rating, sentiment label, priority, resolution "
        "status, and response time."
    )
    lines.append("")
    lines.append("Key design elements:")
    lines.append("")
    lines.append("- **10 product areas:** billing, performance, usability, support, pricing, features, onboarding, exports, integrations, account access")
    lines.append("- **6 channels:** support_ticket, review, nps_survey, chat, email, contact_form")
    lines.append("- **5 customer segments:** freelancer, small_business, startup, enterprise, agency")
    lines.append("- **4 plan types:** free, basic, pro, enterprise")
    lines.append(
        "- **Business correlation rules:** negative feedback concentrated in email and support "
        "channels; critical priority skewed toward billing and account access; response times "
        "escalate with priority level"
    )
    lines.append("")
    lines.append("## NLP and Machine Learning Approach")
    lines.append("")
    lines.append(
        "Text is normalized into a `clean_message` column (lowercase, URL removal, punctuation "
        "stripping with multilingual character preservation). Features are extracted with "
        "**TF-IDF** using unigrams and bigrams (`ngram_range=(1, 2)`), up to 5,000 features."
    )
    lines.append("")
    lines.append("Three supervised classifiers are compared:")
    lines.append("")
    lines.append("- **Logistic Regression** with balanced class weights")
    lines.append("- **LinearSVC** with balanced class weights")
    lines.append("- **Multinomial Naive Bayes** as a probabilistic baseline")
    lines.append("")
    lines.append(
        f"Model selection prioritizes **negative recall** with **macro F1** as tie-breaker. "
        f"The best model (**{findings['best_model_name']}**) achieves "
        f"{_format_rate(findings['best_model_negative_recall'])} negative recall."
    )
    lines.append("")
    lines.append("## Business Intelligence Layer")
    lines.append("")
    lines.append(
        "Beyond classification, the pipeline produces operational intelligence artifacts:"
    )
    lines.append("")
    lines.append(
        "- **`product_area_summary.csv`** — per-area volume, negative counts, unresolved critical "
        "backlog, response times, and resolution rates"
    )
    lines.append(
        "- **`high_risk_feedback.csv`** — escalation-ready list of negative, high/critical, "
        "unresolved feedback sorted by priority and response time"
    )
    lines.append(
        "- **`topic_insights.md`** — business-oriented risk analysis with segment and channel breakdowns"
    )
    lines.append(
        "- **`executive_summary.md`** — consolidated leadership brief with KPIs and recommended actions"
    )
    lines.append("")
    lines.append("## Key Results")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total feedback | {findings['total_feedback']:,} |")
    lines.append(f"| Negative feedback rate | {_format_rate(findings['negative_rate'])} |")
    lines.append(f"| High-risk feedback count | {findings['high_risk_count']} |")
    lines.append(f"| Unresolved critical feedback | {findings['unresolved_critical_count']} |")
    lines.append(
        f"| Top risk product area | {findings['top_risk_area']} "
        f"({findings['top_risk_area_unresolved_critical']} unresolved critical) |"
    )
    lines.append(
        f"| Best model | {findings['best_model_name']} "
        f"(negative recall: {_format_rate(findings['best_model_negative_recall'])}) |"
    )
    lines.append("")
    lines.append("## Tools Used")
    lines.append("")
    lines.append("- Python")
    lines.append("- pandas")
    lines.append("- NumPy")
    lines.append("- scikit-learn")
    lines.append("- matplotlib")
    lines.append("- joblib")
    lines.append("- Markdown reporting")
    lines.append("")
    lines.append("## Portfolio Value")
    lines.append("")
    lines.append("This project demonstrates end-to-end Data Science and AI capabilities:")
    lines.append("")
    lines.append("- **Unstructured text analysis** — ingesting and normalizing multilingual feedback")
    lines.append("- **NLP preprocessing** — cleaning pipelines with derived feature engineering")
    lines.append("- **Supervised text classification** — TF-IDF vectorization and classical ML models")
    lines.append("- **Model evaluation** — multi-metric comparison with business-driven selection criteria")
    lines.append("- **Business metric design** — negative recall prioritization for operational triage")
    lines.append("- **Customer/product analytics** — segment, channel, and product-area risk analysis")
    lines.append("- **Executive reporting** — stakeholder-ready summaries grounded in computed KPIs")
    lines.append("- **AI/data storytelling** — translating technical outputs into actionable recommendations")
    lines.append("")
    lines.append("## What This Project Demonstrates")
    lines.append("")
    lines.append("- NLP preprocessing and text normalization")
    lines.append("- Supervised machine learning for sentiment classification")
    lines.append("- Model selection based on business criteria (negative recall)")
    lines.append("- Customer feedback intelligence and risk prioritization")
    lines.append("- Product-area operational analytics")
    lines.append("- Stakeholder-facing reporting for leadership and portfolio review")
    lines.append("")
    lines.append("## Future Improvements")
    lines.append("")
    lines.append("- Validate pipeline on **real customer feedback** datasets")
    lines.append("- Experiment with **embedding-based** representations (e.g., sentence transformers)")
    lines.append("- Add **topic modeling** (LDA, NMF) for unsupervised theme discovery")
    lines.append("- Build a **product-area classifier** to reduce reliance on metadata labels")
    lines.append("- Add a **GenAI synthesis layer** for draft summaries grounded in validated outputs")
    lines.append("- Deploy an interactive **Streamlit dashboard** for stakeholder exploration")
    lines.append("")
    lines.append("## Generated Artifacts")
    lines.append("")
    lines.append("- `data/raw/customer_feedback_raw.csv`")
    lines.append("- `data/processed/customer_feedback_clean.csv`")
    lines.append("- `reports/eda_summary.md`")
    lines.append("- `reports/model_metrics.md`")
    lines.append("- `reports/sentiment_model_comparison.csv`")
    lines.append("- `reports/topic_insights.md`")
    lines.append("- `reports/product_area_summary.csv`")
    lines.append("- `reports/high_risk_feedback.csv`")
    lines.append("- `reports/executive_summary.md`")
    lines.append("- `reports/lumenvox_project_summary.md`")
    lines.append("- `models/sentiment_model.pkl`")
    lines.append("- `models/tfidf_vectorizer.pkl`")
    lines.append("- `models/model_metadata.json`")
    lines.append("- `visuals/*.png`")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROJECT_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    inputs = load_inputs()
    findings = get_top_findings(inputs)
    generate_executive_summary(inputs, findings)
    generate_project_summary(inputs, findings)
    print("Executive summary saved to: reports/executive_summary.md")
    print("Project summary saved to: reports/lumenvox_project_summary.md")


if __name__ == "__main__":
    main()
