# LumenVox — Executive Summary

## Executive Overview

LumenVox transforms unstructured customer feedback from a fictional SaaS company into actionable intelligence for product, support, and operations leaders. Across **1,000 feedback records**, **40.4%** of messages are negative, **135** items require immediate escalation, and **71** critical cases remain unresolved. The selected sentiment model (**LinearSVC**) achieves **100.0% negative recall**, supporting reliable triage of dissatisfied customers.

## Executive KPI Snapshot

| KPI | Value |
|-----|-------|
| Total feedback records | 1,000 |
| Negative feedback rate | 40.4% |
| High-risk unresolved feedback | 135 |
| Unresolved critical feedback | 71 |
| Top risk product area | billing (21 unresolved critical) |
| Highest negative-rate channel | email (64.5%) |
| Best sentiment model | LinearSVC |
| Negative recall | 100.0% |

## Business Problem

SaaS companies collect thousands of open-text feedback items every month through support tickets, product reviews, NPS surveys, live chat, email, and contact forms. Leadership teams struggle to answer basic questions: What are customers complaining about? Which product areas drive churn risk? Where should engineering prioritize fixes? Without structured analytics, valuable signals remain buried in unstructured, multilingual feedback scattered across channels.

## Dataset Overview

- **Total feedback records:** 1,000
- **Date range:** 2025-01-01 to 2025-12-31
- **Language split:** en 603 (60.3%), es 397 (39.7%)
- **Channels:** contact_form, chat, email, support_ticket, nps_survey, review
- **Customer segments:** freelancer, agency, small_business, enterprise, startup
- **Product areas:** performance, support, usability, features, account_access, integrations, onboarding, exports, billing, pricing

## Key Feedback Patterns

- **Negative feedback:** 404 records (40.4% of total volume)
- **Top negative channel:** email (64.5% negative rate)
- **Top negative segment:** freelancer (95 negative records)
- **Top negative product area:** account_access (50 negative records)
- **Unresolved critical backlog:** 71 cases

## Sentiment Modeling Results

- **Best model:** LinearSVC
- **Negative recall:** 100.0%
- **Macro F1:** 0.981
- **Weighted F1:** 0.980
- **Selection rationale:** Negative recall was prioritized because missing negative feedback is more costly than false alarms — dissatisfied customers who go undetected represent churn and reputation risk. Macro F1 serves as the tie-breaker when models achieve comparable negative recall.

## Product Area Risk Insights

- **Top risk area:** billing (21 unresolved critical cases)
- **Slowest response area:** billing (55.4 hours average response time)
- **High-risk feedback count:** 135 (negative, high/critical priority, unresolved)
- **Critical high-risk feedback:** 49 cases

## Recommended Business Actions

1. **Escalate billing backlog** — 21 unresolved critical cases require immediate product and support review.
2. **Review billing response workflows** — average response time of 55.4 hours exceeds acceptable SLA thresholds for high-impact areas.
3. **Audit the email channel** — 64.5% negative rate suggests workflow or expectation gaps that drive dissatisfaction.
4. **Target freelancer segment outreach** — 95 negative records indicate concentrated pain points worth proactive account management.
5. **Deploy sentiment triage with LinearSVC** — use automated negative detection (100.0% recall) to route urgent feedback before SLA breaches.
6. **Clear the 135-item high-risk queue** — prioritize the 49 critical cases and establish weekly escalation reviews.
7. **Address account_access product friction** — 50 negative records make this the highest-volume dissatisfaction driver for engineering prioritization.

## Limitations

- The dataset is **synthetic** and designed for portfolio demonstration, not production decision-making.
- Sentiment labels were **generated programmatically** alongside the synthetic messages, which can inflate model performance compared to noisy real-world labels.
- Model metrics may be **higher than real-world conditions** where language is messier, sarcasm is common, and labels are inconsistent.
- Findings require **human validation** and ongoing **taxonomy refinement** before operational deployment.

## Next Steps

- Conduct **error analysis** on misclassified sentiment cases.
- Validate findings against **real customer feedback** samples.
- Build a **product-area classifier** to automate routing beyond metadata.
- Add **time trend analysis** to track sentiment and resolution over months.
- Optionally add a **GenAI-assisted summary layer** grounded in validated outputs.
- Optionally build a **Streamlit dashboard** for interactive exploration by stakeholders.

## Generated Artifacts

- `data/processed/customer_feedback_clean.csv`
- `reports/eda_summary.md`
- `reports/model_metrics.md`
- `reports/topic_insights.md`
- `reports/product_area_summary.csv`
- `reports/high_risk_feedback.csv`
- `reports/executive_summary.md`
- `reports/lumenvox_project_summary.md`
- `reports/sentiment_model_comparison.csv`
- `models/sentiment_model.pkl`
- `models/tfidf_vectorizer.pkl`
- `models/model_metadata.json`
- `visuals/*.png`