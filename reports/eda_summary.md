# LumenVox — Exploratory Data Analysis Summary

This report translates raw customer feedback into an executive view of sentiment, operational performance, and product risk areas. It is generated automatically from the cleaned dataset.

## Dataset Overview

- **Row count:** 1000
- **Column count:** 23
- **Date range:** 2025-01-01 to 2025-12-31
- **Language split:** en 603 (60.3%), es 397 (39.7%)

## Key Distributions

**Sentiment distribution**

- **negative**: 404 (40.4%)
- **neutral**: 281 (28.1%)
- **positive**: 315 (31.5%)

**Channel distribution**

- **contact_form**: 178 (17.8%)
- **chat**: 171 (17.1%)
- **email**: 169 (16.9%)
- **support_ticket**: 164 (16.4%)
- **nps_survey**: 163 (16.3%)
- **review**: 155 (15.5%)

**Priority distribution**

- **low**: 221 (22.1%)
- **medium**: 376 (37.6%)
- **high**: 310 (31.0%)
- **critical**: 93 (9.3%)

**Top product areas by volume**

- **performance**: 108 (20.3%)
- **support**: 107 (20.1%)
- **usability**: 107 (20.1%)
- **features**: 106 (19.9%)
- **account_access**: 105 (19.7%)

## Main Negative Feedback Areas

Of 404 negative feedback items, the following product areas account for the largest share:

- **account_access**: 50 negatives (12.4% of all negatives)
- **usability**: 48 negatives (11.9% of all negatives)
- **features**: 44 negatives (10.9% of all negatives)
- **onboarding**: 43 negatives (10.6% of all negatives)
- **billing**: 39 negatives (9.7% of all negatives)

## Channels with Highest Negative Feedback Concentration

Negative rate is computed as negative feedback divided by total feedback per channel, ranked from highest to lowest concentration.

- **email**: 64.5% negative (109 of 169)
- **support_ticket**: 64.0% negative (105 of 164)
- **chat**: 58.5% negative (100 of 171)
- **contact_form**: 27.0% negative (48 of 178)
- **nps_survey**: 14.1% negative (23 of 163)
- **review**: 12.3% negative (19 of 155)

## Average Response Time by Priority

- **low**: 12.2 hours
- **medium**: 27.7 hours
- **high**: 59.7 hours
- **critical**: 98.2 hours

## Resolution Rate by Priority

- **low**: 94.6%
- **medium**: 81.4%
- **high**: 55.2%
- **critical**: 23.7%

## Initial Business Insights

- Negative feedback concentrates most heavily in the **email** channel, where 64.5% of items are negative — a priority channel for service quality intervention.
- The **account_access** product area dominates negative feedback, making it a strong candidate for product and engineering focus.
- Response times escalate with severity: critical issues average 98.2 hours versus 12.2 hours for low-priority items, indicating that the most severe cases also wait the longest.
- There are **71 unresolved critical** issues — a small but high-risk backlog that warrants immediate operational attention.

## Suggested Next Analytical Steps

- Build TF-IDF features and classical sentiment classification baselines.
- Apply topic modeling within product areas to surface recurring themes.
- Produce segment and plan cross-tabs to compare feedback patterns across customer tiers.
- Add time-series trend analysis to track sentiment and volume over months.
- Extend executive reporting with automated, stakeholder-ready summaries.

## Generated Artifacts

- `data/processed/customer_feedback_clean.csv`
- `visuals/sentiment_distribution.png`
- `visuals/feedback_by_channel.png`
- `visuals/feedback_by_area.png`
- `visuals/negative_feedback_by_area.png`
- `visuals/avg_response_time_by_priority.png`
- `visuals/resolution_rate_by_priority.png`
