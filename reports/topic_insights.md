# LumenVox — Product Area & Topic Insights Report

## Objective

This report identifies where negative, critical, and unresolved feedback is concentrated across product areas, channels, and customer segments so product, support, and operations teams can prioritize action on the highest-risk issues.

## Executive Summary

- **135 high-risk feedback items** are negative, high or critical priority, and still unresolved.
- **billing** leads unresolved critical backlog with **21** open critical items.
- **email** has the highest negative rate at **64.5%** of all feedback on that channel.
- **billing** shows the slowest average response time at **55.4 hours**.
- **71 unresolved critical** cases remain in the overall backlog and require escalation attention.

## Product Area Risk Summary

Product areas ranked by unresolved critical feedback, negative volume, and average response time.

| product_area | total_feedback | negative_feedback | negative_rate | unresolved_critical_feedback | avg_response_time_hours | resolution_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| billing | 89 | 39 | 43.8% | 21 | 55.4 | 59.6% |
| performance | 108 | 37 | 34.3% | 17 | 53.4 | 63.9% |
| account_access | 105 | 50 | 47.6% | 15 | 54.7 | 61.9% |
| integrations | 101 | 35 | 34.7% | 9 | 52.9 | 59.4% |
| onboarding | 95 | 43 | 45.3% | 3 | 30.2 | 71.6% |
| features | 106 | 44 | 41.5% | 2 | 29.7 | 79.2% |
| exports | 95 | 36 | 37.9% | 2 | 30.5 | 73.7% |
| usability | 107 | 48 | 44.9% | 1 | 33.8 | 76.6% |
| pricing | 87 | 33 | 37.9% | 1 | 33.5 | 80.5% |
| support | 107 | 39 | 36.4% | 0 | 32.8 | 81.3% |

## Highest Risk Feedback

- **High-risk record count:** 135

Top 5 high-risk feedback examples:

- **FB-00575** — `account_access` | priority: **critical** | channel: **email** | segment: **freelancer** | response time: **164.2 hours**
  - Message: Un acceso no autorizado a nuestro workspace no fue detectado a tiempo. Como freelancer, esto afecta mucho mi flujo de trabajo.
- **FB-00153** — `account_access` | priority: **critical** | channel: **chat** | segment: **freelancer** | response time: **162.8 hours**
  - Message: Admin accounts lost permissions after the last security update.
- **FB-00010** — `billing` | priority: **critical** | channel: **email** | segment: **freelancer** | response time: **162.5 hours**
  - Message: The invoice total doesn't match what I agreed to at checkout. As a freelancer, this matters a lot to my workflow.
- **FB-00812** — `performance` | priority: **critical** | channel: **email** | segment: **freelancer** | response time: **159.3 hours**
  - Message: The app crashes constantly when loading large reports.
- **FB-00902** — `features` | priority: **critical** | channel: **email** | segment: **startup** | response time: **157.4 hours**
  - Message: Key features advertised on the website simply don't exist yet. We're scaling fast and need this resolved quickly.

## Negative Feedback by Segment

Customer segments ranked by negative feedback volume and share of all negative items.

- **freelancer**: 95 negatives (23.5% of negatives)
- **agency**: 84 negatives (20.8% of negatives)
- **enterprise**: 79 negatives (19.6% of negatives)
- **startup**: 74 negatives (18.3% of negatives)
- **small_business**: 72 negatives (17.8% of negatives)

## Channel Risk

Channels ranked by negative rate (negative feedback / total feedback).

- **email**: 64.5% negative (109 of 169)
- **support_ticket**: 64.0% negative (105 of 164)
- **chat**: 58.5% negative (100 of 171)
- **contact_form**: 27.0% negative (48 of 178)
- **nps_survey**: 14.1% negative (23 of 163)
- **review**: 12.3% negative (19 of 155)

## Operational Risks

- **Unresolved critical backlog:** 71 critical items remain open, with concentration in areas such as **billing** (21 items).
- **Slow response in high-risk areas:** billing averages 55.4 hours among areas with unresolved critical feedback, delaying resolution for the most severe issues.
- **Customer trust risk:** 135 negative high-priority items are still unresolved, increasing churn risk for affected accounts.
- **Escalation need:** Critical unresolved feedback in billing, account access, performance, and integrations should trigger immediate product and support escalation workflows.

## Recommended Actions

- Prioritize unresolved critical issues in account access, billing, performance, and integrations.
- Create escalation rules for critical unresolved feedback.
- Review support workflows for **email**, the channel with the highest negative concentration.
- Track negative feedback rate weekly by product area.
- Use high-risk feedback samples for qualitative review and root-cause analysis.

## Limitations

- The dataset is synthetic and designed for portfolio demonstration.
- Product areas are generated labels, not validated taxonomies.
- A real-world implementation would require human validation and taxonomy refinement before executive action.

## Generated Artifacts

- `reports/product_area_summary.csv`
- `reports/high_risk_feedback.csv`
- `reports/topic_insights.md`
- `visuals/negative_feedback_by_segment.png`
- `visuals/unresolved_critical_by_area.png`
- `visuals/negative_rate_by_channel.png`
- `visuals/avg_response_time_by_area.png`
