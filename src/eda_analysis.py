"""Exploratory data analysis and business reporting for LumenVox feedback."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "customer_feedback_clean.csv"
VISUALS_DIR = PROJECT_ROOT / "visuals"
REPORT_PATH = PROJECT_ROOT / "reports" / "eda_summary.md"

PRIORITY_ORDER = ["low", "medium", "high", "critical"]
SENTIMENT_ORDER = ["negative", "neutral", "positive"]


def load_clean_data(path: str) -> pd.DataFrame:
    """Load the processed feedback dataset."""
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def _save_figure(fig: plt.Figure, filename: str) -> None:
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_sentiment_distribution_chart(df: pd.DataFrame) -> None:
    counts = df["sentiment_label"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index, counts.values)
    ax.set_title("Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Feedback Items")
    _save_figure(fig, "sentiment_distribution.png")


def create_feedback_by_channel_chart(df: pd.DataFrame) -> None:
    counts = df["channel"].value_counts()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(counts.index, counts.values)
    ax.set_title("Feedback Volume by Channel")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Number of Feedback Items")
    ax.tick_params(axis="x", rotation=45)
    _save_figure(fig, "feedback_by_channel.png")


def create_feedback_by_area_chart(df: pd.DataFrame) -> None:
    counts = df["product_area"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(counts.index, counts.values)
    ax.set_title("Feedback Volume by Product Area")
    ax.set_xlabel("Product Area")
    ax.set_ylabel("Number of Feedback Items")
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    _save_figure(fig, "feedback_by_area.png")


def create_negative_feedback_by_area_chart(df: pd.DataFrame) -> None:
    negatives = df[df["sentiment_label"] == "negative"]
    counts = negatives["product_area"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(counts.index, counts.values)
    ax.set_title("Negative Feedback by Product Area")
    ax.set_xlabel("Product Area")
    ax.set_ylabel("Number of Negative Feedback Items")
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    _save_figure(fig, "negative_feedback_by_area.png")


def create_avg_response_time_by_priority_chart(df: pd.DataFrame) -> None:
    means = (
        df.groupby("priority")["response_time_hours"]
        .mean()
        .reindex(PRIORITY_ORDER)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(means.index, means.values)
    ax.set_title("Average Response Time by Priority")
    ax.set_xlabel("Priority")
    ax.set_ylabel("Average Response Time (hours)")
    _save_figure(fig, "avg_response_time_by_priority.png")


def create_resolution_rate_by_priority_chart(df: pd.DataFrame) -> None:
    rates = (
        df.groupby("priority")["resolved"].mean().reindex(PRIORITY_ORDER) * 100
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(rates.index, rates.values)
    ax.set_title("Resolution Rate by Priority")
    ax.set_xlabel("Priority")
    ax.set_ylabel("Resolution Rate (%)")
    ax.set_ylim(0, 100)
    _save_figure(fig, "resolution_rate_by_priority.png")


def _format_counts(counts: pd.Series) -> str:
    total = counts.sum()
    lines = []
    for label, count in counts.items():
        share = (count / total * 100) if total else 0
        lines.append(f"- **{label}**: {count} ({share:.1f}%)")
    return "\n".join(lines)


def generate_eda_summary(df: pd.DataFrame) -> None:
    """Compute statistics and write the business-oriented EDA report."""
    row_count = len(df)
    column_count = df.shape[1]
    date_min = df["date"].min().date()
    date_max = df["date"].max().date()

    language_counts = df["language"].value_counts()
    sentiment_counts = df["sentiment_label"].value_counts().reindex(
        SENTIMENT_ORDER, fill_value=0
    )
    channel_counts = df["channel"].value_counts()
    priority_counts = df["priority"].value_counts().reindex(PRIORITY_ORDER, fill_value=0)
    area_counts = df["product_area"].value_counts()

    negatives = df[df["sentiment_label"] == "negative"]
    total_negatives = len(negatives)
    neg_by_area = negatives["product_area"].value_counts()
    top_neg_areas = neg_by_area.head(5)

    total_by_channel = df["channel"].value_counts()
    neg_by_channel = negatives["channel"].value_counts().reindex(
        total_by_channel.index, fill_value=0
    )
    channel_neg_rate = (neg_by_channel / total_by_channel * 100).sort_values(
        ascending=False
    )

    avg_response_by_priority = (
        df.groupby("priority")["response_time_hours"].mean().reindex(PRIORITY_ORDER)
    )
    resolution_by_priority = (
        df.groupby("priority")["resolved"].mean().reindex(PRIORITY_ORDER) * 100
    )

    unresolved_critical = int(df["is_unresolved_critical"].sum()) if "is_unresolved_critical" in df.columns else int(
        ((df["priority"] == "critical") & (~df["resolved"])).sum()
    )

    top_area_negative = top_neg_areas.index[0] if not top_neg_areas.empty else "n/a"
    top_channel_negative = channel_neg_rate.index[0] if not channel_neg_rate.empty else "n/a"
    top_channel_rate = channel_neg_rate.iloc[0] if not channel_neg_rate.empty else 0.0

    low_avg = avg_response_by_priority.get("low", float("nan"))
    critical_avg = avg_response_by_priority.get("critical", float("nan"))
    escalates = critical_avg > low_avg

    # Language split string
    language_split = ", ".join(
        f"{lang} {count} ({count / row_count * 100:.1f}%)"
        for lang, count in language_counts.items()
    )

    lines = []
    lines.append("# LumenVox — Exploratory Data Analysis Summary")
    lines.append("")
    lines.append(
        "This report translates raw customer feedback into an executive view of "
        "sentiment, operational performance, and product risk areas. It is generated "
        "automatically from the cleaned dataset."
    )
    lines.append("")

    lines.append("## Dataset Overview")
    lines.append("")
    lines.append(f"- **Row count:** {row_count}")
    lines.append(f"- **Column count:** {column_count}")
    lines.append(f"- **Date range:** {date_min} to {date_max}")
    lines.append(f"- **Language split:** {language_split}")
    lines.append("")

    lines.append("## Key Distributions")
    lines.append("")
    lines.append("**Sentiment distribution**")
    lines.append("")
    lines.append(_format_counts(sentiment_counts))
    lines.append("")
    lines.append("**Channel distribution**")
    lines.append("")
    lines.append(_format_counts(channel_counts))
    lines.append("")
    lines.append("**Priority distribution**")
    lines.append("")
    lines.append(_format_counts(priority_counts))
    lines.append("")
    lines.append("**Top product areas by volume**")
    lines.append("")
    lines.append(_format_counts(area_counts.head(5)))
    lines.append("")

    lines.append("## Main Negative Feedback Areas")
    lines.append("")
    lines.append(
        f"Of {total_negatives} negative feedback items, the following product areas "
        "account for the largest share:"
    )
    lines.append("")
    for area, count in top_neg_areas.items():
        share = (count / total_negatives * 100) if total_negatives else 0
        lines.append(f"- **{area}**: {count} negatives ({share:.1f}% of all negatives)")
    lines.append("")

    lines.append("## Channels with Highest Negative Feedback Concentration")
    lines.append("")
    lines.append(
        "Negative rate is computed as negative feedback divided by total feedback per "
        "channel, ranked from highest to lowest concentration."
    )
    lines.append("")
    for channel, rate in channel_neg_rate.items():
        total_ch = int(total_by_channel.get(channel, 0))
        neg_ch = int(neg_by_channel.get(channel, 0))
        lines.append(f"- **{channel}**: {rate:.1f}% negative ({neg_ch} of {total_ch})")
    lines.append("")

    lines.append("## Average Response Time by Priority")
    lines.append("")
    for priority in PRIORITY_ORDER:
        value = avg_response_by_priority.get(priority, float("nan"))
        lines.append(f"- **{priority}**: {value:.1f} hours")
    lines.append("")

    lines.append("## Resolution Rate by Priority")
    lines.append("")
    for priority in PRIORITY_ORDER:
        value = resolution_by_priority.get(priority, float("nan"))
        lines.append(f"- **{priority}**: {value:.1f}%")
    lines.append("")

    lines.append("## Initial Business Insights")
    lines.append("")
    lines.append(
        f"- Negative feedback concentrates most heavily in the **{top_channel_negative}** "
        f"channel, where {top_channel_rate:.1f}% of items are negative — a priority "
        "channel for service quality intervention."
    )
    lines.append(
        f"- The **{top_area_negative}** product area dominates negative feedback, making "
        "it a strong candidate for product and engineering focus."
    )
    if escalates:
        lines.append(
            f"- Response times escalate with severity: critical issues average "
            f"{critical_avg:.1f} hours versus {low_avg:.1f} hours for low-priority items, "
            "indicating that the most severe cases also wait the longest."
        )
    else:
        lines.append(
            f"- Critical issues average {critical_avg:.1f} hours to first response, "
            f"compared with {low_avg:.1f} hours for low-priority items."
        )
    lines.append(
        f"- There are **{unresolved_critical} unresolved critical** issues — a small but "
        "high-risk backlog that warrants immediate operational attention."
    )
    lines.append("")

    lines.append("## Suggested Next Analytical Steps")
    lines.append("")
    lines.append(
        "- Build TF-IDF features and classical sentiment classification baselines."
    )
    lines.append(
        "- Apply topic modeling within product areas to surface recurring themes."
    )
    lines.append(
        "- Produce segment and plan cross-tabs to compare feedback patterns across "
        "customer tiers."
    )
    lines.append(
        "- Add time-series trend analysis to track sentiment and volume over months."
    )
    lines.append(
        "- Extend executive reporting with automated, stakeholder-ready summaries."
    )
    lines.append("")

    lines.append("## Generated Artifacts")
    lines.append("")
    lines.append("- `data/processed/customer_feedback_clean.csv`")
    lines.append("- `visuals/sentiment_distribution.png`")
    lines.append("- `visuals/feedback_by_channel.png`")
    lines.append("- `visuals/feedback_by_area.png`")
    lines.append("- `visuals/negative_feedback_by_area.png`")
    lines.append("- `visuals/avg_response_time_by_priority.png`")
    lines.append("- `visuals/resolution_rate_by_priority.png`")
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_clean_data(str(CLEAN_PATH))

    create_sentiment_distribution_chart(df)
    create_feedback_by_channel_chart(df)
    create_feedback_by_area_chart(df)
    create_negative_feedback_by_area_chart(df)
    create_avg_response_time_by_priority_chart(df)
    create_resolution_rate_by_priority_chart(df)

    generate_eda_summary(df)

    print("EDA charts saved to: visuals/")
    print("EDA report saved to: reports/eda_summary.md")


if __name__ == "__main__":
    main()
