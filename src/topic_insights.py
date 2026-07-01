"""Product area and topic insights for LumenVox customer feedback."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "customer_feedback_clean.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
VISUALS_DIR = PROJECT_ROOT / "visuals"

TOPIC_REPORT_PATH = REPORTS_DIR / "topic_insights.md"
PRODUCT_AREA_SUMMARY_PATH = REPORTS_DIR / "product_area_summary.csv"
HIGH_RISK_FEEDBACK_PATH = REPORTS_DIR / "high_risk_feedback.csv"

PRIORITY_ORDER = ["low", "medium", "high", "critical"]

REQUIRED_COLUMNS = [
    "feedback_id",
    "date",
    "channel",
    "customer_segment",
    "plan_type",
    "language",
    "message",
    "rating",
    "product_area",
    "sentiment_label",
    "priority",
    "resolved",
    "response_time_hours",
    "is_negative",
    "is_critical",
    "is_unresolved",
    "is_unresolved_critical",
]

BOOLEAN_COLUMNS = [
    "resolved",
    "is_negative",
    "is_critical",
    "is_unresolved",
    "is_unresolved_critical",
]

_TRUE_STRINGS = {"true", "1", "yes", "y", "t"}
_FALSE_STRINGS = {"false", "0", "no", "n", "f"}

HIGH_RISK_COLUMNS = [
    "feedback_id",
    "date",
    "channel",
    "customer_segment",
    "plan_type",
    "language",
    "product_area",
    "priority",
    "rating",
    "resolved",
    "response_time_hours",
    "message",
]

SUMMARY_TABLE_COLUMNS = [
    "product_area",
    "total_feedback",
    "negative_feedback",
    "negative_rate",
    "unresolved_critical_feedback",
    "avg_response_time_hours",
    "resolution_rate",
]

MESSAGE_TRUNCATE_LENGTH = 200


def _coerce_boolean(series: pd.Series) -> pd.Series:
    """Safely coerce a column to boolean without treating invalid values as True."""
    if series.dtype == bool:
        return series

    normalized = series.astype(str).str.strip().str.lower()
    result = pd.Series(False, index=series.index, dtype=bool)
    result[normalized.isin(_TRUE_STRINGS)] = True
    result[normalized.isin(_FALSE_STRINGS)] = False
    return result


def _save_figure(fig: plt.Figure, filename: str) -> None:
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _show_no_data(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _truncate_message(message: str, max_length: int = MESSAGE_TRUNCATE_LENGTH) -> str:
    text = str(message).replace("\n", " ").strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def load_clean_data(path: Path) -> pd.DataFrame:
    """Load and validate the processed feedback dataset."""
    df = pd.read_csv(path, parse_dates=["date"])

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for column in BOOLEAN_COLUMNS:
        df[column] = _coerce_boolean(df[column])

    return df


def build_product_area_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate risk and operational metrics by product area."""
    grouped = df.groupby("product_area", as_index=False).agg(
        total_feedback=("feedback_id", "count"),
        negative_feedback=("is_negative", "sum"),
        critical_feedback=("is_critical", "sum"),
        unresolved_feedback=("is_unresolved", "sum"),
        unresolved_critical_feedback=("is_unresolved_critical", "sum"),
        avg_rating=("rating", "mean"),
        avg_response_time_hours=("response_time_hours", "mean"),
        resolution_rate=("resolved", "mean"),
    )

    grouped["negative_rate"] = grouped["negative_feedback"] / grouped["total_feedback"]
    grouped = grouped.sort_values(
        by=["unresolved_critical_feedback", "negative_feedback", "avg_response_time_hours"],
        ascending=False,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(PRODUCT_AREA_SUMMARY_PATH, index=False)
    return grouped


def identify_high_risk_feedback(df: pd.DataFrame) -> pd.DataFrame:
    """Extract negative, high/critical, unresolved feedback for escalation."""
    mask = (
        (df["sentiment_label"] == "negative")
        & (df["priority"].isin(["high", "critical"]))
        & (df["resolved"] == False)  # noqa: E712
    )
    high_risk = df.loc[mask, HIGH_RISK_COLUMNS].copy()
    high_risk["priority"] = pd.Categorical(
        high_risk["priority"],
        categories=["critical", "high"],
        ordered=True,
    )
    high_risk = high_risk.sort_values(
        by=["priority", "response_time_hours"],
        ascending=[True, False],
    )
    high_risk["priority"] = high_risk["priority"].astype(str)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    high_risk.to_csv(HIGH_RISK_FEEDBACK_PATH, index=False)
    return high_risk


def create_negative_feedback_by_segment_chart(df: pd.DataFrame) -> None:
    """Bar chart of negative feedback count by customer segment."""
    negatives = df[df["is_negative"]]
    counts = negatives["customer_segment"].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    if counts.empty:
        _show_no_data(ax, "Negative Feedback by Customer Segment")
    else:
        ax.bar(counts.index, counts.values)
        ax.set_title("Negative Feedback by Customer Segment")
        ax.set_xlabel("Customer Segment")
        ax.set_ylabel("Number of Negative Feedback Items")
        ax.tick_params(axis="x", rotation=45)
        plt.setp(ax.get_xticklabels(), ha="right")

    _save_figure(fig, "negative_feedback_by_segment.png")


def create_unresolved_critical_by_area_chart(df: pd.DataFrame) -> None:
    """Bar chart of unresolved critical feedback count by product area."""
    unresolved_critical = df[df["is_unresolved_critical"]]
    counts = unresolved_critical["product_area"].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    if counts.empty:
        _show_no_data(ax, "Unresolved Critical Feedback by Product Area")
    else:
        ax.bar(counts.index, counts.values)
        ax.set_title("Unresolved Critical Feedback by Product Area")
        ax.set_xlabel("Product Area")
        ax.set_ylabel("Number of Unresolved Critical Items")
        ax.tick_params(axis="x", rotation=45)
        plt.setp(ax.get_xticklabels(), ha="right")

    _save_figure(fig, "unresolved_critical_by_area.png")


def create_negative_rate_by_channel_chart(df: pd.DataFrame) -> None:
    """Bar chart of negative rate by channel."""
    total_by_channel = df["channel"].value_counts()
    negative_by_channel = df[df["is_negative"]]["channel"].value_counts().reindex(
        total_by_channel.index,
        fill_value=0,
    )
    negative_rate = (negative_by_channel / total_by_channel * 100).sort_values(
        ascending=False,
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    if negative_rate.empty:
        _show_no_data(ax, "Negative Rate by Channel")
    else:
        ax.bar(negative_rate.index, negative_rate.values)
        ax.set_title("Negative Rate by Channel")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Negative Rate (%)")
        ax.tick_params(axis="x", rotation=45)
        plt.setp(ax.get_xticklabels(), ha="right")

    _save_figure(fig, "negative_rate_by_channel.png")


def create_avg_response_time_by_area_chart(df: pd.DataFrame) -> None:
    """Bar chart of average response time by product area."""
    means = (
        df.groupby("product_area")["response_time_hours"]
        .mean()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    if means.empty:
        _show_no_data(ax, "Average Response Time by Product Area")
    else:
        ax.bar(means.index, means.values)
        ax.set_title("Average Response Time by Product Area")
        ax.set_xlabel("Product Area")
        ax.set_ylabel("Average Response Time (hours)")
        ax.tick_params(axis="x", rotation=45)
        plt.setp(ax.get_xticklabels(), ha="right")

    _save_figure(fig, "avg_response_time_by_area.png")


def _format_rate(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_summary_table(product_area_summary: pd.DataFrame) -> str:
    lines = [
        "| product_area | total_feedback | negative_feedback | negative_rate | "
        "unresolved_critical_feedback | avg_response_time_hours | resolution_rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in product_area_summary.iterrows():
        lines.append(
            f"| {row['product_area']} | {int(row['total_feedback'])} | "
            f"{int(row['negative_feedback'])} | {_format_rate(row['negative_rate'])} | "
            f"{int(row['unresolved_critical_feedback'])} | "
            f"{row['avg_response_time_hours']:.1f} | "
            f"{_format_rate(row['resolution_rate'])} |"
        )
    return "\n".join(lines)


def generate_topic_insights_report(
    df: pd.DataFrame,
    product_area_summary: pd.DataFrame,
    high_risk_feedback: pd.DataFrame,
) -> None:
    """Write the business-oriented product area and topic insights report."""
    high_risk_count = len(high_risk_feedback)
    total_unresolved_critical = int(df["is_unresolved_critical"].sum())

    top_unresolved_area_row = product_area_summary.iloc[0]
    top_unresolved_area = top_unresolved_area_row["product_area"]
    top_unresolved_count = int(top_unresolved_area_row["unresolved_critical_feedback"])

    total_by_channel = df["channel"].value_counts()
    negative_by_channel = df[df["is_negative"]]["channel"].value_counts().reindex(
        total_by_channel.index,
        fill_value=0,
    )
    channel_negative_rate = (negative_by_channel / total_by_channel).sort_values(
        ascending=False,
    )
    top_channel = channel_negative_rate.index[0]
    top_channel_rate = channel_negative_rate.iloc[0]

    worst_response_row = product_area_summary.sort_values(
        "avg_response_time_hours",
        ascending=False,
    ).iloc[0]
    worst_response_area = worst_response_row["product_area"]
    worst_response_time = worst_response_row["avg_response_time_hours"]

    negatives = df[df["is_negative"]]
    segment_counts = negatives["customer_segment"].value_counts()
    total_negatives = len(negatives)

    lines: list[str] = []
    lines.append("# LumenVox — Product Area & Topic Insights Report")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append(
        "This report identifies where negative, critical, and unresolved feedback is "
        "concentrated across product areas, channels, and customer segments so product, "
        "support, and operations teams can prioritize action on the highest-risk issues."
    )
    lines.append("")

    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        f"- **{high_risk_count} high-risk feedback items** are negative, high or "
        "critical priority, and still unresolved."
    )
    lines.append(
        f"- **{top_unresolved_area}** leads unresolved critical backlog with "
        f"**{top_unresolved_count}** open critical items."
    )
    lines.append(
        f"- **{top_channel}** has the highest negative rate at "
        f"**{top_channel_rate * 100:.1f}%** of all feedback on that channel."
    )
    lines.append(
        f"- **{worst_response_area}** shows the slowest average response time at "
        f"**{worst_response_time:.1f} hours**."
    )
    lines.append(
        f"- **{total_unresolved_critical} unresolved critical** cases remain in the "
        "overall backlog and require escalation attention."
    )
    lines.append("")

    lines.append("## Product Area Risk Summary")
    lines.append("")
    lines.append(
        "Product areas ranked by unresolved critical feedback, negative volume, and "
        "average response time."
    )
    lines.append("")
    lines.append(_format_summary_table(product_area_summary[SUMMARY_TABLE_COLUMNS]))
    lines.append("")

    lines.append("## Highest Risk Feedback")
    lines.append("")
    lines.append(f"- **High-risk record count:** {high_risk_count}")
    lines.append("")
    lines.append("Top 5 high-risk feedback examples:")
    lines.append("")
    for _, row in high_risk_feedback.head(5).iterrows():
        lines.append(
            f"- **{row['feedback_id']}** — `{row['product_area']}` | "
            f"priority: **{row['priority']}** | channel: **{row['channel']}** | "
            f"segment: **{row['customer_segment']}** | "
            f"response time: **{row['response_time_hours']:.1f} hours**"
        )
        lines.append(f"  - Message: {_truncate_message(row['message'])}")
    if high_risk_count == 0:
        lines.append("- No high-risk feedback records matched the filter criteria.")
    lines.append("")

    lines.append("## Negative Feedback by Segment")
    lines.append("")
    lines.append(
        "Customer segments ranked by negative feedback volume and share of all "
        "negative items."
    )
    lines.append("")
    for segment, count in segment_counts.items():
        share = (count / total_negatives * 100) if total_negatives else 0.0
        lines.append(f"- **{segment}**: {count} negatives ({share:.1f}% of negatives)")
    if segment_counts.empty:
        lines.append("- No negative feedback records found.")
    lines.append("")

    lines.append("## Channel Risk")
    lines.append("")
    lines.append("Channels ranked by negative rate (negative feedback / total feedback).")
    lines.append("")
    for channel, rate in channel_negative_rate.items():
        total_ch = int(total_by_channel.get(channel, 0))
        neg_ch = int(negative_by_channel.get(channel, 0))
        lines.append(
            f"- **{channel}**: {rate * 100:.1f}% negative ({neg_ch} of {total_ch})"
        )
    lines.append("")

    slow_areas = product_area_summary[
        product_area_summary["unresolved_critical_feedback"] > 0
    ].sort_values("avg_response_time_hours", ascending=False)
    slow_area_name = (
        slow_areas.iloc[0]["product_area"] if not slow_areas.empty else worst_response_area
    )
    slow_area_hours = (
        slow_areas.iloc[0]["avg_response_time_hours"]
        if not slow_areas.empty
        else worst_response_time
    )

    lines.append("## Operational Risks")
    lines.append("")
    lines.append(
        f"- **Unresolved critical backlog:** {total_unresolved_critical} critical items "
        "remain open, with concentration in areas such as "
        f"**{top_unresolved_area}** ({top_unresolved_count} items)."
    )
    lines.append(
        f"- **Slow response in high-risk areas:** {slow_area_name} averages "
        f"{slow_area_hours:.1f} hours among areas with unresolved critical feedback, "
        "delaying resolution for the most severe issues."
    )
    lines.append(
        f"- **Customer trust risk:** {high_risk_count} negative high-priority items are "
        "still unresolved, increasing churn risk for affected accounts."
    )
    lines.append(
        "- **Escalation need:** Critical unresolved feedback in billing, account access, "
        "performance, and integrations should trigger immediate product and support "
        "escalation workflows."
    )
    lines.append("")

    lines.append("## Recommended Actions")
    lines.append("")
    lines.append(
        "- Prioritize unresolved critical issues in account access, billing, performance, "
        "and integrations."
    )
    lines.append("- Create escalation rules for critical unresolved feedback.")
    lines.append(
        f"- Review support workflows for **{top_channel}**, the channel with the highest "
        "negative concentration."
    )
    lines.append("- Track negative feedback rate weekly by product area.")
    lines.append(
        "- Use high-risk feedback samples for qualitative review and root-cause analysis."
    )
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append("- The dataset is synthetic and designed for portfolio demonstration.")
    lines.append("- Product areas are generated labels, not validated taxonomies.")
    lines.append(
        "- A real-world implementation would require human validation and taxonomy "
        "refinement before executive action."
    )
    lines.append("")

    lines.append("## Generated Artifacts")
    lines.append("")
    lines.append("- `reports/product_area_summary.csv`")
    lines.append("- `reports/high_risk_feedback.csv`")
    lines.append("- `reports/topic_insights.md`")
    lines.append("- `visuals/negative_feedback_by_segment.png`")
    lines.append("- `visuals/unresolved_critical_by_area.png`")
    lines.append("- `visuals/negative_rate_by_channel.png`")
    lines.append("- `visuals/avg_response_time_by_area.png`")
    lines.append("")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TOPIC_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_clean_data(CLEAN_PATH)
    product_area_summary = build_product_area_summary(df)
    high_risk_feedback = identify_high_risk_feedback(df)

    create_negative_feedback_by_segment_chart(df)
    create_unresolved_critical_by_area_chart(df)
    create_negative_rate_by_channel_chart(df)
    create_avg_response_time_by_area_chart(df)

    generate_topic_insights_report(df, product_area_summary, high_risk_feedback)

    print("Topic insights report saved to: reports/topic_insights.md")
    print("Product area summary saved to: reports/product_area_summary.csv")
    print("High-risk feedback saved to: reports/high_risk_feedback.csv")
    print("Topic insight charts saved to: visuals/")


if __name__ == "__main__":
    main()
