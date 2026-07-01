"""Text preprocessing and feature engineering for LumenVox customer feedback."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_COLUMNS = [
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
]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "customer_feedback_raw.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "customer_feedback_clean.csv"

DERIVED_COLUMNS = [
    "clean_message",
    "message_length",
    "word_count",
    "month",
    "is_negative",
    "is_positive",
    "is_critical",
    "is_high_priority",
    "is_unresolved",
    "is_unresolved_critical",
]

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
# Keep letters (incl. accented Spanish characters), digits and whitespace only.
_NON_WORD_PATTERN = re.compile(r"[^0-9a-záéíóúüñ\s]", flags=re.IGNORECASE)
_WHITESPACE_PATTERN = re.compile(r"\s+")

_TRUE_STRINGS = {"true", "1", "yes", "y", "t"}


def _coerce_resolved(series: pd.Series) -> pd.Series:
    """Safely coerce a resolved column to boolean regardless of source dtype."""
    if series.dtype == bool:
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(_TRUE_STRINGS)
    )


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw feedback CSV, parsing dates and coercing resolved to bool."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["resolved"] = _coerce_resolved(df["resolved"])
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if any expected columns are missing."""
    missing = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")


def clean_text(text: str) -> str:
    """Normalize a feedback message for downstream analysis.

    Lowercases, strips URLs, drops non-letter/non-digit characters (while
    preserving Spanish and English letters), and collapses whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_PATTERN.sub(" ", text)
    text = _NON_WORD_PATTERN.sub(" ", text)
    text = _WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def preprocess_feedback(df: pd.DataFrame) -> pd.DataFrame:
    """Add the cleaned message and derived analytical features."""
    df = df.copy()

    df["clean_message"] = df["message"].apply(clean_text)
    df["message_length"] = df["clean_message"].str.len()
    df["word_count"] = (
        df["clean_message"].str.split().apply(lambda tokens: len(tokens) if tokens else 0)
    )
    df["month"] = df["date"].dt.month.astype(int)

    df["is_negative"] = df["sentiment_label"] == "negative"
    df["is_positive"] = df["sentiment_label"] == "positive"
    df["is_critical"] = df["priority"] == "critical"
    df["is_high_priority"] = df["priority"] == "high"
    df["is_unresolved"] = df["resolved"] == False  # noqa: E712 - explicit bool compare
    df["is_unresolved_critical"] = df["is_critical"] & df["is_unresolved"]

    return df


def save_processed_data(df: pd.DataFrame, path: str) -> None:
    """Persist the processed dataset, creating parent directories as needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    df = load_raw_data(str(RAW_PATH))
    validate_schema(df)
    df = preprocess_feedback(df)
    save_processed_data(df, str(PROCESSED_PATH))

    print(f"Row count: {len(df)}")
    print(f"Column count: {df.shape[1]}")
    print("New columns created:")
    for column in DERIVED_COLUMNS:
        print(f"  - {column}")
    print(f"\nOutput path: data/processed/customer_feedback_clean.csv")


if __name__ == "__main__":
    main()
