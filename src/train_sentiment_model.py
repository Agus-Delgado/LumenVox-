"""Train and evaluate classical NLP sentiment classifiers for LumenVox."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "customer_feedback_clean.csv"
MODELS_DIR = PROJECT_ROOT / "models"
VISUALS_DIR = PROJECT_ROOT / "visuals"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_REPORT_PATH = REPORTS_DIR / "model_metrics.md"
COMPARISON_CSV_PATH = REPORTS_DIR / "sentiment_model_comparison.csv"
MODEL_PATH = MODELS_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
TEXT_COLUMN = "clean_message"
TARGET_COLUMN = "sentiment_label"
TEST_SIZE = 0.2
RANDOM_STATE = 42

CONFUSION_MATRIX_FILES = {
    "Logistic Regression": "confusion_matrix_logistic_regression.png",
    "LinearSVC": "confusion_matrix_linear_svc.png",
    "Multinomial Naive Bayes": "confusion_matrix_naive_bayes.png",
}

METRIC_COLUMNS = [
    "model_name",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "negative_precision",
    "negative_recall",
    "negative_f1",
]


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the cleaned feedback dataset."""
    df = pd.read_csv(path)

    missing = {TEXT_COLUMN, TARGET_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
    df = df[df[TEXT_COLUMN] != ""]
    df = df[df[TARGET_COLUMN].notna()]
    df = df[df[TARGET_COLUMN].isin(SENTIMENT_LABELS)]

    return df.reset_index(drop=True)


def prepare_data(df: pd.DataFrame):
    """Split features and target into stratified train/test sets."""
    X = df[TEXT_COLUMN]
    y = df[TARGET_COLUMN]
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_vectorizer() -> TfidfVectorizer:
    """Return a configured TF-IDF vectorizer."""
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2,
        max_df=0.95,
    )


def get_models() -> dict:
    """Return the sentiment classifiers to train and compare."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "LinearSVC": LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Multinomial Naive Bayes": MultinomialNB(),
    }


def extract_metrics(y_true, y_pred, model_name: str) -> dict:
    """Compute classification metrics for a single model."""
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=SENTIMENT_LABELS,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "negative_precision": precision[0],
        "negative_recall": recall[0],
        "negative_f1": f1[0],
    }


def save_confusion_matrix(
    y_true,
    y_pred,
    labels: list[str],
    model_name: str,
    output_path: Path,
) -> None:
    """Save a confusion matrix figure for one model."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels,
        display_labels=labels,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_and_evaluate_models(X_train, X_test, y_train, y_test, vectorizer) -> tuple:
    """Fit vectorizer and models, evaluate on the test set, and save confusion matrices."""
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    results: list[dict] = []
    for model_name, model in get_models().items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        result = extract_metrics(y_test, y_pred, model_name)
        result["model"] = model
        result["classification_report"] = classification_report(
            y_test,
            y_pred,
            labels=SENTIMENT_LABELS,
            zero_division=0,
        )
        save_confusion_matrix(
            y_test,
            y_pred,
            SENTIMENT_LABELS,
            model_name,
            VISUALS_DIR / CONFUSION_MATRIX_FILES[model_name],
        )
        results.append(result)

    return results, vectorizer


def select_best_model(results: list[dict]) -> dict:
    """Select the model with highest negative recall, then macro F1 as tie-breaker."""
    return max(results, key=lambda r: (r["negative_recall"], r["macro_f1"]))


def save_model_artifacts(model, vectorizer, best_model_name: str) -> None:
    """Persist the best model, fitted vectorizer, and selection metadata."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    metadata = {
        "best_model_name": best_model_name,
        "selection_metric": "negative_recall",
        "tie_breaker": "macro_f1",
        "text_column": TEXT_COLUMN,
        "target_column": TARGET_COLUMN,
        "labels": SENTIMENT_LABELS,
        "vectorizer": {
            "type": "TfidfVectorizer",
            "ngram_range": [1, 2],
            "max_features": 5000,
            "min_df": 2,
            "max_df": 0.95,
        },
    }
    MODEL_METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def save_model_comparison(results: list[dict]) -> None:
    """Save model comparison metrics to CSV."""
    rows = [{key: result[key] for key in METRIC_COLUMNS} for result in results]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(COMPARISON_CSV_PATH, index=False)


def generate_model_metrics_report(
    results: list[dict],
    best_model_name: str,
    n_rows: int,
    train_size: int,
    test_size: int,
    labels: list[str],
) -> None:
    """Write the sentiment modeling business report."""
    lines: list[str] = []
    lines.append("# LumenVox — Sentiment Modeling Report")
    lines.append("")
    lines.append("## Modeling Objective")
    lines.append("")
    lines.append(
        "The goal of this block is to classify customer feedback sentiment "
        "(negative, neutral, or positive) from cleaned text using supervised "
        "machine learning models trained on TF-IDF features."
    )
    lines.append("")
    lines.append("## Business Priority")
    lines.append("")
    lines.append(
        "Negative recall is prioritized because missing negative feedback could "
        "delay action on customer pain points. A model that catches more true "
        "negatives helps teams respond sooner to issues that drive churn risk."
    )
    lines.append("")
    lines.append("## Dataset Used")
    lines.append("")
    lines.append(f"- **Rows used:** {n_rows}")
    lines.append(f"- **Train size:** {train_size}")
    lines.append(f"- **Test size:** {test_size}")
    lines.append(f"- **Target labels:** {', '.join(labels)}")
    lines.append(f"- **Text feature:** `{TEXT_COLUMN}`")
    lines.append("")
    lines.append("## Models Compared")
    lines.append("")
    lines.append("- Logistic Regression")
    lines.append("- LinearSVC")
    lines.append("- Multinomial Naive Bayes")
    lines.append("")
    lines.append("## Model Comparison")
    lines.append("")
    lines.append(
        "| model_name | accuracy | macro_f1 | weighted_f1 | "
        "negative_precision | negative_recall | negative_f1 |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for result in results:
        lines.append(
            f"| {result['model_name']} | {result['accuracy']:.4f} | "
            f"{result['macro_f1']:.4f} | {result['weighted_f1']:.4f} | "
            f"{result['negative_precision']:.4f} | {result['negative_recall']:.4f} | "
            f"{result['negative_f1']:.4f} |"
        )
    lines.append("")
    lines.append("## Best Model Selected")
    lines.append("")
    best = next(r for r in results if r["model_name"] == best_model_name)
    lines.append(
        f"**{best_model_name}** was selected because it achieved the highest "
        f"negative recall ({best['negative_recall']:.4f}) among the compared models. "
        "When models tie on negative recall, macro F1-score is used as the tie-breaker."
    )
    lines.append("")
    lines.append("## Detailed Classification Reports")
    lines.append("")
    for result in results:
        lines.append(f"### {result['model_name']}")
        lines.append("")
        lines.append("```")
        lines.append(result["classification_report"].rstrip())
        lines.append("```")
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- Automatically triage incoming feedback by sentiment so support and "
        "product teams can focus on negative items first."
    )
    lines.append(
        "- Track negative feedback volume over time as an early indicator of "
        "customer satisfaction trends."
    )
    lines.append(
        "- Route high-risk negative messages to escalation workflows before "
        "they become churn events."
    )
    lines.append(
        "- Compare model outputs across channels and segments to identify where "
        "dissatisfaction is concentrated."
    )
    lines.append(
        "- Provide a reproducible baseline for future topic and product-area "
        "classification work."
    )
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- The dataset is synthetic and designed for portfolio demonstration.")
    lines.append(
        "- Sentiment labels are generated programmatically, not manually annotated "
        "by human reviewers."
    )
    lines.append(
        "- Because the dataset is synthetic and label-consistent by design, model "
        "performance may be higher than expected on real-world customer feedback."
    )
    lines.append(
        "- Future work should validate model performance against real customer "
        "feedback with expert labeling."
    )
    lines.append(
        "- Classical NLP models may miss sarcasm, ambiguity, or context-dependent "
        "sentiment."
    )
    lines.append("")
    lines.append("## Next Steps")
    lines.append("")
    lines.append("- Product area classification using the same text features.")
    lines.append("- Topic analysis within negative feedback subsets.")
    lines.append("- Error analysis on misclassified feedback to guide improvements.")
    lines.append("- Human-in-the-loop validation on a sample of predictions.")
    lines.append(
        "- Optional GenAI-assisted executive summaries grounded in validated metrics."
    )
    lines.append("")
    lines.append("## Generated Artifacts")
    lines.append("")
    lines.append("- `models/sentiment_model.pkl`")
    lines.append("- `models/tfidf_vectorizer.pkl`")
    lines.append("- `models/model_metadata.json`")
    lines.append("- `reports/sentiment_model_comparison.csv`")
    lines.append("- `visuals/confusion_matrix_logistic_regression.png`")
    lines.append("- `visuals/confusion_matrix_linear_svc.png`")
    lines.append("- `visuals/confusion_matrix_naive_bayes.png`")
    lines.append("")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_dataset(CLEAN_PATH)
    n_rows = len(df)

    X_train, X_test, y_train, y_test = prepare_data(df)
    train_size = len(X_train)
    test_size = len(X_test)

    vectorizer = build_vectorizer()
    results, fitted_vectorizer = train_and_evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test,
        vectorizer,
    )

    best = select_best_model(results)
    best_model_name = best["model_name"]

    save_model_artifacts(best["model"], fitted_vectorizer, best_model_name)
    save_model_comparison(results)
    generate_model_metrics_report(
        results,
        best_model_name,
        n_rows,
        train_size,
        test_size,
        SENTIMENT_LABELS,
    )

    print(f"Best model: {best_model_name}")
    print(f"Negative recall: {best['negative_recall']:.4f}")
    print(f"Model path: models/sentiment_model.pkl")
    print(f"Vectorizer path: models/tfidf_vectorizer.pkl")
    print(f"Metrics report path: reports/model_metrics.md")


if __name__ == "__main__":
    main()
