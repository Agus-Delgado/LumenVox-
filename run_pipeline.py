from src.generate_dataset import main as generate_dataset
from src.preprocess_text import main as preprocess_text
from src.eda_analysis import main as run_eda
from src.train_sentiment_model import main as train_sentiment_model
from src.topic_insights import main as generate_topic_insights
from src.generate_executive_report import main as generate_executive_report


def main():
    print("LumenVox pipeline — Step 1: Dataset generation")
    generate_dataset()

    print("LumenVox pipeline — Step 2: Text preprocessing")
    preprocess_text()

    print("LumenVox pipeline — Step 3: Exploratory data analysis")
    run_eda()

    print("LumenVox pipeline — Step 4: Sentiment modeling")
    train_sentiment_model()

    print("LumenVox pipeline — Step 5: Product area and topic insights")
    generate_topic_insights()

    print("LumenVox pipeline — Step 6: Executive reporting")
    generate_executive_report()


if __name__ == "__main__":
    main()
