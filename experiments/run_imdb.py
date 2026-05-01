import pandas as pd
from nlp_pipeline.pipeline.pipeline import NLPPipeline
from config.configs import default_config
from experiments.experiment_utils import split_data, evaluate


def run():
    df = pd.read_csv("data/IMDB-Dataset.csv")

    df = df.sample(5000)

    X = df["review"]
    y = df["sentiment"].map({"positive": 1, "negative": 0})

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    config = default_config.copy()
    config["preprocessing"]["spacy_model"] = "en_core_web_sm"

    pipeline = NLPPipeline(config)

    print("Treinando modelo...")
    pipeline.fit(X_train, y_train)

    print("\nValidação:")
    y_pred_val = pipeline.predict(X_val)
    val_f1 = evaluate(y_val, y_pred_val)
    print("F1 (val):", val_f1)

    print("\nTeste final:")
    y_pred_test = pipeline.predict(X_test)
    test_f1 = evaluate(y_test, y_pred_test)
    print("F1 (test):", test_f1)


if __name__ == "__main__":
    run()