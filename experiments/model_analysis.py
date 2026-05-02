import pandas as pd

from nlp_pipeline.experiments import ExperimentRunner
from config.configs import default_config


def load_imdb():
    df = pd.read_csv("data/IMDB-Dataset.csv")
    df = df.sample(500, random_state=42)

    X = df["review"].fillna("").astype(str)
    y = df["sentiment"].map({"positive": 1, "negative": 0})

    return X, y


def build_config_space():
    return {
        "preprocessing": [
            # 🔹 baseline
            {"remove_stopwords": False, "normalization": None},

            # 🔹 stopwords
            {"remove_stopwords": True, "normalization": None},

            # 🔹 manter negações
            {"remove_stopwords": "keep_negations", "handle_negations": True},

            # 🔹 lematização
            {"remove_stopwords": True, "normalization": "lemmatization"},

            # 🔹 agressivo
            {
                "remove_stopwords": True,
                "normalization": "lemmatization",
                "handle_negations": True,
                "normalize_emojis": True
            },
        ],

        "vectorization": [
            # 🔹 unigram
            {"method": "tfidf", "ngram_range": (1, 1), "max_features": 5000},

            # 🔹 bigram
            {"method": "tfidf", "ngram_range": (1, 2), "max_features": 10000},

            # 🔹 trigram leve
            {"method": "tfidf", "ngram_range": (1, 3), "max_features": 15000},

            # 🔹 sublinear tf
            {
                "method": "tfidf",
                "ngram_range": (1, 2),
                "sublinear_tf": True,
                "max_features": 10000
            },

            # 🔹 redução de dimensionalidade
            {"method": "tfidf_svd", "svd_dim": 100},
            {"method": "tfidf_svd", "svd_dim": 300},
        ],

        "model": [
            # 🔹 regressão logística
            {
                "model": "logreg",
                "search": "manual",
                "params": {"C": 1.0}
            },

            # 🔹 SVM linear
            {
                "model": "svm",
                "search": "manual"
            },

            # 🔹 Naive Bayes
            {
                "model": "nb",
                "search": "manual"
            },

            # 🔹 Random Forest
            {
                "model": "rf",
                "search": "manual",
                "params": {"n_estimators": 100}
            },

            # 🔹 LightGBM
            {
                "model": "lgbm",
                "search": "manual",
                "params": {"n_estimators": 100}
            },
        ]
    }


def run():
    print("\nCarregando dataset IMDB...")
    X, y = load_imdb()

    print(f"Total de amostras: {len(X)}")

    config_space = build_config_space()

    runner = ExperimentRunner(
        base_config=default_config,
        config_space=config_space,
        strategy="random",   
        n_samples=10         
    )

    print("\nIniciando experimentos...")
    results = runner.run(X, y)

    print("\nExperimentos finalizados!")
    print("\nMelhores resultados:")
    print(results.head(10))


if __name__ == "__main__":
    run()