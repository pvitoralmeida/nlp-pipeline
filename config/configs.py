default_config = {
    "preprocessing": {
        "lowercase": True,
        "remove_stopwords": False,
        "normalization": None,
        "handle_negations": False,
        "remove_urls": True,
        "normalize_emojis": False,
        "spacy_model": "en_core_web_sm"
    },

    "vectorization": {
        "method": "tfidf",
        "ngram_range": (1, 1),
        "max_features": 5000,
        "sublinear_tf": False,
        "norm": "l2",
        "svd_dim": 100
    },

    "model": {
        "model": "logreg",
        "search": "manual",
        "params": {
            "C": 1.0
        }
    }
}