from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np

from gensim.models import KeyedVectors
from gensim.models import Word2Vec


class Vectorizer:
    def __init__(self, config: dict):
        self.config = config
        self.method = config["method"]

        self.vectorizer = None
        self.reducer = None
        self.w2v_model = None

    # ---------------------------
    # FIT
    # ---------------------------
    def fit(self, X):
        if self.method == "bow":
            return self._fit_bow(X)

        elif self.method == "tfidf":
            return self._fit_tfidf(X)

        elif self.method == "tfidf_svd":
            return self._fit_tfidf_svd(X)

        elif self.method == "word2vec":
            return self._fit_word2vec(X)

        else:
            raise ValueError(f"Método inválido: {self.method}")

    # ---------------------------
    # TRANSFORM
    # ---------------------------
    def transform(self, X):
        if self.method in ["bow", "tfidf"]:
            return self.vectorizer.transform(X)

        elif self.method == "tfidf_svd":
            X_vec = self.vectorizer.transform(X)
            return self.reducer.transform(X_vec)

        elif self.method == "word2vec":
            return self._transform_word2vec(X)

    # ---------------------------
    # BOw
    # ---------------------------
    def _fit_bow(self, X):
        self.vectorizer = CountVectorizer(
            binary=self.config.get("binary", False),
            max_features=self.config.get("max_features")
        )
        return self.vectorizer.fit_transform(X)

    # ---------------------------
    # TF-IDF
    # ---------------------------
    def _fit_tfidf(self, X):
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=self.config.get("sublinear_tf", False),
            ngram_range=self.config.get("ngram_range", (1, 1)),
            max_features=self.config.get("max_features"),
            norm=self.config.get("norm", "l2")
        )
        return self.vectorizer.fit_transform(X)

    # ---------------------------
    # TF-IDF + SVD
    # ---------------------------
    def _fit_tfidf_svd(self, X):
        X_vec = self._fit_tfidf(X)

        self.reducer = TruncatedSVD(
            n_components=self.config.get("svd_dim", 100),
            random_state=42
        )

        return self.reducer.fit_transform(X_vec)

    # ---------------------------
    # WORD2VEC
    # ---------------------------
    def _fit_word2vec(self, X):
        tokens = [text.split() for text in X]

        if self.config.get("pretrained_path"):
            self.w2v_model = KeyedVectors.load_word2vec_format(
                self.config["pretrained_path"],
                binary=False
            )
        else:
            self.w2v_model = Word2Vec(
                sentences=tokens,
                vector_size=self.config.get("vector_size", 100),
                window=5,
                min_count=2,
                workers=4
            ).wv

        return self._transform_word2vec(X)

    def _transform_word2vec(self, X):
        tokens = [text.split() for text in X]

        vectors = []

        for sentence in tokens:
            word_vecs = []

            for word in sentence:
                if word in self.w2v_model:
                    word_vecs.append(self.w2v_model[word])

            if word_vecs:
                vectors.append(np.mean(word_vecs, axis=0))
            else:
                vectors.append(np.zeros(self.w2v_model.vector_size))

        return np.array(vectors)