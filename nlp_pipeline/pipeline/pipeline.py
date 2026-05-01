from nlp_pipeline.preprocessing.preprocessor import Preprocessor
from nlp_pipeline.vectorization.vectorizer import Vectorizer
from nlp_pipeline.models.classifier import Classifier


class NLPPipeline:
    def __init__(self, config):
        self.preprocessor = Preprocessor(config["preprocessing"])
        self.vectorizer = Vectorizer(config["vectorization"])
        self.classifier = Classifier(config["model"])

    def fit(self, X, y):
        X_clean = self.preprocessor.fit_transform(X)
        X_vec = self.vectorizer.fit(X_clean)
        self.classifier.fit(X_vec, y)

    def predict(self, X):
        X_clean = self.preprocessor.transform(X)
        X_vec = self.vectorizer.transform(X_clean)
        return self.classifier.predict(X_vec)