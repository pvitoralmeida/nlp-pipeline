from nlp_pipeline.preprocessing.cleaners import *
from nlp_pipeline.preprocessing.linguistic import *
from nlp_pipeline.preprocessing.semantics import *


class Preprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.nlp = None

    def fit(self, X, y=None):
        if self._needs_spacy():
            self.nlp = load_nlp(self.config.get("spacy_model", "pt_core_news_sm"))
        return self

    def transform(self, X):
        return [self._process_text(text) for text in X]

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def _needs_spacy(self):
        return any([
            self.config.get("remove_stopwords"),
            self.config.get("normalization"),
            self.config.get("handle_negations")
        ])

    def _process_text(self, text: str) -> str:

        if not isinstance(text, str):
            text = "" if text is None else str(text)
            
        # CLEANING
        if self.config.get("lowercase"):
            text = to_lowercase(text)

        if self.config.get("remove_urls"):
            text = remove_urls(text)

        if self.config.get("normalize_emojis"):
            text = normalize_emojis(text)

        if self.config.get("remove_punctuation"):
            text = remove_punctuation(text)

        # LINGUISTIC
        if self._needs_spacy():
            doc = self.nlp(text)
            tokens = doc_to_tokens(doc)

            tokens = remove_stopwords(tokens, self.config.get("remove_stopwords"))

            if self.config.get("handle_negations"):
                tokens = handle_negations([t.text for t in tokens])
            else:
                tokens = [t.text for t in tokens]

            if self.config.get("normalization") == "lemmatization":
                doc = self.nlp(" ".join(tokens))
                tokens = lemmatize(doc)

            elif self.config.get("normalization") == "stemming":
                tokens = stem(tokens)

            return " ".join(tokens)

        return text