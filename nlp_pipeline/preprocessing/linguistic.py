import spacy
from nltk.stem import SnowballStemmer


def load_nlp(model="pt_core_news_sm"):
    return spacy.load(model)


def doc_to_tokens(doc):
    return [t for t in doc if not t.is_space]


def remove_stopwords(tokens, mode="default"):
    keep_neg = {"não", "nunca", "jamais", "nem"}

    filtered = []

    for t in tokens:
        if mode == "keep_negations":
            if t.text in keep_neg or not t.is_stop:
                filtered.append(t)
        elif mode is True:
            if not t.is_stop:
                filtered.append(t)
        else:
            filtered.append(t)

    return filtered


def lemmatize(tokens):
    return [t.lemma_ for t in tokens]


stemmer = SnowballStemmer("portuguese")


def stem(tokens):
    return [stemmer.stem(t) for t in tokens]