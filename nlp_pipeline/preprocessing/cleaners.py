import re
import string
from textacy.preprocessing import replace, remove


def to_lowercase(text: str) -> str:
    return text.casefold()


def remove_urls(text: str) -> str:
    return replace.urls(text, "")


def remove_punctuation(text: str) -> str:
    return remove.punctuation(text)