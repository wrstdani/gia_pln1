import nltk
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

import spacy

# Convertir tag generada por NLTK en una WN tag
def tag2wn(tag: str) -> str | None:
    if tag.lower().startswith("n"):
        return wn.NOUN
    elif tag.lower().startswith("v"):
        return wn.VERB
    elif tag.lower().startswith("j"):
        return wn.ADJ
    elif tag.lower().startswith("r"):
        return wn.ADV
    else:
        return None

# Preprocesado básico con NLTK
def preprocess_nltk(text: str, lem: bool = False, sents: bool = False, lang: str = "eng", sw: set = {}) -> list[str] | list[list[str]] | list[tuple[str, str]] | list[list[tuple[str, str]]]:
    if not sents:
        tokenized = regexp_tokenize(text, r"\w+")

        if lem:
            tagged = nltk.pos_tag(tokenized, lang=lang)
            tokenized = [WordNetLemmatizer().lemmatize(w.lower(), t) for w, t in tagged if w.lower() not in sw]

    else:
        tokenized = PunktSentenceTokenizer().tokenize(text)
        tokenized = [preprocess_nltk(s, lem, False) for s in tokenized]

    return tokenized
