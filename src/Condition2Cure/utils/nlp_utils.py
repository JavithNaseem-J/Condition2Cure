import re
import spacy

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def clean_text(text: str) -> str:
    """
    Clean input review text: lowercase, remove HTML tags and non-alphabetic characters,
    remove stopwords, and lemmatize using SpaCy.
    """
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    doc = nlp(text)

    words = [token.lemma_ for token in doc if not token.is_stop]

    return ' '.join(words)