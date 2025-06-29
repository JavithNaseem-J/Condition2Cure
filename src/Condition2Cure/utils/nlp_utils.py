import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """
    Clean input review text: lowercase, remove punctuation, stopwords, lemmatize.
    """
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)    
    text = re.sub(r'[^a-zA-Z\s]', '', text)        
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)
