import nltk
import re
from nltk.stem import WordNetLemmatizer

# you need to download this first
# nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence.lower())

def lemmatization(word):
    return lemmatizer.lemmatize(word, pos='v')

def trim(word):
    return re.sub(r'(.)\1{2,}', r'\1\1', word) # removes any instances of repeated letters over 2 repetitions


def preprocess(token):
    token = token.lower()
    trimmed_token = trim(token)

    lemma_token = lemmatization(trimmed_token)
    
    return lemma_token