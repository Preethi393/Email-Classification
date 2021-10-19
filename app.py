import streamlit as st
import pickle

import nltk

nltk.download('stopwords')
nltk.download(['punkt', 'wordnet'])
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def tokenize(text):
    # 1. Normalize the data by converting to lower case and removing punctuations
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())

    # 2. Tokenizing: split text into words
    tokens = word_tokenize(text)

    # 3. Remove stop words
    words = [w for w in tokens if w not in STOPWORDS]

    # 4. Lemmatize
    lemmed_words = [lemmatizer.lemmatize(w) for w in words]

    clean_tokens = []

    for i in lemmed_words:
        clean_tokens.append(i)

        ## back to string from list
    text = " ".join(clean_tokens)
    return text


tfidf = pickle.load(open('vectors.pkl', 'rb'))
model = pickle.load(open('Lin_model.pkl', 'rb'))

st.title("Email Classifier")

input_email = st.text_input("Enter your email")

if st.button('PREDICT'):

    # 1. preprocess
    text = tokenize(input_email)
    # 2. vectorize
    vector_input = tfidf.transform([text])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 0:
        st.header("Abusive")
    else:
        st.header("Non Abusive")