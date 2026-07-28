import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Keep wordnet download, but don't require punkt (we use regex tokenizer)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    pass  # punkt is optional now

def preprocess(text):
    # Lowercase, keep letters only, tokenize with regex (avoids punkt), and lemmatize
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    WORDS = [Lemmatizer.lemmatize(word) for word in words]
    return " ".join(WORDS)

# Load the trained model and TF-IDF vectorizer
try:
    model = joblib.load('best_sentiment_svm_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    st.success("Model and TF-IDF vectorizer loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or TF-IDF vectorizer: {e}")
    st.stop()  # Stop the app if models cannot be loaded

# Streamlit app layout
st.title("Sentiment Analysis App")
st.write("Enter a review below to classify its sentiment (positive/negative).")

user_input = st.text_area("Review text:", "")

if st.button("Analyze Sentiment"):
    if user_input:
        # Preprocess the input text
        cleaned_text = preprocess(user_input)

        # Transform the cleaned text using the loaded TF-IDF vectorizer
        # Ensure the vectorizer expects a list of documents
        vectorized_text = tfidf_vectorizer.transform([cleaned_text])

        # Make a prediction
        prediction = model.predict(vectorized_text)

        st.subheader("Prediction:")
        if prediction[0] == 'pos':
            st.success("Positive Sentiment! 😄")
        elif prediction[0] == 'neg':
            st.error("Negative Sentiment! 😞")
        else:
            st.info(f"Predicted sentiment: {prediction[0]}")
    else:
        st.warning("Please enter some text to analyze.")
