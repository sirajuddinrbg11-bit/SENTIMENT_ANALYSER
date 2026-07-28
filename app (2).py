
import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK data if not already present
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

Lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]"," ",text)
    words = nltk.word_tokenize(text)
    WORDS = [Lemmatizer.lemmatize(word) for word in words]
    return " ".join(WORDS)

# Load the trained model and TF-IDF vectorizer
try:
    model = joblib.load('best_sentiment_svm_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    st.success("Model and TF-IDF vectorizer loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or TF-IDF vectorizer: {e}")
    st.stop() # Stop the app if models cannot be loaded

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
