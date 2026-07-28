import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn

# Download NLTK data if not already present (wordnet attempted; punkt is optional)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    try:
        nltk.download('wordnet')
    except Exception:
        # If download fails, we'll still run but without lemmatization
        pass

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    # punkt is optional because we fall back to a regex tokenizer below
    pass

# Initialize lemmatizer safely (may remain None if WordNet isn't available)
try:
    Lemmatizer = WordNetLemmatizer()
except Exception:
    Lemmatizer = None

def preprocess(text):
    """
    Lowercase, remove non-letters, tokenize (try NLTK then regex fallback), and lemmatize if possible.
    This avoids crashing when NLTK punkt or wordnet resources are unavailable.
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # Try nltk.word_tokenize (which requires punkt). If punkt is missing, fall back to regex.
    try:
        words = nltk.word_tokenize(text)
        # filter out any non-alpha tokens introduced by tokenization
        words = [w for w in words if re.fullmatch(r"[A-Za-z]+", w)]
    except LookupError:
        words = re.findall(r"\b[a-zA-Z]+\b", text)

    # Lemmatize if available, otherwise return tokens as-is
    if Lemmatizer is not None:
        try:
            words = [Lemmatizer.lemmatize(w) for w in words]
        except Exception:
            # On any unexpected lemmatizer error, skip lemmatization
            pass

    return " ".join(words)

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

# Small helper to show model/runtime diagnostics
def show_diagnostics(exc=None):
    st.write("---")
    st.write("Runtime diagnostics:")
    try:
        st.write("scikit-learn version:", sklearn.__version__)
    except Exception:
        st.write("scikit-learn version: unknown")
    st.write("Model type:", type(model))
    st.write("Model has attribute '_effective_probability'?:", hasattr(model, '_effective_probability'))
    st.write("Model has 'support_' (fitted indicator)?:", hasattr(model, 'support_'))
    if exc is not None:
        st.write("Exception:", str(exc))
    st.write("---")

if st.button("Analyze Sentiment"):
    if user_input:
        # Preprocess the input text
        cleaned_text = preprocess(user_input)

        # Transform the cleaned text using the loaded TF-IDF vectorizer
        # Ensure the vectorizer expects a list of documents
        vectorized_text = tfidf_vectorizer.transform([cleaned_text])

        # Make a prediction (safe wrapper)
        try:
            prediction = model.predict(vectorized_text)
        except AttributeError as e:
            st.error("Model appears incompatible with this environment (AttributeError during prediction).")
            show_diagnostics(e)
            st.info("Possible fixes: reinstall the scikit-learn version used when the model was created, or re-train and re-pickle the model in this environment.")
            # Stop further execution for safety
            st.stop()
        except Exception as e:
            st.error("An error occurred during prediction.")
            show_diagnostics(e)
            st.info("If this persists, consider re-training the model or checking model file compatibility.")
            st.stop()

        st.subheader("Prediction:")
        if prediction[0] == 'pos':
            st.success("Positive Sentiment! 😄")
        elif prediction[0] == 'neg':
            st.error("Negative Sentiment! 😞")
        else:
            st.info(f"Predicted sentiment: {prediction[0]}")
    else:
        st.warning("Please enter some text to analyze.")
