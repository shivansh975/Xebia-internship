import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
import re

# Download stopwords
nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load('vectorizer.pkl')  # Make sure this is a TfidfVectorizer, not overwritten

# English stopwords
stop_words = set(stopwords.words('english'))

# Clean text function
def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text).lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis For Movies", layout="centered")
st.title("🎬 Sentiment Analysis App for Movies")
st.markdown("Enter a movie review to find out its sentiment!")

# User input
user_input = st.text_area("Enter the review:")

# Predict button
if st.button("Predict Sentiment"):
    if not user_input.strip():
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(user_input)
        vector_input = vectorizer.transform([cleaned])  # Use a separate name to avoid confusion
        prediction = model.predict(vector_input)[0]
        sentiment = "👍 Positive" if prediction == 1 else "👎 Negative"
        st.success(f"Prediction: {sentiment}")