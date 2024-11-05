import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the model and vectorizer (modify paths if needed)
model = joblib.load("model.pkl")
vectorizer=joblib.load('vectorizer.pkl')
# Streamlit app
st.title("Email Spam Detection")

st.write("""Enter an email below to classify it as spam or not spam.""")

# Input text from user
user_input = st.text_area("Email Text", "Type your email content here...")

if st.button("Classify"):
    if user_input:
        # Preprocess and transform input
        user_input_vector = vectorizer.transform([user_input])
        prediction = model.predict(user_input_vector)
        if prediction[0] == 1:
            st.write("### This email is classified as: Spam")
        else:
            st.write("### This email is classified as: Not Spam")

st.write("Model trained using Naive Bayes and Count Vectorizer.")
