## Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

## Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

## Load the pre-trained model with ReLU activation
model = load_model('imdb_rnn_model.h5')

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


## WEB PAGE CODE
import streamlit as st
from datetime import date

# Customize the page appearance
st.set_page_config(
    page_title="Movie Review System",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Add custom styles for color themes
st.markdown(
    """
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f8f9fa; /* Light grey for the body */
        color: #343a40;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        font-size: 16px;
        padding: 10px 20px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .header {
        background-color: #8A2BE2; /* Violet header */
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
    }
    .header-title {
        font-size: 30px;
        margin: 0;
    }
    .subtext {
        font-size: 16px;
        margin-top: 10px;
    }
    .container {
        margin: 0 auto;
        max-width: 700px;
        padding: 20px;
        # background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        font-size: 14px;
        color: #666;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown(
    """
    <div class="header">
        <div class="header-title">🎥 Movie Review System</div>
        <div class="subtext">Share your thoughts about the movies you've watched and get insights!</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Main App Container
st.markdown("<div class='container'>", unsafe_allow_html=True)

# 1. Date Selection
st.subheader("When did you watch the movie?")
date_selection = st.selectbox(
    "Choose a time frame:",
    ["Select a time frame", "Less than a week ago", "1–2 weeks ago", "A month ago", "More than a month ago"],
)

# 2. Movie Name
st.subheader("What was the movie?")
movie_name = st.text_input("Type the movie name below (required):")

# 3. Movie Type Selection
st.subheader("Select the movie genre")
movie_types = ['Action', 'Comedy', 'Romance', 'Thriller', 'Drama', 'Horror', 'Sci-Fi', 'Fantasy']
movie_type = st.selectbox("Choose the movie type:", options=["Select a genre"] + movie_types)

# 4. Review Box
st.subheader("Your Review")
review = st.text_area("Write your review here (required):", height=150)

# Submission and Prediction
if st.button("Submit Review"):
    if movie_name.strip() and review.strip():
        # Simulating Model Prediction
        preprocessed_input = preprocess_text(review)  # Replace with your preprocessing function
        prediction = model.predict(preprocessed_input)  # Replace with your model
        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
        score = prediction[0][0]

        # Display Results
        st.success("Review Submitted Successfully!")
        st.markdown(f"### Sentiment: **{sentiment}**")
        st.markdown(f"### Prediction Score: **{score:.2f}**")
    else:
        if not movie_name.strip():
            st.error("Please enter the movie name.")
        if not review.strip():
            st.error("Please write your movie review.")

# Footer
st.markdown("</div>", unsafe_allow_html=True)  # Close container
st.markdown("<div class='footer'>Built with ❤️ using Streamlit</div>", unsafe_allow_html=True)

