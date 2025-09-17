# spam_app.py
import streamlit as st
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Page config
st.set_page_config(page_title="Spam Detection App", layout="centered")

st.title("📩 Spam Detection App")
st.write("Classify messages as **Spam** or **Not Spam** Doing with Yamini.")

# ------------------------------
# Load dataset (from CSV file if available)
# ------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("spam_dataset.csv")
    except Exception:
        data = {
            "message": [
                "Congratulations! You won a free lottery ticket.",
                "Hello, how are you doing today?",
                "Free entry in 2 a weekly competition to win prizes!",
                "Are we still meeting for lunch tomorrow?",
                "Call now to claim your free prize.",
                "Can you send me the report by today?",
            ],
            "label": ["spam", "ham", "spam", "ham", "spam", "ham"],
        }
        df = pd.DataFrame(data)
    return df

df = load_data()

# ------------------------------
# Preprocessing
# ------------------------------
def clean_text(text):
    text = text.lower()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    return text

df["clean_message"] = df["message"].apply(clean_text)

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_message"])
y = df["label"]

# Train model
model = MultinomialNB()
model.fit(X, y)

# ------------------------------
# User Input
# ------------------------------
st.subheader("Try it out")
user_input = st.text_area("Enter a message to classify:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        cleaned = clean_text(user_input)
        vect = vectorizer.transform([cleaned])
        prediction = model.predict(vect)[0]
        if prediction == "spam":
            st.error(" This message is classified as **SPAM**")
        else:
            st.success(" This message is classified as **NOT SPAM**")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.write("© 2025 Spam Detection Demo | Built with Streamlit")
