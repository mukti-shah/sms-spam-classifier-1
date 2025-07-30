import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import pickle

nltk.download('punkt')
nltk.download('stopwords')

tfidf_vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

stop_words = stopwords.words('english')
stemmer = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)

    a= []
    for i in text:
        i = i.strip(string.punctuation)
        if i.isalnum() and i not in stop_words:
            a.append(stemmer.stem(i))
            
    return " ".join(a)


st.title("Spam SMS Detection")

st.subheader("Enter the SMS text below:")
user_input = st.text_area("SMS Text", "")

if st.button("Predict"):
    if not user_input:
        st.error("Please enter an SMS text.")
        st.stop()

    transformed_text = transform_text(user_input)

    vectorized_text = tfidf_vectorizer.transform([transformed_text])

    result = model.predict(vectorized_text)[0]

    if result == 1:
        st.header("This SMS is spam.")
    else:
        st.header("This SMS is not spam.")