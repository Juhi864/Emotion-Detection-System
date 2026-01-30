import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
import joblib  # or import your model accordingly

# Load your trained model and vectorizer
model = joblib.load('model.joblib')  # Replace with your model file
vectorizer = joblib.load('vectorize.joblib')  # Replace with your vectorizer file

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

def process_input_and_predict(custom_text):
    # Lemmatize the input
    lemmatized_input = [lemmatizer.lemmatize(word.lower()) for word in custom_text.split()]
    lemmatized_input_str = ' '.join(lemmatized_input)  # Join back to string for vectorization
    
    # Vectorize the lemmatized input
    vectorized_input = vectorizer.transform([lemmatized_input_str])
    
    # Make the prediction using the trained model
    prediction = model.predict(vectorized_input)[0]
    
    # Return the prediction in human-readable form
    return "Positive" if prediction == 1 else "Negative"

# Streamlit UI
st.title('Sentiment Analysis App')
st.write("Enter a text to analyze its sentiment:")

# Take custom input from the user
custom_text = st.text_area("Input Text", "I really love this job")

if st.button('Predict Sentiment'):
    if custom_text:
        predicted_sentiment = process_input_and_predict(custom_text)
        st.success(f'Predicted Sentiment: {predicted_sentiment}')
    else:
        st.error('Please enter some text for sentiment analysis.')

