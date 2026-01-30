import pickle
import numpy as np

def load_model_and_predict(text):
    """
    Load the saved sentiment analysis model and make predictions
    
    Parameters:
    text (str or list): Input text(s) for sentiment analysis
    
    Returns:
    prediction: Predicted sentiment(s)
    """
    # Load the saved model
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Make predictions
    # Note: You might need to preprocess the text the same way you did during training
    try:
        if isinstance(text, str):
            prediction = model.predict([text])
        else:
            prediction = model.predict(text)
        return prediction
    except Exception as e:
        return f"Error making prediction: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Test with single text
    sample_text = "This movie was really great! I enjoyed every minute of it."
    prediction = load_model_and_predict(sample_text)
    print(f"\nText: {sample_text}")
    print(f"Predicted sentiment: {prediction}")
    
    # Test with multiple texts
    sample_texts = [
        "i hate this job very much",
        "The service was okay, nothing special.",
        "I absolutely love this place, highly recommended!"
    ]
    predictions = load_model_and_predict(sample_texts)
    print("\nBatch predictions:")
    for text, pred in zip(sample_texts, predictions):
        if pred=="E" and  text:
            print(f"\nText: {text}")
            print(f"Predicted sentiment: Negative")
        if pred=="r" and text :
            print(f"\nText: {text}")
            print(f"Predicted sentiment:Positive")   

        else:
            print("Unknown") 

        # print(f"\nText: {text}")
        # print(f"Predicted sentiment: {pred}")        
        # print(pred)