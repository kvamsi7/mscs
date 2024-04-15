import sys
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle

model_path = r'D:\Masters\mscs\CS5720-Neural Network and Deep Learning\NN_Final_Project\model_lstm_73.h5'
tokenizer_path = r'D:\Masters\mscs\CS5720-Neural Network and Deep Learning\NN_Final_Project\tokenizer.pkl'

def load_keras_model(model_path):
    """Load the Keras model from the specified path."""
    try:
        model = load_model(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        sys.exit(1)

def load_tokenizer(tokenizer_path):
    """Load the tokenizer used during model training."""
    try:
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        print("Tokenizer loaded successfully.")
        return tokenizer
    except Exception as e:
        print(f"An error occurred while loading the tokenizer: {e}")
        sys.exit(1)

def prepare_text(text, tokenizer, max_len=100):
    """Convert text to padded sequence."""
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=max_len)
    return padded_sequence

def predict_sentiment(model, text, tokenizer):
    """Predict sentiment from text using the loaded model and tokenizer."""
    processed_text = prepare_text(text, tokenizer)
    prediction = model.predict(processed_text)
    if np.argmax(prediction) > 0.5:
        return 'Positive'
    else:
        return 'Negative'

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_model.py <model_path> <tokenizer_path> <text_to_evaluate>")
        sys.exit(1)

    #model_path = sys.argv[1]
    #tokenizer_path = sys.argv[2]
    text_to_evaluate = sys.argv[1]

    # Load the model and tokenizer
    model = load_keras_model(model_path)
    tokenizer = load_tokenizer(tokenizer_path)

    # Perform prediction
    sentiment = predict_sentiment(model, text_to_evaluate, tokenizer)
    print(f"Predicted sentiment: {sentiment}")
