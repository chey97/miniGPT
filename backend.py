from flask import Flask, request, jsonify
import torch
from Sample import get_data_and_vocab, words_to_tensor, tensor_to_words, infer_recursive
from transformerModule import Transformer
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

# Load the trained model
def load_model(model_path):
    print("Loading model...")
    # Get model hyperparameters from vocabulary size
    _, _, _, _, word_to_ix, _ = get_data_and_vocab()
    vocab_size = len(word_to_ix)
    embed_size = 512
    num_layers = 4
    heads = 3

    # Create model
    model = Transformer(vocab_size, embed_size, num_layers, heads)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded successfully.")
    return model

# Load the trained model
model_path = 'trainedModels/trained_model3.pth'  # Provide the path to your trained model
model = load_model(model_path)
print("Model path:", model_path)

# API endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    print("Received POST request to /predict endpoint.")
    # Get input data from the request
    input_data = request.json.get('input_data')
    print("Input data:", input_data)

    # Convert input data to tensors
    _, _, _, _, word_to_ix, _ = get_data_and_vocab()
    input_tensor = words_to_tensor([input_data], word_to_ix)
    print("Input tensor:", input_tensor)

    # Perform inference
    predicted_tensor = infer_recursive(model, input_tensor)
    print("Predicted tensor:", predicted_tensor)

    # Convert predicted tensor to words
    predicted_words = tensor_to_words(predicted_tensor, word_to_ix)
    print("Predicted words:", predicted_words)

    # Return the predicted answer
    return jsonify({'predicted_answer': predicted_words[0]})

if __name__ == '__main__':
    app.run(debug=True)
