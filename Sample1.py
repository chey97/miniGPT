# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import pprint
import os
import time
import pandas as pd
from transformerModule import Transformer

def get_data_from_csv(csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Extract input and target phrases from the DataFrame
    data_words = df['input_phrase'].tolist()
    target_words = df['target_phrase'].tolist()
    
    # Remove '\u200b' character from input and target phrases
    data_words = [phrase.replace('\u200b', '') for phrase in data_words]
    target_words = [phrase.replace('\u200b', '') for phrase in target_words]
    
    # Create training_data dictionary
    training_data = {data_words[i]: target_words[i] for i in range(len(data_words))}

    return training_data,data_words, target_words

# Function to obtain training data, vocab and mapping from word to index and vice versa
def get_data_and_vocab():
    
    # Get training data from CSV file
    training_data, data_words, target_words = get_data_from_csv("test_data.csv")
    
    # Splitting input phrases into words and converting to lowercase
    data_words_split = [x.split(" ") for x in data_words]
    data_words_flattened = [element.lower() for nestedlist in data_words_split for element in nestedlist]

    # Splitting target phrases into words and converting to lowercase
    target_words_split = [x.split(" ") for x in target_words]
    target_words_flattened = [element.lower() for nestedlist in target_words_split for element in nestedlist]

    # Combining all words from input and target phrases, removing duplicates, and converting to list
    vocabulary_words = list(set(data_words_flattened + target_words_flattened))

    
    # Ensure <end> token is at the end of vocabulary list, and there's a blank at the beginning
    vocabulary_words.remove(".")
    vocabulary_words.append(".")
    vocabulary_words.insert(0, "") 
    
    # Create mappings from word to index and index to word
    word_to_ix = {vocabulary_words[k].lower(): k for k in range(len(vocabulary_words))}
    ix_to_word = {v: k for k, v in word_to_ix.items()}
    
    # Return all the necessary data and mappings
    return training_data, data_words, target_words, vocabulary_words, word_to_ix, ix_to_word

# Function to convert a batch of sequences of words to a tensor of indices
def words_to_tensor(seq_batch, device=None):
    index_batch = []
    
    # Loop over sequences in the batch
    for seq in seq_batch:
        word_list = seq.lower().split(" ")
        indices = [word_to_ix[word] for word in word_list if word in word_to_ix]
        t = torch.tensor(indices)
        if device is not None:
            t = t.to(device)  # Transfer tensor to the specified device
        index_batch.append(t)
    
    # Pad tensors to have the same length
    return pad_tensors(index_batch)

# Function to convert a tensor of indices to a list of sequences of words
def tensor_to_words(tensor):
    index_batch = tensor.cpu().numpy().tolist()
    res = []
    for indices in index_batch:
        words = []
        for ix in indices:
            words.append(ix_to_word[ix].lower())  # Convert index to word
            if ix == word_to_ix["."]:
                break  # Stop when <end> token is encountered
        res.append(" ".join(words))
    return res

# Function to pad a list of tensors to the same length
def pad_tensors(list_of_tensors):
    tensor_count = len(list_of_tensors) if not torch.is_tensor(list_of_tensors) else list_of_tensors.shape[0]
    max_dim = max(t.shape[0] for t in list_of_tensors)  # Find the maximum length
    res = []
    for t in list_of_tensors:
        # Create a zero tensor of the desired shape
        res_t = torch.zeros(max_dim, *t.shape[1:]).type(t.dtype).to(t.device)
        res_t[:t.shape[0]] = t  # Copy the original tensor into the padded tensor
        res.append(res_t)
    
    # Concatenate tensors along a new dimension
    res = torch.cat(res)
    firstDim = len(list_of_tensors)
    secondDim = max_dim
    
    # Reshape the result to have the new dimension first
    return res.reshape(firstDim, secondDim, *res.shape[1:])

# Input: PyTorch model (Transformer module), 
# data: input tensor (batch_size x max_token_count) 
# targets: ground truth/expected output tensor (batch_size x max_token_count)
# optimizer: PyTorch optimizer to use and criterion (which loss function to use)
# Function to train the model recursively over each sequence and token
def train_recursive(model, data, targets, optimizer, criterion):
    model.train()  # Set model to training mode
    optimizer.zero_grad()  # Zero the gradients
    total_loss = 0  # Initialize total loss
    batch_size, token_count, token_count_out = data.shape[0], data.shape[1], targets.shape[1]
    
    # Loop over sequences in the batch
    for b in range(batch_size):
        end_encountered = False
        cur_count = 0
        # Loop over tokens in the sequence
        while not end_encountered:
            target_vector = torch.zeros(model.vocab_size).to(data.device)  # Initialize target vector

            if cur_count != token_count_out:
                expected_next_token_idx = targets[b, cur_count]  # Get index of expected next token
                target_vector[expected_next_token_idx] = 1  # Set the corresponding element of the target vector to 1
            
            # Concatenate current input and output tokens and pass through model
            if cur_count > 0:
                model_input = data[b].reshape(token_count).to(data.device)
                part_of_output = targets[b, :cur_count].to(data.device)
                model_input = torch.cat((model_input, part_of_output))
            else:
                model_input = data[b]
            out = model(model_input.reshape(1, token_count + cur_count))
            
            # Compute loss and accumulate total loss
            loss = criterion(out, target_vector.reshape(out.shape))
            total_loss += loss
            cur_count += 1
            
            # Stop when the end of the sequence is reached
            if cur_count > token_count_out:
                end_encountered = True
    
    # Backpropagate gradients and update model parameters
    total_loss.backward()
    optimizer.step()
    return total_loss.item() / batch_size

# Function to perform inference recursively for each sequence in a batch
def infer_recursive(model, input_vectors, max_output_token_count=10):
    model.eval()  # Set model to evaluation mode
    outputs = []

    # Loop over sequences in the batch
    for i in range(input_vectors.shape[0]):
        print(f"Infering sequence {i}")
        input_vector = input_vectors[i].reshape(1, input_vectors.shape[1])
        predicted_sequence = []
        wc = 0  # Initialize word count

        with torch.no_grad():  # Disable gradient computation
            while True:
                output = model(input_vector)  # Pass current input through model
                predicted_index = output[0, :].argmax().item()  # Get index of predicted token
                predicted_sequence.append(predicted_index)  # Append predicted index to sequence
                # Stop when <end> token is predicted or the maximum output length is reached
                if predicted_index == word_to_ix['.'] or wc > max_output_token_count:
                    break
                # Append predicted token to input and increment word count
                input_vector = torch.cat([input_vector, torch.tensor([[predicted_index]])], dim=1)
                wc += 1
        outputs.append(torch.tensor(predicted_sequence))  # Append predicted sequence to outputs
    outputs = pad_tensors(outputs)  # Pad predicted sequences to the same length
    return outputs

def example_training_and_inference():
    global model
    # Get training data and vocabulary
    _, data_words, target_words, _, _, _ = get_data_and_vocab()

    # Get model hyperparameters from vocabulary size
    vocab_size = len(word_to_ix)
    embed_size = 512
    num_layers = 4
    heads = 3
    device = torch.device("cpu")
    
     # Load the saved model (if it exists)
    model_path = "trainedModels/trained_model15.pth" 
    try:
        model = Transformer(vocab_size, embed_size, num_layers, heads).to(device)
        model.load_state_dict(torch.load(model_path))
        print("Loaded saved model from:", model_path)
    except FileNotFoundError:
        print("No saved model found. Training from scratch.")
        model = Transformer(vocab_size, embed_size, num_layers, heads).to(device)

    # Create model, optimizer, and loss function
    model = Transformer(vocab_size, embed_size, num_layers, heads).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    criterion = nn.CrossEntropyLoss()

    # Convert training data to tensors
    data = words_to_tensor(data_words, device=device)
    targets = words_to_tensor(target_words, device=device)

    # Train model for 100 epochs
    for epoch in range(100):
        avg_loss = train_recursive(model, data, targets, optimizer, criterion)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')

    # Perform inference on training data
    input_vector = words_to_tensor(data_words, device=device)
    predicted_vector = infer_recursive(model, input_vector)
    predicted_words = tensor_to_words(predicted_vector)

    # Print training data and model output
    print("\n\n\n")
    print("Training Data:")
    pprint.pprint(training_data)
    print("\n\n")
    print("Model Inference:")
    result_data = {data_words[k]: predicted_words[k] for k in range(len(predicted_words))}
    pprint.pprint(result_data)

# Define a variable to keep track of the model number
model_number = 1

# Function to save the model with a unique name
def save_model(model):
    global model_number
    model_name = f'trainedModels/trained_model{model_number}.pth'
    # Check if the file already exists
    while os.path.exists(model_name):
        model_number += 1
        model_name = f'trainedModels/trained_model{model_number}.pth'
    # Save the model with the unique name
    torch.save(model.state_dict(), model_name)
    print(f"Model saved at: {model_name}")
    # Increment the model number for the next model
    model_number += 1

# Main function to call the demonstration function
if __name__ == "__main__":
    # Get training data and vocabulary
    training_data, data_words, target_words, vocabulary_words, word_to_ix, ix_to_word = get_data_and_vocab()
    
    # Run the example training and inference function
    example_training_and_inference()
    
    # Save the trained model
    global model
    save_model(model)