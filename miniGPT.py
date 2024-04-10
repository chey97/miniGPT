# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import pprint

# Function to obtain training data, vocab and mapping from word to index and vice versa
def get_data_and_vocab():
    # Define training data
    training_data = {
        "how are you": "i am fine <end>",
        "who is john": "a nice person <end>",
        "who is nice": "john <end>",
        "where is john": "at home <end>",
        "how is john": "i dont know <end>",
        "who are you": "mini gpt model <end>"
    }
    
    # Extract input and target phrases
    data_words = [k for k, _ in training_data.items()]
    target_words = [v for _, v in training_data.items()]
    
    # Build vocabulary from training data
    vocabulary_words = list(set([element.lower() for nestedlist in [x.split(" ") for x in data_words] for element in nestedlist] + [element.lower() for nestedlist in [x.split(" ") for x in target_words] for element in nestedlist]))
    
    # Ensure <end> token is at the end of vocabulary list, and there's a blank at the beginning
    vocabulary_words.remove("<end>")
    vocabulary_words.append("<end>")
    vocabulary_words.insert(0, "")
    
    # Create mappings from word to index and index to word
    word_to_ix = {vocabulary_words[k].lower(): k for k in range(len(vocabulary_words))}
    ix_to_word = {v: k for k, v in word_to_ix.items()}
    
    # Return all the necessary data and mappings
    return training_data, data_words, target_words, vocabulary_words, word_to_ix, ix_to_word