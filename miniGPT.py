# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import pprint
import selfAttentionModule, transformerModule, transformerBlock


# Function to obtain training data, vocab and mapping from word to index and vice versa
def get_data_and_vocab():
    # Define training data
    training_data = {
        "how are you": "i am fine <end>",
        "who is john": "a nice person <end>",
        "who is nice": "john <end>",
        "where is john": "at home <end>",
        "how is john": "i dont know <end>",
        "who are you": "mini gpt model <end>",
    }

    # Extract input and target phrases
    data_words = [k for k, _ in training_data.items()]
    target_words = [v for _, v in training_data.items()]

    # Build vocabulary from training data
    vocabulary_words = list(
        set(
            [
                element.lower()
                for nestedlist in [x.split(" ") for x in data_words]
                for element in nestedlist
            ]
            + [
                element.lower()
                for nestedlist in [x.split(" ") for x in target_words]
                for element in nestedlist
            ]
        )
    )

    # Ensure <end> token is at the end of vocabulary list, and there's a blank at the beginning
    vocabulary_words.remove("<end>")
    vocabulary_words.append("<end>")
    vocabulary_words.insert(0, "")

    # Create mappings from word to index and index to word
    word_to_ix = {vocabulary_words[k].lower(): k for k in range(len(vocabulary_words))}
    ix_to_word = {v: k for k, v in word_to_ix.items()}

    # Return all the necessary data and mappings
    return (
        training_data,
        data_words,
        target_words,
        vocabulary_words,
        word_to_ix,
        ix_to_word,
    )


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
            if ix == word_to_ix["<end>"]:
                break  # Stop when <end> token is encountered
        res.append(" ".join(words))
    return res

# Function to pad a list of tensors to the same length
def pad_tensors(list_of_tensors):
    tensor_count = (
        len(list_of_tensors)
        if not torch.is_tensor(list_of_tensors)
        else list_of_tensors.shape[0]
    )
    max_dim = max(t.shape[0] for t in list_of_tensors)  # Find the maximum length
    res = []
    for t in list_of_tensors:
        # Create a zero tensor of the desired shape
        res_t = torch.zeros(max_dim, *t.shape[1:]).type(t.dtype).to(t.device)
        res_t[: t.shape[0]] = t  # Copy the original tensor into the padded tensor
        res.append(res_t)

    # Concatenate tensors along a new dimension
    res = torch.cat(res)
    firstDim = len(list_of_tensors)
    secondDim = max_dim

    # Reshape the result to have the new dimension first
    return res.reshape(firstDim, secondDim, *res.shape[1:])


# Main function to call the demonstration function
if __name__ == "__main__":
    # Get training data and vocabulary
    (
        training_data,
        data_words,
        target_words,
        vocabulary_words,
        word_to_ix,
        ix_to_word,
    ) = get_data_and_vocab()
    # Run the example training and inference function
    # example_training_and_inference()
    print(vocabulary_words)
    x = words_to_tensor(vocabulary_words)
    print(x)
    print(tensor_to_words(x))
    