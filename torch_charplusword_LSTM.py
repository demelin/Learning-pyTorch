""" A PyTorch implementation of an recurrent neural network language model which uses a combination of word-level and 
character-level embeddings to represent individual words, in order to achieve an improved performance relative to a
strictly word-based variant. """

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.autograd import Variable
import numpy as np

# Declare some constants
CHAR_EMBED_DIM = 8
CHAR_HIDDEN_DIM = 8
WORD_EMBED_DIM = 16
WORD_HIDDEN_DIM = 16
LR = 0.001
NUM_EPOCHS = 300
REPORT_FREQ = 50

# Toy data for POS-tagging
training_data = [
    ("The dog ate the apple".lower().split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".lower().split(), ["NN", "V", "DET", "NN"])
]

test_data = [
    ("The dog read everybody".lower().split(), ["DET", "NN", "V", "NN"]),
    ("Everybody ate everybody".lower().split(), ["NN", "V", "NN"])
]


# Declare helper functions
def prepare_sequence_input(sequence, idx_dict):
    """ Transforms a sequence of items to a variable containing the sequence of corresponding, unique indices. """
    idx_sequence = [idx_dict[item] for item in sequence]
    return Variable(torch.LongTensor(idx_sequence))


def prepare_char_input(sequence_list, idx_dict):
    """ Like above, but for character-level encoding. Produces a list of lists of character idx. """
    char_word_list = list()
    for word in sequence_list:
        char_word = [idx_dict[char] for char in list(word)]
        char_word_list.append(Variable(torch.LongTensor(char_word)))
    return char_word_list


def flatten(source):
    """ Flattens a tuple of lists into a list """
    flattened = list()
    for l in source:
        for i in l:
            flattened.append(i)
    return flattened


def create_to_idx_dict(source):
    """ Creates a dictionary of item-specific indices form a tuple of lists. """
    idx_dict = dict()
    for i in source:
        if i not in idx_dict:
            idx_dict[i] = len(idx_dict)
    return idx_dict

# Create to_index dictionaries for words and tags
all_sents, all_tags = zip(*training_data)
word_list = flatten(all_sents)
tag_list = flatten(all_tags)

word_to_idx = create_to_idx_dict(word_list)
tag_to_idx = create_to_idx_dict(tag_list)
char_to_idx = create_to_idx_dict(list(''.join(word_list)))


# Set up the LSTM-RNN
class LSTMRNN(nn.Module):
    def __init__(self, vocab_size, alphabet_size, word_embed_dims, char_embed_dims, word_hidden_size, char_hidden_size,
                 target_size):
        super(LSTMRNN, self).__init__()
        # Set network parameters
        self.vocab_size = vocab_size
        self.alphabet_size = alphabet_size
        self.word_embed_dims = word_embed_dims
        self.char_embed_dims = char_embed_dims
        self.word_hidden_size = word_hidden_size
        self.char_hidden_size = char_hidden_size
        self.target_size = target_size
        # Define layers
        self.char_embedding_layer = nn.Embedding(self.alphabet_size, self.char_embed_dims)
        self.word_embedding_layer = nn.Embedding(self.vocab_size, self.word_embed_dims)
        self.char_lstm = nn.LSTM(self.char_embed_dims, self.char_hidden_size)
        self.word_lstm = nn.LSTM(self.word_embed_dims + self.char_hidden_size, self.word_hidden_size)
        self.linear = nn.Linear(self.word_hidden_size, self.target_size)
        # Initialize RNN hidden states
        self.char_lstm_hidden = self.init_hidden(self.char_hidden_size)
        self.word_lstm_hidden = self.init_hidden(self.word_hidden_size)

    @staticmethod
    def init_hidden(hidden_dims):
        """ Resets the hidden states of the RNNs used by the model. """
        return Variable(torch.zeros(1, 1, hidden_dims)), Variable(torch.zeros(1, 1, hidden_dims))

    def forward(self, word_input, char_input):
        """ Defines the forward pass. """
        # Get the character-level word encoding, first
        char_word_encodings = list()
        for char_word in char_input:
            char_embeds = self.char_embedding_layer(char_word)
            char_lstm_out, self.char_lstm_hidden = self.char_lstm(char_embeds.view(len(char_word), 1, -1))
            # Append the final hidden state of the char-LSTMRNN
            char_word_encodings.append(self.char_lstm_hidden[0])
            self.init_hidden(self.char_hidden_size)
        char_word_tensor = torch.cat(char_word_encodings, 0)
        # print(char_word_tensor)

        # Next, run the word-level RNN and predict the POS-tag per word in word input
        word_embeds = self.word_embedding_layer(word_input).view(len(word_input), 1, -1)
        combined_embeds = torch.cat((word_embeds, char_word_tensor), 2)
        word_lstm_out, self.word_lstm_hidden = self.word_lstm(combined_embeds)
        logits = self.linear(word_lstm_out.view(len(word_input), -1))
        prediction = F.log_softmax(logits)
        return prediction

# Initialize the model and declare training parameters
model = LSTMRNN(len(word_to_idx), len(char_to_idx), WORD_EMBED_DIM, CHAR_EMBED_DIM, WORD_HIDDEN_DIM, CHAR_HIDDEN_DIM,
                len(tag_to_idx))
loss_function = nn.NLLLoss()
optimizer = opt.Adam(model.parameters(), lr=LR)

# Training loop
for e in range(NUM_EPOCHS):
    total_loss = torch.FloatTensor([0.0])
    for words, tags in training_data:
        word_seq = prepare_sequence_input(words, word_to_idx)
        char_seq = prepare_char_input(words, char_to_idx)
        tag_seq = prepare_sequence_input(tags, tag_to_idx)
        # Zero model hidden states and grads
        model.zero_grad()
        model.char_lstm_hidden = model.init_hidden(model.char_hidden_size)
        model.word_lstm_hidden = model.init_hidden(model.word_hidden_size)
        # Obtain prediction
        predicted_tags = model(word_seq, char_seq)
        # Calculate loss
        loss = loss_function(predicted_tags, tag_seq)
        # Backprop
        loss.backward()
        # Optimize
        optimizer.step()
        # Update total epoch loss
        total_loss += loss.data
    # Bookkeeping
    if e % REPORT_FREQ == 0 and e != 0:
        print('Epoch: %d | Loss: %f.4' % (e, total_loss.numpy()))

# Check training data performance
print('\nEvaluating model performance on the test data:')
# Invert the tag-idx dictionary for prediction lookup
idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}
# Compare model predictions to gold standard
for words, tags in test_data:
    word_seq = prepare_sequence_input(words, word_to_idx)
    char_seq = prepare_char_input(words, char_to_idx)
    tag_seq = prepare_sequence_input(tags, tag_to_idx)

    predicted_tags = model(word_seq, char_seq)
    predicted_tags = np.argmax(predicted_tags.data.numpy(), axis=1)
    predicted_tags = [idx_to_tag[idx] for idx in predicted_tags]
    print('Examined sentence: %s' % ' '.join(words))
    print('Gold POS tags: %s' % ' '.join(tags))
    print('Predicted tags: %s\n' % ' '.join(predicted_tags))

print('All done!')
