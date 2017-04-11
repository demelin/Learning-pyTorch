""" A decoder network used in a sequence-to-sequence NMT architecture, as outlined in 
http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html. Local attention mechanism is based on
'Effective Approaches to Attention-based Neural Machine Translation' by Luong et al.(2015) . """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

use_cuda = torch.cuda.is_available()


class LocallyAttentiveDecoder(nn.Module):
    """ A Decoder with a local attention mechanism for finding soft alignments between parallel sentence pairs. """

    def __init__(self, output_size, embed_dims, hidden_size, keep_prob, window_radius, num_layers=1):
        super(LocallyAttentiveDecoder, self).__init__()
        self.target_size = output_size
        self.embed_dims = embed_dims
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.window_radius = window_radius
        self.num_layers = num_layers
        self.attn_size = self.window_radius * 2 + 1

        # Layers
        self.embedding = nn.Embedding(self.target_size, self.embed_dims)
        # Equation (7)
        self.get_score = nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size)
        self.v_score = Variable(torch.randn(self.hidden_size, 1), requires_grad=True)
        # Equation (9)
        self.get_position = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_position = Variable(torch.randn(self.hidden_size, 1), requires_grad=True)
        # Equation (5)
        self.get_attn_hidden = nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size)

        self.dropout = nn.Dropout(self.keep_prob)
        self.gru_rnn = nn.GRU(self.hidden_size, self.hidden_size)
        self.to_softmax = nn.Linear(self.hidden_size, self.target_size)

    def init_hidden(self):
        """ Resets the hidden states of the RNNs used by the model. """
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

    def forward(self, input_data, rnn_hidden, enc_outputs, source_length, time_step):
        """ Defines a single step of the forward pass. """
        # Embed the input data
        embeds = self.dropout(self.embedding(input_data)).view(len(input_data), 1, -1)  # [1, 1, 128]
        output = embeds
        # Feed embeddings into the GRU-RNN
        for _ in range(self.num_layers):
            output, rnn_hidden = self.gru_rnn(output, rnn_hidden)  # output.shape = [1, 1, 128]
        # Flatten output down to two dimensions
        output = output.view(1, -1)

        # --- Attention ---
        # Predict aligned word position within the source sentence at time step t
        position = F.tanh(self.get_position(output))
        position = F.sigmoid(torch.mm(position, self.v_position))
        position = source_length * position

        # Produce the inter-sentential alignment ('concat' variant)
        encoder_window = Variable(torch.zeros(self.attn_size, self.hidden_size))
        broadcast_hidden = Variable(torch.zeros(self.attn_size, self.hidden_size))
        start_point = np.max([0, time_step - self.window_radius])
        end_point = np.min([source_length, time_step + self.window_radius])
        for i in range(start_point, end_point):
            encoder_window[i - start_point, :] = enc_outputs[i, :]
            broadcast_hidden[i - start_point, :] = output
        score_input = torch.cat([broadcast_hidden, encoder_window], 1)
        score = F.tanh(self.get_score(score_input))
        score = torch.mm(score, self.v_score)
        align = F.softmax(score)

        # Calculate attention weights
        scalar = torch.exp(-(torch.pow(time_step - position, 2) / self.window_radius))
        attn_weights = align * scalar.expand(align.size())
        # Calculate the context vector (i.e. weighted average over all source hidden states within the attention window)
        context_vector = torch.mm(attn_weights.permute(1, 0), encoder_window)
        # Calculate the attentional hidden state
        attn_hidden = self.get_attn_hidden(torch.cat([output, context_vector], 1))
        # Output predictions
        output = F.log_softmax(self.to_softmax(attn_hidden))
        return output, attn_hidden, attn_weights
