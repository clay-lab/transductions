# Imports
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import sys
import os

# Functions for tracking time
import time
import math

use_cuda = torch.cuda.is_available()

if use_cuda:
    available_device = torch.device('cuda')
else:
    available_device = torch.device('cpu')

# Generic sequential encoder
class EncoderRNN(nn.Module): 
    def __init__(self, vocab_size, hidden_size, recurrent_unit, n_layers=1, max_length=30, dropout_p=0):
        super(EncoderRNN, self).__init__()
        self.num_layers = n_layers
        self.hidden_size = hidden_size
        self.rnn_type = recurrent_unit
        self.max_length = max_length
        self.dropout = nn.Dropout(p=dropout_p)

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        if recurrent_unit == "SRN":
                self.rnn = nn.RNN(hidden_size, hidden_size, num_layers = self.num_layers, dropout = dropout_p)
        elif recurrent_unit == "GRU":
                self.rnn = nn.GRU(hidden_size, hidden_size, num_layers = self.num_layers, dropout = dropout_p)
        elif recurrent_unit == "LSTM":
                self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers = self.num_layers, dropout = dropout_p)
        else:
                print("Invalid recurrent unit type")


    # For succesively generating each new output and hidden layer
    def forward(self, batch):

        #outputs = Variable(torch.zeros(self.max_length, batch_size, self.hidden_size))
        #outputs = outputs.to(device=available_device) # to be used by attention in the decoder
        embedded_source = self.dropout(self.embedding(batch))
        outputs, final_hiddens = self.rnn(embedded_source)
        final_output = outputs[-1]
        #only return the last timestep's h vectors for the last encoder layer
        final_hiddens = final_hiddens[-1]

        return final_output, final_hiddens, outputs

# Generic sequential decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, recurrent_unit, attn=False, n_layers=1, dropout_p=0.1, max_length=30):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.attention = attn

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)

        if recurrent_unit == "SRN":
                self.rnn = nn.RNN(self.hidden_size, self.hidden_size)
        elif recurrent_unit == "GRU":
                self.rnn = nn.GRU(self.hidden_size, self.hidden_size)
        elif recurrent_unit == "LSTM":
                self.rnn = nn.LSTM(self.hidden_size, self.hidden_size)
        elif recurrent_unit == "SquashedLSTM":
                self.rnn = SquashedLSTM(self.hidden_size, self.hidden_size)
        elif recurrent_unit == "ONLSTM":
                self.rnn = ONLSTM(self.hidden_size, self.hidden_size)
        elif recurrent_unit == "UnsquashedGRU":
                self.rnn = UnsquashedGRU(hidden_size, hidden_size)
        else:
                print("Invalid recurrent unit type")

        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.recurrent_unit = recurrent_unit

        # location-based attention
        if attn == "location":
                # Attention vector
                self.attn = nn.Linear(self.hidden_size * 2, self.max_length)

                # Context vector made by combining the attentions
                self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # content-based attention
        if attn == "content": 
                self.v = nn.Parameter(torch.FloatTensor(hidden_size), requires_grad=True)
                nn.init.uniform(self.v, -1, 1) # maybe need cuda
                self.attn_layer = nn.Linear(self.hidden_size * 3, self.hidden_size)
                self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

    # Perform one step of the forward pass
    def forward_step(self, input, hidden, encoder_outputs, input_variable):
        output = self.embedding(input).unsqueeze(0)
        output = self.dropout(output)

        attn_weights = None

        batch_size = input_variable.size()[1]

        # Determine attention weights using location-based attention
        if self.attention == "location":
                if self.recurrent_unit == "LSTM" or self.recurrent_unit == "ONLSTM" or self.recurrent_unit == "SquashedLSTM":
                    attn_weights = F.softmax(self.attn(torch.cat((output[0], hidden[0][0]), 1)))
                else:
                    attn_weights = F.softmax(self.attn(torch.cat((output[0], hidden[0]), 1)))

                attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs.transpose(0,1))
                attn_applied = attn_applied.transpose(0,1)

                output = torch.cat((output[0], attn_applied[0]), 1)
                output = self.attn_combine(output).unsqueeze(0)

        # Determine attention weights using content-based attention
        if self.attention == "content": 
                input_length = input_variable.size()[0] 
                u_i = Variable(torch.zeros(len(encoder_outputs), batch_size))

                u_i = u_i.to(device=available_device)


                for i in range(input_length):
                        if self.recurrent_unit == "LSTM"  or self.recurrent_unit == "ONLSTM" or self.recurrent_unit == "SquashedLSTM":
                                attn_hidden = F.tanh(self.attn_layer(torch.cat((encoder_outputs[i].unsqueeze(0), hidden[0][0].unsqueeze(0), output), 2)))
                        else:
                                attn_hidden = F.tanh(self.attn_layer(torch.cat((encoder_outputs[i].unsqueeze(0), hidden[0].unsqueeze(0), output), 2)))
                        u_i_j = torch.bmm(attn_hidden, self.v.unsqueeze(1).unsqueeze(0))
                        u_i[i] = u_i_j[0].view(-1)


                a_i = F.softmax(u_i.transpose(0,1)) 
                attn_applied = torch.bmm(a_i.unsqueeze(1), encoder_outputs.transpose(0,1))

                attn_applied = attn_applied.transpose(0,1)

                output = torch.cat((output[0], attn_applied[0]), 1)
                output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.rnn(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    # Perform the full forward pass
    def forward(self, hidden, encoder_outputs, training_set, tf_ratio=0.5, evaluation=False):
        input_variable = training_set[0]
        target_variable = training_set[1]

        batch_size = training_set[0].size()[1]

        decoder_input = Variable(torch.LongTensor([0] * batch_size))
        decoder_input = decoder_input.to(device=available_device)

        decoder_hidden = hidden
        
        decoder_outputs = []

        use_tf = True if random.random() < tf_ratio else False

        if use_tf: # Using teacher forcing
            for di in range(target_variable.size()[0]):
                decoder_output, decoder_hidden, decoder_attention = self.forward_step(
                                decoder_input, decoder_hidden, encoder_outputs, input_variable)
                decoder_input = target_variable[di]
                decoder_outputs.append(decoder_output)

        else: # Not using teacher forcing
            if evaluation:
                end_num = 100
            else:
                end_num = target_variable.size()[0]

            for di in range(end_num): 
                decoder_output, decoder_hidden, decoder_attention = self.forward_step(
                            decoder_input, decoder_hidden, encoder_outputs, input_variable) 

                topv, topi = decoder_output.data.topk(1)
                decoder_input = Variable(topi.view(-1))
                decoder_input = decoder_input.to(device=available_device)

                decoder_outputs.append(decoder_output)

                if 1 in topi[0] or 2 in topi[0]:
                    break

        return decoder_outputs 

# GRU modified such that its hidden states are not bounded
class UnsquashedGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(UnsquashedGRU, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.wr = nn.Linear(hidden_size + input_size, hidden_size)
        self.wz = nn.Linear(hidden_size + input_size, hidden_size)
        self.wv = nn.Linear(hidden_size + input_size, hidden_size)
        self.wx = nn.Linear(input_size, hidden_size)
        self.urh = nn.Linear(hidden_size, hidden_size)


    def forward(self, input, hidden):
        hx = hidden
        input_plus_hidden = torch.cat((input, hx), 2)
        r_t = F.sigmoid(self.wr(input_plus_hidden))
        z_t = F.sigmoid(self.wz(input_plus_hidden))
        v_t = F.sigmoid(self.wv(input_plus_hidden))
        h_tilde = F.tanh(self.wx(input) + self.urh(r_t * hx))
        h_t = z_t * hx + v_t * h_tilde

        return h_t, h_t


# CumMax function for use in the ON-LSTM
class CumMax(nn.Module):
        def __init__(self):
                super(CumMax, self).__init__()

        def forward(self, input):
                return torch.cumsum(nn.Softmax(dim=2)(input), 2)

# Ordered Neurons LSTM recurrent unit
class ONLSTM(nn.Module):
        def __init__(self, input_size, hidden_size):
                super(ONLSTM, self).__init__()

                self.hidden_size = hidden_size
                self.input_size = input_size

                self.wi = nn.Linear(hidden_size + input_size, hidden_size)
                self.wf = nn.Linear(hidden_size + input_size, hidden_size)
                self.wg = nn.Linear(hidden_size + input_size, hidden_size)
                self.wo = nn.Linear(hidden_size + input_size, hidden_size)
                self.wftilde = nn.Linear(hidden_size + input_size, hidden_size)
                self.witilde = nn.Linear(hidden_size + input_size, hidden_size)

        def forward(self, input, hidden):
                hx, cx = hidden
                input_plus_hidden = torch.cat((input, hx), 2)

                f_t = F.sigmoid(self.wf(input_plus_hidden))
                i_t = F.sigmoid(self.wi(input_plus_hidden))
                o_t = F.sigmoid(self.wo(input_plus_hidden))
                c_hat_t = F.tanh(self.wg(input_plus_hidden))

                f_tilde_t = CumMax()(self.wftilde(input_plus_hidden))
                i_tilde_t = 1 - CumMax()(self.witilde(input_plus_hidden))

                omega_t = f_tilde_t * i_tilde_t
                f_hat_t = f_t * omega_t + (f_tilde_t - omega_t)
                i_hat_t = i_t * omega_t + (i_tilde_t - omega_t)

                cx = f_hat_t * cx + i_hat_t * c_hat_t
                hx = o_t * F.tanh(cx)

                return hx, (hx, cx)

# LSTM modified so that both its hidden and cell states are bounded
class SquashedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SquashedLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.wi = nn.Linear(hidden_size + input_size, hidden_size)
        self.wf = nn.Linear(hidden_size + input_size, hidden_size)
        self.wg = nn.Linear(hidden_size + input_size, hidden_size)
        self.wo = nn.Linear(hidden_size + input_size, hidden_size)


    def forward(self, input, hidden):
        hx, cx = hidden
        input_plus_hidden = torch.cat((input, hx), 2)
        i_t = F.sigmoid(self.wi(input_plus_hidden))
        f_t = F.sigmoid(self.wf(input_plus_hidden))
        g_t = F.tanh(self.wg(input_plus_hidden))
        o_t = F.sigmoid(self.wo(input_plus_hidden))

        sum_fi = f_t + i_t

        cx = (f_t * cx + i_t * g_t)/sum_fi
        hx = o_t * F.tanh(cx)

        return hx, (hx, cx)

# Tree-based encoder
# This implements the Tree-GRU of Chen et al. 2017, described
# in section 3.2 of this paper: https://arxiv.org/pdf/1707.05436.pdf
class TreeEncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(TreeEncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        emb_size = hidden_size
        self.emb_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_size)


        self.l_rl = nn.Linear(hidden_size, hidden_size)
        self.r_rl = nn.Linear(hidden_size, hidden_size)

        self.l_rr = nn.Linear(hidden_size, hidden_size)
        self.r_rr = nn.Linear(hidden_size, hidden_size)

        self.l_zl = nn.Linear(hidden_size, hidden_size)
        self.r_zl = nn.Linear(hidden_size, hidden_size)


        self.l_zr = nn.Linear(hidden_size, hidden_size)
        self.r_zr = nn.Linear(hidden_size, hidden_size)

        self.l_z = nn.Linear(hidden_size, hidden_size)
        self.r_z = nn.Linear(hidden_size, hidden_size)

        self.l = nn.Linear(hidden_size, hidden_size)
        self.r = nn.Linear(hidden_size, hidden_size)


    def forward(self, training_set):
        input_seq = training_set[0]
        tree = training_set[2]
        embedded_seq = []

        for elt in input_seq:
            embedded_seq.append(self.embedding(Variable(torch.LongTensor([elt])).to(device=available_device)).unsqueeze(0))

        # current_level starts out as a list of word embeddings for the words in the sequence
        # Then, the tree (which is passed as an input - the element at index 2 of training_set) is used to 
        # determine which 2 things in current_level should be merged together to form a new, single unit
        # Those 2 things are replaced with their merged version in current_level, and that process repeats, 
        # with each time step merging at least one pair of adjacent elements in current_level to create a 
        # single new element, until current_level only contains one element. This element is then a single 
        # embedding for the whole tree, and it is what is returned.
        current_level = embedded_seq
        for level in tree:
            next_level = []

            for node in level:

                if len(node) == 1:
                    next_level.append(current_level[node[0]])
                    continue
                left = node[0]
                right = node[1]


                r_l = nn.Sigmoid()(self.l_rl(current_level[left]) + self.r_rl(current_level[right]))
                r_r = nn.Sigmoid()(self.l_rr(current_level[left]) + self.r_rr(current_level[right]))
                z_l = nn.Sigmoid()(self.l_zl(current_level[left]) + self.r_zl(current_level[right]))
                z_r = nn.Sigmoid()(self.l_zr(current_level[left]) + self.r_zr(current_level[right]))
                z = nn.Sigmoid()(self.l_z(current_level[left]) + self.r_z(current_level[right]))
                h_tilde = nn.Tanh()(self.l(r_l * current_level[left]) + self.r(r_r * current_level[right]))
                hidden = z_l * current_level[left] + z_r * current_level[right] + z * h_tilde

                next_level.append(hidden)

            current_level = next_level

        return current_level[0], current_level[0], current_level[0]

# Tree-based decoder
# This implements the binary tree decoder of Chen et al. 2018, described in
# section 3.2 of this paper: http://papers.nips.cc/paper/7521-tree-to-tree-neural-networks-for-program-translation.pdf
# The only difference is that we have implemented it as a GRU instead of an LSTM, but this is
# a straightforward modification of their setup
class TreeDecoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(TreeDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.word_out = nn.Linear(hidden_size, vocab_size)
        self.rnn_l = nn.GRU(hidden_size, hidden_size)
        self.rnn_r = nn.GRU(hidden_size, hidden_size)

    def forward(self, hidden, encoder_outputs, training_set, tf_ratio=0.5, evaluation=False): 
        encoding = hidden
        tree = training_set[3]

        tree_to_use = tree[::-1][1:]

        current_layer = [encoding]

        # This works in revers of the tree encoder: start with a single vector encoding, then 
        # output 2 children from it, and repeat until the whole tree has been generated
        for layer in tree_to_use:
            next_layer = []
            for index, node in enumerate(layer):
                if len(node) == 1:
                    next_layer.append(current_layer[index])
                else:

                    output, left = self.rnn_l(Variable(torch.zeros(1,1,self.hidden_size)).to(device=available_device), current_layer[index])
                    output, right = self.rnn_r(Variable(torch.zeros(1,1,self.hidden_size)).to(device=available_device), current_layer[index])

                    next_layer.append(left)
                    next_layer.append(right)
            current_layer = next_layer

        # Apply a linear layer to each leaf embedding to determine what word is at that leaf
        words_out = []
        for elt in current_layer:
            words_out.append(nn.LogSoftmax()(self.word_out(elt).view(-1).unsqueeze(0)))

        return words_out


class TridentDecoder(nn.Module):
    def __init__(self, arity, vocab_size, hidden_size, max_depth, null_placement="pre"):
        super(TridentDecoder, self).__init__()

        self.arity = arity
        self.null_placement = null_placement

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size + 1 # first one is for null
        self.max_depth = max_depth

        self.hidden2vocab = nn.Sequential(nn.Linear(self.hidden_size, 2*self.hidden_size), nn.Sigmoid(), nn.Linear(2*self.hidden_size, self.vocab_size))
        self.to_children = nn.Sequential(nn.Linear(self.hidden_size, 2*self.hidden_size), nn.Sigmoid(), nn.Linear(2*self.hidden_size, self.arity * self.hidden_size))

    @property
    def null_ix(self):
        return 0

    def forward(self, root_hidden, tree=None):
        batch_size = root_hidden.shape[0]
        batch_outs = []
        for eg_ix in range(batch_size):
            batch_outs.append(self.forward_nobatch(root_hidden[eg_ix, :], tree=(None if tree is None else tree[eg_ix])))

        return nn.utils.rnn.pad_sequence(batch_outs)

    def forward_nobatch(self, root_hidden, tree=None):
        if self.training:
            # assumption: every non-terminal node either has 3 children which are all non-terminals, or is the parent of one terminal node
            assert tree is not None
            raw_scores = torch.stack(list(self.forward_train_helper(root_hidden, tree)))
            return F.log_softmax(raw_scores, dim=1)
        else:
            raw_scores = torch.stack(list(self.forward_eval_helper(root_hidden)))
            #return F.log_softmax(raw_scores, dim=1)
            return raw_scores

    def forward_train_helper(self, root_hidden, root):
        production = self.hidden2vocab(root_hidden)
        if len(root) > 1:
            assert len(root) == self.arity

            assert self.null_placement == "pre"
            yield production
            children_hidden = self._hidden2children(root_hidden)
            for child_ix in range(self.arity):
                yield from self.forward_train_helper(children_hidden[child_ix], root[child_ix])
        else:
            yield production

    def forward_eval_helper(self, root_hidden, depth=0):
        production = self.hidden2vocab(root_hidden)

        if (torch.argmax(production) == self.null_ix) and (depth <= self.max_depth):
            # we chose NOT to output a word here... recurse more
            children_hidden = self._hidden2children(root_hidden)
            for child_ix in range(self.arity):
                yield from self.forward_eval_helper(children_hidden[child_ix], depth+1)
        else:
            yield production

    def _hidden2children(self, hidden):
        return torch.split(self.to_children(hidden), self.hidden_size, dim=-1)

"""
new decoders will take arguments
encoding vector
instruction token
(optional) teacher forcing tree
"""

class GRUTridentDecoder(nn.Module):
    def __init__(self, arity, vocab_size, hidden_size, max_depth, null_placement="pre"):
        super(GRUTridentDecoder, self).__init__()

        self.arity = arity
        self.null_placement = null_placement

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size + 1 # first one is for null
        self.max_depth = max_depth

        self.hidden2vocab = nn.Sequential(nn.Linear(self.hidden_size, 2*self.hidden_size), nn.Sigmoid(), nn.Linear(2*self.hidden_size, self.vocab_size))
        
        # IDEA: in the future, have just one GRU for all the children but feed in a different input for each instead of always using this same dumb vector 
        self.null_vector = torch.zeros(self.hidden_size, requires_grad=False).unsqueeze(0)
        self.per_child_cell = nn.ModuleList()
        for _ in range(self.arity):
            self.per_child_cell.append(nn.GRUCell(self.hidden_size, self.hidden_size))
        
    @property
    def null_ix(self):
        return 0

    def forward(self, root_hidden, training_set):
        root_hidden = root_hidden.unsqueeze(0)
        if self.training:
            # assumption: every non-terminal node either has 3 children which are all non-terminals, or is the parent of one terminal node
            tree = training_set[3]
            raw_scores = torch.stack(list(self.forward_train_helper(root_hidden, tree)))
            return F.log_softmax(raw_scores, dim=1)
        else:
            raw_scores = torch.stack(list(self.forward_eval_helper(root_hidden)))
            return F.log_softmax(raw_scores, dim=1)

    def forward_train_helper(self, root_hidden, root):
        production = self.hidden2vocab(root_hidden)[0]
        if len(root) > 1:
            assert len(root) == self.arity

            assert self.null_placement == "pre"
            yield production
            for child_ix in range(self.arity):
                child_hidden = self.per_child_cell[child_ix](self.null_vector, root_hidden)
                yield from self.forward_train_helper(child_hidden, root[child_ix])
        else:
            yield production

    def forward_eval_helper(self, root_hidden, depth=0):
        production = self.hidden2vocab(root_hidden)[0]
        if (torch.argmax(production) == self.null_ix) and (depth <= self.max_depth):
            # we chose NOT to output a word here... recurse more
            for child_ix in range(self.arity):
                child_hidden = self.per_child_cell[child_ix](self.null_vector, root_hidden)
                yield from self.forward_eval_helper(child_hidden, depth+1)
        else:
            yield production

class AltGRUTridentDecoder(nn.Module):
    def __init__(self, arity, vocab_size, hidden_size, max_depth, null_placement="pre"):
        super(GRUTridentDecoder, self).__init__()

        self.arity = arity
        self.null_placement = null_placement

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size + 1 # first one is for null
        self.max_depth = max_depth

        self.hidden2vocab = nn.Sequential(nn.Linear(self.hidden_size, 2*self.hidden_size), nn.Sigmoid(), nn.Linear(2*self.hidden_size, self.vocab_size))
        
        # IDEA: in the future, have just one GRU for all the children but feed in a different input for each instead of always using this same dumb vector 
        self.null_vector = torch.zeros(self.hidden_size, requires_grad=False).unsqueeze(0)
        self.child_toks = nn.ModuleList()
        for _ in range(self.arity):
            pass
            #self.child_toks.append()
        
    @property
    def null_ix(self):
        return 0

    def forward(self, root_hidden, training_set):
        root_hidden = root_hidden.unsqueeze(0)
        if self.training:
            # assumption: every non-terminal node either has 3 children which are all non-terminals, or is the parent of one terminal node
            tree = training_set[3]
            raw_scores = torch.stack(list(self.forward_train_helper(root_hidden, tree)))
            return F.log_softmax(raw_scores, dim=1)
        else:
            raw_scores = torch.stack(list(self.forward_eval_helper(root_hidden)))
            return F.log_softmax(raw_scores, dim=1)

    def forward_train_helper(self, root_hidden, root):
        production = self.hidden2vocab(root_hidden)[0]
        if len(root) > 1:
            assert len(root) == self.arity

            assert self.null_placement == "pre"
            yield production
            for child_ix in range(self.arity):
                child_hidden = self.per_child_cell[child_ix](self.null_vector, root_hidden)
                yield from self.forward_train_helper(child_hidden, root[child_ix])
        else:
            yield production

    def forward_eval_helper(self, root_hidden, depth=0):
        production = self.hidden2vocab(root_hidden)[0]
        if (torch.argmax(production) == self.null_ix) and (depth <= self.max_depth):
            # we chose NOT to output a word here... recurse more
            for child_ix in range(self.arity):
                child_hidden = self.per_child_cell[child_ix](self.null_vector, root_hidden)
                yield from self.forward_eval_helper(child_hidden, depth+1)
        else:
            yield production
