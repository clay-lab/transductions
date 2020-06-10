
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

from evaluationNEW import *

use_cuda = torch.cuda.is_available()

if use_cuda:
    available_device = torch.device('cuda')
else:
    available_device = torch.device('cpu')

# figure out attention
def train_iterator(train_iterator, val_iterator, encoder, decoder, enc_recurrent_unit, dec_recurrent_unit, attention, directory, prefix, vocab, print_every=1000, learning_rate=0.01, patience=3, epochs = 4):

    print("Training model")

    # Construct optimizers and loss function
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    # Loss values
    total_loss = 0.0

    for epoch in range(epochs):

        print("#####################################")
        print("Epoch {:d} of {:d}".format(epoch, epochs))

        for index, batch in enumerate(train_iterator):

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # Split the target from the input
            # print(batch)
            target = batch.target

            # Forward pass

            # I have some questions about how the Encoders are structured.
            # In particular, why return a mega-tensor of the outputs?
            e_out, e_hid, e_outs = encoder(batch)
            d_outs = decoder(e_hid, e_outs, batch)

            batch_loss = 0.0
            idx = 0
            while idx < min(len(d_outs), len(batch)): # Make this less brittle?
                batch_loss += criterion(d_outs[idx], target[idx])
                idx += 1


            # Backward pass
            # batch_loss *= batch[0][0].size()[1] # Why is this necessary??
            batch_loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += batch_loss / target.size()[0]

            count_since_improved = 0
            best_loss = float('inf')

            if (index % print_every == 0):

                dev_set_loss = 1 - score(val_iterator, encoder, decoder)
                
                print("Batch Loss:   {0:>2f}".format(batch_loss))
                print("Dev-Set Loss: {0:>2f}".format(dev_set_loss))

                # Deal with model saving here