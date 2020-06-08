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


# Functions for evaluation

#TODO: confirm that loss() and sent_remove_brackets() are never used

# Get the model's full-sentence accuracy on a set of examples
def score(val_iterator, encoder1, decoder1, vocab):
    right = 0
    total = 0

    for batch in val_iterator:

        prediction = evaluate(encoder1, decoder1, batch).transpose_(0,1)

        all_sents = logits_to_sentence(prediction, vocab)
        correct_sents = logits_to_sentence(batch.target, vocab)

        for sents in zip(prediction, batch.target):
            # print("Prediction: ", sents[0])
            # print("Correct:    ", sents[1])

            # raise(SystemError)
            if torch.equal(sents[0], sents[1]):
                right += 1
            total += 1

    return right * 1.0 / total
    
# Given a batch as input, get the decoder's outputs (as argmax indices)
MAX_EXAMPLE = 10000
def evaluate(encoder, decoder, batch, max_length=30):
    encoder_output, encoder_hidden, encoder_outputs = encoder(batch)

    decoder_hidden = encoder_hidden

    decoder_outputs = decoder(decoder_hidden, encoder_outputs, batch, tf_ratio=0.0, evaluation=True)


    output_indices = []
    for logit in decoder_outputs:
        topv, topi = logit.data.topk(1)

        output_indices.append(torch.stack([elt[0] for elt in topi]))

        if 1 in topi or 2 in topi:
            break

    return torch.stack(output_indices)

# Convert logits to a sentence
def logits_to_sentence(batch, vocab):
    """
    Returns a list of sentences [list of words] based on the supplied vocab.

    It is necessary to transpose the input batch so that each tensor slice
    represents the ith sentence and not the ith position in all sentences.
    """

    batch_sents = []
    batch.transpose_(0,1)

    for s in batch:
        batch_sents.append([vocab.vocab.itos[i] for i in s])

    return batch_sents
  


