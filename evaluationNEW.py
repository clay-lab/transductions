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
def score(val_iterator, encoder1, decoder1):
    right = 0
    total = 0

    for batch in val_iterator:
        batch_size = len(batch)
        pred_words = evaluate(encoder1, decoder1, batch)

        all_sents = logits_to_sentence(pred_words, index2word)
        correct_sents = logits_to_sentence(batch[1], index2word)

        for sents in zip(all_sents, correct_sents):
            if sents[0] == sents[1]:
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
def logits_to_sentence(pred_words, batch_size=5, end_at_punc=True):

    batch_sents = []
    for i in range(batch_size):
        current_sent = []
        for sentence in batch.source:
            index = sentence[i]
            word = SRC.vocab.itos[index]
            current_sent.append(word)
        batch_sents.append(current_sent)
    return batch_sents
  


