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

def score(val_iterator, encoder, decoder):
    """
    Compare validation target and prediction to get the accuracy of the
    model. If a target tensor (corresponding to a sentence) equals the 
    predicted tensor, the # of correct results is incremented by 1.

    NB: It appears that calculating score at present is very expensive 
    time-wise; most of the time for each run is spent inside the 
        for batch in val_iterator:
    loop. Maybe there's a more efficient way to do this?

    @param val_iterator: Validation iterator
    @param encoder: Model encoder
    @param decoder: Model decoder

    @returns: (# correct predictions) / (# predictions)
    """
    
    right, total = 0, 0

    for batch in val_iterator:

        prediction = evaluate(encoder, decoder, batch)
        for sents in zip(prediction, batch.target.transpose_(0,1)):
            # print("Prediction: ", sents[0])
            # print("Correct:    ", sents[1])
            if torch.equal(sents[0], sents[1]):
                right += 1
            total += 1

    # print("Total Count: ", total)
    # print("Dataset Len: ", len(val_iterator.dataset))
    return right * 1.0 / total
    
# Given a batch as input, get the decoder's outputs (as argmax indices)
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