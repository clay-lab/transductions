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

# def score(iterator, encoder, decoder):
#     """
#     Compare validation target and prediction to get the accuracy of the
#     model. If a target tensor (corresponding to a sentence) equals the 
#     predicted tensor, the # of correct results is incremented by 1.

#     NB: It appears that calculating score at present is very expensive 
#     time-wise; most of the time for each run is spent inside the 
#         for batch in val_iterator:
#     loop. Maybe there's a more efficient way to do this?

#     @param iterator: Validation iterator
#     @param encoder: Model encoder
#     @param decoder: Model decoder

#     @returns: (# correct predictions) / (# predictions)
#     """
    
#     right, total = 0, 0

#     for batch in iterator:

#         prediction = evaluate(encoder, decoder, batch)
#         for sents in zip(prediction, batch.target):
#             # print("Prediction: ", sents[0])
#             # print("Correct:    ", sents[1])
#             if torch.equal(sents[0], sents[1]):
#                 right += 1
#             total += 1
#     # print("Total Count: ", total)
#     # print("Dataset Len: ", len(val_iterator.dataset))
#     return right * 1.0 / total

# Given a batch as input, get the decoder's outputs (as argmax indices)
def evaluate(model, validation_iter, store=store, max_length=30):

    right, total = 0, 0

    for batch in validation_iter:
        model(batch)
        decoder_outputs = model(batch)
                
        # TODO: double check this
        prediction = decoder_outputs.permute(1, 2, 0)
        target = batch.target.permute(1, 0)

        for sents in zip(prediction, target):
            if torch.equal(sents[0], sents[1]):
                right += 1
            total += 1
    # print("Total Count: ", total)
    # print("Dataset Len: ", len(val_iterator.dataset))
    return right * 1.0 / total
