
# This code was adapted from Tom Mccoy's model found at https://github.com/tommccoy1/rnn-hierarchical-biases
# You must have PyTorch installed to run this code.
# You can get it from: http://pytorch.org/

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


from evaluation import *
from models import *
from parsing import *
from training import *


import argparse
from torchtext.data import Field, BucketIterator

parser = argparse.ArgumentParser()
parser.add_argument("--encoder", help="encoder type", type=str, default=None)
parser.add_argument("--decoder", help="decoder type", type=str, default=None)
parser.add_argument("--task", help="task", type=str, default=None)
parser.add_argument("--attention", help="attention type", type=str, default=None)
parser.add_argument("--lr", help="learning rate", type=float, default=None)
parser.add_argument("--hs", help="hidden size", type=int, default=None)
parser.add_argument("--seed", help="random seed", type=float, default=None)
parser.add_argument("--parse_strategy", help="whether to parse correctly or right-branching", type=str, default="correct")
parser.add_argument("--patience", help="patience", type=int, default=3)
# TODO: add argument for trans vocab
args = parser.parse_args()



prefix = args.task
if args.parse_strategy == "right_branching":
    directory = "models/" + args.task + "_" + args.encoder + "_" + args.decoder  + "_" + "RB" + "_" + args.attention + "_" + str(args.lr) + "_" + str(args.hs)
else:
    directory = "models/" + args.task + "_" + args.encoder + "_" + args.decoder  + "_" + args.attention + "_" + str(args.lr) + "_" + str(args.hs)

use_cuda = torch.cuda.is_available()

if use_cuda:
    available_device = torch.device('cuda')
else:
    available_device = torch.device('cpu')

# Assign Files
trainingdata = 'data/' + prefix + '.train'
validationdata = 'data/' + prefix + '.val'
testingdata = 'data/' + prefix + '.test'


# Assign Fields
SRC = Field(sequential=True, lower=True)
TRG = Field(sequential=True, lower=True)

# Define data fields
# TODO: whether trans is included in SRC or TRG vocab
# TODO: MAKE SURE THIS IS INCLUDED IN THE ENCODER/DECODER
# TODO: INCLUDE THE TRANS VOCAB IN THE CLA
datafields = [("source", SRC), ("trans", TRG), ("target", TRG)]



# Load data into dataset
train, valid, test = TabularDataset.splits(path="data", train=trainingdata, validation=validationdata, test=testingdata, format='csv', skip_header=True, fields=datafields)

# Build vocab
SRC.build_vocab(train, valid, test)
TRG.build_vocab(train, valid, test)



# Build iterators
train_iter, val_iter, test_iter = BucketIterator.splits(
    (train, valid, test), 
    batch_size=5, 
    device=available_device, 
    #TODO: Confirm sort_key is dynamic
    sort_key=lambda x: len(x.target), 
    sort_within_batch=True, 
    repeat=False
)
# get vocabs:
# TODO: use ___.vocab.stoi in score function instead of index2text
# print("SRC VOCAB: ", SRC.vocab.stoi, "\n")
# print("TRG VOCAB: ", TRG.vocab.stoi, "\n")


# print("TRANS VOCAB: ", trans.vocab.stoi, "\n")
# print("SRC: ", SRC.vocab.itos)
# print("TRG: ", TRG.vocab.itos)


# Create a directory where the outputs will be saved
if __name__ == "__main__":
    wait_time = random.randint(0, 99)
    time.sleep(wait_time)

    counter = 0
    dir_made = 0

    while not dir_made:
        if not os.path.exists(directory + "_" +  str(counter)):
            directory = directory + "_" + str(counter)
            os.mkdir(directory)
            dir_made = 1

        else:
            counter += 1

    # Implement the random seed
    if args.seed is None:
        random_seed = counter

    random.seed(random_seed)

#***********#
if __name__ == "__main__":

        max_len = max(len(SRC.vocab), len(TRG.vocab))
        # Initialize the encoder and the decoder
        if args.encoder == "Tree":
            encoder = TreeEncoderRNN(max_len, args.hs)
        else:
            encoder = EncoderRNN(max_len, args.hs, args.encoder, max_length=MAX_LENGTH)

        if args.decoder == "Tree":
            # Note that attention is not implemented for the tree decoder
            decoder = TreeDecoderRNN(max_len, args.hs)
        else:
            decoder = DecoderRNN(args.hs, max_len, args.decoder, attn=args.attention, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH)


        encoder = encoder.to(device=available_device)
        decoder = decoder.to(device=available_device)

        # Give torch a random seed
        torch.manual_seed(random_seed)
        if use_cuda:
            torch.cuda.manual_seed_all(random_seed)	


        # Train the model
        trainIters(encoder, decoder, 10000000, args.encoder, args.decoder, args.attention, train_batches, dev_batches, index2word, directory, prefix, print_every=1000, learning_rate=args.lr, patience=args.patience)
        




