# Imports
import os.path

import torch
from torch import optim
from tqdm import tqdm

from nltk.parse import ViterbiParser
import nltk.grammar

import RPNTask

import models
import modelsNEW
import modelsNEWBob

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

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        # self.val = 0
        self.avg = np.nan
        self.sum = 0.0
        self.count = 0.0

    def update(self, val):
        # self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

def train(model, train_iterator, store, args, validation_iter=None, ignore_index=None):
    if validation_iter is None:
        validation_iter = train_iterator

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(weight=None, ignore_index=ignore_index)
    
    # if we DON"T want to organize by epochs, go to this tutorial and CMD-F "epoch": <http://anie.me/On-Torchtext/>
    for epoch in range(args.num_epochs):
    
        loss_meter = AverageMeter()
        model.train()
        with tqdm(train_iterator) as T:
            for batch in T:
                optimizer.zero_grad()

                decoder_outputs = model(batch)
                
                # TODO: double check this
                pred = decoder_outputs.permute(1, 2, 0)
                target = batch.target.permute(1, 0)
                batch_loss = criterion(pred, target)

                batch_loss.backward()
                optimizer.step()

                # item() to turn a 0-dimensional tensor into a regular float
                loss_meter.update(batch_loss.item())
                T.set_postfix(avg_train_loss=loss_meter.avg)

        # TODO: SHAYNA
        eval_stats = evaluate(model, validation_iter, store=store)

        torch.save(model.state_dict(), os.path.join(store.path, CKPT_NAME_LATEST))

