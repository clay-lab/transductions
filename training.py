#from __future__ import unicode_literals, print_function, division

# Imports
import os.path

import torch
from torch import optim
from tqdm import tqdm

from nltk.parse import ViterbiParser
import nltk.grammar

import RPNTask

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

#from evaluationNEW import *

from abc import ABC, abstractmethod
 
class AbstractMetric(ABC):
    @abstractmethod
    def process_batch(self, prediction, target):
        pass
 
    @abstractmethod
    def result(self):
        pass

class AverageMetric(AbstractMetric):
    def __init__(self):
        self.sum = 0
        self.n_total = 0

    def result(self):
        return self.n_correct / self.n_total 

class SentenceLevelAccuracy(AverageMetric):
    def process_batch(self, prediction, target):
        #pred_target = torch.nn.utils.rnn.pad_sequence([pred, target], padding_value=-1)
        pass

def evaluate(model, val_iter, store=None):
    model.eval()

    stats_dict = {}
    store[LOGS_TABLE].append_row(stats_dict)

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        # self.val = 0
        self.avg = -1
        self.sum = 0.0
        self.count = 0.0

    def update(self, val):
        # self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

def train(model, train_iterator, validation_iter, store, args, ignore_index=None):
    # if validation_iter is None:
    #     validation_iter = train_iterator

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(weight=None, ignore_index=ignore_index)
    
    # if we DON"T want to organize by epochs, go to this tutorial and CMD-F "epoch": <http://anie.me/On-Torchtext/>
    for epoch in range(args.epochs):
    
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

