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
import numpy as np

import sys
import os

# Functions for tracking time
import time
import math

#from evaluationNEW import *

from abc import ABC, abstractmethod
 
CKPT_NAME_LATEST = "latest_ckpt.pt"

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
        return 1.0 * self.sum / self.n_total 

class SentenceLevelAccuracy(AverageMetric):

    def process_batch(self, prediction, target, model):  
        correct = (prediction == target).prod(axis=0).sum()    
        total = target.size()[1]
        self.sum += (prediction == target).prod(axis=0).sum()
        self.n_total += target.size()[1]

        if correct > total:
            p = model.scores2sentence(prediction, model.decoder.vocab)
            t = model.scores2sentence(target, model.decoder.vocab)

            print("I got {} / {} right this time!".format(correct, total))
            print("Predictions:\n", p)
            print("Target:\n", t)
            print(prediction == target)
            print((prediction == target).prod(axis=0))

            raise(SystemError)

class TokenLevelAccuracy(AverageMetric):

    def process_batch(self, prediction, target, model): 
        self.sum += (prediction == target).sum()
        self.n_total += target.size()[0] * target.size()[1]

class LengthLevelAccuracy(AverageMetric):

    def __init__(self):
        AverageMetric.__init__(self)
        self.n_total = 1

    def process_batch(self, prediction, target, model): 
        pass

# TODO: can we work this into the Metric hierarchy ^
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val):
        # self.val = val
        self.sum += val
        self.count += 1
    
    def result(self):
        return self.sum / self.count if self.count > 0 else np.nan

def test(model, test_iter, task):

    model.eval()

    with open('{0}-test-results.txt'.format(task), 'w') as f:
        f.write('{0}\t{1}\t{2}\n'.format('Sentence', 'Target', 'Prediction'))

    with torch.no_grad():
        for batch in test_iter:

            logits = model(batch)
            target = batch.target 
            predictions = logits[:target.size()[0], :].argmax(2)

            sentences = model.scores2sentence(batch.source, model.encoder.vocab)
            predictions = model.scores2sentence(predictions, model.decoder.vocab)
            target = model.scores2sentence(target, model.decoder.vocab)

            with open('{0}-test-results.txt'.format(task), 'a') as f:
                for i, _ in enumerate(sentences):
                    f.write('{0}\t{1}\t{2}\n'.format(
                        sentences[i], target[i], predictions[i])
                    )

def evaluate(model, val_iter, criterion=None, logging_meters=None, store=None):

    model.eval()
    stats_dict = dict()

    loss_meter = AverageMeter()

    with torch.no_grad():

        for batch in val_iter:

            # run the model
            logits = model(batch) # seq length (of pred) x batch_size x vocab
            target = batch.target # seq length (of target) x batch_size

            l = logits[:target.size()[0], :].permute(0, 2, 1)
            pred = logits[:target.size()[0], :].argmax(2)

            batch_loss = criterion(l, target)
            loss_meter.update(batch_loss)

            for _, meter in logging_meters.items():
                meter.process_batch(pred, target, model)

        for name, meter in logging_meters.items():
            stats_dict[name] = meter.result()

        stats_dict['loss'] = loss_meter.result()

        # if store is not None:
            # print(store)
            # store["logs"].append_row(stats_dict)

    return stats_dict

def train(model, train_iterator, validation_iter, logging_meters, store, args, ignore_index=None):

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(weight=None, ignore_index=ignore_index)
    
    # if we DON"T want to organize by epochs, go to this tutorial and CMD-F "epoch": <http://anie.me/On-Torchtext/>
    for epoch in range(args.epochs):
    
        loss_meter = AverageMeter()
        new_meters = dict()
        new_meters['sentence-level-accuracy'] = SentenceLevelAccuracy()
        new_meters['token-level-accuracy'] = TokenLevelAccuracy()
        new_meters['length-accuracy'] = LengthLevelAccuracy()

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
                T.set_postfix(avg_train_loss=loss_meter.result())

        # dictionary of stat_name => value
        eval_stats = evaluate(model, validation_iter, criterion, logging_meters=new_meters, store=store)
        for name, stat in eval_stats.items():
            print('{:<25s} {:f}'.format(name, stat))

        torch.save(model.state_dict(), os.path.join(store.path, CKPT_NAME_LATEST))

