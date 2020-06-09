import torchtext
import tree_loaders
import RPNTask
import modelsNEW
import seq2seq

def my_transform(t):
    if t[0].label() == "RPN":
        return RPNTask.polish2reversePolish(t)
    elif t[0].label() == "POLISH":
        return t
    else:
        assert False, "tree not annotated as expected"


SRC = tree_loaders.TreeField(collapse_unary=True)
SRC_SEQ = tree_loaders.TreeSequenceField(SRC)

EXTRACT = tree_loaders.TreeExtractorField(tree_transformation_fun=lambda t: t[0], collapse_unary=False)
TAR = tree_loaders.TreeField(my_transform, collapse_unary=True)
TAR_SEQ = tree_loaders.TreeSequenceField(TAR, inner_order="pre", inner_symbol="NULL")

fields2 = {"source_tree": SRC, "source_seq": SRC_SEQ, "annotation": EXTRACT, 
           "target_tree": TAR, "target_seq": TAR_SEQ}

dataset2 = tree_loaders.PCFGDataset(RPNTask.polish_annotated, 200, fields2, min_length=5, max_length=15, seed=42)

SRC_SEQ.build_vocab(dataset2)
TAR_SEQ.build_vocab(dataset2)

myiter = torchtext.data.Iterator(dataset2, 5, repeat=False)


import numpy as np
import torch
import torch.nn as nn
from torch import optim
import cox.store

from tqdm import tqdm

# taken from here (thanks to the Madry Lab!) 
# https://github.com/MadryLab/robustness/blob/master/robustness/tools/helpers.py
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

def train(iterator, model, learning_rate, num_epochs, ignore_index):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(weight=None, ignore_index=ignore_index)
    
    # if we DON"T want to organize by epochs, go to this tutorial and CMD-F "epoch": <http://anie.me/On-Torchtext/>
    for epoch in range(num_epochs):
    
        loss_meter = AverageMeter()
        model.train()
        with tqdm(iterator) as T:
            for batch in T:
                optimizer.zero_grad()

                decoder_outputs = model(batch)
                
                # TODO: double check this
                pred = decoder_outputs.permute(1, 2, 0)
                target = batch.target_seq.permute(1, 0)
                batch_loss = criterion(pred, target)

                batch_loss.backward()
                optimizer.step()

                # item() to turn a 0-dimensional tensor into a regular float
                loss_meter.update(batch_loss.item())
                T.set_postfix(avg_train_loss=loss_meter.avg)

def save_checkpoint(store, filename, info):
    ckpt_save_path = os.path.join(store.path, filename)
    torch.save(info, ckpt_save_path)

OUT_DIR = "logs/default"
LOGS_TABLE = "logs"
CKPTS_TABLE = "ckpts"
CKPT_NAME_LATEST = "latest_ckpt.pt"
CKPT_NAME_BEST = "best_ckpt.pt"



encoder = modelsNEW.EncoderRNN(len(SRC_SEQ.vocab), hidden_size=30, recurrent_unit="GRU", n_layers=1, max_length=20)
dec = modelsNEW.TridentDecoder(3, len(TAR_SEQ.vocab), hidden_size=30, max_depth=5)
s2s = seq2seq.Seq2Seq(encoder, dec, ["source_seq"], ["middle1"], decoder_train_field_names=["middle1", "source_tree"])


train(myiter, s2s, 0.01, 20, ignore_index=TAR_SEQ.vocab.stoi['<pad>'])