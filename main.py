import RPNTask
import modelsNEW
import seq2seq
import modelsNEWBob

import numpy as np
import torch
import torch.nn as nn
from torch import optim
#import cox.store

from tqdm import tqdm

from torchtext.data import RawField, Field, TabularDataset, BucketIterator
from tree_loaders import TreeField, TreeSequenceField

# these are as good as constants
LOGS_TABLE = "logs"
META_TABLE = "metadata"
CKPT_NAME_LATEST = "latest_ckpt.pt"
CKPT_NAME_BEST = "best_ckpt.pt"

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

        eval_stats = evaluate(model, validation_iter)

        torch.save(model.state_dict(), os.path.join(store.path, CKPT_NAME_LATEST))




# this is being nixed until cox gets fixed for pandas 1.0
"""
def setup_store(args):
    store = cox.store.Store(args.out_dir, args.exp_name)

    # store metadata
    args_dict = args.__dict__
    meta_schema = cox.store.schema_from_dict(args_dict)
    store.add_table(META_TABLE, meta_schema)
    store[META_TABLE].append_row(args_dict)

    # TODO: support columns that aren't float
    logs_schema = {name: float for name, meter in args.logging_meters.items()}
    store.add_table(LOGS_TABLE, logs_schema)

    return store
"""

if __name__ == "__main__":
    # args is just a blank object.
    # will be replaced by actual command line argument with appropriate defaults
    args = type('', (), {})()
    args.read_tree_format = True
    args.data_filename = "demo_savefile.txt"
    args.batch_size = 5
    args.out_dir = "logs/default"
    args.learning_rate = 0.01
    args.num_epochs = 20
    """
    args.logging_meters = {
        "sentence_accuracy": None,
        "token_accuracy": None,
        "length_accuracy": None,
    }
    """
    # args.exp_name = None

    if args.read_tree_format:
        SRC_TREE = TreeField(collapse_unary=True)
        SRC = TreeSequenceField(SRC_TREE)
        ANNOT = RawField()
        TAR_TREE = TreeField(collapse_unary=True)
        TAR = TreeSequenceField(TAR_TREE, inner_order="pre", inner_symbol="NULL", is_target=True)
        fields = [(("source_tree", "source"), (SRC_TREE, SRC)), 
            ("annotation", ANNOT), 
            (("target_tree", "target"), (TAR_TREE, TAR))]
    else:
        SRC = Field()
        ANNOT = RawField()
        TAR = Field(is_target=True)
        fields = [("source", SRC), ("annotation", ANNOT), ("target", TAR)]

    train_dataset = TabularDataset(args.data_filename, "tsv", fields)

    SRC.build_vocab(train_dataset)
    TAR.build_vocab(train_dataset)

    train_iter = BucketIterator(train_dataset, args.batch_size)

    encoder = modelsNEW.EncoderRNN(len(SRC.vocab), hidden_size=30, recurrent_unit="GRU", n_layers=1, max_length=20)
    dec = modelsNEW.TridentDecoder(3, len(TAR.vocab), hidden_size=30, max_depth=5)
    s2s = seq2seq.Seq2Seq(encoder, dec, ["source"], ["middle1"], decoder_train_field_names=["middle1", "source_tree"])

    #store = setup_store(args)

    train(s2s, train_iter, store, args, ignore_index=TAR.vocab.stoi['<pad>'])