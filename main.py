# seq2seq.py
# Main file for seq2seq model. Adapted from Tom McCoy's repository.

import os
import argparse

from torchtext.data import Field, TabularDataset, BucketIterator, RawField

#from models import EncoderRNN, TreeEncoderRNN, DecoderRNN, TreeDecoderRNN
from decoder import DecoderRNN
from encoder import EncoderRNN
import training
import RPNTask
import seq2seq

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import cox.store
from tqdm import tqdm
from tree_loaders import TreeField, TreeSequenceField

# these are as good as constants
LOGS_TABLE = "logs"
META_TABLE = "metadata"
CKPT_NAME_LATEST = "latest_ckpt.pt"
CKPT_NAME_BEST = "best_ckpt.pt"
def setup_store(args):
    store = cox.store.Store(args.outdir, args.expname)

    if args.expname is None:
        # store metadata
        args_dict = args.__dict__
        meta_schema = cox.store.schema_from_dict(args_dict)
        store.add_table(META_TABLE, meta_schema)
        store[META_TABLE].append_row(args_dict)
        # NEW FROM NA
        logging_meters = {
            args.sentacc: None,
            args.tokenacc: None,
            args.lenacc: None,
        }
        # TODO: support columns that aren't float
        logs_schema = {name: float for name, meter in logging_meters.items()}
        store.add_table(LOGS_TABLE, logs_schema)
    else:
        # TODO: check that the parameters match
        old_meta_schema = store[META_TABLE].schema
        old_args = dict(store[META_TABLE].df.loc[0, :])

    return store

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--encoder', help = 'type of encoder used', choices = ['GRU', 'Tree'], type = str, default = 'GRU')
    parser.add_argument('-d', '--decoder', help = 'type of decoder used', choices = ['GRU', 'Tree'], type = str, default = 'GRU')
    parser.add_argument('-t', '--task', help = 'task model is trained to perform', choices = ['negation'], type = str, default = None, required = True)
    parser.add_argument('-a', '--attention', help = 'type of attention used', choices = ['content'], type = str, default = 'content', required = False)
    parser.add_argument('-lr', '--learning-rate', help = 'learning rate', type = float, default = 0.001)
    parser.add_argument('-hs', '--hidden-size', help = 'hidden size', type = int, default = 256)
    #parser.add_argument('-rs', '--random-seed', help='random seed', type=float, default=None)
    #parser.add_argument('-ps', '--parse-strategy', help = 'how to parse (WHAT IS IT PARSING??)', type = str, choices = ['correct', 'right-branching'], default = 'correct')
    parser.add_argument('-p', '--patience', help = 'parience', type = int, default = 3)
    parser.add_argument('-v', '--vocab', help = 'vocabulary used ?? (THIS SHOULD BE CLARIFIED)', type = str, choices = ['SRC', 'TRG'], default = 'SRC')
    parser.add_argument('-pr', '--print-every',	help = 'print training data out after N iterations', metavar = 'N', type = int, default = 1000)
    parser.add_argument('--tree', help='parse sequences as string representations of tree', type=bool, default=False)
    parser.add_argument('-ep', '--epochs', help="num epochs", type=int, default=20)
    parser.add_argument('-b', '--batches', help='batch size', type=int, default=5)
    #     args.logging_meters = { NA
    #     "sentence_accuracy": None,
    #     "token_accuracy": None,
    #     "length_accuracy": None,
    # }
    # args.exp_name = None
    parser.add_argument('-SA', '--sentacc', help='sentence accuracy', type=None, default=None)
    parser.add_argument('-TA', '--tokenacc', help='token accuracy', type=None, default=None)
    parser.add_argument('-LA', '--lenacc', help='length accuracy', type=None, default=None)
    parser.add_argument('-exp', '--expname', help='exp_name', type=None, default=None)

    parser.add_argument('-out', '--outdir', help='directory in which to place cox store', type=None, default='logs/default')
    return parser.parse_args()

def main():
    args = parse_arguments()
 
    store = setup_store(parse_arguments())

    # Device specification
    available_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Create Datasets
    trainingdata = args.task + '.train'
    validationdata = args.task + '.val'
    testingdata = args.task + '.test'
    # NA 80-93
    if args.tree:
        SRC_TREE = TreeField(collapse_unary=True)
        SRC = TreeSequenceField(SRC_TREE)
        TRANS = RawField()
        TAR_TREE = TreeField(collapse_unary=True)
        TAR = TreeSequenceField(TAR_TREE, inner_order="pre", inner_symbol="NULL", is_target=True)
        datafields = [(("source_tree", "source"), (SRC_TREE, SRC)), 
            ("annotation", TRANS), 
            (("target_tree", "target"), (TAR_TREE, TAR))]
    else:
        SRC = Field(sequential=True, lower=True) # Source vocab
        TRG = Field(sequential=True, lower=True) # Target vocab
        TRANS = SRC if args.vocab == "SRC" else TRG
        datafields = [("source", SRC), ("annotation", TRANS), ("target", TRG)]

    train, valid, test = TabularDataset.splits(
        path = "data",
        train = trainingdata,
        validation = validationdata,
        test = testingdata,
        format = 'csv',
        skip_header = True,
        fields = datafields
    )

    SRC.build_vocab(train, valid, test)
    TRG.build_vocab(train, valid, test)

    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train, valid, test), 
        batch_size = 5, 
        device = available_device, 
        sort_key = lambda x: len(x.target), 
        sort_within_batch = True, 
        repeat = False
    )

    encoder = EncoderRNN(len(SRC.vocab), hidden_size=args.hidden_size, recurrent_unit=args.encoder, num_layers=1, max_length=30)
    #dec = modelsNEW.TridentDecoder(3, len(TAR.vocab), hidden_size=30, max_depth=5)
    #s2s = seq2seq.Seq2Seq(encoder, dec, ["source"], ["middle1"], decoder_train_field_names=["middle1", "source_tree"])
    #dec = DecoderRNN(hidden_size=HIDDEN_SIZE,vocab=TAR_SEQ.vocab, encoder_vocab=SRC_SEQ.vocab, recurrent_unit=DEC_TYPE, num_layers=NUM_LAYERS, max_length=MAX_LENGTH, attention_type=ATT_TYPE, dropout=DROPOUT)
    dec = DecoderRNN(hidden_size=args.hidden_size, vocab=TRG.vocab, encoder_vocab=SRC.vocab, recurrent_unit=args.decoder, num_layers=1, max_length=30, attention_type='additive', dropout=0.3)
    s2s = seq2seq.Seq2Seq(encoder, dec, ["source"], ["middle0", "annotation", "middle1", "source"], decoder_train_field_names=["middle0", "annotation", "middle1", "source", "target"])
    
    if CKPT_NAME_LATEST in os.listdir(store.path):
        ckpt_path = os.path.join(store.path, CKPT_NAME_LATEST)
        s2s.load_state_dict(torch.load(ckpt_path))

    training.train(s2s, train_iter, val_iter, store, args, ignore_index=TRG.vocab.stoi['<pad>'])

import warnings
warnings.warn("""If you have Pandas 1.0 you must make the following change manually for cox to work:
https://github.com/MadryLab/cox/pull/3/files
Use cox.__files__ to find where cox is installed""")

if __name__ == '__main__':
    main()