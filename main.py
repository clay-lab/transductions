# seq2seq.py
# Main file for seq2seq model. Adapted from Tom McCoy's repository.

import os
import argparse

from torchtext.data import Field, TabularDataset, BucketIterator, RawField

from models import EncoderRNN, TreeEncoderRNN, DecoderRNN, TreeDecoderRNN
import training
import RPNTask
import modelsNEW
import seq2seq
import modelsNEWBob

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import cox.store
from tqdm import tqdm
import os.path
from tree_loaders import TreeField, TreeSequenceField


def parse_arguments():

	parser = argparse.ArgumentParser()

	parser.add_argument('-e', '--encoder', help = 'type of encoder used', choices = ['GRU', 'Tree'], type = str, default = 'GRU')
	parser.add_argument('-d', '--decoder', help = 'type of decoder used', choices = ['GRU', 'Tree'], type = str, default = 'GRU')
	parser.add_argument('-t', '--task', help = 'task model is trained to perform', choices = ['negation'], type = str, default = None, required = True)
	parser.add_argument('-a', '--attention', help = 'type of attention used', choices = ['content'], type = str, default = None, required = True)
	parser.add_argument('-lr', '--learning-rate', help = 'learning rate', type = float, default = 0.001)
	parser.add_argument('-hs', '--hidden-size', help = 'hidden size', type = int, default = 256)
	parser.add_argument('-rs', '--random-seed', help='random seed', type=float, default=None)
	parser.add_argument('-ps', '--parse-strategy', help = 'how to parse (WHAT IS IT PARSING??)', type = str, choices = ['correct', 'right-branching'], default = 'correct')
	parser.add_argument('-p', '--patience', help = 'parience', type = int, default = 3)
	parser.add_argument('-v', '--vocab', help = 'vocabulary used ?? (THIS SHOULD BE CLARIFIED)', type = str, choices = ['SRC', 'TRG'], default = 'SRC')
	parser.add_argument('-pr', '--print-every',	help = 'print training data out after N iterations', metavar = 'N', type = int, default = 1000)
    # args.read_tree_format = True NA
    parser.add_argument('--tree', help='read_tree_format', type=bool, default=False)
    # args.num_epochs = 20 NA
    parser.add_argument('-ep', '--epochs', help="num epochs", type=int, default=20)
    # args.batch_size = 5 NA
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
	return parser.parse_args()
    


# def create_directories(args):
# 	if args.parse_strategy == "right_branching":
# 		path = args.task + "_" + args.encoder + "_" + args.decoder  + "_" + "RB" + "_" + args.attention + "_" + str(args.learning_rate) + "_" + str(args.hidden_size)
# 		return os.path.join("models", path)
# 	else:
# 		path = args.task + "_" + args.encoder + "_" + args.decoder  + "_" + args.attention + "_" + str(args.learning_rate) + "_" + str(args.hidden_size)
# 		return os.path.join("models", path)

# this is being nixed until cox gets fixed for pandas 1.0


def main():
	
	args = parse_arguments()
	# directory = create_directories(args)

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
        ANNOT = RawField()
        TAR_TREE = TreeField(collapse_unary=True)
        TAR = TreeSequenceField(TAR_TREE, inner_order="pre", inner_symbol="NULL", is_target=True)
        fields = [(("source_tree", "source"), (SRC_TREE, SRC)), 
            ("annotation", ANNOT), 
            (("target_tree", "target"), (TAR_TREE, TAR))]
    # else:
    #     SRC = Field()
    #     ANNOT = RawField()
    #     TAR = Field(is_target=True)
    #     fields = [("source", SRC), ("annotation", ANNOT), ("target", TAR)]

	SRC = Field(sequential=True, lower=True) # Source vocab
	TRG = Field(sequential=True, lower=True) # Target vocab
	TRANS = SRC if args.vocab == "SRC" else TRG
	datafields = [("source", SRC), ("trans", TRANS), ("target", TRG)]

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
    # NA 127-159
    # these are as good as constants
    LOGS_TABLE = "logs"
    META_TABLE = "metadata"
    CKPT_NAME_LATEST = "latest_ckpt.pt"
    CKPT_NAME_BEST = "best_ckpt.pt"
    def setup_store(args):
    store = cox.store.Store(args.out_dir, args.exp_name)

    # store metadata
    args_dict = args.__dict__
    meta_schema = cox.store.schema_from_dict(args_dict)
    store.add_table(META_TABLE, meta_schema)
    store[META_TABLE].append_row(args_dict)
    # NEW FROM NA
    logging_meters = {
        args.sentacc
        args.tokenacc
        args.lenacc
    }
    # TODO: support columns that aren't float
    logs_schema = {name: float for name, meter in logging_meters.items()}
    store.add_table(LOGS_TABLE, logs_schema)

    return store

    encoder = modelsNEW.EncoderRNN(len(SRC.vocab), hidden_size=30, recurrent_unit="GRU", n_layers=1, max_length=20)
    dec = modelsNEW.TridentDecoder(3, len(TAR.vocab), hidden_size=30, max_depth=5)
    #dec = modelsNEWBob.DecoderRNN(hidden_size=100,vocab=TAR_SEQ.vocab, encoder_vocab=SRC_SEQ.vocab, recurrent_unit="GRU", num_layers=1, max_length=30, attention_type='additive', dropout=0.3)
    s2s = seq2seq.Seq2Seq(encoder, dec, ["source"], ["middle1"], decoder_train_field_names=["middle1", "source_tree"])

    store = setup_store(parse_arguments())

    training.train(s2s, train_iter, store, args, ignore_index=TAR.vocab.stoi['<pad>'])

    # IF YOU HAVE PANDAS 1.0 YOU MUST MAKE THIS CHANGE MANUALLY
    # https://github.com/MadryLab/cox/pull/3/files


    
    # NOT NECESSARY SEE LINES 127-134
	# # Instantiate Models
	# max_len = max(len(SRC.vocab), len(TRG.vocab))
	# if args.encoder == "Tree":
	# 	encoder = TreeEncoderRNN(max_len, args.hidden_size)
	# else :
	# 	encoder = EncoderRNN(max_len, args.hidden_size, args.encoder)

	# if args.decoder == "Tree":
	# 	decoder = TreeDecoderRNN(max_len, args.hidden_size)
	# else:
	# 	decoder = DecoderRNN(max_len, args.hidden_size, args.decoder, attn=args.attention, n_layers=1, dropout_p=0.1)

	# training.train_iterator(
	# 	train_iter,
	# 	val_iter,
	# 	encoder,
	# 	decoder,
	# 	args.encoder,
	# 	args.decoder,
	# 	args.attention,
	# 	directory,
	# 	args.task,
	# 	SRC,
	# 	learning_rate = args.learning_rate,
	# 	patience = args.patience,
	# 	print_every = args.print_every
	# )



if __name__ == '__main__':
	main()