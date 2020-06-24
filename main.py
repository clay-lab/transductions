# seq2seq.py
# Main file for seq2seq model. Adapted from Tom McCoy's repository.

import os
import argparse

from torchtext.data import Field, TabularDataset, BucketIterator, RawField

from models import EncoderRNN, DecoderRNN #, TreeDecoderRNN, TreeEncoderRNN
import training
import RPNTask
import seq2seq
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import cox.store
from tqdm import tqdm
from tree_loaders import TreeField, TreeSequenceField
from typing import Dict

# these are as good as constants
LOGS_TABLE = "logs"
META_TABLE = "metadata"
CKPT_NAME_LATEST = "latest_ckpt.pt"
CKPT_NAME_BEST = "best_ckpt.pt"

def setup_store(args: Dict):
	taskname = args.outdir + "/{0}-results-".format(args.task)
	lt = (time.localtime(time.time()))
	date = "{0}-{1}-{2}".format(str(lt[1]), str(lt[2]), str(lt[0])[2:])
	counter = 0 
	directory = taskname + date + "_{0}".format(str(counter))   
	while os.path.isdir(directory):
		directory = taskname + date + "_{0}".format(str(counter))
		counter += 1
	# print(directory[13:])
	# exit()
	store = cox.store.Store(directory, args.expname)

	if args.expname is None:
		# store metadata
		args_dict = args.__dict__
		meta_schema = cox.store.schema_from_dict(args_dict)
		store.add_table(META_TABLE, meta_schema)
		store[META_TABLE].append_row(args_dict)

		logging_meters = dict()
		if args.sentacc: logging_meters['sentence-level-accuracy'] = training.SentenceLevelAccuracy()
		if args.tokenacc: logging_meters['token-level-accuracy'] = training.TokenLevelAccuracy()
		if args.tokens is not None:
			for token in args.tokens.split('-'):
				logging_meters['{0}-accuracy'.format(token)] = training.SpecTokenAccuracy(token)
		logging_meters['loss'] = training.AverageMetric()

		# TODO: support columns that aren't float
		logs_schema = {name: float for name, meter in logging_meters.items()}
		store.add_table(LOGS_TABLE, logs_schema)
	else:
		# TODO: check that the parameters match
		# TODO: load logging meters
		old_meta_schema = store[META_TABLE].schema
		old_args = dict(store[META_TABLE].df.loc[0, :])

	return store, logging_meters, directory[13:]

def parse_arguments():

	parser = argparse.ArgumentParser()

	parser.add_argument('-e', '--encoder', help = 'type of encoder used', choices = ['GRU', 'LSTM', 'SRN', 'Tree'], type = str, default = 'GRU')
	parser.add_argument('-d', '--decoder', help = 'type of decoder used', choices = ['GRU', 'LSTM', 'SRN', 'Tree'], type = str, default = 'GRU')
	parser.add_argument('-t', '--task', help = 'task model is trained to perform', type = str, required = True)
	parser.add_argument('-a', '--attention', help = 'type of attention used', choices = ['location', 'additive', 'multiplicative', 'dotproduct'], type = str, default = None, required = False)
	parser.add_argument('-lr', '--learning-rate', help = 'learning rate', type = float, default = 0.01)
	parser.add_argument('-hs', '--hidden-size', help = 'hidden size', type = int, default = 256)
	parser.add_argument('-l', '--layers', help='number of layers for encoder and decoder', type = int, default=1)
	parser.add_argument('--max-length', help='limits the length of decoded sequecnes', type = int, default=30)
	parser.add_argument('-rs', '--random-seed', help='random seed', type=float, default=None)
	parser.add_argument('-p', '--patience', help = 'parience', type = int, default = 3)
	parser.add_argument('-v', '--vocab', help = 'which vocabulary contains the transformation annotation', type = str, choices = ['SRC', 'TRG'], default = 'TRG')
	parser.add_argument('-do', '--dropout', help = 'how much dropout to use', type = float, default=0.0)
	parser.add_argument('--input-format', help='input files could contain string representations of trees or just plain sequences', type=str, choices = ['sequences', 'trees'], default='sequences')
	parser.add_argument('-ep', '--epochs', help="number of epochs", type=int, default=40)
	parser.add_argument('-b', '--batch-size', help='batch size', type=int, default=5)
	parser.add_argument('-sa', '--sentacc', help='sentence accuracy', type=bool, default=True)
	parser.add_argument('-ta', '--tokenacc', help='token accuracy', type=bool, default=True)
	parser.add_argument('-la', '--lenacc', help='length accuracy', type=bool, default=True)
	parser.add_argument('-exp', '--expname', help='exp_name', type=None, default=None)
	parser.add_argument('-out', '--outdir', help='directory in which to place cox store', type=None, default='logs/default')
	parser.add_argument('-to', '--tokens', help='list of tokens for logit-level-accuracy', type=str, default=None)

	return parser.parse_args()

def main():
	args = parse_arguments()
 
	store, logging_meters, directory = setup_store(args)

	# Device specification
	available_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	# Create Datasets
	trainingdata = args.task + '.train'
	validationdata = args.task + '.val'
	testingdata = args.task + '.test'
	# NA 80-93
	if args.input_format == 'trees':
		SRC_TREE = TreeField(collapse_unary=True)
		SRC = TreeSequenceField(SRC_TREE)
		TRANS = RawField()
		TAR_TREE = TreeField(collapse_unary=True)
		TAR = TreeSequenceField(TAR_TREE, inner_order="pre", inner_symbol="NULL", is_target=True)
		datafields = [(("source_tree", "source"), (SRC_TREE, SRC)), 
			("annotation", TRANS), 
			(("target_tree", "target"), (TAR_TREE, TAR))]
	else:
		SRC = Field(lower=True, eos_token="<eos>") # Source vocab
		TRG = Field(lower=True, eos_token="<eos>") # Target vocab
		TRANS = SRC if args.vocab == "SRC" else TRG
		datafields = [("source", SRC), ("annotation", TRANS), ("target", TRG)]

	train, valid, test = TabularDataset.splits(
		path = "data",
		train = trainingdata,
		validation = validationdata,
		test = testingdata,
		format = 'tsv',
		skip_header = True,
		fields = datafields
	)
	# print(train[0].source)
	# exit()

	SRC.build_vocab(train, valid, test)
	TRG.build_vocab(train, valid, test)
	# print(SRC.vocab.stoi)

	train_iter, val_iter, test_iter = BucketIterator.splits(
		(train, valid, test), 
		batch_size = 5, 
		device = available_device, 
		sort_key = lambda x: len(x.target), 
		sort_within_batch = True, 
		repeat = False
	)
	# for batch in train_iter:
	# 	print(batch.source)
	# 	exit()


	encoder = EncoderRNN(hidden_size=args.hidden_size, vocab = SRC.vocab, recurrent_unit=args.encoder, num_layers=args.layers, dropout=args.dropout)
	tree_decoder_names = ['Tree']
	if args.decoder not in tree_decoder_names:
		dec = DecoderRNN(hidden_size=args.hidden_size, vocab=TRG.vocab, encoder_vocab=SRC.vocab, recurrent_unit=args.decoder, num_layers=args.layers, max_length=args.max_length, attention_type=args.attention, dropout=args.dropout)
		s2s = seq2seq.Seq2Seq(encoder, dec, ["source"], ["middle0", "annotation", "middle1", "source"], decoder_train_field_names=["middle0", "annotation", "middle1", "source", "target"])
	else:
		assert False

	if CKPT_NAME_LATEST in os.listdir(store.path):
		ckpt_path = os.path.join(store.path, CKPT_NAME_LATEST)
		s2s.load_state_dict(torch.load(ckpt_path))

	training.train(s2s, train_iter, val_iter, logging_meters, store, args, ignore_index=TRG.vocab.stoi['<pad>'])
	training.test(s2s, test_iter, args.task, directory)


if __name__ == '__main__':
	import warnings
	warnings.warn(
		"If you have Pandas 1.0 you must make the following change manually for cox to work: https://github.com/MadryLab/cox/pull/3/files")

	main()


