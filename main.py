# seq2seq.py
# Main file for seq2seq model. Adapted from Tom McCoy's repository.

import os
import argparse
import json

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

import test
from typing import List

# these are as good as constants
LOGS_TABLE = "logs"
META_TABLE = "metadata"
CKPT_NAME_LATEST = "latest_ckpt.pt"
CKPT_NAME_BEST = "best_ckpt.pt"

def train_model(args: Dict):

	# Create directories where model and logs will be saved.
	# Directory structure is as follows:
	#
	# models/
	#   task-month-day-year-N/
	#     model.pt
	# 
	# logs/
	#   task-month-day-year-N/
	#     <some random hash>/
	#       COX DATA HERE?
	# 
	# where `N` is incremented automatically to ensure a unique path

	exp_name = args.task
	exp_time = time.strftime('%d-%m-%y', time.gmtime())
	exp_count = 0

	while True:
		exp_path = '{0}-{1}-{2}'.format(exp_name, exp_time, exp_count)
		if os.path.isdir(os.path.join(args.outdir, exp_path)):
			exp_count += 1
		else:
			os.mkdir(os.path.join(args.outdir, exp_path))
			os.mkdir(os.path.join('models', exp_path))
			break

	logging_dir = os.path.join(args.outdir, exp_path)
	model_dir = os.path.join('models', exp_path)

	# We need to save not only the model parameters, but also enough information
	# about the models structure that we can initialize a model compatable with
	# the saved parameters before loading the state dictionary. We'll write out
	# the relevant arguments to the file `model.structure` in `model_dir`.

	with open(os.path.join(model_dir, 'arguments.txt'), 'w') as f:
		for key, value in vars(args).items():
			f.write('{0}: {1}\n'.format(key, value))

	store, logging_meters = setup_store(args, logging_dir)

	# Device specification
	available_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	# Create Datasets
	trainingdata = args.task + '.train'
	validationdata = args.task + '.val'
	testingdata = args.task + '.test'
	
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
		path = 'data',
		train = trainingdata,
		validation = validationdata,
		test = testingdata,
		format = 'tsv',
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

	encoder = EncoderRNN(hidden_size=args.hidden_size, vocab = SRC.vocab, recurrent_unit=args.encoder, num_layers=args.layers, dropout=args.dropout)
	encoder.to(available_device)
	tree_decoder_names = ['Tree']
	if args.decoder not in tree_decoder_names:
		dec = DecoderRNN(hidden_size=args.hidden_size, vocab=TRG.vocab, encoder_vocab=SRC.vocab, recurrent_unit=args.decoder, num_layers=args.layers, max_length=args.max_length, attention_type=args.attention, dropout=args.dropout)
		dec.to(available_device)
		model = seq2seq.Seq2Seq(encoder, dec, ["source"], ["middle0", "annotation", "middle1", "source"], decoder_train_field_names=["middle0", "annotation", "middle1", "source", "target"])
		model.to(available_device)
	else:
		assert False

	# if CKPT_NAME_LATEST in os.listdir(store.path):
	# 	ckpt_path = os.path.join(store.path, CKPT_NAME_LATEST)
	# 	model.load_state_dict(torch.load(ckpt_path))

	training.train(model, train_iter, val_iter, logging_meters, store, args,
		save_dir = model_dir, ignore_index=TRG.vocab.stoi['<pad>'])

def test_model(args: Dict):
	
	# Load the saved model structure
	structure_path = os.path.join('models', args.model, 'arguments.txt')
	structure = {}
	with open(structure_path, 'r') as f:
		for line in f:
			(key, value) = line.split(': ')
			structure[key] = value.strip()

	# Pull out relevant parameters
	hidden_size = int(structure['hidden_size'])
	layers = int(structure['layers'])
	max_length = int(structure['max_length'])
	dropout = float(structure['dropout'])
	encoder = str(structure['encoder'])
	decoder = str(structure['decoder'])
	attention = str(structure['attention'])
	if attention == 'None':
		attention = None
	vocab = str(structure['vocab'])
	trainedtask = str(structure['task'])

	# Device specification
	available_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	# Create datasets and vocabulary
	SRC = Field(lower=True, eos_token="<eos>") # Source vocab
	TRG = Field(lower=True, eos_token="<eos>") # Target vocab
	TRANS = SRC if vocab == "SRC" else TRG
	datafields = [("source", SRC), ("annotation", TRANS), ("target", TRG)]

	vocabsources = set()
	vocabsources.add(trainedtask + '.train')
	vocabsources.add(trainedtask + '.val')
	vocabsources.add(trainedtask + '.test')
	if args.task is not None:
		for task in args.task:
			vocabsources.add(task + '.test')

	datasets = []
	for v in vocabsources:
		d = TabularDataset(os.path.join('data', v), format = 'tsv', 
			skip_header = True, fields = datafields)
		datasets.append(datasets)

	SRC.build_vocab(*[TabularDataset(os.path.join('data', v), format = 'tsv', 
			skip_header = True, fields = datafields) for v in vocabsources])
	TRG.build_vocab(*[TabularDataset(os.path.join('data', v), format = 'tsv', 
			skip_header = True, fields = datafields) for v in vocabsources])

	enc = EncoderRNN(hidden_size=hidden_size, vocab = SRC.vocab, 
		recurrent_unit=encoder, num_layers=layers, dropout=dropout)
	dec = DecoderRNN(hidden_size=hidden_size, vocab=TRG.vocab, 
		encoder_vocab=SRC.vocab, recurrent_unit=decoder, num_layers=layers, 
		max_length=max_length, attention_type=attention, dropout=dropout)

	enc.to(available_device)
	dec.to(available_device)

	model = seq2seq.Seq2Seq(enc, dec, ["source"], ["middle0", "annotation", "middle1", "source"], decoder_train_field_names=["middle0", "annotation", "middle1", "source", "target"])

	# Does this need to be re-done once the model is loaded?
	model.to(available_device)

	model_path = os.path.join('models', args.model, 'checkpoint.pt')
	model.load_state_dict(torch.load(model_path))
	model.eval()

	iterators = []
	for d in [TabularDataset(os.path.join('data', v), format = 'tsv', skip_header = True, fields = datafields) for v in vocabsources]:
		i = BucketIterator(d, batch_size = 5, device = available_device, sort_key = lambda x: len(x.target), sort_within_batch = True, repeat = False)
		iterators.append(i)

	if args.task is not None:
		test.test(model, name = args.model, data = iterators)
	else:
		test.repl(model, name = args.model)


def setup_store(args: Dict, logging_dir: str):

	store = cox.store.Store(logging_dir, args.expname)

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

	return store, logging_meters

def parse_arguments():

	parser = argparse.ArgumentParser()
	subparser = parser.add_subparsers()

	# Create the training subparser and add its arguments
	trn = subparser.add_parser('train')
	trn.set_defaults(func = train_model)

	trn.add_argument('-m', '--model', 
		help = 'type of model (encoder and decoder) used', type = str, 
		choices = ['GRU', 'LSTM', 'SRN', 'Tree'], default = 'GRU')
	trn.add_argument('-e', '--encoder', 
		help = 'type of encoder used', type = str, 
		choices = ['GRU', 'LSTM', 'SRN', 'Tree'], default = 'GRU')
	trn.add_argument('-d', '--decoder', 
		help = 'type of decoder used', type = str, 
		choices = ['GRU', 'LSTM', 'SRN', 'Tree'], default = 'GRU')
	trn.add_argument('-t', '--task', help = 'task model is trained to perform', 
		type = str, required = True)
	trn.add_argument('-a', '--attention', help = 'type of attention used', 
		choices = ['location', 'additive', 'multiplicative', 'dotproduct'], 
		type = str, default = None)
	trn.add_argument('-lr', '--learning-rate', help = 'learning rate', 
		type = float, default = 0.01)
	trn.add_argument('-hs', '--hidden-size', help = 'hidden size', type = int, 
		default = 256)
	trn.add_argument('-l', '--layers', 
		help='number of layers for encoder and decoder', type = int, default=1)
	trn.add_argument('--max-length', help='maximum length of decoded sequecnes', 
		type = int, default=30)
	trn.add_argument('-rs', '--random-seed', help='random seed', type = float, 
		default=None)
	trn.add_argument('-p', '--patience', 
		help = 'number of changes model has to improve loss by DELTA to avoid early stopping',
		type = int, default = 3)
	trn.add_argument('-dt', '--delta', 
		help = 'amount model needs to improve by to avoid early stopping', 
		type = float, default = 0.005)
	trn.add_argument('-v', '--vocab', 
		help = 'which vocabulary contains the transformation annotation', 
		type = str, choices = ['SRC', 'TRG'], default = 'TRG')
	trn.add_argument('-do', '--dropout', help = 'how much dropout to use', 
		type = float, default=0.0)
	trn.add_argument('-i', '--input-format', 
		help = 'input files could contain string representations of trees or just plain sequences', 
		type = str, choices = ['sequences', 'trees'], default = 'sequences')
	trn.add_argument('-ep', '--epochs', help = 'number of epochs to train for', 
		type = int, default = 40)
	trn.add_argument('-b', '--batch-size', help = 'batch size', type = int, 
		default = 5)
	trn.add_argument('-sa', '--sentacc', help = 'sentence accuracy', type = bool,
		default = True)
	trn.add_argument('-ta', '--tokenacc', help = 'token accuracy', type = bool, 
		default = True)
	trn.add_argument('-la', '--lenacc', help = 'length accuracy', type = bool, 
		default = True)
	trn.add_argument('-to', '--tokens', 
		help = 'list of tokens for logit-level-accuracy', type = str, 
		default = None)
	trn.add_argument('-exp', '--expname', help = 'experiment name', 
		type = str, default = None)
	trn.add_argument('-o', '--outdir', 
		help = 'directory in which to place cox store', type = str, 
		default = 'logs')

	# Create the testing subparser and add its arguments
	tst = subparser.add_parser('test')
	tst.set_defaults(func = test_model)

	tst.add_argument('-t', '--task', 
		help = 'tasks to test model on. If no tasks are provided, you will enter a REPL to test the provided model on custom inputs',
		type = str, nargs = '+', default = None)
	tst.add_argument('-m', '--model', help = 'name of model to test', type = str,
		required = True)

	return parser.parse_args()

if __name__ == '__main__':

	args = parse_arguments()
	args.func(args)
