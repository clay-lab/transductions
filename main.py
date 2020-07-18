# seq2seq.py
# Main file for seq2seq model. Adapted from Tom McCoy's repository.

import os
import argparse

from torchtext.data import Field, TabularDataset, BucketIterator, RawField

from models import EncoderRNN, DecoderRNN, GRUTridentDecoder#, GRUTridentDecoderAttn
from metrics import SentenceLevelAccuracy, TokenLevelAccuracy, SpecTokenAccuracy, LengthAccuracy, AverageMetric
import training
import RPNTask
import seq2seq

import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import cox.store
from cox.store import Store
from tqdm import tqdm
from tree_loaders import TreeField, TreeSequenceField#, pad_arity_factory
from typing import Dict

import test
from typing import List
import pickle

# these are as good as constants
CKPT_NAME_LATEST = "latest_ckpt.pt"
CKPT_NAME_BEST = "best_ckpt.pt"

def get_iterators(args: Dict, source, target, datafields, loc):
	"""
	Constructs train-val-test splits from the task.{train, val, test} files.

	@param args: dictionary of arguments and their values
	@returns: a tuple of BucketIterators in (train, val, test) form.
	"""

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	task_train = args.task + '.train'
	task_val = args.task + '.val'
	task_test = args.task + '.test'

	trn_data, val_data, test_data = TabularDataset.splits(
		path = loc,
		train = task_train,
		validation = task_val,
		test = task_test,
		format = 'tsv',
		skip_header = True,
		fields = datafields
	)

	train, val, test = BucketIterator.splits(
		(trn_data, val_data, test_data), 
		batch_size = 5, 
		device = device, 
		sort_key = lambda x: len(x.target), 
		sort_within_batch = True, 
		repeat = False
	)

	source.build_vocab(trn_data, val_data, test_data)
	target.build_vocab(trn_data, val_data, test_data)

	return (train, val, test)

def train_model(args: Dict):

	"""
	Experiments have the following directory structure:

	EXPDIR/
		task-a/
		task-b/
		TASK/
			data/
				TASK.train
				TASK.val
				TASK.test
				other-test.test
				another-test.test
			ENC-DEC-ATTN/
				model-1/
				model-2/
				model-3/
					logs/RANDOM-HASH/...
					model.pt
					SRC.vocab
					TRG.vocab
					...

	The EXP-DIR is passed in with the -E/--exp-dir flag on the command line.
	TASK is passed in with the -t/--task flag. ENC, DEC, ATTN are similar.

	In order to run an experiment, the trainer will expect a data/ subdir with
	a full train and val split. Everything else will be generated during 
	training.
	"""

	# Get all paths needed for model training
	base_exp_dir = args.exp_dir
	task = args.task
	model_structure = '{0}-{1}-{2}'.format(args.encoder, args.decoder, args.attention)
	model_count = 1

	if not os.path.isdir(base_exp_dir):
		print('ERROR: The provided experiment directory \'{0}\' is not a valid directory.'.format(base_exp_dir))
		raise SystemError

	exp_path = os.path.join(base_exp_dir, task)

	if not os.path.isdir(exp_path):
		print('ERROR: The provided task directory \'{0}\' is not a valid directory.'.format(exp_path))
		raise SystemError

	data_dir = os.path.join(exp_path, 'data')

	if not os.path.isdir(data_dir):
		print('ERROR: The provided data directory \'{0}\' is not a valid directory.'.format(data_dir))
		raise SystemError

	for v in ['.train', '.val']:
		if not os.path.isfile(os.path.join(data_dir, args.task + v)):
			print('ERROR: You must provide a \'{0}\' file for training.'.format(v))
			raise SystemError

	structure_dir = os.path.join(exp_path, model_structure)
	if not os.path.isdir(structure_dir):
		os.mkdir(structure_dir)

	while True:
		model_name = 'model-{0}'.format(model_count)
		if os.path.isdir(os.path.join(structure_dir, model_name)):
			model_count += 1
		else:
			model_dir = os.path.join(structure_dir, model_name)
			log_dir = os.path.join(model_dir, 'logs')
			os.mkdir(model_dir)
			os.mkdir(log_dir)
			break

	store, logging_meters = setup_store(args, log_dir)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	arity=6

	if args.source_format == 'trees' and args.target_format == 'trees':
		SRC_TREE = TreeField(collapse_unary=True)
		SRC = TreeSequenceField(SRC_TREE)
		TRANS = RawField()
		TRG_TREE = TreeField(tree_transformation_fun=pad_arity_factory(arity),collapse_unary=True)
		TRG = TreeSequenceField(TRG_TREE, inner_order="pre", inner_symbol="NULL", is_target=True)
		datafields = [(("source_tree", "source"), (SRC_TREE, SRC)), 
			("annotation", TRANS), 
			(("target_tree", "target"), (TRG_TREE, TRG))]
	elif args.source_format == 'trees': # (trees, sequences)
		SRC_TREE = TreeField(collapse_unary=True)
		SRC = TreeSequenceField(SRC_TREE)
		TRANS = RawField()
		TRG = Field(lower=True, eos_token="<eos>") # Target vocab
		datafields = [(("source_tree", "source"), (SRC_TREE, SRC)), 
			("annotation", TRANS), ("target", TRG)]
	elif args.target_format == 'trees': # (sequences, trees)
		SRC = Field(lower=True, eos_token="<eos>") # Source vocab
		TRANS = RawField()
		TRG_TREE = TreeField(tree_transformation_fun=pad_arity_factory(arity), collapse_unary=True)
		TRG = TreeSequenceField(TRG_TREE, inner_order="pre", inner_symbol="NULL", is_target=True)
		datafields = [("source", SRC), ("annotation", TRANS), 
			(("target_tree", "target"), (TRG_TREE, TRG))]
	else: # (sequences, sequences)
		SRC = Field(lower=True, eos_token="<eos>") # Source vocab
		TRG = Field(lower=True, eos_token="<eos>") # Target vocab
		TRANS = SRC if args.vocab == "SRC" else TRG
		datafields = [("source", SRC), ("annotation", TRANS), ("target", TRG)]

	# Get iterators
	train_iter, val_iter, test_iter = get_iterators(args, SRC, TRG, datafields, data_dir)

	# Pickle vocabularies. This must happen after iterators are created.
	pickle.dump(SRC, open(os.path.join(model_dir, 'SRC.vocab'), 'wb') )
	pickle.dump(TRG, open(os.path.join(model_dir, 'TRG.vocab'), 'wb') )
	if args.target_format == 'trees':
		pickle.dump(TRG_TREE, open(os.path.join(model_dir, 'TRG_TREE.vocab'), 'wb') )
	if args.source_format == 'trees':
		pickle.dump(SRC_TREE, open(os.path.join(model_dir, 'SRC_TREE.vocab'), 'wb') )

	encoder = EncoderRNN(hidden_size=args.hidden_size, vocab = SRC.vocab, recurrent_unit=args.encoder, num_layers=args.layers, dropout=args.dropout)
	tree_decoder_names = ['Tree']
	if args.decoder not in tree_decoder_names:
		dec = DecoderRNN(hidden_size=args.hidden_size, vocab=TRG.vocab, encoder_vocab=SRC.vocab, recurrent_unit=args.decoder, num_layers=args.layers, max_length=args.max_length, attention_type=args.attention, dropout=args.dropout)
		model = seq2seq.Seq2Seq(encoder, dec, ["source"], ["middle0", "annotation", "middle1", "source"], decoder_train_field_names=["middle0", "annotation", "middle1", "source", "target"])
	else:
		#dec = TridentDecoder(arity=3, vocab_size=len(TRG.vocab), hidden_size=args.hidden_size, max_depth=5)
		dec = GRUTridentDecoderAttn(arity=arity, vocab=TRG.vocab, hidden_size=args.hidden_size, max_depth=5, all_annotations=["sem"], encoder_vocab=SRC.vocab, attention_type="additive")
		model = seq2seq.Seq2Seq(encoder, dec, ["source"], ["middle0", "annotation", "middle1", "source"], decoder_train_field_names=["middle0", "annotation", "middle1", "source", "target_tree"])
	
	model.to(device)

	training.train(model, train_iter, val_iter, logging_meters, store, args,
		save_dir = model_dir, ignore_index=TRG.vocab.stoi['<pad>'])

def test_model(args: Dict):

	base_exp_dir = os.path.join(args.exp_dir, args.task)
	structure_name = args.structure
	model_name = args.model
	data_dir = os.path.join(base_exp_dir, 'data')

	model_dir = os.path.join(base_exp_dir, structure_name, model_name)

	logging_dir = os.path.join(model_dir, 'logs')
	if 'training' in os.listdir(logging_dir):
		metadata = Store(logging_dir, 'training')['metadata'].df
	else:
		metadata = os.listdir(logging_dir)[0]

	# Pull out relevant parameters
	hidden_size = int(metadata['hidden_size'][0])
	layers = int(metadata['layers'][0])
	max_length = int(metadata['max_length'][0])
	dropout = float(metadata['dropout'][0])
	encoder = str(metadata['encoder'][0])
	decoder = str(metadata['decoder'][0])
	attention = str(metadata['attention'][0])
	if '==' in attention:
		attention = None
	vocab = str(metadata['vocab'][0])
	trainedtask = str(metadata['task'][0])
	input_format = str(metadata['input_format'][0])

	# Device specification
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	if input_format == 'trees':		
		SRC = pickle.load(open(os.path.join(model_dir, 'SRC.vocab'), 'rb'))
		TRG = pickle.load(open(os.path.join(model_dir, 'TRG.vocab'), 'rb'))
		TRG.inner_order = None # don't print the annotations about tree structure (the "None"s) when converting tree to sequence
		SRC_TREE = pickle.load(open(os.path.join(model_dir, 'SRC_TREE.vocab'), 'rb'))
		TRG_TREE = pickle.load(open(os.path.join(model_dir, 'TRG_TREE.vocab'), 'rb'))
		TRANS = RawField()
		datafields = [(("source_tree", "source"), (SRC_TREE, SRC)), 
			("annotation", TRANS), 
			(("target_tree", "target"), (TRG_TREE, TRG))]
	else:
		# Create datasets and vocabulary
		SRC = pickle.load(open(os.path.join(model_dir, 'SRC.vocab'), 'rb'))
		TRG = pickle.load(open(os.path.join(model_dir, 'TRG.vocab'), 'rb'))
		TRANS = SRC if vocab == "SRC" else TRG
		datafields = [("source", SRC), ("annotation", TRANS), ("target", TRG)]
	
	enc = EncoderRNN(hidden_size=hidden_size, vocab = SRC.vocab, 
		recurrent_unit=encoder, num_layers=layers, dropout=dropout)

	tree_decoder_names = ['Tree']
	if decoder not in tree_decoder_names:
		dec = DecoderRNN(hidden_size=hidden_size, vocab=TRG.vocab, 
			encoder_vocab=SRC.vocab, recurrent_unit=decoder, num_layers=layers, 
			max_length=max_length, attention_type=attention, dropout=dropout)

		model = seq2seq.Seq2Seq(enc, dec, ["source"], ["middle0", "annotation", 
			"middle1", "source"], decoder_train_field_names=["middle0", "annotation", 
			"middle1", "source", "target"])
	else:
		# TODO: replace max depth with max length -- to find depth just take log_3 max length
		dec = GRUTridentDecoder(arity=7, vocab=TRG.vocab, hidden_size=hidden_size, all_annotations=["POLISH", "RPN"], max_depth=5)
		model = seq2seq.Seq2Seq(enc, dec, ["source"], ["middle0", "annotation"], decoder_train_field_names=["middle0", "annotation", "target_tree"])

	model.to(device)
	
	print(attention)

	model_path = os.path.join(model_dir, 'checkpoint.pt')
	model.load_state_dict(torch.load(model_path))
	model.eval()

	if args.files is None:
		
		test.repl(model, args = args, datafields = datafields)
	
	else:

		iterators = {}

		for f in args.files:
			f_data = TabularDataset(os.path.join(data_dir, f + '.test'), 
				format = 'tsv', skip_header = True, fields = datafields)
			iterator = BucketIterator(f_data, batch_size = 5, 
				device = device, sort_key = lambda x: len(x.target), 
				sort_within_batch = True, repeat = False)
			iterators[f] = iterator


		test.test(model, args = args, data = iterators)

def show_model_log(args: Dict):

	base_exp_dir = os.path.join(args.exp_dir, args.task)
	structure_name = args.structure
	model_name = args.model
	data_dir = os.path.join(base_exp_dir, 'data')
	model_dir = os.path.join(base_exp_dir, structure_name, model_name)
	logging_dir = os.path.join(model_dir, 'logs')
	# exp_dir = os.path.join(logging_dir, args.log)

	store = Store(logging_dir, args.log)
	metadata = store['metadata'].df
	logs = store['logs'].df

	print(metadata)
	print(logs)

def setup_store(args: Dict, logging_dir: str, logname = 'training'):

	LOGS_TABLE = "logs"
	META_TABLE = "metadata"

	store = cox.store.Store(logging_dir, logname)

	if args.expname is None:
		# store metadata
		args_dict = args.__dict__
		meta_schema = cox.store.schema_from_dict(args_dict)
		store.add_table(META_TABLE, meta_schema)
		store[META_TABLE].append_row(args_dict)

		logging_meters = dict()
		if args.sentacc: logging_meters['sentence-level-accuracy'] = SentenceLevelAccuracy()
		if args.tokenacc: logging_meters['token-level-accuracy'] = TokenLevelAccuracy()
		if args.tokens is not None:
			for token in args.tokens.split('-'):
				logging_meters['{0}-accuracy'.format(token)] = SpecTokenAccuracy(token)
		logging_meters['loss'] = AverageMetric()

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
	"""
	Creates argument parsers for the main entry point.

	@return: A Namespace object containing the arguments passed to the 
		script, plus any default values for arguments which weren't set.
	"""

	parser = argparse.ArgumentParser()
	subparser = parser.add_subparsers()

	trn = subparser.add_parser('train')
	tst = subparser.add_parser('test')
	log = subparser.add_parser('log')

	trn.set_defaults(func = train_model)
	tst.set_defaults(func = test_model)
	log.set_defaults(func = show_model_log)

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
	trn.add_argument('-E', '--exp-dir',
		help='experiment directory', type=str, required=True)
	trn.add_argument('-S', '--source-format',
		type = str, choices = ['sequences', 'trees'], default = 'sequences',
		help = 'format of source data')
	trn.add_argument('-T', '--target-format',
		type = str, choices = ['sequences', 'trees'], default = 'sequences',
		help = 'format of target data')
	trn.add_argument('-exp', '--expname', help = 'experiment name', 
		type = str, default = None)

	tst.add_argument('-t', '--task', 
		help = 'name of task model was trained on',
		type = str, default = None)
	tst.add_argument('-m', '--model', help = 'name of model to test', type = str,
		required = True)
	tst.add_argument('-sa', '--sentacc', help = 'sentence accuracy', type = bool,
		default = True)
	tst.add_argument('-ta', '--tokenacc', help = 'token accuracy', type = bool, 
		default = True)
	tst.add_argument('-la', '--lenacc', help = 'length accuracy', type = bool, 
		default = True)
	tst.add_argument('-to', '--tokens', 
		help = 'list of tokens for logit-level-accuracy', type = str, 
		default = None)
	tst.add_argument('-E', '--exp-dir',
		help = 'experiment directory', type=str, required=True)
	tst.add_argument('-S', '--structure',
		help = 'structure of model', type=str, required=True)
	tst.add_argument('-f', '--files',
		help = 'testing files', type=str, nargs = '+')
	tst.add_argument('-exp', '--expname', help = 'experiment name', 
		type = str, default = None)

	log.add_argument('-m', '--model', help = 'name of model to show logs of', 
		type = str, required = True)
	log.add_argument('-t', '--task', help = 'task name', 
		type = str, default = None)
	log.add_argument('-E', '--exp-dir',
		help = 'experiment directory', type=str, required=True)
	log.add_argument('-S', '--structure',
		help = 'structure of model', type=str, required=True)
	log.add_argument('-l', '--log',
		help = 'name of log', type=str, default='training')


	return parser.parse_args()

if __name__ == '__main__':
	"""
	Parses the arguments passed on the command line. If `train`, we call
	the train_model() method; if `test`, we call the test_model() method.
	"""

	args = parse_arguments()
	args.func(args)