# seq2seq.py
# 
# Main file for seq2seq model. Adapted from Tom McCoy's repository.

import os
import argparse

import torch
from torchtext.data import Field, TabularDataset, BucketIterator

from models import EncoderRNN, TreeEncoderRNN, DecoderRNN, TreeDecoderRNN
import training

def parse_arguments():

	parser = argparse.ArgumentParser()

	parser.add_argument(
		'-e', '--encoder',
		help = 'type of encoder used',
		choices = ['GRU', 'Tree'],
		type = str,
		default = 'GRU'
	)
	parser.add_argument(
		'-d', '--decoder',
		help = 'type of decoder used',
		choices = ['GRU', 'Tree'],
		type = str,
		default = 'GRU'
	)
	parser.add_argument(
		'-t', '--task',
		help = 'task model is trained to perform',
		choices = ['negation'],
		type = str,
		default = None,
		required = True
	)
	parser.add_argument(
		'-a', '--attention',
		help = 'type of attention used',
		choices = ['content'],
		type = str,
		default = None,
		required = True
	)
	parser.add_argument(
		'-lr', '--learning-rate',
		help = 'learning rate',
		type = float,
		default = 0.001
	)
	parser.add_argument(
		'-hs', '--hidden-size',
		help = 'hidden size',
		type = int,
		default = 256
	)
	parser.add_argument(
		'-rs', '--random-seed', 
		help='random seed', 
		type=float, 
		default=None
	)
	parser.add_argument(
		'-ps', '--parse-strategy',
		help = 'how to parse (WHAT IS IT PARSING??)',
		type = str,
		choices = ['correct', 'right-branching'],
		default = 'correct'
	)
	parser.add_argument(
		'-p', '--patience',
		help = 'parience',
		type = int,
		default = 3
	)
	parser.add_argument(
		'-v', '--vocab',
		help = 'vocabulary used ?? (THIS SHOULD BE CLARIFIED)',
		type = str,
		choices = ['SRC', 'TRG'],
		default = 'SRC'
	)
	parser.add_argument(
		'-pr', '--print-every',
		help = 'print training data out after N iterations',
		metavar = 'N',
		type = int,
		default = 1000
	)
	return parser.parse_args()

def create_directories(args):
	if args.parse_strategy == "right_branching":
		path = args.task + "_" + args.encoder + "_" + args.decoder  + "_" + "RB" + "_" + args.attention + "_" + str(args.learning_rate) + "_" + str(args.hidden_size)
		return os.path.join("models", path)
	else:
		path = args.task + "_" + args.encoder + "_" + args.decoder  + "_" + args.attention + "_" + str(args.learning_rate) + "_" + str(args.hidden_size)
		return os.path.join("models", path)

def main():
	
	args = parse_arguments()
	directory = create_directories(args)

	# Device specification
	available_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	# Create Datasets
	trainingdata = args.task + '.train'
	validationdata = args.task + '.val'
	testingdata = args.task + '.test'

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

	# Instantiate Models
	max_len = max(len(SRC.vocab), len(TRG.vocab))
	if args.encoder == "Tree":
		encoder = TreeEncoderRNN(max_len, args.hidden_size)
	else :
		encoder = EncoderRNN(max_len, args.hidden_size, args.encoder)

	if args.decoder == "Tree":
		decoder = TreeDecoderRNN(max_len, args.hidden_size)
	else:
		decoder = DecoderRNN(max_len, args.hidden_size, args.decoder, attn=args.attention, n_layers=1, dropout_p=0.1)

	training.train_iterator(
		train_iter,
		val_iter,
		encoder,
		decoder,
		args.encoder,
		args.decoder,
		args.attention,
		directory,
		args.task,
		SRC,
		learning_rate = args.learning_rate,
		patience = args.patience,
		print_every = args.print_every
	)

if __name__ == '__main__':
	main()