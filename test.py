# test.py
# 
# Code for testing a trained model on provided datasets.

from cmd import Cmd

from typing import List, Dict
from tqdm import tqdm
import torch
from torchtext.data import Field, TabularDataset, BucketIterator
from torch.nn.utils.rnn import pad_sequence
import os

from seq2seq import Seq2Seq
from metrics import SentenceLevelAccuracy

from main import setup_store
import numpy as np

class Model():

	def __init__(self, name: str):
		self.name = name

class ModelREPL(Cmd):
	"""
	REPL for a Seq2Seq model to test its predictions on arbitrary input.
	"""

	prompt = '> '

	def __init__(self, model: Seq2Seq, name: str, datafields):
		super(ModelREPL, self).__init__()
		self.model = model
		self.intro = 'Enter sequences into the {0} model for evaluation.'.format(name)
		self.datafields = datafields

	def default(self, args):
		"""
		Default command executed, if no command is specified. Since we have no
		'commands' per-se, this will always be called.
		"""
		tempfile = 'tmp'
		with open(tempfile, 'w') as f:
			transformation, source = args.split(' ', 1)
			f.write('{0}\t{1}\t{0}\n'.format(source, transformation))
			f.write('{0}\t{1}\t{0}\n'.format(source, transformation))

		# SRC = Field(lower=True, eos_token="<eos>")
		# TRG = Field(lower=True, eos_token="<eos>")
		# datafields = [("source", SRC), ("annotation", TRG), ("target", TRG)]
		dataset = TabularDataset(tempfile, format = 'tsv', fields = self.datafields, skip_header = False)
		# SRC.build_vocab(dataset)
		# TRG.build_vocab(dataset)
		iterator = BucketIterator(dataset, batch_size = 2,
				sort_key = lambda x: len(x.target), sort_within_batch = True, 
				repeat = False)

		# self.model.encoder.vocab = SRC.vocab
		# self.model.decoder.vocab = TRG.vocab

		for batch in iterator:
			logits = self.model(batch)
			prediction = logits.argmax(2)
			sentence = self.model.scores2sentence(prediction, self.model.decoder.vocab)
			print(sentence[0])

		os.remove(tempfile)

	def do_quit(self, args):
		"""
		Exits the REPL.
		"""
		raise SystemExit

def repl(model: Seq2Seq, args, datafields):
	"""
	Enters an interactive read-evaluate-print loop (REPL) with the provided 
	model, where you enter a sequence into the prompt and the model evaluates
	the sequences and prints is prediction to standard output.

	@param model: The provided Seq2Seq model.
	@param name: The name of the model.


	"""

	prompt = ModelREPL(model, name = args.model, datafields = datafields)
	prompt.cmdloop()

def test(model: Seq2Seq, args, data: Dict):
	"""
	Runs model.test() on the provided list of tasks. It is presumed that each
	task corresponds to a test split of data named `task.test` in the data/
	directory.

	@param model: The provided Seq2Seq model.
	@param args: The namespace of arguments passed on the command line. 
	@param data: A list of TabularDatasets.
	"""

	# name = args.model

	base_exp_dir = os.path.join(args.exp_dir, args.task)
	structure_name = args.structure
	model_name = args.model

	model_dir = os.path.join(base_exp_dir, structure_name, model_name)
	results_dir = os.path.join(model_dir, 'results')
	logging_dir = os.path.join(model_dir, 'logs')

	if not os.path.isdir(results_dir):
		os.mkdir(results_dir)

	available_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	model.eval()

	for key, iterator in data.items():

		store, meters = setup_store(args, logging_dir, logname = key)
		stats_dict = {}

		# Prepare output files
		outfile = os.path.join(results_dir, key + '.tsv')
		print('Testing model on {}'.format(key))
		print('Writing results to {}'.format(outfile))
		with open(outfile, 'w') as f:
			f.write('{0}\t{1}\t{2}\n'.format('source', 'target', 'prediction'))

		with torch.no_grad():
			with tqdm(iterator) as t:
				for batch in t:

					logits = model(batch)
					logits_max = logits.argmax(2)

					# pad sequence combines them into a single tensor and pads whichever is shorter
					# then we split along that new dimension to recover separate prediction and target tensors
					pad_token = model.decoder.vocab['<pad>']
					padded_combined = pad_sequence([logits_max, batch.target], padding_value=pad_token)
					prediction_padded, target_padded = padded_combined[:, 0, :], padded_combined[:, 1, :]

					for mkey, meter in meters.items():
						meter.process_batch(prediction_padded, target_padded, model)
					
					source = model.scores2sentence(batch.source, model.encoder.vocab)
					prediction = model.scores2sentence(logits_max, model.decoder.vocab)
					target = model.scores2sentence(batch.target, model.decoder.vocab)

					with open(outfile, 'a') as f:
						for i, s in enumerate(source):
							f.write('{0}\t{1}\t{2}\n'.format(s, target[i], prediction[i]))

				for mkey, meter in meters.items():
					print('{0}:\t{1}'.format(mkey, meter.result()))
					stats_dict[mkey] = meter.result()
					meter.reset()
			if store is not None:
				store["logs"].append_row(stats_dict)