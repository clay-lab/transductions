# test.py
# 
# Code for testing a trained model on provided datasets.

from cmd import Cmd
from seq2seq import Seq2Seq
from typing import List, Dict
from tqdm import tqdm
import torch
from torchtext.data import Field, TabularDataset, BucketIterator
import os

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
		with open('tmp', 'w') as f:
			transformation, source = args.split(' ', 1)
			f.write('{0}\t{1}\t{0}\n'.format(source, transformation))
			f.write('{0}\t{1}\t{0}'.format(source, transformation))

		# SRC = Field(lower=True, eos_token="<eos>")
		# TRG = Field(lower=True, eos_token="<eos>")
		# datafields = [("source", SRC), ("annotation", TRG), ("target", TRG)]
		dataset = TabularDataset('tmp', format = 'tsv', fields = self.datafields, 
			skip_header = False)
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

	def do_quit(self, args):
		"""
		Exits the REPL.
		"""
		raise SystemExit

def repl(model: Seq2Seq, name: str, datafields):
	"""
	Enters an interactive read-evaluate-print loop (REPL) with the provided 
	model, where you enter a sequence into the prompt and the model evaluates
	the sequences and prints is prediction to standard output.

	@param model: The provided Seq2Seq model.
	@param name: The name of the model.


	"""

	prompt = ModelREPL(model, name = name, datafields = datafields)
	prompt.cmdloop()

def test(model: Seq2Seq, name: str, data: Dict):
	"""
	Runs model.test() on the provided list of tasks. It is presumed that each
	task corresponds to a test split of data named `task.test` in the data/
	directory.

	@param model: The provided Seq2Seq model.
	@param name: The name of the model.
	@param data: A list of TabularDatasets.
	"""

	available_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	if not os.path.exists('results'):
		os.makedir('results')

	for key, iterator in data.items():

		keyless = key[:-len('.test')]
		outfile = os.path.join('results', name + '-' + keyless + '.tsv')
		print('Testing model on {}'.format(key))
		print('Writing results to {}'.format(outfile))
		with open(outfile, 'w') as f:
			f.write('{0}\t{1}\t{2}\n'.format('source', 'target', 'prediction'))


		with torch.no_grad():
			with tqdm(iterator) as t:
				for batch in t:

					# raise SystemExit

					logits = model(batch)
					
					source = model.scores2sentence(batch.source, model.encoder.vocab)
					prediction = model.scores2sentence(logits.argmax(2), model.decoder.vocab)
					target = model.scores2sentence(batch.target, model.decoder.vocab)

					with open(outfile, 'a') as f:
						for i, s in enumerate(source):
							f.write('{0}\t{1}\t{2}\n'.format(s, target[i], prediction[i]))