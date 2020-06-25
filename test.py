# test.py
# 
# Code for testing a trained model on provided datasets.

from cmd import Cmd
from seq2seq import Seq2Seq

class Model():

	def __init__(self, name: str):
		self.name = name

class ModelREPL(Cmd):
	"""
	REPL for a Seq2Seq model to test its predictions on arbitrary input.
	"""

	prompt = '> '

	def __init__(self, model: Seq2Seq):
		super(ModelREPL, self).__init__()
		self.model = model
		self.intro = 'Enter sequences into the {0} model for evaluation.'.format(model.name)

	def default(self, args):
		"""
		Default command executed, if no command is specified. Since we have no
		'commands' per-se, this will always be called.
		"""
		print('Hello, {0}'.format(args))

	def do_quit(self, args):
		"""
		Exits the REPL.
		"""
		raise SystemExit

def repl(model: Seq2Seq):

	prompt = ModelREPL(model)
	prompt.cmdloop()


if __name__ == '__main__':
	testmodel = Model('neg-6-24-GRU')
	repl(testmodel)