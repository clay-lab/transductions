import torch
import torchtext.data as tt
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import numpy as np
import os
from early_stopping import EarlyStopping
from abc import abstractmethod
import seq2seq as ss
from typing import Dict
import cox.store as cx
 
CKPT_NAME_LATEST = "latest_ckpt.pt"

class AverageMetric:
	"""
	A base class for metrics computed at the end of an epoch. If the metric
	in question is dependent on the evaluation of prediction and target data,
	subclass and implement the process_batch method. If the metric is known
	without this computation, simply instantiate an instance of AverageMetric
	and set the new value with the update method.
	"""

	def __init__(self):
		self.correct = 0
		self.total = 0

	def reset(self):
		self.correct = 0
		self.total = 0

	@abstractmethod
	def process_batch(self, prediction, target):
		"""
		Implement this method if you need to update the metric based on 
		calculations of the model's output for a particular set of training
		data. For instance, computing the sentence level accuracy.

		Internally, this method should call self.update(...) to update new
		values for the metric.

		@param prediction: A tensor of predicted output.
		@param target: The reference output.
		"""
		pass

	def update(self, correct: int, total=1):
		"""
		Updates the metric's values for correct and total predictions.

		@param correct: The number of correct predictions.
		@param total: The total number of predictions made. Defaults to 1 in 
			cases where the metric is not weighted by batch size, like loss.
		"""
		self.correct += correct
		self.total += total

	def result(self):
		return 1.0 * self.correct / self.total if self.total > 0 else np.nan

class SentenceLevelAccuracy(AverageMetric):
	"""
	Computes the accuracy of a model at the sentence level. A predicted 
	sequence is correct if it is equal to the target sequence.
	"""

	def process_batch(self, prediction, target):  
		correct = (prediction == target).prod(axis=0)
		total = correct.size()[0]
		self.update(correct.sum(), total)

class TokenLevelAccuracy(AverageMetric):
	"""
	Computes the accuracy of a model at the token level. For each index in a
	predicted sequence the number of correct predictions is incremented if the
	tokens at that index in the predicted sequence and the target sequence are
	the same.
	"""

	def process_batch(self, prediction, target): 
		# TODO: Does this still work if pred and target are different sizes?
		correct = (prediction == target).sum()
		total = target.size()[0] * target.size()[1]
		self.update(correct, total)

class LengthLevelAccuracy(AverageMetric):

	def __init__(self):
		AverageMetric.__init__(self)
		self.total = 1

	def process_batch(self, prediction, target): 
		pass

def predict(model: ss.Seq2Seq, source: torch.Tensor):

	# build batch from tensor
	pass

	model.eval()
	with torch.no_grad():

		logits = model(batch)
		predictions = logits[:source.size()[0], :].argmax(2)
		sentences = model.scores2sentence(predictions, model.decoder.vocab)

		return sentences

def test(model: ss.Seq2Seq, test_iter: tt.Iterator, task: str, filename: str):

	model.eval()

	if not os.path.exists('results'):
		os.makedirs('results')
		
	outfile = os.path.join('results', filename + '.tsv')
	with open(outfile, 'w') as f:
		f.write('{0}\t{1}\t{2}\n'.format('source', 'target', 'prediction'))
	with torch.no_grad():
		print("Testing on test data")
		with tqdm(test_iter) as t:
			for batch in t:

				logits = model(batch)
				target = batch.target 
				predictions = logits[:target.size()[0], :].argmax(2)

				sentences = model.scores2sentence(batch.source, model.encoder.vocab)
				predictions = model.scores2sentence(predictions, model.decoder.vocab)
				target = model.scores2sentence(target, model.decoder.vocab)

				with open(outfile, 'a') as f:
					for i, _ in enumerate(sentences):
						f.write('{0}\t{1}\t{2}\n'.format(
							sentences[i], target[i], predictions[i])
						)

def evaluate(model: ss.Seq2Seq, val_iter: tt.Iterator, epoch: int, 
		         args: Dict, criterion=None, logging_meters=None, store=None):

	model.eval()
	stats_dict = dict()

	with torch.no_grad():
		print("Evaluating epoch {0}/{1} on val data".format(epoch + 1, args.epochs))
		with tqdm(val_iter) as V:
			for batch in V:

				logits = model(batch) # seq length x batch_size x vocab
				target = batch.target # seq length x batch_size
				l = logits[:target.size()[0], :].permute(0, 2, 1)
				predictions = logits[:target.size()[0], :].argmax(2)

				batch_loss = criterion(l, target)

				for name, meter in logging_meters.items():
					if name == 'loss':
						meter.update(batch_loss)
					else:
						meter.process_batch(predictions, target)

			for name, meter in logging_meters.items():
				stats_dict[name] = meter.result()
				meter.reset()

		if store is not None:
			store["logs"].append_row(stats_dict)

	return stats_dict

def train(model: ss.Seq2Seq, train_iterator: tt.Iterator, 
	        validation_iter: tt.Iterator, logging_meters: Dict, 
	        store: cx.Store, args: Dict, ignore_index=None):

	optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
	criterion = nn.CrossEntropyLoss(weight=None, ignore_index=ignore_index)
	early_stopping = EarlyStopping(patience = args.patience, verbose = False,
		filename = os.path.join(store.path, CKPT_NAME_LATEST), delta=0.005)
	
	for epoch in range(args.epochs):

		model.train()
		print("Training epoch {0}/{1} on train data".format(epoch + 1, args.epochs))
		with tqdm(train_iterator) as T:
			for batch in T:
				optimizer.zero_grad()

				decoder_outputs = model(batch)
				pred = decoder_outputs.permute(1, 2, 0)
				target = batch.target.permute(1, 0)
				batch_loss = criterion(pred, target)

				batch_loss.backward()
				optimizer.step()

				logging_meters['loss'].update(batch_loss.item())
				T.set_postfix(loss=logging_meters['loss'].result())

		eval_stats = evaluate(model, validation_iter, epoch, args, criterion,
		                      logging_meters=logging_meters, store=store)

		for name, stat in eval_stats.items():
			if 'accuracy' in name:
				stat = stat * 100
			sign = '%' if 'accuracy' in name else ''
			print('{:<25s} {:.5} {:s}'.format(name, stat, sign))

		early_stopping(eval_stats['loss'], model)
		if early_stopping.early_stop:
			print("Early stopping. Loading model from last saved checkoint.")
			model.load_state_dict(torch.load(os.path.join(store.path, CKPT_NAME_LATEST)))
			break

		torch.save(model.state_dict(), os.path.join(store.path, CKPT_NAME_LATEST))

