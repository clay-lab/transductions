from abc import abstractmethod
import seq2seq as ss

class AverageMetric:
	"""
	A base class for metrics computed at the end of an epoch. If the metric
	in question is dependent on the evaluation of prediction and target data,
	subclass and implement the process_batch method. If the metric is known
	without this computation, simply instantiate an instance of AverageMetric
	and set the new value with the update method.
	"""

	def __init__(self, tokens=None):
		self.correct = 0
		self.total = 0
		self.tokens = tokens

	def reset(self):
		self.correct = 0
		self.total = 0

	@abstractmethod
	def process_batch(self, prediction, target, model):
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

	def process_batch(self, prediction, target, model:ss.Seq2Seq):
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

	def process_batch(self, prediction, target, model): 
		# TODO: Does this still work if pred and target are different sizes?		
		correct = (prediction == target).sum()
		total = target.size()[0] * target.size()[1]
		self.update(correct, total)

class SpecTokenAccuracy(AverageMetric):
	"""
	Computes the accuracy of a model at the specified token level. For each index in a
	predicted sequence the number of correct predictions is incremented if the
	tokens at that index in the predicted sequence and the target sequence are
	the same for a given token.
	"""
	def process_batch(self, prediction, target, model):

		logits = [model.decoder.vocab[token] for token in self.tokens.split("-")]
		correct = 0
		total = 0
		for logit in logits:
			if logit in target:
				for row in range(prediction.size()[0]):
					for i in range(prediction.size()[1]):
						correct += ((logit in prediction[row][i]) and (logit in target[row][i]))
				total += (target == logit).sum().item()
		self.update(correct, total)