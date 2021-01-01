from abc import abstractmethod
from torch import Tensor

class BaseMetric:
  """
  An abstract metric, to be given a specific implementation.
  """

  @property
  def accuracy(self):
    if self.total > 0:
      return self.correct / float(self.total)
    else:
      return 0.0
  
  @property
  def name(self):
    return type(self).__name__

  def __init__(self):
    self.correct = 0
    self.total = 0
  
  def reset(self):
    self.correct = 0
    self.total = 0
  
  def _update(self, correct, total):
    self.correct += correct
    self.total += total
  
  @abstractmethod
  def compute(self, prediction: Tensor, target: Tensor):
    """
    Should return a tuple of (delta_c, delta_t) representing the
    number of correct predictions and the number of total predictions in
    a given batch.
    """
    raise NotImplementedError

  def __call__(self, prediction: Tensor, target: Tensor):
    delta_c, delta_t = self.compute(prediction, target)
    self._update(delta_c, delta_t)

class SequenceAccuracy(BaseMetric):
  """
  Computes full-sequence accuracy by tokens. If all tokens in a sequence are
  correct, the sequence scores 1.0; otherwise, it scores 0.0.
  """

  def compute(self, prediction: Tensor, target: Tensor):
    prediction = prediction.argmax(1)
    correct = (prediction == target).prod(axis=0)
    total = correct.size[0]
    return correct, total

class TokenAccuracy(BaseMetric):
  """
  Computes the token-level accuracy; every correct token is +1.0.
  """

  def __init__(self, pad_token_id = None):
    super().__init__()
    self._pad = pad_token_id
  
  def compute(self, prediction: Tensor, target: Tensor):
    prediction = prediction.argmax(1)
    correct = (prediction == target).sum()
    total = target.numel()

    if self._pad:
      correct -= ((prediction == target) & (target == self._pad)).sum()
      total -= (target == self._pad).sum()

    return correct, total

class LossMetric(BaseMetric):
  """
  Computes the average loss over the epoch.
  """

  def __init__(self, loss_fn):
    super().__init__()
    self._loss_fn = loss_fn
  
  def compute(self, prediction: Tensor, target: Tensor):
    loss = self._loss_fn(prediction, target)
    return loss, 1
  
  @property
  def name(self):
    return "Average Loss"