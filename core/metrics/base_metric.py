from abc import abstractmethod
from torch import Tensor
import torch
from typing import List

import logging
log = logging.getLogger(__name__)

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

    correct = (prediction == target).prod(axis=1)
    total = correct.shape[0]
    correct = correct.sum()

    return correct, total

class NthTokenAccuracy(BaseMetric):
  """
  Computes the accuracy at the specified token.
  """

  @property
  def name(self):
    if self.n == 1:
      return "1st Token Accuracy"
    elif self.n == 2:
      return "2nd Token Accuracy"
    elif self.n == 3:
      return "3rd Token Accuray"
    else:
      return f"{self.n}th Token Accuracy"

  def __init__(self, n: int = 0):
    super().__init__()
    self.n = n
  
  def compute(self, prediction: Tensor, target: Tensor):
    prediction = prediction.argmax(1)
    correct = (prediction[:,self.n] == target[:,self.n])
    total = correct.shape[0]
    correct = correct.sum()

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

class LengthAccuracy(BaseMetric):
  """
  Are the predicted sequences the same length as the target seqs?
  """

  def __init__(self, pad_token_id = None):
    super().__init__()
    self._pad = pad_token_id

  def compute(self, prediction: Tensor, target: Tensor):
    prediction = prediction.argmax(1)
    pred_seq_len = (prediction != self._pad).sum(axis=1)
    target_seq_len = (target != self._pad).sum(axis=1)
    matches = torch.eq(pred_seq_len, target_seq_len)
    correct = matches.int().sum()
    total = matches.shape[0]

    return correct, total

class LinearAError(BaseMetric):
  """
  Given a target of the form
    Alice near Mary sees herself -> See ( Alice , Alice ) & near ( Alice , Mary )
  Count the number of outputs which mix up the antecedent of 'herself', interpreting
  the VP as 'Mary sees herself', resulting in
    See ( Mary  , Mary  ) & near ( ............. )
  """

  def __init__(self, pad_token_id = None, and_token_id = None):
    super().__init__()
    self._pad = pad_token_id
    self._and = and_token_id

  def compute(self, prediction: Tensor, target: Tensor):

    # total = # target sequences of the form "<sos> verb ( X , X ) & p ( X , Y )"
    # correct = # predictions of the form "<sos> verb (Y, Y) ...."
    prediction = prediction.argmax(1)

    total = 0
    correct = 0
    for idx, b in enumerate(target):
      if b.shape[0] > 7:
        t = torch.logical_and(b[3] == b[5], b[7] == self._and)
        if t:
          if prediction[idx][3] == prediction[idx][5] and prediction[idx][3] == b[12]:
            correct += 1
          total += 1
    return correct, total

class LinearBError(BaseMetric):
  """
  Given a target of the form
    victor beside leo knows himself -> <sos> know ( victor , victor ) & beside ( victor , leo )	

  Test if the output is of form:  
    know ( victor , leo ) & beside ( .... )
  """

  def __init__(self, pad_token_id = None, and_token_id = None):
    super().__init__()
    self._pad = pad_token_id
    self._and = and_token_id

  def compute(self, prediction: Tensor, target: Tensor):

    # total = # target sequences of the form "<sos> verb ( X , X ) & p ( X , Y )"
    # correct = # predictions of the form "verb (X, Y) ...."
    prediction = prediction.argmax(1)

    total = 0
    correct = 0
    for idx, b in enumerate(target):
      if b.shape[0] > 7:
        t = torch.logical_and(b[3] == b[5], b[7] == self._and)
        if t:
          if torch.logical_and(prediction[idx][5] == b[12], prediction[idx][3] != prediction[idx][5]):
            correct += 1
          total += 1

    return correct, total

class LinearCError(BaseMetric):
  """
  Given a target of the form
    victor beside leo knows himself -> <sos> know ( victor , victor ) & beside ( victor , leo )	

  Test if the output is of form:  
    know ( leo , victor ) & beside ( .... )

  i.e., reflexives are structural but subj-verb is linear
  """

  def __init__(self, pad_token_id = None, and_token_id = None):
    super().__init__()
    self._pad = pad_token_id
    self._and = and_token_id

  def compute(self, prediction: Tensor, target: Tensor):

    # total = # target sequences of the form "<sos> verb ( X , X ) & p ( X , Y )"
    # correct = # predictions of the form "verb (X, Y) ...."
    prediction = prediction.argmax(1)

    total = 0
    correct = 0
    for idx, b in enumerate(target):
      if b.shape[0] > 7:
        t = torch.logical_and(b[3] == b[5], b[7] == self._and)
        if t:
          if torch.logical_and(prediction[idx][3] == b[12], prediction[idx][3] != prediction[idx][5]):
            correct += 1
          total += 1

    return correct, total

class HierarchicalAccuracy(BaseMetric):
  """
  Given a target of the form
    victor beside leo knows himself -> <sos> know ( victor , victor ) & beside ( victor , leo )	

  Test if the output is of form:  
    know ( victor , victor ) & beside ( .... )
  """

  def __init__(self, pad_token_id = None, and_token_id = None):
    super().__init__()
    self._pad = pad_token_id
    self._and = and_token_id

  def compute(self, prediction: Tensor, target: Tensor):

    # total = # target sequences of the form "<sos> verb ( X , X ) & p ( X , Y )"
    # correct = # predictions of the form "verb (X, Y) ...."
    prediction = prediction.argmax(1)

    total = 0
    correct = 0
    for idx, b in enumerate(target):
      if b.shape[0] > 7:
        t = torch.logical_and(b[3] == b[5], b[7] == self._and)
        if t:
          if torch.logical_and(prediction[idx][3] == b[3], prediction[idx][3] == prediction[idx][5]):
            correct += 1
          total += 1

    return correct, total

class NegAccuracy(BaseMetric):
  """
  If target contains a NEG token, check if the prediction has a NEG token in
  the same position.
  """

  def __init__(self, neg_token_id = 0):
    super().__init__()
    self._neg = neg_token_id
  
  def compute(self, prediction: Tensor, target: Tensor):
    prediction = prediction.argmax(1)

    tar_locs = target[:,] == self._neg
    total = 0
    correct = 0

    for k in torch.nonzero(tar_locs):
      total += 1
      seq, loc = k
      
      if prediction[seq][loc] == self._neg:
        correct += 1

    return correct, total

class FullSequenceModAccuracy(BaseMetric):
  """
  Calculates the accuracy of predictions modulo a set of equivalence classes
  defined on the tokens. Allows for a more "forgiving" measure of accuracy. 
  """

  @staticmethod
  def _merge_classes(lists):
    result = []
    for r in lists:
      for i, rr in enumerate(result):
        if r[0] == rr[0]:
          result[i] += r[1:]
          break
      else:
        result.append(r)
    return [sorted(list(set(r))) for r in result]

  def __init__(self, classes: List):
    super().__init__()
    self._classes = FullSequenceModAccuracy._merge_classes(classes)
  
  def compute(self, prediction: Tensor, target: Tensor):

    correct, total = 0, 0

    prediction = prediction.argmax(1)

    for c in self._classes:
      for toke in c[1:]:
        prediction[prediction==toke] = c[0]
        target[target==toke] = c[0]
    
    correct = (prediction == target).prod(axis=1)
    total = correct.shape[0]
    correct = correct.sum()

    return correct, total