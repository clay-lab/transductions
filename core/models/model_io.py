# model_io.py
#
# Defines an interface which represents inputs and outputs of a coder.

from typing import Dict
from torch import Tensor

class ModelIO:
  """
  Object containing tensors of input and output sequences. An object of this
  type should be passed to and returned from a TransductionModel encoder and
  decoder.
  """

  def __init__(self, attrs: Dict[str, Tensor] = {}):
    
    self.set_attributes(attrs)
  
  def set_attribute(self, attr: str, val: Tensor):
    setattr(self, attr, val)
  
  def set_attributes(self, attrs: Dict[str, Tensor]):
    for attr, val in attrs.items():
      self.set_attribute(attr, val)

  def __repr__(self) -> str:
    msg = type(self).__name__ + "\n"
    padding = ""
    for attr in self.__dict__:
      padding = " "
      msg += padding + "{}: {}\n".format(attr, getattr(self, attr).shape)
    return msg