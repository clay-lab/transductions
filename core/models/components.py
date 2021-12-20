# components.py
#
# Defines abstract components, to be used as building blocks for a
# Transductions model

import logging
from abc import abstractmethod
from typing import List
from torch import Tensor
from torch.nn import Module
from omegaconf import DictConfig

# Library imports
from core.models.model_io import ModelIO

log = logging.getLogger(__name__)


class TransductionComponent(Module):

    # BEGIN computed properties

    @property
    @abstractmethod
    def vocab_size(self):
        return

    @property
    def num_layers(self) -> int:
        return int(self.cfg.num_layers)

    @property
    def hidden_size(self) -> int:
        return int(self.cfg.hidden_size)

    @property
    def embedding_size(self) -> int:
        return int(self.cfg.embedding_size)

    @property
    def dropout_p(self) -> float:
        return float(self.cfg.dropout)

    @property
    def unit_type(self) -> str:
        return str(self.cfg.unit).upper()

    # END computed properties

    def __init__(self, cfg: DictConfig):

        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(self, inputs: ModelIO) -> ModelIO:
        """
        Runs the forward pass for the component, where inputs are sent through
        the various units and packaged as the appropriate output type.
        """
        return

    @abstractmethod
    def to_tokens(self, tensor: Tensor):
        """
        Converts a tensor of indices into an array of strings. Implementation
        depends on how the vocab object is defined for the component.
        """
        return

    @abstractmethod
    def to_ids(self, tokens: List):
        """
        Converts a list of tokens [<sos>, 'the', 'man', ... , <eos>] into a list
        of ids in the component's vocabulary.
        """
        pass
