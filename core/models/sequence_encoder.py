# sequence_encoder.py
# 
# Provides SequenceEncoder module.

import torch
from torch import nn
from omegaconf import DictConfig
from torchtext.vocab import Vocab

# Library imports
from core.models.model_io import ModelIO
from core.models.positional_encoding import PositionalEncoding

class SequenceEncoder(torch.nn.Module):

  @property
  def vocab_size(self):
    return len(self._vocabulary)

  def __init__(self, cfg: DictConfig, vocab: Vocab):
    
    super(SequenceEncoder, self).__init__()

    self._num_layers = cfg.num_layers
    self._hidden_size = cfg.hidden_size
    self._unit_type = cfg.unit.upper()
    
    self._vocabulary = vocab

    self._embedding = torch.nn.Embedding(self.vocab_size, self._hidden_size)
    self._dropout = torch.nn.Dropout(p=cfg.dropout)

    if self._unit_type == 'SRN':
      self._unit = nn.RNN(self._hidden_size, self._hidden_size, num_layers = self._num_layers, dropout = cfg.dropout)
    elif self._unit_type == 'GRU':
      self._unit = nn.GRU(self._hidden_size, self._hidden_size, num_layers = self._num_layers, dropout = cfg.dropout)
    elif self._unit_type == 'LSTM':
      self._unit = nn.LSTM(self._hidden_size, self._hidden_size, num_layers = self._num_layers, dropout = cfg.dropout)
    elif self._unit_type == 'TRANSFORMER':
      layer = nn.TransformerEncoderLayer(self._hidden_size, cfg.num_heads)
      self._unit = nn.TransformerEncoder(layer, num_layers=self._num_layers)
      self._pos_enc = PositionalEncoding(self._hidden_size, cfg.dropout)
      self._embedding = nn.Sequential(self._embedding, self._pos_enc)
    else:
      raise ValueError('Invalid recurrent unit type "{}".'.format(self._unit_type))
  
  def forward(self, enc_input: ModelIO) -> ModelIO:
    """
      Compute the forward pass.
    """
    embedded = self._dropout(self._embedding(enc_input.source))
    enc = self._unit(embedded)

    output = ModelIO()
    if isinstance(enc, tuple):
      enc_outputs, enc_hidden = enc
      output.set_attributes({
        "enc_outputs" : enc_outputs,
        "enc_hidden" : enc_hidden
      })
    else:
      enc_outputs = enc
      output.set_attributes({
        "enc_outputs" : enc_outputs
      })

    return output