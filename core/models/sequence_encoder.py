# sequence_encoder.py
# 
# Provides SequenceEncoder module.

import torch
from torch import nn, Tensor
from omegaconf import DictConfig
from torchtext.vocab import Vocab

# Library imports
from core.models.model_io import ModelIO
from core.models.positional_encoding import PositionalEncoding

class SequenceEncoder(torch.nn.Module):

  @property
  def vocab_size(self):
    return len(self.vocab)
  
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
  def unit_type(self) -> str:
    return str(self.cfg.unit).upper()
  
  @property
  def dropout_p(self) -> float:
    return float(self.cfg.dropout)

  def __init__(self, cfg: DictConfig, vocab: Vocab):
    
    super().__init__()

    self.cfg = cfg
    self.vocab = vocab

    # self._num_layers = cfg.num_layers
    # self._hidden_size = cfg.hidden_size
    # self._embedding_size = cfg.embedding_size
    # self._unit_type = cfg.unit.upper()
    # self._dropout_p = cfg.dropout
    
    # self._vocabulary = vocab

    embedding = torch.nn.Embedding(self.vocab_size, self.embedding_size)

    if self.unit_type == 'SRN':
      unit = nn.RNN(
        self.embedding_size, 
        self.hidden_size, 
        num_layers = self.num_layers, 
        dropout = self.dropout_p
      )
    elif self.unit_type == 'GRU':
      unit = nn.GRU(
        self.embedding_size, 
        self.hidden_size, 
        num_layers = self.num_layers, 
        dropout = self.dropout_p
      )
    elif self.unit_type == 'LSTM':
      unit = nn.LSTM(
        self.embedding_size, 
        self.hidden_size, 
        num_layers = self.num_layers, 
        dropout = self.dropout_p
      )
    elif self.unit_type == 'TRANSFORMER':
      layer = nn.TransformerEncoderLayer(self.hidden_size, cfg.num_heads)
      unit = nn.TransformerEncoder(layer, num_layers=self.num_layers)
      pos_enc = PositionalEncoding(self.hidden_size, self.dropout_p)
      embedding = nn.Sequential(embedding, pos_enc)
    else:
      raise ValueError('Invalid recurrent unit type "{}".'.format(self._unit_type))

    self.module = nn.Sequential(
      embedding,
      unit
    )
  
  def tok_to_id(self, tokens: Tensor):
    pass
  
  def forward(self, enc_input: ModelIO) -> ModelIO:
    """
      Compute the forward pass.
    """
    enc = self.module(enc_input.source)

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