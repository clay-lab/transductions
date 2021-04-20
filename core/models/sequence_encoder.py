# sequence_encoder.py
# 
# Provides SequenceEncoder module.

import torch
import logging
from typing import Any, List
from torch import nn, Tensor
from omegaconf import DictConfig
from torchtext.vocab import Vocab
import numpy as np

# Library imports
from core.models.model_io import ModelIO
from core.models.bert_encoder import BERTEncoder
from core.models.components import TransductionComponent
from core.models.positional_encoding import PositionalEncoding

log = logging.getLogger(__name__)

class SequenceEncoder(TransductionComponent):

  def __new__(cls, cfg: DictConfig, vocab=None) -> Any:
    """
    This is an alternative to using a Factory paradigm. We always return 
    some subtype of SequenceEncoder, depending on what kind of unit we want
    to create. Currently, this is split between "BERT-like" encoders from
    huggingface and our own custom implementations of recurrent / transformer
    encoders.
    """
    unit_type = str(cfg.unit).upper()
    
    if unit_type == "BERT":
      return BERTEncoder(cfg=cfg)
    else:
      return TransductionSequenceEncoder(cfg=cfg, vocab=vocab)


class TransductionSequenceEncoder(TransductionComponent):

  @property
  def vocab_size(self):
    return len(self.vocab)

  def __init__(self, cfg: DictConfig, vocab: Vocab):
    
    super().__init__(cfg)

    self.vocab = vocab

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
  
  def to_tokens(self, idx_tensor: Tensor, show_special=False):
    outputs = np.empty(idx_tensor.detach().cpu().numpy().shape, dtype=object)

    for idr, r in enumerate(idx_tensor):
      for idc, _ in enumerate(r):
        string = self.vocab.itos[idx_tensor[idr][idc]]
        if string not in ['<sos>', '<eos>', '<pad>'] or show_special:
          outputs[idr][idc] = self.vocab.itos[idx_tensor[idr][idc]]
    
    batch_strings = []
    for r in outputs:
      batch_strings.append(r[r != np.array(None)])

    return batch_strings
  
  def to_ids(self, tokens: List):
    return [self.vocab.stoi[t] for t in tokens]
  
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