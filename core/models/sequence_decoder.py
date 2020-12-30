# sequence_decoder.py
# 
# Provides SequenceDecoder module.

import torch
from omegaconf import DictConfig
from torchtext.vocab import Vocab

class SequenceDecoder(torch.nn.Module):

  @property
  def vocab_size(self):
    return len(self._vocabulary)

  def __init__(self, cfg: DictConfig, vocab: Vocab):
    
    super(SequenceDecoder, self).__init__()

    self._num_layers = cfg.num_layers
    self._hidden_size = cfg.hidden_size
    self._unit_type = cfg.unit.upper()
    self._max_length = cfg.max_length
    self._embedding_size = cfg.embedding_size
    self._attention = cfg.attention.lower()
    if self._attention != 'none':
      self._embedding_size += self._hidden_size
    
    self._vocabulary = vocab
    self._pad_index = self._vocabulary.stoi['pad']
    self._cls_index = self._vocabulary.stoi['cls']

    self._embedding = torch.nn.Embedding(self.vocab_size, self._hidden_size)
    self._dropout = torch.nn.Dropout(p=cfg.dropout)

    if self._num_layers == 1:
      assert cfg.dropout == 0, "Dropout must be zero if num_layers = 1"

    if self._unit_type == "SRN":
      self._unit = torch.nn.RNN(self._embedding_size, self._hidden_size, num_layers=self._num_layers, dropout=cfg.dropout)
    elif self._unit_type == "GRU":
      self._unit = torch.nn.GRU(self._embedding_size, self._hidden_size, num_layers=self._num_layers, dropout=cfg.dropout)
    elif self._unit_type == "LSTM":
      self._unit = torch.nn.LSTM(self._embedding_size, self._hidden_size, num_layers=self._num_layers, dropout=cfg.dropout)
    elif self._unit_type == "TRANSFORMER":
      raise NotImplementedError
    else:
      raise ValueError("Invalid unit type '{}''.".format(self._unit_type))
  
    self._out = torch.nn.Linear(self._hidden_size, self.vocab_size)
  
  def forward(self, input, hidden, cell):
    
    # Strip 0th dimension only!
    input = input.unsqueeze(0).long()

    # print("Input:", input.shape)
    # print("Hidden:", hidden.shape)
    # print("Cell:", [c.shape for c in cell])

    embedded = self._dropout(self._embedding(input))
    output, (hidden, cell) = self._unit(embedded, (hidden, cell))
    output = self._out(output)

    return output, hidden, cell
    