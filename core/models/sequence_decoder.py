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
    self.out = torch.nn.Linear(self._hidden_size, self.vocab_size)

    if self._num_layers == 1:
      assert cfg.dropout == 0, "Dropout must be zero if num_layers = 1"

    if self._unit_type == "SRN":
      self.unit = torch.nn.RNN(self._embedding_size, self._hidden_size, num_layers=self._num_layers, dropout=cfg.dropout)
    elif self._unit_type == "GRU":
      self.unit = torch.nn.GRU(self._embedding_size, self._hidden_size, num_layers=self._num_layers, dropout=cfg.dropout)
    elif self._unit_type == "LSTM":
      self.unit = torch.nn.LSTM(self._embedding_size, self._hidden_size, num_layers=self._num_layers, dropout=cfg.dropout)
    elif self._unit_type == 'TRANSFORMER':
      pass
    else:
      raise ValueError('Invalid recurrent unit type "{}".'.format(self._unit_type))
  
  def forward(self, batch, target=None):
    
    # During training, force outputs to be the same length as the target
    gen_length = self._max_length if target is None else target.shape[0]
    
    inputs = torch.squeeze(batch).long()
    embedded = self._embedding(inputs)
    dropped = self._dropout(embedded)
    output = self.out(dropped)

    if target.shape[1] == 1:
      output = torch.unsqueeze(output, 0)
    
    return output[:,:, :gen_length]
    