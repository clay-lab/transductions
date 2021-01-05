# attention.py
# 
# Defines attention modules.

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod

from torch import Tensor
from torchtext.vocab import Vocab

def create_mask(source: Tensor, vocab: Vocab) -> Tensor:

  source = source.T
  batch_size, max_len = source.shape
  pad_index = vocab.stoi['<pad>']

  mask = source.mul(source.ne(pad_index)).type(torch.bool)
  return mask


class Attention(nn.Module):

  @abstractmethod
  def forward(self, enc_outputs: Tensor, dec_hiddens: Tensor, src_mask: Tensor) -> Tensor:
    raise NotImplementedError

class MultiplicativeAttention(Attention):

  def __init__(self, dec_size, enc_size=None, attn_size=None):
    super().__init__()

    self.dec_size = dec_size
    self.enc_size = dec_size if enc_size is None else enc_size
    self.attn_size = dec_size if attn_size is None else attn_size

    self.key_map = nn.Linear(self.dec_size, self.attn_size)
    self.val_map = nn.Linear(self.enc_size, self.attn_size)

  def forward(self, enc_outputs: Tensor, dec_hiddens: Tensor, src_mask: Tensor) -> Tensor:
    # want dims of [batch_size, seq_len, hidden_size]
    enc_outputs = enc_outputs.permute(1,0,2)
    value = self.val_map(enc_outputs)
    key = self.key_map(dec_hiddens).unsqueeze(2)

    weights = torch.bmm(value, key).squeeze(2)
    weights[~src_mask] = -float("Inf")

    return F.softmax(weights, dim=1)
