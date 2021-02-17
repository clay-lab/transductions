# attention.py
# 
# Defines attention modules.

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod, ABC

from torch import Tensor
from torchtext.vocab import Vocab
from transformers import DistilBertTokenizer

def create_mask(source: Tensor, vocab: Vocab) -> Tensor:

  source = source.T
  batch_size, max_len = source.shape
  if vocab is None:
    # BERT Encoder, get the index from the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    pad_index = tokenizer.convert_tokens_to_ids('[PAD]')
  else:
    pad_index = vocab.stoi['<pad>']

  mask = source.mul(source.ne(pad_index)).type(torch.bool)
  return mask

class Attention(nn.Module, ABC):

  @abstractmethod
  def forward(self, enc_outputs: Tensor, dec_hiddens: Tensor, src_mask: Tensor) -> Tensor:
    pass

class MultiplicativeAttention(Attention):

  def __init__(self, dec_size: int, enc_size: int = None, attn_size: int = None):
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

class AdditiveAttention(Attention):

  def __init__(self, dec_size: int, enc_size: int = None):
    super().__init__()

    self.dec_size = dec_size
    self.enc_size = dec_size if enc_size is None else enc_size

    self.enc_map = nn.Linear(self.enc_size, self.dec_size)
    self.dec_map = nn.Linear(self.dec_size, self.dec_size) 
    self.v = nn.Parameter(torch.rand(self.dec_size), requires_grad=True)

  def forward(self, enc_outputs: Tensor, dec_hiddens: Tensor, src_mask: Tensor) -> Tensor:
    # want dims of [batch_size, seq_len, hidden_size]
    enc_outputs = enc_outputs.permute(1,0,2)
    dec_hiddens = dec_hiddens.unsqueeze(1)
    mapped_enc = self.enc_map(enc_outputs)
    mapped_dec = self.dec_map(dec_hiddens)
    weights = mapped_enc + mapped_dec
    weights = torch.tanh(weights) @ self.v
    weights[~src_mask] = -float("Inf")      
    return F.softmax(weights, dim=1)

class DotProductAttention(Attention):

  def __init__(self):
    super().__init__()
  
  def forward(self, enc_outputs: Tensor, dec_hiddens: Tensor, src_mask: Tensor) -> Tensor:
    value = enc_outputs.permute(1,0,2)
    key = dec_hiddens.unsqueeze(2)
    weights = torch.bmm(value, key)
    weights = weights.squeeze(2)
    weights[~src_mask] = -float("Inf")
    # result is of shape (batch, seq_len)
    return F.softmax(weights, dim=1)
