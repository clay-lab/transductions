# bert_encoder.py
# 
# Provides BERTEncoder module.

from typing import Dict
import torch
from torch import nn
from omegaconf import DictConfig
from torchtext.vocab import Vocab
from transformers import BertTokenizer, BertModel

# Library imports
from core.models.model_io import ModelIO

class BERTEncoder(nn.Module):

  def __init__(self, cfg: DictConfig, vocab: Vocab):

    super().__init__()
    self.module = BertModel.from_pretrained('bert-base-uncased')

    # Freeze BERT layers to speed up training
    for param in self.module.parameters():
      param.requires_grad = False
  
  def forward(self, enc_input: ModelIO) -> ModelIO:

    encoded = self.module(enc_input.source)
    return ModelIO({"enc_outputs" : encoded.last_hidden_state})
