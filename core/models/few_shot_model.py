# few_shot_model.py
# 
# Model wrapper for GPT-2-eqsue models.

import logging
from torch import nn
from omegaconf import DictConfig
from transformers import GPT2Tokenizer, GPT2LMHeadModel

log = logging.getLogger(__name__)

class FewShotModel(nn.Module):
  
  def __init__(cfg: DictConfig):

    self._model_name = cfg.model_name

    self._tokenizer = GPT2Tokenizer.from_pretrained(self._model_name)
    self._model = GPT2LMHeadModel.from_pretrained(self._model_name, return_dict = True)

    