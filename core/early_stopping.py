# early_stopping.py
#
# Provides a way to stop training early if the model isn't making any headway.

import logging
import torch
from omegaconf import DictConfig

# Library imports
from core.models.base_model import TransductionModel

log = logging.getLogger(__name__)

class EarlyStopping:
  
  def __init__(self, cfg: DictConfig,):

    self.patience = cfg.patience
    self.tolerance = cfg.tolerance

    # Counters
    self.counter = 0
    self.best_score = None
    self.should_stop = False
    self.should_save = True

  def __call__(self, loss):
    
    score = -loss

    if self.best_score is None:
      self.best_score = score
    elif score < self.best_score + self.tolerance:
      self.counter += 1
      log.info(f'Early stopping counter: {self.counter} / {self.patience}')
      self.should_stop = self.counter >= self.patience
      self.should_save = False
    else:
      self.best_score = score
      self.counter = 0
      self.should_save = True
    
    return self.should_stop, self.should_save
  

