# few_shot_trainer.py
# 
# Training loop for few-shot GPT-2 models

import logging 
import torch
from omegaconf import DictConfig

# Library imports
from core.models.few_shot_model import FewShotModel
from core.dataset.few_shot_dataset import FewShotDataset

log = logging.getLogger(__name__)

class FewShotTrainer:

  def __init__(cfg: DictConfig):

    self._model = FewShotModel(cfg.experiment.model)
    self._dataset = FewShotDataset(cfg.experiment.dataset)