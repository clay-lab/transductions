import logging
import hydra
import torch
from omegaconf import OmegaConf
from core.models.TransductionModel import TransductionModel
from core.dataset.TransductionDataset import TransductionDataset

log = logging.getLogger(__name__)

class Trainer:
  """
  Handles interface between:
    - TransductionModel
    - Dataset
    - Checkpoint?
    - Visualizer
  """

  def __init__(self, cfg):
    self._cfg = cfg
    self._instantiate()

  def _instantiate(self):

    if self._cfg.training.cuda > -1 and torch.cuda.is_available():
      device = "cuda"
      torch.cuda.set_device(self._cfg.training.cuda)
    else:
      device = "cpu"
    
    self._device = device
    log.info("DEVICE: {}".format(self._device))

    self._dataset = TransductionDataset(self._cfg)
    self._model = TransductionModel(self._cfg, self._dataset.vocab)
  
  def train(self):

    self._is_training = True