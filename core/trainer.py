import logging
import hydra
import torch
from tqdm import tqdm
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
    - Visualizer?
  """

  def __init__(self, cfg):
    self._cfg = cfg
    self._instantiate()

  def _instantiate(self):

    self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info("DEVICE: {}".format(self._device))

    self._dataset = TransductionDataset(self._cfg, self._device)
    log.info(self._dataset)

    self._model = TransductionModel(self._cfg, self._dataset, self._device)
    log.info(self._model)
  
  def train(self):

    self._is_training = True
    log.info("Beginning training")

    criterion = torch.nn.CrossEntropyLoss(weight=None)
    optimizer = torch.optim.SGD(self._model.parameters(), lr=self._cfg.training.lr)

    for epoch in range(self._cfg.training.epochs):

      log.info("EPOCH %i / %i", epoch + 1, self._cfg.training.epochs)

      self._model.train()
      with tqdm(self._dataset.iterators['train']) as T:
        for batch in T:

          optimizer.zero_grad()

          output = self._model(batch)

          # Loss expects:
          #   input:  [batch_size, classes, seq_len]
          #   target: [batch_size, seq_len]
          predictions = output
          target = batch.target.permute(1, 0)

          loss = criterion(predictions, target)
          loss.backward()
          optimizer.step()
