import os
import logging
from typing import List
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

# Library imports
from core.metrics.base_metric import BaseMetric

log = logging.getLogger(__name__)

class Meter:
  """
  A collection of metrics which is used to monitor model performance during
  training.
  """

  def __init__(self, bundle: List[BaseMetric]):
    self._metrics = bundle
    
    self._tensorboard_dir = os.path.join(os.getcwd(), "tensorboard")
    self._writer = SummaryWriter(log_dir=self._tensorboard_dir)
    log.info("Logging with tensorboard; view with `tensorboard --logdir={}`".format(self._tensorboard_dir))
  
  def __call__(self, prediction: Tensor, target: Tensor):
    for metric in self._metrics:
      metric(prediction, target)
  
  def _log_to_tensorboard(self, stage: str, step: int):
    for metric in self._metrics:
      metric_name = metric.name + '/' + stage
      metric_value = metric.accuracy
      self._writer.add_scalar(metric_name, metric_value, step)

  def _display_metrics(self):
    msg = "Meter:\n"
    for i, metric in enumerate(self._metrics):
      if i > 0:
        msg += "\n"
      msg += "{}:\t{:4.3f}".format(metric.name, metric.accuracy)
    log.info(msg)

  def reset(self):
    for metric in self._metrics:
      metric.reset()
  
  def log(self, stage: str, step: int):
    self._log_to_tensorboard(stage=stage, step=step)
    self._display_metrics()
