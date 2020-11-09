import logging
from omegaconf import DictConfig
from torchtext.data import Field, TabularDataset, BucketIterator, RawField

log = logging.getLogger(__name__)

class TransductionDataset:

  def __init__(self, config: DictConfig):
    log.info("Initializing dataset")