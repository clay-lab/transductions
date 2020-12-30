# TransductionModel.py
# 
# Provides the base class for transduction models.

import logging
import torch
from torchtext.data.batch import Batch
from omegaconf import DictConfig

# library imports
from core.models.sequence_encoder import SequenceEncoder
from core.models.sequence_decoder import SequenceDecoder
from core.dataset.TransductionDataset import TransductionDataset

log = logging.getLogger(__name__)

class TransductionModel(torch.nn.Module):
  """
  Provides the base class for transduction models.
  """

  def __init__(self, cfg: DictConfig, dataset: TransductionDataset, device):
    
    log.info("Initializing model")
    super(TransductionModel, self).__init__()

    self.device = device
    
    encoder_cfg = cfg.model.encoder
    encoder_vcb = dataset.source_field.vocab

    decoder_cfg = cfg.model.decoder
    decoder_vcb = dataset.target_field.vocab


    if cfg.experiment.dataset.source_format == 'sequence':
      self._encoder = SequenceEncoder(encoder_cfg, encoder_vcb)
    
    if cfg.experiment.dataset.target_format == 'sequence':
      self._decoder = SequenceDecoder(decoder_cfg, decoder_vcb)
    
    self._encoder.to(self.device)
    self._decoder.to(self.device)
  
  def forward(self, batch: Batch):
    """
    Runs the forward pass.
    """
    encoded, hidden = self._encoder(batch)
    decoded, hidden = self._decoder(hidden, batch.target)
    return decoded
  