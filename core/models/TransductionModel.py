# TransductionModel.py
# 
# Provides the base class for transduction models.
import hydra
import logging
import torch
from omegaconf import OmegaConf, DictConfig
from core.models.sequence_encoder import SequenceEncoder
from core.models.sequence_decoder import SequenceDecoder

log = logging.getLogger(__name__)

class TransductionModel(torch.nn.Module):
  """
  Provides the base class for transduction models.
  """

  def __init__(self, cfg: DictConfig, dataset, device):
    
    log.info("Initializing model")
    super(TransductionModel, self).__init__()

    self.device = device
    encoder_cfg = cfg.model.encoder
    decoder_cfg = cfg.model.decoder

    if cfg.experiment.dataset.source_format == 'sequence':
      self._encoder = SequenceEncoder(encoder_cfg, dataset.source_field.vocab)
    
    if cfg.experiment.dataset.target_format == 'sequence':
      self._decoder = SequenceDecoder(decoder_cfg, dataset.target_field.vocab)
    
    self._encoder.to(self.device)
    self._decoder.to(self.device)
  
  def forward(self, batch):
    """
    Runs the forward pass; called by `optimize` and `test`.
    """
    encoded, state = self._encoder(batch)
    decoded = self._decoder(state, target=batch.target)
    return decoded

  def backward(self, *args, **kwargs):
    """
    Runs the backward pass.
    """
    raise NotImplementedError("Backward call must be implemented.")

  def optimize(self, epoch, batch_size):
    """
    Calculate loss and gradients, and update the weights. Called during every 
    training iteration.
    """
    self._num_epochs = epoch
    self._num_batches += 1
    self._num_samples += batch_size

    self.forward(epoch=epoch)

    self.backward()
  