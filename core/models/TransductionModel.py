# TransductionModel.py
# 
# Provides the base class for transduction models.

import torch
import random
import logging
from omegaconf import DictConfig
from torch._C import dtype
from torchtext.data.batch import Batch

# library imports
from core.models.sequence_encoder import SequenceEncoder
from core.models.sequence_decoder import SequenceDecoder
from core.dataset.TransductionDataset import TransductionDataset

log = logging.getLogger(__name__)

class TransductionModel(torch.nn.Module):
  """
  Provides the base class for transduction models.
  """

  @property
  def max_len(self):
    return 3

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
  
  def forward(self, batch: Batch, tf_prob = None):
    """
    Runs the forward pass.

    batch (torchtext Batch): batch of [source, annotation, target]
    tf_prob (float in range [0, 1]): if present, probability of using teacher
      forcing.
    """

    batch_size = batch.source.shape[1]
    target_voc = self._decoder.vocab_size
    target_len = batch.target.shape[0] if hasattr(batch, 'target') else self.max_len
    SOS_TOK = self._decoder._vocabulary.stoi['<sos>']

    outputs = torch.zeros(target_len, batch_size, target_voc).to(self.device)
    input = torch.Tensor([SOS_TOK for i in range(batch_size)]).int()

    # outputs, hidden
    _, hidden = self._encoder(batch.source)

    for t in range(target_len):
      
      output, hidden = self._decoder(input, hidden)
      outputs[t] = output

      if tf_prob:
        input = output.argmax(1) if random.random() < tf_prob else batch.target[t]
      else: 
        input = batch.target[t]

    return outputs
  
 