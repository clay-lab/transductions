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
from core.dataset.base_dataset import TransductionDataset

log = logging.getLogger(__name__)

class TransductionModel(torch.nn.Module):
  """
  Provides the base class for transduction models.
  """

  @property
  def max_len(self):
    return 30

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
    else:
      raise NotImplementedError
    
    if cfg.experiment.dataset.target_format == 'sequence':
      self._decoder = SequenceDecoder(decoder_cfg, decoder_vcb)
    else:
      raise NotImplementedError
    
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
    EOS_TOK = self._decoder._vocabulary.stoi['<eos>']
    PAD_TOK = self._decoder._vocabulary.stoi['<pad>']

    outputs = torch.zeros(target_len, batch_size, target_voc).to(self.device)
    outputs[:,:,PAD_TOK] = 1.0
    input = torch.Tensor([SOS_TOK for _ in range(batch_size)]).int()
    has_finished = torch.zeros(batch_size).to(self.device)
    t = 0

    # outputs, hidden
    _, hidden = self._encoder(batch.source)

    while True:

      # Compute forward step
      output, hidden = self._decoder(input, hidden)
      outputs[t] = output
      best_guess = output.argmax(2).squeeze(0)

      # Check if output is <eos>
      for i, _ in enumerate(best_guess):
        if best_guess[i] == EOS_TOK:
          has_finished[i] = 1

      # Decide teacher-forcing
      if not hasattr(batch, 'target'):
        input = best_guess
        if tf_prob is not None:
          log.warning('You have specified a teacher-forcing ratio but your batch does not contain a target; teacher-forcing is not supported without a target.')
      else:
        if tf_prob is not None:
          input = best_guess if random.random() < tf_prob else batch.target[t]
        else: 
          input = batch.target[t]

      # Break
      if has_finished.prod().item() != 0 or t == target_len - 1:
        break
      else:
        t += 1

    return outputs
  
 