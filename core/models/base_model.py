# TransductionModel.py
# 
# Provides the base class for transduction models.

import torch
import logging
from omegaconf import DictConfig
from torchtext.vocab import Vocab
from torchtext.data.batch import Batch
from transformers.utils.dummy_pt_objects import BertModel

# library imports
from core.models.sequence_encoder import SequenceEncoder
from core.models.bert_encoder import BERTEncoder
from core.models.sequence_decoder import SequenceDecoder
from core.models.model_io import ModelIO
from core.dataset.base_dataset import TransductionDataset

log = logging.getLogger(__name__)

class TransductionModel(torch.nn.Module):
  """
  Provides the base class for a sequence-to-sequence model. Models are 
  specified as encoder/decoder pairs in the `config/model` directory.
  Encoders and decoders are responsible for implementing their own 
  forward pass logic. 

  Inputs to the encoders and decoders are sent through `ModelInput` objects,
  which contain attributes like `source`, `target`, `enc_hidden`, and so on.
  """

  def __init__(self, cfg: DictConfig, src_vocab: Vocab, tgt_vocab: Vocab, device):
    
    log.info("Initializing model")
    super(TransductionModel, self).__init__()

    self.device = device
    
    encoder_cfg = cfg.model.encoder
    decoder_cfg = cfg.model.decoder

    if cfg.dataset.source_format == 'sequence':
      if encoder_cfg.unit == 'BERT':
        self._encoder = BERTEncoder(encoder_cfg, src_vocab)
      else:
        self._encoder = SequenceEncoder(encoder_cfg, src_vocab)
    else:
      raise NotImplementedError
    
    if cfg.dataset.target_format == 'sequence':
      self._decoder = SequenceDecoder.newDecoder(src_vocab, tgt_vocab, decoder_cfg, encoder_cfg)
    else:
      raise NotImplementedError
    
    self._encoder.to(self.device)
    self._decoder.to(self.device)
  
  def forward(self, batch: Batch, tf_ratio: float = 0.0):
    """
    Runs the forward pass.

    batch (torchtext Batch): batch of [source, annotation, target]
    tf_ratio (float in range [0, 1]): if present, probability of using teacher
      forcing.
    """

    enc_input = ModelIO({"source" : batch.source})
    enc_output = self._encoder(enc_input)

    enc_output.set_attributes({
      "source" : batch.source, 
      "transform" : batch.annotation
    })

    if hasattr(batch, 'target'):
      enc_output.set_attribute("target", batch.target)

    dec_output = self._decoder(enc_output, tf_ratio=tf_ratio)

    return dec_output.dec_outputs
