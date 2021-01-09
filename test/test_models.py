# test_models.py
# 
# Testing code related to instantiating models.

import logging
import unittest
import torch
from omegaconf import OmegaConf, DictConfig
from torchtext.data import Field
import sys, os

# Library imports
# Since we are outside the main tree (/core/...), we 
# need to insert the root back into the syspath
DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..")
sys.path.append(ROOT)

from core.models.base_model import TransductionModel

class TestModels(unittest.TestCase):

  def test_srn_inattentive(self):
    """
    Tests instantiation of a simple SRN model without attention.
    """
    cfg_string = """
      model:
        encoder:
          unit: SRN
          type: sequence
          dropout: 0
          num_layers: 1
          hidden_size: 256
          max_length: 0
          embedding_size: 256
          bidirectional: False
        decoder:
          unit: SRN
          type: sequence
          dropout: 0
          num_layers: 1
          max_length: 30
          hidden_size: 256
          attention: null
          embedding_size: 256
      dataset:
        source_format: sequence
        target_format: sequence
        transform_field: source
    """

    cfg = OmegaConf.create(cfg_string)

    # Construct fake vocabularies so the model
    # won't complain
    test_field = Field() 
    test_field.build_vocab()
    test_field.vocab.stoi = {}
    test_field.vocab.stoi['pad'] = 0
    test_field.vocab.stoi['cls'] = 1

    src_vocab = tgt_vocab = test_field.vocab
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransductionModel(cfg, src_vocab, tgt_vocab, device)

if __name__ == "__main__":
  unittest.main()