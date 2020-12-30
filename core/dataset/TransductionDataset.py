import logging
import os
import hydra
import torch
from typing import Dict
import fileinput
import re
import numpy as np
from omegaconf import DictConfig
import pickle
from torch._C import dtype
from torchtext.data import Field, TabularDataset, BucketIterator, RawField

log = logging.getLogger(__name__)

class TransductionDataset:

  @property
  def iterators(self):
    return self._iterators
  
  def _process_raw_data(self, cfg: DictConfig):
    """
    Creates .pt files from input in the data/raw/ directory. Each created
    file represents a train, test, val, or gen split, or a separate tracking
    split.

    Note that {train, test, val, gen} are fully disjoint and their union is
    simple the input file. The remaining tracking splits are not necessarily
    disjoint from any of the regular splits, and are used only to track a
    model's performance on certain subsets of dataset during and after training.
    """
    log.info('Processing raw data')
    log.info('Creating dataset at {}'.format(self._processed_path))
    self._create_regular_splits(cfg)
    self._create_tracking_splits(cfg)

  def _create_regular_splits(self, cfg: DictConfig):
    """
    Creates train, val, test splits, and a gen split if withholding patterns
    are specified in configuration. All files are written to the processed
    data path as .pt files.
    """

    splits = cfg.experiment.dataset.splits
    to_withhold = re.compile('|'.join(cfg.experiment.dataset.withholding))

    s_names = list(splits.keys())
    s_paths = [os.path.join(self._processed_path, s + '.pt') for s in s_names]
    s_probs = [float(p) / 100.0 for p in list(splits.values())]
    g_path = os.path.join(self._processed_path, 'gen.pt')

    for split in s_paths:
      if os.path.isfile(split):
        os.remove(split)
    if os.path.isfile(g_path):
      os.remove(g_path)

    with open(self._raw_path) as raw_data:
      next(raw_data)
      for line in raw_data:
        if to_withhold.match(line):
          with open(g_path, 'a') as f:
            f.write(line)
        else:
          split = np.random.choice(s_paths, p=s_probs)
          with open(split, 'a') as f:
            f.write(line)
  
  def _create_tracking_splits(self, cfg: DictConfig):
    """
    Creates .pt split files based on named tracking patterns provided in 
    configuration. All inputs which match these patterns are saved to the
    corresponding tracking file.
    """

    tracking = cfg.experiment.dataset.tracking
    t_names = list(tracking.keys())
    t_paths = [os.path.join(self._processed_path, t + '.pt') for t in t_names]
    t_patterns = list(tracking.values())
    t_patterns = [re.compile(t) for t in t_patterns]

    for path in t_paths:
      if os.path.exists(path):
        os.remove(path)
    
    with open(self._raw_path) as raw_data:
      next(raw_data)
      for line in raw_data:
        for i, pattern in enumerate(t_patterns):
          if pattern.match(line):
            with open(t_paths[i], 'a') as f:
              f.write(line)

  def _create_iterators(self, cfg: DictConfig):
    """
    Constructs TabularDatasets and iterators for the processed data.
    """
    source_format = cfg.experiment.dataset.source_format
    target_format = cfg.experiment.dataset.target_format

    if source_format == 'sequence' and target_format == 'sequence':

      self.source_field = Field(lower=True, eos_token="<eos>") 
      self.target_field = Field(lower=True, eos_token="<eos>") 
      self.transform_field = self.source_field

      datafields = [("source", self.source_field), 
                    ("annotation", self.transform_field), 
                    ("target", self.target_field)]
      
      self._iterators = {}
      self._in_sample_data = []
      
      for file in os.listdir(self._processed_path):
        if file.endswith('.pt'):
          split = file.split('.')[0]
          dataset = TabularDataset(
            path = os.path.join(self._processed_path, file),
            format = 'tsv',
            skip_header = True,
            fields = datafields)
          iterator = BucketIterator(
            dataset,
            device = self._device,
            batch_size = cfg.training.batch_size,
            sort_key = lambda x: len(x.target), 
            sort_within_batch = True, 
            repeat = False)
          
          self._iterators[split] = iterator
          if split in ['train', 'test', 'val']:
            self._in_sample_data.append(dataset)

  def id_to_token(self, idx_tensor, vocab_str: str):
    """
    Returns a tensor containing tokens from the specified vocabulary. 

    Parameters:
      - idx_tensor (torch.tensor of shape [batch, seq_len]): index tensor of    
          inputs
      - vocab_str (str): one of 'source', 'target'
    """

    vocab = self.source_field.vocab if vocab_str == 'source' else self.target_field.vocab
    outputs = np.empty(idx_tensor.detach().cpu().numpy().shape, dtype=object)

    for idr, r in enumerate(idx_tensor):
      for idc, _ in enumerate(r):
        outputs[idr][idc] = vocab.itos[idx_tensor[idr][idc]]
    return outputs

  def __init__(self, cfg: DictConfig, device):

    log.info("Initializing dataset")

    self._processed_path = os.path.join(hydra.utils.get_original_cwd(), 
      'data/experiments', cfg.experiment.name)
    self._raw_path = os.path.join(hydra.utils.get_original_cwd(), 'data/raw', 
      cfg.experiment.dataset.input)
    self._device = device
    
    if not os.path.exists(self._processed_path):
      os.mkdir(self._process_path)
      splits = self._process_raw_data(cfg)
    else:
      if cfg.experiment.dataset.overwrite:
        splits = self._process_raw_data(cfg)
    
    self._create_iterators(cfg)

    self.source_field.build_vocab(*self._in_sample_data)
    self.target_field.build_vocab(*self._in_sample_data)

    pickle.dump(self.source_field, open('source.pt', 'wb'))
    pickle.dump(self.target_field, open('target.pt', 'wb'))

  def __repr__(self):
    message = "Dataset: %s \n" % self.__class__.__name__
    for attr in self.__dict__:
      if '_iterators' in attr:
        message += "Splits:\n"
        for s in getattr(self, attr):
          message += "\t'{}' ({} sequences)\n".format(s, len(getattr(self, attr)[s].dataset) + 1)
      elif '_path' in attr:
        clean_name = {'_processed_path' : 'Processed data',
                      '_raw_path' : 'Raw data'}
        message += "{}:\t '{}'\n".format(clean_name[attr], getattr(self, attr))
      elif '_in_sample_data' in attr:
        message += "In-sample Data:\n"
        isd = [str(d) for d in getattr(self, attr)]
        message += "\t{}\n".format(isd)
      # message += "{}:\t {}\n".format(attr, getattr(self, attr))
    # for attr in self.__dict__:
    #     if "transform" in attr:
    #         message += "{}{} {}= {}\n".format(COLORS.IPurple, attr, COLORS.END_NO_TOKEN, getattr(self, attr))
    # for attr in self.__dict__:
    #     if attr.endswith("_dataset"):
    #         dataset = getattr(self, attr)
    #         if isinstance(dataset, list):
    #             if len(dataset) > 1:
    #                 size = ", ".join([str(len(d)) for d in dataset])
    #             else:
    #                 size = len(dataset[0])
    #         elif dataset:
    #             size = len(dataset)
    #         else:
    #             size = 0
    #         if attr.startswith("_"):
    #             attr = attr[1:]
    #         message += "Size of {}{} {}= {}\n".format(COLORS.IPurple, attr, COLORS.END_NO_TOKEN, size)
    # for key, attr in self.__dict__.items():
    #     if key.endswith("_sampler") and attr:
    #         message += "{}{} {}= {}\n".format(COLORS.IPurple, key, COLORS.END_NO_TOKEN, attr)
    # message += "{}Batch size ={} {}".format(COLORS.IPurple, COLORS.END_NO_TOKEN, self.batch_size)
    return message

