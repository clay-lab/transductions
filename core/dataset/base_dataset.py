import logging
import os
import hydra
import re
import numpy as np
from omegaconf import DictConfig
from typing import Dict
import shutil
import pickle
from torchtext.data import Field, TabularDataset, BucketIterator
from transformers import DistilBertTokenizer

log = logging.getLogger(__name__)

class TransductionDataset:

  @property
  def iterators(self):
    return self._iterators

  @property
  def datafields(self):
    return [("source", self.source_field), 
            ("annotation", self.transform_field), 
            ("target", self.target_field)]
  
  @property
  def transform_field(self):
    if self._trns_field == 'source':
      return self.source_field
    elif self._trns_field == 'target':
      return self.target_field
    else:
      log.error("`transform_field` must be either 'source' or 'target'; you supplied {}!".format(self.trns_field))
      raise ValueError("Invalid `transform_field`: {}".format(self.trns_field))

  
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

    splits = cfg.dataset.splits
    if 'withholding' in cfg.dataset.keys():
      to_withhold = re.compile('|'.join(cfg.dataset.withholding))
      g_path = os.path.join(self._processed_path, 'gen.pt')
      if os.path.isfile(g_path):
        os.remove(g_path)

    s_names = list(splits.keys())
    s_paths = [os.path.join(self._processed_path, s + '.pt') for s in s_names]
    s_probs = [float(p) / 100.0 for p in list(splits.values())]

    for split in s_paths:
      if os.path.isfile(split):
        os.remove(split)

    with open(self._raw_path) as raw_data:
      next(raw_data)
      for line in raw_data:
        if 'withholding' in cfg.dataset.keys() and bool(to_withhold.search(line)):
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

    if 'tracking' in cfg.dataset.keys():
      tracking = cfg.dataset.tracking
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
            if bool(pattern.search(line)):
              with open(t_paths[i], 'a') as f:
                f.write(line)

  def _create_iterators(self, cfg: DictConfig):
    """
    Constructs TabularDatasets and iterators for the processed data.
    """
    
    self._iterators = {}
    self._in_sample_data = []
    
    for file in os.listdir(self._processed_path):
      if file.endswith('.pt'):
        split = file.split('.')[0]
        dataset = TabularDataset(
          path = os.path.join(self._processed_path, file),
          format = 'tsv',
          skip_header = True,
          fields = self.datafields)
        iterator = BucketIterator(
          dataset,
          device = self._device,
          batch_size = cfg.hyperparameters.batch_size,
          sort_key = lambda x: len(x.target), 
          sort_within_batch = True, 
          repeat = False)
        
        self._iterators[split] = iterator
        if split in ['train', 'test', 'val']:
          self._in_sample_data.append(dataset)

  def id_to_token(self, idx_tensor, vocab_str: str, show_special = False):
    """
    Returns a tensor containing tokens from the specified vocabulary. 

    Parameters:
      - idx_tensor (torch.tensor of shape [batch, seq_len]): index tensor of    
          inputs
      - vocab_str (str): one of 'source', 'target'
      - show_special (bool): include special tokens like <pad>, <sos>, <eos>
    """

    vocab = self.source_field.vocab if vocab_str == 'source' else self.target_field.vocab
    outputs = np.empty(idx_tensor.detach().cpu().numpy().shape, dtype=object)

    for idr, r in enumerate(idx_tensor):
      for idc, _ in enumerate(r):
        string = vocab.itos[idx_tensor[idr][idc]]
        if string not in ['<sos>', '<eos>', '<pad>'] or show_special:
          outputs[idr][idc] = vocab.itos[idx_tensor[idr][idc]]
    
    batch_strings = []
    for r in outputs:
      batch_strings.append(r[r != np.array(None)])

    return batch_strings

  def __init__(self, cfg: DictConfig, device, fields: Dict = None, BERT = False):

    log.info("Initializing dataset")

    data_path = os.path.join(hydra.utils.get_original_cwd(), 'data')
    # data_path = '/Users/jacksonpetty/Documents/Development/transductions/data'
    process_root_path = os.path.join(data_path, 'processed')
    self._processed_path = os.path.join(process_root_path, cfg.dataset.name)
    self._raw_path = os.path.join(data_path, 'raw', cfg.dataset.input)
    self._device = device

    self._trns_field = cfg.dataset.transform_field.lower()

    if not os.path.exists(data_path):
      os.mkdir(data_path)
    
    if not os.path.exists(process_root_path):
      os.mkdir(process_root_path)
    
    if not os.path.exists(self._processed_path):
      os.mkdir(self._processed_path)
      self._process_raw_data(cfg)
    else:
      if cfg.dataset.overwrite:
        shutil.rmtree(self._processed_path)
        os.mkdir(self._processed_path)
        self._process_raw_data(cfg)

    # Construct fields
    source_format = cfg.dataset.source_format.lower()
    target_format = cfg.dataset.target_format.lower()

    if fields is not None:
      log.info("Using provided fields: {}".format(list(fields.keys())))
      if source_format == 'sequence' and target_format == 'sequence':
        self.source_field = fields['source']
        self.target_field = fields['target']

        if self.source_field is None:
          tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased') if BERT else str.split
          lower = False if BERT else True
          pad = tokenizer.convert_tokens_to_ids(tokenizer.pad_token) if BERT else '<pad>'
          sos = tokenizer.convert_tokens_to_ids(tokenizer.bos_token) if BERT else '<sos>'
          eos = None if BERT else '<eos>'
          tok_fun = tokenizer.encode if BERT else tokenizer
          self.source_field = Field(lower=lower, eos_token=eos, init_token=sos,
                                      tokenize=tok_fun, use_vocab=lower, pad_token=pad)
      else:
        raise NotImplementedError
    else:
      log.info("Constructing fields from dataset.")
      if source_format == 'sequence' and target_format == 'sequence':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased') if BERT else str.split
        lower = False if BERT else True
        pad = tokenizer.convert_tokens_to_ids(tokenizer.pad_token) if BERT else '<pad>'
        sos = tokenizer.convert_tokens_to_ids(tokenizer.bos_token) if BERT else '<sos>'
        eos = None if BERT else '<eos>'
        tok_fun = tokenizer.encode if BERT else tokenizer
        self.source_field = Field(lower=lower, eos_token=eos, init_token=sos,
                                    tokenize=tok_fun, use_vocab=lower, pad_token=pad)
        self.target_field = Field(lower=True, eos_token='<eos>', init_token='<sos>') 
      else:
        raise NotImplementedError
    
    # Construct iterators
    self._create_iterators(cfg)

    if fields is None:
      if not BERT:
        self.source_field.build_vocab(*self._in_sample_data)
        pickle.dump(self.source_field, open('source.pt', 'wb'))	
        
      self.target_field.build_vocab(*self._in_sample_data)	
      pickle.dump(self.target_field, open('target.pt', 'wb'))

  def __repr__(self):
    message = "{}(\n".format(self.__class__.__name__)
    for attr in self.__dict__:
      padding = " "
      if '_iterators' in attr:
        message += padding + "splits: ["
        for i, s in enumerate(getattr(self, attr)):
          if i > 0:
            message += ', '
          message += "{} ({} sequences)".format(s, len(getattr(self, attr)[s].dataset) + 1)
        message += ']\n'
        message += padding + 'fields: ['
        for i, k in enumerate(self.datafields):
          if i > 0:
            message += ', '
          message += f"{k[0]}"
        message += ']'
    message += "\n)"
    return message
