import logging
import random
import torch
import numpy as np
import hydra
import os
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from omegaconf import DictConfig
from typing import Dict, Tuple
from cmd import Cmd
import pickle
from torchtext.data import Batch
from torchtext.vocab import Vocab
import re

# Library imports
from core.models.base_model import TransductionModel
# from core.models.model_io import ModelIO
from core.dataset.base_dataset import TransductionDataset
from core.metrics.base_metric import *
from core.metrics.meter import Meter
from core.early_stopping import EarlyStopping
# from core.tools.trajectory import Trajectory

log = logging.getLogger(__name__)

class Trainer:
  """
  Handles interface between:
    - TransductionModel
    - TransductionDataset
    - Checkpoint
    - Meter
  """

  def __init__(self, cfg: DictConfig):

    self._cfg = cfg
    self._instantiate()

  def _instantiate(self):

    self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info("DEVICE: {}".format(self._device))

    # Check if a random seed should be set
    if 'seed' in self._cfg.experiment.hyperparameters.keys():
      random_seed = int(self._cfg.experiment.hyperparameters.seed)
      np.random.seed(random_seed)
      random.seed(random_seed)
      torch.manual_seed(random_seed)
      torch.cuda.manual_seed(random_seed)
      torch.backends.cudnn.deterministic = True

  def _normalize_lengths(self, output: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Pad the output or target with decoder's <pad> so that loss can be computed.
    """

    diff = int(output.shape[-1] - target.shape[-1])
    pad_idx = self._model._decoder.vocab.stoi['<pad>']

    if diff == 0:
      pass
    elif diff > 0:
      # output is longer than target. Add <pad> index to end 
      # of target until length is equal
      target = F.pad(input=target, pad=(0, diff), value=pad_idx)
    else:
      # target is longer than output. Add one-hot tensor hot in
      # index == pad_idx to end until length is equal
      padding = torch.zeros(output.shape[0], output.shape[1], -diff).to(self._device)
      padding[:, pad_idx] = 1.0
      output = torch.cat((output, padding), dim=2)

    return output.to(self._device), target.to(self._device)
  
  def _load_model(self, cfg: DictConfig, src_vocab: Vocab, tgt_vocab: Vocab, from_path: str = None):

    model = TransductionModel(cfg, src_vocab, tgt_vocab, self._device)
    log.info(model)

    if from_path is not None:
      log.info("Loading model weights from {}".format(from_path))
      model.load_state_dict(torch.load(from_path))
    
    return model

  def _load_dataset(self, cfg: DictConfig, from_paths: Dict = None):

    BERT = cfg.model.encoder.unit == 'BERT'

    if from_paths is not None:
      src_field = pickle.load(open(from_paths['source'], 'rb')) if not BERT else None
      tgt_field = pickle.load(open(from_paths['target'], 'rb'))
      dataset = TransductionDataset(cfg, self._device, {'source': src_field, 'target': tgt_field}, BERT=BERT)
    else:
      dataset = TransductionDataset(cfg, self._device, BERT=BERT)
    
    log.info(dataset)
    return dataset
  
  def _load_checkpoint(self, from_path: str = None):
    """
    Sets up self._model and self._dataset. If given a path, it will attempt to load
    these from disk; otherwise, it will create them from scratch.
    """

    if from_path is not None:
      chkpt_dir = hydra.utils.to_absolute_path(from_path)
      field_paths = {
        'source' : os.path.join(chkpt_dir, 'source.pt'),
        'target' : os.path.join(chkpt_dir, 'target.pt')
      }
      model_path = os.path.join(chkpt_dir, 'model.pt')
      self._dataset = self._load_dataset(self._cfg.experiment, field_paths)
    else:
      model_path = None
      self._dataset = self._load_dataset(self._cfg.experiment)

    src_field = self._dataset.source_field
    tgt_field = self._dataset.target_field
    src_vocab = src_field.vocab if hasattr(src_field, 'vocab') else None
    self._model = self._load_model(self._cfg.experiment, src_vocab, tgt_field.vocab, model_path)

  def train(self):
    
    # Load checkpoint
    self._load_checkpoint()

    log.info("Beginning training")

    lr = self._cfg.experiment.hyperparameters.lr
    tf_ratio = self._cfg.experiment.hyperparameters.tf_ratio
    if not tf_ratio:
      # TODO: Should probably come up with better logic to handle the 'null' case
      tf_ratio = 0.0
    epochs = self._cfg.experiment.hyperparameters.epochs
    optimizer = torch.optim.SGD(self._model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=None, ignore_index=self._model._decoder.PAD_IDX)

    early_stoping = EarlyStopping(self._cfg.experiment.hyperparameters)

    # Metrics
    seq_acc = SequenceAccuracy()
    tok_acc = TokenAccuracy(self._dataset.target_field.vocab.stoi['<pad>'])
    len_acc = LengthAccuracy(self._dataset.target_field.vocab.stoi['<pad>'])
    first_acc = NthTokenAccuracy(n=1)
    avg_loss = LossMetric(F.cross_entropy)
    
    meter = Meter([seq_acc, tok_acc, len_acc, first_acc, avg_loss])

    for epoch in range(epochs):

      log.info("EPOCH %i / %i", epoch + 1, epochs)

      log.info("Computing metrics for 'train' dataset")
      self._model.train()

      with tqdm(self._dataset.iterators['train']) as T:
        for batch in T:

          optimizer.zero_grad()

          # Loss expects:
          #   output:  [batch_size, classes, seq_len]
          #   target: [batch_size, seq_len]
          output = self._model(batch, tf_ratio=tf_ratio).permute(1, 2, 0)
          target = batch.target.permute(1, 0)
          output, target = self._normalize_lengths(output, target)

          # TODO: Loss should ignore <pad>, ...maybe others?
          loss = criterion(output, target)

          loss.backward()
          optimizer.step()

          # Compute metrics
          meter(output, target)

          T.set_postfix(trn_loss='{:4.3f}'.format(loss.item()))
        
        meter.log(stage='train', step=epoch)
        meter.reset()
      
      # Perform val, test, gen, ... passes
      with torch.no_grad():

        log.info("Computing metrics for 'val' dataset")
        with tqdm(self._dataset.iterators['val']) as V:
          val_loss = 0.0
          for batch in V:

            output = self._model(batch).permute(1, 2, 0)
            target = batch.target.permute(1, 0)
            output, target = self._normalize_lengths(output, target)

            meter(output, target)

            # Compute average validation loss
            val_loss += F.cross_entropy(output, target) / len(batch)
            V.set_postfix(val_loss='{:4.3f}'.format(val_loss.item()))

          meter.log(stage='val', step=epoch)
          meter.reset()

        log.info("Computing metrics for 'test' dataset")
        with tqdm(self._dataset.iterators['test']) as T:
          for batch in T:
            
            output = self._model(batch).permute(1, 2, 0)
            target = batch.target.permute(1, 0)
            output, target = self._normalize_lengths(output, target)

            meter(output, target)

          meter.log(stage='test', step=epoch)
          meter.reset()

        if 'gen' in list(self._dataset.iterators.keys()):
          log.info("Computing metrics for 'gen' dataset")
          with tqdm(self._dataset.iterators['gen']) as G:
            for batch in G:

              output = self._model(batch).permute(1, 2, 0)
              target = batch.target.permute(1, 0)
              output, target = self._normalize_lengths(output, target)

              meter(output, target)

            meter.log(stage='gen', step=epoch)
            meter.reset()
        
        other_iters = list(self._dataset.iterators.keys())
        other_iters = [o for o in other_iters if o not in ['train', 'val', 'test', 'gen']]
        for itr in other_iters:
          log.info("Computing metrics for '{}' dataset".format(itr))
          with tqdm(self._dataset.iterators[itr]) as I:
            for batch in I:

              output = self._model(batch).permute(1, 2, 0)
              target = batch.target.permute(1, 0)
              output, target = self._normalize_lengths(output, target)

              meter(output, target)

            meter.log(stage=itr, step=epoch)
            meter.reset()

      should_stop, should_save = early_stoping(val_loss)
      if should_save:
        torch.save(self._model.state_dict(), 'model.pt')
      if should_stop:
        break

  def eval(self, eval_cfg: DictConfig):

    # Load checkpoint data
    self._load_checkpoint(eval_cfg.checkpoint_dir)

    # Create meter
    seq_acc = SequenceAccuracy()
    tok_acc = TokenAccuracy(self._dataset.target_field.vocab.stoi['<pad>'])
    len_acc = LengthAccuracy(self._dataset.target_field.vocab.stoi['<pad>'])
    first_acc = NthTokenAccuracy(n=1)
    avg_loss = LossMetric(F.cross_entropy)

    meter = Meter([seq_acc, tok_acc, len_acc, first_acc, avg_loss])

    log.info("Beginning evaluation")
    self._model.eval()

    for key in list(self._dataset.iterators.keys()):

      log.info('Evaluating model on {} dataset'.format(key))

      with open('{}.tsv'.format(key), 'a') as f:
        f.write('source\ttarget\tprediction\n')

        with tqdm(self._dataset.iterators[key]) as T:
          for batch in T:

            source = batch.source.permute(1, 0)
            prediction = self._model(batch).permute(1, 2, 0)
            target = batch.target.permute(1, 0)
            prediction, target = self._normalize_lengths(prediction, target)

            # TODO: SHOULD WE USE normed ouputs instead of prediction ehre?
            meter(prediction, target)

            # src_toks = self._dataset.id_to_token(source, 'source')
            pred_toks = self._dataset.id_to_token(prediction.argmax(1), 'target')
            tgt_toks = self._dataset.id_to_token(target, 'target')

            for seq in range(len(tgt_toks)):
              # src_line = ' '.join(src_toks[seq])
              src_line = 'SRC'
              tgt_line = ' '.join(tgt_toks[seq])
              pred_line = ' '.join(pred_toks[seq])

              f.write('{}\t{}\t{}\n'.format(src_line, tgt_line, pred_line))
      
        meter.log(stage=key, step=0)
        meter.reset()

  def tpdr(self, tpdr_cfg: DictConfig):

    # Load checkpoint data
    self._load_checkpoint(tpdr_cfg.checkpoint_dir)

    log.info("Beginning TPDR REPL")

    repl = ModelArithmeticREPL(self._model, self._dataset)
    repl.cmdloop()

  def repl(self, repl_cfg: DictConfig):

    # Load checkpoint data
    self._load_checkpoint(repl_cfg.checkpoint_dir)

    # Create PCA Trajectory plotter
    # trajectory = Trajectory()
    # self._model.traj = trajectory

    log.info("Beginning REPL")

    repl = ModelREPL(self._model, self._dataset)
    repl.cmdloop()


class ModelREPL(Cmd):
  """
  A REPL for interacting with a pre-trained model.
  """

  prompt = '> '

  def __init__(self, model: TransductionModel, dataset: TransductionDataset):
    super(ModelREPL, self).__init__()
    self.intro = 'Enter sequences into the model for evaluation.'
    self._model = model
    self._dataset = dataset
  
  def batchify(self, args):
    """
    Turn the REPL input into a batch for the model to process.
    """

    transf, source = args.split(' ', 1)

    source = source.split(' ')
    source.append('<eos>')
    source.insert(0, '<sos>')
    source = [[self._dataset.source_field.vocab.stoi[s]] for s in source]
    source = torch.LongTensor(source)

    transf = ['<sos>', transf, '<eos>']
    transf = [[self._dataset.transform_field.vocab.stoi[t]] for t in transf]
    transf = torch.LongTensor(transf)

    batch = Batch()
    batch.source = source
    batch.annotation = transf

    return batch

  def default(self, args):

    batch = self.batchify(args)

    prediction = self._model(batch, plot_trajectories=True).permute(1, 2, 0).argmax(1)
    prediction = self._dataset.id_to_token(prediction, 'target')[0]
    prediction = ' '.join(prediction)

    source = self._dataset.id_to_token(batch.source.permute(1, 0), 'source')[0]
    source = ' '.join(source)

    transformation = self._dataset.id_to_token(batch.annotation.permute(1, 0), 'source')[0]
    transformation = ' '.join(transformation)

    result = "{} → {} → {}".format(source, transformation, prediction)
    log.info(result)

  
  def do_quit(self, args):
    log.info("Exiting REPL.")
    raise SystemExit

class ModelArithmeticREPL(ModelREPL):
  """
  A REPL for doing expression math
  """

  prompt = 'λ '

  def default(self, args):

    expr_list = re.split("\[|\]", args)
    expr_list = list(filter(None, [e.strip() for e in expr_list]))

    transform = expr_list[0]
    unbatched_expressions = expr_list[1:]

    expressions = []

    for e in unbatched_expressions:
      if e in "+-":
        expressions.append(e)
      else:
        batch = self.batchify(f"{transform} {e}")
        expressions.append(batch)
    
    prediction = self._model.forward_expression(expressions).permute(1, 2, 0).argmax(1)
    prediction = self._dataset.id_to_token(prediction, 'target')[0]
    prediction = ' '.join(prediction)

    source = ' '.join(args.split(' ')[1:])

    result = "{} = {}".format(source, prediction)
    log.info(result)
  
  def do_quit(self, args):
    log.info("Exiting REPL.")
    raise SystemExit