import logging
import torch
import hydra
import os
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from omegaconf import DictConfig
from cmd import Cmd
from torchtext.data import Batch

# Library imports
from core.models.base_model import TransductionModel
from core.models.model_io import ModelIO
from core.dataset.base_dataset import TransductionDataset
from core.metrics.base_metric import TokenAccuracy, LossMetric, LengthAccuracy
from core.metrics.meter import Meter
from core.early_stopping import EarlyStopping

log = logging.getLogger(__name__)

class Trainer:
  """
  Handles interface between:
    - TransductionModel
    - TransductionDataset
    - Meter
  """

  def __init__(self, cfg: DictConfig):

    print(cfg)

    self._cfg = cfg
    self._instantiate()

  def _instantiate(self):

    self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info("DEVICE: {}".format(self._device))

    self._dataset = TransductionDataset(self._cfg.experiment, self._device)
    log.info(self._dataset)

    self._model = TransductionModel(self._cfg.experiment, self._dataset, self._device)
    log.info(self._model)
  
  def train(self):

    log.info("Beginning training")

    lr = self._cfg.experiment.hyperparameters.lr
    epochs = self._cfg.experiment.hyperparameters.epochs
    optimizer = torch.optim.SGD(self._model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=None)

    early_stoping = EarlyStopping(self._cfg.experiment.hyperparameters)

    # Metrics
    token_acc = TokenAccuracy(self._dataset.target_field.vocab.stoi['<pad>'])
    length_acc = LengthAccuracy(self._dataset.target_field.vocab.stoi['<pad>'])
    avg_loss = LossMetric(F.cross_entropy)
    meter = Meter([token_acc, avg_loss])

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
          output = self._model(batch, tf_ratio=0.5).permute(1, 2, 0)
          target = batch.target.permute(1, 0)

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
          for batch in V:

            output = self._model(batch).permute(1, 2, 0)
            target = batch.target.permute(1, 0)

            meter(output, target)

            # Compute validation loss
            val_loss = F.cross_entropy(output, target)
            T.set_postfix(val_loss='{:4.3f}'.format(val_loss.item()))

          meter.log(stage='val', step=epoch)
          meter.reset()

        log.info("Computing metrics for 'test' dataset")
        with tqdm(self._dataset.iterators['test']) as T:
          for batch in T:
            
            output = self._model(batch).permute(1, 2, 0)
            target = batch.target.permute(1, 0)

            meter(output, target)

          meter.log(stage='test', step=epoch)
          meter.reset()

        if 'gen' in list(self._dataset.iterators.keys()):
          log.info("Computing metrics for 'gen' dataset")
          with tqdm(self._dataset.iterators['gen']) as G:
            for batch in G:

              output = self._model(batch).permute(1, 2, 0)
              target = batch.target.permute(1, 0)

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

              meter(output, target)

            meter.log(stage=itr, step=epoch)
            meter.reset()

      should_stop, should_save = early_stoping(val_loss)
      if should_save:
        torch.save(self._model.state_dict(), 'model.pt')
      if should_stop:
        break

  def eval(self, eval_cfg: DictConfig):

    # Create meter
    # Metrics
    token_acc = TokenAccuracy(self._dataset.target_field.vocab.stoi['<pad>'])
    meter = Meter([token_acc])

    # Load the pre-trained model weights
    chkpt_dir = hydra.utils.to_absolute_path(eval_cfg.checkpoint_dir)
    model_path = os.path.join(chkpt_dir, 'model.pt')

    log.info("Loading model weights from {}".format(model_path))
    self._model.load_state_dict(torch.load(model_path))

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

            meter(prediction, target)

            src_toks = self._dataset.id_to_token(source, 'source')
            pred_toks = self._dataset.id_to_token(prediction.argmax(1), 'target')
            tgt_toks = self._dataset.id_to_token(target, 'target')

            for seq in range(len(src_toks)):
              src_line = ' '.join(src_toks[seq])
              tgt_line = ' '.join(tgt_toks[seq])
              pred_line = ' '.join(pred_toks[seq])

              f.write('{}\t{}\t{}\n'.format(src_line, tgt_line, pred_line))
      
        meter.log(stage=key, step=0)
        meter.reset()

  def repl(self, eval_cfg: DictConfig):

    # Load the pre-trained model weights
    chkpt_dir = hydra.utils.to_absolute_path(eval_cfg.checkpoint_dir)
    model_path = os.path.join(chkpt_dir, 'model.pt')

    log.info("Loading model weights from {}".format(model_path))
    self._model.load_state_dict(torch.load(model_path))

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

    prediction = self._model(batch).permute(1, 2, 0).argmax(1)
    prediction = self._dataset.id_to_token(prediction, 'target')[0]
    prediction = ' '.join(prediction)

    source = self._dataset.id_to_token(batch.source.permute(1, 0), 'source')[0]
    source = ' '.join(source)

    result = "{} → {}".format(source, prediction)
    log.info(result)
  
  def do_quit(self, args):
    log.info("Exiting REPL.")
    raise SystemExit