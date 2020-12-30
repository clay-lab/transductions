import logging
import torch
import hydra
import os
import torch.nn as nn
from tqdm import tqdm
from omegaconf import DictConfig
from cmd import Cmd
from torchtext.data import Batch
import numpy as np

# Library imports
from core.models.TransductionModel import TransductionModel
from core.dataset.TransductionDataset import TransductionDataset

log = logging.getLogger(__name__)

class Trainer:
  """
  Handles interface between:
    - TransductionModel
    - Dataset
    - Checkpoint?
    - Visualizer?
  """

  def __init__(self, cfg: DictConfig):

    self._cfg = cfg
    self._instantiate()

  def _instantiate(self):

    self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info("DEVICE: {}".format(self._device))

    self._dataset = TransductionDataset(self._cfg, self._device)
    log.info(self._dataset)

    self._model = TransductionModel(self._cfg, self._dataset, self._device)
    log.info(self._model)
  
  def train(self):

    log.info("Beginning training")

    lr = self._cfg.hyperparameters.lr
    optimizer = torch.optim.SGD(self._model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=None)

    for epoch in range(self._cfg.hyperparameters.epochs):

      log.info("EPOCH %i / %i", epoch + 1, self._cfg.hyperparameters.epochs)

      self._model.train()
      with tqdm(self._dataset.iterators['train']) as T:
        for batch in T:

          optimizer.zero_grad()

          # Loss expects:
          #   output:  [batch_size, classes, seq_len]
          #   target: [batch_size, seq_len]
          output = self._model(batch).permute(1, 2, 0)
          target = batch.target.permute(1, 0)

          loss = criterion(output, target)

          loss.backward()

          optimizer.step()

          T.set_postfix(trn_loss=loss.item())
      
      torch.save(self._model.state_dict(), 'model.pt')

  def eval(self, eval_cfg: DictConfig):

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

            print(batch.source)
            raise SystemExit

            source = batch.source.permute(1, 0)
            prediction = self._model(batch).permute(1, 2, 0)
            target = batch.target.permute(1, 0)

            src_toks = self._dataset.id_to_token(source, 'source')
            pred_toks = self._dataset.id_to_token(prediction.argmax(1), 'target')
            tgt_toks = self._dataset.id_to_token(target, 'target')

            for seq in range(len(src_toks)):
              src_line = ' '.join(src_toks[seq])
              tgt_line = ' '.join(tgt_toks[seq])
              pred_line = ' '.join(pred_toks[seq])

              f.write('{}\t{}\t{}\n'.format(src_line, tgt_line, pred_line))

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

  def default(self, args):

    transf, source = args.split(' ', 1)
    
    source_txt = source
    transf_txt = transf

    source = source.split(' ')
    source.append('<eos>')
    transf = [transf, '<eos>']

    source = [[self._dataset.source_field.vocab.stoi[s]] for s in source]
    source = torch.LongTensor(source)

    transf = [[self._dataset.transform_field.vocab.stoi[t]] for t in transf]
    transf = torch.LongTensor(transf)

    zrs = [[0] for i in range(self._model.max_len)]
    target = torch.LongTensor(zrs)

    batch = Batch()
    batch.source = source
    batch.annotation = transf
    batch.target = target

    prediction = self._model(batch).permute(1, 2, 0).argmax(1)
    prediction = self._dataset.id_to_token(prediction, 'target').flatten()

    result = "{} → {} → {}".format(source_txt, transf_txt, ' '.join(prediction))
    log.info(result)
  
  def do_quit(self, args):
    log.info("Exiting REPL.")
    raise SystemExit