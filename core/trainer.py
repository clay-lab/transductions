import logging
import torch
import hydra
import os
import torch.nn as nn
from tqdm import tqdm
from omegaconf import DictConfig
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

    lr = self._cfg.training.lr
    optimizer = torch.optim.SGD(self._model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=None)

    for epoch in range(self._cfg.training.epochs):

      log.info("EPOCH %i / %i", epoch + 1, self._cfg.training.epochs)

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
    self._model.eval()

    log.info("Beginning evaluation")
    with tqdm(self._dataset.iterators['test']) as T:

      with open('test.tsv', 'a') as f:

        f.write('source\ttarget\tprediction\n')

        for batch in T:

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


