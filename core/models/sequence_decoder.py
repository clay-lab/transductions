# sequence_decoder.py
# 
# Provides SequenceDecoder module.

import torch
import torch.nn.functional as F
import random
import logging
from omegaconf import DictConfig
from torchtext.vocab import Vocab

# Library imports
from core.models.model_io import ModelIO

log = logging.getLogger(__name__)

if torch.cuda.is_available():
    avd = torch.device('cuda')
else:
    avd = torch.device('cpu')

class SequenceDecoder(torch.nn.Module):

  @property
  def vocab_size(self):
    return len(self._vocabulary)
  
  @property
  def PAD_IDX(self):
    return self._vocabulary.stoi['<pad>']
  
  @property
  def SOS_IDX(self):
    return self._vocabulary.stoi['<sos>']
  
  @property
  def EOS_IDX(self):
    return self._vocabulary.stoi['<eos>']

  def __init__(self, cfg: DictConfig, vocab: Vocab):
    
    super(SequenceDecoder, self).__init__()

    self._num_layers = cfg.num_layers
    self._hidden_size = cfg.hidden_size
    self._unit_type = cfg.unit.upper()
    self._max_length = cfg.max_length
    self._embedding_size = cfg.embedding_size
    self._attention = cfg.attention.lower()
    if self._attention != 'none':
      self._embedding_size += self._hidden_size
    
    self._vocabulary = vocab
    self._pad_index = self._vocabulary.stoi['pad']
    self._cls_index = self._vocabulary.stoi['cls']

    self._embedding = torch.nn.Embedding(self.vocab_size, self._hidden_size)
    self._dropout = torch.nn.Dropout(p=cfg.dropout)

    if self._num_layers == 1:
      assert cfg.dropout == 0, "Dropout must be zero if num_layers = 1"

    if self._unit_type == "SRN":
      self._unit = torch.nn.RNN(self._embedding_size, self._hidden_size, num_layers=self._num_layers, dropout=cfg.dropout)
    elif self._unit_type == "GRU":
      self._unit = torch.nn.GRU(self._embedding_size, self._hidden_size, num_layers=self._num_layers, dropout=cfg.dropout)
    elif self._unit_type == "LSTM":
      self._unit = torch.nn.LSTM(self._embedding_size, self._hidden_size, num_layers=self._num_layers, dropout=cfg.dropout)
    elif self._unit_type == "TRANSFORMER":
      raise NotImplementedError
    else:
      log.error("Invalid unit type '{}''.".format(self._unit_type))
      raise ValueError("Invalid unit type '{}''.".format(self._unit_type))
  
    self._out = torch.nn.Linear(self._hidden_size, self.vocab_size)

  def forward(self, dec_input: ModelIO, tf_ratio: float) -> ModelIO:
    """
    Computes the forward pass of the decoder.

    Paramters:
      - dec_input: wrapper object for the various inputs to the decoder. This
          allows for variadic parameters to account for various units' different
          input requirements (i.e., LSTMs require a `cell`)
      - tf_ratio (float in range [0.0, 1.0]): chance that teacher_forcing is
          used for a given batch. If tf_ratio is not `None`, a `target` must
          be present in `dec_input`.
    """

    batch_size = dec_input.source.shape[1]
    if hasattr(dec_input, 'target'):
      gen_len = dec_input.target.shape[0]
    else:
      gen_len = self._max_length
    
    teacher_forcing = random.random() < tf_ratio
    if teacher_forcing and not hasattr(dec_input, 'target'):
      log.error("You must provide a 'target' to use teacher forcing.")
      raise SystemError

    """
    Create or extract step inputs from the `dec_input` object.
    - x0: [trans_seq_len × batch_size] 
          The initial input to the decoder. For simple seq-2-seq tasks,
          this is simply the `transformation` token. For more complicated
          tasks, it may be a sequence of transformation tokens.
    - enc_outputs: The hidden states produced at all timesteps from the encoder.
    - h0: The final hidden state of the encoded vectors. We generate this from
          the `enc_outputs` value.
    """
    x0 = dec_input.transform[1:-1] # remove the `<sos>` and `<eos>` tokens
    enc_outputs = dec_input.enc_outputs
    h0 = enc_outputs[-1]

    x, h = x0, h0

    dec_step_input = ModelIO()
    dec_step_input.set_attributes({
      "x": x,
      "h": h,
    })

    # Skeletons for the decoder outputs
    has_finished = torch.zeros(batch_size).to(avd)
    dec_outputs = torch.zeros(gen_len, batch_size, self.vocab_size).to(avd)
    dec_outputs[:,:,self.PAD_IDX] = 1.0
    dec_hiddens = torch.zeros(gen_len, batch_size, self._hidden_size).to(avd)

    for i in range(gen_len):

      # Get forward_step pass
      step_result = self.forward_step(dec_step_input)
      step_prediction = step_result.y.argmax(dim=1)

      # Update outputs
      dec_outputs[i] = step_result.y
      dec_hiddens[i] = step_result.h[-1]

      # Check if we're done
      has_finished[step_prediction == self.EOS_IDX] = 1
      if has_finished.prod() == 1: # TODO: use `bool` tensor here
        break
      else:
        # Otherwise, iterate x, h and repeat
        x = dec_input.target[i] if teacher_forcing else step_prediction
        h = step_result.h

        dec_step_input.set_attributes({
          "x": x,
          "h": h,
        })

    output = ModelIO({
      "dec_outputs" : dec_outputs,
      "dec_hiddens" : dec_hiddens
    })

    return output

  def forward_step(self, step_input: ModelIO) -> ModelIO:

    unit_input = F.relu(self._embedding(step_input.x))
    h = step_input.h

    if len(unit_input.shape) == 2:
      unit_input = unit_input.unsqueeze(0)
    
    if len(h.shape) == 2:
      h = h.unsqueeze(0)

    # print("unit_input:", unit_input.shape)
    # print("h:", h.shape)

    # TODO: Figure out why the original code has unit_input.unsqueeze(0) and
    #       why we have to unsqueeze h....probably changes with an LSTM.....

    _, state = self._unit(unit_input, h)
    y = self._out(state[-1])

    step_result = ModelIO({
      "y" : y,
      "h" : state
    })

    return step_result
