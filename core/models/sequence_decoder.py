# sequence_decoder.py
# 
# Provides SequenceDecoder module.

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import random
import logging
from omegaconf import DictConfig
from torchtext.vocab import Vocab
from abc import abstractmethod

# Library imports
from core.models.model_io import ModelIO
from core.models.positional_encoding import PositionalEncoding
from core.models.attention import create_mask, MultiplicativeAttention, AdditiveAttention, DotProductAttention

log = logging.getLogger(__name__)

if torch.cuda.is_available():
    avd = torch.device('cuda')
else:
    avd = torch.device('cpu')

class SequenceDecoder(torch.nn.Module):

  @staticmethod
  def newDecoder(src_vocab: Vocab, tgt_vocab: Vocab, dec_cfg: DictConfig, enc_cfg: DictConfig = None):

    unit_type = dec_cfg.unit.upper()

    if unit_type == 'SRN':
      return SRNSequenceDecoder(src_vocab, tgt_vocab, dec_cfg, enc_cfg)
    elif unit_type == 'GRU':
      return GRUSequenceDecoder(src_vocab, tgt_vocab, dec_cfg, enc_cfg)
    elif unit_type == 'LSTM':
      return LSTMSequenceDecoder(src_vocab, tgt_vocab, dec_cfg, enc_cfg)
    elif unit_type == 'TRANSFORMER':
      return TransformerSequenceDecoder(tgt_vocab, dec_cfg)
    else:
      log.error(f"Unknown decoder type '{unit_type}'.")

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
  
  @property
  def attention_type(self):
    if self._attention_type is not None:
      return self._attention_type.upper()
    else:
      return self._attention_type

  def __init__(self, src_vocab: Vocab, tgt_vocab: Vocab, dec_cfg: DictConfig, enc_cfg: DictConfig):
    
    super(SequenceDecoder, self).__init__()

    self._num_layers = dec_cfg.num_layers
    self._hidden_size = dec_cfg.hidden_size
    self._unit_type = dec_cfg.unit.upper()
    self._max_length = dec_cfg.max_length
    self._embedding_size = dec_cfg.embedding_size
    self._dropout_p = dec_cfg.dropout
    self._attention_type = dec_cfg.attention

    if self.attention_type:
      self.unit_input_size = self._embedding_size + self._hidden_size
    else:
      self.unit_input_size = self._embedding_size
    
    self._vocabulary = tgt_vocab
    self._src_vocab = src_vocab

    self._embedding = torch.nn.Embedding(self.vocab_size, self._embedding_size)

    if self._num_layers == 1:
      assert self._dropout_p == 0, "Dropout must be zero if num_layers = 1"
    
    self._unit = nn.Module()
  
    self._out = torch.nn.Linear(self._hidden_size, self.vocab_size)

    # Attention
    if self.attention_type is not None:
      if self.attention_type == 'MULTIPLICATIVE':
        self._attention = MultiplicativeAttention(self._hidden_size, enc_cfg.hidden_size)
      elif self.attention_type == 'ADDITIVE':
        self._attention = AdditiveAttention(self._hidden_size, enc_cfg.hidden_size)
      elif self.attention_type == 'DOTPRODUCT':
        self._attention = DotProductAttention()
      else:
        raise NotImplementedError

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

    seq_len, batch_size, _ = dec_input.enc_outputs.shape

    teacher_forcing = random.random() < tf_ratio
    if teacher_forcing and not hasattr(dec_input, 'target'):
      log.error("You must provide a 'target' to use teacher forcing.")
      raise SystemError

    # Okay so we still need this, but should we be padding
    # the outputs or something when not using teacher forcing?
    if hasattr(dec_input, 'target'):
      gen_len = dec_input.target.shape[0]
    else:
      gen_len = self._max_length
  
    # Get input to decoder unit
    dec_step_input = self._get_step_inputs(dec_input)

    # Skeletons for the decoder outputs
    has_finished = torch.zeros(batch_size, dtype=torch.bool).to(avd)
    dec_outputs = torch.zeros(gen_len, batch_size, self.vocab_size).to(avd)
    dec_outputs[:,:,self.PAD_IDX] = 1.0
    dec_hiddens = torch.zeros(gen_len, batch_size, self._hidden_size).to(avd)
    
    # Attention
    if self.attention_type is not None:
      attention = torch.zeros(gen_len, batch_size, seq_len).to(avd)
      src_mask = create_mask(dec_input.source, self._src_vocab)
    else:
      src_mask = None

    for i in range(gen_len):

      # Get forward_step pass
      step_result = self.forward_step(dec_step_input, src_mask)
      step_prediction = step_result.y.argmax(dim=1)

      # Update results
      dec_outputs[i] = step_result.y
      dec_hiddens[i] = step_result.h[-1]
      if self.attention_type is not None:
        attention[i] = step_result.attn

      # Check if we're done
      has_finished[step_prediction == self.EOS_IDX] = True
      if all(has_finished): 
        break
      else:
        # Otherwise, iterate x, h and repeat
        x = dec_input.target[i] if teacher_forcing else step_prediction

        # A little hacky, but we want to use every value from the step result
        # For the next step EXCEPT x, which should come from either the target
        # or the step_prediction.
        step_result.set_attribute('x', x)
        step_result.set_attribute('enc_outputs', dec_input.enc_outputs)
        dec_step_input = self._get_step_inputs(step_result)

    output = ModelIO({
      "dec_outputs" : dec_outputs,
      "dec_hiddens" : dec_hiddens
    })

    if self.attention_type is not None:
      output.set_attribute("attention", attention)

    return output
  
  def compute_attention(self, unit_input: Tensor, enc_outputs: Tensor, h: Tensor, src_mask: Tensor) -> Tensor:

    attn = self._attention(enc_outputs, h[-1], src_mask).unsqueeze(1)
    enc_out = enc_outputs.permute(1,0,2)
    weighted_enc_out = torch.bmm(attn, enc_out).permute(1,0,2)
    return torch.cat((unit_input, weighted_enc_out), dim=2), attn.squeeze(1)

  def forward_step(self, step_input: ModelIO, src_mask: Tensor = None) -> ModelIO:

    h = step_input.h
    h = h.unsqueeze(0) if len(h.shape) == 2 else h
    unit_input = F.relu(self._embedding(step_input.x))
    unit_input = unit_input.unsqueeze(0) if len(unit_input.shape) == 2 else unit_input
    if src_mask is not None:
      unit_input, attn = self.compute_attention(unit_input, step_input.enc_outputs, h, src_mask)

    # print("unit_input", unit_input.shape)
    # print("attn", attn.shape)

    _, state = self._unit(unit_input, h)
    y = self._out(state[-1])

    step_result = ModelIO({ "y" : y, "h" : state })
    if src_mask is not None:
      step_result.set_attribute("attn", attn)

    return step_result

  def _get_step_inputs(self, dec_inputs: ModelIO) -> ModelIO:

    if hasattr(dec_inputs, 'h'):
      h = dec_inputs.h
    elif hasattr(dec_inputs, 'enc_hidden'):
      h = dec_inputs.enc_hidden
    else:
      h = dec_inputs.enc_outputs[-1]
      # log.error(f"I don't have any hidden state to use for the step from {dec_inputs}.")
      # raise SystemError

    batch_size = h.shape[0]

    if hasattr(dec_inputs, 'x'):
      # Not the first step. Use outputs from previous step instead
      x = dec_inputs.x
    elif hasattr(dec_inputs, 'transform'):
      # Use the transformation token from the input
      x = dec_inputs.transform[1:-1] # strip <sos> and <eos> tokens
    else:
      log.error(f"I don't have any input to use for the step from {dec_inputs}.")
      raise SystemError

    dec_step_input = ModelIO({
      "x": x,
      "h": h
    })

    if hasattr(dec_inputs, 'enc_outputs'):
      dec_step_input.set_attribute('enc_outputs', dec_inputs.enc_outputs)

    return dec_step_input

class LSTMSequenceDecoder(SequenceDecoder):

  def __init__(self, src_vocab: Vocab, tgt_vocab: Vocab, dec_cfg: DictConfig, enc_cfg: DictConfig):
    super(LSTMSequenceDecoder, self).__init__(src_vocab, tgt_vocab, dec_cfg, enc_cfg)
    self._unit = nn.LSTM(self.unit_input_size, self._hidden_size, num_layers=self._num_layers, dropout=dec_cfg.dropout)

  def _get_step_inputs(self, dec_inputs: ModelIO) -> ModelIO:

    # TODO: This is hacky....make the logic nicer here.
    if hasattr(dec_inputs, 'enc_hidden') and isinstance(dec_inputs.enc_hidden, tuple):
      dec_inputs.c = dec_inputs.enc_hidden[1]
      dec_inputs.enc_hidden = dec_inputs.enc_hidden[0]

    # Get default implementation
    step_input = super()._get_step_inputs(dec_inputs)
    batch_size = step_input.h.shape[0]

    if hasattr(dec_inputs, 'c'):
      # We're @ timestep t>0, and the decoder has already produced a 'c'
      step_input.set_attribute('c', dec_inputs.c)
    else:
      # We're @ timestep t=0, so create an initial 'c' of all zeros.
      c = torch.zeros(self._num_layers, batch_size, self._hidden_size).to(avd)
      step_input.set_attribute('c', c)

    return step_input

  def forward_step(self, step_input: ModelIO, src_mask: Tensor = None) -> ModelIO:

    unit_input = F.relu(self._embedding(step_input.x))
    h, c = step_input.h, step_input.c

    if len(unit_input.shape) == 2:
      unit_input = unit_input.unsqueeze(0)
    
    if len(h.shape) == 2:
      h = h.unsqueeze(0)
    
    hidden = (h, c)

    if src_mask is not None:
      unit_input, attn = self.compute_attention(unit_input, step_input.enc_outputs, h, src_mask)

    _, state = self._unit(unit_input, hidden)
    y = self._out(state[0][-1])

    step_result = ModelIO({ "y" : y, "h" : state[0], "c" : state[1] })
    if src_mask is not None:
      step_result.set_attribute("attn", attn)

    return step_result
    
class SRNSequenceDecoder(SequenceDecoder):

  def __init__(self, src_vocab: Vocab, tgt_vocab: Vocab, dec_cfg: DictConfig, enc_cfg: DictConfig):
    super(SRNSequenceDecoder, self).__init__(src_vocab, tgt_vocab, dec_cfg, enc_cfg)
    print(self._num_layers)
    self._unit = nn.RNN(self.unit_input_size, self._hidden_size, num_layers=self._num_layers, dropout=dec_cfg.dropout)

class GRUSequenceDecoder(SequenceDecoder):

  def __init__(self, src_vocab: Vocab, tgt_vocab: Vocab, dec_cfg: DictConfig, enc_cfg: DictConfig):
    super(GRUSequenceDecoder, self).__init__(src_vocab, tgt_vocab, dec_cfg, enc_cfg)
    self._unit = nn.GRU(self.unit_input_size, self._hidden_size, num_layers=self._num_layers, dropout=dec_cfg.dropout)

class TransformerSequenceDecoder(nn.Module):

  @property
  def vocab_size(self):
    return len(self._vocabulary)
  
  @property
  def EOS_IDX(self):
    return self._vocabulary.stoi['<eos>']
  
  @property
  def PAD_IDX(self):
    return self._vocabulary.stoi['<pad>']

  def __init__(self, vocab: Vocab, cfg: DictConfig):
    
    super().__init__()
    
    # Parameters
    self._num_layers = cfg.num_layers
    self._hidden_size = cfg.hidden_size
    self._unit_type = cfg.unit.upper()
    self._max_length = cfg.max_length
    self._embedding_size = cfg.embedding_size
    self._dropout_p = cfg.dropout
    self._num_heads = cfg.num_heads
    self._vocabulary = vocab

    # Model layers
    self._embedding = nn.Sequential(
      torch.nn.Embedding(self.vocab_size, self._hidden_size),
      PositionalEncoding(self._hidden_size, self._dropout_p, self._max_length)
    )
    layer = nn.TransformerDecoderLayer(d_model=self._hidden_size, nhead=self._num_heads, dropout=self._dropout_p)
    self._unit = nn.TransformerDecoder(layer, num_layers=self._num_layers)
    self._out = torch.nn.Linear(self._hidden_size, self.vocab_size)
    
  def _generate_square_subsequent_mask(self, sz: int) -> Tensor:
    """
    Taken from PyTorch's implementation:
    https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer.generate_square_subsequent_mask
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
  
  def forward(self, dec_input: ModelIO, tf_ratio: float) -> ModelIO:
    """
    Try and keep the same signature as SequenceDecoder.
    """

    seq_len, batch_size, _ = dec_input.enc_outputs.shape

    teacher_forcing = random.random() < tf_ratio
    if teacher_forcing and not hasattr(dec_input, 'target'):
      log.error("You must provide a 'target' to use teacher forcing.")
      raise SystemError

    if hasattr(dec_input, 'target'):
      gen_len = dec_input.target.shape[0]
    else:
      gen_len = self._max_length

    # tgt = inputs to the decoder, starting with TRANS token and appending
    #       dec_input.target[i] or predicted token
    # mem = encoder outputs, used for multi-headed attention
    tgt = dec_input.transform[1:-1] # strip <sos> and <eos> tokens
    mem = dec_input.enc_outputs

    has_finished = torch.zeros(batch_size, dtype=torch.bool).to(avd)

    for i in range(gen_len):

      # Re-embed every time since we need positional encoding to take into
      # account the new tokens in context of the old ones.
      tgt_emb = self._embedding(tgt)

      # Ensures that once a model outputs token <t> @ position i, it will
      # always output <t> @ i even for further timesteps
      tgt_mask = self._generate_square_subsequent_mask(tgt.shape[0]).to(avd)
      
      # Calculate the next predicted token from output
      out = self._out(self._unit(tgt=tgt_emb, memory=mem, tgt_mask=tgt_mask))
      predicted = out[-1].argmax(dim=1)
      
      has_finished[predicted == self.EOS_IDX] = True
      if all(has_finished): 
        break
      else:
        new_tgt = dec_input.target[i] if teacher_forcing else predicted
        tgt = torch.cat((tgt, new_tgt.unsqueeze(0)), dim=0).to(avd)

    return ModelIO({"dec_outputs": out})
  