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
from core.models.sequence_decoder import SequenceDecoder, BetterSequenceDecoder
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
      self._encoder = SequenceEncoder(encoder_cfg, src_vocab)
    else:
      raise NotImplementedError
    
    if cfg.dataset.target_format == 'sequence':
      # self._decoder = BetterSequenceDecoder(
      #   src_vocab=src_vocab, 
      #   tgt_vocab=tgt_vocab, 
      #   dec_cfg=decoder_cfg, 
      #   enc_cfg=encoder_cfg
      # )
      self._decoder = SequenceDecoder.newDecoder(src_vocab, tgt_vocab, decoder_cfg, encoder_cfg)
    else:
      raise NotImplementedError
    
    self._encoder.to(self.device)
    self._decoder.to(self.device)
  
  def forward_expression_eos(self, expressions):
    """
    Same as `forward_expression`, but this performs arithmetic before the
    <eos> token.

    A                        | B
    <sos> alice sees herself | <eos> -
    <sos> alice meets claire | <eos> +
    <sos> grace meets claire | <eos>

    A -> encoder -> (outputs, hidden) * 3

    (<eos>, hidden) -> encoder -> (output_e, hidden_e)

    ([outputs, output_e], hidden_e) -> decoder

    """

    representations = []
    sources = []

    # Compute forward pass on each sub-expression
    for term in expressions:
      if isinstance(term, str):
        representations.append(term)
      else:
        enc_input = ModelIO({"source" : term.source[0:-1,:]})
        sources.append(term.source)
        enc_output = self._encoder(enc_input)
        representations.append(enc_output)
    
    # Perform arithmetic operation on reps
    buffer = {
      "enc_hidden" : None,
      "enc_outputs" : None
    }
    should_operate = False
    for r in representations:
      if isinstance(r, str):
        should_operate = torch.add if r == "+" else torch.subtract
      else:
        if not should_operate:
          for key in buffer:
            try:
              buffer[key] = getattr(r, key)
            except:
              buffer[key] = None
        else:
          for key in buffer:
            try:
              buffer[key] = should_operate(buffer[key], getattr(r, key))
            except:
              buffer[key] = None

          should_operate = False
    
    # Run final foward pass on [<eos>] with the previously computed hidden state
    # as input. We concatenate the enc_outputs to the previous enc_outputs, and
    # replace the previous enc_hidden with this enc_hidden.
    eos_token = expressions[0].source[-1,:].unsqueeze(1)
    eos_input = ModelIO({"source" : eos_token})
    test_out = self._encoder.forward_with_hidden(eos_input, hidden=buffer['enc_hidden'])

    # Concatenate the previously-computed outputs with output of <eos> encoding
    new_outputs = torch.cat((buffer['enc_outputs'], test_out.enc_outputs))    
    buffer['enc_outputs'] = new_outputs
    buffer['enc_hidden'] = test_out.enc_hidden

    # Ensure that the dimensions match
    assert buffer['enc_outputs'].shape[0:2] == expressions[0].source.shape, "Computed outputs don't match the length of the input"
    assert buffer['enc_hidden'].shape[0] == 1, "Hidden state should be of length 1"

    dec_inputs = ModelIO({
      "source" : sources[0],
      "transform" : expressions[0].annotation
    })

    for key in buffer:
      if buffer[key] is not None:
        dec_inputs.set_attribute(key, buffer[key])

    dec_output = self._decoder(dec_inputs, tf_ratio = 0.0)
    return dec_output.dec_outputs
  
  def forward_batch_expr(self, batch: Batch):
    """
    Splits each entry in a batch into sub-expressions, encodes them, and
    performs arithmetic on the encodings.
    """

    # print("Mock Batch")
    # mock_in = ModelIO({"source" : batch.source})
    # mock_out = self._encoder(mock_in)
    # print(mock_out)

    reduced_encodings = []

    for exp in batch.source.permute(1,0):

      # Split a single expression at the indices where there is an <unk>
      # token, corresponding to one of the operators ("+", "-") which wasn't
      # present in the training vocabulary.
      operator_indices = ((exp==0).nonzero(as_tuple=True)[0])
      sub_exps = list(torch.tensor_split(exp, operator_indices))

      # We need to pad each sub-expression with an <sos> and <eos> token, if
      # They don't already have one.
      sos_tok = sub_exps[0][0]
      eos_tok = sub_exps[-1][-1]

      # The general format for sub-expressions here is:
      #   0th: <sos> .....
      #   1st--peultimate: <unk> .....
      #   last: <unk> ..... <eos>
      # So, we need to (a) append an <eos> token to the 0th sub-exp,
      # (b) switch <unk> to <sos> and append an <eos> token to the first--through 
      # penultimate sub-exps, and (c) switch <unk> to <sos> for the final sub-exp

      for i in range(len(sub_exps[0:-1])):
        # Append <eos> to 0...penultimate
        sub_exps[i] = torch.cat((sub_exps[i], torch.tensor([eos_tok])))
      
      for i in range(len(sub_exps[1:])):
        # Change <unk> to <sos> in 1...last
        sub_exps[i+1][0] = sos_tok
      
      encodings = []
      for s in sub_exps:
        enc_input = ModelIO({"source" : s.unsqueeze(0).permute(1,0)})
        enc_output = self._encoder(enc_input)
        encodings.append(enc_output)
      
      results = {}
      for key in encodings[0].__dict__.keys():
        if type(getattr(encodings[0], key)) is tuple:
          a = getattr(encodings[0], key)[0] - getattr(encodings[1], key)[0] + getattr(encodings[2], key)[0]
          b = getattr(encodings[0], key)[1] - getattr(encodings[1], key)[0] + getattr(encodings[2], key)[1]
          results[key] = (a, b)
        else:
          results[key] = getattr(encodings[0], key) - getattr(encodings[1], key) + getattr(encodings[2], key)
      
      reduced_encodings.append(ModelIO(results))

    # Collapse array of ModelIO's into single ModelIO by stacking the enc_hidden
    # and enc_output tensors
    dec_input = ModelIO({
      # "enc_outputs" : torch.hstack([r.enc_outputs for r in reduced_encodings]),
      # "enc_hidden" : torch.hstack([r.enc_hidden for r in reduced_encodings]),
      "source" : batch.source,
      "transform" : batch.annotation
    })

    for key in reduced_encodings[0].__dict__.keys():
      if type(getattr(reduced_encodings[0], key)) is tuple:
        dec_input.set_attribute(
          key, 
          tuple(map(torch.hstack, zip(*[getattr(r, key) for r in reduced_encodings])))
          # torch.hstack([getattr(r, key) for r in reduced_encodings])
        )
      else:
        dec_input.set_attribute(key, torch.hstack([getattr(r, key) for r in reduced_encodings]))

    dec_output = self._decoder(dec_input, tf_ratio = 0.0)
    return dec_output.dec_outputs

  def forward_expression(self, expressions):
    """
    ...
    """

    representations = []
    sources = []

    # Compute forward pass on each sub-expression
    for term in expressions:
      if isinstance(term, str):
        representations.append(term)
      else:
        enc_input = ModelIO({"source" : term.source})
        sources.append(term.source)
        enc_output = self._encoder(enc_input)
        representations.append(enc_output)

    # Perform arithmetic operation on representations
    buffer = {
      "enc_hidden" : None,
      "enc_outputs" : None
    }
    should_operate = False
    for r in representations:
      if isinstance(r, str):
        should_operate = torch.add if r == "+" else torch.subtract
      else:
        if not should_operate:
          for key in buffer:
            try:
              buffer[key] = getattr(r, key)
            except:
              buffer[key] = None
        else:
          for key in buffer:
            try:
              buffer[key] = should_operate(buffer[key], getattr(r, key))
            except:
              buffer[key] = None

          should_operate = False
    
    dec_inputs = ModelIO({
      "source" : sources[0],
      "transform" : expressions[0].annotation
    })

    for key in buffer:
      if buffer[key] is not None:
        dec_inputs.set_attribute(key, buffer[key])

    dec_output = self._decoder(dec_inputs, tf_ratio = 0.0)
    return dec_output.dec_outputs
  
  def forward(self, batch: Batch, tf_ratio: float = 0.0, plot_trajectories=False):
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

    # print(enc_output)
    # raise SystemExit

    if hasattr(batch, 'target'):
      enc_output.set_attribute("target", batch.target)

    dec_output = self._decoder(enc_output, tf_ratio=tf_ratio)

    return dec_output.dec_outputs
