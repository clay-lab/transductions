# bert_encoder.py
# 
# Provides BERTEncoder module.

from torch import nn
from omegaconf import DictConfig
from torchtext.vocab import Vocab
from transformers import DistilBertModel, DistilBertConfig
from transformers.models.distilbert.modeling_distilbert import Embeddings

# Library imports
from core.models.model_io import ModelIO
from core.models.positional_encoding import PositionalEncoding

class PositionalBertEmbeddings(Embeddings):

  def __init__(self, config):
    super().__init__(config)

    self.pos_enc = PositionalEncoding(768)
  
  def forward(self, input_ids):
      embeddings = super().forward(input_ids)
      embeddings = self.pos_enc(embeddings)

      return embeddings
      

  # def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
  #   embeddings = super().forward(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds)


class BERTEncoder(nn.Module):

  def __init__(self, cfg: DictConfig, vocab: Vocab):

    super().__init__()

    config = DistilBertConfig()
    config.sinusoidal_pos_embds = True

    self.module = DistilBertModel.from_pretrained('distilbert-base-uncased', config=config)

    # Add in our own positional encodings
    embedding_layer = PositionalBertEmbeddings(self.module.config)
    self.module.embeddings = embedding_layer

    if cfg.should_freeze:
      # Freeze BERT layers to speed up training
      for param in self.module.parameters():
        param.requires_grad = False
    
    layer = nn.TransformerEncoderLayer(self._hidden_size, cfg.num_heads)
    self.unit = nn.TransformerEncoder(layer, num_layers=self._num_layers)

  def forward(self, enc_input: ModelIO) -> ModelIO:

    embedded = self.module(enc_input.source)
    print(embedded)
    raise SystemError
    encoded = self.unit(embedded.last_hidden_state)
    return ModelIO({"enc_outputs" : encoded.last_hidden_state})
