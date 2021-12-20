# bert_encoder.py
#
# Provides BERTEncoder module.

from typing import List

from core.models.components import TransductionComponent
from core.models.model_io import ModelIO
from core.models.positional_encoding import PositionalEncoding
from omegaconf import DictConfig
from torch import Tensor, nn
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer
from transformers.models.distilbert.modeling_distilbert import Embeddings


class PositionalBertEmbeddings(Embeddings):
    def __init__(self, config):
        super().__init__(config)

        self.pos_enc = PositionalEncoding(768)

    def forward(self, input_ids):
        embeddings = super().forward(input_ids)
        embeddings = self.pos_enc(embeddings)

        return embeddings


class BERTEncoder(TransductionComponent):
    @property
    def hidden_size(self) -> int:
        return 768

    @property
    def num_heads(self) -> int:
        return 4

    @property
    def num_layers(self) -> int:
        return 1

    @property
    def is_frozen(self) -> bool:
        return bool(self.cfg.should_freeze)

    def __init__(self, cfg: DictConfig):

        super().__init__(cfg)

        config = DistilBertConfig()
        config.sinusoidal_pos_embds = True

        self.module = DistilBertModel.from_pretrained(
            "distilbert-base-uncased", config=config
        )

        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        # Add in our own positional encodings
        embedding_layer = PositionalBertEmbeddings(self.module.config)
        self.module.embeddings = embedding_layer

        if self.is_frozen:
            for param in self.module.parameters():
                param.requires_grad = False

        layer = nn.TransformerEncoderLayer(self.hidden_size, self.num_heads)
        self.unit = nn.TransformerEncoder(layer, num_layers=self.num_layers)

    def forward(self, enc_input: ModelIO) -> ModelIO:

        embedded = self.module(enc_input.source)
        encoded = self.unit(embedded.last_hidden_state)
        return ModelIO({"enc_outputs": encoded})

    def to_tokens(self, idx_tensor: Tensor, show_special=False):

        outputs = idx_tensor.detach().cpu().numpy()
        batch_strings = []
        for o in outputs:
            batch_strings.append(
                self.tokenizer.convert_ids_to_tokens(
                    o, skip_special_tokens=not show_special
                )
            )

        return batch_strings

    def to_ids(self, tokens: List):
        return self.tokenizer.convert_tokens_to_ids(tokens)
