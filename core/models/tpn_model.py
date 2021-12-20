import torch
import torch.nn as nn
from omegaconf import DictConfig

from core.models.model_io import ModelIO
from core.models.binding_operations import BindingLayer


class TensorProductEncoder(nn.Module):
    @property
    def num_roles(self) -> int:
        return int(self.cfg.model.num_roles)

    @property
    def num_fillers(self) -> int:
        return int(self.cfg.model.num_fillers)

    @property
    def role_dim(self) -> int:
        return int(self.cfg.model.role_dim)

    @property
    def filler_dim(self) -> int:
        return int(self.cfg.model.filler_dim)

    @property
    def embed_squeeze_dim(self) -> int:
        if self.cfg.model.embed_squeeze_dim is None:
            return None
        else:
            return int(self.cfg.model.embed_squeeze_dim)

    @property
    def embed_squeeze(self) -> bool:
        return self.embed_squeeze_dim is not None

    @property
    def final_layer_width(self) -> int:
        if self.cfg.model.final_layer_width is None:
            return None
        else:
            return int(self.cfg.model.final_layer_width)

    @property
    def has_final_layer(self) -> bool:
        return self.final_layer_width is not None

    @property
    def binder(self) -> str:
        return str(self.cfg.model.binder)

    def _build_model(self, pretrained_embeddings=None) -> None:

        if self.embed_squeeze is False:
            self.filler_embedding = nn.Embedding(self.num_fillers, self.filler_dim)
        else:
            self.filler_embedding = nn.Sequential(
                nn.Embedding(self.num_fillers, self.embed_squeeze_dim),
                nn.Linear(self.embed_squeeze_dim, self.filler_dim),
            )

        if pretrained_embeddings is not None:
            self.filler_embedding.load_state_dict(
                {"weight": torch.FloatTensor(pretrained_embeddings.cuda())}
            )

        self.role_embedding = nn.Embedding(self.num_roles, self.role_dim)

        self.output_layer = nn.Sequential(BindingLayer.for_type(self.binder))
        if self.final_layer_width is not None:
            self.output_layer.add_module(
                "1", nn.Linear(self.filler_dim * self.role_dim, self.final_layer_width)
            )

    def __init__(self, cfg: DictConfig, device, pretrained_embeddings=None):

        super(TensorProductEncoder, self).__init__()

        self.cfg = cfg
        self.device = device
        self._build_model(pretrained_embeddings=pretrained_embeddings)
        self.to(self.device)

    def forward(self, input: ModelIO):

        filler_list = input.fillers
        role_list = input.roles

        embedded_fillers = self.filler_embedding(filler_list)
        embedded_roles = self.role_embedding(role_list)
        output = ModelIO(
            {"bound_embeddings": self.output_layer((embedded_fillers, embedded_roles))}
        )

        return output
