# main.py
# 
# Entry point for the transductions library.

from typing import Dict
import hydra
from omegaconf import OmegaConf, DictConfig
from core.trainer import Trainer

@hydra.main(config_path="config", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
  if cfg.pretty_print:
    print(OmegaConf.to_yaml(cfg))
  
  trainer = Trainer(cfg)
  trainer.train()

if __name__ == "__main__":
  main()