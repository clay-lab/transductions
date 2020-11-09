# main.py
# 
# Entry point for the transductions library.

import hydra
from omegaconf import DictConfig, OmegaConf
from core.trainer import Trainer

@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg) -> None:
  if cfg.pretty_print:
    print(OmegaConf.to_yaml(cfg))
  
  trainer = Trainer(cfg)
  trainer.train()

if __name__ == "__main__":
  main()