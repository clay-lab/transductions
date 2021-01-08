# main.py
# 
# Entry point for the transductions library.

import hydra
from omegaconf import OmegaConf, DictConfig

# Library imports
from core.few_shot_trainer import FewShotTrainer

@hydra.main(config_path="conf", config_name="few-shot")
def main(cfg: DictConfig) -> None:
  if cfg.pretty_print:
    print(OmegaConf.to_yaml(cfg))
  
  # trainer = Trainer(cfg)
  # trainer.train()

if __name__ == "__main__":
  main()