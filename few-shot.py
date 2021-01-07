# few-shot.py
# 
# Entry point for running few-shot experiments.

import hydra
from omegaconf import OmegaConf, DictConfig

# Library imports
from core.few_shot_trainer import FewShotTrainer

@hydra.main(config_path="conf", config_name="few-shot")
def main(cfg: DictConfig) -> None:
  if cfg.pretty_print:
    print(OmegaConf.to_yaml(cfg))
  
  trainer = FewShotTrainer(cfg)
  trainer.train()

if __name__ == "__main__":
  main()