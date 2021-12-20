# arith_eval.py
#
# Evaluate model on defined arithmetic expressions.

import os
import hydra
from omegaconf import OmegaConf, DictConfig
from core.trainer import Trainer


@hydra.main(config_path="conf", config_name="eval_arith.yaml")
def main(cfg: DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))

    # Load checkpoint configuration
    chkpt_dir = hydra.utils.to_absolute_path(cfg.checkpoint_dir)
    chkpt_cfg_path = os.path.join(chkpt_dir, ".hydra", "config.yaml")
    chkpt_cfg = OmegaConf.load(chkpt_cfg_path)

    trainer = Trainer(chkpt_cfg)
    trainer.arith_eval(eval_cfg=cfg)


if __name__ == "__main__":
    main()
