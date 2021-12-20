# repl.py
#
# Enter an interactive REPL for the model.

import os
import hydra
from omegaconf import OmegaConf, DictConfig
from core.trainer import Trainer


@hydra.main(config_path="conf", config_name="tpdn.yaml")
def tpdn(cfg: DictConfig) -> None:

    # Load checkpoint configuration
    chkpt_dir = hydra.utils.to_absolute_path(cfg.checkpoint_dir)
    chkpt_cfg_path = os.path.join(chkpt_dir, ".hydra", "config.yaml")
    chkpt_cfg = OmegaConf.load(chkpt_cfg_path)

    trainer = Trainer(chkpt_cfg)
    trainer.fit_tpdn(tpdn_cfg=cfg)


if __name__ == "__main__":
    tpdn()
