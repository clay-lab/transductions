# arith.py
#
# Enter an interactive REPL for sentence arithmetic.

import os

import hydra
from omegaconf import DictConfig, OmegaConf

from core.trainer import Trainer


@hydra.main(config_path="conf", config_name="repl.yaml")
def main(cfg: DictConfig) -> None:

    # Load checkpoint configuration: Since the REPL's own
    # entrypoint is different than that of the saved model,
    # we have to load them separately as different config
    # files.
    chkpt_dir = hydra.utils.to_absolute_path(cfg.checkpoint_dir)
    chkpt_cfg_path = os.path.join(chkpt_dir, ".hydra", "config.yaml")
    chkpt_cfg = OmegaConf.load(chkpt_cfg_path)

    trainer = Trainer(chkpt_cfg)
    trainer.arith(repl_cfg=cfg)


if __name__ == "__main__":
    main()
