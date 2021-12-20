# early_stopping.py
#
# Provides a way to stop training early if the model isn't making any headway.

import logging
from omegaconf import DictConfig

log = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(
        self,
        cfg: DictConfig,
    ):

        self.patience = cfg.patience
        self.tolerance = cfg.tolerance
        self.active = cfg.early_stopping

        # Counters
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.should_save = True

    def __call__(self, loss):

        if not self.active:
            return False, True

        score = loss

        if self.best_score is None:
            self.best_score = score
        else:
            delta = self.best_score - score
            if delta <= self.tolerance:
                self.counter += 1
                log.info(f"Early stopping counter: {self.counter} / {self.patience}")
                log.info(
                    "Pausing model saving until validation loss decreases by {}".format(
                        self.tolerance
                    )
                )
                self.should_stop = self.counter >= self.patience
                self.should_save = False
            else:
                self.best_score = score
                self.counter = 0
                self.should_save = True
                self.should_stop = False

        return self.should_stop, self.should_save
