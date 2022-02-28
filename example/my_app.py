import logging

import hydra
import wandb
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(config_path="./conf", config_name="config")
def dummy_training(cfg: DictConfig) -> None:
    """A dummy function to minimize
    Minimum is 0.0 at:
    lr = 0.12, dropout=0.33, db=mnist, batch_size=4

    to execute code --> python my_app.py --multirun +sweeper=wandb
    """
    dropout = cfg.model.dropout
    batch_size = cfg.experiment.batch_size
    database = cfg.experiment.db
    lr = cfg.experiment.lr

    out = float(
        abs(dropout - 0.33)
        + int(database == "mnist")
        + abs(lr - 0.12)
        + abs(batch_size - 4)
    )
    logger.info(
        f"dummy_training(dropout={dropout:.3f}, lr={lr:.3f}, db={database}, batch_size={batch_size}) = {out:.3f}",
    )
    assert wandb.run is not None
    wandb.log({"out": out})


if __name__ == "__main__":
    dummy_training()
