import logging
import threading
from time import sleep

import hydra
import wandb
from hydra.types import TaskFunction
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class DummyTraining(TaskFunction):
    def __init__(self) -> None:
        self.awake = False

    def background_calculation(self, result_available, progress):
        # here goes some long calculation
        for i in range(100):
            sleep(0.1)
            progress.append(i + 1)

        # when the calculation is done, the result is stored in a global variable
        result_available.set()

    def __call__(self, cfg: DictConfig) -> float:
        """
        A dummy function to minimize
        Minimum is 0.0 at:
        lr = 0.12, dropout=0.33, db=mnist, batch_size=4

        ------------------------------------------------------------------------
        to execute code locally:
          `python my_app.py --multirun +sweeper=wandb`
        ------------------------------------------------------------------------


        ------------------------------------------------------------------------
        to execute code on SLURM cluster to test wandb pre-emption:
          `python my_app.py --multirun +sweeper=wandb +launcher=submitit_remote`
          then execute `scancel --signal=USR1 <JOB_ID>`

          --> requires submit it launcher to be installed
              `pip install hydra-submitit-launcher --upgrade`
        ------------------------------------------------------------------------
        """
        # Simulating running some blocking background task ~10s long for testing out pre-emption via
        # the submitit plugin and sending `scancel --signal=USR1 <JOB_ID>` to this job.
        progress = []
        result_available = threading.Event()
        thread = threading.Thread(
            target=self.background_calculation, args=(result_available, progress)
        )
        self.awake = False
        thread.start()
        # poll every 5s for the result to be available before continuing
        while not result_available.wait(timeout=5):
            logger.info(f"{len(progress)}% done...")
        self.awake = True

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
        wandb.log({"out": out})

        if cfg.error:
            raise RuntimeError("cfg.error is True")

        if cfg.return_type == "float":
            return out

        if cfg.return_type == "dict":
            return dict(name="objective", type="objective", value=out)

        if cfg.return_type == "list":
            return [dict(name="objective", type="objective", value=out)]

        if cfg.return_type == "none":
            return None


if __name__ == "__main__":
    dummy_training = DummyTraining()
    app = hydra.main(config_path="conf", config_name="config")(dummy_training.__call__)
    app()
