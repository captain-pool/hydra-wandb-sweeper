from typing import List, Optional

from hydra import TaskFunction
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext
from omegaconf import DictConfig

from hydra_plugins.hydra_wandb_sweeper.config import WandbConfig


class WandbSweeper(Sweeper):
    """Class to interface with Wandb"""

    def __init__(self, wandb_sweep_config: WandbConfig, params: Optional[DictConfig]):
        from hydra_plugins.hydra_wandb_sweeper._impl import WandbSweeperImpl

        self.sweeper = WandbSweeperImpl(wandb_sweep_config, params)

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        return self.sweeper.setup(
            hydra_context=hydra_context, task_function=task_function, config=config
        )

    def sweep(self, arguments: List[str]) -> None:
        return self.sweeper.sweep(arguments)
