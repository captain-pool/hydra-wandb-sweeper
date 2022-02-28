from dataclasses import dataclass, field
from typing import Any, Dict, MutableSequence, Optional

from hydra.core import config_store
from omegaconf import DictConfig, ListConfig


@dataclass
class WandbParameterSpec:
    """Representation of all the options to define
    a Wandb parameter.
    """

    # (int or float) Maximum and minimum values.
    # if int, for int uniform-distributed hyperparameters. If float, for uniform-distributed hyperparameters.
    min: Optional[float] = None
    max: Optional[float] = None

    # (float) Mean parameter for normal or log normal-distributed hyperparameters.
    mu: Optional[float] = 0

    # (float) Standard deviation parameter for normal or log normal-distributed hyperparameters.
    sigma: Optional[float] = 1

    # (float) Quantization step size for quantized hyperparameters.
    q: Optional[float] = 1

    # Specify how values will be distributed
    # Supported distributions and required parameters
    # "constant": ["value"],
    # "categorical": ["values"],
    # "int_uniform": ["min", "max"],
    # "uniform": ["min", "max"],
    # "q_uniform": ["min", "max"],
    # "log_uniform": ["min", "max"],
    # "q_log_uniform": ["min", "max"],
    # "inv_log_uniform": ["min", "max"],
    # "normal": ["mu", "sigma"],
    # "q_normal": ["mu", "sigma"],
    # "log_normal": ["mu", "sigma"],
    # "q_log_normal": ["mu", "sigma"],
    distribution: Optional[str] = None
    value: Optional[float] = None
    values: Optional[MutableSequence] = None


@dataclass
class WandbConfig:
    name: str
    method: str
    count: Optional[int] = None
    metric: Optional[DictConfig] = DictConfig({})
    num_agents: Optional[int] = 1
    sweep_id: Optional[str] = None
    entity: Optional[str] = None
    project: Optional[str] = None
    early_terminate: Optional[DictConfig] = DictConfig({})
    tags: Optional[ListConfig] = ListConfig([])

    # Notes can contain a string, a list, or any OmegaConf type (e.g., if you wanted to pass a config value
    # that interoplated into a ListConfig, like ${hydra.overrides.task})
    notes: Optional[Any] = None


@dataclass
class WandbSweeperConf:
    wandb_sweep_config: WandbConfig
    _target_: str = "hydra_plugins.hydra_wandb_sweeper.wandb_sweeper.WandbSweeper"

    # default parametrization of the search space
    # can be specified:
    # - as a string, like commandline arguments
    # - as a list, for categorical variables
    # - as a full scalar specification
    params: Dict[str, Any] = field(default_factory=dict)


config_store.ConfigStore.instance().store(
    group="hydra/sweeper", name="wandb", node=WandbSweeperConf, provider="wandb_sweeper"
)
