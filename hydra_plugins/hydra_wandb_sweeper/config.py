from dataclasses import dataclass, field
from typing import Any, Dict, MutableSequence, Optional

from hydra.core.config_store import ConfigStore
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
    # "q_uniform": ["min", "max", "q"],
    # "log_uniform": ["min", "max"],
    # "q_log_uniform": ["min", "max", "q"],
    # "inv_log_uniform": ["min", "max"],
    # "normal": ["mu", "sigma"],
    # "q_normal": ["mu", "sigma", "q"],
    # "log_normal": ["mu", "sigma"],
    # "q_log_normal": ["mu", "sigma", "q"],
    distribution: Optional[str] = None
    value: Optional[float] = None
    values: Optional[MutableSequence] = None


@dataclass
class WandbConfig:
    name: str = "hydra_wandb_sweep"
    method: str = "bayes"

    # number of function evaluations to perform per agent
    count: Optional[int] = None

    metric: Optional[DictConfig] = DictConfig({})

    # number of agents to launch in a batch until budget is reached
    num_agents: Optional[int] = 1

    sweep_id: Optional[str] = None
    entity: Optional[str] = None
    project: Optional[str] = None
    early_terminate: Optional[DictConfig] = DictConfig({})
    tags: Optional[ListConfig] = ListConfig([])

    # total number of agents to launch
    budget: Optional[int] = 1

    # Notes can contain a string, a list, or any OmegaConf type (e.g., if you wanted to pass a config value
    # that interoplated into a ListConfig, like ${hydra.overrides.task})
    notes: Optional[Any] = None

    # maximum authorized failure rate for a batch of wandb agents' runs (since each agent can execute >= 1 runs)
    max_run_failure_rate: float = 0.0

    # maximum authorized failure rate for a batch of wandb agents launched by the launcher
    max_agent_failure_rate: float = 0.0


@dataclass
class WandbSweeperConf:
    wandb_sweep_config: WandbConfig = WandbConfig()
    _target_: str = "hydra_plugins.hydra_wandb_sweeper.wandb_sweeper.WandbSweeper"

    # default parametrization of the search space
    # can be specified:
    # - as a string, like commandline arguments
    # - as a list, for categorical variables
    # - as a full scalar specification
    params: Dict[str, Any] = field(default_factory=dict)


ConfigStore.instance().store(
    group="hydra/sweeper", name="wandb", node=WandbSweeperConf, provider="wandb_sweeper"
)
