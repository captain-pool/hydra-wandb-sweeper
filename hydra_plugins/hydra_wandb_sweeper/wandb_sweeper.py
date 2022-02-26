import functools
import itertools
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, MutableSequence, Optional, Union

import omegaconf
import wandb
from hydra.core import plugins
from hydra.core.hydra_config import HydraConfig
from hydra.core.override_parser import overrides_parser
from hydra.core.override_parser.types import (
    ChoiceSweep,
    IntervalSweep,
    Override,
    RangeSweep,
    Transformer,
)
from hydra.plugins import sweeper
from omegaconf import DictConfig, ListConfig, OmegaConf
from wandb.apis import InternalApi
from wandb.sdk.wandb_sweep import _get_sweep_url

from hydra_plugins.hydra_wandb_sweeper.config import WandbParameterSpec

LOGGER = logging.getLogger(__name__)

SUPPORTED_DISTRIBUTIONS = {
    "constant": ["value"],
    "categorical": ["values"],
    "int_uniform": ["min", "max"],
    "uniform": ["min", "max"],
    "q_uniform": ["min", "max", "q"],
    "log_uniform": ["min", "max"],
    "q_log_uniform": ["min", "max", "q"],
    "inv_log_uniform": ["min", "max"],
    "normal": ["mu", "sigma"],
    "q_normal": ["mu", "sigma", "q"],
    "log_normal": ["mu", "sigma"],
    "q_log_normal": ["mu", "sigma", "q"],
}


def is_wandb_override(override: Override) -> bool:
    if (
        override.is_delete()
        or override.is_add()
        or override.is_force_add()
        or override.is_hydra_override()
    ):
        return False
    else:
        return True


def get_parameter(distribution, *args) -> Dict[str, Any]:
    keys = SUPPORTED_DISTRIBUTIONS[distribution]
    parameter = {"distribution": distribution}
    for key, value in zip(keys, args):
        if value is None:
            raise TypeError(f"{key} must be assigned a value.")
        if distribution not in ("constant", "categorical"):
            if (key == "min" or key == "max") and "int" in distribution:
                if not isinstance(value, int):
                    raise TypeError(f"{value} assigned to {key} must be an integer.")
            else:
                value = float(value)
        if distribution in "constant" and isinstance(value, MutableSequence):
            assert len(value) == 1
            value = value[0]
        if isinstance(value, ListConfig):
            value = OmegaConf.to_container(value, resolve=True)
        parameter[key] = value
    return parameter


def create_wandb_param_from_config(
    config: Union[MutableSequence[Any], MutableMapping[str, Any]]
) -> Any:
    if isinstance(config, MutableSequence):
        if isinstance(config, ListConfig):
            config = OmegaConf.to_container(config, resolve=True)
        assert len(config) > 0
        distribution = "constant" if len(config) == 1 else "categorical"
        return get_parameter(distribution, config)
    if isinstance(config, MutableMapping):
        specs = WandbParameterSpec(**config)
        distribution = specs.distribution
        if distribution not in SUPPORTED_DISTRIBUTIONS:
            raise ValueError(
                f"{distribution} not supported. "
                f"Supported Distributions: {list(SUPPORTED_DISTRIBUTIONS.keys())}"
            )
        supported_params = SUPPORTED_DISTRIBUTIONS[distribution]
        init_params = [getattr(specs, p) for p in supported_params]
        param = get_parameter(specs.distribution, *init_params)
        return param
    param = get_parameter("constant", config)
    return param


def create_wandb_param_from_override(override: Override) -> Any:
    value = override.value()
    distribution = None
    if getattr(value, "tags", None):
        assert len(value.tags) == 1  # TODO: support 'grid' search method, i.e., no tags
        distribution = list(value.tags)[0]
        if distribution not in SUPPORTED_DISTRIBUTIONS:
            raise ValueError(
                f"{distribution} not supported. "
                f"Supported Distributions: {list(SUPPORTED_DISTRIBUTIONS.keys())}"
            )
    if not override.is_sweep_override():
        return value
    if override.is_interval_sweep():
        assert isinstance(value, IntervalSweep)
        distribution = distribution or "uniform"
        if "uniform" not in distribution or "q" in distribution:
            raise ValueError(
                f"Type IntervalSweep only supports non-quantized uniform distributions"
            )
        return get_parameter(distribution, value.start, value.end)
    if override.is_choice_sweep():
        assert isinstance(value, ChoiceSweep)
        choices = [x for x in override.sweep_iterator(transformer=Transformer.encode)]
        distribution = distribution or "categorical"
        if distribution != "categorical":
            raise ValueError(f"Type ChoiceSweep doesn't allow {distribution}")
        return get_parameter(distribution, choices)
    if override.is_range_sweep():
        assert isinstance(value, RangeSweep)
        distribution = distribution or "q_uniform"
        if "uniform" not in distribution or "q" not in distribution:
            raise ValueError(
                f"Type RangeSweep only supports quantized uniform distributions"
            )
        return get_parameter(distribution, value.start, value.end, value.step)

    raise NotImplementedError(f"{override} not supported by WandB sweeper")


def _flatten_dict_gen(d, parent_key, sep):
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep=sep).items()
        else:
            yield new_key, v


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = "."):
    return dict(_flatten_dict_gen(d, parent_key, sep))


# TODO: support for 'grid' sweep method. If user provides 'grid' as the search
# method, then we shouldn't be using SUPPORTED_DISTRIBUTIONS & "distribution"
# and instead return "value" or "values" if the user provides a single value
# or a list of values. This means we need to verify that "distribution" isn't
# provided by the user in any of the params to be sweeped through.
class WandbSweeper(sweeper.Sweeper):
    def __init__(
        self, wandb_sweep_config: omegaconf.DictConfig, params: Optional[DictConfig]
    ) -> None:
        self.wandb_sweep_config = wandb_sweep_config
        self._sweep_id = None
        self.params: Dict[str, Any] = {}

        # setup wandb params from hydra.sweep.params
        if params is not None:
            assert isinstance(params, DictConfig)
            self.params = {
                str(x): create_wandb_param_from_config(y) for x, y in params.items()
            }

        self.agent_run_count = self.wandb_sweep_config.get("count", None)

        self.job_idx = itertools.count(0)

    def setup(self, config, hydra_context, task_function) -> None:
        self.config = config

        self._task_function = functools.partial(
            self.wandb_task,
            task_function=task_function,
            count=self.agent_run_count,
        )
        self.hydra_context = hydra_context

        self.launcher = plugins.Plugins.instance().instantiate_launcher(
            config=config,
            hydra_context=hydra_context,
            task_function=self._task_function,
        )
        self.sweep_dict = {
            "name": self.wandb_sweep_config.name,
            "method": self.wandb_sweep_config.method,
            "parameters": self.params,
        }
        early_terminate = self.wandb_sweep_config.early_terminate
        metric = self.wandb_sweep_config.metric

        if metric:
            self.sweep_dict.update({"metric": omegaconf.OmegaConf.to_container(metric)})

        if early_terminate:
            self.sweep_dict.update(
                {"early_terminate": omegaconf.OmegaConf.to_container(early_terminate)}
            )

    @property
    def sweep_id(self):
        return self._sweep_id

    def sweep(self, arguments: List[str]) -> None:
        parser = overrides_parser.OverridesParser.create()
        parsed = parser.parse_overrides(arguments)
        sweep_id = self.wandb_sweep_config["sweep_id"]

        wandb_params = self.sweep_dict["parameters"]
        hydra_overrides = []

        # Separating command-line overrides meant for wandb params
        # from overrides meant for hydra configuration
        for override in parsed:
            if is_wandb_override(override):
                wandb_params[
                    override.get_key_element()
                ] = create_wandb_param_from_override(override)
            else:
                hydra_overrides.append(override)

        LOGGER.info(
            f"WandbSweeper(method={self.wandb_sweep_config.method}, "
            f"num_agents={self.wandb_sweep_config.num_agets}, count={self.wandb_sweep_config.count}, "
            f"entity={self.wandb_sweep_config.entity}, project={self.wandb_sweep_config.project}, "
            f"name={self.wandb_sweep_config.name})"
        )
        LOGGER.info(f"with parameterization {wandb_params}")
        LOGGER.info(f"Sweep output dir: {self.config.hydra.sweep.dir}")

        wandb_api = InternalApi()
        if not sweep_id:
            # Wandb sweep controller will only sweep through
            # params provided by self.sweep_dict
            sweep_id = wandb.sweep(
                self.sweep_dict,
                entity=self.wandb_sweep_config.entity,
                project=self.wandb_sweep_config.project,
            )
            LOGGER.info(
                f"Starting Sweep with ID: {sweep_id} at URL: {_get_sweep_url(wandb_api, sweep_id)}"
            )
        else:
            LOGGER.info(
                f"Reusing Sweep with ID: {sweep_id} at URL: {_get_sweep_url(wandb_api, sweep_id)}"
            )
        self._sweep_id = sweep_id

        # Repeating hydra overrides to match the number of wandb agents
        # requested. Each agent will interact with the wandb cloud controller
        # to receive hyperparams to send to its associated task function.
        overrides = []
        for _ in range(self.wandb_sweep_config.num_agents):
            overrides.append(
                tuple(
                    f"{override.get_key_element()}={override.value()}"
                    for override in hydra_overrides
                )
            )
        self.validate_batch_is_legal(overrides)

        # Hydra launcher will launch a wandb agent for each hydra override (which
        # will contain the base configuration to be overridden by wandb cloud controller)
        # It's recommended to set hydra.wandb_sweep_config.count to 1 if using the submitit
        # plugin -> https://docs.wandb.ai/guides/sweeps/faq#how-should-i-run-sweeps-on-slurm
        # TODO: would be nice to have a budget-based approach and allow repeatedly launching
        # batches of agents until a budget is reached.
        returns = self.launcher.launch(overrides, initial_job_idx=next(self.job_idx))

    def wandb_task(self, base_config, task_function, count):
        def run():
            runtime_cfg = HydraConfig.get()
            LOGGER.info("Agent initializing wandb...")
            # TODO: allow user to pass in their own notes, name, and tags
            with wandb.init(
                name=Path(os.getcwd()).name,
                settings=wandb.Settings(start_method="thread"),
                notes=OmegaConf.to_yaml(runtime_cfg.overrides.task),
                tags=[
                    base_config.experiment.name,
                    base_config.model._target_,
                    base_config.dataset.name,
                ],
            ) as run:
                override_dotlist = [
                    f"{dot}={val}" for dot, val in run.config.as_dict().items()
                ]
                override_config = OmegaConf.from_dotlist(override_dotlist)
                config = OmegaConf.merge(base_config, override_config)

                # update any values not set by sweep (won't update those already configured by sweep)
                config_dict = OmegaConf.to_container(
                    config, resolve=True, throw_on_missing=True
                )
                config_dot_dict = flatten_dict(config_dict)
                run.config.setdefaults(config_dot_dict)

                LOGGER.info(
                    f"Agent initialized with id={run.id}, name={run.name}, "
                    f"config={flatten_dict(run.config.as_dict())} at URL: {run.get_url()}"
                )
                LOGGER.info(f"Agent {run.id} executing task function...")
                task_function(config, run)
                LOGGER.info(f"Agent {run.id} finished executing task function")

        if not self.sweep_id:
            raise ValueError(f"sweep_id cannot be {self.sweep_id}")

        LOGGER.info("Launching Agent...")
        wandb.agent(self.sweep_id, function=run, count=count)
