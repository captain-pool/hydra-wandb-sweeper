import logging
import os
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, MutableSequence, Optional, Union

import __main__
import omegaconf
import wandb
import yaml
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
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, ListConfig, OmegaConf
from wandb.apis import InternalApi
from wandb.sdk.lib import filesystem
from wandb.sdk.wandb_setup import _EarlyLogger
from wandb.sdk.wandb_sweep import _get_sweep_url

from hydra_plugins.hydra_wandb_sweeper.config import WandbParameterSpec

# TODO: switch to lazy %-style logging  (will make code look less readable though)
# https://docs.python.org/3/howto/logging.html#optimization
logger = logging.getLogger(__name__)

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


__original_cwd__ = os.getcwd()
__main_file__ = __main__.__file__


# Used by wandb.sweep since it checks if __stage_dir__ in wandb.old.core is set in
# order to create it for eventually saving the sweep config yaml to. If it's not set it defaults
# to 'wandb' + os.sep
wandb.old.core._set_stage_dir(".wandb" + os.sep)


def _my_gitrepo_init(self, root=None, remote="origin", lazy=True):
    self.remote_name = remote
    self._root = __original_cwd__ if root is None else root
    self._repo = None
    if not lazy:
        self.repo


# Monkeypatching GitRepo to use the original working directory as the root.
# This will allow wandb's code save features to be used properly when the hydra
# cwd is different than the code directory.
# Have to patch out here for submitit pickling purposes since out here is executed in the original working directory
# before the task function is pickled via executor.map_array. After unpickling at the node for task fn execution,
# the reference wandb.sdk.lib.git.GitRepo.__init__ -> _my_gitrepo_init is still preserved with original_cwd within
# _my_gitrepo_init preserving its previous value.
wandb.sdk.lib.git.GitRepo.__init__ = _my_gitrepo_init


def _my_save_config_file_from_dict(config_filename, config_dict):
    s = b"wandb_version: 1"
    if config_dict:  # adding an empty dictionary here causes a parse error
        s += b"\n\n" + yaml.dump(
            config_dict,
            Dumper=yaml.SafeDumper,
            default_flow_style=False,
            allow_unicode=True,
            encoding="utf-8",
        )
    data = s.decode("utf-8")
    if ".wandb" not in config_filename and "wandb" in config_filename:
        config_filename = config_filename.replace("wandb", ".wandb", 1)
        os.environ[wandb.env.SWEEP_PARAM_PATH] = config_filename
    filesystem._safe_makedirs(os.path.dirname(config_filename))
    with open(config_filename, "w") as conf_file:
        conf_file.write(data)


# Monkeypatching fn used by wandb.agent for creating the sweep config yaml file. It ignores __stage_dir__ which is
# where wandb.init stores its files (/path/to/.wandb/). Note that wandb.sweep uses a different __stage_dir__ which
# needed to be separately set above (very annoying).
# NOTE: This fn will eventually be moved in wandb.
wandb.sdk.lib.config_util.save_config_file_from_dict = _my_save_config_file_from_dict


def _my_get_program_relpath_from_gitrepo(
    program: str, _logger: Optional[_EarlyLogger] = None
) -> Optional[str]:
    repo = wandb.sdk.lib.git.GitRepo()
    root = repo.root
    if not root:
        root = os.getcwd()
    full_path_to_program = os.path.join(
        root, os.path.relpath(__original_cwd__, root), program
    )
    if os.path.exists(full_path_to_program):
        relative_path = os.path.relpath(full_path_to_program, start=root)
        if "../" in relative_path:
            if _logger:
                _logger.warning("could not save program above cwd: %s" % program)
            return None
        return relative_path

    if _logger:
        _logger.warning("could not find program at %s" % program)
    return None


# Monkeypatching to force wandb to use the original cwd when creating
# full_path_to_program. Otherwise the hydra cwd would be used, which could
# be located away from the code directory.
# Patching out here for same reasons explained in previous patch comment.
wandb.sdk.wandb_settings._get_program_relpath_from_gitrepo = (
    _my_get_program_relpath_from_gitrepo
)


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
class WandbSweeper(Sweeper):
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

        self.agent_run_count: Optional[int] = None
        self.job_idx: Optional[int] = None

        self.wandb_tags = self.wandb_sweep_config.tags
        self.wandb_notes = self.wandb_sweep_config.notes
        assert OmegaConf.is_list(self.wandb_tags)
        assert OmegaConf.is_config(self.wandb_notes)

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.agent_run_count = self.wandb_sweep_config.count
        self._task_function = lambda task_cfg: (
            self.wandb_task(
                base_config=task_cfg,
                task_function=task_function,
                count=self.agent_run_count,
            )
        )
        self.hydra_context = hydra_context
        self.job_idx = 0

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

        self.wandb_tags = (
            OmegaConf.to_container(self.wandb_tags) if self.wandb_tags else None
        )
        self.wandb_notes = (
            str(OmegaConf.to_container(self.wandb_notes)) if self.wandb_notes else None
        )

        # For keeping track of original code working directory without resorting to hydra.util get_original_cwd()
        # since HydraConfig hasn't been instantiated yet (happens after launch, which is too late)
        self.program = __main_file__
        self.program_relpath = __main_file__

    @property
    def sweep_id(self) -> Any:
        return self._sweep_id

    def sweep(self, arguments: List[str]) -> None:
        assert self.config is not None
        assert self.launcher is not None
        assert self.job_idx is not None

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

        logger.info(
            f"WandbSweeper(method={self.wandb_sweep_config.method}, "
            f"num_agents={self.wandb_sweep_config.num_agents}, count={self.wandb_sweep_config.count}, "
            f"entity={self.wandb_sweep_config.entity}, project={self.wandb_sweep_config.project}, "
            f"name={self.wandb_sweep_config.name})"
        )
        logger.info(f"with parameterization {wandb_params}")
        logger.info(
            f"Sweep output dir: {to_absolute_path(self.config.hydra.sweep.dir)}"
        )

        # Creating this folder early so that wandb.init can write to this location.
        Path(to_absolute_path(self.config.hydra.sweep.dir)).mkdir(
            exist_ok=True, parents=True
        )
        (Path(to_absolute_path(self.config.hydra.sweep.dir)) / ".wandb").mkdir(
            exist_ok=True, parents=False
        )

        # Unfortuately wandb.sweep doesn't pay attn to this since it uses InternalApi which uses the old Settings
        # class that uses wandb.old.core for retrieving the wandb dir. It has its own __stage_dir__.
        os.environ["WANDB_DIR"] = to_absolute_path(self.config.hydra.sweep.dir)

        wandb_api = InternalApi()
        if not sweep_id:
            # Need to set PROGRAM env var to original program location since wandb.sweep can't take in a
            # wandb.Settings object, unlike wandb.init
            os.environ[wandb.env.PROGRAM] = self.program

            # Wandb sweep controller will only sweep through
            # params provided by self.sweep_dict
            sweep_id = wandb.sweep(
                self.sweep_dict,
                entity=self.wandb_sweep_config.entity,
                project=self.wandb_sweep_config.project,
            )
            logger.info(
                f"Starting Wandb sweep with ID: {sweep_id} at URL: {_get_sweep_url(wandb_api, sweep_id)}"
            )
        else:
            logger.info(
                f"Reusing Wandb sweep with ID: {sweep_id} at URL: {_get_sweep_url(wandb_api, sweep_id)}"
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
        returns = self.launcher.launch(overrides, initial_job_idx=self.job_idx)
        # self.job_idx += len(returns)  # TODO: use for budget approach

    def wandb_task(
        self,
        *,
        base_config: DictConfig,
        task_function: TaskFunction,
        count: Optional[int] = 1,
    ) -> None:
        runtime_cfg = HydraConfig.get()
        sweep_dir = Path(to_absolute_path(runtime_cfg.sweep.dir))
        sweep_subdir = sweep_dir / Path(runtime_cfg.sweep.subdir)

        # Need to set PROGRAM env var to original program location since passing it through wandb_settings doesn't
        # apply to wandb.agent which checks where the program is located in order to do code-save-related things.
        # However, with wandb.init we're fine since we pass wandb_settings to it.
        os.environ[wandb.env.PROGRAM] = self.program
        wandb_settings = wandb.Settings(
            start_method="thread",
            program=self.program,
            program_relpath=self.program_relpath,
        )

        def run() -> Any:
            logger.info("Agent initializing wandb...")
            # TODO: test resuming sweeps and resuming runs. Will need to set resume=True after preemption. Can check
            # via self.config after overriding checkpoint(self, *args: Any, **kwargs: Any) in BaseSubmititLauncher
            # and providing sweep_overrides = {'hydra.sweeper.wandb_sweep_config.resume': True} to **kwargs
            with wandb.init(
                name=sweep_subdir.name,
                group=sweep_dir.name,
                settings=wandb_settings,
                notes=self.wandb_notes,
                tags=self.wandb_tags,
                dir=str(sweep_dir),
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

                logger.info(
                    f"Agent initialized with id={run.id}, name={run.name}, "
                    f"config={flatten_dict(run.config.as_dict())} at URL: {run.get_url()}"
                )
                logger.info(f"Agent {run.id} executing task function...")
                ret = task_function(config)
                logger.info(f"Agent {run.id} finished executing task function")
                return ret

        if not self.sweep_id:
            raise ValueError(f"sweep_id cannot be {self.sweep_id}")

        logger.info("Launching Wandb agent...")
        wandb.agent(self.sweep_id, function=run, count=count)
