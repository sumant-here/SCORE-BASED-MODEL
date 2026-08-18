"""Ablation study configuration generator and combinatorial parameter matrix expander."""

import copy
import itertools
from typing import Any, Dict, List, Union
from src.utils.config import Config
from src.experiments.registry import DEFAULT_DEV_CONFIG


def expand_ablation_matrix(ablation_config: Union[Config, dict]) -> List[Config]:
    """Expand ablation matrix config into individual run configurations.

    Args:
        ablation_config: Configuration dictionary specifying lists for parameters to ablate.

    Returns:
        List of single-run Config instances.
    """
    raw = ablation_config.to_dict() if isinstance(ablation_config, Config) else copy.deepcopy(ablation_config)
    base_name = raw.get("experiment_name", "ablation")

    # Parameters to search for lists
    sweep_keys = {}

    def extract_sweeps(d: dict, prefix: str = ""):
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, list) and not (k in ["channel_multipliers", "attention_resolutions", "selected_classes"] and all(isinstance(x, int) for x in v)):
                sweep_keys[full_key] = v
            elif isinstance(v, dict):
                extract_sweeps(v, full_key)

    extract_sweeps(raw)

    if not sweep_keys:
        # No sweep parameters, return single config
        return [Config(raw)]

    # Generate Cartesian product of all sweep parameters
    keys = list(sweep_keys.keys())
    values = list(sweep_keys.values())
    combinations = list(itertools.product(*values))

    run_configs = []
    for idx, combo in enumerate(combinations):
        cfg = copy.deepcopy(raw)
        name_parts = [base_name]

        for k, val in zip(keys, combo):
            # Set nested value in cfg
            parts = k.split(".")
            target = cfg
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = val

            # Append short tag to experiment name
            short_k = parts[-1]
            val_str = str(val).replace("[", "").replace("]", "").replace(" ", "")
            name_parts.append(f"{short_k}_{val_str}")

        cfg["experiment_name"] = "_".join(name_parts)
        run_configs.append(Config(cfg))

    return run_configs
