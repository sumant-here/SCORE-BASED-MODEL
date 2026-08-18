"""Configuration loader and schema management."""

import copy
from pathlib import Path
from typing import Any, Dict, Union
import yaml


class Config(dict):
    """Recursive attribute-accessible dictionary for configuration management."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict) and not isinstance(value, Config):
                self[key] = Config(value)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config back into standard Python dictionary."""
        result = {}
        for key, value in self.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, Config) else item for item in value
                ]
            else:
                result[key] = value
        return result

    def copy(self) -> "Config":
        """Deep copy configuration."""
        return Config(copy.deepcopy(self.to_dict()))


def load_config(config_path: Union[str, Path]) -> Config:
    """Load YAML configuration file into a Config object.

    Args:
        config_path: Path to YAML file.

    Returns:
        Config object.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return Config(data)


def save_config(config: Union[Config, dict], save_path: Union[str, Path]) -> None:
    """Save Config or dict into a YAML file."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.to_dict() if isinstance(config, Config) else config
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
