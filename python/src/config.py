import yaml
from pathlib import Path
from typing import List, Optional

from dataclasses import dataclass, field


@dataclass
class GenerationDefaults:
    temperature: float = 0.0
    max_tokens: int = 16384
    top_p: float = 1.0
    stop: Optional[List[str]] = None


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    context_length: int = 4096
    defaults: GenerationDefaults = field(default_factory=GenerationDefaults)


def load_config(config_path: Optional[str] = None) -> ServerConfig:
    """Load server config.

    If config_path is provided, load overrides from that YAML file.
    Otherwise, return built-in defaults.
    """
    config = ServerConfig()

    if config_path is None:
        return config

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with open(path) as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse config at {path}: {e}")

    if config_data is None:
        return config

    server_data = config_data.get("server", {})
    if "host" in server_data:
        config.host = server_data["host"]
    if "port" in server_data:
        config.port = int(server_data["port"])
    if "context_length" in server_data:
        config.context_length = int(server_data["context_length"])

    defaults_data = config_data.get("defaults", {})
    if defaults_data:
        config.defaults = GenerationDefaults(
            temperature=defaults_data.get("temperature", config.defaults.temperature),
            max_tokens=defaults_data.get("max_tokens", config.defaults.max_tokens),
            top_p=defaults_data.get("top_p", config.defaults.top_p),
            stop=defaults_data.get("stop", config.defaults.stop),
        )

    return config
