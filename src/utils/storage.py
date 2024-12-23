import json
import os
from typing import Any, Dict

from utils.logging import k_log


class KubinStorage:
    def __init__(self, base_dir: str = "configs"):
        self.base_dir = base_dir
        self.ui_dir = os.path.join(base_dir, "ui")
        self._ensure_directories()
        self._settings_cache = {}

    def _ensure_directories(self) -> None:
        os.makedirs(self.ui_dir, exist_ok=True)

    def _get_block_filepath(self, block_name: str) -> str:
        return os.path.join(self.ui_dir, f"{block_name}_settings.json")

    def _load(self, block_name: str) -> Dict[str, Any]:
        if block_name in self._settings_cache:
            return self._settings_cache[block_name]

        filepath = self._get_block_filepath(block_name)
        if not os.path.exists(filepath):
            return {}

        try:
            with open(filepath, "r") as f:
                settings = json.load(f)
                self._settings_cache[block_name] = settings
                k_log(f"loaded ui settings for {block_name} from {filepath}")
                return settings
        except (json.JSONDecodeError, IOError):
            return {}

    def save(self, block_name: str, settings: Dict[str, Any]) -> None:
        filepath = self._get_block_filepath(block_name)
        with open(filepath, "w") as f:
            json.dump(settings, f, indent=2)
        self._settings_cache[block_name] = settings

    def get(self, block_name: str, key: str, default: Any = None) -> Any:
        block_settings = self._load(block_name)

        if "." in key:
            current = block_settings
            for part in key.split("."):
                if isinstance(current, dict):
                    current = current.get(part, default)
                else:
                    return default
            return current

        return block_settings.get(key, default)


def get_value(storage, block, name, def_value):
    value = storage.get(block, name, def_value)
    return value
