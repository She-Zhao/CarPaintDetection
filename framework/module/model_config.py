import yaml
import json
from pathlib import Path

class ModelConfigManager:
    def __init__(self, config_dir: Path=None):
        self.root = Path(__file__).parent.parent / 'cfg' if config_dir is None else config_dir
        self.param = self._load_param(self.root / 'param.json')
        self.config = self._load_config(self.root / 'config.yaml')

    def _load_param(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            param_data = json.load(f)
        return param_data
        
    def _load_config(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        return config_data
