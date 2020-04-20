from typing import Dict

import json
from olm.occlusion.strategy import STRATEGY_REGISTRY


class Config:
    def __init__(self, **kwargs: Dict[str, str]) -> None:
        strategy = kwargs.pop("strategy", None)
        if strategy is None:
            raise ValueError("No strategy specified.")

        strategy = strategy.lower()
        if strategy not in STRATEGY_REGISTRY.keys():
            raise ValueError("Unknown strategy '%s'." % strategy)

        self.batch_size = kwargs.pop("batch_size", 32)
        self.seed = kwargs.pop("seed", 1111)
        self.strategy = STRATEGY_REGISTRY[strategy](**kwargs)

    @classmethod
    def from_dict(cls, dct: Dict[str, str]) -> "Config":
        """Constructs a `Config` from a Python dictionary of parameters."""
        config = cls(**dct)
        # for key, value in dct.items():
        #     setattr(config, key, value)
        return config

    @classmethod
    def from_json_file(cls, json_file: str) -> "Config":
        """Constructs a `Config` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))
