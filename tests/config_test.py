import os
import pytest

from tests import FIXTURES_ROOT

from xbert import Config
from xbert.occlusion.strategies import UnkReplacement

TEST_CONFIG = {
        "strategy": "unk_replacement",
        "unk_token": "__UNK__",
        "seed": 1234
}


def test_config_create():
    with pytest.raises(ValueError) as excinfo:
        Config(value="test")
    assert "no strategy" in str(excinfo.value).lower()

    with pytest.raises(ValueError) as excinfo:
        Config(strategy="not_available")
    assert "unknown strategy" in str(excinfo.value).lower()

    conf = Config(**TEST_CONFIG)

    assert isinstance(conf.strategy, UnkReplacement)
    assert conf.batch_size == 32
    assert conf.seed == 1234


def test_config_create_from_dict():
    conf = Config.from_dict(TEST_CONFIG)
    assert isinstance(conf.strategy, UnkReplacement)
    assert conf.batch_size == 32
    assert conf.seed == 1234


def test_config_create_from_json_file():
    path = os.path.join(FIXTURES_ROOT, "test_config.json")
    conf = Config.from_json_file(path)
    assert isinstance(conf.strategy, UnkReplacement)
    assert conf.batch_size == 32
    assert conf.seed == 1234
