import pytest

from processlens.config import CONFIG_DIR, load_config

CONFIGS = ["data", "model", "costs", "rootcause", "benchmark", "agent", "monitoring"]


@pytest.mark.parametrize("name", CONFIGS)
def test_config_loads(name: str) -> None:
    assert isinstance(load_config(name), dict)


def test_split_fractions_sum_to_one() -> None:
    split = load_config("model")["split"]
    assert abs(split["train_frac"] + split["val_frac"] + split["test_frac"] - 1.0) < 1e-9


def test_stochastic_configs_are_seeded() -> None:
    for name in ["data", "model", "rootcause", "benchmark", "agent"]:
        assert "seed" in load_config(name), name


def test_missing_config_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist", CONFIG_DIR)
