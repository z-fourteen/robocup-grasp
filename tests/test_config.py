import pytest

from robot_grasp.config import load_reconstruction_config
from robot_grasp.errors import ValidationError


def test_default_reconstruction_config():
    config = load_reconstruction_config(None)
    assert config.voxel_length == 0.0025
    assert config.sdf_trunc == 0.01
    assert config.depth_min == 0.1
    assert config.depth_max == 1.5
    assert config.depth_quantile_low == 0.0
    assert config.depth_quantile_high == 1.0
    assert config.use_mask is True


def test_unknown_config_key_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("voxel_length: 0.005\ntypo_option: true\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="unknown keys.*typo_option"):
        load_reconstruction_config(path)


def test_non_numeric_config_has_actionable_error(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("voxel_length: tiny\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="finite numeric"):
        load_reconstruction_config(path)


def test_invalid_depth_quantiles_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("depth_quantile_low: 0.9\ndepth_quantile_high: 0.1\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="depth quantiles"):
        load_reconstruction_config(path)
