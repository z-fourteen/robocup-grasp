import pytest

from robot_grasp.errors import ValidationError
from robot_grasp.io_utils import prepare_output_dir


def test_existing_empty_output_requires_overwrite(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValidationError, match="already exists"):
        prepare_output_dir(output)
    assert prepare_output_dir(output, overwrite=True) == output


def test_overwrite_does_not_clean_unrelated_files(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("user data", encoding="utf-8")
    prepare_output_dir(output, overwrite=True)
    assert marker.read_text(encoding="utf-8") == "user data"
