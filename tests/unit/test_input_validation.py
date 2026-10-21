import pytest

from core.input_validation import validate_image_dimensions, validate_parameters


def test_validate_parameters_accepts_known_valid_values():
    assert validate_parameters(2, 50, "db4", "uniform") is True


def test_validate_parameters_rejects_invalid_quality():
    with pytest.raises(ValueError):
        validate_parameters(2, 120, "db4", "uniform")


def test_validate_image_dimensions_rejects_too_small_values():
    with pytest.raises(ValueError):
        validate_image_dimensions(32, 64)
