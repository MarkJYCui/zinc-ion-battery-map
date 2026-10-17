"""Tests for core.formulation_utils volumetric balance and safeguards."""
import pytest
from core.formulation_utils import calculate_diluent_volume, Component
from core.exceptions import OversaturatedFormulationWarning


def test_calculate_diluent_volume_normal():
    components = [Component(volume=50.0), Component(volume=100.0)]
    diluent = calculate_diluent_volume(components, target_vol=200.0)
    assert diluent == 50.0


def test_calculate_diluent_volume_oversaturated_triggers_warning():
    components = [Component(volume=150.0), Component(volume=100.0)]
    with pytest.warns(OversaturatedFormulationWarning) as record:
        diluent = calculate_diluent_volume(components, target_vol=200.0)
    
    assert diluent == 0.0
    assert len(record) == 1
    assert "exceeded by stock components" in str(record[0].message)
    assert "Clamping diluent volume to 0.0" in str(record[0].message)


def test_calculate_diluent_volume_exact_match():
    components = [Component(volume=120.0), Component(volume=80.0)]
    diluent = calculate_diluent_volume(components, target_vol=200.0)
    assert diluent == 0.0

