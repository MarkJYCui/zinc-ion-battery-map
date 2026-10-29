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


def test_convert_concentration_to_mM_and_solubility_formatting():
    from discovery.formula_utils import convert_concentration_to_mM, format_solubility_conc
    from discovery.domain_config import ZIBDomainConfig
    from models import ExtractedAdditive, Concentration, ZincIonBatteryContext
    from optimization.concentration_anchor import ConcentrationAnchorEngine
    from discovery.literature_engine import LiteratureDiscoveryEngine

    # 1. Molar (M, mol L-1, mol/L, mol dm-3) -> mM (* 1000)
    assert convert_concentration_to_mM(0.1, "M") == (100.0, "mM")
    assert convert_concentration_to_mM(0.1, "mol L-1") == (100.0, "mM")
    assert convert_concentration_to_mM(0.05, "mol/L") == (50.0, "mM")
    assert convert_concentration_to_mM(0.2, "mol·L⁻¹") == (200.0, "mM")
    # Even if unit was defaulted to mM, raw_text with mol L-1 is recognized
    assert convert_concentration_to_mM(0.1, "mM", raw_text="0.1 mol L-1") == (100.0, "mM")

    # 2. Micromolar (uM, µM, umol/L) -> mM (/ 1000)
    assert convert_concentration_to_mM(500.0, "µM") == (0.5, "mM")
    assert convert_concentration_to_mM(250.0, "uM") == (0.25, "mM")

    # 3. Mass/volume (g/L, mg/mL) -> mM when MW > 0, preserved when MW == 0
    val_mm, unit_mm = convert_concentration_to_mM(18.217, "g/L", mw=182.17)
    assert val_mm == pytest.approx(100.0, rel=1e-3)
    assert unit_mm == "mM"
    assert convert_concentration_to_mM(10.0, "mg/mL", mw=0.0) == (10.0, "mg/mL")

    # 4. format_solubility_conc switches from mM to M at >= 100 mM
    assert format_solubility_conc(45.6) == "45.6 mM"
    assert format_solubility_conc(99.9) == "99.9 mM"
    assert format_solubility_conc(100.0) == "0.10 M"
    assert format_solubility_conc(1850.0) == "1.85 M"

    # 5. Domain config normalize_unit handles mol L-1 and Unicode superscripts
    cfg = ZIBDomainConfig()
    assert cfg.normalize_unit("mol L-1") == "M"
    assert cfg.normalize_unit("mol·L⁻¹") == "M"
    assert cfg.normalize_unit("mmol L-1") == "mM"

    # 6. LiteratureDiscoveryEngine & ConcentrationAnchorEngine convert 0.1 M -> 100.0 mM
    add_molar = ExtractedAdditive(
        name="D-Sorbitol",
        smiles="O[C@H]([C@H](O)[C@@H](O)[C@H](O)CO)CO",
        optimal_concentration=Concentration(value=0.1, unit="M", raw_text="0.1 mol L-1"),
        battery_context=ZincIonBatteryContext(baseline_electrolyte="Unspecified / Not reported"),
    )
    lit_eng = LiteratureDiscoveryEngine()
    t1, _ = lit_eng.generate_two_tier_candidates([add_molar], use_pubchem=False)
    assert len(t1) == 1
    assert t1[0].reported_concentration == pytest.approx(100.0)
    assert t1[0].reported_concentration_unit == "mM"
    assert t1[0].parent_baseline_electrolyte == "Unspecified / Not reported"

    anchor_eng = ConcentrationAnchorEngine()
    anchor = anchor_eng.anchor_from_extracted_additive(add_molar, target_electrolyte="2 M ZnSO4")
    assert anchor.published_concentration == pytest.approx(100.0)
    assert anchor.is_baseline_unspecified is True


