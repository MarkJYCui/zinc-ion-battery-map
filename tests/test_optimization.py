"""
Unit and integration tests for Phase 2: Autonomous Design of Experiments (DoE) & Robotic Dispensing Engine.
Materials Acceleration Platform (MAP) for Zinc-Ion Batteries (ZIB).

Following strict Test-Driven Development (TDD) rules.
Tests cover:
- Pydantic schema validation for Phase 2 models
- Electrolyte-aware concentration anchoring across all 6 aqueous ZIB salts
- Delaney ESOL solubility ceiling enforcement (80% saturation cap)
- Multi-factor DoE sampling (LHS and CCD) with electrolyte-specific salt bounds
- Robotic liquid handling volumetric balance, coordinate mapping, and safety alerts
- End-to-end CLI & JSON orchestration interfacing Phase 1 ExtractionResult
"""

import json
import pytest
from pydantic import ValidationError

from models import (
    ConcentrationAnchor,
    DoEParameter,
    FormulationPoint,
    DoEPlan,
    RoboticVial,
    DispensingSchedule,
    ExtractionResult,
    ExtractedAdditive,
    CandidateMolecule,
    Concentration,
    ZincIonBatteryContext,
    PaperMetadata,
    FormulatedCellResult,
)
from optimization.concentration_anchor import (
    ConcentrationAnchorEngine,
    CANONICAL_ELECTROLYTE_PROFILES,
    ELECTROLYTE_SCALING_RULES,
)
from unittest.mock import patch, MagicMock
from optimization.doe_engine import DoEEngine
from optimization.dispenser import RoboticDispenserEngine
from optimization.planner_cli import run_planner, parse_args
from optimization.concentration_anchor import (
    fetch_pubchem_data,
    get_solubility,
    calculate_target_concentration,
    scale_additive,
    DEMAND_RATIOS,
)
from core.exceptions import OversaturatedFormulationWarning





def test_concentration_anchor_schema():
    """Verify ConcentrationAnchor model validation and fields."""
    anchor = ConcentrationAnchor(
        additive_name="D-Sorbitol",
        smiles="C(C(C(C(C(CO)O)O)O)O)O",
        published_concentration=15.0,
        published_unit="mM",
        parent_electrolyte="2 M ZnSO4",
        parent_salt="ZnSO4",
        parent_molarity=2.0,
        target_electrolyte="1 M Zn(OTf)2",
        target_electrolyte_salt="Zn(OTf)2",
        target_salt_molarity=1.0,
        scaling_factor=0.7,
        shifted_anchor_conc=10.5,
        anchor_unit="mM",
        min_conc=2.1,
        max_conc=31.5,
        solubility_limit_mM=5000.0,
        solubility_ceiling_mM=4000.0,
        is_capped_by_solubility=False,
        electrolyte_adjustment_rationale="Synergistic OTf- solvation modulation reduces additive requirement.",
        functional_groups=["polyol"],
    )
    assert anchor.target_electrolyte_salt == "Zn(OTf)2"
    assert "Synergistic" in anchor.electrolyte_adjustment_rationale
    assert anchor.shifted_anchor_conc == 10.5


def test_doe_parameter_and_formulation_point():
    """Verify DoEParameter and FormulationPoint schemas."""
    param = DoEParameter(
        name="additive_conc_mM",
        display_name="D-Sorbitol Concentration",
        unit="mM",
        min_val=2.1,
        max_val=31.5,
        baseline_val=10.5,
    )
    assert param.min_val == 2.1

    pt = FormulationPoint(
        sample_id="S01",
        additive_conc_mM=10.5,
        salt_conc_M=1.0,
        temperature_C=25.0,
        point_type="center",
    )
    assert pt.sample_id == "S01"
    assert pt.point_type == "center"


def test_robotic_vial_and_dispensing_schedule_schemas():
    """Verify RoboticVial and DispensingSchedule schema integrity."""
    vial = RoboticVial(
        vial_id="A1",
        rack_format="24-vial rack",
        row="A",
        col=1,
        additive_conc_mM=10.0,
        salt_conc_M=1.0,
        target_volume_uL=2000.0,
        additive_stock_uL=200.0,
        salt_stock_uL=666.7,
        water_uL=1133.3,
        warnings=[],
    )
    assert vial.vial_id == "A1"

    schedule = DispensingSchedule(
        schedule_id="SCHED_001",
        target_electrolyte_salt="Zn(OTf)2",
        electrolyte_adjustment_rationale="Target Zn(OTf)2 requires dedicated stock solution.",
        rack_format="24-vial rack",
        total_vials=1,
        total_volume_per_vial_uL=2000.0,
        additive_stock_conc_mM=100.0,
        salt_stock_conc_M=3.0,
        salt_stock_name="Zn(OTf)2 stock solution (3.0 M)",
        vials=[vial],
        warnings=[],
    )
    assert schedule.target_electrolyte_salt == "Zn(OTf)2"
    assert schedule.total_vials == 1


def test_canonical_electrolyte_profiles():
    """Verify canonical electrolyte profiles and their operational ranges."""
    engine = ConcentrationAnchorEngine()
    profiles = engine.get_canonical_profiles()
    assert "ZnSO4" in profiles
    assert profiles["ZnSO4"]["default_molarity"] == 2.0
    assert profiles["ZnSO4"]["molarity_range"] == [1.0, 3.0]
    assert "Zn(OTf)2" in profiles
    assert profiles["Zn(OTf)2"]["default_molarity"] == 1.0
    assert profiles["Zn(OTf)2"]["molarity_range"] == [0.5, 2.5]
    assert "Zn(ClO4)2" in profiles
    assert profiles["Zn(ClO4)2"]["default_molarity"] == 2.0
    assert profiles["Zn(ClO4)2"]["molarity_range"] == [1.0, 3.5]
    assert "Zn(TFSI)2" in profiles
    assert profiles["Zn(TFSI)2"]["default_molarity"] == 4.0
    assert profiles["Zn(TFSI)2"]["molarity_range"] == [2.0, 7.5]
    assert "ZnCl2" in profiles
    assert profiles["ZnCl2"]["default_molarity"] == 5.0
    assert profiles["ZnCl2"]["molarity_range"] == [2.0, 10.0]
    assert "Zn(CH3COO)2" in profiles
    assert profiles["Zn(CH3COO)2"]["default_molarity"] == 1.0
    assert profiles["Zn(CH3COO)2"]["molarity_range"] == [0.5, 2.5]


def test_polyol_scaling_znso4_to_znotf2():
    """
    Test polyol (D-Sorbitol) transfer from 2 M ZnSO4 to 1 M Zn(OTf)2.
    Should apply 0.6x - 0.8x scaling factor due to synergistic OTf- participation.
    """
    engine = ConcentrationAnchorEngine()
    anchor = engine.calculate_anchor(
        additive_name="D-Sorbitol",
        published_conc_mM=20.0,
        parent_electrolyte="2 M ZnSO4",
        target_electrolyte="1 M Zn(OTf)2",
        smiles="C(C(C(C(C(CO)O)O)O)O)O",
    )
    assert anchor.target_electrolyte_salt == "Zn(OTf)2"
    assert 0.6 <= anchor.scaling_factor <= 0.8
    assert anchor.shifted_anchor_conc == pytest.approx(20.0 * anchor.scaling_factor, rel=0.001)
    assert anchor.min_conc == pytest.approx(anchor.shifted_anchor_conc * 0.2, rel=0.001)
    assert anchor.max_conc == pytest.approx(anchor.shifted_anchor_conc * 3.0, rel=0.001)
    assert "OTf" in anchor.electrolyte_adjustment_rationale or "triflate" in anchor.electrolyte_adjustment_rationale.lower()


def test_quaternary_ammonium_scaling_znso4_to_zncl2():
    """
    Test quaternary ammonium transfer from 2 M ZnSO4 to concentrated 5 M ZnCl2.
    Should apply a 1.5x - 2.0x multiplier to overcome EDL compaction from high ionic strength.
    """
    engine = ConcentrationAnchorEngine()
    anchor = engine.calculate_anchor(
        additive_name="Tetramethylammonium bromide",
        published_conc_mM=10.0,
        parent_electrolyte="2 M ZnSO4",
        target_electrolyte="5 M ZnCl2",
        smiles="C[N+](C)(C)C.[Br-]",
    )
    assert anchor.target_electrolyte_salt == "ZnCl2"
    assert 1.5 <= anchor.scaling_factor <= 2.0
    assert anchor.shifted_anchor_conc == pytest.approx(10.0 * anchor.scaling_factor, rel=0.001)
    assert "EDL" in anchor.electrolyte_adjustment_rationale or "ionic strength" in anchor.electrolyte_adjustment_rationale.lower()


def test_quaternary_ammonium_in_zntfsi2_solubility_alert():
    """
    Test quaternary ammonium in Zn(TFSI)2.
    Must enforce strict solubility check and precipitation warning for hydrophobic [NR4][TFSI].
    """
    engine = ConcentrationAnchorEngine()
    anchor = engine.calculate_anchor(
        additive_name="Tetramethylammonium bromide",
        published_conc_mM=10.0,
        parent_electrolyte="2 M ZnSO4",
        target_electrolyte="4 M Zn(TFSI)2",
        smiles="C[N+](C)(C)C.[Br-]",
    )
    assert anchor.target_electrolyte_salt == "Zn(TFSI)2"
    assert "precipitation" in anchor.electrolyte_adjustment_rationale.lower() or "hydrophobic" in anchor.electrolyte_adjustment_rationale.lower()


def test_thiocarbonyl_scaling_znso4_to_zncl2():
    """
    Test thiocarbonyl (Thiourea) transfer from 2 M ZnSO4 to 5 M ZnCl2.
    Should apply a 1.5x multiplier against competitive Cl- surface adsorption.
    """
    engine = ConcentrationAnchorEngine()
    anchor = engine.calculate_anchor(
        additive_name="Thiourea",
        published_conc_mM=10.0,
        parent_electrolyte="2 M ZnSO4",
        target_electrolyte="5 M ZnCl2",
        smiles="C(=S)(N)N",
    )
    assert anchor.target_electrolyte_salt == "ZnCl2"
    assert anchor.scaling_factor == pytest.approx(1.5, rel=0.001)
    assert anchor.shifted_anchor_conc == pytest.approx(15.0, rel=0.001)
    assert "competitive" in anchor.electrolyte_adjustment_rationale.lower() or "cl-" in anchor.electrolyte_adjustment_rationale.lower()


def test_same_salt_scaling_factor_is_one():
    """Test that transfer within the same salt defaults to 1.0x scaling."""
    engine = ConcentrationAnchorEngine()
    anchor = engine.calculate_anchor(
        additive_name="D-Sorbitol",
        published_conc_mM=15.0,
        parent_electrolyte="2 M ZnSO4",
        target_electrolyte="2 M ZnSO4",
        smiles="C(C(C(C(C(CO)O)O)O)O)O",
    )
    assert anchor.scaling_factor == 1.0
    assert anchor.shifted_anchor_conc == 15.0


def test_esol_solubility_ceiling_enforcement():
    """
    Test that Delaney ESOL solubility ceiling (80% of saturation limit)
    caps max_conc when 3.0x anchor exceeds the ceiling.
    """
    engine = ConcentrationAnchorEngine()
    anchor = engine.calculate_anchor(
        additive_name="1-Octanol",
        published_conc_mM=10.0,
        parent_electrolyte="2 M ZnSO4",
        target_electrolyte="2 M ZnSO4",
        smiles="CCCCCCCCO",
    )
    assert anchor.solubility_limit_mM is not None
    assert anchor.solubility_ceiling_mM is not None
    assert anchor.solubility_ceiling_mM == pytest.approx(0.8 * anchor.solubility_limit_mM, rel=0.001)
    assert anchor.is_capped_by_solubility is True
    assert anchor.max_conc <= anchor.solubility_ceiling_mM
    assert "solubility" in anchor.electrolyte_adjustment_rationale.lower()


def test_doe_salt_bounds_adapts_to_target_chemistry():
    """
    Verify Factor 2 (zinc salt concentration) dynamically reflects
    target electrolyte bounds, rather than hardcoding ZnSO4.
    """
    anchor_engine = ConcentrationAnchorEngine()
    doe_engine = DoEEngine()

    anchor_otf = anchor_engine.calculate_anchor(
        additive_name="D-Sorbitol",
        published_conc_mM=10.0,
        parent_electrolyte="2 M ZnSO4",
        target_electrolyte="1 M Zn(OTf)2",
        smiles="C(C(C(C(C(CO)O)O)O)O)O",
    )
    plan_otf = doe_engine.create_plan(anchor=anchor_otf, method="lhs", num_samples=9, seed=42)
    assert plan_otf.target_electrolyte_salt == "Zn(OTf)2"
    salt_param = next(p for p in plan_otf.parameters if p.name == "salt_conc_M")
    assert salt_param.min_val == 0.5
    assert salt_param.max_val == 2.5

    anchor_cl = anchor_engine.calculate_anchor(
        additive_name="Thiourea",
        published_conc_mM=10.0,
        parent_electrolyte="2 M ZnSO4",
        target_electrolyte="5 M ZnCl2",
        smiles="C(=S)(N)N",
    )
    plan_cl = doe_engine.create_plan(anchor=anchor_cl, method="lhs", num_samples=9, seed=42)
    assert plan_cl.target_electrolyte_salt == "ZnCl2"
    salt_param_cl = next(p for p in plan_cl.parameters if p.name == "salt_conc_M")
    assert salt_param_cl.min_val == 2.0
    assert salt_param_cl.max_val == 10.0


def test_lhs_matrix_generation_properties():
    """Test Latin Hypercube Sampling properties, bounds, and stratification."""
    anchor_engine = ConcentrationAnchorEngine()
    doe_engine = DoEEngine()
    anchor = anchor_engine.calculate_anchor(
        additive_name="D-Sorbitol",
        published_conc_mM=15.0,
        parent_electrolyte="2 M ZnSO4",
        target_electrolyte="1 M Zn(OTf)2",
        smiles="C(C(C(C(C(CO)O)O)O)O)O",
    )
    plan = doe_engine.create_plan(anchor=anchor, method="lhs", num_samples=9, seed=42)
    assert len(plan.formulations) == 9
    assert plan.method == "lhs"

    add_param = next(p for p in plan.parameters if p.name == "additive_conc_mM")
    salt_param = next(p for p in plan.parameters if p.name == "salt_conc_M")

    for f in plan.formulations:
        assert add_param.min_val <= f.additive_conc_mM <= add_param.max_val
        assert salt_param.min_val <= f.salt_conc_M <= salt_param.max_val
        assert f.point_type == "lhs"


def test_ccd_matrix_generation_properties():
    """
    Test Central Composite Design (CCD) 2-factor matrix.
    Should generate 4 factorial points + 4 axial points + center points (default 3).
    Total points = 4 + 4 + 3 = 11.
    """
    anchor_engine = ConcentrationAnchorEngine()
    doe_engine = DoEEngine()
    anchor = anchor_engine.calculate_anchor(
        additive_name="D-Sorbitol",
        published_conc_mM=15.0,
        parent_electrolyte="2 M ZnSO4",
        target_electrolyte="1 M Zn(OTf)2",
        smiles="C(C(C(C(C(CO)O)O)O)O)O",
    )
    plan = doe_engine.create_plan(anchor=anchor, method="ccd", ccd_center_points=3)
    assert plan.method == "ccd"
    assert len(plan.formulations) == 11

    center_pts = [f for f in plan.formulations if f.point_type == "center"]
    factorial_pts = [f for f in plan.formulations if f.point_type == "factorial"]
    axial_pts = [f for f in plan.formulations if f.point_type == "axial"]

    assert len(center_pts) == 3
    assert len(factorial_pts) == 4
    assert len(axial_pts) == 4

    add_param = next(p for p in plan.parameters if p.name == "additive_conc_mM")
    salt_param = next(p for p in plan.parameters if p.name == "salt_conc_M")

    expected_add_mid = (add_param.min_val + add_param.max_val) / 2.0
    expected_salt_mid = (salt_param.min_val + salt_param.max_val) / 2.0

    for cp in center_pts:
        assert cp.additive_conc_mM == pytest.approx(expected_add_mid, rel=0.001)
        assert cp.salt_conc_M == pytest.approx(expected_salt_mid, rel=0.001)


def test_doe_with_temperature_factor():
    """Test 3-factor DoE incorporating temperature parameter."""
    anchor_engine = ConcentrationAnchorEngine()
    doe_engine = DoEEngine()
    anchor = anchor_engine.calculate_anchor(
        additive_name="D-Sorbitol",
        published_conc_mM=10.0,
        parent_electrolyte="2 M ZnSO4",
        target_electrolyte="1 M Zn(OTf)2",
        smiles="C(C(C(C(C(CO)O)O)O)O)O",
    )
    plan = doe_engine.create_plan(
        anchor=anchor,
        method="lhs",
        num_samples=9,
        seed=42,
        temperature_range=[20.0, 60.0],
    )
    temp_param = next((p for p in plan.parameters if p.name == "temperature_C"), None)
    assert temp_param is not None
    assert temp_param.min_val == 20.0
    assert temp_param.max_val == 60.0
    for f in plan.formulations:
        assert f.temperature_C is not None
        assert 20.0 <= f.temperature_C <= 60.0


def test_dispensing_volumetric_balance():
    """
    Test volumetric conservation:
    additive_stock_uL + salt_stock_uL + water_uL == target_volume_uL
    for every vial in the schedule.
    """
    anchor_engine = ConcentrationAnchorEngine()
    doe_engine = DoEEngine()
    dispenser = RoboticDispenserEngine()

    anchor = anchor_engine.calculate_anchor(
        additive_name="D-Sorbitol",
        published_conc_mM=15.0,
        parent_electrolyte="2 M ZnSO4",
        target_electrolyte="1 M Zn(OTf)2",
        smiles="C(C(C(C(C(CO)O)O)O)O)O",
    )
    plan = doe_engine.create_plan(anchor=anchor, method="lhs", num_samples=9, seed=42)
    schedule = dispenser.generate_schedule(plan=plan, rack_format="24-vial rack", total_volume_uL=2000.0)

    assert schedule.total_vials == 9
    assert schedule.target_electrolyte_salt == "Zn(OTf)2"
    assert "Zn(OTf)2" in schedule.salt_stock_name

    for vial in schedule.vials:
        total_dispensed = vial.additive_stock_uL + vial.salt_stock_uL + vial.water_uL
        assert total_dispensed == pytest.approx(vial.target_volume_uL, abs=0.001)
        assert vial.water_uL >= 0.0


def test_dispensing_rack_and_plate_coordinate_mapping():
    """Verify physical coordinate layout for 24-vial rack and 96-well plate."""
    dispenser = RoboticDispenserEngine()
    rack_ids_24 = dispenser.generate_coordinate_grid("24-vial rack", count=8)
    assert rack_ids_24[0] == ("A1", "A", 1)
    assert rack_ids_24[5] == ("A6", "A", 6)
    assert rack_ids_24[6] == ("B1", "B", 1)

    plate_ids_96 = dispenser.generate_coordinate_grid("96-well plate", count=15)
    assert plate_ids_96[0] == ("A1", "A", 1)
    assert plate_ids_96[11] == ("A12", "A", 12)
    assert plate_ids_96[12] == ("B1", "B", 1)


def test_liquid_handling_viscosity_and_pipette_warnings():
    """
    Verify liquid handling alerts:
    - High salt viscosity warning when ZnCl2 > 4 M or WiSE Zn(TFSI)2
    - Pipetting accuracy warning when dispensed volume < 5 uL
    """
    anchor_engine = ConcentrationAnchorEngine()
    doe_engine = DoEEngine()
    dispenser = RoboticDispenserEngine()

    anchor = anchor_engine.calculate_anchor(
        additive_name="Thiourea",
        published_conc_mM=10.0,
        parent_electrolyte="2 M ZnSO4",
        target_electrolyte="5 M ZnCl2",
        smiles="C(=S)(N)N",
    )
    plan = doe_engine.create_plan(anchor=anchor, method="lhs", num_samples=6, seed=42)
    schedule = dispenser.generate_schedule(plan=plan, rack_format="24-vial rack", total_volume_uL=2000.0)

    viscosity_warns = [w for w in schedule.warnings if "viscosity" in w.lower() or "zncl2" in w.lower()]
    assert len(viscosity_warns) > 0

    schedule_small = dispenser.generate_schedule(
        plan=plan, rack_format="96-well plate", total_volume_uL=50.0, additive_stock_conc_mM=1000.0
    )
    pipette_warns = [w for v in schedule_small.vials for w in v.warnings if "< 5" in w or "5.0" in w or "threshold" in w.lower()]
    assert len(pipette_warns) > 0


def test_planner_cli_direct_flags(capsys):
    """Test CLI execution using direct parameter flags."""
    args = [
        "--additive", "D-Sorbitol",
        "--published-conc", "15",
        "--parent-electrolyte", "2 M ZnSO4",
        "--target-electrolyte", "Zn(OTf)2",
        "--method", "lhs",
        "--samples", "9",
        "--seed", "42",
        "--json",
    ]
    code = run_planner(args)
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["target_electrolyte_salt"] == "Zn(OTf)2"
    assert data["total_vials"] == 9
    assert len(data["vials"]) == 9
    assert "Zn(OTf)2" in data["salt_stock_name"]


def test_planner_cli_ccd_from_extraction_json(tmp_path, capsys):
    """
    Test piping an ExtractionResult JSON directly from Phase 1
    into Phase 2 planner CLI with target chemistry and CCD design.
    """
    extraction_payload = {
        "paper_metadata": {
            "paper_id": "paper_test_01",
            "title": "Engineering Interfacial Solvation Sheath for Dendrite-Free Zn Anodes",
        },
        "additives": [
            {
                "name": "D-Mannitol",
                "smiles": "C(C(C(C(C(CO)O)O)O)O)O",
                "optimal_concentration": {"value": 20.0, "unit": "mM", "raw_text": "20 mM"},
                "concentration_range_tested": [{"value": 20.0, "unit": "mM", "raw_text": "20 mM"}],
                "functional_groups": ["polyol"],
                "primary_mechanisms": ["Solvation sheath modulation"],
                "mechanism_summary": "Modulates Zn²⁺ solvation structure.",
                "battery_context": {
                    "baseline_electrolyte": "2 M ZnSO4",
                    "solvent": "H2O",
                    "anode_material": "Zn foil",
                    "cell_format": "Symmetric",
                },
                "confidence_score": 0.95,
                "extraction_evidence": "Optimal 20 mM D-mannitol achieved 2000 h.",
            }
        ],
        "candidate_molecules": [],
        "extraction_summary": "Extracted D-Mannitol.",
        "source_type": "pdf",
    }
    json_path = tmp_path / "extraction_result.json"
    json_path.write_text(json.dumps(extraction_payload), encoding="utf-8")

    args = [
        "--from-extraction", str(json_path),
        "--target-electrolyte", "Zn(OTf)2",
        "--method", "ccd",
        "--json",
    ]
    code = run_planner(args)
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["target_electrolyte_salt"] == "Zn(OTf)2"
    assert data["total_vials"] == 11
    assert "OTf" in data["electrolyte_adjustment_rationale"] or "triflate" in data["electrolyte_adjustment_rationale"].lower()


def test_all_six_salts_scaling_for_all_five_functional_groups():
    """
    Audit cross-salt scaling factors across all 6 canonical ZIB salts
    (ZnSO4, Zn(OTf)2, Zn(ClO4)2, Zn(TFSI)2, ZnCl2, Zn(CH3COO)2)
    for all 5 functional group families:
    1. polyols (D-Sorbitol)
    2. quaternary ammoniums (TMABr)
    3. thiocarbonyls (Thiourea)
    4. amides/ureas (Urea)
    5. sulfoxides (DMSO)
    """
    engine = ConcentrationAnchorEngine()
    test_cases = [
        # (FG name, Molecule name, SMILES, target_salt, conc_mM, expected_factor)
        # Polyol
        ("polyol", "D-Sorbitol", "C(C(C(C(C(CO)O)O)O)O)O", "ZnSO4", 15.0, 1.0),
        ("polyol", "D-Sorbitol", "C(C(C(C(C(CO)O)O)O)O)O", "Zn(OTf)2", 15.0, 0.7),
        ("polyol", "D-Sorbitol", "C(C(C(C(C(CO)O)O)O)O)O", "Zn(TFSI)2", 15.0, 0.6),
        ("polyol", "D-Sorbitol", "C(C(C(C(C(CO)O)O)O)O)O", "ZnCl2", 15.0, 1.2),
        ("polyol", "D-Sorbitol", "C(C(C(C(C(CO)O)O)O)O)O", "Zn(ClO4)2", 15.0, 0.9),
        ("polyol", "D-Sorbitol", "C(C(C(C(C(CO)O)O)O)O)O", "Zn(CH3COO)2", 15.0, 0.9),
        # Quaternary Ammonium
        ("quaternary_ammonium", "TMABr", "C[N+](C)(C)C.[Br-]", "ZnSO4", 10.0, 1.0),
        ("quaternary_ammonium", "TMABr", "C[N+](C)(C)C.[Br-]", "Zn(OTf)2", 10.0, 0.85),
        ("quaternary_ammonium", "TMABr", "C[N+](C)(C)C.[Br-]", "Zn(TFSI)2", 10.0, 0.5),
        ("quaternary_ammonium", "TMABr", "C[N+](C)(C)C.[Br-]", "ZnCl2", 10.0, 1.75),
        ("quaternary_ammonium", "TMABr", "C[N+](C)(C)C.[Br-]", "Zn(ClO4)2", 10.0, 1.0),
        ("quaternary_ammonium", "TMABr", "C[N+](C)(C)C.[Br-]", "Zn(CH3COO)2", 10.0, 1.0),
        # Thiocarbonyl
        ("thiocarbonyl", "Thiourea", "NC(=S)N", "ZnSO4", 10.0, 1.0),
        ("thiocarbonyl", "Thiourea", "NC(=S)N", "Zn(OTf)2", 10.0, 0.8),
        ("thiocarbonyl", "Thiourea", "NC(=S)N", "Zn(TFSI)2", 10.0, 0.7),
        ("thiocarbonyl", "Thiourea", "NC(=S)N", "ZnCl2", 10.0, 1.5),
        ("thiocarbonyl", "Thiourea", "NC(=S)N", "Zn(ClO4)2", 10.0, 1.0),
        ("thiocarbonyl", "Thiourea", "NC(=S)N", "Zn(CH3COO)2", 10.0, 1.0),
        # Amide/Urea
        ("amide_urea", "Urea", "NC(=O)N", "ZnSO4", 20.0, 1.0),
        ("amide_urea", "Urea", "NC(=O)N", "Zn(OTf)2", 20.0, 0.85),
        ("amide_urea", "Urea", "NC(=O)N", "Zn(TFSI)2", 20.0, 0.75),
        ("amide_urea", "Urea", "NC(=O)N", "ZnCl2", 20.0, 1.2),
        ("amide_urea", "Urea", "NC(=O)N", "Zn(ClO4)2", 20.0, 0.9),
        ("amide_urea", "Urea", "NC(=O)N", "Zn(CH3COO)2", 20.0, 1.0),
        # Sulfoxide
        ("sulfoxide", "DMSO", "CS(=O)C", "ZnSO4", 25.0, 1.0),
        ("sulfoxide", "DMSO", "CS(=O)C", "Zn(OTf)2", 25.0, 0.8),
        ("sulfoxide", "DMSO", "CS(=O)C", "Zn(TFSI)2", 25.0, 0.7),
        ("sulfoxide", "DMSO", "CS(=O)C", "ZnCl2", 25.0, 1.2),
        ("sulfoxide", "DMSO", "CS(=O)C", "Zn(ClO4)2", 25.0, 1.0),
        ("sulfoxide", "DMSO", "CS(=O)C", "Zn(CH3COO)2", 25.0, 1.0),
    ]

    for fg, name, smiles, target_salt, conc_mM, expected_factor in test_cases:
        anchor = engine.calculate_anchor(
            additive_name=name,
            published_conc_mM=conc_mM,
            parent_electrolyte="2 M ZnSO4",
            target_electrolyte=target_salt,
            smiles=smiles,
        )
        assert anchor.target_electrolyte_salt == target_salt
        assert anchor.scaling_factor == pytest.approx(expected_factor, abs=0.01), (
            f"Failed for {fg} ({name}) transitioning from 2 M ZnSO4 to {target_salt}: "
            f"expected {expected_factor}, got {anchor.scaling_factor}"
        )
        assert anchor.shifted_anchor_conc == pytest.approx(conc_mM * expected_factor, abs=0.02)


def test_cross_salt_scaling_non_znso4_parents():
    """
    Verify cross-salt scaling when parent electrolyte is NOT ZnSO4:
    - Zn(OTf)2 (demand 0.7) -> ZnSO4 (demand 1.0): factor = 1.0 / 0.7 = 1.429
    - Zn(OTf)2 (demand 0.7) -> ZnCl2 (demand 1.2): factor = 1.2 / 0.7 = 1.714
    - ZnCl2 (demand 1.5) -> Zn(OTf)2 (demand 0.8): factor = 0.8 / 1.5 = 0.533
    - ZnCl2 (demand 1.5) -> ZnSO4 (demand 1.0): factor = 1.0 / 1.5 = 0.667
    """
    engine = ConcentrationAnchorEngine()

    # Polyol: Zn(OTf)2 -> ZnSO4
    anchor_otf_to_so4 = engine.calculate_anchor(
        additive_name="D-Sorbitol",
        published_conc_mM=10.0,
        parent_electrolyte="1 M Zn(OTf)2",
        target_electrolyte="2 M ZnSO4",
        smiles="C(C(C(C(C(CO)O)O)O)O)O",
    )
    assert anchor_otf_to_so4.scaling_factor == pytest.approx(1.429, abs=0.01)
    assert anchor_otf_to_so4.shifted_anchor_conc == pytest.approx(14.29, abs=0.05)
    assert "without Zn(OTf)2 co-ion assistance" in anchor_otf_to_so4.electrolyte_adjustment_rationale

    # Polyol: Zn(OTf)2 -> ZnCl2
    anchor_otf_to_cl = engine.calculate_anchor(
        additive_name="D-Sorbitol",
        published_conc_mM=10.0,
        parent_electrolyte="1 M Zn(OTf)2",
        target_electrolyte="5 M ZnCl2",
        smiles="C(C(C(C(C(CO)O)O)O)O)O",
    )
    assert anchor_otf_to_cl.scaling_factor == pytest.approx(1.714, abs=0.01)
    assert anchor_otf_to_cl.shifted_anchor_conc == pytest.approx(17.14, abs=0.05)
    assert "Cross-salt transition from Zn(OTf)2" in anchor_otf_to_cl.electrolyte_adjustment_rationale

    # Thiocarbonyl: ZnCl2 -> Zn(OTf)2
    anchor_cl_to_otf = engine.calculate_anchor(
        additive_name="Thiourea",
        published_conc_mM=15.0,
        parent_electrolyte="5 M ZnCl2",
        target_electrolyte="1 M Zn(OTf)2",
        smiles="NC(=S)N",
    )
    assert anchor_cl_to_otf.scaling_factor == pytest.approx(0.533, abs=0.01)
    assert anchor_cl_to_otf.shifted_anchor_conc == pytest.approx(8.0, abs=0.05)

    # Thiocarbonyl: ZnCl2 -> ZnSO4
    anchor_cl_to_so4 = engine.calculate_anchor(
        additive_name="Thiourea",
        published_conc_mM=15.0,
        parent_electrolyte="5 M ZnCl2",
        target_electrolyte="2 M ZnSO4",
        smiles="NC(=S)N",
    )
    assert anchor_cl_to_so4.scaling_factor == pytest.approx(0.667, abs=0.01)
    assert anchor_cl_to_so4.shifted_anchor_conc == pytest.approx(10.0, abs=0.05)


def test_wise_zntfsi2_trace_vs_cosolvent_polyol_scaling():
    """
    Test trace (<100 mM) vs co-solvent (>=100 mM) scaling rules in WiSE Zn(TFSI)2:
    - Trace regime: 0.6x multiplier
    - Co-solvent regime: 0.8x multiplier
    """
    engine = ConcentrationAnchorEngine()

    # Trace polyol: 15 mM in 2 M ZnSO4 -> Zn(TFSI)2
    anchor_trace = engine.calculate_anchor(
        additive_name="Glycerol",
        published_conc_mM=15.0,
        parent_electrolyte="2 M ZnSO4",
        target_electrolyte="4 M Zn(TFSI)2",
        smiles="C(C(CO)O)O",
    )
    assert anchor_trace.scaling_factor == pytest.approx(0.6, abs=0.01)
    assert anchor_trace.shifted_anchor_conc == pytest.approx(9.0, abs=0.01)
    assert "Water-in-salt (WiSE)" in anchor_trace.electrolyte_adjustment_rationale

    # Co-solvent polyol: 250 mM in 2 M ZnSO4 -> Zn(TFSI)2
    anchor_cosolvent = engine.calculate_anchor(
        additive_name="Ethylene glycol",
        published_conc_mM=250.0,
        parent_electrolyte="2 M ZnSO4",
        target_electrolyte="4 M Zn(TFSI)2",
        smiles="OCCO",
    )
    assert anchor_cosolvent.scaling_factor == pytest.approx(0.8, abs=0.01)
    assert anchor_cosolvent.shifted_anchor_conc == pytest.approx(200.0, abs=0.01)
    assert "Co-solvent WiSE regime" in anchor_cosolvent.electrolyte_adjustment_rationale


def test_esol_logs_override_and_saturation_cap():
    """
    Test esol_logs_override parameter in calculate_anchor:
    - Verifies that passing pre-calculated logS (e.g. from Phase 1 CandidateMolecule)
      computes saturation limit and caps max_conc at 80% ceiling.
    """
    engine = ConcentrationAnchorEngine()
    # Sparingly soluble hypothetical additive with logS = -3.5 (S ~ 0.316 mM, 80% ceiling ~ 0.253 mM)
    anchor = engine.calculate_anchor(
        additive_name="Hydrophobic-Test",
        published_conc_mM=5.0,
        parent_electrolyte="2 M ZnSO4",
        target_electrolyte="2 M ZnSO4",
        esol_logs_override=-3.5,
    )
    assert anchor.solubility_limit_mM == pytest.approx(0.316, abs=0.01)
    assert anchor.solubility_ceiling_mM == pytest.approx(0.253, abs=0.01)
    assert anchor.is_capped_by_solubility is True
    assert anchor.max_conc == pytest.approx(anchor.solubility_ceiling_mM, abs=0.01)
    assert anchor.shifted_anchor_conc <= anchor.solubility_ceiling_mM


def test_robotic_dispenser_deck_capacity_and_negative_water_alerts():
    """
    Verify liquid handling alerts for:
    1. Deck capacity overflow (> 24 on a 24-vial rack or > 96 on a 96-well plate)
    2. Negative water balance and stock feasibility alert when stock is less concentrated than target.
    """
    anchor_engine = ConcentrationAnchorEngine()
    doe_engine = DoEEngine()
    dispenser = RoboticDispenserEngine()

    anchor = anchor_engine.calculate_anchor(
        additive_name="D-Sorbitol",
        published_conc_mM=15.0,
        parent_electrolyte="2 M ZnSO4",
        target_electrolyte="1 M Zn(OTf)2",
        smiles="C(C(C(C(C(CO)O)O)O)O)O",
    )

    # 1. Plan with 28 samples on a 24-vial rack raises ValueError for deck capacity overflow
    plan_large = doe_engine.create_plan(anchor=anchor, method="lhs", num_samples=28, seed=42)
    with pytest.raises(ValueError, match="Deck capacity overflow: 28 new samples with an offset of 0 exceeds the maximum 24 wells for a 24-vial rack."):
        dispenser.generate_schedule(plan=plan_large, rack_format="24-vial rack")

    # 2. Insufficient salt stock concentration causing negative water balance
    plan_std = doe_engine.create_plan(anchor=anchor, method="lhs", num_samples=5, seed=42)
    # Target Zn(OTf)2 has max conc ~2.5 M, but user provides 0.5 M stock
    with pytest.warns(OversaturatedFormulationWarning) as warn_record:
        sched_bad_stock = dispenser.generate_schedule(
            plan=plan_std,
            rack_format="24-vial rack",
            salt_stock_conc_M=0.5,
        )
    assert len(warn_record) > 0
    assert any("Clamping diluent volume to 0.0" in str(w.message) for w in warn_record)
    assert any("Stock feasibility alert" in w for w in sched_bad_stock.warnings)
    assert any("Negative water balance alert" in w for w in sched_bad_stock.warnings)
    assert all(v.water_uL >= 0.0 for v in sched_bad_stock.vials)



def test_robotic_dispenser_deck_slot_and_flow_rates():
    """
    Test custom OT-2 deck slot assignment and flow rate parameters:
    - Verifies ot2_deck_slot and flow rates propagate to DispensingSchedule and all RoboticVials.
    """
    anchor_engine = ConcentrationAnchorEngine()
    doe_engine = DoEEngine()
    dispenser = RoboticDispenserEngine()

    anchor = anchor_engine.calculate_anchor(
        additive_name="D-Sorbitol",
        published_conc_mM=15.0,
        parent_electrolyte="2 M ZnSO4",
        target_electrolyte="1 M Zn(OTf)2",
        smiles="C(C(C(C(C(CO)O)O)O)O)O",
    )
    plan = doe_engine.create_plan(anchor=anchor, method="lhs", num_samples=6, seed=42)
    sched = dispenser.generate_schedule(
        plan=plan,
        deck_slot=4,
        aspiration_rate_ul_s=15.0,
        dispense_rate_ul_s=20.0,
    )

    assert sched.ot2_deck_slot == 4
    assert sched.default_aspiration_rate_ul_s == 15.0
    assert sched.default_dispense_rate_ul_s == 20.0
    for v in sched.vials:
        assert v.slot == 4
        assert v.aspiration_rate_ul_s == 15.0
        assert v.dispense_rate_ul_s == 20.0


def test_doe_lhs_uniformity_and_reproducibility():
    """
    Test Latin Hypercube Sampling (LHS) stratification uniformity and seed reproducibility.
    """
    anchor_engine = ConcentrationAnchorEngine()
    doe_engine = DoEEngine()

    anchor = anchor_engine.calculate_anchor(
        additive_name="D-Sorbitol",
        published_conc_mM=15.0,
        parent_electrolyte="2 M ZnSO4",
        target_electrolyte="1 M Zn(OTf)2",
        smiles="C(C(C(C(C(CO)O)O)O)O)O",
    )

    plan1 = doe_engine.create_plan(anchor=anchor, method="lhs", num_samples=10, seed=123)
    plan2 = doe_engine.create_plan(anchor=anchor, method="lhs", num_samples=10, seed=123)

    # Identical formulations with same seed
    for f1, f2 in zip(plan1.formulations, plan2.formulations):
        assert f1.additive_conc_mM == f2.additive_conc_mM
        assert f1.salt_conc_M == f2.salt_conc_M

    # Space filling check: all values strictly bounded
    for f in plan1.formulations:
        assert anchor.min_conc <= f.additive_conc_mM <= anchor.max_conc
        assert 0.5 <= f.salt_conc_M <= 2.5


def test_doe_ccd_rotatable_and_face_centered():
    """
    Test Central Composite Design (CCD) modes:
    - Face-centered (alpha=1.0)
    - Orthogonal / rotatable (alpha=sqrt(2))
    Verifies that all axial points remain strictly within the physical parameter bounds.
    """
    anchor_engine = ConcentrationAnchorEngine()
    doe_engine = DoEEngine()

    anchor = anchor_engine.calculate_anchor(
        additive_name="Thiourea",
        published_conc_mM=10.0,
        parent_electrolyte="2 M ZnSO4",
        target_electrolyte="1 M Zn(OTf)2",
        smiles="NC(=S)N",
    )

    plan_face = doe_engine.create_plan(anchor=anchor, method="ccd", alpha_mode="face", ccd_center_points=3)
    assert len(plan_face.formulations) == 11
    for f in plan_face.formulations:
        assert anchor.min_conc <= f.additive_conc_mM <= anchor.max_conc
        assert 0.5 <= f.salt_conc_M <= 2.5

    plan_ortho = doe_engine.create_plan(anchor=anchor, method="ccd", alpha_mode="orthogonal", ccd_center_points=3)
    assert len(plan_ortho.formulations) == 11
    for f in plan_ortho.formulations:
        assert anchor.min_conc <= f.additive_conc_mM <= anchor.max_conc
        assert 0.5 <= f.salt_conc_M <= 2.5


def test_anchor_from_extracted_additive():
    """
    Test direct programmatic Phase 1 -> Phase 2 handoff from an ExtractedAdditive object.
    Verifies automatic extraction of optimal concentration, baseline electrolyte, and SMILES.
    """
    engine = ConcentrationAnchorEngine()
    additive = ExtractedAdditive(
        name="D-Mannitol",
        smiles="C(C(C(C(C(CO)O)O)O)O)O",
        optimal_concentration=Concentration(value=25.0, unit="mM", raw_text="25 mM"),
        concentration_range_tested=[Concentration(value=25.0, unit="mM", raw_text="25 mM")],
        functional_groups=["polyol"],
        primary_mechanisms=["Solvation sheath modulation"],
        battery_context=ZincIonBatteryContext(
            baseline_electrolyte="2 M ZnSO4",
            solvent="H2O",
            anode_material="Zn foil",
            cell_format="Symmetric",
        ),
    )

    anchor = engine.anchor_from_extracted_additive(additive, target_electrolyte="1 M Zn(OTf)2")
    assert anchor.additive_name == "D-Mannitol"
    assert anchor.published_concentration == 25.0
    assert anchor.parent_electrolyte == "2 M ZnSO4"
    assert anchor.target_electrolyte_salt == "Zn(OTf)2"
    assert anchor.scaling_factor == pytest.approx(0.7, abs=0.01)
    assert anchor.shifted_anchor_conc == pytest.approx(17.5, abs=0.01)
    assert "polyol" in anchor.functional_groups


def test_anchor_from_candidate_molecule_with_esol_cap():
    """
    Test programmatic handoff for an AI-recommended CandidateMolecule:
    Verifies that candidate.logs (Delaney ESOL) is utilized as esol_logs_override
    and strictly enforces the 80% saturation cap.
    """
    engine = ConcentrationAnchorEngine()
    candidate = CandidateMolecule(
        name="Hydrophobic-Polymer",
        smiles="CCCC(=O)NC",
        similarity_score=0.82,
        similarity_metric="Tanimoto (Morgan R=2)",
        molecular_weight=101.15,
        logp=0.45,
        logs=-3.0,  # Saturation = 10^(-3.0 + 3.0) = 1.0 mM; 80% ceiling = 0.8 mM
        tpsa=29.1,
        hbd=1,
        hba=1,
        rotatable_bonds=2,
        rationale="High similarity to reference amide.",
        parent_baseline_electrolyte="2 M ZnSO4",
    )

    anchor = engine.anchor_from_candidate_molecule(
        candidate=candidate,
        parent_conc_mM=20.0,
        target_electrolyte="1 M Zn(OTf)2",
    )

    assert anchor.additive_name == "Hydrophobic-Polymer"
    assert anchor.solubility_limit_mM == pytest.approx(1.0, abs=0.01)
    assert anchor.solubility_ceiling_mM == pytest.approx(0.8, abs=0.01)
    assert anchor.is_capped_by_solubility is True
    assert anchor.max_conc == pytest.approx(0.8, abs=0.01)
    assert anchor.shifted_anchor_conc <= 0.8


def test_anchor_from_extraction_result_both_modes():
    """
    Test end-to-end handoff from an ExtractionResult object:
    1. Direct handoff of parent literature additive.
    2. Handoff of AI-recommended CandidateMolecule inheriting parent concentration.
    """
    engine = ConcentrationAnchorEngine()
    extraction = ExtractionResult(
        paper_metadata=PaperMetadata(
            paper_id="paper_001",
            title="Polyol additives in aqueous zinc-ion batteries",
        ),
        additives=[
            ExtractedAdditive(
                name="D-Sorbitol",
                smiles="C(C(C(C(C(CO)O)O)O)O)O",
                optimal_concentration=Concentration(value=20.0, unit="mM", raw_text="20 mM"),
                functional_groups=["polyol"],
                battery_context=ZincIonBatteryContext(
                    baseline_electrolyte="2 M ZnSO4",
                ),
            )
        ],
        candidate_molecules=[
            CandidateMolecule(
                name="Xylitol",
                smiles="C(C(C(CO)O)O)O",
                similarity_score=0.91,
                similarity_metric="Tanimoto (Morgan R=2)",
                molecular_weight=152.15,
                logp=-1.4,
                logs=0.5,
                tpsa=91.15,
                hbd=5,
                hba=5,
                rotatable_bonds=4,
                rationale="Structurally similar C5 polyol.",
                parent_baseline_electrolyte="2 M ZnSO4",
            )
        ],
        extraction_summary="Extracted D-Sorbitol.",
        source_type="pdf",
    )

    # 1. Parent literature additive
    anchor_parent = engine.anchor_from_extraction_result(
        extraction=extraction,
        target_electrolyte="1 M Zn(OTf)2",
    )
    assert anchor_parent.additive_name == "D-Sorbitol"
    assert anchor_parent.published_concentration == 20.0
    assert anchor_parent.shifted_anchor_conc == pytest.approx(14.0, abs=0.01)

    # 2. Recommended candidate molecule
    anchor_cand = engine.anchor_from_extraction_result(
        extraction=extraction,
        target_electrolyte="1 M Zn(OTf)2",
        candidate_index=0,
    )
    assert anchor_cand.additive_name == "Xylitol"
    assert anchor_cand.published_concentration == 20.0
    assert anchor_cand.shifted_anchor_conc == pytest.approx(14.0, abs=0.01)


def test_doe_ccd_three_factors_with_temperature():
    """
    Test 3-factor CCD (Additive conc, Salt conc, Temperature):
    For k=3: 2^3 factorial (8) + 2*3 axial (6) + 3 center points = 17 formulations.
    Verifies that all 17 formulations strictly respect physical factor bounds.
    """
    anchor_engine = ConcentrationAnchorEngine()
    doe_engine = DoEEngine()

    anchor = anchor_engine.calculate_anchor(
        additive_name="D-Sorbitol",
        published_conc_mM=15.0,
        parent_electrolyte="2 M ZnSO4",
        target_electrolyte="1 M Zn(OTf)2",
        smiles="C(C(C(C(C(CO)O)O)O)O)O",
    )

    plan_3factor = doe_engine.create_plan(
        anchor=anchor,
        method="ccd",
        temperature_range=[20.0, 60.0],
        ccd_center_points=3,
        alpha_mode="face",
    )

    assert len(plan_3factor.formulations) == 17
    factorial_pts = [f for f in plan_3factor.formulations if f.point_type == "factorial"]
    axial_pts = [f for f in plan_3factor.formulations if f.point_type == "axial"]
    center_pts = [f for f in plan_3factor.formulations if f.point_type == "center"]

    assert len(factorial_pts) == 8
    assert len(axial_pts) == 6
    assert len(center_pts) == 3

    for f in plan_3factor.formulations:
        assert anchor.min_conc <= f.additive_conc_mM <= anchor.max_conc
        assert 0.5 <= f.salt_conc_M <= 2.5
        assert 20.0 <= f.temperature_C <= 60.0


def test_viscosity_throttled_flow_rates_default():
    """
    Verify automatic 50% flow rate throttling (25 uL/s) for viscous electrolytes:
    - 5 M ZnCl2: throttled to 25 uL/s
    - 4 M Zn(TFSI)2: throttled to 25 uL/s
    - 2 M ZnSO4: standard 50 uL/s
    """
    anchor_engine = ConcentrationAnchorEngine()
    doe_engine = DoEEngine()
    dispenser = RoboticDispenserEngine()

    # 1. Viscous 5 M ZnCl2
    anchor_cl = anchor_engine.calculate_anchor(
        additive_name="Thiourea",
        published_conc_mM=10.0,
        parent_electrolyte="2 M ZnSO4",
        target_electrolyte="5 M ZnCl2",
        smiles="NC(=S)N",
    )
    plan_cl = doe_engine.create_plan(anchor=anchor_cl, method="lhs", num_samples=6)
    sched_cl = dispenser.generate_schedule(plan=plan_cl)
    assert sched_cl.default_aspiration_rate_ul_s == 25.0
    assert sched_cl.default_dispense_rate_ul_s == 25.0
    assert any("High viscosity alert" in w for w in sched_cl.warnings)

    # 2. Viscous 4 M Zn(TFSI)2
    anchor_tfsi = anchor_engine.calculate_anchor(
        additive_name="D-Sorbitol",
        published_conc_mM=15.0,
        parent_electrolyte="2 M ZnSO4",
        target_electrolyte="4 M Zn(TFSI)2",
        smiles="C(C(C(C(C(CO)O)O)O)O)O",
    )
    plan_tfsi = doe_engine.create_plan(anchor=anchor_tfsi, method="lhs", num_samples=6)
    sched_tfsi = dispenser.generate_schedule(plan=plan_tfsi)
    assert sched_tfsi.default_aspiration_rate_ul_s == 25.0
    assert sched_tfsi.default_dispense_rate_ul_s == 25.0
    assert any("High viscosity alert" in w for w in sched_tfsi.warnings)

    # 3. Standard non-viscous 2 M ZnSO4
    anchor_so4 = anchor_engine.calculate_anchor(
        additive_name="D-Sorbitol",
        published_conc_mM=15.0,
        parent_electrolyte="2 M ZnSO4",
        target_electrolyte="2 M ZnSO4",
        smiles="C(C(C(C(C(CO)O)O)O)O)O",
    )
    plan_so4 = doe_engine.create_plan(anchor=anchor_so4, method="lhs", num_samples=6)
    sched_so4 = dispenser.generate_schedule(plan=plan_so4)
    assert sched_so4.default_aspiration_rate_ul_s == 50.0
    assert sched_so4.default_dispense_rate_ul_s == 50.0


def test_planner_cli_candidate_selection_flag(tmp_path, capsys):
    """
    Verify CLI selection of candidate molecules using --candidate-index flag
    when parsing Phase 1 ExtractionResult JSON.
    """
    extraction_payload = {
        "paper_metadata": {
            "paper_id": "paper_test_02",
            "title": "Solvation structure of zinc electrolytes",
        },
        "additives": [
            {
                "name": "D-Mannitol",
                "smiles": "C(C(C(C(C(CO)O)O)O)O)O",
                "optimal_concentration": {"value": 20.0, "unit": "mM", "raw_text": "20 mM"},
                "battery_context": {"baseline_electrolyte": "2 M ZnSO4"},
            }
        ],
        "candidate_molecules": [
            {
                "name": "Erythritol",
                "smiles": "C(C(C(CO)O)O)O",
                "similarity_score": 0.88,
                "similarity_metric": "Tanimoto (Morgan R=2)",
                "molecular_weight": 122.12,
                "logp": -1.2,
                "logs": 0.8,
                "tpsa": 80.9,
                "hbd": 4,
                "hba": 4,
                "rotatable_bonds": 3,
                "rationale": "C4 polyol analog.",
                "parent_baseline_electrolyte": "2 M ZnSO4",
            }
        ],
        "extraction_summary": "Extracted D-Mannitol.",
        "source_type": "pdf",
    }
    json_path = tmp_path / "extraction_result.json"
    json_path.write_text(json.dumps(extraction_payload), encoding="utf-8")

    args = [
        "--from-extraction", str(json_path),
        "--candidate-index", "0",
        "--target-electrolyte", "Zn(OTf)2",
        "--method", "lhs",
        "--samples", "8",
        "--json",
    ]
    code = run_planner(args)
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["target_electrolyte_salt"] == "Zn(OTf)2"
    assert "Erythritol" in data["schedule_id"]
    assert data["total_vials"] == 8


# ==============================================================================
# Phase 3 -> Phase 2 Closed-Loop Adaptive DoE Unit Tests
# ==============================================================================

def test_adaptive_doe_gradient_exploration_and_interpolation():
    """
    Verify Phase 3 -> Phase 2 Active Learning:
    - Given prior results: 0.0, 5.0, and 20.0 mM where 20.0 mM significantly outperforms 0 and 5 mM,
      detects upward gradient.
    - Generates next batch of 2 candidate formulations balancing:
      * Interpolation / refinement: 15.0 mM (probing intermediate gap)
      * Exploitation / ascent: 25.0 mM (probing beyond current best in gradient direction).
    - Preserves sample index continuity (S04, S05).
    - Records iteration_round and prior_observations.
    """
    anchor_engine = ConcentrationAnchorEngine()
    doe_engine = DoEEngine()

    anchor = anchor_engine.calculate_anchor(
        additive_name="D-Sorbitol",
        published_conc_mM=15.0,
        parent_electrolyte="2 M ZnSO4",
        target_electrolyte="1 M Zn(OTf)2",
        smiles="C(C(C(C(C(CO)O)O)O)O)O",
    )
    # Ensure generous solubility ceiling for unconstrained gradient exploration
    anchor.solubility_ceiling_mM = 100.0

    prior_results = [
        FormulatedCellResult(
            sample_id="S01",
            vial_id="A1",
            additive_conc_mM=0.0,
            salt_conc_M=1.0,
            performance_score=40.0,
            iteration_round=0,
        ),
        FormulatedCellResult(
            sample_id="S02",
            vial_id="A2",
            additive_conc_mM=5.0,
            salt_conc_M=1.0,
            performance_score=60.0,
            iteration_round=0,
        ),
        FormulatedCellResult(
            sample_id="S03",
            vial_id="A3",
            additive_conc_mM=20.0,
            salt_conc_M=1.0,
            performance_score=100.0,
            iteration_round=0,
        ),
    ]

    adaptive_plan = doe_engine.generate_adaptive_plan(
        anchor=anchor,
        prior_results=prior_results,
        num_samples=2,
        iteration_round=1,
    )

    assert adaptive_plan.method == "adaptive"
    assert adaptive_plan.iteration_round == 1
    assert len(adaptive_plan.prior_observations) == 3
    assert len(adaptive_plan.formulations) == 2

    concs = [f.additive_conc_mM for f in adaptive_plan.formulations]
    assert concs == [15.0, 25.0]

    # Sample ID sequencing
    assert adaptive_plan.formulations[0].sample_id == "S04"
    assert adaptive_plan.formulations[1].sample_id == "S05"

    # Point types
    assert adaptive_plan.formulations[0].point_type == "adaptive_interpolation"
    assert adaptive_plan.formulations[1].point_type == "adaptive_exploitation"

    # Metadata tracking
    assert adaptive_plan.metadata["best_prior_conc_mM"] == 20.0
    assert adaptive_plan.metadata["best_prior_score"] == 100.0
    assert adaptive_plan.metadata["is_capped_by_solubility"] is False


def test_adaptive_doe_delaney_esol_ceiling_guardrail():
    """
    Verify Delaney ESOL 80% saturation physical guardrail:
    - If upward gradient suggests 25.0 mM, but Delaney ESOL saturation ceiling is 22.0 mM,
      the proposal MUST be capped at 22.0 mM.
    - Attaches explanatory warning alert and sets is_capped_by_solubility flag.
    """
    anchor_engine = ConcentrationAnchorEngine()
    doe_engine = DoEEngine()

    anchor = anchor_engine.calculate_anchor(
        additive_name="Test-Additive",
        published_conc_mM=15.0,
        parent_electrolyte="2 M ZnSO4",
        target_electrolyte="1 M Zn(OTf)2",
    )
    # Impose Delaney ESOL saturation ceiling of 22.0 mM
    anchor.solubility_limit_mM = 27.5
    anchor.solubility_ceiling_mM = 22.0
    anchor.is_capped_by_solubility = False

    prior_results = [
        FormulatedCellResult(
            sample_id="S01",
            vial_id="A1",
            additive_conc_mM=0.0,
            salt_conc_M=1.0,
            performance_score=40.0,
        ),
        FormulatedCellResult(
            sample_id="S02",
            vial_id="A2",
            additive_conc_mM=5.0,
            salt_conc_M=1.0,
            performance_score=60.0,
        ),
        FormulatedCellResult(
            sample_id="S03",
            vial_id="A3",
            additive_conc_mM=20.0,
            salt_conc_M=1.0,
            performance_score=100.0,
        ),
    ]

    adaptive_plan = doe_engine.generate_adaptive_plan(
        anchor=anchor,
        prior_results=prior_results,
        num_samples=2,
        iteration_round=1,
    )

    concs = [f.additive_conc_mM for f in adaptive_plan.formulations]
    assert concs[0] == 15.0
    assert concs[1] == 22.0  # Capped at ceiling rather than 25.0 mM!

    assert adaptive_plan.metadata["is_capped_by_solubility"] is True
    assert anchor.is_capped_by_solubility is True

    warnings = adaptive_plan.metadata["warnings"]
    assert any("Delaney ESOL" in w and "capped at ceiling" in w for w in warnings)
    assert any("22.00 mM" in w for w in warnings)


def test_adaptive_doe_downward_gradient_and_interior_peak():
    """
    Test adaptive DoE handling for:
    1. Downward gradient: lowest tested conc (e.g. 5 mM) is best -> probes lower (exploitation) and interpolates.
    2. Interior peak: middle tested conc (e.g. 10 mM) is best -> interpolates adjacent gaps.
    """
    anchor_engine = ConcentrationAnchorEngine()
    doe_engine = DoEEngine()

    anchor = anchor_engine.calculate_anchor(
        additive_name="Thiourea",
        published_conc_mM=10.0,
        parent_electrolyte="2 M ZnSO4",
        target_electrolyte="1 M Zn(OTf)2",
    )
    anchor.solubility_ceiling_mM = 100.0

    # 1. Downward gradient
    prior_downward = [
        FormulatedCellResult(sample_id="S01", vial_id="A1", additive_conc_mM=5.0, salt_conc_M=1.0, performance_score=95.0),
        FormulatedCellResult(sample_id="S02", vial_id="A2", additive_conc_mM=15.0, salt_conc_M=1.0, performance_score=60.0),
        FormulatedCellResult(sample_id="S03", vial_id="A3", additive_conc_mM=30.0, salt_conc_M=1.0, performance_score=30.0),
    ]
    plan_down = doe_engine.generate_adaptive_plan(anchor=anchor, prior_results=prior_downward, num_samples=2)
    down_concs = [f.additive_conc_mM for f in plan_down.formulations]
    assert down_concs[0] < 5.0  # Explores lower concentration
    assert 5.0 < down_concs[1] < 15.0  # Interpolates gap between 5 and 15

    # 2. Interior peak
    prior_interior = [
        FormulatedCellResult(sample_id="S01", vial_id="A1", additive_conc_mM=0.0, salt_conc_M=1.0, performance_score=40.0),
        FormulatedCellResult(sample_id="S02", vial_id="A2", additive_conc_mM=10.0, salt_conc_M=1.0, performance_score=95.0),
        FormulatedCellResult(sample_id="S03", vial_id="A3", additive_conc_mM=25.0, salt_conc_M=1.0, performance_score=50.0),
    ]
    plan_int = doe_engine.generate_adaptive_plan(anchor=anchor, prior_results=prior_interior, num_samples=2)
    int_concs = [f.additive_conc_mM for f in plan_int.formulations]
    assert 0.0 < int_concs[0] < 10.0  # Interpolation in lower gap
    assert 10.0 < int_concs[1] < 25.0  # Interpolation in upper gap


def test_robotic_dispenser_vial_offset_and_continuity():
    """
    Verify robotic dispenser rack coordinate continuity across iterations:
    - offset=3, count=2 allocates A4 and A5 without collision.
    - offset=6, count=2 allocates B1 and B2.
    - Passing existing_vials automatically detects offset and schedules next empty rack positions.
    - Deck capacity alert triggers when total rack allocation exceeds rack format limit.
    """
    dispenser = RoboticDispenserEngine()

    grid_offset3 = dispenser.generate_coordinate_grid(rack_format="24-vial rack", count=2, offset=3)
    assert len(grid_offset3) == 2
    assert grid_offset3[0] == ("A4", "A", 4)
    assert grid_offset3[1] == ("A5", "A", 5)

    grid_offset6 = dispenser.generate_coordinate_grid(rack_format="24-vial rack", count=2, offset=6)
    assert len(grid_offset6) == 2
    assert grid_offset6[0] == ("B1", "B", 1)
    assert grid_offset6[1] == ("B2", "B", 2)

    # Schedule generation with existing vials
    anchor_engine = ConcentrationAnchorEngine()
    doe = DoEEngine()
    anchor = anchor_engine.calculate_anchor(
        additive_name="D-Sorbitol",
        published_conc_mM=15.0,
        parent_electrolyte="2 M ZnSO4",
        target_electrolyte="1 M Zn(OTf)2",
    )
    plan_round0 = doe.create_plan(anchor=anchor, num_samples=3)
    sched_round0 = dispenser.generate_schedule(plan=plan_round0)
    assert len(sched_round0.vials) == 3
    assert [v.vial_id for v in sched_round0.vials] == ["A1", "A2", "A3"]
    assert [f.coordinate for f in plan_round0.formulations] == ["A1", "A2", "A3"]

    # Round 1 schedule passing existing vials
    plan_round1 = FormulationPoint(sample_id="S04", additive_conc_mM=15.0, salt_conc_M=1.0, point_type="adaptive")
    plan1 = DoEPlan(
        plan_id="DOE_ADAPTIVE_R1",
        additive_name="D-Sorbitol",
        target_electrolyte_salt="Zn(OTf)2",
        target_salt_default_molarity=1.0,
        target_salt_molarity_range=[0.5, 2.5],
        method="adaptive",
        anchor=anchor,
        parameters=plan_round0.parameters,
        formulations=[
            FormulationPoint(sample_id="S04", additive_conc_mM=15.0, salt_conc_M=1.0, point_type="adaptive_interpolation"),
            FormulationPoint(sample_id="S05", additive_conc_mM=25.0, salt_conc_M=1.0, point_type="adaptive_exploitation"),
        ],
        electrolyte_adjustment_rationale="Adaptive round 1",
        iteration_round=1,
    )
    sched_round1 = dispenser.generate_schedule(plan=plan1, existing_vials=sched_round0.vials)
    assert len(sched_round1.vials) == 2
    assert [v.vial_id for v in sched_round1.vials] == ["A4", "A5"]
    assert [f.coordinate for f in plan1.formulations] == ["A4", "A5"]
    assert sched_round1.iteration_round == 1

    # Deck capacity overflow detection
    with pytest.raises(ValueError, match="Deck capacity overflow: 2 new samples with an offset of 23 exceeds the maximum 24 wells for a 24-vial rack."):
        dispenser.generate_schedule(
            plan=plan1,
            rack_format="24-vial rack",
            initial_vial_offset=23,
        )

    # Unsupported container format detection
    with pytest.raises(ValueError, match="Unsupported container format: 48-well plate"):
        dispenser.generate_schedule(
            plan=plan1,
            rack_format="48-well plate",
        )


def test_fetch_pubchem_data_caching():
    """Verify that fetch_pubchem_data caches results with LRU cache."""
    fetch_pubchem_data.cache_clear()
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"Record": {"RecordNumber": 5780}}
        mock_get.return_value = mock_response

        res1 = fetch_pubchem_data("D-Sorbitol")
        res2 = fetch_pubchem_data("D-Sorbitol")

        assert res1 == {"Record": {"RecordNumber": 5780}}
        assert res2 == res1
        # Due to @lru_cache, requests.get is called only once
        assert mock_get.call_count == 1


def test_fetch_pubchem_data_exponential_backoff_retry():
    """Verify that HTTP 429 triggers exponential backoff and retries."""
    fetch_pubchem_data.cache_clear()
    with patch("requests.get") as mock_get, patch("time.sleep") as mock_sleep:
        resp_429 = MagicMock()
        resp_429.status_code = 429

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {"success": True}

        # First two calls return 429, third returns 200
        mock_get.side_effect = [resp_429, resp_429, resp_200]

        res = fetch_pubchem_data("Thiourea")
        assert res == {"success": True}
        assert mock_get.call_count == 3
        # Backoff delays: 1.0 * (2^0) = 1.0, 1.0 * (2^1) = 2.0
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)


def test_fetch_pubchem_data_persistent_failure_raises():
    """Verify that persistent 429/503 raises ConnectionError after max_retries."""
    fetch_pubchem_data.cache_clear()
    with patch("requests.get") as mock_get, patch("time.sleep"):
        resp_503 = MagicMock()
        resp_503.status_code = 503
        mock_get.return_value = resp_503

        with pytest.raises(ConnectionError) as exc_info:
            fetch_pubchem_data("UnreachableAdditive")

        assert "persistent after 4 retries" in str(exc_info.value)
        assert mock_get.call_count == 4


def test_relative_cross_salt_scaling_logic():
    """Verify relative cross-salt demand ratio scaling calculation."""
    # ZnCl2 (1.2) -> Zn(TFSI)2 (0.6): relative_ratio = 0.6 / 1.2 = 0.5
    conc_scaled = calculate_target_concentration(
        parent_concentration=15.0,
        parent_salt="ZnCl2",
        target_salt="Zn(TFSI)2",
        functional_group_multiplier=1.0,
    )
    assert conc_scaled == 7.5

    # ZnSO4 (1.0) -> Zn(OTf)2 (0.7): relative_ratio = 0.7 / 1.0 = 0.7
    conc_otf = scale_additive(
        parent_concentration=20.0,
        parent_salt="ZnSO4",
        target_salt="Zn(OTf)2",
        functional_group_multiplier=1.0,
    )
    assert conc_otf == 14.0

    # With functional group multiplier (e.g., 1.2)
    conc_mult = calculate_target_concentration(
        parent_concentration=10.0,
        parent_salt="ZnCl2",
        target_salt="Zn(TFSI)2",
        functional_group_multiplier=1.2,
    )
    # 10.0 * 0.5 * 1.2 = 6.0
    assert conc_mult == 6.0




