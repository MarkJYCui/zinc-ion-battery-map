"""
Integration and End-to-End Pipeline Tests for the Autonomous Materials Acceleration Platform (MAP).
Tests sequential workflow across:
Phase 1 (Discovery) ➔ Phase 2 (Formulation & Robotic Dispensing) ➔ Phase 3 (Electrochemical Analysis & Reporting)

Verifies:
1. Complete Phase 1 -> Phase 2 -> Phase 3 sequential data flow with zero data loss across model transitions.
2. Phase 2 correctly scales anchor concentrations based on Phase 1 candidate properties.
3. Enforces Delaney ESOL solubility ceilings (80% saturation cap).
4. DoE sampling (LHS / CCD) and Robotic liquid handling schedules (OT-2 coordinates, aspiration/dispense rates).
5. Phase 3 correctly ingests and analyzes formulated cell runs (Approach C failure-aware dual-tier metrics).
6. Full campaign orchestrator (MAPOrchestrator) and top-level CLI (main.py).
7. Error handling and validation across all boundaries.
"""

import json
import subprocess
import sys
from pathlib import Path
import pytest
import numpy as np


from models import (
    ExtractionResult,
    CandidateMolecule,
    ExtractedAdditive,
    Concentration,
    ZincIonBatteryContext,
    ConcentrationAnchor,
    DoEPlan,
    DispensingSchedule,
    ElectrochemicalAnalysisReport,
    CampaignResult,
    FormulatedCellResult,
)
from optimization.concentration_anchor import ConcentrationAnchorEngine
from optimization.doe_engine import DoEEngine
from optimization.dispenser import RoboticDispenserEngine
from analysis.analyzer_cli import run_analysis
from map_orchestrator import MAPOrchestrator, create_benchmark_cycler_file


# ==============================================================================
# 1. Phase 1 -> Phase 2 Hand-off Tests
# ==============================================================================

def test_phase1_to_phase2_handoff_with_extraction_result(tmp_path):
    """
    Verify hand-off from Phase 1 ExtractionResult JSON to Phase 2 formulation:
    - Loads published D-Sorbitol (15 mM in 2 M ZnSO4).
    - Shifts to 1 M Zn(OTf)2 (0.7x scaling factor -> 10.5 mM anchor).
    - Exploration bounds [2.1 mM, 31.5 mM].
    - Delaney ESOL ceiling computed.
    - Generates 9-sample LHS DoE plan.
    - Generates 24-vial dispensing schedule with OT-2 coordinates and flow rates.
    - Asserts zero data loss in model transitions.
    """
    extraction_file = Path("samples/test_extraction.json")
    assert extraction_file.exists(), "samples/test_extraction.json must exist"

    with open(extraction_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    extraction = ExtractionResult.model_validate(data)
    assert len(extraction.additives) > 0
    additive = extraction.additives[0]

    orchestrator = MAPOrchestrator(output_dir=tmp_path)
    plan, schedule = orchestrator.run_formulation(
        extraction=extraction,
        target_salt="Zn(OTf)2",
        method="lhs",
        num_samples=9,
        rack_format="24-vial rack",
        total_volume_uL=2000.0,
        seed=42,
    )

    # Verify ConcentrationAnchor calculations
    anchor = plan.anchor
    assert anchor.additive_name == "D-Sorbitol"
    assert anchor.published_concentration == 15.0
    assert anchor.parent_salt == "ZnSO4"
    assert anchor.target_electrolyte_salt == "Zn(OTf)2"
    assert anchor.scaling_factor == 0.7
    assert anchor.shifted_anchor_conc == pytest.approx(10.5, abs=0.01)
    assert anchor.min_conc == pytest.approx(2.1, abs=0.01)
    assert anchor.max_conc == pytest.approx(31.5, abs=0.01)
    assert "polyol" in anchor.functional_groups
    assert "Synergistic" in anchor.electrolyte_adjustment_rationale

    # Delaney ESOL ceiling
    assert anchor.solubility_ceiling_mM is not None
    assert anchor.solubility_ceiling_mM > 0
    assert not anchor.is_capped_by_solubility

    # Verify DoE Plan
    assert plan.method == "lhs"
    assert len(plan.formulations) == 9
    for pt in plan.formulations:
        assert 2.1 <= pt.additive_conc_mM <= 31.5
        assert 0.5 <= pt.salt_conc_M <= 2.5

    # Verify Robotic Dispensing Schedule & OT-2 details
    assert schedule.target_electrolyte_salt == "Zn(OTf)2"
    assert schedule.total_vials == 9
    assert schedule.total_volume_per_vial_uL == 2000.0
    assert schedule.ot2_deck_slot == 1
    assert schedule.default_aspiration_rate_ul_s == 50.0
    assert schedule.default_dispense_rate_ul_s == 50.0

    vial_ids = [v.vial_id for v in schedule.vials]
    assert vial_ids[:6] == ["A1", "A2", "A3", "A4", "A5", "A6"]
    assert vial_ids[6:9] == ["B1", "B2", "B3"]

    for vial in schedule.vials:
        total_vol = vial.additive_stock_uL + vial.salt_stock_uL + vial.water_uL
        assert total_vol == pytest.approx(2000.0, abs=0.01)
        assert vial.slot == 1
        assert vial.aspiration_rate_ul_s == 50.0
        assert vial.dispense_rate_ul_s == 50.0


def test_phase1_to_phase2_candidate_molecule_solubility_ceiling_cap(tmp_path):
    """
    Verify Delaney ESOL 80% saturation solubility ceiling enforcement for a candidate:
    - When unconstrained upper exploration bound exceeds Delaney ESOL 80% ceiling,
      is_capped_by_solubility must be True, and max_conc must be truncated to ceiling.
    """
    # Sparingly soluble aromatic candidate: 4-phenylphenol (MW ~170, high logP ~3.2, low logS)
    candidate = CandidateMolecule(
        name="4-Phenylphenol",
        smiles="Oc1ccc(-c2ccccc2)cc1",
        similarity_score=0.45,
        molecular_weight=170.21,
        logp=3.20,
        logs=-3.1,
        tpsa=20.23,
        hbd=1,
        hba=1,
        rotatable_bonds=1,
        rationale="Hydrophobic interphase candidate",
        sdl_formulation_notes="Check solubility limits",
        parent_baseline_electrolyte="2 M ZnSO4",
    )

    orchestrator = MAPOrchestrator(output_dir=tmp_path)
    # High baseline conc (e.g. 10.0 mM) whose 3.0x upper bound (30 mM) exceeds solubility ceiling
    plan, schedule = orchestrator.run_formulation(
        extraction=candidate,
        target_salt="ZnSO4",
        method="lhs",
        num_samples=6,
        published_conc_override=10.0,
    )

    anchor = plan.anchor
    assert anchor.solubility_ceiling_mM is not None
    # For 4-phenylphenol, Delaney ESOL ceiling is typically ~1-3 mM
    assert anchor.is_capped_by_solubility is True
    assert anchor.max_conc == pytest.approx(anchor.solubility_ceiling_mM, abs=0.01)
    assert anchor.shifted_anchor_conc <= anchor.solubility_ceiling_mM
    assert "capped at Delaney ESOL 80% saturation" in anchor.electrolyte_adjustment_rationale

    # All DoE formulations must respect the solubility ceiling
    for pt in plan.formulations:
        assert pt.additive_conc_mM <= anchor.solubility_ceiling_mM + 1e-4


def test_phase2_viscosity_rate_reduction():
    """
    Verify that high-viscosity salts (e.g. 5 M ZnCl2 or 4 M Zn(TFSI)2)
    trigger 50% slow aspiration/dispense rates (25 uL/s) on OT-2 liquid handler.
    """
    anchor_engine = ConcentrationAnchorEngine()
    anchor = anchor_engine.calculate_anchor(
        additive_name="Thiourea",
        published_conc_mM=10.0,
        parent_electrolyte="2 M ZnSO4",
        target_electrolyte="5 M ZnCl2",
        smiles="NC(=S)N",
    )

    doe = DoEEngine()
    plan = doe.create_plan(anchor=anchor, method="lhs", num_samples=6)

    dispenser = RoboticDispenserEngine()
    schedule = dispenser.generate_schedule(plan=plan, rack_format="24-vial rack")

    assert schedule.target_electrolyte_salt == "ZnCl2"
    assert schedule.default_aspiration_rate_ul_s == 25.0
    assert schedule.default_dispense_rate_ul_s == 25.0
    for vial in schedule.vials:
        assert vial.aspiration_rate_ul_s == 25.0
        assert vial.dispense_rate_ul_s == 25.0

    # High viscosity warning must be present
    assert any("High viscosity alert" in w for w in schedule.warnings)


# ==============================================================================
# 2. Phase 2 -> Phase 3 Hand-off & Failure-Aware Cycling Tests
# ==============================================================================

def test_phase2_to_phase3_handoff_with_active_cycling_windowing(tmp_path):
    """
    Verify Phase 2 -> Phase 3 handoff with simulated cycling data:
    - Formulated cell run with 100 hours of cycling and short-circuit failure at 60 h.
    - Rest-period stripping & normalization.
    - Approach C dual-tier metrics:
      * active_cycling_metrics: reflects healthy cycling (< 60 h, ~35 mV overpotential)
      * total_test_summary: nominal duration 100 h, short-circuit detected, failure time 60 h.
    - Output of both JSON report and interactive Plotly HTML dashboard.
    """
    csv_path = tmp_path / "cell_A1_cycling.csv"
    create_benchmark_cycler_file(
        filepath=csv_path,
        duration_hours=100.0,
        short_at_hour=60.0,
        plateau_voltage_v=0.035,
        current_ma=1.131,
    )
    assert csv_path.exists()

    orchestrator = MAPOrchestrator(output_dir=tmp_path)
    report = orchestrator.run_analysis(
        data_source=csv_path,
        technique="galvanostatic_symmetric",
        area_cm2=1.13097,
        sample_id="A1",
    )

    assert report.sample_id == "A1"
    assert report.technique == "galvanostatic_symmetric"
    assert report.symmetric_cell is not None
    symm = report.symmetric_cell

    # Verify Approach C failure-aware dual-tier metrics
    assert symm.short_circuit_detected is True
    assert symm.failure_time_hours == pytest.approx(60.0, abs=2.5)

    assert symm.active_cycling_metrics is not None
    active = symm.active_cycling_metrics
    assert active["active_lifespan_hours"] == pytest.approx(60.0, abs=2.5)
    assert active["mean_overpotential_mv"] == pytest.approx(35.0, abs=3.0)
    assert active["mean_hysteresis_mv"] == pytest.approx(70.0, abs=6.0)

    assert symm.total_test_summary is not None
    total = symm.total_test_summary
    assert total["nominal_duration_hours"] == pytest.approx(100.0, abs=1.0)
    assert total["short_circuit_detected"] is True
    assert total["failure_time_hours"] == pytest.approx(60.0, abs=2.5)

    # HTML visualization dashboard must exist
    assert report.visualization_html_path is not None
    html_file = Path(report.visualization_html_path)
    assert html_file.exists()
    html_content = html_file.read_text(encoding="utf-8")
    assert "ZIB Characterization Report" in html_content
    assert "Failure" in html_content or "Short Circuit" in html_content


# ==============================================================================
# 3. Full End-to-End Campaign Orchestration Tests
# ==============================================================================

def test_full_campaign_end_to_end_orchestration(tmp_path):
    """
    Verify complete MAPOrchestrator.run_full_campaign():
    - Loads extraction from samples/test_extraction.json.
    - Shifts to 1 M Zn(OTf)2.
    - Runs 5-sample LHS DoE.
    - Generates dispensing schedule.
    - Runs benchmark cycler simulations across formulation points.
    - Generates unified CampaignResult, DoE plan JSON, Dispensing JSON, and HTML reports.
    - Evaluates and identifies best candidate.
    """
    output_dir = tmp_path / "campaign_001"
    orchestrator = MAPOrchestrator(output_dir=output_dir)

    result = orchestrator.run_full_campaign(
        from_extraction="samples/test_extraction.json",
        target_salt="Zn(OTf)2",
        doe_method="lhs",
        num_samples=5,
        use_benchmark=True,
        seed=42,
    )

    assert isinstance(result, CampaignResult)
    assert result.target_salt == "Zn(OTf)2"
    assert result.formulation_plan is not None
    assert len(result.formulation_plan.formulations) == 5
    assert result.dispensing_schedule is not None
    assert len(result.dispensing_schedule.vials) == 5
    assert len(result.cell_results) == 5

    # Check disk artifacts
    assert (output_dir / "campaign_summary.json").exists()
    assert (output_dir / "doe_plan.json").exists()
    assert (output_dir / "dispensing_schedule.json").exists()

    # Phase 2 -> Phase 3 sample tracking & data integrity assertion
    for i, cell in enumerate(result.cell_results):
        form_pt = result.formulation_plan.formulations[i]
        vial = result.dispensing_schedule.vials[i]
        assert cell.sample_id == form_pt.sample_id
        assert cell.vial_id == vial.vial_id
        assert cell.point_type == form_pt.point_type
        assert cell.rack_format == vial.rack_format
        assert cell.row == vial.row
        assert cell.col == vial.col
        assert cell.slot == vial.slot
        assert cell.aspiration_rate_ul_s == vial.aspiration_rate_ul_s
        assert cell.dispense_rate_ul_s == vial.dispense_rate_ul_s
        assert cell.additive_conc_mM == pytest.approx(vial.additive_conc_mM)
        assert cell.salt_conc_M == pytest.approx(vial.salt_conc_M)
        assert cell.target_volume_uL == vial.target_volume_uL
        assert cell.status == "completed"
        assert cell.analysis_report is not None

        # Verify summary_metrics and processed_data metadata tracking
        rep = cell.analysis_report
        assert rep.summary_metrics["sample_id"] == cell.sample_id
        assert rep.summary_metrics["vial_id"] == cell.vial_id
        assert rep.summary_metrics["point_type"] == cell.point_type
        assert rep.summary_metrics["slot"] == vial.slot
        assert rep.processed_data.metadata["sample_id"] == cell.sample_id

    # Pareto-optimal candidate identification
    assert len(result.pareto_optimal_candidates) > 0
    for p_cand in result.pareto_optimal_candidates:
        assert p_cand.is_pareto_optimal is True

    # Best candidate selection from Pareto frontier
    assert result.best_candidate is not None
    assert result.best_candidate in result.pareto_optimal_candidates
    assert result.best_candidate.sample_id in [f.sample_id for f in result.formulation_plan.formulations]
    assert result.best_candidate.analysis_report is not None
    assert result.best_candidate.performance_score >= max(c.performance_score for c in result.cell_results)


def test_full_campaign_with_live_query_discovery(tmp_path):
    """
    Verify complete closed-loop Phase 1 -> Phase 2 -> Phase 3 starting from a natural-language query:
    - Runs Phase 1 discovery with query="sorbitol".
    - Flows directly into Phase 2 formulation and robotic dispensing.
    - Flows directly into Phase 3 characterization, failure-aware windowing, and Pareto selection.
    - Asserts full campaign completion with Pareto frontier candidate identified.
    """
    output_dir = tmp_path / "campaign_query_live"
    orchestrator = MAPOrchestrator(output_dir=output_dir)

    result = orchestrator.run_full_campaign(
        query="sorbitol",
        target_salt="Zn(OTf)2",
        doe_method="lhs",
        num_samples=5,
        use_benchmark=True,
        seed=42,
    )

    assert isinstance(result, CampaignResult)
    assert result.target_salt == "Zn(OTf)2"
    assert len(result.cell_results) == 5
    assert result.best_candidate is not None
    assert result.best_candidate.is_pareto_optimal is True
    assert (output_dir / "campaign_summary.json").exists()


def test_main_cli_run_pipeline_execution(tmp_path):
    """
    Verify top-level CLI main.py run-pipeline execution via subprocess:
    python main.py run-pipeline --from-extraction samples/test_extraction.json
                                --target-salt "Zn(OTf)2"
                                --samples 4
                                --use-benchmark
                                --output-dir <tmp_path>
    """
    cmd = [
        sys.executable,
        "main.py",
        "run-pipeline",
        "--from-extraction",
        "samples/test_extraction.json",
        "--target-salt",
        "Zn(OTf)2",
        "--samples",
        "4",
        "--use-benchmark",
        "--output-dir",
        str(tmp_path / "cli_run"),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI failed with error:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    summary_file = tmp_path / "cli_run" / "campaign_summary.json"
    assert summary_file.exists()
    with open(summary_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["target_salt"] == "Zn(OTf)2"
    assert len(data["cell_results"]) == 4


# ==============================================================================
# 4. Error Handling and Edge Case Tests
# ==============================================================================

def test_pipeline_error_handling_invalid_inputs(tmp_path):
    """
    Verify proper exception raising for missing or invalid inputs:
    - Non-existent extraction file
    - Missing additive and published conc
    - Invalid DoE method
    """
    orchestrator = MAPOrchestrator(output_dir=tmp_path)

    # Missing extraction file
    with pytest.raises(FileNotFoundError):
        orchestrator.run_discovery(from_extraction="non_existent_file.json")

    # Missing required inputs
    with pytest.raises(ValueError, match="Either provide extraction data"):
        orchestrator.run_full_campaign()

    # Invalid DoE method
    with pytest.raises(ValueError, match="Unsupported DoE method"):
        orchestrator.run_formulation(
            extraction={"name": "Sorbitol", "optimal_concentration": {"value": 15.0}},
            target_salt="ZnSO4",
            method="invalid_method",
        )


# ==============================================================================
# 5. Closed-Loop Adaptive Iteration Integration Tests
# ==============================================================================

def test_closed_loop_two_iteration_campaign(tmp_path):
    """
    Verify complete autonomous closed-loop campaign:
    Round 0 (Initial 3 DoE formulations A1-A3) ➔ Phase 3 Evaluation
    ➔ Feedback into Phase 2 Adaptive Sequential DoE (proposing 2 new formulations A4-A5)
    ➔ Phase 3 Evaluation ➔ Global Pareto Frontier selection across all 5 evaluated cells.
    """
    output_dir = tmp_path / "closed_loop_two_iter"
    orchestrator = MAPOrchestrator(output_dir=output_dir)

    campaign = orchestrator.run_closed_loop_campaign(
        from_extraction="samples/test_extraction.json",
        target_salt="Zn(OTf)2",
        initial_samples=3,
        max_iterations=2,
        samples_per_iteration=2,
        use_benchmark=True,
        seed=42,
    )

    # 1. Verification of iterative rounds
    assert campaign.total_iterations == 2
    assert campaign.iteration_round == 1

    # 2. Total aggregated formulations & vials (3 from R0 + 2 from R1 = 5)
    assert len(campaign.cell_results) == 5
    assert len(campaign.dispensing_schedule.vials) == 5
    assert campaign.dispensing_schedule.total_vials == 5

    # 3. Robotic Dispensing continuity without coordinate collision
    expected_vials = ["A1", "A2", "A3", "A4", "A5"]
    assert [v.vial_id for v in campaign.dispensing_schedule.vials] == expected_vials
    assert [c.vial_id for c in campaign.cell_results] == expected_vials

    # 4. Sample IDs sequence across iterations
    expected_sample_ids = ["S01", "S02", "S03", "S04", "S05"]
    assert [c.sample_id for c in campaign.cell_results] == expected_sample_ids

    # 5. Iteration round tagging on cell results
    for c in campaign.cell_results[:3]:
        assert c.iteration_round == 0
    for c in campaign.cell_results[3:]:
        assert c.iteration_round == 1

    # 6. Global Pareto frontier evaluation
    assert len(campaign.pareto_optimal_candidates) > 0
    for cand in campaign.pareto_optimal_candidates:
        assert cand.is_pareto_optimal is True

    # Best candidate chosen from global Pareto frontier
    assert campaign.best_candidate is not None
    assert campaign.best_candidate in campaign.pareto_optimal_candidates
    max_score = max(c.performance_score for c in campaign.cell_results)
    assert campaign.best_candidate.performance_score == max_score

    # 7. Campaign history tracking
    assert len(campaign.history) == 2
    assert campaign.history[0]["iteration_round"] == 0
    assert campaign.history[0]["new_samples"] == 3
    assert campaign.history[1]["iteration_round"] == 1
    assert campaign.history[1]["new_samples"] == 2

    # 8. Disk artifacts for both rounds
    assert (output_dir / "campaign_summary.json").exists()
    assert (output_dir / "doe_plan.json").exists()
    assert (output_dir / "dispensing_schedule.json").exists()
    assert (output_dir / "doe_plan_round_1.json").exists()
    assert (output_dir / "dispensing_schedule_round_1.json").exists()


def test_main_cli_closed_loop_campaign_execution(tmp_path):
    """
    Verify top-level CLI execution for closed-loop active learning with --iterations 2:
    python main.py run-pipeline --from-extraction samples/test_extraction.json
                                --target-salt "Zn(OTf)2"
                                --samples 3
                                --iterations 2
                                --samples-per-iteration 2
                                --use-benchmark
                                --output-dir <tmp_path>
    """
    output_dir = tmp_path / "cli_closed_loop"
    cmd = [
        sys.executable,
        "main.py",
        "run-pipeline",
        "--from-extraction",
        "samples/test_extraction.json",
        "--target-salt",
        "Zn(OTf)2",
        "--samples",
        "3",
        "--iterations",
        "2",
        "--samples-per-iteration",
        "2",
        "--use-benchmark",
        "--output-dir",
        str(output_dir),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI closed-loop failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    summary_file = output_dir / "campaign_summary.json"
    assert summary_file.exists()
    with open(summary_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["target_salt"] == "Zn(OTf)2"
    assert data["total_iterations"] == 2
    assert len(data["cell_results"]) == 5
    assert len(data["history"]) == 2
    assert data["best_candidate"] is not None
    assert (output_dir / "doe_plan_round_1.json").exists()
    assert (output_dir / "dispensing_schedule_round_1.json").exists()

