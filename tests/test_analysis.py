"""
Unit and Integration Tests for Phase 3: Electrochemical Analysis & Ingestion Engine.
Covers:
1. Parsers (ASCII BioLogic/Landt/Arbin, Excel, MPR binary, format detection).
2. Processing (TimeScaler, RestDetector, MathEngine, ElectrochemicalNormalizer).
3. Electrochemical Techniques (CV, LSV, EIS, Asymmetric Half-Cell, Full Cell).
4. Adaptive Failure-Aware Cycling Windowing & Dual-Tier Metric Aggregation (Symmetric Cells).
5. Interactive Dual-Range Visualization (Approach C in Plotly Dashboard: range slider, buttons, failure line).
6. Manual Windowing & CLI Overrides (--max-time-hours, --max-cycles, --time-window, --trim-post-failure).
7. Automated Watchdog Ingestion Pipeline.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from models import (
    ElectrochemicalTechnique,
    ElectrodeGeometry,
    CyclerRawData,
    NormalizedElectrochemicalData,
    SymmetricCellAnalysis,
    AsymmetricHalfCellAnalysis,
    FullCellAnalysis,
    LSVAnalysis,
    EISAnalysis,
    CVAnalysis,
    ElectrochemicalAnalysisReport,
    SymmetricCellCycleMetric,
)
from analysis.parsers import (
    AsciiCyclerParser,
    ExcelCyclerParser,
    MprCyclerParser,
    parse_cycler_file,
)
from analysis.processing.time_scaler import TimeScaler
from analysis.processing.rest_detector import RestDetector
from analysis.processing.math_engine import MathEngine
from analysis.processing.normalizer import ElectrochemicalNormalizer
from analysis.techniques.galvanostatic import (
    SymmetricCellAnalyzer,
    AsymmetricCellAnalyzer,
    FullCellAnalyzer,
)
from analysis.techniques.lsv import LSVAnalyzer
from analysis.techniques.eis import EISAnalyzer
from analysis.techniques.cv import CVAnalyzer
from analysis.visualization.plot_engine import ElectrochemicalPlotEngine
from analysis.analyzer_cli import (
    detect_technique,
    truncate_normalized_data,
    run_analysis,
    parse_args,
)
from analysis.ingestion.watchdog import CyclerWatchdog
from export_engine import (
    generate_electrochemical_dataset_csv,
    generate_electrochemical_dataset_xlsx,
)


# ==============================================================================
# Fixtures & Synthetic Data Generators
# ==============================================================================

@pytest.fixture
def disc_geometry():
    """Standard 12 mm diameter electrode disc (area ~ 1.131 cm²)."""
    return ElectrodeGeometry(area_cm2=1.13097, active_mass_mg=2.5)


def create_synthetic_symmetric_cell_data(
    duration_hours: float = 100.0,
    short_at_hour: float | None = 60.0,
    point_interval_s: float = 60.0,
) -> NormalizedElectrochemicalData:
    """
    Generates synthetic Zn||Zn symmetric cell data.
    If short_at_hour is provided, healthy +/-35 mV plateaus transition to near 0 mV short-circuit flatline.
    """
    total_seconds = int(duration_hours * 3600)
    times = np.arange(0, total_seconds, point_interval_s, dtype=float)
    n_pts = len(times)

    # 1 hour strip (+1 mA), 1 hour plate (-1 mA)
    half_period_s = 3600.0
    period_s = 7200.0
    phase = (times % period_s) < half_period_s

    current = np.where(phase, 1.0, -1.0)
    voltage = np.where(phase, 0.035, -0.035)

    if short_at_hour is not None:
        short_s = short_at_hour * 3600.0
        short_mask = times >= short_s
        voltage[short_mask] = 0.001 * np.where(phase[short_mask], 1.0, -1.0)

    areal_cap = (np.abs(current) * times / 3600.0) / 1.13097
    cycle_num = (times // period_s).astype(int) + 1

    return NormalizedElectrochemicalData(
        time_s=times.tolist(),
        voltage_v=voltage.tolist(),
        current_ma=current.tolist(),
        current_density_ma_cm2=(current / 1.13097).tolist(),
        capacity_mah=(areal_cap * 1.13097).tolist(),
        areal_capacity_mah_cm2=areal_cap.tolist(),
        cycle_number=cycle_num.tolist(),
        display_time=(times / 3600.0).tolist(),
        time_unit="h",
    )


# ==============================================================================
# 1. Parsers Tests
# ==============================================================================

def test_ascii_parser_biologic_mpt(tmp_path):
    mpt_content = """BT-Lab ASCII FILE
Device : VMP3
Nb header lines : 4
time/s\tEcell/V\t<I>/mA\tcycle number
0.0\t-0.035\t-1.0\t1
60.0\t-0.035\t-1.0\t1
120.0\t0.035\t1.0\t1
180.0\t0.035\t1.0\t1
"""
    mpt_file = tmp_path / "test_symmetric.mpt"
    mpt_file.write_text(mpt_content)

    parser = AsciiCyclerParser()
    assert parser.can_parse(mpt_file) is True
    data = parser.parse(mpt_file)

    assert len(data.time_s) == 4
    assert data.time_s[1] == 60.0
    assert data.voltage_v[0] == -0.035
    assert data.current_ma[2] == 1.0
    assert data.cycle_number == [1, 1, 1, 1]


def test_ascii_parser_csv_landt(tmp_path):
    csv_content = """Time(s),Voltage(V),Current(mA),Capacity(mAh)
0.0,1.2,10.0,0.0
10.0,1.25,10.0,0.027
20.0,1.3,10.0,0.055
"""
    csv_file = tmp_path / "landt_export.csv"
    csv_file.write_text(csv_content)

    data = parse_cycler_file(csv_file)
    assert len(data.voltage_v) == 3
    assert data.current_ma[0] == 10.0


def test_ascii_parser_arbin_unit_scaling(tmp_path):
    """Verifies that Arbin files with Current in Amperes and Capacity in Ah scale to mA and mAh."""
    arbin_csv = """Test_Time(s),Voltage(V),Current(A),Charge_Capacity(Ah),Discharge_Capacity(Ah)
0.0,1.20,0.005,0.0,0.0
60.0,1.25,0.005,0.0000833,0.0
120.0,1.30,-0.005,0.0000833,0.0000833
"""
    file_path = tmp_path / "arbin_export.csv"
    file_path.write_text(arbin_csv)

    data = parse_cycler_file(file_path)
    assert len(data.time_s) == 3
    # 0.005 A -> 5.0 mA
    assert data.current_ma[0] == pytest.approx(5.0, abs=0.01)
    assert data.current_ma[2] == pytest.approx(-5.0, abs=0.01)
    # 0.0000833 Ah -> 0.0833 mAh
    assert data.capacity_charge_mah is not None
    assert data.capacity_charge_mah[1] == pytest.approx(0.0833, abs=0.01)


def test_ascii_parser_neware_format(tmp_path):
    """Verifies that Neware cycler files with Step_Index and Cycle_Index parse cleanly."""
    neware_csv = """Time(s),Step_Index,Cycle_Index,Current(mA),Voltage(V),Capacity(mAh)
0.0,1,1,1.13,1.35,0.0
60.0,1,1,1.13,1.36,0.0188
120.0,2,1,-1.13,1.25,0.0376
180.0,3,2,1.13,1.37,0.0564
"""
    file_path = tmp_path / "neware_export.csv"
    file_path.write_text(neware_csv)

    data = parse_cycler_file(file_path)
    assert len(data.time_s) == 4
    assert data.step_index == [1, 1, 2, 3]
    assert data.cycle_number == [1, 1, 1, 2]
    assert data.current_ma[0] == pytest.approx(1.13)


def test_mpr_parser_biologic_binary(tmp_path):
    """Verifies that BioLogic binary .mpr files decode accurately with channel mappings."""
    import struct
    header = b'BIO-LOGIC FILE VMP3\x00\x00\x00'
    vmp_tag = b'VMP data\x00\x00'
    num_rows = 3
    num_cols = 4
    col_ids = bytes([4, 6, 7, 8])  # 4: time/s, 6: Ewe/V, 7: I/mA, 8: cycle
    row_data = (
        struct.pack('<ffff', 0.0, 1.20, 10.0, 1.0)
        + struct.pack('<ffff', 60.0, 1.25, 10.0, 1.0)
        + struct.pack('<ffff', 120.0, 1.30, 10.0, 1.0)
    )
    content = header + vmp_tag + struct.pack('<IB', num_rows, num_cols) + col_ids + row_data

    mpr_file = tmp_path / "biologic_test.mpr"
    mpr_file.write_bytes(content)

    parser = MprCyclerParser()
    assert parser.can_parse(mpr_file) is True
    data = parser.parse(mpr_file)

    assert len(data.time_s) == 3
    assert data.time_s == [0.0, 60.0, 120.0]
    assert data.voltage_v[0] == pytest.approx(1.20, rel=1e-3)
    assert data.current_ma[0] == pytest.approx(10.0, rel=1e-3)
    assert data.cycle_number == [1, 1, 1]


def test_excel_parser_builtin_zip(tmp_path):
    """Verifies zero-dependency .xlsx workbook parsing via built-in zipfile and XML."""
    import zipfile

    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '</Types>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Channel_1" sheetId="1" r:id="rId1"/></sheets>'
        '</workbook>'
    )
    sheet1 = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetData>'
        '<row r="1">'
        '<c r="A1"><v>Time(s)</v></c>'
        '<c r="B1"><v>Voltage(V)</v></c>'
        '<c r="C1"><v>Current(mA)</v></c>'
        '</row>'
        '<row r="2">'
        '<c r="A2"><v>0.0</v></c>'
        '<c r="B2"><v>1.25</v></c>'
        '<c r="C2"><v>10.0</v></c>'
        '</row>'
        '<row r="3">'
        '<c r="A3"><v>60.0</v></c>'
        '<c r="B3"><v>1.30</v></c>'
        '<c r="C3"><v>10.0</v></c>'
        '</row>'
        '</sheetData>'
        '</worksheet>'
    )

    xlsx_file = tmp_path / "cycler_data.xlsx"
    with zipfile.ZipFile(xlsx_file, "w") as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/worksheets/sheet1.xml", sheet1)

    parser = ExcelCyclerParser()
    assert parser.can_parse(xlsx_file) is True
    data = parser.parse(xlsx_file)

    assert len(data.time_s) == 2
    assert data.time_s == [0.0, 60.0]
    assert data.voltage_v == [1.25, 1.30]
    assert data.current_ma == [10.0, 10.0]


def test_parser_unsupported_extension(tmp_path):
    bad_file = tmp_path / "unsupported.xyz"
    bad_file.write_text("dummy")

    with pytest.raises(ValueError, match="Unsupported cycler file format"):
        parse_cycler_file(bad_file)


# ==============================================================================
# 2. Processing Pipeline Tests
# ==============================================================================

def test_time_scaler_conversions():
    scaler = TimeScaler()
    assert scaler.auto_scale(30.0)[1] == "s"
    assert scaler.auto_scale(120.0)[1] == "min"
    assert scaler.auto_scale(7200.0)[1] == "h"

    converted = scaler.scale_series([0, 3600, 7200], target_unit="h")
    assert converted == [0.0, 1.0, 2.0]


def test_rest_detector():
    detector = RestDetector()
    currents = [0.001, 0.002, 1.0, 1.0, 0.001]
    is_rest = detector.detect_rest_steps(currents, current_threshold_ma=0.01)
    assert is_rest == [True, True, False, False, True]


def test_rest_detector_trimming_and_zeroing():
    """
    Verifies that RestDetector accurately strips leading OCV rest steps,
    re-zeros time to t=0 at the first cycling step, and records trimmed duration.
    """
    times = [0.0, 600.0, 1200.0, 1800.0, 2400.0, 3000.0]
    volts = [1.4, 1.41, 1.41, 1.35, 1.34, 1.33]
    currs = [0.0, 0.001, 0.001, -1.13, -1.13, -1.13]
    raw = CyclerRawData(
        time_s=times,
        voltage_v=volts,
        current_ma=currs,
    )

    trimmed, trimmed_s = RestDetector.trim_initial_rest(raw, current_threshold_ma=0.005)
    assert trimmed_s == 1800.0
    assert len(trimmed.time_s) == 3
    assert trimmed.time_s[0] == 0.0
    assert trimmed.time_s[1] == 600.0
    assert trimmed.time_s[2] == 1200.0
    assert trimmed.current_ma[0] == -1.13
    assert trimmed.metadata["trimmed_rest_seconds"] == 1800.0
    assert trimmed.metadata["trimmed_points_count"] == 3


def test_rest_detector_no_rest_case():
    """Verifies that RestDetector leaves data unmodified when cell starts cycling immediately."""
    raw = CyclerRawData(
        time_s=[0.0, 60.0, 120.0],
        voltage_v=[0.035, 0.035, 0.035],
        current_ma=[1.13, 1.13, 1.13],
    )
    trimmed, trimmed_s = RestDetector.trim_initial_rest(raw, current_threshold_ma=0.005)
    assert trimmed_s == 0.0
    assert len(trimmed.time_s) == 3
    assert trimmed.time_s[0] == 0.0


def test_math_engine_plateau_and_derivatives():
    engine = MathEngine()
    vals = [1.0, 1.05, 1.0, 0.99, 1.02]
    med, std = engine.calculate_plateau_median_std(vals)
    assert round(med, 2) == 1.0
    assert std < 0.05

    v = [1.0, 1.1, 1.2, 1.3]
    q = [0.0, 0.1, 0.25, 0.3]
    dq_dv = engine.compute_dq_dv(v, q)
    assert len(dq_dv) == 4


def test_electrochemical_normalizer(disc_geometry):
    normalizer = ElectrochemicalNormalizer(geometry=disc_geometry)
    raw = CyclerRawData(
        time_s=[0.0, 3600.0],
        voltage_v=[1.0, 1.1],
        current_ma=[1.13097, 1.13097],
        capacity_charge_mah=[0.0, 1.13097],
        capacity_discharge_mah=[0.0, 0.0],
    )
    norm = normalizer.process(raw, auto_trim_rest=False)
    assert pytest.approx(norm.current_density_ma_cm2[0], rel=1e-3) == 1.0
    assert pytest.approx(norm.areal_capacity_mah_cm2[1], rel=1e-3) == 1.0
    assert norm.time_unit == "h"


# ==============================================================================
# 3. Techniques Analysis Tests (CV, LSV, EIS, Asymmetric, Full Cell)
# ==============================================================================

def test_cv_analyzer():
    # Duck-shaped cyclic voltammogram
    v = [-0.2, 0.0, 0.15, 0.3, 0.15, 0.0, -0.08, -0.2]
    i = [0.0, 1.0, 5.0, 0.0, -1.0, -3.0, -4.0, 0.0]
    data = NormalizedElectrochemicalData(
        time_s=list(range(len(v))),
        voltage_v=v,
        current_ma=i,
        current_density_ma_cm2=i,
        capacity_mah=[0.0] * len(v),
        areal_capacity_mah_cm2=[0.0] * len(v),
        display_time=list(range(len(v))),
        time_unit="h",
    )
    cv_res = CVAnalyzer().analyze(data)
    assert cv_res.anodic_peak_potential_epa_v == 0.15
    assert cv_res.cathodic_peak_potential_epc_v == -0.08
    assert cv_res.anodic_peak_current_ipa_ma == 5.0
    assert cv_res.cathodic_peak_current_ipc_ma == -4.0
    assert cv_res.peak_separation_delta_ep_mv == 230.0
    assert cv_res.nucleation_overpotential_mv is not None
    assert cv_res.nucleation_overpotential_mv > 0.0


def test_lsv_analyzer():
    v = np.linspace(-0.2, -1.4, 50).tolist()
    # Exponential cathodic current for HER
    j = (-0.01 * np.exp(np.linspace(0, 7, 50))).tolist()
    data = NormalizedElectrochemicalData(
        time_s=list(range(50)),
        voltage_v=v,
        current_ma=j,
        current_density_ma_cm2=j,
        capacity_mah=[0.0] * 50,
        areal_capacity_mah_cm2=[0.0] * 50,
        display_time=list(range(50)),
        time_unit="h",
    )
    lsv_res = LSVAnalyzer().analyze(data)
    assert lsv_res.her_onset_potential_v is not None
    assert lsv_res.corrosion_potential_ecorr_v is not None
    assert lsv_res.tafel_slope_cathodic_mv_per_dec is not None
    assert lsv_res.electrochemical_stability_window_v is None


def test_lsv_electrochemical_stability_window():
    """Verifies ESW extraction when LSV spans cathodic HER onset to anodic decomposition limit."""
    v = np.linspace(-1.0, 2.5, 70).tolist()
    j = []
    for pot in v:
        if pot < -0.4:
            j.append(float(-1.0 * np.exp(-3.0 * (pot + 0.4))))
        elif pot > 1.8:
            j.append(float(1.0 * np.exp(3.0 * (pot - 1.8))))
        else:
            j.append(float(0.001 * pot))
    data = NormalizedElectrochemicalData(
        time_s=list(range(70)),
        voltage_v=v,
        current_ma=j,
        current_density_ma_cm2=j,
        capacity_mah=[0.0] * 70,
        areal_capacity_mah_cm2=[0.0] * 70,
        display_time=list(range(70)),
        time_unit="h",
    )
    lsv_res = LSVAnalyzer(her_threshold_ma_cm2=-1.0).analyze(data)
    assert lsv_res.her_onset_potential_v is not None
    assert lsv_res.electrochemical_stability_window_v is not None
    assert lsv_res.electrochemical_stability_window_v >= 2.0


def test_eis_analyzer():
    freqs = [100000.0, 10000.0, 1000.0, 100.0, 10.0, 1.0, 0.1]
    z_real = [2.0, 3.0, 6.0, 10.0, 12.0, 15.0, 20.0]
    z_imag = [0.0, 1.5, 4.0, 5.0, 4.0, 1.0, 0.5]
    raw = CyclerRawData(
        frequency_hz=freqs,
        z_real_ohm=z_real,
        z_imag_ohm=z_imag,
    )
    eis_res = EISAnalyzer().analyze_raw(raw)
    assert eis_res.solution_resistance_rs_ohm > 0
    assert eis_res.charge_transfer_resistance_rct_ohm > 0
    assert eis_res.equivalent_circuit == "R_s + R_ct // CPE"


def test_asymmetric_cell_analyzer():
    # Plating (negative current) followed by stripping (positive current)
    # 2 cycles
    n_cycle = 40
    times = list(range(n_cycle * 2))
    # Cycle 1: plate then strip
    pol1 = [-1.0] * 10 + [1.0] * 10
    v1 = [-0.08] * 3 + [-0.04] * 7 + [0.2] * 10
    q1 = [0.05 * i for i in range(10)] + [0.045 * i for i in range(10)]
    # Cycle 2: plate then strip
    pol2 = [-1.0] * 10 + [1.0] * 10
    v2 = [-0.07] * 3 + [-0.038] * 7 + [0.2] * 10
    q2 = [0.05 * i for i in range(10)] + [0.046 * i for i in range(10)]

    data = NormalizedElectrochemicalData(
        time_s=times,
        voltage_v=v1 + v2,
        current_ma=pol1 + pol2,
        current_density_ma_cm2=pol1 + pol2,
        capacity_mah=q1 + q2,
        areal_capacity_mah_cm2=q1 + q2,
        display_time=times,
        time_unit="h",
    )
    asymm_res = AsymmetricCellAnalyzer().analyze(data)
    assert asymm_res.total_cycles >= 1
    assert asymm_res.average_ce_percent > 70.0
    assert asymm_res.mean_nucleation_overpotential_mv > 0


def test_full_cell_analyzer():
    cyc_nums = [1] * 20 + [2] * 20 + [3] * 20
    v = [1.2] * 60
    i = [1.0] * 60
    q = [100.0] * 20 + [95.0] * 20 + [90.0] * 20
    data = NormalizedElectrochemicalData(
        time_s=list(range(60)),
        voltage_v=v,
        current_ma=i,
        current_density_ma_cm2=i,
        capacity_mah=[1.0] * 60,
        areal_capacity_mah_cm2=[1.0] * 60,
        specific_capacity_mah_g=q,
        cycle_number=cyc_nums,
        display_time=list(range(60)),
        time_unit="h",
    )
    fc_res = FullCellAnalyzer().analyze(data)
    assert fc_res.total_cycles == 3
    assert fc_res.capacity_retention_percent <= 100.0
    assert fc_res.energy_efficiency_percent is not None
    assert fc_res.average_energy_efficiency_percent is not None


def test_full_cell_energy_efficiency_trapezoidal_integration():
    """
    Verifies that FullCellAnalyzer accurately calculates round-trip Energy Efficiency (EE%)
    using numerical trapezoidal integration of E = integral(V * I dt).
    """
    t_chg = np.linspace(0, 3600, 20)
    v_chg = np.linspace(1.4, 1.7, 20)
    i_chg = np.ones(20) * 1.13097

    t_disch = np.linspace(3600, 7200, 20)
    v_disch = np.linspace(1.5, 1.2, 20)
    i_disch = -np.ones(20) * 1.13097

    t_all = np.concatenate([t_chg, t_disch])
    v_all = np.concatenate([v_chg, v_disch])
    i_all = np.concatenate([i_chg, i_disch])
    cyc_all = np.ones(40, dtype=int)
    cap_all = np.cumsum(np.abs(i_all) * (7200 / 40) / 3600.0)

    data = NormalizedElectrochemicalData(
        time_s=t_all.tolist(),
        voltage_v=v_all.tolist(),
        current_ma=i_all.tolist(),
        current_density_ma_cm2=(i_all / 1.13097).tolist(),
        capacity_mah=cap_all.tolist(),
        areal_capacity_mah_cm2=(cap_all / 1.13097).tolist(),
        cycle_number=cyc_all.tolist(),
        display_time=(t_all / 3600.0).tolist(),
        time_unit="h",
    )

    fc_res = FullCellAnalyzer().analyze(data)
    assert fc_res.total_cycles == 1
    assert fc_res.energy_efficiency_percent is not None
    assert 75.0 <= fc_res.energy_efficiency_percent <= 95.0
    assert len(fc_res.energy_efficiencies) == 1


# ==============================================================================
# 4. Adaptive Failure-Aware Cycling Windowing & Dual-Tier Metric Aggregation
# ==============================================================================

def test_symmetric_cell_healthy_cycling_metrics():
    """Verifies that a healthy 100h cell has identical full and active cycling metrics."""
    data = create_synthetic_symmetric_cell_data(duration_hours=100.0, short_at_hour=None)
    analyzer = SymmetricCellAnalyzer()
    res = analyzer.analyze(data)

    assert res.short_circuit_detected is False
    assert res.failure_mode == "none"
    assert res.failure_cycle is None
    assert res.failure_time_hours is None
    assert res.total_lifespan_hours >= 99.0
    assert pytest.approx(res.mean_overpotential_mv, abs=2.0) == 35.0

    # Dual-tier metrics validation
    assert res.active_cycling_metrics is not None
    assert res.total_test_summary is not None

    active = res.active_cycling_metrics
    assert pytest.approx(active["mean_overpotential_mv"], abs=2.0) == 35.0
    assert pytest.approx(active["mean_hysteresis_mv"], abs=4.0) == 70.0
    assert active["active_lifespan_hours"] == res.total_lifespan_hours
    assert active["active_cycles"] == res.total_cycles

    summary = res.total_test_summary
    assert summary["nominal_duration_hours"] == res.total_lifespan_hours
    assert summary["failure_mode"] == "none"
    assert summary["short_circuit_detected"] is False
    assert "without failure" in summary["diagnostic_message"]


def test_symmetric_cell_dendrite_short_isolation_vs_full_metrics():
    """
    CRITICAL TDD: A 100h test failing at 60h with 0V post-failure flatline
    must isolate active metrics strictly to pre-failure window (eta ~ 35 mV)
    while full-test metrics reflect flatline dilution (eta < 25 mV).
    """
    data = create_synthetic_symmetric_cell_data(duration_hours=100.0, short_at_hour=60.0)
    analyzer = SymmetricCellAnalyzer(short_circuit_threshold_mv=5.0)
    res = analyzer.analyze(data)

    assert res.short_circuit_detected is True
    assert res.failure_mode == "dendritic_short"
    assert res.failure_time_hours is not None
    assert 59.0 <= res.failure_time_hours <= 62.0

    # Full-test metrics are corrupted by post-failure flatlines (~1 mV)
    assert res.mean_overpotential_mv < 30.0

    # Pre-failure active cycling metrics strictly isolate healthy cycling (~35 mV)
    assert res.active_cycling_metrics is not None
    active = res.active_cycling_metrics
    assert pytest.approx(active["mean_overpotential_mv"], abs=2.0) == 35.0
    assert pytest.approx(active["mean_hysteresis_mv"], abs=4.0) == 70.0
    assert active["active_lifespan_hours"] <= 62.0
    assert active["active_cycles"] < res.total_cycles

    # Total test summary records full test metadata
    summary = res.total_test_summary
    assert summary["nominal_duration_hours"] >= 99.0
    assert summary["short_circuit_detected"] is True
    assert summary["failure_time_hours"] == res.failure_time_hours
    assert "Dendrite Short-Circuit Detected" in summary["diagnostic_message"]


# ==============================================================================
# 5. Interactive Dual-Range Visualization (Approach C in Plotly Dashboard)
# ==============================================================================

def test_plotly_dashboard_approach_c_viewport_and_controls(tmp_path):
    """
    Verifies that the HTML dashboard implements Approach C:
    - Default visible x-axis window is [0, t_fail + buffer].
    - Full 100h dataset is retained in the browser trace.
    - Plotly Range Selector buttons: [0 to Failure (Auto)], [Full Test], [Initial 24h].
    - Range slider is visible below the chart.
    - Vertical dashed failure line and diagnostic annotation tag are present.
    """
    data = create_synthetic_symmetric_cell_data(duration_hours=100.0, short_at_hour=60.0)
    analyzer = SymmetricCellAnalyzer()
    symm_res = analyzer.analyze(data)

    report = ElectrochemicalAnalysisReport(
        sample_id="Zn_Zn_Failure_Test",
        file_source="synthetic_100h.csv",
        technique=ElectrochemicalTechnique.GALVANOSTATIC_SYMMETRIC,
        geometry=ElectrodeGeometry(area_cm2=1.13097),
        processed_data=data,
        symmetric_cell=symm_res,
        active_cycling_metrics=symm_res.active_cycling_metrics,
        total_test_summary=symm_res.total_test_summary,
    )

    plot_engine = ElectrochemicalPlotEngine(output_dir=tmp_path)
    out_html = plot_engine.render_report(report)
    assert out_html.exists()

    html_content = out_html.read_text(encoding="utf-8")

    # 1. Full data retained (all 100 hours of points in the HTML JSON)
    assert "plot_vt" in html_content
    assert "Plotly.newPlot" in html_content

    # Extract plot_vt layout from HTML
    # Check Approach C viewport default range [0, ~65.0]
    assert '"range": [0,' in html_content

    # Check Range Selector Buttons
    assert "0 to Failure (Auto)" in html_content
    assert "Full Test" in html_content
    assert "Initial 24h" in html_content

    # Check Range Slider
    assert '"rangeslider": {"visible": true}' in html_content.lower()

    # Check Failure Line Shape
    assert '"dash": "dash"' in html_content
    assert "#d62728" in html_content

    # Check Diagnostic Annotation Tag
    assert "Dendrite Short-Circuit Detected at" in html_content
    assert "DENDRITIC SHORT-CIRCUIT DETECTED" in html_content


def test_plotly_dashboard_healthy_test_defaults(tmp_path):
    """Verifies that healthy test dashboard does not set failure line and shows full duration."""
    data = create_synthetic_symmetric_cell_data(duration_hours=50.0, short_at_hour=None)
    symm_res = SymmetricCellAnalyzer().analyze(data)

    report = ElectrochemicalAnalysisReport(
        sample_id="Zn_Zn_Healthy_Test",
        file_source="synthetic_healthy.csv",
        technique=ElectrochemicalTechnique.GALVANOSTATIC_SYMMETRIC,
        geometry=ElectrodeGeometry(area_cm2=1.13097),
        processed_data=data,
        symmetric_cell=symm_res,
        active_cycling_metrics=symm_res.active_cycling_metrics,
        total_test_summary=symm_res.total_test_summary,
    )

    plot_engine = ElectrochemicalPlotEngine(output_dir=tmp_path)
    out_html = plot_engine.render_report(report)
    html_content = out_html.read_text(encoding="utf-8")

    assert "STABLE REVERSIBLE CYCLING" in html_content
    assert "DENDRITIC SHORT-CIRCUIT DETECTED" not in html_content


# ==============================================================================
# 6. Manual Windowing & CLI Overrides
# ==============================================================================

def test_truncate_normalized_data_max_time():
    data = create_synthetic_symmetric_cell_data(duration_hours=100.0)
    truncated = truncate_normalized_data(data, max_time_hours=40.0)
    assert max(truncated.time_s) <= 40.0 * 3600.0
    assert max(truncated.display_time) <= 40.0


def test_truncate_normalized_data_time_window():
    data = create_synthetic_symmetric_cell_data(duration_hours=100.0)
    truncated = truncate_normalized_data(data, time_window=[10.0, 30.0])
    assert min(truncated.time_s) >= 10.0 * 3600.0
    assert max(truncated.time_s) <= 30.0 * 3600.0


def test_truncate_normalized_data_max_cycles():
    data = create_synthetic_symmetric_cell_data(duration_hours=100.0)
    truncated = truncate_normalized_data(data, max_cycles=5)
    assert max(truncated.cycle_number) <= 5


def test_cli_parse_args_windowing_flags():
    args = parse_args([
        "--file", "dummy.csv",
        "--max-time-hours", "50.5",
        "--max-cycles", "25",
        "--time-window", "5.0", "45.0",
        "--trim-post-failure",
    ])
    assert args.max_time_hours == 50.5
    assert args.max_cycles == 25
    assert args.time_window == [5.0, 45.0]
    assert args.trim_post_failure is True


def test_run_analysis_with_manual_windowing(tmp_path):
    csv_file = tmp_path / "test_100h_symm.csv"
    data = create_synthetic_symmetric_cell_data(duration_hours=100.0, short_at_hour=60.0)

    # Write CSV
    header = "Time(s),Voltage(V),Current(mA)\n"
    rows = "\n".join(f"{t},{v},{i}" for t, v, i in zip(data.time_s[:200], data.voltage_v[:200], data.current_ma[:200]))
    csv_file.write_text(header + rows)

    out_json = tmp_path / "report.json"
    report = run_analysis(
        filepath=csv_file,
        technique_str="galvanostatic_symmetric",
        output_json=str(out_json),
        max_time_hours=1.5,
        no_html=True,
    )

    assert out_json.exists()
    assert max(report.processed_data.time_s) <= 1.5 * 3600.0


def test_run_analysis_trim_post_failure(tmp_path):
    csv_file = tmp_path / "shorting_cell.csv"
    data = create_synthetic_symmetric_cell_data(duration_hours=10.0, short_at_hour=4.0, point_interval_s=120.0)

    header = "Time(s),Voltage(V),Current(mA)\n"
    rows = "\n".join(f"{t},{v},{i}" for t, v, i in zip(data.time_s, data.voltage_v, data.current_ma))
    csv_file.write_text(header + rows)

    report = run_analysis(
        filepath=csv_file,
        technique_str="galvanostatic_symmetric",
        trim_post_failure=True,
        no_html=True,
    )

    assert report.symmetric_cell.short_circuit_detected is True
    # The processed data was cropped strictly to pre-failure window
    assert max(report.processed_data.display_time) <= report.symmetric_cell.failure_time_hours + 0.1


# ==============================================================================
# 7. Automated Watchdog Ingestion Pipeline
# ==============================================================================

def test_watchdog_workflow(tmp_path):
    watch_dir = tmp_path / "inbox"
    processed_dir = tmp_path / "archived"
    watch_dir.mkdir()

    processed_reports = []

    def callback(file_path):
        processed_reports.append(file_path.name)
        return None

    watchdog = CyclerWatchdog(
        watch_dir=watch_dir,
        processed_dir=processed_dir,
        callback=callback,
        stabilization_wait_s=0.01,
    )

    # Place an export file in inbox
    test_file = watch_dir / "sample_cycler_01.csv"
    test_file.write_text("Time(s),Voltage(V),Current(mA)\n0,1,0\n")

    watchdog.poll_once()

    assert "sample_cycler_01.csv" in processed_reports
    # File was archived to processed_dir
    assert (processed_dir / "sample_cycler_01.csv").exists()
    assert not test_file.exists()


def test_technique_auto_detection():
    # Symmetric
    symm = CyclerRawData(voltage_v=[-0.035, 0.035], current_ma=[-1.0, 1.0])
    assert detect_technique(symm) == ElectrochemicalTechnique.GALVANOSTATIC_SYMMETRIC

    # Full cell
    fc = CyclerRawData(voltage_v=[0.8, 1.6], current_ma=[1.0, -1.0])
    assert detect_technique(fc) == ElectrochemicalTechnique.GALVANOSTATIC_FULL_CELL

    # EIS
    eis = CyclerRawData(frequency_hz=[1000.0], z_real_ohm=[5.0], z_imag_ohm=[1.0])
    assert detect_technique(eis) == ElectrochemicalTechnique.EIS

    # LSV
    lsv = CyclerRawData(voltage_v=list(np.linspace(0.0, -1.0, 20)), current_ma=list(range(20)))
    assert detect_technique(lsv) == ElectrochemicalTechnique.LSV


def test_symmetric_cell_polarization_cutoff_failure():
    """Verifies polarization cutoff failure detection when potential exceeds 1.0V."""
    data = create_synthetic_symmetric_cell_data(duration_hours=50.0, short_at_hour=None)
    # Inject large voltage spike after hour 30 (+1.25V on strip, -1.25V on plate)
    t_arr = np.array(data.time_s)
    v_arr = np.array(data.voltage_v)
    period_s = 7200.0
    phase = (t_arr % period_s) < 3600.0
    spike_mask = t_arr >= 30.0 * 3600.0
    v_arr[spike_mask] = np.where(phase[spike_mask], 1.25, -1.25)

    passivated_data = NormalizedElectrochemicalData(
        time_s=data.time_s,
        voltage_v=v_arr.tolist(),
        current_ma=data.current_ma,
        current_density_ma_cm2=data.current_density_ma_cm2,
        capacity_mah=data.capacity_mah,
        areal_capacity_mah_cm2=data.areal_capacity_mah_cm2,
        cycle_number=data.cycle_number,
        display_time=data.display_time,
        time_unit="h",
    )

    analyzer = SymmetricCellAnalyzer(polarization_cutoff_v=1.0)
    res = analyzer.analyze(passivated_data)

    assert res.failure_mode == "polarization_cut_off"
    assert res.failure_time_hours is not None
    assert 29.0 <= res.failure_time_hours <= 33.0
    assert "Polarization Cutoff Exceeded" in res.total_test_summary["diagnostic_message"]
    assert res.active_cycling_metrics["active_lifespan_hours"] <= 33.0


def test_symmetric_cell_soft_short_failure():
    """Verifies soft-short failure detection when plateau micro-oscillations exceed 15 mV std."""
    data = create_synthetic_symmetric_cell_data(duration_hours=50.0, short_at_hour=None)
    t_arr = np.array(data.time_s)
    v_arr = np.array(data.voltage_v)

    # Add high-frequency noise of 40 mV amplitude after hour 20 (cycle ~10)
    rng = np.random.default_rng(42)
    noise_mask = t_arr >= 20.0 * 3600.0
    v_arr[noise_mask] += rng.normal(0, 0.04, size=int(np.sum(noise_mask)))

    noisy_data = NormalizedElectrochemicalData(
        time_s=data.time_s,
        voltage_v=v_arr.tolist(),
        current_ma=data.current_ma,
        current_density_ma_cm2=data.current_density_ma_cm2,
        capacity_mah=data.capacity_mah,
        areal_capacity_mah_cm2=data.areal_capacity_mah_cm2,
        cycle_number=data.cycle_number,
        display_time=data.display_time,
        time_unit="h",
    )

    analyzer = SymmetricCellAnalyzer()
    res = analyzer.analyze(noisy_data)

    assert res.soft_short_detected is True
    assert res.failure_mode == "soft_short"
    assert res.failure_time_hours is not None
    assert 19.0 <= res.failure_time_hours <= 25.0
    assert "Soft-Short" in res.total_test_summary["diagnostic_message"]


def test_dual_tier_metric_report_json_serialization(tmp_path):
    """Verifies that JSON analysis reports faithfully preserve both active_cycling_metrics and total_test_summary."""
    data = create_synthetic_symmetric_cell_data(duration_hours=100.0, short_at_hour=60.0)
    analyzer = SymmetricCellAnalyzer()
    symm_res = analyzer.analyze(data)

    report = ElectrochemicalAnalysisReport(
        sample_id="Zn_Zn_DualTier_Test",
        file_source="test_raw.mpt",
        technique=ElectrochemicalTechnique.GALVANOSTATIC_SYMMETRIC,
        geometry=ElectrodeGeometry(area_cm2=1.13097),
        processed_data=data,
        symmetric_cell=symm_res,
        active_cycling_metrics=symm_res.active_cycling_metrics,
        total_test_summary=symm_res.total_test_summary,
    )

    json_str = report.model_dump_json(indent=2)
    parsed = json.loads(json_str)

    # Top-level dual-tier metrics
    assert "active_cycling_metrics" in parsed
    assert "total_test_summary" in parsed
    assert parsed["active_cycling_metrics"]["active_lifespan_hours"] <= 62.0
    assert pytest.approx(parsed["active_cycling_metrics"]["mean_overpotential_mv"], abs=2.0) == 35.0
    assert parsed["total_test_summary"]["nominal_duration_hours"] >= 99.0
    assert parsed["total_test_summary"]["short_circuit_detected"] is True

    # Nested symmetric_cell metrics
    assert "active_cycling_metrics" in parsed["symmetric_cell"]
    assert "total_test_summary" in parsed["symmetric_cell"]


def test_plotly_dashboard_100h_full_trace_retention_in_html(tmp_path):
    """
    Verifies that the generated HTML preserves all 100 hours of raw trace data
    while restricting the viewport range to [0, t_fail + buffer] (Approach C).
    """
    data = create_synthetic_symmetric_cell_data(duration_hours=100.0, short_at_hour=60.0)
    symm_res = SymmetricCellAnalyzer().analyze(data)

    report = ElectrochemicalAnalysisReport(
        sample_id="Zn_Zn_Viewport_Test",
        file_source="test_100h.csv",
        technique=ElectrochemicalTechnique.GALVANOSTATIC_SYMMETRIC,
        geometry=ElectrodeGeometry(area_cm2=1.13097),
        processed_data=data,
        symmetric_cell=symm_res,
        active_cycling_metrics=symm_res.active_cycling_metrics,
        total_test_summary=symm_res.total_test_summary,
    )

    engine = ElectrochemicalPlotEngine(output_dir=tmp_path)
    html_file = engine.render_report(report)
    content = html_file.read_text(encoding="utf-8")

    # Full data check: 100.0 hours points exist in the trace
    assert "99.9" in content or "100.0" in content or "99.8" in content

    # Viewport default check: range upper bound is failure_time + buffer (~65 h)
    t_fail = symm_res.failure_time_hours
    expected_max = round(t_fail + max(5.0, 0.05 * t_fail), 2)
    assert f'"range": [0, {expected_max}]' in content


def test_cycler_template_generators_and_roundtrip(tmp_path):
    """
    Verifies that the reference template generators (BioLogic .mpt, Landt .csv, Arbin .xlsx)
    produce valid, parseable files that pass end-to-end through run_analysis().
    """
    from analysis.parsers.template_generators import (
        generate_template_biologic_mpt,
        generate_template_landt_csv,
        generate_template_arbin_xlsx,
    )

    # 1. BioLogic .mpt
    mpt_bytes = generate_template_biologic_mpt(duration_hours=20.0)
    assert b"BT-Lab ASCII FILE" in mpt_bytes
    assert b"Nb header lines : 4" in mpt_bytes
    mpt_path = tmp_path / "sample_biologic.mpt"
    mpt_path.write_bytes(mpt_bytes)

    mpt_report = run_analysis(mpt_path, technique_str="auto", no_html=True)
    assert mpt_report.technique == ElectrochemicalTechnique.GALVANOSTATIC_SYMMETRIC
    assert mpt_report.symmetric_cell is not None
    assert mpt_report.symmetric_cell.total_cycles == 10
    assert pytest.approx(mpt_report.symmetric_cell.mean_overpotential_mv, abs=2.0) == 35.0

    # 2. Landt .csv
    csv_bytes = generate_template_landt_csv(duration_hours=20.0)
    assert b"Time(s),Voltage(V),Current(mA),Capacity(mAh)" in csv_bytes
    csv_path = tmp_path / "sample_landt.csv"
    csv_path.write_bytes(csv_bytes)

    csv_report = run_analysis(csv_path, technique_str="auto", no_html=True)
    assert csv_report.technique == ElectrochemicalTechnique.GALVANOSTATIC_SYMMETRIC
    assert csv_report.symmetric_cell is not None
    assert csv_report.symmetric_cell.total_cycles == 10
    assert pytest.approx(csv_report.symmetric_cell.mean_overpotential_mv, abs=2.0) == 35.0

    # 3. Arbin .xlsx
    xlsx_bytes = generate_template_arbin_xlsx(duration_hours=20.0)
    assert len(xlsx_bytes) > 500
    xlsx_path = tmp_path / "sample_arbin.xlsx"
    xlsx_path.write_bytes(xlsx_bytes)

    xlsx_report = run_analysis(xlsx_path, technique_str="auto", no_html=True)
    assert xlsx_report.technique == ElectrochemicalTechnique.GALVANOSTATIC_SYMMETRIC
    assert xlsx_report.symmetric_cell is not None
    assert xlsx_report.symmetric_cell.total_cycles == 10
    assert pytest.approx(xlsx_report.symmetric_cell.mean_overpotential_mv, abs=2.0) == 35.0


# ==============================================================================
# 8. Arbin Capacity Auto-Healer & Multi-Row Header Ingestion Tests (Chat 1)
# ==============================================================================

def test_arbin_accumulated_capacity_half_cell(tmp_path):
    """
    Verifies that Arbin half-cell files where Discharge_Capacity(Ah) and Charge_Capacity(Ah)
    accumulate monotonically across cycles without resetting to zero per cycle are correctly
    healed using step- and polarity-aware capacity deltas (ΔQ = Q_end - Q_start).
    """
    # Zn||Cu half-cell cycling: 3 cycles of plating (1.0 mAh/cm²) and stripping
    # Geometry: 1.0 cm²
    rows = [
        "Data_Point,Test_Time(s),Step_Index,Cycle_Index,Current(A),Voltage(V),Charge_Capacity(Ah),Discharge_Capacity(Ah)"
    ]

    t = 0.0
    dt = 120.0
    step = 1
    cum_chg_ah = 0.0
    cum_disch_ah = 0.0

    # 3 cycles: Cycle 1 CE ~98%, Cycle 2 CE ~97%, Cycle 3 CE ~96%
    ce_targets = [0.98, 0.97, 0.96]
    for cyc_idx, target_ce in enumerate(ce_targets, start=1):
        # 1. Plating step: negative current -0.001 A (-1.0 mA) for 3600 s -> 0.001 Ah (1.0 mAh)
        step = (cyc_idx - 1) * 2 + 1
        plate_dt = 3600.0 / 30.0
        for _ in range(31):
            rows.append(f"{len(rows)},{t:.1f},{step},{cyc_idx},-0.001,-0.040,{cum_chg_ah:.7f},{cum_disch_ah:.7f}")
            t += plate_dt
            cum_disch_ah += 0.001 * (plate_dt / 3600.0)

        # 2. Stripping step: positive current +0.001 A (+1.0 mA) for target_ce * 3600 s
        step = (cyc_idx - 1) * 2 + 2
        strip_dt = (target_ce * 3600.0) / 30.0
        for _ in range(31):
            rows.append(f"{len(rows)},{t:.1f},{step},{cyc_idx},0.001,0.200,{cum_chg_ah:.7f},{cum_disch_ah:.7f}")
            t += strip_dt
            cum_chg_ah += 0.001 * (strip_dt / 3600.0)

    csv_content = "\n".join(rows) + "\n"
    csv_file = tmp_path / "arbin_accumulated_half_cell.csv"
    csv_file.write_text(csv_content, encoding="utf-8")

    # Ingestion & normalization
    raw_data = parse_cycler_file(csv_file)
    normalizer = ElectrochemicalNormalizer(geometry=ElectrodeGeometry(area_cm2=1.0))
    norm_data = normalizer.process(raw_data, auto_trim_rest=False)

    # Technique evaluation
    analyzer = AsymmetricCellAnalyzer()
    res = analyzer.analyze(norm_data)

    assert res.total_cycles == 3
    assert len(res.cycle_metrics) == 3

    # Check cycle 1
    m1 = res.cycle_metrics[0]
    assert m1.cycle_number == 1
    assert pytest.approx(m1.plating_capacity_mah_cm2, rel=0.03) == 1.0
    assert m1.coulombic_efficiency_percent == pytest.approx(98.0, abs=1.5)

    # Check cycle 2 (accumulated Ah did not corrupt the delta)
    m2 = res.cycle_metrics[1]
    assert m2.cycle_number == 2
    assert pytest.approx(m2.plating_capacity_mah_cm2, rel=0.03) == 1.0
    assert m2.coulombic_efficiency_percent == pytest.approx(97.0, abs=1.5)

    # Check cycle 3
    m3 = res.cycle_metrics[2]
    assert m3.cycle_number == 3
    assert pytest.approx(m3.plating_capacity_mah_cm2, rel=0.03) == 1.0
    assert m3.coulombic_efficiency_percent == pytest.approx(96.0, abs=1.5)

    # Average CE
    assert 96.0 <= res.average_ce_percent <= 98.0


def test_capacity_corrupt_nonmonotonic_fallback_to_trapezoidal_integration():
    """
    Verifies that MathEngine and AsymmetricCellAnalyzer fall back cleanly to numerical
    trapezoidal integration (∫ |I| dt / 3600) when capacity columns are corrupt,
    non-monotonic, or missing.
    """
    times = [0.0, 1800.0, 3600.0]
    current = [-1.0, -1.0, -1.0]

    # 1. Missing capacity column (None)
    q_val, method = MathEngine.compute_step_capacity_delta(times, current, raw_capacity=None, expected_polarity=-1)
    assert method == "trapezoidal_integral"
    assert q_val == pytest.approx(1.0, abs=1e-3)

    # 2. Corrupt non-monotonic capacity with sudden drop (e.g. 0.8 -> 0.2 while current flows)
    corrupt_q = [0.0, 0.8, 0.2]
    q_val_c, method_c = MathEngine.compute_step_capacity_delta(times, current, raw_capacity=corrupt_q, expected_polarity=-1)
    assert method_c == "trapezoidal_integral"
    assert q_val_c == pytest.approx(1.0, abs=1e-3)

    # 3. All zeros while current is actively flowing
    zeros_q = [0.0, 0.0, 0.0]
    q_val_z, method_z = MathEngine.compute_step_capacity_delta(times, current, raw_capacity=zeros_q, expected_polarity=-1)
    assert method_z == "trapezoidal_integral"
    assert q_val_z == pytest.approx(1.0, abs=1e-3)

    # 4. Valid monotonic delta: 5.0 -> 6.0
    valid_q = [5.0, 5.5, 6.0]
    q_val_v, method_v = MathEngine.compute_step_capacity_delta(times, current, raw_capacity=valid_q, expected_polarity=-1)
    assert method_v == "delta"
    assert q_val_v == pytest.approx(1.0, abs=1e-5)


def test_full_cell_automated_cycle_coulombic_efficiency():
    """
    Verifies that FullCellAnalyzer automatically computes true cycle-by-cycle Coulombic
    Efficiency (CE = Q_discharge / Q_charge * 100%) rather than a static default.
    """
    # 2 cycles:
    # Cycle 1: Charge 1.0 mAh (I = +1 mA, 1h), Discharge 0.95 mAh (I = -1 mA, 0.95h) -> CE = 95.0%
    # Cycle 2: Charge 1.0 mAh (I = +1 mA, 1h), Discharge 0.92 mAh (I = -1 mA, 0.92h) -> CE = 92.0%
    t_list = []
    v_list = []
    i_list = []
    c_list = []
    chg_q_list = []
    disch_q_list = []

    t_now = 0.0
    cum_chg = 0.0
    cum_disch = 0.0

    for cyc in [1, 2]:
        # Charge: 1h at +1 mA (10 points)
        for _ in range(10):
            t_list.append(t_now)
            v_list.append(1.55)
            i_list.append(1.0)
            c_list.append(cyc)
            chg_q_list.append(cum_chg)
            disch_q_list.append(cum_disch)
            dt = 360.0
            t_now += dt
            cum_chg += 1.0 * (dt / 3600.0)

        # Discharge: 0.95h (cyc 1) or 0.92h (cyc 2) at -1 mA
        disch_target = 0.95 if cyc == 1 else 0.92
        for _ in range(10):
            t_list.append(t_now)
            v_list.append(1.25)
            i_list.append(-1.0)
            c_list.append(cyc)
            chg_q_list.append(cum_chg)
            disch_q_list.append(cum_disch)
            dt = (disch_target * 3600.0) / 10.0
            t_now += dt
            cum_disch += 1.0 * (dt / 3600.0)

    data = NormalizedElectrochemicalData(
        time_s=t_list,
        voltage_v=v_list,
        current_ma=i_list,
        current_density_ma_cm2=i_list,
        capacity_mah=(np.array(chg_q_list) + np.array(disch_q_list)).tolist(),
        areal_capacity_mah_cm2=(np.array(chg_q_list) + np.array(disch_q_list)).tolist(),
        specific_capacity_mah_g=(np.array(chg_q_list) + np.array(disch_q_list)).tolist(),
        capacity_charge_mah=chg_q_list,
        capacity_discharge_mah=disch_q_list,
        cycle_number=c_list,
        display_time=(np.array(t_list) / 3600.0).tolist(),
        time_unit="h",
    )

    fc_res = FullCellAnalyzer().analyze(data)
    assert fc_res.total_cycles == 2
    assert fc_res.coulombic_efficiencies[0] == pytest.approx(95.0, abs=1.5)
    assert fc_res.coulombic_efficiencies[1] == pytest.approx(92.0, abs=1.5)
    assert fc_res.average_coulombic_efficiency_percent == pytest.approx(93.5, abs=1.5)


def test_multi_row_header_auto_healing_ascii(tmp_path):
    """
    Verifies that multi-row headers in ASCII files (Row 1 names, Row 2 units)
    are cleanly healed, units are extracted (A -> mA, Ah -> mAh), and data is read from Row 3.
    """
    content = """Record,Test_Time,Voltage,Current,Charge_Capacity,Discharge_Capacity
,s,V,A,Ah,Ah
1,0.0,1.20,0.005,0.0,0.0
2,60.0,1.25,0.005,0.0000833,0.0
3,120.0,1.30,-0.005,0.0000833,0.0000833
"""
    file_path = tmp_path / "multi_row_cycler.csv"
    file_path.write_text(content, encoding="utf-8")

    data = parse_cycler_file(file_path)
    assert len(data.time_s) == 3
    # 0.005 A -> 5.0 mA
    assert data.current_ma[0] == pytest.approx(5.0, abs=0.01)
    assert data.current_ma[2] == pytest.approx(-5.0, abs=0.01)
    # 0.0000833 Ah -> 0.0833 mAh
    assert data.capacity_charge_mah is not None
    assert data.capacity_charge_mah[1] == pytest.approx(0.0833, abs=0.01)


def test_multi_row_header_auto_healing_excel(tmp_path):
    """
    Verifies that Excel workbooks (.xlsx) with Row 1 names and Row 2 units
    cleanly detect unit rows and scale SI units appropriately.
    """
    import zipfile

    types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '</Types>'
    )
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    wb_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Channel_1" sheetId="1" r:id="rId1"/></sheets>'
        '</workbook>'
    )
    # Row 1: Header names
    # Row 2: Units (s, V, A, Ah)
    # Row 3: Numeric data
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetData>'
        '<row r="1">'
        '<c r="A1" t="inlineStr"><is><t>Test_Time</t></is></c>'
        '<c r="B1" t="inlineStr"><is><t>Voltage</t></is></c>'
        '<c r="C1" t="inlineStr"><is><t>Current</t></is></c>'
        '<c r="D1" t="inlineStr"><is><t>Charge_Capacity</t></is></c>'
        '</row>'
        '<row r="2">'
        '<c r="A2" t="inlineStr"><is><t>s</t></is></c>'
        '<c r="B2" t="inlineStr"><is><t>V</t></is></c>'
        '<c r="C2" t="inlineStr"><is><t>A</t></is></c>'
        '<c r="D2" t="inlineStr"><is><t>Ah</t></is></c>'
        '</row>'
        '<row r="3">'
        '<c r="A3"><v>0.0</v></c>'
        '<c r="B3"><v>1.25</v></c>'
        '<c r="C3"><v>0.010</v></c>'
        '<c r="D3"><v>0.0001</v></c>'
        '</row>'
        '<row r="4">'
        '<c r="A4"><v>60.0</v></c>'
        '<c r="B4"><v>1.28</v></c>'
        '<c r="C4"><v>0.010</v></c>'
        '<c r="D4"><v>0.0002</v></c>'
        '</row>'
        '</sheetData>'
        '</worksheet>'
    )

    xlsx_path = tmp_path / "multi_row_test.xlsx"
    with zipfile.ZipFile(xlsx_path, "w") as zf:
        zf.writestr("[Content_Types].xml", types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("xl/workbook.xml", wb_xml)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)

    parser = ExcelCyclerParser()
    assert parser.can_parse(xlsx_path) is True
    data = parser.parse(xlsx_path)

    assert len(data.time_s) == 2
    # 0.010 A -> 10.0 mA
    assert data.current_ma[0] == pytest.approx(10.0, rel=1e-3)
    # 0.0001 Ah -> 0.1 mAh
    assert data.capacity_charge_mah is not None
    assert data.capacity_charge_mah[0] == pytest.approx(0.1, rel=1e-3)


def test_fuzzy_header_aliases_and_durations(tmp_path):
    """
    Verifies that BioLogic (<I>/mA), Neware (Chg-Cap, Cur(mA)), Chinese column names,
    square brackets, and hh:mm:ss duration timestamps are all correctly parsed.
    """
    # 1. Chinese headers with square brackets
    cn_csv = """时间/s,电压/V,电流/mA,充电容量/mAh,放电容量/mAh
0.0,1.20,10.0,0.0,0.0
60.0,1.25,10.0,0.166,0.0
120.0,1.15,-10.0,0.166,0.166
"""
    f_cn = tmp_path / "cn_cycler.csv"
    f_cn.write_text(cn_csv, encoding="utf-8")
    d_cn = parse_cycler_file(f_cn)
    assert len(d_cn.time_s) == 3
    assert d_cn.current_ma[0] == 10.0
    assert d_cn.capacity_charge_mah[1] == pytest.approx(0.166, abs=1e-3)

    # 2. Square brackets: Current [A], Voltage [V], Time [min]
    bracket_csv = """Time [min],Voltage [V],Current [A],Capacity [Ah]
0.0,1.20,0.005,0.0
1.0,1.25,0.005,0.0000833
"""
    f_br = tmp_path / "bracket_cycler.csv"
    f_br.write_text(bracket_csv, encoding="utf-8")
    d_br = parse_cycler_file(f_br)
    assert d_br.time_s[1] == pytest.approx(60.0, rel=1e-3)  # 1 min -> 60 s
    assert d_br.current_ma[0] == pytest.approx(5.0, rel=1e-3)  # 0.005 A -> 5 mA

    # 3. Neware duration string: hh:mm:ss
    duration_csv = """Time,Voltage(V),Current(mA),Capacity(mAh)
00:00:00,1.20,10.0,0.0
00:01:30,1.25,10.0,0.25
01:00:00,1.30,10.0,10.0
"""
    f_dur = tmp_path / "duration_cycler.csv"
    f_dur.write_text(duration_csv, encoding="utf-8")
    d_dur = parse_cycler_file(f_dur)
    assert d_dur.time_s == [0.0, 90.0, 3600.0]


def test_biologic_mpt_long_header_and_calendar_datetime(tmp_path):
    """
    Verifies that BioLogic EC-Lab ASCII (.mpt) files with long metadata preambles
    (>50 lines, e.g. 'Nb header lines : 55') and calendar datetime timestamps in 'time/s'
    are correctly auto-detected, parsed, and normalized to elapsed seconds.
    """
    mpt_lines = [
        "EC-Lab ASCII FILE\n",
        "Nb header lines : 55\n",
    ]
    # Add filler metadata lines up to line 54
    for i in range(3, 55):
        mpt_lines.append(f"Param_{i} : Some metadata value {i}\n")

    # Line 55 is the actual column header row
    mpt_lines.append(
        "mode\tox/red\terror\tNs\ttime/s\tcontrol/mA\tEwe/V\tI/mA\tQ discharge/mA.h\tQ charge/mA.h\tcycle number\n"
    )
    # Add data rows with calendar datetimes
    mpt_lines.append("1\t1\t0\t0\t02/09/2026 21:54:04.000\t0.3\t0.035\t0.3\t0.0\t0.0\t1\n")
    mpt_lines.append("1\t1\t0\t0\t02/09/2026 21:54:34.000\t0.3\t0.036\t0.3\t0.0\t0.0025\t1\n")
    mpt_lines.append("1\t1\t0\t0\t02/09/2026 21:55:04.000\t0.3\t0.037\t0.3\t0.0\t0.0050\t1\n")

    f_mpt = tmp_path / "test_biologic.mpt"
    f_mpt.write_text("".join(mpt_lines), encoding="utf-8")

    parsed = parse_cycler_file(f_mpt)
    assert len(parsed.time_s) == 3
    assert parsed.time_s == [0.0, 30.0, 60.0]  # normalized to elapsed seconds
    assert pytest.approx(parsed.voltage_v[0], abs=1e-4) == 0.035
    assert pytest.approx(parsed.current_ma[0], abs=1e-4) == 0.3
    assert parsed.capacity_charge_mah is not None
    assert pytest.approx(parsed.capacity_charge_mah[1], abs=1e-4) == 0.0025


def test_gamry_dta_potentiostat_curve(tmp_path):
    """
    Verifies that Gamry Framework .DTA text files with CURVE TABLE markers
    and Vf / Im columns are correctly ingested and unit-scaled into CyclerRawData.
    """
    gamry_lines = [
        "EXPLAIN\n",
        "TAG\tCHRONOP\n",
        "TITLE\tZinc Plating Experiment\n",
        "DATE\t10/13/2026\n",
        "TIME\t21:30:00\n",
        "CURVE\tTABLE\n",
        "Pt\tTime\tVf\tIm\tVu\tSig\tAch\n",
        "#\ts\tV\tA\tV\tV\tV\n",
        "0\t0.000\t0.0350\t0.0010\t0.0350\t0.000\t0.000\n",
        "1\t10.000\t0.0360\t0.0010\t0.0360\t0.000\t0.000\n",
        "2\t20.000\t0.0370\t0.0010\t0.0370\t0.000\t0.000\n",
    ]
    f_dta = tmp_path / "experiment.dta"
    f_dta.write_text("".join(gamry_lines), encoding="utf-8")

    parsed = parse_cycler_file(f_dta)
    assert len(parsed.time_s) == 3
    assert parsed.time_s == [0.0, 10.0, 20.0]
    assert pytest.approx(parsed.voltage_v[0], abs=1e-4) == 0.0350
    # Im in A should scale to mA (0.001 A -> 1.0 mA)
    assert pytest.approx(parsed.current_ma[0], abs=1e-3) == 1.0


def test_gamry_dta_eis_zcurve(tmp_path):
    """
    Verifies that Gamry Framework .DTA EIS files with ZCURVE TABLE markers,
    Freq, Zreal, and Zimag columns are correctly recognized and ingested.
    """
    gamry_eis_lines = [
        "EXPLAIN\n",
        "TAG\tEISPOT\n",
        "ZCURVE\tTABLE\n",
        "Pt\tTime\tFreq\tZreal\tZimag\tZsig\tZmod\tZphz\n",
        "#\ts\tHz\tohm\tohm\tV\tohm\t°\n",
        "0\t0.0\t100000.0\t15.20\t-0.50\t0.01\t15.21\t-1.88\n",
        "1\t0.5\t10000.0\t16.80\t-2.10\t0.01\t16.93\t-7.12\n",
        "2\t1.0\t1000.0\t19.50\t-5.30\t0.01\t20.21\t-15.20\n",
    ]
    f_eis = tmp_path / "impedance.dta"
    f_eis.write_text("".join(gamry_eis_lines), encoding="utf-8")

    parsed = parse_cycler_file(f_eis)
    assert parsed.frequency_hz is not None
    assert len(parsed.frequency_hz) == 3
    assert parsed.frequency_hz[0] == 100000.0
    assert pytest.approx(parsed.z_real_ohm[0], abs=1e-2) == 15.20
    assert pytest.approx(parsed.z_imag_ohm[0], abs=1e-2) == -0.50


def test_metrohm_autolab_exported_csv(tmp_path):
    """
    Verifies that Metrohm Autolab NOVA exported CSV files with WE(1).Potential (V)
    and WE(1).Current (A) columns are cleanly ingested and normalized.
    """
    autolab_csv = """Time (s),WE(1).Potential (V),WE(1).Current (A)
0.0,0.035,0.001
30.0,0.036,0.001
60.0,0.037,0.001
"""
    f_autolab = tmp_path / "autolab_export.csv"
    f_autolab.write_text(autolab_csv, encoding="utf-8")

    parsed = parse_cycler_file(f_autolab)
    assert len(parsed.time_s) == 3
    assert pytest.approx(parsed.voltage_v[0], abs=1e-4) == 0.035
    assert pytest.approx(parsed.current_ma[0], abs=1e-3) == 1.0  # 0.001 A -> 1.0 mA


def test_biologic_mpr_binary_file():
    """
    Verifies that real BioLogic .mpr binary files are decoded by MprCyclerParser
    into standardized CyclerRawData.
    """
    real_mpr = Path(r"reference_repos/cellpy/testdata/data/biol.mpr")
    if not real_mpr.is_file():
        pytest.skip("biol.mpr fixture not present")

    parsed = parse_cycler_file(real_mpr)
    assert len(parsed.time_s) > 1000
    assert len(parsed.voltage_v) == len(parsed.time_s)
    assert len(parsed.current_ma) == len(parsed.time_s)
    assert min(parsed.voltage_v) > 0.0
    assert parsed.step_index is not None


def test_column_mapping_prefers_test_time_over_step_time_and_date():
    """
    Verifies that resolve_best_column_mapping picks Test_Time(s) as the primary 'time' channel
    when competing with Step_Time(s) and Date_Time, preventing cycle profile scrambling.
    """
    from analysis.parsers.base import match_canonical_channel, resolve_best_column_mapping

    raw_headers = [
        "Data_Point",
        "Date_Time",
        "Test_Time(s)",
        "Step_Time(s)",
        "Cycle_Index",
        "Step_Index",
        "Current(A)",
        "Voltage(V)",
    ]
    candidate_matches = {}
    for i, h in enumerate(raw_headers):
        m = match_canonical_channel(h)
        if m:
            candidate_matches[i] = m

    assert candidate_matches[1] == "time"        # Date_Time
    assert candidate_matches[2] == "time"        # Test_Time(s)
    assert candidate_matches[3] == "step_time"   # Step_Time(s)

    col_map = resolve_best_column_mapping(candidate_matches, raw_headers)

    # Test_Time(s) (col 2) must win the 'time' channel, NOT Date_Time (col 1) or Step_Time (col 3)
    assert col_map.get(2) == "time"
    assert 1 not in col_map  # Date_Time eliminated in favor of Test_Time(s)
    assert col_map.get(3) == "step_time"
    assert col_map.get(4) == "cycle_number"
    assert col_map.get(5) == "step_index"
    assert col_map.get(6) == "current"
    assert col_map.get(7) == "voltage"


def test_mpr_dynamic_offset_calculation():
    """
    Verifies that decode_mpr_data_module correctly calculates the data_offset
    dynamically for v2/v3 BioLogic MPR files using len(raw_bytes) - (n_points * row_size).
    """
    from analysis.parsers.third_party.biologic_mpr import decode_mpr_data_module
    import numpy as np

    n_points = 2
    n_cols = 2
    col_ids = np.array([4, 6], dtype="<u2")
    header_prefix = np.array([n_points], dtype="<u4").tobytes() + np.array([n_cols], dtype="u1").tobytes() + col_ids.tobytes()
    pad_len = 406 - len(header_prefix)
    padding = b"\x00" * pad_len
    assert len(header_prefix + padding) == 406

    data_dtype = np.dtype([("time/s", "<f8"), ("Ewe/V", "<f4")])
    data_records = np.array([(0.0, 0.035), (30.0, 0.040)], dtype=data_dtype).tobytes()

    module_bytes = header_prefix + padding + data_records
    data_mod = {"data": module_bytes, "version": 3}

    decoded = decode_mpr_data_module(data_mod)
    assert "time/s" in decoded
    assert "Ewe/V" in decoded
    assert len(decoded["time/s"]) == 2
    assert pytest.approx(decoded["time/s"][1]) == 30.0
    assert pytest.approx(decoded["Ewe/V"][0]) == 0.035
    assert pytest.approx(decoded["Ewe/V"][1]) == 0.040


# ==============================================================================
# 9. Savitzky-Golay Signal Processing, dQ/dV Differential Capacity & First-Class Figures
# ==============================================================================

def test_savitzky_golay_filter_degree_preservation():
    """
    Verifies that the pure NumPy Savitzky-Golay filter preserves polynomials
    of degree <= polyorder (fundamental mathematical theorem of Savitzky-Golay convolution).
    """
    x = np.linspace(-2.0, 2.0, 101)
    # Quadratic: y = 3x^2 - 2x + 5
    y_quad = 3.0 * (x ** 2) - 2.0 * x + 5.0

    y_filtered = MathEngine.savitzky_golay_filter(y_quad, window_length=15, polyorder=2)

    # Interior points (beyond half-window of 7 points) match exactly within floating point precision
    np.testing.assert_allclose(y_filtered[7:-7], y_quad[7:-7], rtol=1e-10, atol=1e-10)
    # Boundary points with reflection padding stay close (< 4% relative difference)
    np.testing.assert_allclose(y_filtered, y_quad, rtol=0.05, atol=1.0)


def test_savitzky_golay_filter_input_validation_and_coercion():
    """
    Verifies that savitzky_golay_filter handles edge cases gracefully:
    - Coerces even window_length to odd (window_length + 1)
    - Coerces window_length < 3 to 3
    - Clamps window_length exceeding data length
    - Clamps polyorder >= window_length
    - Returns input copy for short arrays (< 3 elements)
    """
    # Short array
    short = np.array([1.0, 2.0])
    res_short = MathEngine.savitzky_golay_filter(short, window_length=5, polyorder=2)
    np.testing.assert_array_equal(res_short, short)

    x = np.linspace(0, 10, 50)
    y = np.sin(x)

    # Even window length coerced to odd
    res_even = MathEngine.savitzky_golay_filter(y, window_length=10, polyorder=2)
    assert len(res_even) == len(y)
    assert not np.any(np.isnan(res_even))

    # Window length larger than len(y)
    res_huge = MathEngine.savitzky_golay_filter(y, window_length=100, polyorder=2)
    assert len(res_huge) == len(y)
    assert not np.any(np.isnan(res_huge))

    # Polyorder >= window_length
    res_order = MathEngine.savitzky_golay_filter(y, window_length=5, polyorder=5)
    assert len(res_order) == len(y)
    assert not np.any(np.isnan(res_order))


def test_savitzky_golay_filter_derivative():
    """
    Verifies that savitzky_golay_filter with deriv=1 correctly evaluates numerical derivatives:
    For y = 2x^2 + 3x + 1, dy/dx = 4x + 3.
    """
    dx = 0.05
    x = np.arange(0, 5, dx)
    y = 2.0 * (x ** 2) + 3.0 * x + 1.0
    expected_dy = 4.0 * x + 3.0

    dy_filtered = MathEngine.savitzky_golay_filter(y, window_length=11, polyorder=2, deriv=1, delta=dx)

    # Interior points match dy/dx
    np.testing.assert_allclose(dy_filtered[10:-10], expected_dy[10:-10], rtol=1e-5, atol=1e-5)


def test_compute_dq_dv_savgol():
    """
    Verifies compute_dq_dv_savgol calculates smoothed differential capacity (dQ/dV vs V)
    and successfully recovers peak position despite high-frequency noise.
    """
    v = np.linspace(1.0, 1.8, 100)
    # Synthetic capacity with a sigmoid step (e.g. phase transition at V = 1.45 V)
    q_ideal = 10.0 / (1.0 + np.exp(-(v - 1.45) / 0.03))

    rng = np.random.default_rng(42)
    q_noisy = q_ideal + rng.normal(0, 0.05, size=len(v))

    dq_dv = MathEngine.compute_dq_dv_savgol(v, q_noisy, window_length=15, polyorder=3)

    assert len(dq_dv) == len(v)
    assert not np.any(np.isnan(dq_dv))
    assert not np.any(np.isinf(dq_dv))

    # Peak must be centered near 1.45 V
    peak_idx = int(np.argmax(dq_dv))
    peak_v = v[peak_idx]
    assert 1.40 <= peak_v <= 1.50


def test_compute_dq_dv_dispatch_savgol():
    """
    Verifies MathEngine.compute_dq_dv dispatches filter_type='savgol'
    and matches direct compute_dq_dv_savgol output.
    """
    v = np.linspace(1.2, 1.6, 50).tolist()
    q = np.linspace(0.0, 50.0, 50).tolist()

    res_dispatch = MathEngine.compute_dq_dv(v, q, filter_type="savgol", window_length=11, polyorder=2)
    res_direct = MathEngine.compute_dq_dv_savgol(v, q, window_length=11, polyorder=2)

    assert len(res_dispatch) == len(v)
    np.testing.assert_allclose(res_dispatch, res_direct, rtol=1e-10)


def test_plot_engine_create_cv_figure():
    """
    Verifies ElectrochemicalPlotEngine.create_cv_figure renders cyclic voltammogram
    with anodic/cathodic peak markers and onset annotations.
    """
    from analysis.visualization.plot_engine import create_cv_figure

    v = np.concatenate([np.linspace(-0.2, 0.6, 50), np.linspace(0.6, -0.2, 50)])
    i = np.sin(np.linspace(0, 2 * np.pi, 100)) * 5.0
    norm_data = NormalizedElectrochemicalData(
        time_s=list(range(100)),
        display_time=[t / 3600.0 for t in range(100)],
        voltage_v=v.tolist(),
        current_ma=i.tolist(),
        current_density_ma_cm2=i.tolist(),
        capacity_mah=[0.0] * 100,
        areal_capacity_mah_cm2=[0.0] * 100,
    )
    cv_res = CVAnalysis(
        technique="cv",
        anodic_peak_potential_epa_v=0.45,
        cathodic_peak_potential_epc_v=-0.12,
        anodic_peak_current_ipa_ma=4.8,
        cathodic_peak_current_ipc_ma=-4.9,
        peak_separation_delta_ep_mv=570.0,
        current_ratio_ipa_ipc=0.98,
        plating_onset_potential_v=-0.05,
        stripping_onset_potential_v=0.08,
    )
    report = ElectrochemicalAnalysisReport(
        sample_id="CV_Test_Sample",
        file_source="cv_cycle.mpt",
        technique="cv",
        geometry=ElectrodeGeometry(area_cm2=1.0),
        processed_data=norm_data,
        cv=cv_res,
    )

    fig = create_cv_figure(report)
    assert fig is not None
    assert "Cyclic Voltammogram" in fig.layout.title.text

    trace_names = [t.name for t in fig.data]
    assert any("E_pa" in n for n in trace_names)
    assert any("E_pc" in n for n in trace_names)


def test_plot_engine_create_lsv_figures():
    """
    Verifies ElectrochemicalPlotEngine.create_lsv_figures creates dual polarization
    and Tafel slope kinetics figures with E_corr and HER annotations.
    """
    from analysis.visualization.plot_engine import create_lsv_figures

    v = np.linspace(-0.4, 0.2, 100)
    j = np.exp((v + 0.1) * 15.0) - np.exp(-(v + 0.1) * 15.0)

    norm_data = NormalizedElectrochemicalData(
        time_s=list(range(100)),
        display_time=[t / 3600.0 for t in range(100)],
        voltage_v=v.tolist(),
        current_ma=j.tolist(),
        current_density_ma_cm2=j.tolist(),
        capacity_mah=[0.0] * 100,
        areal_capacity_mah_cm2=[0.0] * 100,
    )
    lsv_res = LSVAnalysis(
        technique="lsv",
        corrosion_potential_ecorr_v=-0.10,
        corrosion_current_jcorr_ma_cm2=0.015,
        tafel_slope_cathodic_mv_per_dec=115.0,
        tafel_slope_anodic_mv_per_dec=88.0,
        her_onset_potential_v=-0.25,
    )
    report = ElectrochemicalAnalysisReport(
        sample_id="LSV_Test_Sample",
        file_source="lsv_scan.csv",
        technique="lsv",
        geometry=ElectrodeGeometry(area_cm2=1.0),
        processed_data=norm_data,
        lsv=lsv_res,
    )

    fig_lsv, fig_tafel = create_lsv_figures(report)
    assert fig_lsv is not None
    assert fig_tafel is not None
    assert "LSV Polarization Curve" in fig_lsv.layout.title.text
    assert "Tafel Slope Kinetics" in fig_tafel.layout.title.text


def test_plot_engine_create_eis_figure():
    """
    Verifies ElectrochemicalPlotEngine.create_eis_figure renders Nyquist plot
    with strict 1:1 aspect ratio (scaleanchor='x', scaleratio=1) and Rs / apex annotations.
    """
    from analysis.visualization.plot_engine import create_eis_figure

    zr = [2.5, 5.0, 10.0, 18.0, 25.0, 30.0]
    zi = [0.0, -3.0, -8.0, -10.0, -6.0, 0.0]
    norm_data = NormalizedElectrochemicalData(
        time_s=list(range(6)),
        display_time=[0.0] * 6,
        voltage_v=[0.0] * 6,
        current_ma=[0.0] * 6,
        current_density_ma_cm2=[0.0] * 6,
        capacity_mah=[0.0] * 6,
        areal_capacity_mah_cm2=[0.0] * 6,
        metadata={"z_real_ohm": zr, "z_imag_ohm": zi, "frequency_hz": [100000, 10000, 1000, 100, 10, 1]},
    )
    eis_res = EISAnalysis(
        technique="eis",
        solution_resistance_rs_ohm=2.5,
        charge_transfer_resistance_rct_ohm=27.5,
        characteristic_frequency_hz=100.0,
        equivalent_circuit="R_s + R_ct // CPE",
    )
    report = ElectrochemicalAnalysisReport(
        sample_id="EIS_Test_Sample",
        file_source="eis_spec.dta",
        technique="eis",
        geometry=ElectrodeGeometry(area_cm2=1.0),
        processed_data=norm_data,
        eis=eis_res,
    )

    fig = create_eis_figure(report)
    assert fig is not None
    assert "Nyquist Impedance Spectrum" in fig.layout.title.text
    assert fig.layout.yaxis.scaleanchor == "x"
    assert fig.layout.yaxis.scaleratio == 1


def test_plot_engine_create_dqdv_figure():
    """
    Verifies ElectrochemicalPlotEngine.create_dqdv_figure creates differential
    capacity plot with Savitzky-Golay smoothed curve.
    """
    from analysis.visualization.plot_engine import create_dqdv_figure

    v = np.linspace(1.0, 1.8, 50).tolist()
    q = np.linspace(0.0, 100.0, 50).tolist()
    norm_data = NormalizedElectrochemicalData(
        time_s=list(range(50)),
        display_time=[t / 3600.0 for t in range(50)],
        voltage_v=v,
        current_ma=[1.0] * 50,
        current_density_ma_cm2=[1.0] * 50,
        capacity_mah=q,
        areal_capacity_mah_cm2=q,
    )
    report = ElectrochemicalAnalysisReport(
        sample_id="FullCell_DQDV_Test",
        file_source="full_cell.csv",
        technique="galvanostatic_full_cell",
        geometry=ElectrodeGeometry(area_cm2=1.0),
        processed_data=norm_data,
    )

    fig = create_dqdv_figure(report, window_length=9, polyorder=2)
    assert fig is not None
    assert "Differential Capacity Analysis dQ/dV" in fig.layout.title.text
    assert len(fig.data) == 1
    assert fig.data[0].name == "dQ/dV (Savitzky-Golay)"


def test_lsv_pre_sweep_settling_trimming():
    """
    Verifies that initial pre-sweep settling/OCP points do not distort
    corrosion potential (E_corr) or Tafel slopes in LSVAnalyzer.
    """
    # 2 pre-sweep points followed by monotonic sweep from -3.0 V to +1.0 V
    v_presweep = [0.05, -2.80]
    v_sweep = np.linspace(-3.0, 1.0, 100).tolist()
    v_total = v_presweep + v_sweep

    # Current has zero-crossing during active sweep near -0.1 V
    j_presweep = [0.0, -0.7]
    j_sweep = (np.array(v_sweep) - (-0.1)) * 2.0  # zero crossing at -0.1 V
    j_total = j_presweep + j_sweep.tolist()

    norm_data = NormalizedElectrochemicalData(
        time_s=list(range(len(v_total))),
        display_time=[t / 3600.0 for t in range(len(v_total))],
        voltage_v=v_total,
        current_ma=j_total,
        current_density_ma_cm2=j_total,
        capacity_mah=[0.0] * len(v_total),
    )

    analyzer = LSVAnalyzer()
    res = analyzer.analyze(norm_data)

    # E_corr should be near -0.1 V, NOT the pre-sweep OCP point (0.05 V)
    assert res.corrosion_potential_ecorr_v is not None
    assert abs(res.corrosion_potential_ecorr_v - (-0.1)) < 0.05
    assert res.her_onset_potential_v is not None


def test_eis_analyzer_local_minimum_rct():
    """
    Verifies that EISAnalyzer identifies high-frequency solution resistance (R_s)
    and charge-transfer resistance (R_ct) from the first local minimum after the apex.
    """
    # Synthetic Nyquist semicircle (Rs = 5 Ohm, Rct = 50 Ohm, apex = 25 Ohm) followed by Warburg tail
    zr = [5.0, 6.0, 10.0, 20.0, 30.0, 45.0, 55.0, 58.0, 65.0, 80.0]
    zi = [0.2, 5.0, 15.0, 25.0, 20.0, 10.0, 2.0, 4.0, 12.0, 25.0]
    freq = [100000.0, 50000.0, 10000.0, 1000.0, 500.0, 100.0, 10.0, 5.0, 1.0, 0.1]

    raw = CyclerRawData(
        time_s=list(range(len(zr))),
        voltage_v=[0.0] * len(zr),
        current_ma=[0.0] * len(zr),
        frequency_hz=freq,
        z_real_ohm=zr,
        z_imag_ohm=zi,
        metadata={"source_file": "sample_eis.mpr"},
    )

    analyzer = EISAnalyzer()
    res = analyzer.analyze(raw)

    assert res.solution_resistance_rs_ohm == 5.0
    # Apex is at zr=20, zi=25; first local min after apex is at index 6 (zr=55, zi=2)
    # R_ct = 55 - 5 = 50 Ohm
    assert res.charge_transfer_resistance_rct_ohm == 50.0
    assert res.characteristic_frequency_hz == 1000.0


def test_detect_technique_improved():
    """
    Verifies that detect_technique accurately identifies LSV, EIS, CV, and
    Galvanostatic symmetric cell cycling.
    """
    # 1. EIS
    raw_eis = CyclerRawData(
        time_s=[1, 2, 3],
        voltage_v=[0.0, 0.0, 0.0],
        current_ma=[0.0, 0.0, 0.0],
        frequency_hz=[1000.0, 100.0, 10.0],
        z_real_ohm=[5.0, 25.0, 50.0],
        z_imag_ohm=[1.0, 20.0, 2.0],
    )
    assert detect_technique(raw_eis) == ElectrochemicalTechnique.EIS

    # 2. LSV (monotonic sweep with 1 pre-sweep point, range -3V to +3V)
    v_lsv = [0.05] + np.linspace(-3.0, 3.0, 50).tolist()
    raw_lsv = CyclerRawData(
        time_s=list(range(len(v_lsv))),
        voltage_v=v_lsv,
        current_ma=v_lsv,
    )
    assert detect_technique(raw_lsv) == ElectrochemicalTechnique.LSV

    # 3. Symmetric cell (alternating low voltage between -0.05 V and +0.05 V)
    v_symm = [0.05, -0.05] * 20
    raw_symm = CyclerRawData(
        time_s=list(range(len(v_symm))),
        voltage_v=v_symm,
        current_ma=[1.0, -1.0] * 20,
    )
    assert detect_technique(raw_symm) == ElectrochemicalTechnique.GALVANOSTATIC_SYMMETRIC


def test_cycle_summary_excel_and_csv_ingestion(tmp_path):
    """
    Verifies that Excel and CSV files containing 3-column cycle summary tables
    (Cycle number, Capacity, Coulombic Efficiency) are properly ingested and
    analyzed as Full Cell tests with dual-axis capacity and CE profiles.
    """
    import zipfile
    import xml.etree.ElementTree as ET

    # 1. Create a 3-column summary CSV file with 'Columbic Efficiency' spelling variant
    csv_file = tmp_path / "full_cell_summary.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("Cycle number,Capacity,Columbic Efficiency\n")
        f.write(",mAh g-1,%\n")
        f.write("1,170.0,100.0\n")
        f.write("2,165.0,95.5\n")
        f.write("3,160.0,98.2\n")
        f.write("4,155.0,99.1\n")
        f.write("5,150.0,99.5\n")

    report_csv = run_analysis(csv_file, technique_str="auto")
    assert report_csv.technique == ElectrochemicalTechnique.GALVANOSTATIC_FULL_CELL
    assert report_csv.full_cell is not None
    fc_csv = report_csv.full_cell
    assert fc_csv.total_cycles == 5
    assert fc_csv.initial_discharge_capacity_mah_g == 170.0
    assert fc_csv.final_discharge_capacity_mah_g == 150.0
    assert pytest.approx(fc_csv.capacity_retention_percent, abs=0.1) == 88.24
    assert len(fc_csv.coulombic_efficiencies) == 5
    assert fc_csv.coulombic_efficiencies[0] == 100.0
    assert fc_csv.coulombic_efficiencies[1] == 95.5

    # 2. Verify dual-axis HTML dashboard generation
    plot_engine = ElectrochemicalPlotEngine(output_dir=tmp_path)
    html = plot_engine._render_full_cell_dashboard(report_csv)
    assert "yaxis2" in html
    assert "Coulombic Efficiency" in html
    assert "Discharge Capacity" in html
    assert "plot_cap" in html

    # 3. If the user's uploaded temp xlsx exists, test it directly
    user_temp_xlsx = Path(r"C:\Users\10073\AppData\Local\Temp\tmpoxv2snar.xlsx")
    if user_temp_xlsx.is_file():
        report_xlsx = run_analysis(user_temp_xlsx, technique_str="auto")
        assert report_xlsx.technique == ElectrochemicalTechnique.GALVANOSTATIC_FULL_CELL
        assert report_xlsx.full_cell is not None
        assert report_xlsx.full_cell.total_cycles == 36
        assert len(report_xlsx.full_cell.coulombic_efficiencies) == 36
        assert len(report_xlsx.full_cell.discharge_capacities_mah_g) == 36
        html_xlsx = plot_engine._render_full_cell_dashboard(report_xlsx)
        assert "yaxis2" in html_xlsx
        assert "plot_cap" in html_xlsx


def test_generate_electrochemical_dataset_csv():
    """
    Verifies that generate_electrochemical_dataset_csv creates clean, Origin/Prism compatible
    timeseries CSV with proper headers and numerical values.
    """
    import csv
    import io

    norm_data = NormalizedElectrochemicalData(
        time_s=[0.0, 1800.0, 3600.0, 7200.0],
        voltage_v=[0.035, 0.035, -0.035, 0.035],
        current_ma=[1.0, 1.0, -1.0, 1.0],
        current_density_ma_cm2=[0.884, 0.884, -0.884, 0.884],
        capacity_mah=[0.0, 0.5, 1.0, 2.0],
    )
    report = ElectrochemicalAnalysisReport(
        sample_id="SAMPLE_TEST_CSV",
        file_source="test_cycling.mpt",
        technique="galvanostatic_symmetric",
        geometry=ElectrodeGeometry(area_cm2=1.13097),
        processed_data=norm_data,
    )

    csv_output = generate_electrochemical_dataset_csv(report)
    reader = list(csv.reader(io.StringIO(csv_output)))
    assert len(reader) == 5  # 1 header row + 4 data rows
    assert reader[0] == ["Time (h)", "Potential (V)", "Current (mA)", "Current Density (mA/cm²)", "Capacity (mAh)"]
    assert reader[1] == ["0.0", "0.035", "1.0", "0.884", "0.0"]
    assert reader[2] == ["0.5", "0.035", "1.0", "0.884", "0.5"]
    assert reader[3] == ["1.0", "-0.035", "-1.0", "-0.884", "1.0"]
    assert reader[4] == ["2.0", "0.035", "1.0", "0.884", "2.0"]


def test_generate_electrochemical_dataset_xlsx_structure():
    """
    Verifies that generate_electrochemical_dataset_xlsx creates a valid multi-sheet
    Excel workbook containing Timeseries_Data, Cycle_Summary, and Technique_Diagnostics.
    """
    import zipfile
    import io

    norm_data = NormalizedElectrochemicalData(
        time_s=[0.0, 3600.0, 7200.0],
        voltage_v=[0.035, -0.035, 0.035],
        current_ma=[1.0, -1.0, 1.0],
        current_density_ma_cm2=[0.884, -0.884, 0.884],
        capacity_mah=[0.0, 1.0, 2.0],
    )
    symm = SymmetricCellAnalysis(
        total_cycles=2,
        total_lifespan_hours=2.0,
        mean_overpotential_mv=35.0,
        mean_hysteresis_mv=70.0,
        cycle_metrics=[
            SymmetricCellCycleMetric(
                cycle_number=1,
                stripping_plateau_v=0.035,
                plating_plateau_v=-0.035,
                overpotential_mv=35.0,
                voltage_hysteresis_mv=70.0,
                areal_capacity_mah_cm2=0.884,
            ),
            SymmetricCellCycleMetric(
                cycle_number=2,
                stripping_plateau_v=0.036,
                plating_plateau_v=-0.036,
                overpotential_mv=36.0,
                voltage_hysteresis_mv=72.0,
                areal_capacity_mah_cm2=0.884,
            ),
        ],
        active_cycling_metrics={"active_lifespan_hours": 2.0, "active_cycles": 2},
    )
    report = ElectrochemicalAnalysisReport(
        sample_id="SAMPLE_SYMM_XLSX",
        file_source="symm.mpr",
        technique="galvanostatic_symmetric",
        geometry=ElectrodeGeometry(area_cm2=1.13097),
        processed_data=norm_data,
        symmetric_cell=symm,
    )

    xlsx_bytes = generate_electrochemical_dataset_xlsx(report)
    assert isinstance(xlsx_bytes, bytes)
    assert len(xlsx_bytes) > 0

    with zipfile.ZipFile(io.BytesIO(xlsx_bytes)) as z:
        # Check OpenXML workbook files
        assert "xl/workbook.xml" in z.namelist()
        wb_xml = z.read("xl/workbook.xml").decode("utf-8")
        assert 'name="Timeseries_Data"' in wb_xml
        assert 'name="Cycle_Summary"' in wb_xml
        assert 'name="Technique_Diagnostics"' in wb_xml

        # Sheet 1: Timeseries
        sheet1_xml = z.read("xl/worksheets/sheet1.xml").decode("utf-8")
        assert "Time (h)" in sheet1_xml
        assert "Potential (V)" in sheet1_xml

        # Sheet 2: Cycle Summary
        sheet2_xml = z.read("xl/worksheets/sheet2.xml").decode("utf-8")
        assert "Cycle Number" in sheet2_xml
        assert "Plating Plateau (V)" in sheet2_xml
        assert "Overpotential (mV)" in sheet2_xml

        # Sheet 3: Diagnostics
        sheet3_xml = z.read("xl/worksheets/sheet3.xml").decode("utf-8")
        assert "SAMPLE_SYMM_XLSX" in sheet3_xml
        assert "Total Lifespan" in sheet3_xml
        assert "Mean Overpotential" in sheet3_xml


def test_generate_electrochemical_dataset_xlsx_all_techniques():
    """
    Verifies technique-adaptive diagnostics sheet generation for Full Cell, CV, LSV, and EIS.
    """
    import zipfile
    import io

    # 1. Full Cell
    fc_report = ElectrochemicalAnalysisReport(
        sample_id="FC_01",
        file_source="fc.csv",
        technique="galvanostatic_full_cell",
        geometry=ElectrodeGeometry(area_cm2=1.0, active_mass_mg=2.0),
        processed_data=NormalizedElectrochemicalData(
            time_s=[0.0, 3600.0],
            voltage_v=[1.6, 1.0],
            current_ma=[-1.0, -1.0],
            current_density_ma_cm2=[-1.0, -1.0],
            capacity_mah=[0.0, 1.0],
        ),
        full_cell=FullCellAnalysis(
            total_cycles=5,
            initial_discharge_capacity_mah_g=150.0,
            final_discharge_capacity_mah_g=140.0,
            capacity_retention_percent=93.3,
            average_coulombic_efficiency_percent=99.2,
            cycle_numbers=[1, 2, 3, 4, 5],
            discharge_capacities_mah_g=[150.0, 148.0, 145.0, 142.0, 140.0],
            coulombic_efficiencies=[98.5, 99.0, 99.2, 99.5, 99.6],
        ),
    )
    fc_xlsx = generate_electrochemical_dataset_xlsx(fc_report)
    with zipfile.ZipFile(io.BytesIO(fc_xlsx)) as z:
        s3 = z.read("xl/worksheets/sheet3.xml").decode("utf-8")
        assert "Initial Discharge Capacity" in s3
        assert "Capacity Retention" in s3
        assert "Average Coulombic Efficiency" in s3

    # 2. CV
    cv_report = ElectrochemicalAnalysisReport(
        sample_id="CV_01",
        file_source="cv.mpt",
        technique="cv",
        geometry=ElectrodeGeometry(area_cm2=1.0),
        processed_data=NormalizedElectrochemicalData(
            time_s=[0.0, 10.0],
            voltage_v=[-0.2, 0.4],
            current_ma=[-5.0, 8.0],
            current_density_ma_cm2=[-5.0, 8.0],
            capacity_mah=[0.0, 0.0],
        ),
        cv=CVAnalysis(
            anodic_peak_potential_epa_v=0.12,
            cathodic_peak_potential_epc_v=-0.08,
            peak_separation_delta_ep_mv=200.0,
            anodic_peak_current_ipa_ma=8.0,
            cathodic_peak_current_ipc_ma=-5.0,
            current_ratio_ipa_ipc=1.6,
        ),
    )
    cv_xlsx = generate_electrochemical_dataset_xlsx(cv_report)
    with zipfile.ZipFile(io.BytesIO(cv_xlsx)) as z:
        s3 = z.read("xl/worksheets/sheet3.xml").decode("utf-8")
        assert "Peak Separation" in s3
        assert "Anodic Peak Potential" in s3

    # 3. LSV
    lsv_report = ElectrochemicalAnalysisReport(
        sample_id="LSV_01",
        file_source="lsv.dta",
        technique="lsv",
        geometry=ElectrodeGeometry(area_cm2=1.0),
        processed_data=NormalizedElectrochemicalData(
            time_s=[0.0, 60.0],
            voltage_v=[-1.2, 0.2],
            current_ma=[0.1, 10.0],
            current_density_ma_cm2=[0.1, 10.0],
            capacity_mah=[0.0, 0.0],
        ),
        lsv=LSVAnalysis(
            corrosion_potential_ecorr_v=-0.98,
            corrosion_current_jcorr_ma_cm2=0.012,
            tafel_slope_cathodic_mv_per_dec=120.0,
            tafel_slope_anodic_mv_per_dec=85.0,
            her_onset_potential_v=-1.15,
        ),
    )
    lsv_xlsx = generate_electrochemical_dataset_xlsx(lsv_report)
    with zipfile.ZipFile(io.BytesIO(lsv_xlsx)) as z:
        s3 = z.read("xl/worksheets/sheet3.xml").decode("utf-8")
        assert "Corrosion Potential" in s3
        assert "Cathodic Tafel Slope" in s3

    # 4. EIS
    eis_report = ElectrochemicalAnalysisReport(
        sample_id="EIS_01",
        file_source="eis.mpr",
        technique="eis",
        geometry=ElectrodeGeometry(area_cm2=1.0),
        processed_data=NormalizedElectrochemicalData(
            time_s=[0.0, 10.0],
            voltage_v=[0.0, 0.0],
            current_ma=[0.0, 0.0],
            current_density_ma_cm2=[0.0, 0.0],
            capacity_mah=[0.0, 0.0],
        ),
        eis=EISAnalysis(
            solution_resistance_rs_ohm=4.5,
            charge_transfer_resistance_rct_ohm=42.0,
            characteristic_frequency_hz=1250.0,
        ),
    )
    eis_xlsx = generate_electrochemical_dataset_xlsx(eis_report)
    with zipfile.ZipFile(io.BytesIO(eis_xlsx)) as z:
        s3 = z.read("xl/worksheets/sheet3.xml").decode("utf-8")
        assert "Solution Resistance" in s3
        assert "Charge Transfer Resistance" in s3


# ==============================================================================
# 11. Integrated Full-Story Laboratory Experiment Dossier (Multi-Sheet XLSX & Executive DOCX)
# ==============================================================================

import io
import zipfile
import xml.etree.ElementTree as ET
from models import (
    CampaignResult,
    DispensingSchedule,
    RoboticVial,
    DoEPlan,
    ConcentrationAnchor,
    FormulationPoint,
    FormulatedCellResult,
)
from export_engine import (
    generate_campaign_summary_xlsx,
    generate_campaign_summary_docx,
    generate_campaign_summary_txt,
)


@pytest.fixture
def sample_campaign():
    """Create a fully-populated mock CampaignResult bridging Modules 1, 2, and 3."""
    anchor = ConcentrationAnchor(
        additive_name="Triethyl Phosphate",
        published_concentration=25.0,
        published_unit="mM",
        parent_electrolyte="2M ZnSO4",
        parent_salt="ZnSO4",
        parent_molarity=2.0,
        target_electrolyte="3M Zn(OTf)2",
        target_electrolyte_salt="Zn(OTf)2",
        target_salt_molarity=3.0,
        scaling_factor=1.25,
        shifted_anchor_conc=31.25,
        anchor_unit="mM",
        min_conc=10.0,
        max_conc=50.0,
        solubility_limit_mM=60.0,
        solubility_ceiling_mM=48.0,
        is_capped_by_solubility=True,
        electrolyte_adjustment_rationale="Zn2+ desolvation sheath restructuring and dipole coordination",
        functional_groups=["hydroxyl", "ether"],
    )

    doe_plan = DoEPlan(
        plan_id="DOE_PLAN_001",
        additive_name="Triethyl Phosphate",
        target_electrolyte_salt="Zn(OTf)2",
        target_salt_default_molarity=3.0,
        target_salt_molarity_range=[1.0, 4.0],
        method="sobol",
        anchor=anchor,
        parameters=[],
        formulations=[
            FormulationPoint(sample_id="SMP_01", additive_conc_mM=20.0, salt_conc_M=2.5, point_type="corner"),
            FormulationPoint(sample_id="SMP_02", additive_conc_mM=31.25, salt_conc_M=3.0, point_type="center"),
            FormulationPoint(sample_id="SMP_03", additive_conc_mM=45.0, salt_conc_M=3.5, point_type="corner"),
        ],
        electrolyte_adjustment_rationale="Solvation coordination tuning",
    )

    vials = [
        RoboticVial(
            vial_id=f"A{i}",
            rack_format="24-vial rack",
            row="A",
            col=i,
            additive_conc_mM=float(15 * i),
            salt_conc_M=2.0 + 0.3 * i,
            target_volume_uL=1000.0,
            additive_stock_uL=float(100 * i),
            salt_stock_uL=500.0,
            water_uL=1000.0 - float(100 * i) - 500.0,
            slot=2,
            aspiration_rate_ul_s=40.0,
            dispense_rate_ul_s=45.0,
        )
        for i in range(1, 6)
    ]

    sched = DispensingSchedule(
        schedule_id="SCHED_001",
        target_electrolyte_salt="Zn(OTf)2",
        electrolyte_adjustment_rationale="Solvation coordination tuning",
        rack_format="24-vial rack",
        total_vials=5,
        total_volume_per_vial_uL=1000.0,
        additive_stock_conc_mM=200.0,
        salt_stock_conc_M=5.0,
        salt_stock_name="Zn(OTf)2 5M Stock",
        vials=vials,
        ot2_deck_slot=2,
        default_aspiration_rate_ul_s=40.0,
        default_dispense_rate_ul_s=45.0,
    )

    # Symmetric cell analysis report for champion
    champion_report = ElectrochemicalAnalysisReport(
        sample_id="SMP_02",
        file_source="smp_02_cycling.mpr",
        technique="galvanostatic_symmetric",
        geometry=ElectrodeGeometry(area_cm2=1.13, anode_material="Zn foil"),
        processed_data=NormalizedElectrochemicalData(
            time_s=[0.0, 3600.0, 7200.0],
            voltage_v=[0.02, -0.02, 0.02],
            current_ma=[1.13, -1.13, 1.13],
            current_density_ma_cm2=[1.0, -1.0, 1.0],
            capacity_mah=[0.5, 0.5, 0.5],
        ),
        symmetric_cell=SymmetricCellAnalysis(
            total_cycles=92,
            total_lifespan_hours=185.5,
            mean_overpotential_mv=42.1,
            mean_hysteresis_mv=45.0,
            failure_mode="dendritic_short",
            short_circuit_detected=True,
            soft_short_detected=False,
            failure_cycle=92,
            failure_time_hours=185.5,
            active_cycling_metrics={
                "active_lifespan_hours": 185.5,
                "mean_overpotential_mv": 42.1,
                "mean_hysteresis_mv": 45.0,
                "active_cycles": 92,
            },
            total_test_summary={
                "failure_mode": "dendritic_short",
                "short_circuit_detected": True,
                "failure_time_hours": 185.5,
            },
        ),
    )

    cell_1 = FormulatedCellResult(
        sample_id="SMP_01",
        vial_id="A1",
        additive_conc_mM=20.0,
        salt_conc_M=2.5,
        performance_score=0.742,
        is_pareto_optimal=False,
        iteration_round=1,
    )
    cell_2 = FormulatedCellResult(
        sample_id="SMP_02",
        vial_id="A2",
        additive_conc_mM=31.25,
        salt_conc_M=3.0,
        performance_score=0.915,
        is_pareto_optimal=True,
        iteration_round=1,
        analysis_report=champion_report,
    )
    cell_3 = FormulatedCellResult(
        sample_id="SMP_03",
        vial_id="A3",
        additive_conc_mM=45.0,
        salt_conc_M=3.5,
        performance_score=0.860,
        is_pareto_optimal=True,
        iteration_round=2,
    )

    campaign = CampaignResult(
        campaign_id="CAMP_EXP_001",
        target_salt="Zn(OTf)2",
        formulation_plan=doe_plan,
        dispensing_schedule=sched,
        cell_results=[cell_1, cell_2, cell_3],
        pareto_optimal_candidates=[cell_2, cell_3],
        best_candidate=cell_2,
        output_dir="./test_output",
        created_at="2026-10-15T18:00:00Z",
        iteration_round=2,
        total_iterations=2,
        history=[
            {"iteration_round": 1, "new_samples": 2, "total_cells": 2, "best_sample_id": "SMP_02", "best_score": 0.915},
            {"iteration_round": 2, "new_samples": 1, "total_cells": 3, "best_sample_id": "SMP_02", "best_score": 0.915},
        ],
    )
    return campaign


def test_generate_campaign_summary_xlsx_five_sheets(sample_campaign):
    """
    Verify generate_campaign_summary_xlsx produces a valid OpenXML .xlsx archive
    containing exactly the 5 cohesive, publication-structured worksheets:
      1_Executive_Overview, 2_Literature_Baseline, 3_Robotic_Dispensing,
      4_Evaluated_Cells, 5_Diagnostic_Summary.
    """
    xlsx_bytes = generate_campaign_summary_xlsx(sample_campaign)
    assert len(xlsx_bytes) > 0

    with zipfile.ZipFile(io.BytesIO(xlsx_bytes)) as z:
        # Check OpenXML core package structure
        assert "[Content_Types].xml" in z.namelist()
        assert "_rels/.rels" in z.namelist()
        assert "xl/workbook.xml" in z.namelist()
        assert "xl/styles.xml" in z.namelist()

        # Parse workbook.xml to inspect sheet definitions
        wb_xml = z.read("xl/workbook.xml")
        root = ET.fromstring(wb_xml)
        ns = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        sheets = root.findall(".//ns:sheet", ns)
        sheet_names = [s.attrib["name"] for s in sheets]

        expected_sheets = [
            "1_Executive_Overview",
            "2_Literature_Baseline",
            "3_Robotic_Dispensing",
            "4_Evaluated_Cells",
            "5_Diagnostic_Summary",
        ]
        assert sheet_names == expected_sheets

        # Sheet 1: Executive Overview
        s1 = z.read("xl/worksheets/sheet1.xml").decode("utf-8")
        assert "CAMP_EXP_001" in s1
        assert "Zn(OTf)2" in s1
        assert "SMP_02" in s1
        assert "0.915" in s1

        # Sheet 2: Literature Baseline
        s2 = z.read("xl/worksheets/sheet2.xml").decode("utf-8")
        assert "Triethyl Phosphate" in s2
        assert "ZnSO4" in s2
        assert "25" in s2
        assert "1.25" in s2
        assert "48" in s2  # Delaney ESOL ceiling
        assert "desolvation sheath" in s2

        # Sheet 3: Robotic Dispensing
        s3 = z.read("xl/worksheets/sheet3.xml").decode("utf-8")
        assert "A1" in s3
        assert "A5" in s3
        assert "24-vial rack" in s3

        # Sheet 4: Evaluated Cells
        s4 = z.read("xl/worksheets/sheet4.xml").decode("utf-8")
        assert "SMP_01" in s4
        assert "SMP_02" in s4
        assert "SMP_03" in s4
        assert "Dendritic Short Circuit" in s4
        assert "dendritic_short" not in s4

        # Sheet 5: Diagnostic Summary
        s5 = z.read("xl/worksheets/sheet5.xml").decode("utf-8")
        assert "SMP_02" in s5
        assert "galvanostatic_symmetric" in s5
        assert "185.5" in s5
        assert "42.1" in s5


def test_generate_campaign_summary_docx_five_sections_and_tables(sample_campaign):
    """
    Verify generate_campaign_summary_docx produces a valid OpenXML .docx document
    containing all 5 formal numbered sections and structured report tables.
    """
    docx_bytes = generate_campaign_summary_docx(sample_campaign)
    assert len(docx_bytes) > 0

    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as z:
        assert "[Content_Types].xml" in z.namelist()
        assert "_rels/.rels" in z.namelist()
        assert "word/document.xml" in z.namelist()

        doc_xml = z.read("word/document.xml")
        root = ET.fromstring(doc_xml)  # Validates XML syntax
        xml_text = doc_xml.decode("utf-8")

        # Verify all 5 formal numbered headings
        assert "1. Executive Summary &amp; Autonomous MAP Campaign Overview" in xml_text or "1. Executive Summary" in xml_text
        assert "2. Module 1 Literature Mining, Delaney ESOL Solubility Capping &amp; Scaling" in xml_text or "2. Module 1" in xml_text
        assert "3. Module 2 DoE Formulation Matrix &amp; Opentrons OT-2 Pipetting Protocol" in xml_text or "3. Module 2" in xml_text
        assert "4. Module 3 Battery Cycling Evaluation, Polarization &amp; Failure Diagnostics" in xml_text or "4. Module 3" in xml_text
        assert "5. Top Recommended Formulation &amp; Future Screening Recommendations" in xml_text or "5. Top Recommended" in xml_text

        # Verify structured OpenXML tables exist (<w:tbl>)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        tables = root.findall(".//w:tbl", ns)
        assert len(tables) >= 5, f"Expected at least 5 structured tables in DOCX, found {len(tables)}"

        # Verify key narrative contents
        assert "CAMP_EXP_001" in xml_text
        assert "Zn(OTf)2" in xml_text
        assert "SMP_02" in xml_text
        assert "Triethyl Phosphate" in xml_text
        assert "Dendritic Short Circuit" in xml_text
        assert "dendritic_short" not in xml_text


def test_generate_campaign_summary_txt_structure(sample_campaign):
    """
    Verify generate_campaign_summary_txt generates the cohesive narrative
    across all 5 sections in clean text format.
    """
    txt_str = generate_campaign_summary_txt(sample_campaign)
    assert "SECTION 1: EXECUTIVE SUMMARY & AUTONOMOUS MAP CAMPAIGN OVERVIEW" in txt_str
    assert "SECTION 2: MODULE 1 LITERATURE MINING, DELANEY ESOL SOLUBILITY CAPPING & SCALING" in txt_str
    assert "SECTION 3: MODULE 2 DoE FORMULATION MATRIX & OPENTRONS OT-2 PIPETTING PROTOCOL" in txt_str
    assert "SECTION 4: MODULE 3 BATTERY CYCLING EVALUATION, POLARIZATION & FAILURE DIAGNOSTICS" in txt_str
    assert "SECTION 5: TOP RECOMMENDED FORMULATION & FUTURE SCREENING RECOMMENDATIONS" in txt_str
    assert "CAMP_EXP_001" in txt_str
    assert "SMP_02" in txt_str
    assert "Dendritic Short Circuit" in txt_str
    assert "dendritic_short" not in txt_str


def test_generate_campaign_summary_xlsx_adaptive_techniques():
    """
    Verify Sheet 5 (5_Diagnostic_Summary) adapts dynamically across different
    electrochemical techniques (asymmetric, full-cell, CV, LSV, EIS).
    """
    # 1. Asymmetric Half Cell
    asym_report = ElectrochemicalAnalysisReport(
        sample_id="ASYM_01",
        file_source="asym.mpr",
        technique="galvanostatic_asymmetric",
        geometry=ElectrodeGeometry(area_cm2=1.0),
        processed_data=NormalizedElectrochemicalData(
            time_s=[0.0, 100.0],
            voltage_v=[0.0, 0.5],
            current_ma=[1.0, 1.0],
            current_density_ma_cm2=[1.0, 1.0],
            capacity_mah=[0.1, 0.1],
        ),
        asymmetric_cell=AsymmetricHalfCellAnalysis(
            total_cycles=50,
            average_ce_percent=99.2,
            final_ce_percent=99.5,
            cumulative_capacity_mah_cm2=50.0,
            mean_nucleation_overpotential_mv=18.5,
        ),
    )
    anchor = ConcentrationAnchor(
        additive_name="Glucose",
        published_concentration=10.0,
        parent_electrolyte="2M ZnSO4",
        parent_salt="ZnSO4",
        parent_molarity=2.0,
        target_electrolyte="2M ZnSO4",
        target_electrolyte_salt="ZnSO4",
        target_salt_molarity=2.0,
        scaling_factor=1.0,
        shifted_anchor_conc=10.0,
        min_conc=5.0,
        max_conc=20.0,
        electrolyte_adjustment_rationale="Baseline",
    )
    doe_plan = DoEPlan(
        plan_id="DOE_002",
        additive_name="Glucose",
        target_electrolyte_salt="ZnSO4",
        target_salt_default_molarity=2.0,
        target_salt_molarity_range=[1.0, 3.0],
        method="lhs",
        anchor=anchor,
        parameters=[],
        formulations=[],
        electrolyte_adjustment_rationale="Baseline",
    )
    sched = DispensingSchedule(
        schedule_id="SCHED_002",
        target_electrolyte_salt="ZnSO4",
        electrolyte_adjustment_rationale="Baseline",
        rack_format="24-vial rack",
        total_vials=0,
        total_volume_per_vial_uL=1000.0,
        additive_stock_conc_mM=100.0,
        salt_stock_conc_M=2.0,
        salt_stock_name="ZnSO4 Stock",
        vials=[],
    )
    cand = FormulatedCellResult(
        sample_id="ASYM_01",
        vial_id="A1",
        additive_conc_mM=10.0,
        salt_conc_M=2.0,
        performance_score=0.99,
        analysis_report=asym_report,
    )
    campaign = CampaignResult(
        campaign_id="CAMP_ASYM",
        target_salt="ZnSO4",
        formulation_plan=doe_plan,
        dispensing_schedule=sched,
        cell_results=[cand],
        best_candidate=cand,
        output_dir="./test_output",
    )

    xlsx_bytes = generate_campaign_summary_xlsx(campaign)
    with zipfile.ZipFile(io.BytesIO(xlsx_bytes)) as z:
        s5 = z.read("xl/worksheets/sheet5.xml").decode("utf-8")
        assert "Coulombic Efficiency" in s5
        assert "Nucleation Overpotential" in s5

    # 2. CV
    cv_report = ElectrochemicalAnalysisReport(
        sample_id="CV_01",
        file_source="cv.mpr",
        technique="cv",
        geometry=ElectrodeGeometry(area_cm2=1.0),
        processed_data=NormalizedElectrochemicalData(
            time_s=[0.0, 10.0],
            voltage_v=[-0.2, 0.2],
            current_ma=[-1.0, 1.0],
            current_density_ma_cm2=[-1.0, 1.0],
            capacity_mah=[0.0, 0.0],
        ),
        cv=CVAnalysis(
            anodic_peak_potential_epa_v=0.15,
            cathodic_peak_potential_epc_v=-0.12,
            peak_separation_delta_ep_mv=270.0,
            anodic_peak_current_ipa_ma=5.2,
            cathodic_peak_current_ipc_ma=-4.8,
        ),
    )
    cand.sample_id = "CV_01"
    cand.analysis_report = cv_report
    xlsx_bytes_cv = generate_campaign_summary_xlsx(campaign)
    with zipfile.ZipFile(io.BytesIO(xlsx_bytes_cv)) as z:
        s5 = z.read("xl/worksheets/sheet5.xml").decode("utf-8")
        assert "Peak Potential Separation" in s5
        assert "270" in s5

    # 3. EIS
    eis_report = ElectrochemicalAnalysisReport(
        sample_id="EIS_01",
        file_source="eis.mpr",
        technique="eis",
        geometry=ElectrodeGeometry(area_cm2=1.0),
        processed_data=NormalizedElectrochemicalData(
            time_s=[0.0, 10.0],
            voltage_v=[0.0, 0.0],
            current_ma=[0.0, 0.0],
            current_density_ma_cm2=[0.0, 0.0],
            capacity_mah=[0.0, 0.0],
        ),
        eis=EISAnalysis(
            solution_resistance_rs_ohm=3.8,
            charge_transfer_resistance_rct_ohm=35.2,
            characteristic_frequency_hz=1450.0,
        ),
    )
    cand.sample_id = "EIS_01"
    cand.analysis_report = eis_report
    xlsx_bytes_eis = generate_campaign_summary_xlsx(campaign)
    with zipfile.ZipFile(io.BytesIO(xlsx_bytes_eis)) as z:
        s5 = z.read("xl/worksheets/sheet5.xml").decode("utf-8")
        assert "Solution Resistance" in s5
        assert "Charge Transfer Resistance" in s5


def test_generate_campaign_summary_empty_and_none_safe():
    """
    Verify report generators handle empty campaigns or None attributes gracefully
    without raising unhandled exceptions or generating corrupted XML.
    """
    anchor = ConcentrationAnchor(
        additive_name="None",
        published_concentration=0.0,
        parent_electrolyte="2M ZnSO4",
        parent_salt="ZnSO4",
        parent_molarity=2.0,
        target_electrolyte="2M ZnSO4",
        target_electrolyte_salt="ZnSO4",
        target_salt_molarity=2.0,
        scaling_factor=1.0,
        shifted_anchor_conc=0.0,
        min_conc=0.0,
        max_conc=0.0,
        electrolyte_adjustment_rationale="None",
    )
    doe_plan = DoEPlan(
        plan_id="DOE_EMPTY",
        additive_name="None",
        target_electrolyte_salt="ZnSO4",
        target_salt_default_molarity=2.0,
        target_salt_molarity_range=[1.0, 3.0],
        method="lhs",
        anchor=anchor,
        parameters=[],
        formulations=[],
        electrolyte_adjustment_rationale="None",
    )
    sched = DispensingSchedule(
        schedule_id="SCHED_EMPTY",
        target_electrolyte_salt="ZnSO4",
        electrolyte_adjustment_rationale="None",
        rack_format="24-vial rack",
        total_vials=0,
        total_volume_per_vial_uL=1000.0,
        additive_stock_conc_mM=0.0,
        salt_stock_conc_M=0.0,
        salt_stock_name="Stock",
        vials=[],
    )
    campaign = CampaignResult(
        campaign_id="CAMP_EMPTY",
        target_salt="ZnSO4",
        formulation_plan=doe_plan,
        dispensing_schedule=sched,
        cell_results=[],
        best_candidate=None,
        output_dir="./test_output",
        history=[],
    )

    xlsx = generate_campaign_summary_xlsx(campaign)
    assert len(xlsx) > 0
    with zipfile.ZipFile(io.BytesIO(xlsx)) as z:
        wb_xml = z.read("xl/workbook.xml")
        ET.fromstring(wb_xml)
        for i in range(1, 6):
            s_xml = z.read(f"xl/worksheets/sheet{i}.xml")
            ET.fromstring(s_xml)

    docx = generate_campaign_summary_docx(campaign)
    assert len(docx) > 0
    with zipfile.ZipFile(io.BytesIO(docx)) as z:
        doc_xml = z.read("word/document.xml")
        ET.fromstring(doc_xml)

    txt = generate_campaign_summary_txt(campaign)
    assert "SECTION 1" in txt
    assert "END OF LABORATORY EXPERIMENT DOSSIER" in txt



