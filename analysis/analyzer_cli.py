"""
Autonomous Electrochemical Characterization CLI & Orchestration Engine.
Phase 3 of the Materials Acceleration Platform (MAP) for aqueous Zinc-Ion Batteries.

Usage:
    python -m analysis.analyzer_cli --file data/cycler_export.csv --technique auto --area-cm2 1.13
    python -m analysis.analyzer_cli --watch-dir ./cycler_inbox --output-json reports/summary.json
"""

import sys
import json
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
import numpy as np

from models import (
    ElectrochemicalTechnique,
    ElectrodeGeometry,
    CyclerRawData,
    NormalizedElectrochemicalData,
    ElectrochemicalAnalysisReport,
)
from analysis.parsers import parse_cycler_file
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
from analysis.ingestion.watchdog import CyclerWatchdog

logger = logging.getLogger(__name__)


def detect_technique(raw_data: CyclerRawData) -> ElectrochemicalTechnique:
    """Infers electrochemical technique automatically from raw data profiles."""
    fn = str(raw_data.metadata.get("source_file", "")).lower()

    if raw_data.frequency_hz is not None and raw_data.z_real_ohm is not None:
        if len(raw_data.z_real_ohm) > 0:
            return ElectrochemicalTechnique.EIS
    if any(k in fn for k in ["_eis", "-eis", ".eis", "peis", "geis"]) and raw_data.z_real_ohm and len(raw_data.z_real_ohm) > 0:
        return ElectrochemicalTechnique.EIS

    # Check for Cycle Summary Table (e.g. Cycle number, Capacity, Coulombic Efficiency)
    if raw_data.metadata.get("is_cycle_summary"):
        return ElectrochemicalTechnique.GALVANOSTATIC_FULL_CELL
    if raw_data.cycle_number is not None and len(raw_data.cycle_number) > 0:
        if raw_data.capacity_discharge_mah is not None or raw_data.coulombic_efficiency is not None:
            if not raw_data.voltage_v or len(raw_data.voltage_v) == 0 or np.all(np.array(raw_data.voltage_v) >= 0.0):
                if len(raw_data.time_s) == len(raw_data.cycle_number):
                    return ElectrochemicalTechnique.GALVANOSTATIC_FULL_CELL

    v = raw_data.voltage_v
    i = raw_data.current_ma
    if not v or not i or len(v) < 2:
        return ElectrochemicalTechnique.UNKNOWN

    v_arr = np.array(v, dtype=float)
    min_v = float(np.min(v_arr))
    max_v = float(np.max(v_arr))
    delta_v = max_v - min_v

    # Flat potential plateau / unipolar constant-potential deposition chronopotentiometry
    if delta_v < 0.02:
        return ElectrochemicalTechnique.UNKNOWN

    v_diffs = np.diff(v_arr)
    act_diffs = v_diffs[np.abs(v_diffs) > 0.0002]
    if len(act_diffs) > 10:
        pos_ratio = float(np.sum(act_diffs > 0)) / len(act_diffs)
        neg_ratio = float(np.sum(act_diffs < 0)) / len(act_diffs)
        is_lsv_monotonic = (pos_ratio > 0.90 or neg_ratio > 0.90) and (delta_v > 0.05)
    else:
        is_lsv_monotonic = False

    if is_lsv_monotonic or (any(k in fn for k in ["_lsv", "-lsv", ".lsv", "lsv_"]) and delta_v > 0.05):
        return ElectrochemicalTechnique.LSV

    if any(k in fn for k in ["_cv", "-cv", ".cv", "cv_"]):
        return ElectrochemicalTechnique.CV

    if min_v < -0.005 and max_v > 0.005 and abs(min_v) < 0.5 and abs(max_v) < 0.5 and abs(min_v + max_v) < 0.2:
        return ElectrochemicalTechnique.GALVANOSTATIC_SYMMETRIC
    elif min_v > 0.4 and max_v > 0.8:
        return ElectrochemicalTechnique.GALVANOSTATIC_FULL_CELL
    elif min_v < -0.01 and max_v > 0.2:
        return ElectrochemicalTechnique.GALVANOSTATIC_ASYMMETRIC
    else:
        return ElectrochemicalTechnique.CV


def truncate_normalized_data(
    data: NormalizedElectrochemicalData,
    max_time_hours: Optional[float] = None,
    max_cycles: Optional[int] = None,
    time_window: Optional[Union[List[float], Tuple[float, float]]] = None,
) -> NormalizedElectrochemicalData:
    """Manually truncates normalized electrochemical data by time, window, or cycle count."""
    n = len(data.time_s)
    if n == 0:
        return data

    mask = np.ones(n, dtype=bool)
    t_arr = np.array(data.time_s, dtype=float)

    if max_time_hours is not None:
        mask = mask & (t_arr <= max_time_hours * 3600.0)

    if time_window is not None and len(time_window) == 2:
        min_s = float(time_window[0]) * 3600.0
        max_s = float(time_window[1]) * 3600.0
        mask = mask & (t_arr >= min_s) & (t_arr <= max_s)

    if max_cycles is not None and data.cycle_number is not None and len(data.cycle_number) == n:
        c_arr = np.array(data.cycle_number)
        mask = mask & (c_arr <= max_cycles)

    if np.all(mask):
        return data

    indices = np.where(mask)[0]

    def filter_list(lst):
        if lst is None or len(lst) != n:
            return lst
        return [lst[i] for i in indices]

    return NormalizedElectrochemicalData(
        time_s=filter_list(data.time_s),
        voltage_v=filter_list(data.voltage_v),
        current_ma=filter_list(data.current_ma),
        current_density_ma_cm2=filter_list(data.current_density_ma_cm2),
        capacity_mah=filter_list(data.capacity_mah),
        capacity_charge_mah=filter_list(data.capacity_charge_mah),
        capacity_discharge_mah=filter_list(data.capacity_discharge_mah),
        coulombic_efficiency=filter_list(data.coulombic_efficiency),
        areal_capacity_mah_cm2=filter_list(data.areal_capacity_mah_cm2),
        specific_capacity_mah_g=filter_list(data.specific_capacity_mah_g),
        dq_dv=filter_list(data.dq_dv),
        cycle_number=filter_list(data.cycle_number),
        step_index=filter_list(data.step_index),
        display_time=filter_list(data.display_time),
        time_unit=data.time_unit,
        rest_trimmed_seconds=data.rest_trimmed_seconds,
        metadata=data.metadata.copy(),
    )


def run_analysis(
    filepath: Union[str, Path],
    technique_str: str = "auto",
    area_cm2: float = 1.13097,
    active_mass_mg: Optional[float] = None,
    drop_rest_min: Optional[float] = None,
    output_json: Optional[str] = None,
    output_html: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    no_html: bool = False,
    max_time_hours: Optional[float] = None,
    max_cycles: Optional[int] = None,
    time_window: Optional[List[float]] = None,
    trim_post_failure: bool = False,
) -> ElectrochemicalAnalysisReport:
    """Executes end-to-end ingestion, normalization, analysis, and dashboard generation."""
    path = Path(filepath)
    logger.info(f"Ingesting cycler file: {path.name}")
    raw_data = parse_cycler_file(path)

    if technique_str.lower() == "auto":
        technique = detect_technique(raw_data)
    else:
        technique = ElectrochemicalTechnique(technique_str.lower())

    geometry = ElectrodeGeometry(area_cm2=area_cm2, active_mass_mg=active_mass_mg)
    normalizer = ElectrochemicalNormalizer(geometry=geometry)
    normalized_data = normalizer.process(
        raw_data=raw_data,
        auto_trim_rest=(drop_rest_min is None or drop_rest_min >= 0),
        drop_rest_min=drop_rest_min,
        compute_dq_dv=True,
    )

    # Apply manual windowing overrides
    if max_time_hours is not None or max_cycles is not None or time_window is not None:
        normalized_data = truncate_normalized_data(
            normalized_data,
            max_time_hours=max_time_hours,
            max_cycles=max_cycles,
            time_window=time_window,
        )

    symm_res = None
    asymm_res = None
    fc_res = None
    lsv_res = None
    eis_res = None
    cv_res = None
    summary_metrics: Dict[str, Any] = {}
    truncated_json_data = None

    if technique == ElectrochemicalTechnique.GALVANOSTATIC_SYMMETRIC:
        symm_res = SymmetricCellAnalyzer().analyze(normalized_data)
        summary_metrics = {
            "total_cycles": symm_res.total_cycles,
            "lifespan_h": symm_res.total_lifespan_hours,
            "mean_overpotential_mv": symm_res.mean_overpotential_mv,
            "mean_hysteresis_mv": symm_res.mean_hysteresis_mv,
            "short_circuit": symm_res.short_circuit_detected,
            "failure_mode": symm_res.failure_mode,
        }
        # Defer truncation to ensure the full trace is retained in the HTML report for Plotly's "Full Test" zoom.
        truncated_json_data = None
        if trim_post_failure and symm_res.failure_time_hours is not None and (symm_res.short_circuit_detected or symm_res.failure_mode != "none"):
            truncated_json_data = truncate_normalized_data(
                normalized_data,
                max_time_hours=symm_res.failure_time_hours,
            )
    elif technique == ElectrochemicalTechnique.GALVANOSTATIC_ASYMMETRIC:
        asymm_res = AsymmetricCellAnalyzer().analyze(normalized_data)
        summary_metrics = {
            "total_cycles": asymm_res.total_cycles,
            "average_ce_percent": asymm_res.average_ce_percent,
            "final_ce_percent": asymm_res.final_ce_percent,
            "mean_nucleation_overpotential_mv": asymm_res.mean_nucleation_overpotential_mv,
        }
    elif technique == ElectrochemicalTechnique.GALVANOSTATIC_FULL_CELL:
        fc_res = FullCellAnalyzer().analyze(normalized_data)
        summary_metrics = {
            "total_cycles": fc_res.total_cycles,
            "capacity_retention_percent": fc_res.capacity_retention_percent,
            "average_ce_percent": fc_res.average_coulombic_efficiency_percent,
        }
    elif technique == ElectrochemicalTechnique.LSV:
        lsv_res = LSVAnalyzer().analyze(normalized_data)
        summary_metrics = {
            "her_onset_v": lsv_res.her_onset_potential_v,
            "ecorr_v": lsv_res.corrosion_potential_ecorr_v,
            "jcorr_ma_cm2": lsv_res.corrosion_current_jcorr_ma_cm2,
            "tafel_slope_cathodic_mv_dec": lsv_res.tafel_slope_cathodic_mv_per_dec,
        }
    elif technique == ElectrochemicalTechnique.EIS:
        eis_res = EISAnalyzer().analyze_raw(raw_data)
        summary_metrics = {
            "solution_resistance_rs_ohm": eis_res.solution_resistance_rs_ohm,
            "charge_transfer_resistance_rct_ohm": eis_res.charge_transfer_resistance_rct_ohm,
            "f_apex_hz": eis_res.characteristic_frequency_hz,
        }
    elif technique == ElectrochemicalTechnique.CV:
        cv_res = CVAnalyzer().analyze(normalized_data)
        summary_metrics = {
            "epa_v": cv_res.anodic_peak_potential_epa_v,
            "epc_v": cv_res.cathodic_peak_potential_epc_v,
            "delta_ep_mv": cv_res.peak_separation_delta_ep_mv,
            "current_ratio_ipa_ipc": cv_res.current_ratio_ipa_ipc,
        }

    report = ElectrochemicalAnalysisReport(
        sample_id=path.stem,
        file_source=str(path.name),
        technique=technique,
        geometry=geometry,
        processed_data=normalized_data,
        symmetric_cell=symm_res,
        asymmetric_cell=asymm_res,
        full_cell=fc_res,
        lsv=lsv_res,
        eis=eis_res,
        cv=cv_res,
        summary_metrics=summary_metrics,
        active_cycling_metrics=symm_res.active_cycling_metrics if symm_res else None,
        total_test_summary=symm_res.total_test_summary if symm_res else None,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    if not no_html:
        plot_engine = (
            ElectrochemicalPlotEngine(output_dir=output_dir)
            if output_dir
            else ElectrochemicalPlotEngine()
        )
        html_path = plot_engine.render_report(report, output_path=output_html)
        report.visualization_html_path = str(html_path)

    if truncated_json_data is not None:
        report.processed_data = truncated_json_data

    if output_json:
        out_json_path = Path(output_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        out_json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"JSON analysis report written to: {out_json_path}")

    return report


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous Electrochemical Characterization, Parsing & Analysis Engine for ZIBs."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--file",
        "-f",
        type=str,
        help="Path to cycler export file (.csv, .txt, .mpt, .xlsx, .mpr).",
    )
    group.add_argument(
        "--watch-dir",
        "-w",
        type=str,
        help="Directory to watch for incoming cycler files.",
    )

    parser.add_argument(
        "--technique",
        "-t",
        type=str,
        default="auto",
        help="Electrochemical technique: auto, galvanostatic_symmetric, galvanostatic_asymmetric, galvanostatic_full_cell, lsv, eis, cv.",
    )
    parser.add_argument(
        "--area-cm2",
        type=float,
        default=1.13097,
        help="Electrode active surface area in cm^2 (default: 1.13097 cm^2 for 12mm disc).",
    )
    parser.add_argument(
        "--active-mass-mg",
        type=float,
        default=None,
        help="Cathode active material mass in mg (for specific capacity in mAh/g).",
    )
    parser.add_argument(
        "--drop-rest-min",
        type=float,
        default=None,
        help="Optional manual rest period trimming duration in minutes.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Destination path for structured JSON analysis report.",
    )
    parser.add_argument(
        "--output-html",
        type=str,
        default=None,
        help="Destination path for standalone Plotly HTML dashboard.",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Suppress HTML chart generation.",
    )

    # Adaptive Failure-Aware Cycling Windowing & Overrides
    parser.add_argument(
        "--max-time-hours",
        type=float,
        default=None,
        help="Manually truncate the analysis window by elapsed time.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Manually truncate the analysis window by cycle count.",
    )
    parser.add_argument(
        "--time-window",
        type=float,
        nargs=2,
        default=None,
        help="Analyze an explicit time slice (min_h max_h).",
    )
    parser.add_argument(
        "--trim-post-failure",
        action="store_true",
        help="Strictly crop exported time-series to pre-failure data.",
    )

    return parser.parse_args(args)


def main():
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parsed = parse_args()

    if parsed.watch_dir:
        watchdog = CyclerWatchdog(
            watch_dir=parsed.watch_dir,
            callback=lambda f: run_analysis(
                filepath=f,
                technique_str=parsed.technique,
                area_cm2=parsed.area_cm2,
                active_mass_mg=parsed.active_mass_mg,
                drop_rest_min=parsed.drop_rest_min,
                output_json=parsed.output_json,
                output_html=parsed.output_html,
                no_html=parsed.no_html,
                max_time_hours=parsed.max_time_hours,
                max_cycles=parsed.max_cycles,
                time_window=parsed.time_window,
                trim_post_failure=parsed.trim_post_failure,
            ),
        )
        print(f"[*] Starting autonomous cycler watchdog on {parsed.watch_dir}...")
        watchdog.run_polling_loop()
        return

    report = run_analysis(
        filepath=parsed.file,
        technique_str=parsed.technique,
        area_cm2=parsed.area_cm2,
        active_mass_mg=parsed.active_mass_mg,
        drop_rest_min=parsed.drop_rest_min,
        output_json=parsed.output_json,
        output_html=parsed.output_html,
        no_html=parsed.no_html,
        max_time_hours=parsed.max_time_hours,
        max_cycles=parsed.max_cycles,
        time_window=parsed.time_window,
        trim_post_failure=parsed.trim_post_failure,
    )

    print("\n============================================================")
    print(" AUTONOMOUS ZIB CHARACTERIZATION REPORT ".center(60, "="))
    print("============================================================")
    print(f"Sample ID:        {report.sample_id}")
    print(f"Technique:        {report.technique.value}")
    print(f"Source File:      {report.file_source}")
    print(f"Electrode Area:   {report.geometry.area_cm2:.4f} cm^2")
    print("\nKey Analytical Metrics:")
    for k, v in report.summary_metrics.items():
        print(f"  * {k}: {v}")

    if report.visualization_html_path:
        print(f"\nInteractive HTML: file:///{Path(report.visualization_html_path).resolve().as_posix()}")
    print("============================================================\n")


if __name__ == "__main__":
    main()
