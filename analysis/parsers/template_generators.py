"""
Reference Cycler File Template Generators for BioLogic, Landt, and Arbin/Neware.
Provides downloadable authentic reference files to guide users on cycler file formatting
and enable instant verification of the electrochemical analysis engine.
"""

import io
from typing import Dict, List, Any
import numpy as np

from export_engine import make_multi_sheet_xlsx


def generate_template_biologic_mpt(
    duration_hours: float = 20.0,
    current_ma: float = 1.131,
    plateau_voltage_v: float = 0.035,
    point_interval_s: float = 60.0,
    cycle_period_s: float = 7200.0,
) -> bytes:
    """
    Generate authentic BioLogic EC-Lab ASCII (.mpt) reference cycling file.
    Includes BT-Lab header preamble and tab-delimited columns.
    """
    total_seconds = int(duration_hours * 3600.0)
    times = np.arange(0, total_seconds, point_interval_s, dtype=float)
    half_period = cycle_period_s / 2.0
    phase = (times % cycle_period_s) < half_period

    current = np.where(phase, current_ma, -current_ma)
    voltage = np.where(phase, plateau_voltage_v, -plateau_voltage_v)

    np.random.seed(42)
    noise = np.random.normal(0, 0.0005, size=len(times))
    voltage += noise
    cycle_number = (times // cycle_period_s).astype(int) + 1

    lines = [
        "BT-Lab ASCII FILE\n",
        "Device : VMP3\n",
        "Nb header lines : 4\n",
        "time/s\tEcell/V\t<I>/mA\tcycle number\n",
    ]
    for t, v, i, c in zip(times, voltage, current, cycle_number):
        lines.append(f"{t:.1f}\t{v:.5f}\t{i:.4f}\t{c}\n")

    return "".join(lines).encode("utf-8")


def generate_template_landt_csv(
    duration_hours: float = 20.0,
    current_ma: float = 1.131,
    plateau_voltage_v: float = 0.035,
    point_interval_s: float = 60.0,
    cycle_period_s: float = 7200.0,
) -> bytes:
    """
    Generate standard Landt Battery Testing System CSV (.csv) reference file.
    Comma-delimited format with Time(s), Voltage(V), Current(mA), Capacity(mAh).
    """
    total_seconds = int(duration_hours * 3600.0)
    times = np.arange(0, total_seconds, point_interval_s, dtype=float)
    half_period = cycle_period_s / 2.0
    phase = (times % cycle_period_s) < half_period

    current = np.where(phase, current_ma, -current_ma)
    voltage = np.where(phase, plateau_voltage_v, -plateau_voltage_v)

    np.random.seed(42)
    noise = np.random.normal(0, 0.0005, size=len(times))
    voltage += noise

    capacity_step = (np.abs(current) * point_interval_s) / 3600.0
    cumulative_capacity = np.cumsum(capacity_step)

    lines = ["Time(s),Voltage(V),Current(mA),Capacity(mAh)\n"]
    for t, v, i, c in zip(times, voltage, current, cumulative_capacity):
        lines.append(f"{t:.1f},{v:.5f},{i:.4f},{c:.4f}\n")

    return "".join(lines).encode("utf-8")


def generate_template_arbin_xlsx(
    duration_hours: float = 20.0,
    current_ma: float = 1.131,
    plateau_voltage_v: float = 0.035,
    point_interval_s: float = 60.0,
    cycle_period_s: float = 7200.0,
) -> bytes:
    """
    Generate standard Arbin / Neware Excel (.xlsx) reference workbook.
    Includes unit scaling demonstrations: Current in Amperes (A) and Capacity in Ampere-hours (Ah).
    """
    total_seconds = int(duration_hours * 3600.0)
    times = np.arange(0, total_seconds, point_interval_s, dtype=float)
    half_period = cycle_period_s / 2.0
    phase = (times % cycle_period_s) < half_period

    current_a = np.where(phase, current_ma / 1000.0, -current_ma / 1000.0)
    voltage = np.where(phase, plateau_voltage_v, -plateau_voltage_v)

    np.random.seed(42)
    noise = np.random.normal(0, 0.0005, size=len(times))
    voltage += noise
    cycle_number = (times // cycle_period_s).astype(int) + 1

    dt_h = point_interval_s / 3600.0
    chg_ah = np.cumsum(np.where(phase, current_a * dt_h, 0.0))
    dchg_ah = np.cumsum(np.where(~phase, np.abs(current_a) * dt_h, 0.0))

    headers = [
        "Test_Time(s)",
        "Voltage(V)",
        "Current(A)",
        "Charge_Capacity(Ah)",
        "Discharge_Capacity(Ah)",
        "Cycle_Index",
    ]
    rows: List[List[Any]] = [headers]
    for t, v, i, c_chg, c_dchg, cyc in zip(times, voltage, current_a, chg_ah, dchg_ah, cycle_number):
        rows.append([round(float(t), 1), round(float(v), 5), round(float(i), 6), round(float(c_chg), 6), round(float(c_dchg), 6), int(cyc)])

    return make_multi_sheet_xlsx({"Record": rows})

