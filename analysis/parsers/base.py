"""
Base classes and channel heuristics for Potentiostat & Cycler Data Parsers.
Standardizes heterogeneous vendor column names into SI-aligned CyclerRawData.
Enhanced with cellpy-inspired canonical aliases and multi-row unit healers
(MIT License, see third_party_licenses/cellpy_LICENSE.txt).
"""

import re
import csv
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

from models import CyclerRawData

CHANNEL_ALIASES: Dict[str, Tuple[str, ...]] = {
    "time": (
        "time/s", "time(s)", "time (s)", "time", "test_time(s)", "test_time",
        "test time (s)", "test time", "total_time(s)", "total time(s)", "total time", "total_time",
        "t/s", "t (s)", "t", "runtime(s)", "runtime", "elapsed_time", "elapsed time", "duration/s", "duration",
        "date_time", "datetime", "date time", "relative_time(s)", "relative_time", "file_time",
        "time/min", "time(min)", "time/h", "time(h)", "time/hr", "time(hr)", "time(ms)", "time/ms",
        "testtime", "cur_time", "curtime",
        "时间", "测试时间", "相对时间", "绝对时间",
    ),
    "step_time": (
        "step_time(s)", "step_time", "step time (s)", "step time", "sub_step_time",
        "step-time", "steptime", "工步时间",
    ),
    "voltage": (
        "ewe/v", "ewe(v)", "ewe (v)", "ewe", "ecell/v", "ecell(v)", "ecell (v)", "ecell", "<ecell>/v",
        "voltage/v", "voltage(v)", "voltage (v)", "voltage", "volt/v", "volt", "v", "vf", "vu", "vfield",
        "potential/v", "potential(v)", "potential", "cell_voltage", "cell voltage", "potential (v)",
        "we(1).potential (v)", "we(1).potential/v", "we(1).potential", "we.potential", "ewe-ece/v", "ewe-ece",
        "u/v", "u(v)", "u", "aux_voltage_1(v)", "aux_voltage_1", "reference_voltage(v)", "ref_voltage",
        "working_electrode_potential", "we_potential", "potential/mv", "potential(mv)", "voltage/mv", "voltage(mv)",
        "volt(v)", "v(v)", "u/mv", "u(mv)", "cell_volt", "cellvolt",
        "电压", "电位", "工作电极电位", "电池电压",
    ),
    "current": (
        "current/ma", "current(ma)", "current (ma)", "current/a", "current(a)",
        "current (a)", "current", "<i-ewe>/ma", "<i>/ma", "i/ma", "i(ma)",
        "i (ma)", "im", "i/a", "i(a)", "i (a)", "i", "curr", "curr/ma", "curr(ma)",
        "we(1).current (a)", "we(1).current/a", "we(1).current", "we(1).current (ma)", "we(1).current/ma", "we.current",
        "curr/a", "curr(a)", "cell_current", "cell current", "cur(ma)", "cur/ma", "cur(a)", "cur/a",
        "cur", "current(ua)", "current/ua", "current/µa", "current(µa)", "i/ua", "i(ua)",
        "cur_ma", "cur_a", "current_a", "current_ma", "<i-ewe>", "<i>",
        "电流", "工作电流", "电池电流",
    ),
    "step_index": (
        "step_index", "step index", "step_id", "step id", "step", "step_no",
        "step number", "step no", "state", "step type", "step(n)", "step_index(n)",
        "sub_step_index", "step-index", "step_idx", "stepidx", "ns", "step_name",
        "step_number(n)", "sub_step", "steptype", "step type(n)",
        "工步", "工步序号", "工步号", "步骤", "状态",
    ),
    "cycle_number": (
        "cycle_index", "cycle index", "cycle_number", "cycle number",
        "cycle", "cycle_no", "cycle no", "cyc#", "cycle count", "cycle(n)", "cyc",
        "cycle_id", "cycle id", "cycle-index", "cycle_idx", "cycleidx",
        "cycle_index(n)", "cycleno", "cycle#", "cycle(n)",
        "循环", "循环序号", "循环号", "圈数", "周次",
    ),
    "capacity_charge": (
        "charge_capacity(mah)", "charge capacity(mah)", "charge_capacity(ah)",
        "charge capacity (ah)", "charge capacity", "charge_capacity",
        "chg_cap(mah)", "chg. cap. (mah)", "capacity_charge", "q_charge",
        "charge capacity/mah", "chg_capacity", "q charge/ma.h", "q charge/mah",
        "charge_cap(ah)", "chg_cap(ah)", "chg_cap", "chg cap", "ccap", "q_chg",
        "q_charge(ah)", "charge_cap", "chg.cap(ah)", "charge capacity/ah",
        "charge_capacity(ua.h)", "charge_capacity(uah)", "q charge/a.h", "q charge/ah",
        "chg-cap(mah)", "chg-cap(ah)", "chg-cap", "chg_cap_mah", "chg_cap_ah", "chgcap",
        "charge_capacity_ah", "charge_capacity_mah", "chg_capacity(mah)", "chg_capacity(ah)",
        "c_cap(mah)", "c_cap(ah)", "c_capacity",
        "充电容量", "充电容量(mah)", "充电容量(ah)", "充电容量/mah", "充电容量/ah",
    ),
    "capacity_discharge": (
        "discharge_capacity(mah)", "discharge capacity(mah)", "discharge_capacity(ah)",
        "discharge capacity (ah)", "discharge capacity", "discharge_capacity",
        "dchg_cap(mah)", "dchg. cap. (mah)", "capacity_discharge", "q_discharge",
        "discharge capacity/mah", "dchg_capacity", "q discharge/ma.h", "q discharge/mah",
        "discharge_cap(ah)", "dchg_cap(ah)", "dchg_cap", "dchg cap", "dcap", "q_dchg",
        "q_discharge(ah)", "discharge_cap", "dchg.cap(ah)", "discharge capacity/ah",
        "discharge_capacity(ua.h)", "discharge_capacity(uah)", "q discharge/a.h", "q discharge/ah",
        "dchg-cap(mah)", "dchg-cap(ah)", "dchg-cap", "dchg_cap_mah", "dchg_cap_ah", "dchgcap",
        "discharge_capacity_ah", "discharge_capacity_mah", "dchg_capacity(mah)", "dchg_capacity(ah)",
        "d_cap(mah)", "d_cap(ah)", "d_capacity",
        "discharge capacity(mah/g)", "discharge capacity (mah/g)", "discharge capacity/mah/g",
        "specific discharge capacity", "discharge specific capacity", "specific_discharge_capacity",
        "放电容量", "放电容量(mah)", "放电容量(ah)", "放电容量/mah", "放电容量/ah", "放电比容量", "放电比容量(mah/g)",
    ),
    "capacity": (
        "capacity(mah)", "capacity (mah)", "capacity/mah", "capacity(ah)", "capacity (ah)", "capacity/ah",
        "capacity", "cap", "cap(mah)", "cap(ah)", "q", "q(mah)", "q(ah)", "(q-qo)/ma.h", "(q-qo)/mah",
        "total_capacity", "net_capacity", "cap/mah", "cap/ah", "capacity_mah", "capacity_ah",
        "specific capacity", "specific_capacity", "capacity(mah/g)", "capacity (mah/g)", "capacity/mah/g",
        "cap(mah/g)", "cap (mah/g)", "capacity(mah g-1)", "capacity (mah g-1)",
        "capacity / (mah/g)", "discharge capacity / (mah/g)", "capacity / mah g-1",
        "specific capacity (mah/g)", "specific capacity(mah/g)", "discharge specific capacity (mah/g)",
        "specific discharge capacity (mah/g)", "specific discharge capacity",
        "q charge/discharge/ma.h", "q charge/discharge/mah", "q charge/discharge",
        "q charge/discharge/a.h", "q charge/discharge/ah",
        "容量", "容量(mah)", "容量(ah)", "容量/mah", "容量/ah", "比容量", "比容量(mah/g)",
    ),
    "coulombic_efficiency": (
        "coulombic_efficiency", "coulombic efficiency", "coulombic efficiency(%)", "coulombic efficiency (%)",
        "coulombic_efficiency(%)", "coulombic_efficiency (%)", "coulombic eff(%)", "coulombic eff (%)",
        "coulombic eff", "coulombic_eff", "coulomb efficiency", "coulombic",
        "columbic_efficiency", "columbic efficiency", "columbic efficiency(%)", "columbic efficiency (%)",
        "columbic_efficiency(%)", "columbic_efficiency (%)", "columbic eff(%)", "columbic eff (%)",
        "columbic eff", "columbic_eff", "columb efficiency", "columbic",
        "coulombic efficiency/%", "columbic efficiency/%", "coulombic efficiency / %", "columbic efficiency / %",
        "coulomb efficiency (%)", "columb efficiency (%)", "coulomb efficiency(%)", "columb efficiency(%)",
        "coulombic-efficiency", "columbic-efficiency",
        "ce(%)", "ce (%)", "ce", "ce/%", "ce / %", "ce(pct)", "ce (pct)",
        "eff(%)", "eff (%)", "efficiency(%)", "efficiency (%)", "efficiency",
        "库仑效率", "库伦效率", "库仑效率(%)", "库伦效率(%)", "库仑效率 / %", "库伦效率 / %", "效率", "效率(%)",
    ),
    "frequency": (
        "freq/hz", "freq(hz)", "frequency/hz", "frequency(hz)", "frequency (hz)",
        "frequency", "freq", "f/hz", "f (hz)", "f", "frequency (f/hz)", "频率",
    ),
    "z_real": (
        "z_real", "z_real_ohm", "zreal", "z' / ohm", "z'/ohm",
        "re(z)", "zr", "z_re", "re(z)/ohm", "z real", "z' (ohm)", "real(z)", "z'(ohm)",
        "z' / ω", "z'/ω", "z_re(ohm)", "实部",
    ),
    "z_imag": (
        "z_imag", "z_imag_ohm", "zimag", "-z'' / ohm", "-z''/ohm",
        "-im(z)", "zim", "-z_im", "-im(z)/ohm", "z''/ohm", "z imag", "-z'' (ohm)",
        "imag(z)", "im(z)", "-z''(ohm)", "z'' (ohm)", "-z'' / ω", "-z''/ω", "z_im(ohm)", "虚部",
    ),
}

# Known unit regex tokens
_UNIT_TOKENS = frozenset({
    "a", "ma", "ua", "µa", "ah", "mah", "uah", "µah",
    "v", "mv", "s", "sec", "secs", "min", "mins", "h", "hr", "hrs", "ms",
    "ohm", "ohms", "ω", "hz", "khz", "mhz", "w", "mw", "wh", "mwh",
    "mah/g", "ah/kg", "ma/g", "a/g", "%",
})


def clean_header_str(text: Any) -> str:
    """Normalize header string for channel matching."""
    s = str(text or "").strip().lower()
    # Normalize brackets and braces to parentheses
    s = s.replace("[", "(").replace("]", ")")
    s = s.replace("{", "(").replace("}", ")")
    # Replace line breaks, tabs, multiple spaces, underscores with single space
    s = re.sub(r'[\r\n\t_]+', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\s*\(\s*', '(', s)
    s = re.sub(r'\s*\)\s*', ')', s)
    s = re.sub(r'\s*/\s*', '/', s)
    return s.strip()


def strip_channel_prefix(cleaned: str) -> str:
    """Strip cycler channel prefixes like 'channel 1 current' or 'ch1_current'."""
    return re.sub(r'^(channel\s*\d+|ch\s*\d+|aux\s*\d+)\s*[:\-\s]\s*', '', cleaned).strip()


def match_canonical_channel(raw_header: Any) -> Optional[str]:
    """Match raw column header against canonical channel aliases."""
    if raw_header is None:
        return None
    cleaned = clean_header_str(raw_header)
    if not cleaned:
        return None

    stripped = strip_channel_prefix(cleaned)

    # 1. Exact match
    for canonical, aliases in CHANNEL_ALIASES.items():
        for alias in aliases:
            cleaned_alias = clean_header_str(alias)
            if cleaned == cleaned_alias or stripped == cleaned_alias:
                return canonical

    # 2. Substring / boundary match
    for canonical, aliases in CHANNEL_ALIASES.items():
        for alias in aliases:
            cleaned_alias = clean_header_str(alias)
            if len(cleaned_alias) <= 1:
                continue
            # Match boundary separated by space
            if cleaned.startswith(cleaned_alias + ' ') or cleaned.endswith(' ' + cleaned_alias):
                return canonical
            if stripped.startswith(cleaned_alias + ' ') or stripped.endswith(' ' + cleaned_alias):
                return canonical

    # 3. Base name before unit separator (/ or ()
    base_name = re.split(r'[/(\[]', cleaned)[0].strip()
    base_stripped = re.split(r'[/(\[]', stripped)[0].strip()
    for b in (base_name, base_stripped):
        if b and b != cleaned:
            for canonical, aliases in CHANNEL_ALIASES.items():
                for alias in aliases:
                    cleaned_alias = clean_header_str(alias)
                    if b == cleaned_alias:
                        return canonical

    return None


def resolve_best_column_mapping(matches: Dict[int, str], raw_headers: List[str]) -> Dict[int, str]:
    """
    Given candidate {col_idx: canonical_channel} and raw_headers, resolves channel collisions
    when multiple columns map to the same canonical channel (e.g. Test_Time vs Date_Time,
    or Voltage vs Aux_Voltage). Returns a deduplicated {col_idx: canonical_channel} mapping
    with the highest-specificity column selected for each channel.
    """
    by_channel: Dict[str, List[int]] = {}
    for col_idx, canon in matches.items():
        by_channel.setdefault(canon, []).append(col_idx)

    resolved: Dict[int, str] = {}

    def score_col(c_idx: int, canon: str) -> float:
        header_raw = raw_headers[c_idx] if c_idx < len(raw_headers) else ""
        clean = clean_header_str(header_raw)

        if canon == "time":
            if any(k in clean for k in ("test time", "test_time", "total time", "total_time", "runtime", "elapsed", "duration")):
                return 100.0
            if any(k in clean for k in ("time/s", "time(s)", "t/s", "t(s)", "time")):
                return 80.0
            if any(k in clean for k in ("relative", "file_time")):
                return 50.0
            return 20.0  # date, datetime, timestamp

        elif canon == "voltage":
            if any(k in clean for k in ("ewe/v", "ewe(v)", "voltage/v", "voltage(v)", "cell voltage", "ecell", "potential/v")):
                return 100.0
            if any(k in clean for k in ("ewe", "voltage", "potential", "u/v", "u(v)", "u", "volt")):
                return 80.0
            return 30.0  # aux, ref, ewe-ece

        elif canon == "current":
            if any(k in clean for k in ("current/ma", "current(ma)", "current/a", "current(a)", "i/ma", "i(ma)", "i/a", "i(a)", "<i", "we(1).current")):
                return 100.0
            if any(k in clean for k in ("current", "curr", "i")):
                return 80.0
            return 30.0

        elif canon == "step_index":
            if any(k in clean for k in ("step index", "step_index", "step no", "step_no", "step number", "step id", "工步序号", "工步号")):
                return 100.0
            return 80.0

        elif canon == "cycle_number":
            if any(k in clean for k in ("cycle index", "cycle_index", "cycle no", "cycle_no", "cycle number", "cycle_number", "cyc#", "循环序号", "循环号")):
                return 100.0
            return 80.0

        elif canon == "capacity_charge":
            if any(k in clean for k in ("charge capacity", "charge_capacity", "chg cap", "chg_cap", "q charge", "q_charge", "充电容量")):
                return 100.0
            return 60.0

        elif canon == "capacity_discharge":
            if any(k in clean for k in ("discharge capacity", "discharge_capacity", "dchg cap", "dchg_cap", "q discharge", "q_discharge", "放电容量")):
                return 100.0
            return 60.0

        elif canon == "coulombic_efficiency":
            if any(k in clean for k in ("coulombic", "columbic", "库仑", "库伦")):
                return 100.0
            if any(k in clean for k in ("ce", "efficiency")):
                return 80.0
            return 50.0

        return 50.0

    for canon, col_indices in by_channel.items():
        if len(col_indices) == 1:
            resolved[col_indices[0]] = canon
        else:
            best_col = max(col_indices, key=lambda c: (score_col(c, canon), -c))
            resolved[best_col] = canon

    return resolved


def detect_channel_unit(header_str: Any, unit_str: Any = None) -> Optional[str]:
    """
    Extract canonical SI unit from header string or dedicated unit row string.
    Returns: 'a', 'ma', 'ua', 'ah', 'mah', 'uah', 'v', 'mv', 's', 'min', 'h', 'ms', 'ohm', 'hz'.
    """
    combined = f"{clean_header_str(header_str)} {clean_header_str(unit_str)}".lower()

    # Current units
    if re.search(r'(?:\(|/|\[|\b)(ma|milliamp|milliamps)(?:\)|/|\]|\b)', combined):
        return 'ma'
    if re.search(r'(?:\(|/|\[|\b)(ua|µa|microamp|microamps)(?:\)|/|\]|\b)', combined):
        return 'ua'
    if re.search(r'(?:\(|/|\[|\b)(a|amp|amps|ampere|amperes)(?:\)|/|\]|\b)', combined):
        # Ensure it's not Ah or mA
        if not re.search(r'(?:mah|ah|ma)', combined):
            return 'a'

    # Capacity units
    if re.search(r'(?:\(|/|\[|\b)(mah|ma\.h|ma\*h|milliamp-hour|milliamp-hours)(?:\)|/|\]|\b)', combined):
        return 'mah'
    if re.search(r'(?:\(|/|\[|\b)(uah|µah|ua\.h)(?:\)|/|\]|\b)', combined):
        return 'uah'
    if re.search(r'(?:\(|/|\[|\b)(ah|a\.h|a\*h|amp-hour|amp-hours|ampere-hour)(?:\)|/|\]|\b)', combined):
        return 'ah'

    # Time units
    if re.search(r'(?:\(|/|\[|\b)(ms|millisecond|milliseconds)(?:\)|/|\]|\b)', combined):
        return 'ms'
    if re.search(r'(?:\(|/|\[|\b)(min|mins|minute|minutes)(?:\)|/|\]|\b)', combined):
        return 'min'
    if re.search(r'(?:\(|/|\[|\b)(h|hr|hrs|hour|hours)(?:\)|/|\]|\b)', combined):
        return 'h'
    if re.search(r'(?:\(|/|\[|\b)(s|sec|secs|second|seconds)(?:\)|/|\]|\b)', combined):
        return 's'

    # Voltage units
    if re.search(r'(?:\(|/|\[|\b)(mv|millivolt|millivolts)(?:\)|/|\]|\b)', combined):
        return 'mv'
    if re.search(r'(?:\(|/|\[|\b)(v|volt|volts)(?:\)|/|\]|\b)', combined):
        return 'v'

    return None


def is_unit_row(row_cells: List[Any]) -> bool:
    """
    Check if a row consists of units (e.g., Row 2 of a multi-row header: s, V, A, Ah).
    Returns True if cells contain unit tokens and fail numeric conversion.
    """
    if not row_cells:
        return False

    unit_hits = 0
    numeric_count = 0
    non_empty = 0

    for cell in row_cells:
        if cell is None:
            continue
        c_str = str(cell).strip()
        if not c_str:
            continue
        non_empty += 1

        cleaned = clean_header_str(c_str).strip("()")
        if cleaned in _UNIT_TOKENS or f"({cleaned})" in _UNIT_TOKENS:
            unit_hits += 1

        try:
            float(c_str.replace(',', '.'))
            numeric_count += 1
        except ValueError:
            pass

    # A unit row must have at least one recognized unit token and very few or no numeric data points
    if non_empty > 0 and unit_hits >= 1 and numeric_count == 0:
        return True
    return False


COMMON_DATETIME_FORMATS = (
    "%m/%d/%Y %H:%M:%S.%f",
    "%d/%m/%Y %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S.%f",
    "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
)


def parse_duration_to_seconds(val: Any, unit: Optional[str] = None) -> float:
    """
    Parse a time cell into float seconds, supporting float numbers, unit multipliers,
    calendar datetime strings (e.g. BioLogic '02/09/2026 21:54:04.6800'),
    and duration strings like 'hh:mm:ss' or 'hh:mm:ss.fff'.
    """
    if isinstance(val, (int, float)):
        v = float(val)
        if unit == 'h':
            return v * 3600.0
        elif unit == 'min':
            return v * 60.0
        elif unit == 'ms':
            return v / 1000.0
        return v

    if isinstance(val, datetime):
        return val.timestamp()

    s = str(val or "").strip()
    if not s:
        return 0.0

    # 1. Handle calendar datetimes (BioLogic / Arbin exports)
    if ('/' in s or '-' in s) and ':' in s:
        for fmt in COMMON_DATETIME_FORMATS:
            try:
                dt = datetime.strptime(s, fmt)
                return dt.timestamp()
            except ValueError:
                pass

    # 2. Handle hh:mm:ss duration strings (e.g. 01:30:00.5)
    if ':' in s:
        parts = s.split(':')
        try:
            if len(parts) == 3:
                h = float(parts[0])
                m = float(parts[1])
                sec = float(parts[2].replace(',', '.'))
                return h * 3600.0 + m * 60.0 + sec
            elif len(parts) == 2:
                m = float(parts[0])
                sec = float(parts[1].replace(',', '.'))
                return m * 60.0 + sec
        except ValueError:
            pass

    # 3. Numeric string
    v = float(s.replace(',', '.'))
    if unit == 'h':
        return v * 3600.0
    elif unit == 'min':
        return v * 60.0
    elif unit == 'ms':
        return v / 1000.0
    return v


def detect_delimiter(sample_lines: List[str]) -> str:
    """Detect delimiter (comma, tab, semicolon, pipe) from sample lines."""
    delimiters = [',', '\t', ';', '|']
    scores = {d: 0 for d in delimiters}
    for line in sample_lines:
        for d in delimiters:
            if d in line:
                scores[d] += line.count(d)
    best_delim = max(scores, key=scores.get)
    return best_delim if scores[best_delim] > 0 else ','


def clean_numeric_cell(val_raw: Any) -> Optional[float]:
    """
    Clean and convert heterogeneous cycler cell values into float.
    Handles:
    - Pure numbers (int, float)
    - Percent strings: '99.5%', '99.5 %', '100%' -> 99.5
    - Unit strings: '120.5 mAh/g', '120.5 mAh', '1.0 mA/cm2', '2.5 V' -> 120.5
    - Cycle prefixes: 'Cycle 1', 'Cyc 1', '#1', 'No. 1' -> 1.0
    - Localized decimals: '120,5' -> 120.5, '1 200,5' -> 1200.5
    - Whitespace & non-breaking space: ' 120.5 \xa0' -> 120.5
    - Scientific notation: '1.2e-3' -> 0.0012
    - None, empty, NaN, '-', 'N/A' -> None
    """
    if val_raw is None:
        return None
    if isinstance(val_raw, (int, float)):
        import math
        return float(val_raw) if not math.isnan(val_raw) else None
    if isinstance(val_raw, bool):
        return 1.0 if val_raw else 0.0

    s = str(val_raw).strip().replace('\xa0', ' ')
    if not s or s.lower() in ('none', 'nan', 'null', 'n/a', 'na', '-', '--'):
        return None

    # Handle comma/dot formatting
    if ',' in s and '.' not in s:
        s = s.replace(',', '.')
    else:
        s = s.replace(',', '')

    try:
        return float(s)
    except ValueError:
        pass

    # Extract first signed float or int using regex
    m = re.search(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', s)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            pass

    return None


class BaseCyclerParser(ABC):
    """Abstract base cycler parser interface."""

    @abstractmethod
    def can_parse(self, filepath: Union[str, Path]) -> bool:
        """Return True if this parser can handle the given file."""
        pass

    @abstractmethod
    def parse(self, filepath: Union[str, Path]) -> CyclerRawData:
        """Parse raw file and return standardized CyclerRawData."""
        pass
