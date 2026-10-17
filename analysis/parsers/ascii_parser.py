"""
ASCII Delimited Cycler & Potentiostat Data Parser.
Supports CSV, TXT, MPT, DAT files from BioLogic (EC-Lab), Arbin, Landt, Neware, Gamry.
Enhanced with multi-row header auto-healing and canonical unit detection.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

from models import CyclerRawData
from analysis.parsers.base import (
    BaseCyclerParser,
    match_canonical_channel,
    detect_delimiter,
    clean_header_str,
    clean_numeric_cell,
    detect_channel_unit,
    is_unit_row,
    parse_duration_to_seconds,
    resolve_best_column_mapping,
)


class AsciiCyclerParser(BaseCyclerParser):
    """Parses delimited ASCII files from major battery cyclers and potentiostats."""

    SUPPORTED_EXTENSIONS = frozenset({'.csv', '.txt', '.mpt', '.dat', '.dta'})

    def _read_preamble_lines(self, f, default_count: int = 250) -> List[str]:
        """Read preamble lines, dynamically expanding if 'Nb header lines : N' is found."""
        lines = []
        target_count = default_count
        for _ in range(600):
            line = f.readline()
            if not line:
                break
            lines.append(line)
            if len(lines) <= 25:
                m = re.search(r'nb header lines\s*:\s*(\d+)', line.lower())
                if m:
                    hint = int(m.group(1))
                    target_count = max(target_count, hint + 10)
            if len(lines) >= target_count:
                break
        return lines

    def _detect_encoding(self, path: Path) -> str:
        """Detect file encoding among UTF-8, GBK, and Latin-1."""
        for enc in ('utf-8-sig', 'utf-8', 'gbk', 'latin-1'):
            try:
                with open(path, 'r', encoding=enc) as f:
                    sample = self._read_preamble_lines(f)
                header_idx, _ = self._find_header_row(sample)
                if header_idx is not None:
                    return enc
            except Exception:
                continue
        return 'utf-8'

    def can_parse(self, filepath: Union[str, Path]) -> bool:
        path = Path(filepath)
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            return False

        try:
            enc = self._detect_encoding(path)
            with open(path, 'r', encoding=enc, errors='ignore') as f:
                sample_lines = self._read_preamble_lines(f)
            header_idx, _ = self._find_header_row(sample_lines)
            return header_idx is not None
        except Exception:
            return False

    def _find_header_row(self, lines: List[str]) -> Tuple[Optional[int], Optional[str]]:
        """Identify line index of column header and delimiter."""
        nb_header_hint = None
        for line in lines[:30]:
            m = re.search(r'nb header lines\s*:\s*(\d+)', line.lower())
            if m:
                nb_header_hint = int(m.group(1)) - 1
                break

        if nb_header_hint is not None and nb_header_hint < len(lines):
            target_line = lines[nb_header_hint]
            delim = detect_delimiter([target_line])
            cols = [clean_header_str(c) for c in target_line.split(delim)]
            matches = [c for c in cols if match_canonical_channel(c) is not None]
            if len(matches) >= 2:
                return nb_header_hint, delim

        # Gamry CURVE / ZCURVE table marker detection
        for idx, line in enumerate(lines[:300]):
            line_upper = line.strip().upper()
            if line_upper.startswith("CURVE") or line_upper.startswith("ZCURVE") or "CURVE\tTABLE" in line_upper or "ZCURVE\tTABLE" in line_upper:
                for h_idx in range(idx + 1, min(idx + 6, len(lines))):
                    candidate = lines[h_idx].strip()
                    if candidate and not candidate.startswith("#"):
                        delim = detect_delimiter([candidate])
                        cols = [clean_header_str(c) for c in candidate.split(delim)]
                        matches = [c for c in cols if match_canonical_channel(c) is not None]
                        if len(matches) >= 2:
                            return h_idx, delim

        for idx, line in enumerate(lines[:300]):
            delim = detect_delimiter([line])
            cols = [clean_header_str(c) for c in line.split(delim)]
            matches = [c for c in cols if match_canonical_channel(c) is not None]
            if len(matches) >= 2:
                return idx, delim

            # Also check if merging line and line+1 forms a composite header
            if idx + 1 < len(lines):
                next_line = lines[idx + 1]
                next_cols = [clean_header_str(c) for c in next_line.split(delim)]
                composite_cols = [f"{c1} {c2}".strip() for c1, c2 in zip(cols, next_cols)]
                comp_matches = [c for c in composite_cols if match_canonical_channel(c) is not None]
                if len(comp_matches) >= 2:
                    return idx, delim

        return None, None

    def parse(self, filepath: Union[str, Path]) -> CyclerRawData:
        """Parse raw ASCII cycler file into CyclerRawData."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        encoding = self._detect_encoding(path)
        with open(path, 'r', encoding=encoding, errors='ignore') as f:
            lines = self._read_preamble_lines(f)

        header_idx, delimiter = self._find_header_row(lines)
        if header_idx is None or delimiter is None:
            raise ValueError(f"Could not identify valid tabular cycler columns in {path.name}")

        header_line = lines[header_idx].strip()
        raw_headers = [c.strip() for c in header_line.split(delimiter)]

        # Check if row following header is a unit row or composite header
        next_is_unit = False
        unit_row_cells: List[str] = []
        if header_idx + 1 < len(lines):
            next_line = lines[header_idx + 1].strip()
            unit_row_cells = [c.strip() for c in next_line.split(delimiter)]
            next_is_unit = is_unit_row(unit_row_cells)

        candidate_matches: Dict[int, str] = {}
        candidate_units: Dict[int, str] = {}

        for i, h in enumerate(raw_headers):
            canonical = match_canonical_channel(h)
            unit_cell = unit_row_cells[i] if i < len(unit_row_cells) else None

            # Try composite matching if needed
            if not canonical and unit_cell:
                canonical = match_canonical_channel(f"{h} {unit_cell}")

            if not canonical:
                continue

            candidate_matches[i] = canonical
            detected_u = detect_channel_unit(h, unit_cell)
            if detected_u:
                candidate_units[i] = detected_u

        col_mapping = resolve_best_column_mapping(candidate_matches, raw_headers)
        header_units = {k: candidate_units[k] for k in col_mapping if k in candidate_units}

        time_s: List[float] = []
        voltage_v: List[float] = []
        current_ma: List[float] = []
        step_index: List[int] = []
        cycle_number: List[int] = []
        cap_charge: List[float] = []
        cap_discharge: List[float] = []
        frequency_hz: List[float] = []
        z_real_ohm: List[float] = []
        z_imag_ohm: List[float] = []

        has_step = any(v == 'step_index' for v in col_mapping.values())
        has_cycle = any(v == 'cycle_number' for v in col_mapping.values())
        has_cap_chg = any(v == 'capacity_charge' for v in col_mapping.values())
        has_cap_dchg = any(v == 'capacity_discharge' for v in col_mapping.values())
        has_generic_cap = any(v == 'capacity' for v in col_mapping.values())
        has_ce = any(v == 'coulombic_efficiency' for v in col_mapping.values())
        has_voltage = any(v == 'voltage' for v in col_mapping.values())
        has_freq = any(v == 'frequency' for v in col_mapping.values())
        has_zreal = any(v == 'z_real' for v in col_mapping.values())
        has_zimag = any(v == 'z_imag' for v in col_mapping.values())

        is_cycle_summary = (
            not has_voltage
            and not has_zreal
            and (has_cycle or has_ce)
            and (has_generic_cap or has_cap_dchg or has_cap_chg or has_ce)
        )

        metadata = {
            'source_file': path.name,
            'header_row': header_idx,
            'delimiter': delimiter,
            'raw_columns': raw_headers,
            'mapped_channels': {raw_headers[k]: v for k, v in col_mapping.items() if k < len(raw_headers)},
            'detected_units': {raw_headers[k]: header_units[k] for k in header_units if k < len(raw_headers)},
            'has_unit_row': next_is_unit,
            'is_cycle_summary': is_cycle_summary,
        }

        data_start_line = header_idx + 2 if next_is_unit else header_idx + 1

        coulombic_efficiency: List[float] = []

        with open(path, 'r', encoding=encoding, errors='ignore') as f:
            for _ in range(data_start_line):
                f.readline()

            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith('#') or line_str.startswith(';'):
                    continue

                parts = [p.strip() for p in line_str.split(delimiter)]
                row_vals: Dict[str, float] = {}
                try:
                    for col_idx, canon_name in col_mapping.items():
                        raw_val_str = parts[col_idx] if col_idx < len(parts) else None
                        if raw_val_str is None or not raw_val_str:
                            continue
                        unit = header_units.get(col_idx)

                        if canon_name == 'time':
                            val = parse_duration_to_seconds(raw_val_str, unit=unit)
                            if val is None or val == 0.0:
                                val = clean_numeric_cell(raw_val_str)
                        elif canon_name == 'step_time':
                            continue
                        else:
                            val = clean_numeric_cell(raw_val_str)
                            if val is not None:
                                if canon_name == 'current':
                                    if unit == 'a':
                                        val *= 1000.0
                                    elif unit == 'ua':
                                        val /= 1000.0
                                elif canon_name in ('capacity_charge', 'capacity_discharge', 'capacity'):
                                    if unit == 'ah':
                                        val *= 1000.0
                                    elif unit == 'uah':
                                        val /= 1000.0
                                elif canon_name == 'voltage':
                                    if unit == 'mv':
                                        val /= 1000.0

                        if val is not None:
                            row_vals[canon_name] = val

                    if not row_vals:
                        continue

                    if is_cycle_summary:
                        cyc = int(row_vals.get('cycle_number', len(cycle_number) + 1))
                        cap_val = float(row_vals.get('capacity_discharge', row_vals.get('capacity', row_vals.get('capacity_charge', 0.0))))
                        ce_val = float(row_vals.get('coulombic_efficiency', 99.5)) if has_ce else 99.5
                        if 0.0 < ce_val <= 1.05:
                            ce_val *= 100.0

                        chg_val = float(row_vals.get('capacity_charge', 0.0))
                        if not has_cap_chg and ce_val > 0:
                            chg_val = round(cap_val / (ce_val / 100.0), 4)

                        time_s.append(float((cyc - 1) * 3600.0))
                        voltage_v.append(1.2)  # Nominal full-cell potential for schema integrity
                        current_ma.append(1.0)
                        cycle_number.append(cyc)
                        cap_discharge.append(cap_val)
                        cap_charge.append(chg_val)
                        if has_ce or ce_val is not None:
                            coulombic_efficiency.append(ce_val)
                        continue

                    if 'voltage' not in row_vals and 'z_real' not in row_vals:
                        continue

                    curr_val = row_vals.get('current', 0.0)
                    volt_val = row_vals.get('voltage', 0.0)
                    time_s.append(row_vals.get('time', float(len(voltage_v))))
                    voltage_v.append(volt_val)
                    current_ma.append(curr_val)

                    if has_step:
                        step_index.append(int(row_vals.get('step_index', 0)))
                    if has_cycle:
                        cycle_number.append(int(row_vals.get('cycle_number', 0)))

                    if has_cap_chg:
                        cap_charge.append(float(row_vals.get('capacity_charge', 0.0)))
                    if has_cap_dchg:
                        cap_discharge.append(float(row_vals.get('capacity_discharge', 0.0)))
                    if has_generic_cap and not has_cap_chg and not has_cap_dchg:
                        q_val = float(row_vals.get('capacity', 0.0))
                        if curr_val < -0.005:
                            cap_discharge.append(q_val)
                            cap_charge.append(0.0)
                        else:
                            cap_charge.append(q_val)
                            cap_discharge.append(0.0)

                    if has_ce and 'coulombic_efficiency' in row_vals:
                        ce_v = float(row_vals['coulombic_efficiency'])
                        if 0.0 < ce_v <= 1.05:
                            ce_v *= 100.0
                        coulombic_efficiency.append(ce_v)

                    if has_freq:
                        frequency_hz.append(float(row_vals.get('frequency', 0.0)))
                    if has_zreal:
                        z_real_ohm.append(float(row_vals.get('z_real', 0.0)))
                    if has_zimag:
                        z_imag_ohm.append(float(row_vals.get('z_imag', 0.0)))
                except (ValueError, IndexError):
                    continue

        if not time_s:
            raise ValueError(f"No valid numeric data points parsed from {path.name}")

        if time_s and time_s[0] > 1e7:
            t0 = time_s[0]
            time_s = [round(t - t0, 4) for t in time_s]

        final_has_cap_chg = has_cap_chg or (has_generic_cap and len(cap_charge) == len(time_s)) or (is_cycle_summary and len(cap_charge) == len(time_s))
        final_has_cap_dchg = has_cap_dchg or (has_generic_cap and len(cap_discharge) == len(time_s)) or (is_cycle_summary and len(cap_discharge) == len(time_s))
        final_has_ce = len(coulombic_efficiency) == len(time_s) and len(coulombic_efficiency) > 0

        if is_cycle_summary:
            metadata['summary_coulombic_efficiency'] = coulombic_efficiency
            metadata['summary_cycle_number'] = cycle_number
            metadata['summary_capacity'] = cap_discharge

        return CyclerRawData(
            time_s=time_s,
            voltage_v=voltage_v,
            current_ma=current_ma,
            step_index=step_index if has_step else None,
            cycle_number=cycle_number if (has_cycle or is_cycle_summary) else None,
            capacity_charge_mah=cap_charge if final_has_cap_chg else None,
            capacity_discharge_mah=cap_discharge if final_has_cap_dchg else None,
            coulombic_efficiency=coulombic_efficiency if final_has_ce else None,
            frequency_hz=frequency_hz if has_freq else None,
            z_real_ohm=z_real_ohm if has_zreal else None,
            z_imag_ohm=z_imag_ohm if has_zimag else None,
            metadata=metadata,
        )
