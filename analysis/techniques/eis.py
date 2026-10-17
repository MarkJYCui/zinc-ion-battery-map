"""
Electrochemical Impedance Spectroscopy (EIS) Equivalent Circuit & Nyquist Engine.
Extracts:
1. Ohmic electrolyte solution resistance (R_s) at high-frequency intercept.
2. Charge transfer resistance (R_ct) from semicircle diameter.
3. Characteristic apex frequency (f_apex).
"""

from typing import Optional, Tuple, Union
import numpy as np

from models import CyclerRawData, NormalizedElectrochemicalData, EISAnalysis


class EISAnalyzer:
    """Analyzes EIS Nyquist spectra for bulk electrolyte and charge-transfer resistances."""

    def analyze(self, data: Union[CyclerRawData, NormalizedElectrochemicalData]) -> EISAnalysis:
        if isinstance(data, CyclerRawData):
            freq_raw = data.frequency_hz
            zr_raw = data.z_real_ohm
            zi_raw = data.z_imag_ohm
        elif isinstance(data, NormalizedElectrochemicalData):
            meta = data.metadata or {}
            freq_raw = meta.get("frequency_hz")
            zr_raw = meta.get("z_real_ohm")
            zi_raw = meta.get("z_imag_ohm")
        else:
            return EISAnalysis()

        freq = np.array(freq_raw, dtype=float) if freq_raw else np.array([])
        zr = np.array(zr_raw, dtype=float) if zr_raw else np.array([])
        zi = np.array(zi_raw, dtype=float) if zi_raw else np.array([])

        if len(zr) < 3 or len(zi) < 3:
            return EISAnalysis()

        # In standard Nyquist plots, y-axis is -Im(Z)
        neg_zi = -zi if np.mean(zi) < 0 else zi

        # If frequencies are provided, sort by descending frequency (high freq first)
        if len(freq) == len(zr) and len(freq) > 0:
            sort_order = np.argsort(freq)[::-1]
            freq = freq[sort_order]
            zr = zr[sort_order]
            neg_zi = neg_zi[sort_order]

        # High-frequency intercept (R_s): minimum -Im(Z) in high-frequency region
        hf_window = max(2, min(len(zr), max(2, len(zr) // 5)))
        rs_idx = int(np.argmin(neg_zi[:hf_window]))
        rs = float(zr[rs_idx])

        # Semicircle apex (maximum -Im(Z))
        apex_idx = int(np.argmax(neg_zi))
        f_apex = float(freq[apex_idx]) if len(freq) > 0 and len(freq) > apex_idx else None
        apex_zi = float(neg_zi[apex_idx])

        # Default initial estimate of R_ct as 2 * apex height
        rct = 2.0 * apex_zi

        # Refined R_ct from the first local minimum after apex where -Im(Z) bottoms out
        local_min_idx = None
        for k in range(apex_idx + 1, len(neg_zi) - 1):
            if neg_zi[k] <= neg_zi[k - 1] and neg_zi[k] <= neg_zi[k + 1]:
                local_min_idx = k
                break

        if local_min_idx is not None and local_min_idx > apex_idx:
            rct_candidate = float(zr[local_min_idx] - rs)
            if rct_candidate > 0:
                rct = rct_candidate

        rs = max(0.01, rs)
        rct = max(0.01, rct)

        return EISAnalysis(
            solution_resistance_rs_ohm=round(rs, 2),
            charge_transfer_resistance_rct_ohm=round(rct, 2),
            characteristic_frequency_hz=round(f_apex, 1) if f_apex is not None else None,
            equivalent_circuit='R_s + R_ct // CPE',
        )

    def analyze_raw(self, raw_data: CyclerRawData) -> EISAnalysis:
        """Backward-compatible alias for analyze."""
        return self.analyze(raw_data)
