"""
Linear Sweep Voltammetry (LSV) & Corrosion/HER Kinetics Engine.
Extracts:
1. Hydrogen Evolution Reaction (HER) onset overpotential at target current density threshold.
2. Corrosion potential (E_corr) and corrosion current density (j_corr).
3. Anodic and cathodic Tafel slopes (mV/decade).
"""

from typing import Optional, Tuple
import numpy as np

from models import NormalizedElectrochemicalData, LSVAnalysis


class LSVAnalyzer:
    """Evaluates HER overpotential and zinc corrosion kinetics from linear sweep data."""

    def __init__(self, her_threshold_ma_cm2: float = -1.0):
        self.her_threshold = her_threshold_ma_cm2

    def analyze(self, data: NormalizedElectrochemicalData) -> LSVAnalysis:
        v = np.array(data.voltage_v, dtype=float)
        j = np.array(data.current_density_ma_cm2, dtype=float)

        if len(v) < 5:
            return LSVAnalysis()

        # Isolate active linear sweep from pre-sweep settling / OCP points
        if len(v) > 10:
            if v[-1] >= v[0]:  # Anodic sweep direction
                start_idx = int(np.argmin(v[: min(50, len(v))]))
            else:  # Cathodic sweep direction
                start_idx = int(np.argmax(v[: min(50, len(v))]))
            v = v[start_idx:]
            j = j[start_idx:]

        # HER onset potential (potential where cathodic current exceeds threshold)
        her_onset = float(v[-1])
        cathodic_mask = j <= self.her_threshold
        if np.any(cathodic_mask):
            her_onset = float(v[cathodic_mask][0])
        else:
            her_onset = float(v[np.argmin(j)])

        # E_corr: minimum absolute current density during the active sweep (zero-crossing)
        abs_j = np.abs(j)
        min_idx = int(np.argmin(abs_j))
        ecorr = float(v[min_idx])

        # Tafel analysis: log10(|j|) vs V in linear regions (30 mV to 200 mV away from E_corr)
        safe_abs_j = np.where(abs_j < 1e-8, 1e-8, abs_j)
        log_j = np.log10(safe_abs_j)

        cath_mask = (v < ecorr - 0.03) & (v > ecorr - 0.20)
        anod_mask = (v > ecorr + 0.03) & (v < ecorr + 0.20)

        beta_c = 120.0  # default Tafel slope mV/dec
        beta_a = 60.0
        j_corr = float(safe_abs_j[min_idx])

        slope_c, inter_c = None, None
        slope_a, inter_a = None, None

        if np.sum(cath_mask) >= 4:
            slope_c, inter_c = np.polyfit(v[cath_mask], log_j[cath_mask], deg=1)
            if slope_c != 0:
                beta_c = abs(1.0 / slope_c) * 1000.0

        if np.sum(anod_mask) >= 4:
            slope_a, inter_a = np.polyfit(v[anod_mask], log_j[anod_mask], deg=1)
            if slope_a != 0:
                beta_a = abs(1.0 / slope_a) * 1000.0

        if np.sum(cath_mask) >= 4 and np.sum(anod_mask) >= 4 and slope_c is not None and slope_a is not None and slope_c != slope_a:
            # Intersection of cathodic and anodic Tafel slopes
            v_corr = (inter_a - inter_c) / (slope_c - slope_a)
            log_j_corr = slope_a * v_corr + inter_a
            j_corr = float(10.0 ** log_j_corr)

        j_corr = max(1e-5, min(j_corr, 50.0))

        # Electrochemical Stability Window (ESW)
        anodic_limit_v = None
        anod_thresh_mask = j >= abs(self.her_threshold)
        if np.any(anod_thresh_mask):
            anodic_limit_v = float(v[anod_thresh_mask][0])
        elif max(v) > ecorr:
            anodic_limit_v = float(max(v))

        esw_v = None
        if anodic_limit_v is not None and her_onset is not None:
            esw_v = round(abs(anodic_limit_v - her_onset), 3)

        return LSVAnalysis(
            her_onset_potential_v=round(her_onset, 3),
            her_current_density_threshold_ma_cm2=self.her_threshold,
            corrosion_potential_ecorr_v=round(ecorr, 3),
            corrosion_current_jcorr_ma_cm2=round(j_corr, 5),
            tafel_slope_cathodic_mv_per_dec=round(beta_c, 1),
            tafel_slope_anodic_mv_per_dec=round(beta_a, 1),
            electrochemical_stability_window_v=esw_v,
        )
