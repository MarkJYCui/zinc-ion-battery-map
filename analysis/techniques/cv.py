"""
Cyclic Voltammetry (CV) Plating/Stripping Reversibility Engine.
Extracts:
1. Anodic peak potential (E_pa) and cathodic peak potential (E_pc).
2. Peak current separation ΔE_p (mV).
3. Peak current ratio (I_pa / |I_pc|).
4. Zinc plating and stripping onset potentials.
"""

from typing import Optional, Tuple
import numpy as np

from models import NormalizedElectrochemicalData, CVAnalysis


class CVAnalyzer:
    """Analyzes cyclic voltammograms for zinc deposition/dissolution reversibility."""

    def analyze(self, data: NormalizedElectrochemicalData) -> CVAnalysis:
        v = np.array(data.voltage_v, dtype=float)
        i = np.array(data.current_ma, dtype=float)

        if len(v) < 5:
            return CVAnalysis()

        anodic_mask = i > 0
        if np.any(anodic_mask):
            anod_sub_idx = int(np.argmax(i[anodic_mask]))
            epa = float(v[anodic_mask][anod_sub_idx])
            ipa = float(i[anodic_mask][anod_sub_idx])
        else:
            epa = float(v[np.argmax(i)])
            ipa = float(np.max(i))

        cathodic_mask = i < 0
        if np.any(cathodic_mask):
            cath_sub_idx = int(np.argmin(i[cathodic_mask]))
            epc = float(v[cathodic_mask][cath_sub_idx])
            ipc = float(i[cathodic_mask][cath_sub_idx])
        else:
            epc = float(v[np.argmin(i)])
            ipc = float(np.min(i))

        delta_ep_mv = abs(epa - epc) * 1000.0

        if abs(ipc) > 1e-6:
            ratio = abs(ipa / ipc)
        else:
            ratio = 1.0

        # Plating onset (cathodic, at 5% of peak cathodic current)
        i_plate_thresh = 0.05 * ipc
        plate_onset_mask = i <= i_plate_thresh
        if np.any(plate_onset_mask):
            plating_onset_v = float(v[plate_onset_mask][0])
        else:
            plating_onset_v = epc

        # Stripping onset (anodic, at 5% of peak anodic current)
        i_strip_thresh = 0.05 * ipa
        strip_onset_mask = i >= i_strip_thresh
        if np.any(strip_onset_mask):
            stripping_onset_v = float(v[strip_onset_mask][0])
        else:
            stripping_onset_v = epa

        # Nucleation overpotential: difference between plating onset and crossover / equilibrium
        zero_crossings = np.where(np.diff(np.signbit(i)))[0]
        if len(zero_crossings) > 0:
            crossover_v = float(v[zero_crossings[0]])
        else:
            crossover_v = stripping_onset_v
        nucleation_overpot_mv = abs(plating_onset_v - crossover_v) * 1000.0

        return CVAnalysis(
            anodic_peak_potential_epa_v=round(epa, 3),
            cathodic_peak_potential_epc_v=round(epc, 3),
            anodic_peak_current_ipa_ma=round(ipa, 3),
            cathodic_peak_current_ipc_ma=round(ipc, 3),
            peak_separation_delta_ep_mv=round(delta_ep_mv, 1),
            current_ratio_ipa_ipc=round(ratio, 3),
            plating_onset_potential_v=round(plating_onset_v, 3),
            stripping_onset_potential_v=round(stripping_onset_v, 3),
            nucleation_overpotential_mv=round(nucleation_overpot_mv, 1),
        )
