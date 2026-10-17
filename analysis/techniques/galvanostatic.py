"""
Galvanostatic Cycling Analysis for Aqueous Zinc-Ion Batteries (ZIB).
Analyzes:
1. Zn||Zn symmetric cells (overpotential, hysteresis, lifetime, dendrite hard/soft-short detection).
2. Zn||Cu / Zn||Ti asymmetric half-cells (Coulombic Efficiency CE, nucleation overpotential).
3. Zn||Cathode full cells (capacity retention, rate capability, CE over cycles).
"""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np

from models import (
    NormalizedElectrochemicalData,
    SymmetricCellCycleMetric,
    ActiveCyclingMetrics,
    TotalTestSummary,
    SymmetricCellAnalysis,
    AsymmetricHalfCellCycleMetric,
    AsymmetricHalfCellAnalysis,
    FullCellAnalysis,
)
from analysis.processing.math_engine import MathEngine


class SymmetricCellAnalyzer:
    """Evaluates zinc plating/stripping reversibility and lifetime in symmetric cells."""

    def __init__(
        self,
        short_circuit_threshold_mv: float = 5.0,
        polarization_cutoff_v: float = 1.0,
        noise_threshold_factor: float = 3.0,
    ):
        self.short_mv = short_circuit_threshold_mv
        self.cutoff_v = polarization_cutoff_v
        self.noise_factor = noise_threshold_factor

    def analyze(self, data: NormalizedElectrochemicalData) -> SymmetricCellAnalysis:
        v_arr = np.array(data.voltage_v, dtype=float)
        i_arr = np.array(data.current_ma, dtype=float)
        t_arr = np.array(data.time_s, dtype=float)
        areal_q = np.array(data.areal_capacity_mah_cm2, dtype=float)

        polarity = np.where(i_arr > 0.005, 1, np.where(i_arr < -0.005, -1, 0))
        transitions = np.where(polarity[:-1] != polarity[1:])[0]
        segment_indices = [0] + (transitions + 1).tolist() + [len(v_arr)]

        cycle_metrics: List[SymmetricCellCycleMetric] = []
        short_detected = False
        soft_short_detected = False
        failure_cycle: Optional[int] = None
        failure_time_h: Optional[float] = None
        failure_mode: Optional[str] = None

        half_cycles = []
        for s_start, s_end in zip(segment_indices[:-1], segment_indices[1:]):
            if s_end - s_start < 2:
                continue
            seg_pol = polarity[s_start + (s_end - s_start) // 2]
            if seg_pol != 0:
                half_cycles.append((seg_pol, s_start, s_end))

        # Pre-compute plateau boundaries to vectorize span calculations
        hc_arr = np.array(half_cycles)
        if len(hc_arr) > 0:
            pols, starts, ends = hc_arr[:, 0], hc_arr[:, 1], hc_arr[:, 2]
            spans = ends - starts
            p_froms = starts + (0.2 * spans).astype(int)
            p_tos = starts + (0.8 * spans).astype(int)
            p_tos = np.maximum(p_tos, p_froms + 1)
        else:
            pols, starts, ends, p_froms, p_tos = [], [], [], [], []

        cycle_idx = 1
        i_half = 0
        while i_half < len(pols) - 1:
            if pols[i_half] != pols[i_half + 1]:
                idx1, idx2 = i_half, i_half + 1
                if pols[idx1] > 0:
                    s_idx, p_idx = idx1, idx2
                else:
                    p_idx, s_idx = idx1, idx2

                # Safe plateau extraction avoiding transition zero-crossings
                v_strip = v_arr[p_froms[s_idx]:p_tos[s_idx]]
                v_plate = v_arr[p_froms[p_idx]:p_tos[p_idx]]
                
                v_strip_med, v_strip_std = float(np.median(v_strip)), float(np.std(v_strip))
                v_plate_med, v_plate_std = float(np.median(v_plate)), float(np.std(v_plate))

                overpotential_mv = abs(v_strip_med - v_plate_med) / 2.0 * 1000.0
                hysteresis_mv = abs(v_strip_med - v_plate_med) * 1000.0
                half_q = float(abs(areal_q[ends[idx2] - 1] - areal_q[starts[idx1]])) / 2.0

                cycle_metrics.append(
                    SymmetricCellCycleMetric(
                        cycle_number=cycle_idx,
                        stripping_plateau_v=round(v_strip_med, 4),
                        plating_plateau_v=round(v_plate_med, 4),
                        overpotential_mv=round(overpotential_mv, 2),
                        voltage_hysteresis_mv=round(hysteresis_mv, 2),
                        areal_capacity_mah_cm2=round(half_q, 4),
                    )
                )

                # Point-by-point sudden drop check locked inside 20-80% window
                short_thresh_v = self.short_mv / 1000.0
                is_soft_short = cycle_idx > 3 and (v_strip_std * 1000.0 > 15.0 or v_plate_std * 1000.0 > 15.0)
                sudden_drop = (np.any(np.abs(v_strip) < short_thresh_v) or np.any(np.abs(v_plate) < short_thresh_v)) and not is_soft_short

                if (overpotential_mv < self.short_mv or sudden_drop) and not short_detected:
                    short_detected = True
                    failure_cycle = cycle_idx
                    failure_time_h = round(float(t_arr[ends[idx2] - 1]) / 3600.0, 2)
                    failure_mode = "dendritic_short"
                elif (abs(v_strip_med) >= self.cutoff_v or abs(v_plate_med) >= self.cutoff_v) and not failure_mode:
                    failure_cycle = cycle_idx
                    failure_time_h = round(float(t_arr[ends[idx2] - 1]) / 3600.0, 2)
                    failure_mode = "polarization_cut_off"
                elif is_soft_short and not soft_short_detected:
                    soft_short_detected = True
                    if not failure_mode:
                        failure_cycle = cycle_idx
                        failure_time_h = round(float(t_arr[ends[idx2] - 1]) / 3600.0, 2)
                        failure_mode = "soft_short"

                cycle_idx += 1
                i_half += 2
            else:
                i_half += 1

        total_lifespan_h = round(float(t_arr[-1]) / 3600.0, 2) if len(t_arr) > 0 else 0.0
        mean_overpotential = float(np.mean([m.overpotential_mv for m in cycle_metrics])) if cycle_metrics else 0.0
        mean_hysteresis = float(np.mean([m.voltage_hysteresis_mv for m in cycle_metrics])) if cycle_metrics else 0.0

        # Dual-tier metrics calculation
        if failure_cycle is not None and failure_cycle > 0:
            active_cycles = [m for m in cycle_metrics if m.cycle_number < failure_cycle]
            if not active_cycles and cycle_metrics:
                active_cycles = [cycle_metrics[0]]
            active_overpot = float(np.mean([m.overpotential_mv for m in active_cycles])) if active_cycles else 0.0
            active_hyst = float(np.mean([m.voltage_hysteresis_mv for m in active_cycles])) if active_cycles else 0.0
            active_lifespan = failure_time_h if failure_time_h is not None else total_lifespan_h
            active_pts = int(np.sum(t_arr <= (active_lifespan * 3600.0))) if len(t_arr) > 0 else 0
            active_count = len(active_cycles)
        else:
            active_overpot = mean_overpotential
            active_hyst = mean_hysteresis
            active_lifespan = total_lifespan_h
            active_pts = len(t_arr)
            active_count = len(cycle_metrics)

        performance_score = round(active_lifespan / (1.0 + active_overpot / 50.0), 3)

        active_metrics = ActiveCyclingMetrics(
            mean_overpotential_mv=round(active_overpot, 2),
            mean_hysteresis_mv=round(active_hyst, 2),
            active_lifespan_hours=round(active_lifespan, 2),
            active_cycles=active_count,
            data_points=active_pts,
            performance_score=performance_score
        )

        diag_msg: Optional[str] = None
        if short_detected:
            diag_msg = f"Dendrite Short-Circuit Detected at {failure_time_h:.1f} h (Cycle {failure_cycle}) - Potential < {self.short_mv:.1f} mV"
        elif soft_short_detected:
            diag_msg = f"Soft-Short / Voltage Micro-Oscillation Detected at {failure_time_h:.1f} h (Cycle {failure_cycle})"
        elif failure_mode == "polarization_cut_off":
            diag_msg = f"Polarization Cutoff Exceeded (Byproduct Passivation) at {failure_time_h:.1f} h (Cycle {failure_cycle}) - Potential >= {self.cutoff_v:.1f} V"
        else:
            diag_msg = "Stable cycling completed without failure"

        total_summary = TotalTestSummary(
            nominal_duration_hours=total_lifespan_h,
            total_data_points=len(t_arr),
            failure_mode=failure_mode if failure_mode else "none",
            failure_time_hours=failure_time_h,
            failure_cycle=failure_cycle,
            short_circuit_detected=short_detected,
            diagnostic_message=diag_msg,
        )

        return SymmetricCellAnalysis(
            total_cycles=len(cycle_metrics),
            total_lifespan_hours=total_lifespan_h,
            mean_overpotential_mv=round(mean_overpotential, 2),
            mean_hysteresis_mv=round(mean_hysteresis, 2),
            short_circuit_detected=short_detected,
            soft_short_detected=soft_short_detected,
            failure_cycle=failure_cycle,
            failure_time_hours=failure_time_h,
            failure_mode=failure_mode if failure_mode else "none",
            cycle_metrics=cycle_metrics,
            active_cycling_metrics=active_metrics.model_dump(),
            total_test_summary=total_summary.model_dump(),
        )


class AsymmetricCellAnalyzer:
    """Evaluates zinc plating/stripping Coulombic Efficiency (CE) on foreign substrate (Cu, Ti)."""

    def analyze(self, data: NormalizedElectrochemicalData) -> AsymmetricHalfCellAnalysis:
        v_arr = np.array(data.voltage_v, dtype=float)
        i_arr = np.array(data.current_ma, dtype=float)
        t_arr = np.array(data.time_s, dtype=float)
        areal_q = np.array(data.areal_capacity_mah_cm2, dtype=float)

        # Determine effective electrode area
        area = 1.13097
        if "area_cm2" in data.metadata:
            try:
                area = float(data.metadata["area_cm2"])
            except (ValueError, TypeError):
                pass
        elif (
            len(data.capacity_mah) > 0
            and len(data.areal_capacity_mah_cm2) > 0
            and data.areal_capacity_mah_cm2[-1] > 1e-6
            and data.capacity_mah[-1] > 1e-6
        ):
            area = data.capacity_mah[-1] / data.areal_capacity_mah_cm2[-1]

        polarity = np.where(i_arr < -0.005, -1, np.where(i_arr > 0.005, 1, 0))
        transitions = np.where(polarity[:-1] != polarity[1:])[0]
        segments = [0] + (transitions + 1).tolist() + [len(v_arr)]

        half_cycles = []
        for s_start, s_end in zip(segments[:-1], segments[1:]):
            if s_end - s_start < 2:
                continue
            pol = polarity[s_start + (s_end - s_start) // 2]
            if pol != 0:
                half_cycles.append((pol, s_start, s_end))

        def _get_segment_delta_q(start: int, end: int, pol: int) -> float:
            """
            Compute true step capacity delta ΔQ for plating or stripping segment.
            First-principles methodology:
            1. Try raw discharge capacity (for plating, pol == -1) or charge capacity (for stripping, pol == 1).
            2. Try alternate raw capacity column if polarity was inverted.
            3. Try areal_capacity_mah_cm2.
            4. Fallback to trapezoidal integration of current over time: ∫ |I| dt / 3600 / area.
            """
            t_seg = t_arr[start:end].tolist()
            i_seg = i_arr[start:end].tolist()

            # 1. Primary raw column
            primary_raw = (
                data.capacity_discharge_mah if pol == -1 else data.capacity_charge_mah
            )
            if primary_raw is not None and len(primary_raw) == len(t_arr):
                val, method = MathEngine.compute_step_capacity_delta(
                    time_s=t_seg,
                    current_ma=i_seg,
                    raw_capacity=primary_raw[start:end],
                    expected_polarity=pol,
                )
                if method == "delta":
                    return val / area

            # 2. Alternate raw column (polarity inverted cycler)
            alt_raw = (
                data.capacity_charge_mah if pol == -1 else data.capacity_discharge_mah
            )
            if alt_raw is not None and len(alt_raw) == len(t_arr):
                val, method = MathEngine.compute_step_capacity_delta(
                    time_s=t_seg,
                    current_ma=i_seg,
                    raw_capacity=alt_raw[start:end],
                    expected_polarity=pol,
                )
                if method == "delta":
                    return val / area

            # 3. Areal capacity column
            if len(areal_q) == len(t_arr):
                val, method = MathEngine.compute_step_capacity_delta(
                    time_s=t_seg,
                    current_ma=i_seg,
                    raw_capacity=areal_q[start:end].tolist(),
                    expected_polarity=pol,
                )
                if method == "delta":
                    return val

            # 4. Fallback: numerical trapezoidal integration
            val, _ = MathEngine.compute_step_capacity_delta(
                time_s=t_seg,
                current_ma=i_seg,
                raw_capacity=None,
                expected_polarity=pol,
            )
            return val / area

        metrics: List[AsymmetricHalfCellCycleMetric] = []
        cycle_idx = 1
        i_half = 0
        while i_half < len(half_cycles) - 1:
            pol1, start1, end1 = half_cycles[i_half]
            pol2, start2, end2 = half_cycles[i_half + 1]

            if pol1 == -1 and pol2 == 1:
                plate_v = v_arr[start1:end1]
                v_dip = float(np.min(plate_v[: max(3, len(plate_v) // 4)]))
                span = len(plate_v)
                v_plateau = float(np.median(plate_v[span // 4 : 3 * span // 4]))
                nucleation_overpotential_mv = abs(v_dip - v_plateau) * 1000.0
                plateau_overpotential_mv = abs(v_plateau) * 1000.0

                q_plate = _get_segment_delta_q(start1, end1, -1)
                q_strip = _get_segment_delta_q(start2, end2, 1)

                ce_percent = (q_strip / q_plate * 100.0) if q_plate > 0 else 0.0
                ce_percent = min(100.0, max(0.0, ce_percent))

                metrics.append(
                    AsymmetricHalfCellCycleMetric(
                        cycle_number=cycle_idx,
                        plating_capacity_mah_cm2=round(q_plate, 4),
                        stripping_capacity_mah_cm2=round(q_strip, 4),
                        coulombic_efficiency_percent=round(ce_percent, 2),
                        nucleation_overpotential_mv=round(nucleation_overpotential_mv, 2),
                        plateau_overpotential_mv=round(plateau_overpotential_mv, 2),
                    )
                )
                cycle_idx += 1
                i_half += 2
            else:
                i_half += 1

        avg_ce = float(np.mean([m.coulombic_efficiency_percent for m in metrics])) if metrics else 0.0
        final_ce = metrics[-1].coulombic_efficiency_percent if metrics else 0.0
        cum_cap = float(np.sum([m.plating_capacity_mah_cm2 for m in metrics])) if metrics else 0.0
        mean_nuc = float(np.mean([m.nucleation_overpotential_mv for m in metrics])) if metrics else 0.0

        return AsymmetricHalfCellAnalysis(
            total_cycles=len(metrics),
            average_ce_percent=round(avg_ce, 2),
            final_ce_percent=round(final_ce, 2),
            cumulative_capacity_mah_cm2=round(cum_cap, 4),
            mean_nucleation_overpotential_mv=round(mean_nuc, 2),
            cycle_metrics=metrics,
        )


class FullCellAnalyzer:
    """Evaluates capacity retention, rate performance, CE, and Energy Efficiency in aqueous ZIB full cells."""

    def analyze(self, data: NormalizedElectrochemicalData) -> FullCellAnalysis:
        cycle_nums: List[int] = []
        q_discharges: List[float] = []
        ces: List[float] = []
        energy_efficiencies: List[float] = []

        if data.metadata.get("is_cycle_summary") or (data.coulombic_efficiency and data.cycle_number and len(data.coulombic_efficiency) == len(data.cycle_number) and len(data.cycle_number) == len(set(data.cycle_number))):
            cycle_nums = list(data.cycle_number) if data.cycle_number else list(range(1, len(data.time_s) + 1))
            if getattr(data, "specific_capacity_mah_g", None):
                q_discharges = [round(float(q), 2) for q in data.specific_capacity_mah_g]
            elif getattr(data, "capacity_discharge_mah", None):
                q_discharges = [round(float(q), 2) for q in data.capacity_discharge_mah]
            elif getattr(data, "capacity_mah", None):
                q_discharges = [round(float(q), 2) for q in data.capacity_mah]
            elif "summary_capacity" in data.metadata:
                q_discharges = [round(float(q), 2) for q in data.metadata["summary_capacity"]]
            else:
                q_discharges = [0.0] * len(cycle_nums)

            raw_ces = []
            if getattr(data, "coulombic_efficiency", None):
                raw_ces = data.coulombic_efficiency
            elif "summary_coulombic_efficiency" in data.metadata:
                raw_ces = data.metadata["summary_coulombic_efficiency"]
            elif getattr(data, "capacity_charge_mah", None) and len(data.capacity_charge_mah) == len(q_discharges):
                raw_ces = [
                    (qd / qc * 100.0) if qc > 0 else 0.0
                    for qd, qc in zip(q_discharges, data.capacity_charge_mah)
                ]
            elif data.metadata.get("is_cycle_summary"):
                raw_ces = [99.5] * len(cycle_nums)
            else:
                raw_ces = [0.0] * len(cycle_nums)

            ces = []
            for c in raw_ces:
                c_val = float(c)
                if 0.0 < c_val <= 1.05:
                    c_val *= 100.0
                ces.append(round(c_val, 2))

            energy_efficiencies = []

        elif data.cycle_number is not None and len(data.cycle_number) == len(data.voltage_v):
            cycles = sorted(list(set(data.cycle_number)))
            cap_source = data.specific_capacity_mah_g if data.specific_capacity_mah_g else data.areal_capacity_mah_cm2
            v_arr = np.array(data.voltage_v)
            i_arr = np.array(data.current_ma)
            t_arr = np.array(data.time_s)
            c_arr = np.array(data.cycle_number)

            # Determine capacity normalization scale factor
            scale = 1.0
            if (
                data.capacity_mah is not None
                and len(data.capacity_mah) > 0
                and data.capacity_mah[-1] > 1e-6
                and len(cap_source) > 0
                and cap_source[-1] > 1e-6
            ):
                scale = cap_source[-1] / data.capacity_mah[-1]

            for cyc in cycles:
                if cyc <= 0:
                    continue
                mask = c_arr == cyc
                cyc_v = v_arr[mask]
                cyc_i = i_arr[mask]
                cyc_t = t_arr[mask]
                cyc_cap = [cap_source[idx] for idx, m in enumerate(mask) if m]

                disch_idx = np.where(mask & (i_arr < -0.005))[0]
                chg_idx = np.where(mask & (i_arr > 0.005))[0]

                # Discharge capacity calculation
                q_disch = 0.0
                if len(disch_idx) >= 2:
                    t_disch = t_arr[disch_idx].tolist()
                    i_disch = i_arr[disch_idx].tolist()
                    raw_disch = (
                        [data.capacity_discharge_mah[k] for k in disch_idx]
                        if data.capacity_discharge_mah is not None
                        else None
                    )
                    val, method = MathEngine.compute_step_capacity_delta(
                        time_s=t_disch,
                        current_ma=i_disch,
                        raw_capacity=raw_disch,
                        expected_polarity=-1,
                    )
                    if method == "delta":
                        q_disch = val * scale
                    else:
                        cap_seg = [cap_source[k] for k in disch_idx]
                        val_c, meth_c = MathEngine.compute_step_capacity_delta(
                            time_s=t_disch,
                            current_ma=i_disch,
                            raw_capacity=cap_seg,
                            expected_polarity=-1,
                        )
                        q_disch = val_c if meth_c == "delta" else val * scale
                elif cyc_cap:
                    q_disch = (
                        max(cyc_cap) - min(cyc_cap)
                        if max(cyc_cap) > min(cyc_cap)
                        else cyc_cap[0]
                    )

                # Charge capacity calculation
                q_chg = 0.0
                if len(chg_idx) >= 2:
                    t_chg = t_arr[chg_idx].tolist()
                    i_chg = i_arr[chg_idx].tolist()
                    raw_chg = (
                        [data.capacity_charge_mah[k] for k in chg_idx]
                        if data.capacity_charge_mah is not None
                        else None
                    )
                    val, method = MathEngine.compute_step_capacity_delta(
                        time_s=t_chg,
                        current_ma=i_chg,
                        raw_capacity=raw_chg,
                        expected_polarity=1,
                    )
                    if method == "delta":
                        q_chg = val * scale
                    else:
                        cap_seg = [cap_source[k] for k in chg_idx]
                        val_c, meth_c = MathEngine.compute_step_capacity_delta(
                            time_s=t_chg,
                            current_ma=i_chg,
                            raw_capacity=cap_seg,
                            expected_polarity=1,
                        )
                        q_chg = val_c if meth_c == "delta" else val * scale

                # Coulombic Efficiency: CE = Q_disch / Q_chg * 100
                if q_chg > 0 and q_disch > 0:
                    ce = round((q_disch / q_chg) * 100.0, 2)
                    ce = min(100.0, max(0.0, ce))
                elif data.coulombic_efficiency is not None and len(data.coulombic_efficiency) == len(data.voltage_v):
                    cyc_ces = [data.coulombic_efficiency[idx] for idx, m in enumerate(mask) if m and data.coulombic_efficiency[idx] > 0]
                    ce = round(float(np.mean(cyc_ces)), 2) if cyc_ces else 0.0
                else:
                    ce = 0.0

                cycle_nums.append(cyc)
                q_discharges.append(round(q_disch, 2))
                ces.append(ce)

                # Energy efficiency calculation: E_disch / E_chg * 100
                chg_sub = cyc_i > 0.001
                disch_sub = cyc_i < -0.001
                if np.sum(chg_sub) >= 2 and np.sum(disch_sub) >= 2:
                    e_chg = float(np.trapezoid(cyc_v[chg_sub] * cyc_i[chg_sub], cyc_t[chg_sub])) if hasattr(np, "trapezoid") else float(np.trapz(cyc_v[chg_sub] * cyc_i[chg_sub], cyc_t[chg_sub]))
                    e_disch = float(np.trapezoid(cyc_v[disch_sub] * np.abs(cyc_i[disch_sub]), cyc_t[disch_sub])) if hasattr(np, "trapezoid") else float(np.trapz(cyc_v[disch_sub] * np.abs(cyc_i[disch_sub]), cyc_t[disch_sub]))
                    if e_chg > 0:
                        ee = round((e_disch / e_chg * 100.0), 2)
                        ee = min(100.0, max(0.0, ee))
                        energy_efficiencies.append(ee)
                    else:
                        energy_efficiencies.append(0.0)
                else:
                    energy_efficiencies.append(0.0)
        else:
            cycle_nums = [1]
            q_discharges = [round(max(data.areal_capacity_mah_cm2), 2)] if data.areal_capacity_mah_cm2 else [0.0]
            ces = [0.0]
            energy_efficiencies = []

        init_q = q_discharges[0] if q_discharges else 0.0
        final_q = q_discharges[-1] if q_discharges else 0.0
        retention = (final_q / init_q * 100.0) if init_q > 0 else 0.0
        valid_ees = [e for e in energy_efficiencies if e > 0]
        avg_ee = round(float(np.mean(valid_ees)), 2) if valid_ees else 0.0
        final_ee = energy_efficiencies[-1] if energy_efficiencies else 0.0

        return FullCellAnalysis(
            initial_discharge_capacity_mah_g=init_q,
            final_discharge_capacity_mah_g=final_q,
            capacity_retention_percent=round(retention, 2),
            average_coulombic_efficiency_percent=round(float(np.mean(ces)), 2) if ces else 0.0,
            energy_efficiency_percent=final_ee,
            average_energy_efficiency_percent=avg_ee,
            total_cycles=len(cycle_nums),
            cycle_numbers=cycle_nums,
            discharge_capacities_mah_g=q_discharges,
            coulombic_efficiencies=ces,
            energy_efficiencies=energy_efficiencies,
        )
