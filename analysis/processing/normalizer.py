"""
Electrode Normalization Engine for Zinc-Ion Battery Characterization.
Normalizes raw cycler currents and capacities by geometric area (cm²) and active mass (mg).
"""

from typing import Optional
import numpy as np

from models import CyclerRawData, ElectrodeGeometry, NormalizedElectrochemicalData
from analysis.processing.rest_detector import RestDetector
from analysis.processing.time_scaler import TimeScaler
from analysis.processing.math_engine import MathEngine


class ElectrochemicalNormalizer:
    """Normalizes raw cycler time series using physical electrode dimensions."""

    def __init__(self, geometry: Optional[ElectrodeGeometry] = None):
        self.geometry = geometry if geometry else ElectrodeGeometry()

    def process(
        self,
        raw_data: CyclerRawData,
        auto_trim_rest: bool = True,
        drop_rest_min: Optional[float] = None,
        compute_dq_dv: bool = True,
    ) -> NormalizedElectrochemicalData:
        """Transform raw cycler data into fully normalized, unit-scaled electrochemical dataset."""
        trimmed_data = raw_data
        trimmed_seconds = 0.0

        is_summary = bool(raw_data.metadata.get("is_cycle_summary"))

        if not is_summary and (auto_trim_rest or drop_rest_min is not None):
            trimmed_data, trimmed_seconds = RestDetector.trim_initial_rest(
                raw_data=raw_data,
                drop_rest_min=drop_rest_min,
            )

        area = self.geometry.area_cm2
        mass_mg = self.geometry.active_mass_mg

        display_time, time_unit = TimeScaler.scale_time_series(trimmed_data.time_s)

        current_density = [round(i / area, 5) for i in trimmed_data.current_ma]

        if is_summary:
            cum_q = trimmed_data.capacity_discharge_mah or trimmed_data.capacity_charge_mah or [0.0] * len(trimmed_data.time_s)
            if mass_mg is not None and mass_mg > 0:
                specific_cap = [round(q * 1000.0 / mass_mg, 3) for q in cum_q]
            else:
                specific_cap = list(cum_q)
            areal_cap = [round(q / area, 5) for q in cum_q]
            dq_dv_values = None
            capacity_mode = "cycle_summary"
        else:
            # Capacity calculation: first-principles unrolled throughput with trapezoidal fallback
            has_chg = (
                trimmed_data.capacity_charge_mah is not None
                and len(trimmed_data.capacity_charge_mah) == len(trimmed_data.time_s)
                and any(q > 0 for q in trimmed_data.capacity_charge_mah)
            )
            has_disch = (
                trimmed_data.capacity_discharge_mah is not None
                and len(trimmed_data.capacity_discharge_mah) == len(trimmed_data.time_s)
                and any(q > 0 for q in trimmed_data.capacity_discharge_mah)
            )

            cum_q = None
            capacity_mode = "numerical_trapezoidal"

            if has_chg or has_disch:
                cum_chg = (
                    MathEngine.unroll_capacity_resets(trimmed_data.capacity_charge_mah)
                    if has_chg
                    else [0.0] * len(trimmed_data.time_s)
                )
                cum_disch = (
                    MathEngine.unroll_capacity_resets(trimmed_data.capacity_discharge_mah)
                    if has_disch
                    else [0.0] * len(trimmed_data.time_s)
                )
                unrolled = [round(float(c) + float(d), 5) for c, d in zip(cum_chg, cum_disch)]

                # Check validity: finite, non-negative, and monotonic
                arr_q = np.array(unrolled, dtype=float)
                if (
                    len(arr_q) > 0
                    and np.all(np.isfinite(arr_q))
                    and np.all(arr_q >= -1e-5)
                    and np.all(np.diff(arr_q) >= -1e-5)
                    and arr_q[-1] > 0
                ):
                    cum_q = unrolled
                    capacity_mode = "unrolled_manufacturer"

            if cum_q is None:
                cum_q, _, _ = MathEngine.integrate_capacity(
                    time_s=trimmed_data.time_s,
                    current_ma=trimmed_data.current_ma,
                )
                capacity_mode = "numerical_trapezoidal"

            areal_cap = [round(q / area, 5) for q in cum_q]

            specific_cap = None
            if mass_mg is not None and mass_mg > 0:
                specific_cap = [round(q * 1000.0 / mass_mg, 3) for q in cum_q]

            dq_dv_values = None
            if compute_dq_dv and len(trimmed_data.voltage_v) >= 5:
                dq_dv_values = MathEngine.compute_dq_dv(
                    voltage_v=trimmed_data.voltage_v,
                    capacity_mah=areal_cap,
                )

        ce_data = trimmed_data.coulombic_efficiency or trimmed_data.metadata.get("summary_coulombic_efficiency")

        return NormalizedElectrochemicalData(
            time_s=trimmed_data.time_s,
            display_time=display_time,
            time_unit=time_unit,
            voltage_v=trimmed_data.voltage_v,
            current_ma=trimmed_data.current_ma,
            current_density_ma_cm2=current_density,
            capacity_mah=cum_q,
            areal_capacity_mah_cm2=areal_cap,
            specific_capacity_mah_g=specific_cap,
            dq_dv=dq_dv_values,
            step_index=trimmed_data.step_index,
            cycle_number=trimmed_data.cycle_number,
            capacity_charge_mah=trimmed_data.capacity_charge_mah,
            capacity_discharge_mah=trimmed_data.capacity_discharge_mah,
            coulombic_efficiency=ce_data,
            rest_trimmed_seconds=trimmed_seconds,
            metadata={
                **trimmed_data.metadata,
                'capacity_mode': capacity_mode,
                'area_cm2': area,
                'active_mass_mg': mass_mg,
                **({'z_real_ohm': trimmed_data.z_real_ohm} if trimmed_data.z_real_ohm is not None else {}),
                **({'z_imag_ohm': trimmed_data.z_imag_ohm} if trimmed_data.z_imag_ohm is not None else {}),
                **({'frequency_hz': trimmed_data.frequency_hz} if trimmed_data.frequency_hz is not None else {}),
            },
        )
