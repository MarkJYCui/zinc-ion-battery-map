"""
Smart Rest-Period Auto-Detection and Trimming for Battery Cycler Ingestion.
Automatically detects and removes leading Open Circuit Voltage (OCV) rest steps
and re-zeros time to begin precisely at the initiation of galvanostatic cycling.
"""

from typing import Tuple, Optional
import numpy as np

from models import CyclerRawData


class RestDetector:
    """Detects initial OCV rest period and trims leading non-cycling data."""

    @staticmethod
    def detect_rest_steps(
        current_ma: list[float],
        current_threshold_ma: float = 0.005,
    ) -> list[bool]:
        """Classify each point as rest or cycling based on current magnitude."""
        return [abs(c) <= current_threshold_ma for c in current_ma]

    @staticmethod
    def trim_initial_rest(
        raw_data: CyclerRawData,
        current_threshold_ma: float = 0.005,
        drop_rest_min: Optional[float] = None,
    ) -> Tuple[CyclerRawData, float]:
        """
        Identify initial rest period and return trimmed CyclerRawData with time zeroed.
        """
        n_points = len(raw_data.time_s)
        if n_points == 0:
            return raw_data, 0.0

        cutoff_idx = 0

        if drop_rest_min is not None and drop_rest_min > 0:
            cutoff_seconds = drop_rest_min * 60.0
            for idx, t in enumerate(raw_data.time_s):
                if t >= cutoff_seconds:
                    cutoff_idx = idx
                    break
        else:
            for idx, curr in enumerate(raw_data.current_ma):
                if abs(curr) > current_threshold_ma:
                    cutoff_idx = idx
                    break
            else:
                cutoff_idx = 0

        if cutoff_idx == 0:
            return raw_data, 0.0

        trimmed_seconds = raw_data.time_s[cutoff_idx]
        zeroed_time = [t - trimmed_seconds for t in raw_data.time_s[cutoff_idx:]]

        metadata = dict(raw_data.metadata)
        metadata["trimmed_rest_seconds"] = trimmed_seconds
        metadata["trimmed_points_count"] = cutoff_idx

        trimmed_data = CyclerRawData(
            time_s=zeroed_time,
            voltage_v=raw_data.voltage_v[cutoff_idx:],
            current_ma=raw_data.current_ma[cutoff_idx:],
            step_index=raw_data.step_index[cutoff_idx:] if raw_data.step_index else None,
            cycle_number=raw_data.cycle_number[cutoff_idx:] if raw_data.cycle_number else None,
            capacity_charge_mah=raw_data.capacity_charge_mah[cutoff_idx:] if raw_data.capacity_charge_mah else None,
            capacity_discharge_mah=raw_data.capacity_discharge_mah[cutoff_idx:] if raw_data.capacity_discharge_mah else None,
            frequency_hz=raw_data.frequency_hz[cutoff_idx:] if raw_data.frequency_hz else None,
            z_real_ohm=raw_data.z_real_ohm[cutoff_idx:] if raw_data.z_real_ohm else None,
            z_imag_ohm=raw_data.z_imag_ohm[cutoff_idx:] if raw_data.z_imag_ohm else None,
            metadata=metadata,
        )

        return trimmed_data, trimmed_seconds
