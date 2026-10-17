"""
Mathematical signal processing engine for battery characterization.
Computes numerical capacity integration Q = ∫I dt, differential capacity dQ/dV,
and smoothing filters using pure NumPy.

Portions of this code are adapted from:
- ampworks: BSD 3-Clause License, Copyright (c) 2025, Alliance for Energy Innovation, LLC
  (See third_party_licenses/ampworks_LICENSE.txt)
- cellpy: MIT License, Copyright (c) 2016-2026 cellpy developers
  (See third_party_licenses/cellpy_LICENSE.txt)
"""

import math
from typing import List, Tuple, Optional, Union
import numpy as np


class MathEngine:
    """Numerical processing methods for electrochemical data."""

    @staticmethod
    def calculate_plateau_median_std(values: List[float]) -> Tuple[float, float]:
        """Compute median and standard deviation of plateau voltage points."""
        arr = np.array(values, dtype=float)
        return float(np.median(arr)), float(np.std(arr))

    @staticmethod
    def integrate_capacity(time_s: List[float], current_ma: List[float]) -> Tuple[List[float], float, float]:
        """Integrate current over time via trapezoidal rule to calculate cumulative capacity."""
        if not time_s or not current_ma or len(time_s) != len(current_ma):
            return [], 0.0, 0.0

        t = np.array(time_s, dtype=float)
        i = np.array(current_ma, dtype=float)

        dt = np.diff(t)
        # Guard against negative dt, irregular timestamps, and microsecond clock jitter
        dt = np.where(dt < 1e-5, 0.0, dt)

        avg_i = (i[:-1] + i[1:]) / 2.0
        dq = avg_i * dt / 3600.0  # mAh

        charge_dq = np.where(dq > 0, dq, 0.0)
        discharge_dq = np.where(dq < 0, -dq, 0.0)

        total_charge = float(np.sum(charge_dq))
        total_discharge = float(np.sum(discharge_dq))

        abs_dq = np.abs(dq)
        cum_q = np.insert(np.cumsum(abs_dq), 0, 0.0)

        return cum_q.tolist(), round(total_charge, 4), round(total_discharge, 4)

    @staticmethod
    def integrate_segment_trapezoid(time_s: List[float], current_ma: List[float]) -> float:
        """
        Numerically integrate |I| dt / 3600 via trapezoidal rule for a segment or step.
        Guards against negative dt, non-monotonic timestamps, and zero length.
        """
        if len(time_s) < 2 or len(current_ma) < 2:
            return 0.0

        t = np.array(time_s, dtype=float)
        i = np.array(current_ma, dtype=float)

        dt = np.diff(t)
        dt = np.where(dt < 1e-5, 0.0, dt)
        avg_i = (np.abs(i[:-1]) + np.abs(i[1:])) / 2.0
        q_mah = float(np.sum(avg_i * dt / 3600.0))
        return round(q_mah, 6)

    @staticmethod
    def unroll_capacity_resets(capacity_series: List[float]) -> List[float]:
        """
        Unroll a capacity column that resets to zero at step/cycle boundaries
        into continuous monotonic throughput (ampworks _ah_wh_throughput algorithm).
        Handles both resetting cyclers (Landt/BioLogic) and pre-accumulated cyclers (Arbin).
        """
        if not capacity_series:
            return []

        arr = np.array(capacity_series, dtype=float)
        if len(arr) == 1:
            return [max(0.0, float(arr[0]))]

        diffs = np.diff(arr)
        # Discard negative drops (resets back to zero) and accumulate only forward increments
        pos_diffs = np.maximum(diffs, 0.0)
        cum = np.insert(np.cumsum(pos_diffs), 0, 0.0)
        return cum.tolist()

    @classmethod
    def compute_step_capacity_delta(
        cls,
        time_s: List[float],
        current_ma: List[float],
        raw_capacity: Optional[List[float]] = None,
        expected_polarity: Optional[int] = None,
    ) -> Tuple[float, str]:
        """
        Compute true capacity delta ΔQ for a step or half-cycle using first-principles.
        
        Methodology inspired by ampworks (_ah_wh) and cellpy (cap_mod_normal):
        1. If raw_capacity is provided, finite, and monotonic non-decreasing over the step
           (diffs >= -1e-5), compute ΔQ = Q_end - Q_start.
        2. If raw_capacity is missing, corrupt, non-monotonic, or zero while current flows,
           fall back to numerical trapezoidal integration: ∫ |I| dt / 3600 (mAh).
           
        Returns:
            Tuple[float, str]: (delta_q_mah, method) where method is 'delta' or 'trapezoidal_integral'.
        """
        if len(time_s) < 2 or len(current_ma) < 2:
            return 0.0, "insufficient_data"

        # Check if raw_capacity is valid and monotonic
        if raw_capacity is not None and len(raw_capacity) == len(time_s):
            q_arr = np.array(raw_capacity, dtype=float)
            if np.all(np.isfinite(q_arr)):
                diffs = np.diff(q_arr)
                # Allow tiny floating-point noise (-1e-5)
                is_monotonic = bool(np.all(diffs >= -1e-5))
                delta_q = float(q_arr[-1] - q_arr[0])

                # Check if current is actively flowing (> 0.005 mA)
                i_arr = np.array(current_ma, dtype=float)
                active_current = bool(np.mean(np.abs(i_arr)) > 0.005)

                if is_monotonic and delta_q > 1e-6:
                    return round(delta_q, 6), "delta"
                elif not active_current:
                    # Current was zero (rest step)
                    return 0.0, "zero_current"

        # Fallback to numerical trapezoidal integration
        fallback_q = cls.integrate_segment_trapezoid(time_s, current_ma)
        return fallback_q, "trapezoidal_integral"

    @staticmethod
    def smooth_signal(y: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Apply uniform moving average smoothing filter."""
        if window_size <= 1:
            return y.copy()

        kernel = np.ones(window_size) / window_size
        pad_size = window_size // 2
        padded = np.pad(y, pad_size, mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        if len(smoothed) > len(y):
            smoothed = smoothed[:len(y)]
        return smoothed

    @classmethod
    def savitzky_golay_filter(
        cls,
        y: Union[List[float], np.ndarray],
        window_length: int = 15,
        polyorder: int = 3,
        deriv: int = 0,
        delta: float = 1.0,
    ) -> np.ndarray:
        """
        Pure NumPy Savitzky-Golay 1D filtering.
        Fits local polynomials of degree `polyorder` via linear least squares.
        Inspired by cellpy.ica (MIT License) and ampworks.dqdv (BSD 3-Clause License).
        """
        arr = np.array(y, dtype=float)
        if len(arr) < 3:
            return arr.copy()

        # Window length must be odd and at least polyorder + 2
        if window_length % 2 == 0:
            window_length += 1
        if window_length > len(arr):
            window_length = len(arr) if len(arr) % 2 != 0 else len(arr) - 1
        if window_length <= polyorder:
            polyorder = max(1, window_length - 1)
        if window_length < 3:
            return arr.copy()

        m = (window_length - 1) // 2
        x = np.arange(-m, m + 1, dtype=float)
        # Vandermonde matrix: cols are x^0, x^1, ..., x^polyorder
        a_mat = x[:, None] ** np.arange(polyorder + 1)
        pinv = np.linalg.pinv(a_mat)
        weights = (pinv[deriv] * math.factorial(deriv)) / (delta ** deriv)

        # Reflection boundary padding
        pad_left = arr[0] - np.abs(arr[1:m+1][::-1] - arr[0]) if len(arr) > m else np.full(m, arr[0])
        pad_right = arr[-1] + np.abs(arr[-m-1:-1][::-1] - arr[-1]) if len(arr) > m else np.full(m, arr[-1])
        padded = np.concatenate([pad_left, arr, pad_right])
        smoothed = np.convolve(padded, weights[::-1], mode="valid")
        return smoothed

    @classmethod
    def compute_dq_dv_savgol(
        cls,
        voltage_v: List[float],
        capacity_mah: List[float],
        window_length: int = 15,
        polyorder: int = 3,
        min_delta_v: float = 0.001,
    ) -> List[float]:
        """
        Calculate differential capacity dQ/dV with Savitzky-Golay smoothing.
        Inspired by cellpy.ica and ampworks.dqdv conventions.
        Pre-smoothes V and Q to suppress high-frequency digitization jitter, computes
        dQ/dV derivative, and applies Savitzky-Golay post-smoothing to reveal electrochemical
        phase transition peaks and degradation features.
        """
        if len(voltage_v) < 3 or len(voltage_v) != len(capacity_mah):
            return [0.0] * len(voltage_v)

        v = np.array(voltage_v, dtype=float)
        q = np.array(capacity_mah, dtype=float)

        # Pre-smooth voltage and capacity series
        v_smooth = cls.savitzky_golay_filter(v, window_length=window_length, polyorder=polyorder)
        q_smooth = cls.savitzky_golay_filter(q, window_length=window_length, polyorder=polyorder)

        dv = np.gradient(v_smooth)
        dq = np.gradient(q_smooth)

        # Guard against division by zero and maintain correct directional sign
        safe_dv = np.where(
            np.abs(dv) < min_delta_v,
            np.where(dv >= 0, min_delta_v, -min_delta_v),
            dv,
        )

        raw_dq_dv = dq / safe_dv
        # Post-smooth dQ/dV
        dq_dv_smooth = cls.savitzky_golay_filter(raw_dq_dv, window_length=window_length, polyorder=polyorder)

        return dq_dv_smooth.tolist()

    @classmethod
    def compute_dq_dv(
        cls,
        voltage_v: List[float],
        capacity_mah: List[float],
        window_size: int = 5,
        min_delta_v: float = 0.001,
        filter_type: str = "moving_average",
        polyorder: int = 3,
        window_length: Optional[int] = None,
    ) -> List[float]:
        """Calculate differential capacity dQ/dV with noise suppression and smoothing."""
        effective_window = window_length if window_length is not None else window_size
        if filter_type.lower() in ("savgol", "savitzky_golay"):
            return cls.compute_dq_dv_savgol(
                voltage_v=voltage_v,
                capacity_mah=capacity_mah,
                window_length=effective_window,
                polyorder=polyorder,
                min_delta_v=min_delta_v,
            )

        if len(voltage_v) < 3 or len(voltage_v) != len(capacity_mah):
            return [0.0] * len(voltage_v)

        v = np.array(voltage_v, dtype=float)
        q = np.array(capacity_mah, dtype=float)

        v_smooth = cls.smooth_signal(v, window_size=window_size)
        q_smooth = cls.smooth_signal(q, window_size=window_size)

        dv = np.gradient(v_smooth)
        dq = np.gradient(q_smooth)

        # Guard against division by zero and maintain correct directional sign when dv == 0
        safe_dv = np.where(np.abs(dv) < min_delta_v, 
                           np.where(dv >= 0, min_delta_v, -min_delta_v), 
                           dv)

        dq_dv = dq / safe_dv
        dq_dv_smoothed = cls.smooth_signal(dq_dv, window_size=window_size)

        return dq_dv_smoothed.tolist()
