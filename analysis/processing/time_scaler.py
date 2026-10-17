"""
Adaptive time scaling for electrochemical time series.
"""

from typing import List, Tuple, Optional
import numpy as np


class TimeScaler:
    """Scales raw seconds to human-readable units for visualization and reporting."""

    @staticmethod
    def scale_time_series(time_s: List[float], force_unit: Optional[str] = None) -> Tuple[List[float], str]:
        """Convert raw SI seconds into adaptive display time array and unit label."""
        if not time_s:
            return [], 's'

        max_t = max(time_s) if time_s else 0.0

        if force_unit:
            unit = force_unit.lower()
            if unit in ('h', 'hr', 'hours'):
                scale = 3600.0
                selected_unit = 'h'
            elif unit in ('min', 'm', 'minutes'):
                scale = 60.0
                selected_unit = 'min'
            elif unit in ('d', 'day', 'days'):
                scale = 86400.0
                selected_unit = 'd'
            else:
                scale = 1.0
                selected_unit = 's'
        else:
            if max_t < 60.0:
                scale = 1.0
                selected_unit = 's'
            elif max_t < 3600.0:
                scale = 60.0
                selected_unit = 'min'
            else:
                scale = 3600.0
                selected_unit = 'h'

        scaled = [round(t / scale, 4) for t in time_s]
        return scaled, selected_unit

    @classmethod
    def auto_scale(cls, max_t: float) -> Tuple[float, str]:
        scaled, unit = cls.scale_time_series([max_t])
        return scaled[0] if scaled else 0.0, unit

    @classmethod
    def scale_series(cls, time_s: List[float], target_unit: Optional[str] = None) -> List[float]:
        scaled, _ = cls.scale_time_series(time_s, force_unit=target_unit)
        return scaled

