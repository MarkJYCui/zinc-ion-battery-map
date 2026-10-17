"""Formulation utility functions for liquid handling and volumetric balance."""
from typing import Sequence, Any, Union
import warnings
from core.exceptions import OversaturatedFormulationWarning


class Component:
    """Representation of a liquid stock component."""
    def __init__(self, volume: float, name: str = ""):
        self.volume = float(volume)
        self.name = name


def calculate_diluent_volume(stock_components: Sequence[Any], target_vol: float) -> float:
    """
    Calculate required diluent volume given stock component volumes and target total volume.

    Args:
        stock_components: Sequence of components with a .volume attribute (or numeric volumes).
        target_vol: Target total volume (e.g. in uL).

    Returns:
        Required diluent volume in uL, clamped to 0.0 if stock exceeds target volume.
    """
    total_stock_vol = sum(getattr(component, "volume", component) for component in stock_components)
    diluent_required = target_vol - total_stock_vol

    # Volumetric Balance & Negative Diluent Safeguard
    if diluent_required < 0.0:
        import warnings
        from core.exceptions import OversaturatedFormulationWarning

        warnings.warn(
            f"Target volume ({target_vol} uL) exceeded by stock components ({total_stock_vol} uL). "
            f"Clamping diluent volume to 0.0.", 
            OversaturatedFormulationWarning,
            stacklevel=2,
        )
        diluent_required = 0.0

    return round(diluent_required, 3)

