"""Custom exceptions and warnings for the Materials Acceleration Platform (MAP)."""


class OversaturatedFormulationWarning(UserWarning):
    """Warning raised when required stock component volumes exceed target formulation volume."""
    pass

