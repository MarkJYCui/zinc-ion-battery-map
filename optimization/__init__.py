"""
Phase 2: Autonomous Design of Experiments (DoE) & Electrolyte-Aware Robotic Formulation Engine.
Materials Acceleration Platform (MAP) for Zinc-Ion Batteries (ZIB).

Exports:
    - ConcentrationAnchorEngine: Dynamically shifts concentration anchors across electrolyte chemistries.
    - CANONICAL_ELECTROLYTE_PROFILES: Standard operating parameters for all 6 major aqueous ZIB salts.
    - ELECTROLYTE_SCALING_RULES: Mechanistic scaling coefficients and rationales.
    - DoEEngine: Multi-factor sampling engine supporting LHS and CCD.
    - RoboticDispenserEngine: Generates physical pipetting schedules for robotic liquid handlers.
    - run_planner: Main orchestrator execution entry point.
"""

from .concentration_anchor import (
    ConcentrationAnchorEngine,
    CANONICAL_ELECTROLYTE_PROFILES,
    ELECTROLYTE_SCALING_RULES,
)
from .doe_engine import DoEEngine
from .dispenser import RoboticDispenserEngine


def __getattr__(name: str):
    if name == "run_planner":
        from .planner_cli import run_planner
        return run_planner
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "ConcentrationAnchorEngine",
    "CANONICAL_ELECTROLYTE_PROFILES",
    "ELECTROLYTE_SCALING_RULES",
    "DoEEngine",
    "RoboticDispenserEngine",
    "run_planner",
]

