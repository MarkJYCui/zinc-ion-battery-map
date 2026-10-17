"""
Autonomous Electrochemical Characterization, Parsing & Analysis Engine.
Phase 3 of the Materials Acceleration Platform (MAP) for aqueous Zinc-Ion Batteries (ZIB).
"""

from .parsers import (
    BaseCyclerParser,
    AsciiCyclerParser,
    ExcelCyclerParser,
    MprCyclerParser,
    parse_cycler_file,
    generate_template_biologic_mpt,
    generate_template_landt_csv,
    generate_template_arbin_xlsx,
)
from .processing import (
    MathEngine,
    TimeScaler,
    RestDetector,
    ElectrochemicalNormalizer,
)
from .techniques import (
    SymmetricCellAnalyzer,
    AsymmetricCellAnalyzer,
    FullCellAnalyzer,
    LSVAnalyzer,
    EISAnalyzer,
    CVAnalyzer,
)
from .visualization import ElectrochemicalPlotEngine
from .ingestion import CyclerWatchdog


def __getattr__(name: str):
    if name == "run_analysis":
        from .analyzer_cli import run_analysis
        return run_analysis
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "BaseCyclerParser",
    "AsciiCyclerParser",
    "ExcelCyclerParser",
    "MprCyclerParser",
    "parse_cycler_file",
    "MathEngine",
    "TimeScaler",
    "RestDetector",
    "ElectrochemicalNormalizer",
    "SymmetricCellAnalyzer",
    "AsymmetricCellAnalyzer",
    "FullCellAnalyzer",
    "LSVAnalyzer",
    "EISAnalyzer",
    "CVAnalyzer",
    "ElectrochemicalPlotEngine",
    "CyclerWatchdog",
    "run_analysis",
    "generate_template_biologic_mpt",
    "generate_template_landt_csv",
    "generate_template_arbin_xlsx",
]
