"""
Multi-vendor cycler and potentiostat ingestion parsers for aqueous Zinc-Ion Batteries.
Supports ASCII (CSV/TXT/MPT/DAT), Excel (XLSX), and BioLogic (MPR) formats.
"""

from pathlib import Path
from typing import Union, List

from models import CyclerRawData
from analysis.parsers.base import BaseCyclerParser, match_canonical_channel, CHANNEL_ALIASES
from analysis.parsers.ascii_parser import AsciiCyclerParser
from analysis.parsers.excel_parser import ExcelCyclerParser
from analysis.parsers.mpr_parser import MprCyclerParser

from analysis.parsers.template_generators import (
    generate_template_biologic_mpt,
    generate_template_landt_csv,
    generate_template_arbin_xlsx,
)

PARSER_REGISTRY: List[BaseCyclerParser] = [
    MprCyclerParser(),
    ExcelCyclerParser(),
    AsciiCyclerParser(),
]


def parse_cycler_file(filepath: Union[str, Path]) -> CyclerRawData:
    """Auto-detect cycler file vendor format and return standardized CyclerRawData."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    for parser in PARSER_REGISTRY:
        if parser.can_parse(path):
            return parser.parse(path)

    raise ValueError(f"Unsupported cycler file format: No suitable parser found for {filepath}")


__all__ = [
    "BaseCyclerParser",
    "AsciiCyclerParser",
    "ExcelCyclerParser",
    "MprCyclerParser",
    "parse_cycler_file",
    "match_canonical_channel",
    "CHANNEL_ALIASES",
    "generate_template_biologic_mpt",
    "generate_template_landt_csv",
    "generate_template_arbin_xlsx",
]
