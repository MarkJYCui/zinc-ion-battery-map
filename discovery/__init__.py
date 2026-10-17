"""
Phase 1: Literature Discovery & Extraction Engine for Zinc-Ion Batteries.
Modular package export.
"""

from .search import SemanticScholarClient, clean_scientific_text
from .extractor import PaperExtractor
from .similarity import (
    RDKitSimilarityEngine,
    CURATED_ZIB_ADDITIVES,
    calculate_delaney_esol,
    calculate_esol,
    get_aromatic_proportion,
)
from .literature_engine import LiteratureDiscoveryEngine, clean_llm_json_output

__all__ = [
    "SemanticScholarClient",
    "clean_scientific_text",
    "PaperExtractor",
    "RDKitSimilarityEngine",
    "CURATED_ZIB_ADDITIVES",
    "calculate_delaney_esol",
    "calculate_esol",
    "get_aromatic_proportion",
    "LiteratureDiscoveryEngine",
    "clean_llm_json_output",
]

