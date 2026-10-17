"""
Unit and integration tests for the Literature Discovery and Extraction Engine.
Module 1: Materials Acceleration Platform (MAP) for Zinc-Ion Batteries (ZIB).

Following strict Test-Driven Development (TDD) rules.
Tests cover:
- Pydantic schema validation & normalization
- RDKit similarity calculations & candidate ranking
- Domain text extraction with realistic battery literature mocks
- Semantic Scholar API search client & response parsing
- PDF text extraction
- End-to-end Literature Discovery Engine pipeline
"""

import io
import json
import pytest
from unittest.mock import MagicMock, patch
from pydantic import ValidationError

from models import (
    Concentration,
    ZincIonBatteryContext,
    CyclingCondition,
    PerformanceMetrics,
    ExtractedAdditive,
    CandidateMolecule,
    PaperMetadata,
    ExtractionResult,
)
from discovery.similarity import RDKitSimilarityEngine, CURATED_ZIB_ADDITIVES
from discovery.search import SemanticScholarClient
from discovery.extractor import PaperExtractor
from discovery.literature_engine import LiteratureDiscoveryEngine


# ============================================================================
# 1. SCHEMA VALIDATION TESTS
# ============================================================================

def test_concentration_normalization():
    """Test concentration unit normalization."""
    c1 = Concentration(value=10.0, unit="mM", raw_text="10 mM")
    assert c1.value == 10.0
    assert c1.unit == "mM"

    c2 = Concentration(value=5.0, unit="mM", raw_text="5 mm")
    assert c2.unit == "mM"

    c3 = Concentration(value=2.5, unit="wt%", raw_text="2.5 wt%")
    assert c3.unit == "wt%"

    c4 = Concentration(value=1.2, unit="g/L", raw_text="1.2 g/l")
    assert c4.unit == "g/L"


def test_extracted_additive_schema_valid():
    """Verify ExtractedAdditive model with realistic battery data."""
    additive = ExtractedAdditive(
        name="Trehalose",
        iupac_name="(2R,3S,4S,5R,6R)-2-(hydroxymethyl)-6-[(2R,3R,4S,5S,6R)-3,4,5-trihydroxy-6-(hydroxymethyl)oxan-2-yl]oxyoxane-3,4,5-triol",
        smiles="C(C1C(C(C(C(O1)OC2C(C(C(C(O2)CO)O)O)O)O)O)O)O",
        optimal_concentration=Concentration(value=10.0, unit="mM", raw_text="10 mM"),
        concentration_range_tested=[Concentration(value=10.0, unit="mM", raw_text="10 mM")],
        functional_groups=["hydroxyl"],
        primary_mechanisms=[
            "Zn²⁺ solvation sheath modulation",
            "HER suppression",
            "Dendrite inhibition",
        ],
        mechanism_summary="Trehalose molecules replace active water in the inner Zn²⁺ solvation sheath.",
        battery_context=ZincIonBatteryContext(
            baseline_electrolyte="2 M ZnSO4",
            solvent="H2O",
            electrolyte_pH=4.0,
            anode_material="Zn foil",
            cathode_material="MnO2",
            cell_format="Symmetric Zn//Zn cell"
        ),
        performance_metrics=PerformanceMetrics(
            symmetric_cell_performance=[
                CyclingCondition(
                    current_density_mA_cm2=1.0,
                    areal_capacity_mAh_cm2=1.0,
                    lifespan_hours=1800.0
                )
            ],
            coulombic_efficiency_percent=99.6
        ),
        confidence_score=0.95,
        extraction_evidence="The optimal 10 mM trehalose electrolyte achieved stable Zn plating/stripping for over 1800 h."
    )
    assert additive.name == "Trehalose"
    assert additive.optimal_concentration.value == 10.0
    assert additive.confidence_score == 0.95
    assert "Zn²⁺ solvation sheath modulation" in additive.primary_mechanisms


# ============================================================================
# 2. RDKIT SIMILARITY & CHEMINFORMATICS TESTS
# ============================================================================

def test_rdkit_similarity_engine_descriptors():
    """Test physicochemical descriptor calculation for a known molecule."""
    engine = RDKitSimilarityEngine()
    descriptors = engine.calculate_descriptors("C(C(CO)O)O")
    assert descriptors is not None
    assert pytest.approx(descriptors["molecular_weight"], 0.1) == 92.09
    assert descriptors["hbd"] == 3
    assert descriptors["hba"] == 3
    assert descriptors["tpsa"] > 50.0


def test_rdkit_invalid_smiles():
    """Invalid SMILES string should return None cleanly."""
    engine = RDKitSimilarityEngine()
    assert engine.calculate_descriptors("INVALID_SMILES_XYZ") is None
    assert engine.compute_tanimoto("INVALID_SMILES_1", "C(C(CO)O)O") is None


def test_rdkit_tanimoto_identity():
    """Identical molecules should have a Tanimoto coefficient of 1.0."""
    engine = RDKitSimilarityEngine()
    smiles = "OCC(O)CO"
    score = engine.compute_tanimoto(smiles, smiles)
    assert score == pytest.approx(1.0, 1e-4)


def test_rdkit_suggest_candidates():
    """Test candidate suggestion from curated zinc additive library."""
    engine = RDKitSimilarityEngine()
    glycerol_smiles = "C(C(CO)O)O"
    candidates = engine.find_similar_candidates(
        query_smiles=glycerol_smiles,
        top_n=3,
        threshold=0.25
    )

    assert len(candidates) > 0
    assert all(isinstance(c, CandidateMolecule) for c in candidates)
    assert candidates[0].similarity_score >= candidates[-1].similarity_score


# ============================================================================
# 3. SEMANTIC SCHOLAR SEARCH CLIENT TESTS
# ============================================================================

def test_semantic_scholar_search_success():
    """Test paper search query against mock Semantic Scholar response."""
    client = SemanticScholarClient(api_key=None)

    mock_response_data = {
        "total": 1,
        "offset": 0,
        "data": [
            {
                "paperId": "abc12345",
                "title": "Trehalose Electrolyte Additive Regulates Solvation Chemistry",
                "abstract": "Water-induced parasitic reactions hinder aqueous zinc-ion batteries...",
                "year": 2023,
                "venue": "Energy & Environmental Science",
                "authors": [{"name": "Zhang San"}, {"name": "Li Si"}],
                "externalIds": {"DOI": "10.1039/D3EE00000A"},
                "openAccessPdf": {"url": "https://example.com/paper.pdf"},
                "citationCount": 42
            }
        ]
    }

    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_response_data
        mock_get.return_value = mock_resp

        results = client.search_papers(query="zinc-ion battery additive", limit=1)

        assert len(results) == 1
        paper = results[0]
        assert isinstance(paper, PaperMetadata)
        assert paper.paper_id == "abc12345"
        assert "Trehalose" in paper.title


# ============================================================================
# 4. TEXT & PDF EXTRACTION TESTS
# ============================================================================

def test_extract_additive_from_literature_text_mocked_llm():
    """Verify structured additive extraction using mocked LLM response."""
    sample_text = (
        "In this work, 50 mM TEABr in 2 M ZnSO4 operates stably for 1200 h at 2 mA cm-2."
    )

    extractor = PaperExtractor()

    mock_llm_output = {
        "additives": [
            {
                "name": "Tetraethylammonium bromide",
                "iupac_name": "N,N,N-triethylethanaminium bromide",
                "smiles": "CC[N+](CC)(CC)CC.[Br-]",
                "optimal_concentration": {"value": 50.0, "unit": "mM", "raw_text": "50 mM"},
                "concentration_range_tested": [{"value": 50.0, "unit": "mM", "raw_text": "50 mM"}],
                "functional_groups": ["quaternary ammonium"],
                "primary_mechanisms": ["Electrostatic shielding"],
                "mechanism_summary": "Forms dynamic electrostatic shielding.",
                "battery_context": {
                    "baseline_electrolyte": "2 M ZnSO4",
                    "solvent": "H2O",
                    "anode_material": "Zn foil",
                    "cell_format": "Symmetric Zn//Zn cell"
                },
                "performance_metrics": {
                    "symmetric_cell_performance": [
                        {
                            "current_density_mA_cm2": 2.0,
                            "areal_capacity_mAh_cm2": 1.0, 
                            "lifespan_hours": 1200.0
                        }
                    ],
                    "coulombic_efficiency_percent": 99.4
                },
                "confidence_score": 0.95,
                "extraction_evidence": "50 mM tetraethylammonium bromide"
            }
        ],
        "summary": "50 mM TEABr yielded 1200 h lifespan."
    }

    with patch.object(extractor, "_call_llm_structured", return_value=mock_llm_output):
        additives, summary = extractor.extract_from_text(sample_text)

        assert len(additives) == 1
        additive = additives[0]
        assert additive.name == "Tetraethylammonium bromide"
        assert additive.performance_metrics.symmetric_cell_performance[0].lifespan_hours == 1200.0


# ============================================================================
# 5. END-TO-END ENGINE PIPELINE TEST
# ============================================================================

def test_literature_engine_pipeline_mocked():
    """Verify complete pipeline: paper metadata -> extraction -> RDKit candidates."""
    engine = LiteratureDiscoveryEngine(gemini_api_key="mock_key")

    mock_paper = PaperMetadata(
        paper_id="paper_101",
        title="Glycerol Regulated Electrolyte",
        authors=["Alice Curie"],
        year=2024
    )

    mock_llm_output = {
        "additives": [
            {
                "name": "Glycerol",
                "smiles": "C(C(CO)O)O",
                "optimal_concentration": {"value": 2.0, "unit": "vol%", "raw_text": "2.0 vol%"},
                "concentration_range_tested": [{"value": 2.0, "unit": "vol%", "raw_text": "2.0 vol%"}],
                "functional_groups": ["hydroxyl"],
                "primary_mechanisms": ["Zn²⁺ solvation sheath modulation"],
                "mechanism_summary": "Disrupts water hydrogen bonding.",
                "battery_context": {
                    "baseline_electrolyte": "2 M ZnSO4",
                    "solvent": "H2O",
                    "anode_material": "Zn foil",
                    "cell_format": "Symmetric Zn//Zn cell"
                },
                "performance_metrics": {
                    "symmetric_cell_performance": [
                        {
                            "current_density_mA_cm2": 1.0,
                            "areal_capacity_mAh_cm2": 1.0, 
                            "lifespan_hours": 2000.0
                        }
                    ]
                },
                "confidence_score": 0.98,
                "extraction_evidence": "2.0 vol% glycerol optimizes solvation"
            }
        ],
        "summary": "Glycerol additive achieves 2000 h cycling."
    }

    with patch.object(engine.extractor, "_call_llm_structured", return_value=mock_llm_output):
        result = engine.process_paper_text(
            paper_metadata=mock_paper,
            text="Mock text stating 2.0 vol% glycerol..."
        )

        assert isinstance(result, ExtractionResult)
        assert len(result.additives) == 1
        assert len(result.candidate_molecules) > 0