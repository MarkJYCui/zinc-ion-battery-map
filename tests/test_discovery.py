"""
Unit and integration tests for the Literature Discovery and Extraction Engine.
Module 1: Materials Acceleration Platform (MAP) for Zinc-Ion Batteries (ZIB).

Following strict Test-Driven Development (TDD) rules.
Tests cover:
- Pydantic schema validation & normalization
- CandidateMolecule baseline electrolyte & compatibility guidance
- RDKit similarity calculations & candidate ranking
- Electrolyte-aware electrochemical compatibility reasoning
- Semantic Scholar API search client & OpenAlex 429 fallback
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
    ElectrolyteCompatibilityDetail,
)
from discovery.similarity import (
    RDKitSimilarityEngine, 
    CURATED_ZIB_ADDITIVES, 
    calculate_delaney_esol,
    assess_electrolyte_compatibility,
    get_aromatic_proportion,
)
from discovery.search import SemanticScholarClient, clean_scientific_text
from discovery.extractor import PaperExtractor, clean_invalid_unicode, dehyphenate_scientific_text
from discovery.domain_config import ZIBDomainConfig
from discovery.pubchem_client import PubChemClient
from discovery.literature_engine import LiteratureDiscoveryEngine, clean_llm_json_output, parse_llm_response


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


def test_candidate_molecule_with_electrolyte_fields():
    """Verify CandidateMolecule schema includes baseline electrolyte and compatibility fields."""
    cand = CandidateMolecule(
        name="D-Sorbitol",
        smiles="C(C(C(C(C(CO)O)O)O)O)O",
        similarity_score=0.78,
        similarity_metric="Tanimoto (Morgan R=2)",
        molecular_weight=182.17,
        logp=-3.0,
        logs=0.5,
        tpsa=121.38,
        hbd=6,
        hba=6,
        rotatable_bonds=5,
        rationale="Tanimoto similarity of 0.78 to target.",
        sdl_formulation_notes="Aqueous solubility limits optimal for automated liquid handling.",
        parent_baseline_electrolyte="2 M ZnSO4",
        electrolyte_compatibility_note=(
            "Validated for sulfate chemistry; verify compatibility before using in other electrolytes; "
            "theoretically compatible with Zn(OTf)2 due to neutral non-ionic solvation sheath modulation."
        )
    )
    assert cand.parent_baseline_electrolyte == "2 M ZnSO4"
    assert "theoretically compatible with Zn(OTf)2" in cand.electrolyte_compatibility_note


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
# 3. ELECTROCHEMICAL COMPATIBILITY REASONING TESTS
# ============================================================================

def test_polyol_compatibility_in_znso4():
    """Polyol additive validated in ZnSO4 should note constructive theoretical compatibility with Zn(OTf)2."""
    engine = RDKitSimilarityEngine()
    # Query with Glycerol (polyol)
    candidates = engine.find_similar_candidates(
        query_smiles="OCC(O)CO",
        top_n=3,
        threshold=0.20,
        baseline_electrolyte="2 M ZnSO4"
    )
    assert len(candidates) > 0
    # D-Sorbitol or D-Mannitol or Ethylene Glycol should be recommended
    sorbitol_cand = next((c for c in candidates if "Sorbitol" in c.name or "Mannitol" in c.name or "Glycol" in c.name), candidates[0])
    assert sorbitol_cand.parent_baseline_electrolyte == "2 M ZnSO4"
    assert "Validated for sulfate chemistry" in sorbitol_cand.electrolyte_compatibility_note
    assert "verify compatibility before using in other electrolytes" in sorbitol_cand.electrolyte_compatibility_note
    assert "theoretically compatible with Zn(OTf)2" in sorbitol_cand.electrolyte_compatibility_note


def test_amide_compatibility_in_znso4():
    """Amide/urea additives in ZnSO4 should note theoretical compatibility with Zn(OTf)2 and Zn(ClO4)2."""
    engine = RDKitSimilarityEngine()
    candidates = engine.find_similar_candidates(
        query_smiles="C(=O)(N)N",  # Urea
        top_n=3,
        threshold=0.10,
        baseline_electrolyte="3 M ZnSO4"
    )
    assert len(candidates) > 0
    assert all(c.parent_baseline_electrolyte == "3 M ZnSO4" for c in candidates)
    # Check that compatibility notes are generated
    for c in candidates:
        assert "Validated for sulfate chemistry" in c.electrolyte_compatibility_note


def test_triflate_parent_electrolyte_compatibility():
    """When parent electrolyte is Zn(OTf)2, note should guide validation in sulfate/BZS suppression."""
    engine = RDKitSimilarityEngine()
    candidates = engine.find_similar_candidates(
        query_smiles="OCC(O)CO",
        top_n=2,
        threshold=0.20,
        baseline_electrolyte="1 M Zn(OTf)2"
    )
    assert len(candidates) > 0
    assert candidates[0].parent_baseline_electrolyte == "1 M Zn(OTf)2"
    assert "Validated for triflate chemistry" in candidates[0].electrolyte_compatibility_note
    assert "verify compatibility before using in other electrolytes" in candidates[0].electrolyte_compatibility_note


def test_unspecified_parent_electrolyte():
    """When no parent electrolyte is given, should provide graceful fallback."""
    engine = RDKitSimilarityEngine()
    candidates = engine.find_similar_candidates(
        query_smiles="OCC(O)CO",
        top_n=2,
        threshold=0.20,
        baseline_electrolyte=None
    )
    assert len(candidates) > 0
    assert candidates[0].parent_baseline_electrolyte is None
    assert "verify baseline salt compatibility" in candidates[0].electrolyte_compatibility_note.lower()


def test_electrolyte_compatibility_detail_schema():
    """Verify ElectrolyteCompatibilityDetail schema structure."""
    detail = ElectrolyteCompatibilityDetail(
        salt="Zn(OTf)2",
        status="Highly Compatible",
        expected_effectiveness="High",
        notes="Forms fluorinated ZnF2 interphase synergistically."
    )
    assert detail.salt == "Zn(OTf)2"
    assert detail.status == "Highly Compatible"
    assert detail.expected_effectiveness == "High"


def test_polyol_all_six_electrolytes_breakdown():
    """Test full 6-salt compatibility breakdown for polyol candidate (e.g. Sorbitol/Glycerol)."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles("OCC(O)CO")  # Glycerol
    note, breakdown = assess_electrolyte_compatibility(mol, baseline_electrolyte="2 M ZnSO4")
    
    assert len(breakdown) == 6
    expected_salts = ["ZnSO4", "Zn(OTf)2", "Zn(ClO4)2", "Zn(TFSI)2", "ZnCl2", "Zn(CH3COO)2"]
    for salt in expected_salts:
        assert salt in breakdown
        assert isinstance(breakdown[salt], ElectrolyteCompatibilityDetail)
        assert breakdown[salt].salt == salt

    # ZnSO4 is the baseline
    assert breakdown["ZnSO4"].status == "Validated"
    assert "solvation sheath" in breakdown["ZnSO4"].notes.lower()
    
    # Zn(OTf)2
    assert breakdown["Zn(OTf)2"].status in ["Highly Compatible", "Compatible"]
    assert "theoretically compatible with zn(otf)2" in note.lower()

    # Zn(ClO4)2
    assert breakdown["Zn(ClO4)2"].status in ["Highly Compatible", "Compatible"]
    assert "desolvation" in breakdown["Zn(ClO4)2"].notes.lower() or "kinetics" in breakdown["Zn(ClO4)2"].notes.lower()

    # Zn(TFSI)2
    assert breakdown["Zn(TFSI)2"].status in ["Highly Compatible", "Compatible"]
    assert "water" in breakdown["Zn(TFSI)2"].notes.lower() or "window" in breakdown["Zn(TFSI)2"].notes.lower()

    # ZnCl2
    assert breakdown["ZnCl2"].status in ["Compatible", "Caution / Conditional"]
    assert "chlorozinc" in breakdown["ZnCl2"].notes.lower() or "complex" in breakdown["ZnCl2"].notes.lower()

    # Zn(CH3COO)2
    assert breakdown["Zn(CH3COO)2"].status in ["Compatible", "Highly Compatible"]
    assert "acetate" in breakdown["Zn(CH3COO)2"].notes.lower() or "buffer" in breakdown["Zn(CH3COO)2"].notes.lower()


def test_amide_urea_all_six_electrolytes_breakdown():
    """Test full 6-salt compatibility breakdown for amide/urea additive."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles("C(=O)(N)N")  # Urea
    note, breakdown = assess_electrolyte_compatibility(mol, baseline_electrolyte="3 M ZnSO4")
    
    assert len(breakdown) == 6
    assert breakdown["ZnSO4"].status == "Validated"
    assert breakdown["ZnCl2"].status in ["Highly Compatible", "Compatible"]
    assert "deep eutectic" in breakdown["ZnCl2"].notes.lower() or "complex" in breakdown["ZnCl2"].notes.lower()
    assert breakdown["Zn(TFSI)2"].status in ["Highly Compatible", "Compatible"]


def test_thiocarbonyl_all_six_electrolytes_breakdown():
    """Test thiocarbonyl (Thiourea) across all 6 chemistries."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles("C(=S)(N)N")  # Thiourea
    note, breakdown = assess_electrolyte_compatibility(mol, baseline_electrolyte="2 M ZnSO4")
    
    assert len(breakdown) == 6
    assert breakdown["ZnSO4"].status == "Validated"
    assert "adsorb" in breakdown["ZnSO4"].notes.lower() or "sulfur" in breakdown["ZnSO4"].notes.lower()
    # In ZnCl2, high Cl- causes competitive adsorption caution
    assert breakdown["ZnCl2"].status == "Caution / Conditional"
    assert "compet" in breakdown["ZnCl2"].notes.lower() or "cl-" in breakdown["ZnCl2"].notes.lower()


def test_quaternary_ammonium_all_six_electrolytes_breakdown():
    """Test quaternary ammonium (TEABr) across all 6 chemistries."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles("CC[N+](CC)(CC)CC")  # TEA+
    note, breakdown = assess_electrolyte_compatibility(mol, baseline_electrolyte="2 M ZnSO4")
    
    assert len(breakdown) == 6
    assert breakdown["ZnSO4"].status == "Validated"
    assert "electrostatic" in breakdown["ZnSO4"].notes.lower()
    # In Zn(TFSI)2, hydrophobic ion-pair precipitation risk is flagged
    assert breakdown["Zn(TFSI)2"].status == "Caution / Conditional"
    assert "precipitat" in breakdown["Zn(TFSI)2"].notes.lower() or "ion pair" in breakdown["Zn(TFSI)2"].notes.lower()
    # In ZnCl2, EDL compaction noted
    assert "double layer" in breakdown["ZnCl2"].notes.lower() or "edl" in breakdown["ZnCl2"].notes.lower() or "ionic strength" in breakdown["ZnCl2"].notes.lower()


def test_sulfoxide_all_six_electrolytes_breakdown():
    """Test sulfoxide (DMSO) across all 6 chemistries."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles("CS(=O)C")  # DMSO
    note, breakdown = assess_electrolyte_compatibility(mol, baseline_electrolyte="2 M ZnSO4")
    
    assert len(breakdown) == 6
    assert breakdown["ZnSO4"].status == "Validated"
    assert breakdown["Zn(TFSI)2"].status == "Highly Compatible"
    assert breakdown["Zn(ClO4)2"].status == "Highly Compatible"
    assert "donor" in breakdown["ZnSO4"].notes.lower() or "solvation" in breakdown["ZnSO4"].notes.lower() or "coordinate" in breakdown["ZnSO4"].notes.lower()


def test_alternative_baseline_electrolytes_validation():
    """Verify that any of the 6 chemistries can serve as the validated baseline."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles("OCC(O)CO")
    
    baselines_and_expected = [
        ("1 M Zn(OTf)2", "Zn(OTf)2"),
        ("2 M Zn(ClO4)2", "Zn(ClO4)2"),
        ("1 M Zn(TFSI)2", "Zn(TFSI)2"),
        ("5 M ZnCl2", "ZnCl2"),
        ("1 M Zn(CH3COO)2", "Zn(CH3COO)2"),
    ]
    
    for baseline, expected_salt in baselines_and_expected:
        note, breakdown = assess_electrolyte_compatibility(mol, baseline_electrolyte=baseline)
        assert breakdown[expected_salt].status == "Validated"


def test_candidate_molecules_populated_with_breakdown():
    """Verify that find_similar_candidates populates electrolyte_compatibility_breakdown."""
    engine = RDKitSimilarityEngine()
    candidates = engine.find_similar_candidates(
        query_smiles="OCC(O)CO",
        top_n=3,
        threshold=0.20,
        baseline_electrolyte="2 M ZnSO4"
    )
    assert len(candidates) > 0
    for cand in candidates:
        assert cand.electrolyte_compatibility_breakdown is not None
        assert len(cand.electrolyte_compatibility_breakdown) == 6
        assert "ZnSO4" in cand.electrolyte_compatibility_breakdown
        assert "Zn(OTf)2" in cand.electrolyte_compatibility_breakdown
        assert cand.electrolyte_compatibility_breakdown["ZnSO4"].status == "Validated"


# ============================================================================
# 4. SEARCH CLIENT & FALLBACK TESTS
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


def test_openalex_fallback_on_429():
    """Test automatic fallback to OpenAlex when Semantic Scholar returns HTTP 429."""
    client = SemanticScholarClient(api_key=None)

    mock_openalex_data = {
        "results": [
            {
                "id": "https://openalex.org/W123456",
                "title": "Sorbitol Regulation of Zinc Anode Interface",
                "abstract_inverted_index": {"Aqueous": [0], "zinc": [1], "batteries": [2]},
                "publication_year": 2024,
                "primary_location": {
                    "source": {"display_name": "Advanced Energy Materials"},
                    "pdf_url": "https://example.com/openalex_paper.pdf"
                },
                "authorships": [{"author": {"display_name": "Wang Wu"}}],
                "doi": "https://doi.org/10.1002/aenm.202400001",
                "cited_by_count": 15
            }
        ]
    }

    with patch("requests.get") as mock_get:
        # First call (Semantic Scholar) returns 429
        resp_429 = MagicMock()
        resp_429.status_code = 429
        # Second call (OpenAlex) returns 200
        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = mock_openalex_data

        mock_get.side_effect = [resp_429, resp_200]

        results = client.search_papers(query="zinc-ion battery additive", limit=1)
        assert len(results) == 1
        assert "Sorbitol" in results[0].title


# ============================================================================
# 5. TEXT & PDF EXTRACTION TESTS
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
        assert additive.battery_context.baseline_electrolyte == "2 M ZnSO4"
        assert additive.performance_metrics.symmetric_cell_performance[0].lifespan_hours == 1200.0


# ============================================================================
# 6. END-TO-END PIPELINE WITH ELECTROLYTE PROPAGATION
# ============================================================================

def test_literature_engine_pipeline_with_electrolyte_compatibility():
    """Verify complete pipeline passes baseline electrolyte into candidate recommendations."""
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
        # All candidate molecules must carry parent baseline electrolyte and compatibility note
        for cand in result.candidate_molecules:
            assert cand.parent_baseline_electrolyte == "2 M ZnSO4"
            assert "Validated for sulfate chemistry" in cand.electrolyte_compatibility_note
            assert "verify compatibility before using in other electrolytes" in cand.electrolyte_compatibility_note


def test_clean_llm_json_output_markdown_fences():
    """Verify clean_llm_json_output safely strips markdown fences from LLM responses."""
    # Test case 1: ```json ... ```
    raw1 = "```json\n{\"additives\": [{\"name\": \"D-Sorbitol\"}], \"summary\": \"Test\"}\n```"
    cleaned1 = clean_llm_json_output(raw1)
    assert not cleaned1.startswith("```")
    assert not cleaned1.endswith("```")
    parsed1 = parse_llm_response(raw1)
    assert parsed1["additives"][0]["name"] == "D-Sorbitol"

    # Test case 2: ``` ... ``` without json tag and with surrounding whitespace
    raw2 = "   ```\n{\"additives\": [], \"summary\": \"Clean\"}\n```   "
    cleaned2 = clean_llm_json_output(raw2)
    assert cleaned2 == '{"additives": [], "summary": "Clean"}'

    # Test case 3: Raw JSON without markdown fences
    raw3 = '{"additives": [], "summary": "Plain"}'
    assert clean_llm_json_output(raw3) == raw3

    # Test case 4: PaperExtractor._call_llm_structured with markdown fences
    extractor = PaperExtractor()
    res = extractor._call_llm_structured(raw1)
    assert res["additives"][0]["name"] == "D-Sorbitol"


def test_get_aromatic_proportion_zero_division():
    """Verify get_aromatic_proportion handles None and 0 heavy atoms without ZeroDivisionError."""
    from rdkit import Chem
    # None mol
    assert get_aromatic_proportion(None) == 0.0

    # 0 heavy atoms (proton or H2)
    mol_h = Chem.MolFromSmiles("[H+]")
    assert get_aromatic_proportion(mol_h) == 0.0

    # Benzene (all aromatic heavy atoms)
    mol_b = Chem.MolFromSmiles("c1ccccc1")
    assert get_aromatic_proportion(mol_b) == 1.0

    # Calculate Delaney ESOL safely for 0 heavy atom molecules
    logs = calculate_delaney_esol(mol_h)
    assert isinstance(logs, float)


def test_semantic_scholar_search_extended_offline_papers_and_query_ranking():
    """Verify SemanticScholarClient returns extended offline sample papers with keyword ranking."""
    client = SemanticScholarClient(api_key=None)
    
    # Query matching trehalose specifically
    papers_trehalose = client.search_papers(query="trehalose solvation sheath", limit=3)
    assert len(papers_trehalose) >= 1
    assert "Trehalose" in papers_trehalose[0].title
    assert papers_trehalose[0].doi is not None
    assert papers_trehalose[0].abstract is not None

    # Query matching thiourea specifically
    papers_thiourea = client.search_papers(query="thiourea interfacial adsorption", limit=3)
    assert len(papers_thiourea) >= 1
    assert "Thiourea" in papers_thiourea[0].title

    # Query matching glucose specifically
    papers_glucose = client.search_papers(query="glucose additive kinetics", limit=3)
    assert len(papers_glucose) >= 1
    assert "Glucose" in papers_glucose[0].title


def test_heuristic_extraction_all_curated_additives_and_regex_patterns():
    """Verify PaperExtractor heuristic extractor extracts additives, concentrations, and baseline salts."""
    extractor = PaperExtractor()

    # Test Glucose extraction with custom concentration
    text_glucose = "In this work, 25 mM glucose in 2 M ZnSO4 delivers 1800 h of stable cycling."
    additives_g, _ = extractor.extract_from_text(text_glucose)
    assert len(additives_g) == 1
    assert additives_g[0].name == "Glucose"
    assert additives_g[0].optimal_concentration.value == 25.0
    assert "Zn²⁺ solvation sheath modulation" in additives_g[0].primary_mechanisms

    # Test Thiourea extraction with custom concentration
    text_thiourea = "We added 12 mM thiourea into 2 M ZnSO4 electrolyte."
    additives_t, _ = extractor.extract_from_text(text_thiourea)
    assert len(additives_t) == 1
    assert additives_t[0].name == "Thiourea"
    assert additives_t[0].optimal_concentration.value == 12.0


def test_clean_scientific_text_comprehensive():
    """Verify clean_scientific_text handles all user edge cases for subscripts, superscripts, and spacing."""
    # User-specified primary test cases:
    assert clean_scientific_text("TBA 2 SO 4 ") == "TBA₂SO₄ "
    assert clean_scientific_text("coupled to α-MnO 2") == "coupled to α-MnO₂"
    assert clean_scientific_text("coupled to α - MnO 2") == "coupled to α-MnO₂"
    assert clean_scientific_text("coupled to α- MnO 2") == "coupled to α-MnO₂"
    assert clean_scientific_text("coupled to MnO 2") == "coupled to MnO₂"
    assert clean_scientific_text("coupled to β-MnO 2") == "coupled to β-MnO₂"

    # Electrolyte salts & units:
    t1 = "tested in 2 M ZnSO 4 with 10 mM trehalose at 2 mA cm -2"
    assert clean_scientific_text(t1) == "tested in 2 M ZnSO₄ with 10 mM trehalose at 2 mA/cm²"

    # Ions, crystal planes & cell notation:
    t2 = "delivering Zn 2+ ions to ( 002 ) plane in Zn // Zn cell"
    assert clean_scientific_text(t2) == "delivering Zn²⁺ ions to (002) plane in Zn//Zn cell"

    # Coordination complexes, basic salts & spacing around brackets:
    t3 = "formation of [ ZnCl 4 ] 2- complexes and Zn 4 SO 4 ( OH ) 6 byproduct ."
    assert clean_scientific_text(t3) == "formation of [ZnCl₄]²⁻ complexes and Zn₄SO₄(OH)₆ byproduct."

    # Multi-element salts & quaternary salts:
    t4 = "Al 2 ( SO 4 ) 3 and Na 2 SO 4 in aqueous electrolyte ."
    assert clean_scientific_text(t4) == "Al₂(SO₄)₃ and Na₂SO₄ in aqueous electrolyte."

    t5 = "(TBA) 2 SO 4 and (NH 4 ) 2 SO 4 with [ NR 4 ][ TFSI ]"
    assert clean_scientific_text(t5) == "(TBA)₂SO₄ and (NH₄)₂SO₄ with [NR₄][TFSI⁻]"

    # Cathodes, MXenes & 4-digit Miller-Bravais indices:
    t6 = "Zn // Cu cell with ( 0002 ) plane, ZnMn 2 O 4 cathode, and Ti 3 C 2 T x MXene"
    assert clean_scientific_text(t6) == "Zn//Cu cell with (0002) plane, ZnMn₂O₄ cathode, and Ti₃C₂Tₓ MXene"

    # User-specified coordination complexes, aqua ligands, hydrates, oxides, anions, and units:
    assert clean_scientific_text("[Zn(OH2)6]2+") == "[Zn(OH₂)₆]²⁺"
    assert clean_scientific_text("Zn₄SO₄(OH)₆·xH2O and H₂") == "Zn₄SO₄(OH)₆·xH₂O and H₂"
    assert clean_scientific_text("[Zn(H₂O) 4 (SF)] 2+") == "[Zn(H₂O)₄(SF)]²⁺"
    assert clean_scientific_text("Li 2 O/Li 2 CO 3") == "Li₂O/Li₂CO₃"
    assert clean_scientific_text("Cl –") == "Cl⁻"
    assert clean_scientific_text("1g L –1") == "1 g L⁻¹"
    assert clean_scientific_text("Zn//NH₄ V₄O₁₀") == "Zn//NH₄V₄O₁₀"

    # Additional complex coordination and hydrate variants:
    assert clean_scientific_text("Zn4SO4(OH)₆·xH₂O") == "Zn₄SO₄(OH)₆·xH₂O"
    assert clean_scientific_text("Zn4SO4(OH)6·xH2O") == "Zn₄SO₄(OH)₆·xH₂O"
    assert clean_scientific_text("Zn<sub>4</sub>SO<sub>4</sub>(OH)<sub>6</sub>·xH<sub>2</sub>O") == "Zn₄SO₄(OH)₆·xH₂O"
    assert clean_scientific_text("Zn4SO4") == "Zn₄SO₄"
    assert clean_scientific_text("[Zn(H2O)6]2+") == "[Zn(H₂O)₆]²⁺"
    assert clean_scientific_text("[Zn(H2O) 6] 2+") == "[Zn(H₂O)₆]²⁺"
    assert clean_scientific_text("[Fe(CN)6]3-") == "[Fe(CN)₆]³⁻"
    assert clean_scientific_text("[Cu(NH3)4]2+") == "[Cu(NH₃)₄]²⁺"
    assert clean_scientific_text("ZnSO4·7H2O") == "ZnSO₄·7H₂O"
    assert clean_scientific_text("ZnSO4 · 7 H2O") == "ZnSO₄·7H₂O"
    assert clean_scientific_text("Cl-rich and F-doped") == "Cl-rich and F-doped"
    assert clean_scientific_text("Cl – and F – and OH –") == "Cl⁻ and F⁻ and OH⁻"
    assert clean_scientific_text("10 mg L -1") == "10 mg L⁻¹"
    assert clean_scientific_text("10 g L^-1") == "10 g L⁻¹"

    # New unit formatting & dash variations:
    assert clean_scientific_text("7.82 g cm −2") == "7.82 g cm⁻²"
    assert clean_scientific_text("5,849 mAh/cm3") == "5,849 mAh/cm³"
    assert clean_scientific_text("100 mA/cm2 and 500 mAh/cm3") == "100 mA/cm² and 500 mAh/cm³"
    assert clean_scientific_text("density of 1.25 g cm-3") == "density of 1.25 g cm⁻³"

    # Leading Abstract stripping:
    assert clean_scientific_text("Abstract: High-performance aqueous zinc battery") == "High-performance aqueous zinc battery"
    assert clean_scientific_text("ABSTRACT: High capacity of 5,849 mAh/cm3 is achieved.") == "High capacity of 5,849 mAh/cm³ is achieved."
    assert clean_scientific_text("Abstract - The cycling stability of Zn//Zn") == "The cycling stability of Zn//Zn"
    assert clean_scientific_text("Abstract. A novel additive is proposed.") == "A novel additive is proposed."
    assert clean_scientific_text("<b>Abstract</b>: Zinc battery...") == "Zinc battery..."
    assert clean_scientific_text("Abstract\nHere we demonstrate...") == "Here we demonstrate..."

    # Ensure ordinary words starting with abstract/abstraction are NOT corrupted:
    assert clean_scientific_text("Abstract algebra in materials modelling") == "Abstract algebra in materials modelling"
    assert clean_scientific_text("Abstraction of proton occurs rapidly") == "Abstraction of proton occurs rapidly"

    # None and empty string handling
    assert clean_scientific_text(None) is None
    assert clean_scientific_text("") == ""


# ============================================================================
# 7. BATCH MULTI-PDF DOCUMENT INGESTION & CHEMICAL EXTRACTION TESTS
# ============================================================================

def test_extract_text_from_pdf_stream_bytes_and_path():
    """Verify PaperExtractor.extract_text_from_pdf works with paths, open file handles, BytesIO, and raw bytes."""
    import os
    import io
    pdf_path = os.path.join("samples", "sample_zib_paper.pdf")
    assert os.path.exists(pdf_path), "sample_zib_paper.pdf must exist in samples directory"

    extractor = PaperExtractor()

    # 1. File path string
    text_path = extractor.extract_text_from_pdf(pdf_path)
    assert len(text_path) > 50
    assert "Tetraethylammonium" in text_path

    # 2. Open file object
    with open(pdf_path, "rb") as f:
        text_f = extractor.extract_text_from_pdf(f)
        assert "Tetraethylammonium" in text_f

    # 3. BytesIO buffer
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    bio = io.BytesIO(pdf_bytes)
    text_bio = extractor.extract_text_from_pdf(bio)
    assert "Tetraethylammonium" in text_bio

    # 4. Raw bytes object
    text_raw = extractor.extract_text_from_pdf(pdf_bytes)
    assert "Tetraethylammonium" in text_raw


def test_recommend_candidates_from_additives_and_rdkit_ranking():
    """Verify recommend_candidates_from_additives computes Morgan fingerprints, Delaney ESOL, and compatibility."""
    engine = LiteratureDiscoveryEngine(gemini_api_key=None)

    additives = [
        ExtractedAdditive(
            name="Tetraethylammonium bromide",
            smiles="CC[N+](CC)(CC)CC.[Br-]",
            optimal_concentration=Concentration(value=50.0, unit="mM", raw_text="50 mM"),
            concentration_range_tested=[Concentration(value=50.0, unit="mM", raw_text="50 mM")],
            functional_groups=["quaternary ammonium"],
            primary_mechanisms=["Dynamic electrostatic shielding", "HER suppression"],
            mechanism_summary="Bulky TEA+ cations form an electrostatic shield.",
            battery_context=ZincIonBatteryContext(baseline_electrolyte="2 M ZnSO4", solvent="H2O")
        ),
        ExtractedAdditive(
            name="D-Sorbitol",
            smiles="C(C(C(C(C(CO)O)O)O)O)O",
            optimal_concentration=Concentration(value=15.0, unit="mM", raw_text="15 mM"),
            concentration_range_tested=[Concentration(value=15.0, unit="mM", raw_text="15 mM")],
            functional_groups=["hydroxyl", "polyol"],
            primary_mechanisms=["Zn²⁺ solvation sheath modulation", "HER suppression"],
            mechanism_summary="Polyol hydroxyl groups modulate Zn²⁺ solvation sheath.",
            battery_context=ZincIonBatteryContext(baseline_electrolyte="2 M ZnSO4", solvent="H2O")
        )
    ]

    candidates = engine.recommend_candidates_from_additives(additives, top_n=8)
    assert len(candidates) >= 2

    # Direct literature findings should have similarity_score = 1.0
    direct_findings = [c for c in candidates if c.similarity_metric == "Direct Literature Finding"]
    assert len(direct_findings) == 2
    direct_names = [c.name for c in direct_findings]
    assert "Tetraethylammonium bromide" in direct_names
    assert "D-Sorbitol" in direct_names

    # Check that descriptors and Delaney ESOL LogS are populated
    for cand in candidates:
        assert cand.molecular_weight > 0
        assert cand.logs is not None
        assert cand.parent_baseline_electrolyte == "2 M ZnSO4"
        assert cand.electrolyte_compatibility_breakdown is not None
        assert len(cand.electrolyte_compatibility_breakdown) > 0


def test_process_multiple_pdfs_batch_execution():
    """Verify batch processing of PDF documents produces ExtractionResult list with candidate molecules."""
    import os
    pdf_path = os.path.join("samples", "sample_zib_paper.pdf")
    engine = LiteratureDiscoveryEngine(gemini_api_key=None)

    results = engine.process_multiple_pdfs([pdf_path])
    assert len(results) == 1

    res = results[0]
    assert isinstance(res, ExtractionResult)
    assert res.source_type == "pdf"
    assert len(res.additives) >= 1
    assert res.additives[0].name == "Tetraethylammonium bromide"
    assert res.additives[0].optimal_concentration.value == 50.0
    assert len(res.candidate_molecules) > 0


def test_zero_fabrication_rule_for_paper_without_additive():
    """Verify PaperExtractor returns empty additive list and does not hallucinate fake additives."""
    extractor = PaperExtractor()
    non_additive_text = (
        "In this study, we investigated the crystallographic phase transitions of MnO2 cathode materials "
        "in conventional 2 M ZnSO4 aqueous electrolytes. No chemical additives or co-solvents were incorporated. "
        "The cell demonstrated stable intercalation over 500 cycles."
    )
    additives, diag = extractor.extract_from_text(non_additive_text)
    assert len(additives) == 0, f"Expected 0 additives for non-additive paper, got {len(additives)}"
    assert "No recognized electrolyte additive" in diag or len(diag) > 0


def test_extract_accurate_concentration_and_tested_range():
    """Verify PaperExtractor extracts the true optimal concentration and tested range."""
    extractor = PaperExtractor()
    paper_text = (
        "We systematically tested trehalose at various concentrations (5 mM, 10 mM, 20 mM, 50 mM) "
        "in 2 M ZnSO4 baseline aqueous electrolyte. The optimal concentration was 20 mM trehalose, "
        "which elevated the hydrogen evolution overpotential and yielded over 1200 h of dendrite-free Zn plating."
    )
    additives, _ = extractor.extract_from_text(paper_text)
    assert len(additives) == 1
    add = additives[0]
    assert add.name == "Trehalose"
    assert add.optimal_concentration.value == 20.0
    assert add.optimal_concentration.unit == "mM"
    assert len(add.concentration_range_tested) >= 4
    tested_vals = [c.value for c in add.concentration_range_tested]
    assert 5.0 in tested_vals
    assert 10.0 in tested_vals
    assert 20.0 in tested_vals
    assert 50.0 in tested_vals


def test_clean_scientific_text_zn2_plus_superscript_variants():
    """Verify clean_scientific_text converts all Zn2+ permutations to Zn²⁺ with proper superscript."""
    cases = [
        ("Zn2+", "Zn²⁺"),
        ("Zn 2+", "Zn²⁺"),
        ("Zn^2+", "Zn²⁺"),
        ("Zn 2^+", "Zn²⁺"),
        ("Zn2+ solvation sheath modulation", "Zn²⁺ solvation sheath modulation"),
        ("coordinates with Zn 2+ ions", "coordinates with Zn²⁺ ions"),
        ("dynamic shielding repels Zn^2+ accumulation", "dynamic shielding repels Zn²⁺ accumulation"),
    ]
    for raw, expected in cases:
        cleaned = clean_scientific_text(raw)
        assert expected in cleaned, f"Expected '{expected}' in '{cleaned}' (from '{raw}')"


def test_extract_edta_cuna2_paper_and_suggest_analogues():
    """Verify PaperExtractor extracts EDTA-CuNa2 chelate additive and suggests Disodium EDTA analogue."""
    extractor = PaperExtractor()
    paper_text = (
        "Reversible aqueous zinc-ion batteries enabled by copper disodium ethylenediaminetetraacetate (EDTA-CuNa2) additive. "
        "Herein, trace copper disodium ethylenediaminetetraacetate (EDTA-CuNa2) is introduced as an electrolyte additive "
        "into 2 M ZnSO4 baseline aqueous electrolyte. The optimal concentration was determined to be 5 mM. "
        "Cu²⁺ cations spontaneously deposit to construct an in situ Cu-Zn alloy interface layer, suppressing dendritic growth."
    )
    additives, summary = extractor.extract_from_text(paper_text)
    assert len(additives) >= 1
    add = additives[0]
    assert "EDTA-CuNa2" in add.name or "copper disodium" in add.name.lower()
    assert add.optimal_concentration is not None
    assert add.optimal_concentration.value == 5.0
    assert add.optimal_concentration.unit == "mM"

    # Verify RDKit similarity candidate suggestion
    lit_engine = LiteratureDiscoveryEngine()
    analogues = lit_engine.recommend_candidates_for_additive(add, top_n=3)
    assert len(analogues) > 0
    names = [c.name for c in analogues]
    assert any("EDTA" in n or "citrate" in n.lower() or "Urea" in n for n in names)


def test_extract_du_dimethylpropyleneurea_ocr_zero_and_analogues():
    """Verify PaperExtractor extracts N,N0-dimethylpropyleneurea (DU) with OCR error and suggests cyclic urea analogue."""
    extractor = PaperExtractor()
    paper_text = (
        "Modulating Zn2+ Solvation Structure via N,N0-dimethylpropyleneurea (DU) for High-Performance Zinc Batteries. "
        "In this study, dipolar N,N0-dimethylpropyleneurea (DU) was added to 2 M ZnSO4 aqueous electrolyte at 10 mM. "
        "The carbonyl oxygen preferentially coordinates with Zn2+ cations, expelling active water molecules and inhibiting hydrogen evolution."
    )
    additives, summary = extractor.extract_from_text(paper_text)
    assert len(additives) >= 1
    add = additives[0]
    assert "dimethylpropyleneurea" in add.name.lower() or "DU" in add.name
    assert add.optimal_concentration is not None
    assert add.optimal_concentration.value == 10.0

    # Verify RDKit similarity candidate suggestion
    lit_engine = LiteratureDiscoveryEngine()
    analogues = lit_engine.recommend_candidates_for_additive(add, top_n=3)
    assert len(analogues) > 0
    names = [c.name for c in analogues]
    assert any("1,3-Dimethyl-2-imidazolidinone" in n or "Urea" in n for n in names)


def test_extract_cuso4_inorganic_salt_additive_and_analogues():
    """Verify PaperExtractor extracts trace CuSO4 inorganic salt additive and suggests MnSO4 analogue."""
    extractor = PaperExtractor()
    paper_text = (
        "Trace CuSO4 electrolyte additive for dendrite-free aqueous zinc anodes. "
        "We incorporated 1 mM CuSO4 into 2 M ZnSO4 baseline electrolyte as a multifunctional additive. "
        "The trace Cu2+ ions establish an in situ replacement alloy buffer layer on the zinc metal anode, guiding uniform zinc deposition."
    )
    additives, summary = extractor.extract_from_text(paper_text)
    assert len(additives) >= 1
    add = additives[0]
    assert "CuSO" in add.name or "copper" in add.name.lower()
    assert add.optimal_concentration is not None
    assert add.optimal_concentration.value == 1.0
    assert add.optimal_concentration.unit == "mM"

    # Verify RDKit similarity candidate suggestion
    lit_engine = LiteratureDiscoveryEngine()
    analogues = lit_engine.recommend_candidates_for_additive(add, top_n=3)
    assert len(analogues) > 0
    names = [c.name for c in analogues]
    assert any("Manganese" in n or "MnSO" in n or "sulfate" in n.lower() or "EDTA" in n for n in names)


def test_extract_sodium_citrate_paper_and_analogues():
    """Verify PaperExtractor extracts sodium citrate (SC) additive and suggests citric acid analogue."""
    extractor = PaperExtractor()
    paper_text = (
        "Interfacial regulation by sodium citrate (SC) additive for durable aqueous zinc batteries. "
        "Sodium citrate (SC) was introduced into 2 M ZnSO4 baseline aqueous electrolyte at an optimal concentration of 25 mM. "
        "Citrate anions adsorb dynamically onto active zinc facets, suppressing hydrogen evolution and side reactions."
    )
    additives, summary = extractor.extract_from_text(paper_text)
    assert len(additives) >= 1
    add = additives[0]
    assert "citrate" in add.name.lower() or "SC" in add.name
    assert add.optimal_concentration is not None
    assert add.optimal_concentration.value == 25.0
    assert add.optimal_concentration.unit == "mM"

    # Verify RDKit similarity candidate suggestion
    lit_engine = LiteratureDiscoveryEngine()
    analogues = lit_engine.recommend_candidates_for_additive(add, top_n=3)
    assert len(analogues) > 0
    names = [c.name for c in analogues]
    assert any("Citric" in n or "Succinic" in n or "EDTA" in n for n in names)


# ============================================================================
# 11. CHEMCROW, LEMAT-SYNTH & PAPERQA2 ARCHITECTURAL INTEGRATION TESTS
# ============================================================================

def test_pubchem_client_name_to_smiles_mocked():
    """Verify PubChemClient resolves compound names to SMILES via mocked PUG REST API."""
    client = PubChemClient()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "PropertyTable": {
            "Properties": [{"CID": 5780, "SMILES": "C(C(C(C(C(CO)O)O)O)O)O"}]
        }
    }
    with patch.object(client.session, "get", return_value=mock_response):
        smi = client.name_to_smiles("sorbitol")
        assert smi == "OCC(O)C(O)C(O)C(O)CO"


def test_pubchem_client_fast_similarity_search_mocked():
    """Verify PubChemClient queries fast 2D similarity structural analogues."""
    client = PubChemClient()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "PropertyTable": {
            "Properties": [
                {"CID": 6251, "SMILES": "C(C(C(C(C(CO)O)O)O)O)O", "Title": "D-Mannitol"},
                {"CID": 5793, "SMILES": "C(C1C(C(C(C(O1)O)O)O)O)O", "Title": "D-Glucose"},
            ]
        }
    }
    with patch.object(client.session, "get", return_value=mock_response):
        hits = client.search_similar_compounds("C(C(C(C(C(CO)O)O)O)O)O", threshold=85, max_records=2)
        assert len(hits) == 2
        assert hits[0]["name"] == "D-Mannitol"
        assert hits[0]["cid"] == 6251
        assert hits[1]["name"] == "D-Glucose"


def test_pubchem_client_offline_graceful_fallback():
    """Verify PubChemClient fails gracefully without throwing uncaught exceptions when offline."""
    import requests
    client = PubChemClient()
    with patch.object(client.session, "get", side_effect=requests.RequestException("DNS lookup failed")):
        assert client.name_to_smiles("unknown_substance_xyz") is None
        assert client.smiles_to_name("CCO") is None
        assert client.search_similar_compounds("CCO") == []


def test_rdkit_similarity_engine_name_resolution_curated_and_aliases():
    """Verify RDKitSimilarityEngine resolves common names and benchmark aliases to valid SMILES."""
    from rdkit import Chem
    engine = RDKitSimilarityEngine()

    test_queries = [
        ("D-Sorbitol", "C(C(C(C(C(CO)O)O)O)O)O"),
        ("EDTA-CuNa2", "[Na+].[Na+].[Cu+2].[O-]C(=O)CN(CCN(CC(=O)[O-])CC(=O)[O-])CC(=O)[O-]"),
        ("DU", "CN1CCCN(C)C1=O"),
        ("CuSO4", "[Cu+2].[O-]S(=O)(=O)[O-]"),
        ("SC", "[Na+].[Na+].[Na+].C(C(=O)[O-])C(CC(=O)[O-])(C(=O)[O-])O"),
        ("TEABr", "CC[N+](CC)(CC)CC.[Br-]"),
    ]
    for name, expected_smi in test_queries:
        resolved = engine.resolve_chemical_name_to_smiles(name)
        assert resolved is not None, f"Failed to resolve {name}"
        mol = Chem.MolFromSmiles(resolved)
        assert mol is not None, f"Resolved SMILES for {name} is not a valid RDKit molecule: {resolved}"


def test_rdkit_similarity_engine_dynamic_pubchem_integration():
    """Verify RDKitSimilarityEngine merges dynamic PubChem results into candidate rankings."""
    mock_client = MagicMock(spec=PubChemClient)
    # Use a novel molecule not in the default pool (e.g., Acetamide)
    mock_client.search_similar_compounds.return_value = [
        {
            "cid": 178,
            "name": "Acetamide (PubChem Discovery)",
            "smiles": "CC(=O)N",
        }
    ]
    engine = RDKitSimilarityEngine(pubchem_client=mock_client)
    # Query with Urea
    candidates = engine.find_similar_candidates(
        query_smiles="C(=O)(N)N",
        top_n=5,
        threshold=0.10,
        baseline_electrolyte="2 M ZnSO4",
        use_pubchem=True
    )
    assert len(candidates) > 0
    # Acetamide should be discovered, scored and ranked
    pubchem_cand = next((c for c in candidates if "PubChem" in c.similarity_metric), None)
    assert pubchem_cand is not None
    assert pubchem_cand.name == "Acetamide (PubChem Discovery)"
    assert pubchem_cand.similarity_score > 0.0
    assert pubchem_cand.logs is not None
    assert pubchem_cand.parent_baseline_electrolyte == "2 M ZnSO4"
    assert "Validated for sulfate chemistry" in pubchem_cand.electrolyte_compatibility_note


def test_paperqa_text_sanitization_unicode_and_hyphens():
    """Verify PaperQA2-inspired unicode sanitization and scientific de-hyphenation across linebreaks."""
    raw_corrupted = "Electrolyte with Zn\ud800\udc00 salt\x00 and trace \u200badditive."
    cleaned = clean_invalid_unicode(raw_corrupted)
    assert "\x00" not in cleaned
    assert "\u200b" not in cleaned

    broken_text = (
        "We report a multi-functional elec-\n"
        "trolyte addi-\n"
        "tive for Zn²⁺ batteries. Specifically, D-\n"
        "sorbitol was utilized."
    )
    dehyphed = dehyphenate_scientific_text(broken_text)
    assert "electrolyte" in dehyphed
    assert "additive" in dehyphed
    assert "D-sorbitol" in dehyphed


def test_lemat_synth_zib_domain_config():
    """Verify LeMat-Synth-inspired ZIBDomainConfig defines proper schema instructions and unit normalizations."""
    config = ZIBDomainConfig()
    assert config.domain_name == "aqueous_zinc_ion_battery_electrolyte_additives"
    assert "mM" in config.supported_concentration_units
    assert "wt%" in config.supported_concentration_units
    assert "ZnSO4" in config.canonical_salts

    # Test unit normalizations
    assert config.normalize_unit("millimolar") == "mM"
    assert config.normalize_unit("mm") == "mM"
    assert config.normalize_unit("weight %") == "wt%"
    assert config.normalize_unit("vol %") == "vol%"

    # Test salt normalizations
    assert config.normalize_salt("zinc sulfate") == "2 M ZnSO4"
    assert config.normalize_salt("Zn(OTf)2") == "Zn(OTf)2"

    # Test prompt synthesis
    prompt = config.get_extraction_prompt("Sample paper abstract text")
    assert "Zinc-Ion Battery" in prompt
    assert "Zero-Fabrication Mandate" in prompt
    assert "optimal_concentration" in prompt


# ============================================================================
# 7. OPEN-ENDED NER, UNSPECIFIED SALTS & BATCH EXTRACTION TESTS
# ============================================================================

def test_open_ended_chemical_ner_extracts_novel_additive():
    """Verify PaperExtractor extracts novel/unseen chemical additive names without forcing them into curated vocabulary."""
    extractor = PaperExtractor()
    novel_paper_text = (
        "Reversible zinc anode electrochemistry enabled by xylitol additive. "
        "Herein, xylitol was investigated as an electrolyte additive for aqueous zinc-ion batteries. "
        "The optimal concentration was found to be 25 mM xylitol, which modified the Zn²⁺ solvation sheath "
        "and suppressed the hydrogen evolution reaction over 1000 hours of cycling."
    )
    additives, _ = extractor.extract_from_text(novel_paper_text)
    assert len(additives) == 1
    add = additives[0]
    assert "xylitol" in add.name.lower()
    assert add.name != "D-Sorbitol"  # Must not force into similar curated polyol!
    assert add.optimal_concentration.value == 25.0
    assert add.optimal_concentration.unit == "mM"


def test_scientific_honesty_unspecified_baseline_salt():
    """Verify that PaperExtractor does NOT hallucinate 2 M ZnSO4 when baseline electrolyte is omitted."""
    extractor = PaperExtractor()
    no_salt_text = (
        "Tannic acid was explored as a multifunctional electrolyte additive for aqueous zinc batteries. "
        "Adding 10 mM tannic acid into the electrolyte regulated Zn²⁺ interfacial migration "
        "and suppressed dendrite formation during long-term cycling."
    )
    additives, _ = extractor.extract_from_text(no_salt_text)
    assert len(additives) == 1
    add = additives[0]
    assert "tannic acid" in add.name.lower()
    assert add.battery_context.baseline_electrolyte == "Unspecified / Not reported"
    assert "2 M ZnSO" not in add.battery_context.baseline_electrolyte

    # Now verify that when a salt IS specified, it extracts that exact salt
    salt_text = (
        "Tannic acid was explored as an electrolyte additive in 1 M Zn(OTf)2 baseline electrolyte. "
        "The optimal concentration was 10 mM."
    )
    additives_salt, _ = extractor.extract_from_text(salt_text)
    assert len(additives_salt) == 1
    assert "Zn(OTf)" in additives_salt[0].battery_context.baseline_electrolyte


def test_extract_batch_from_papers_micro_chunking_offline_and_mocked():
    """Verify PaperExtractor.extract_batch_from_papers micro-chunks papers and calls progress callbacks."""
    extractor = PaperExtractor()

    papers = [
        PaperMetadata(paper_id=f"p_{i}", title=f"Paper {i} with Trehalose additive", abstract=f"Trehalose was used as an electrolyte additive at {10 + i} mM in 2 M ZnSO4.")
        for i in range(14)
    ]

    callback_calls = []

    def mock_cb(batch_idx, total_batches, batch_papers, accum_res):
        callback_calls.append((batch_idx, total_batches, len(batch_papers), len(accum_res)))

    # Test offline mode batching (batch_size=6 -> 3 batches for 14 papers: 6, 6, 2)
    results = extractor.extract_batch_from_papers(papers, batch_size=6, batch_callback=mock_cb)

    assert len(results) == 14
    assert len(callback_calls) == 3
    assert callback_calls[0] == (1, 3, 6, 6)
    assert callback_calls[1] == (2, 3, 6, 12)
    assert callback_calls[2] == (3, 3, 2, 14)

    for rec in results:
        assert "paper" in rec
        assert "additive" in rec
        assert rec["additive"].name == "Trehalose"

    # Test mocked LLM structured batch response
    mock_llm_batch_response = [
        {
            "paper_index": 0,
            "has_additive": True,
            "additives": [
                {
                    "name": "Novel Fluorinated Chelate",
                    "optimal_concentration": {"value": 8.5, "unit": "mM", "raw_text": "8.5 mM"},
                    "concentration_range_tested": [{"value": 8.5, "unit": "mM", "raw_text": "8.5 mM"}],
                    "primary_mechanisms": ["Zn²⁺ solvation sheath modulation"],
                    "battery_context": {"baseline_electrolyte": "1 M Zn(OTf)2", "solvent": "H₂O"}
                }
            ]
        }
    ]

    with patch.object(extractor, "_call_llm_structured", return_value=mock_llm_batch_response):
        test_papers = [PaperMetadata(paper_id="single", title="Novel Chelate study", abstract="Study of novel chelate")]
        llm_results = extractor.extract_batch_from_papers(test_papers, batch_size=6)
        assert len(llm_results) == 1
        assert llm_results[0]["additive"].name == "Novel Fluorinated Chelate"
        assert llm_results[0]["additive"].optimal_concentration.value == 8.5
        assert llm_results[0]["additive"].battery_context.baseline_electrolyte == "1 M Zn(OTf)2"


def test_paper_extractor_and_engine_default_and_custom_model():
    """Verify PaperExtractor and LiteratureDiscoveryEngine default to gemini-3.5-flash-lite and accept custom models."""
    ext_default = PaperExtractor()
    assert ext_default.model_name == "gemini-3.5-flash-lite"

    ext_custom = PaperExtractor(model_name="gemini-3.1-pro")
    assert ext_custom.model_name == "gemini-3.1-pro"

    engine_default = LiteratureDiscoveryEngine()
    assert engine_default.extractor.model_name == "gemini-3.5-flash-lite"

    engine_custom = LiteratureDiscoveryEngine(llm_model="gemini-3.8-flash")
    assert engine_custom.extractor.model_name == "gemini-3.8-flash"


def test_graphene_oxide_paper_extraction_honest_conc_and_clean_name():
    """
    Test extraction on the user's specific paper:
    'Elimination of Zinc Dendrites by Graphene Oxide Electrolyte Additive for Zinc-Ion Batteries'.
    Verifies:
    1. Additive name is cleanly extracted as 'Graphene oxide' (without 'we developed a hybrid electrolyte with')
    2. Optimal concentration is None (honest, NOT fabricated 15 mM)
    3. SMILES is None (nanomaterial, NOT defaulted to ethylene glycol C(CO)O)
    4. Baseline electrolyte is 'Unspecified / Not reported'
    5. Chemical formula resolves to 'GO' via CURATED_FORMULAS
    """
    from app import get_chemical_formula
    title = "Elimination of Zinc Dendrites by Graphene Oxide Electrolyte Additive for Zinc-Ion Batteries"
    abstract = (
        "Zinc-ion batteries have drawn tremendous attention in the electrochemical energy storage market "
        "because of their merits of high safety and low cost, however, the dendrites issue of zinc anodes "
        "remains a challenge. Herein, we developed a hybrid electrolyte with graphene oxide as an additive "
        "with the functions of promoting the uniform distribution of the electric field and reducing the "
        "nucleation overpotential of Zn²⁺, displaying a smooth zinc electrodeposition layer and reaction kinetics. "
        "The resultant zinc symmetric battery with graphene oxide additive affords a stable zinc anode which "
        "provided service for more than 650 h at 1 mA/cm² and an exceptional cycle life of 140 h even at 10 mA/cm². "
        "In addition, scanning electron microscopy images demonstrated that the deposition of zinc dendrites "
        "through the glass fiber separator was the reason for the battery short circuit. It is believed that the "
        "graphene oxide additive can remarkably improve the Coulombic efficiency (99.16%) of the batteries via "
        "stabilizing the Zn plating/stripping behavior and facilitating Zn²⁺ nucleation. Consequently, the full "
        "battery with graphene oxide electrolyte additive exhibited remarkable enhancements in rate performance "
        "and cyclability compared to the additive-free electrolyte. The hybrid electrolyte containing solid "
        "particles opens a chapter for designing advanced and durable zinc-ion batteries."
    )
    full_text = f"{title}\n{abstract}"
    extractor = PaperExtractor()
    extracted, summary = extractor.extract_from_text(full_text)

    assert len(extracted) == 1
    item = extracted[0]
    assert item.name == "Graphene oxide"
    assert "developed a" not in item.name
    assert "electrolyte" not in item.name
    # Concentration must be None since the abstract never mentioned a concentration
    assert item.optimal_concentration is None
    # SMILES must be None for 2D carbon nanomaterial
    assert item.smiles is None
    # Baseline electrolyte is unspecified
    assert "unspecified" in item.battery_context.baseline_electrolyte.lower()
    # Formula resolves to GO
    formula = get_chemical_formula(item.smiles, item.name)
    assert formula == "GO"


def test_gemini_micro_chunk_batch_with_null_concentration_pydantic_validation():
    """Verify that Gemini returning null for concentration validates in ExtractedAdditive without raising ValidationError."""
    from typing import List
    from models import ExtractedAdditive
    from pydantic import TypeAdapter

    null_conc_payload = [
        {
            "name": "Graphene oxide",
            "iupac_name": None,
            "smiles": None,
            "optimal_concentration": None,
            "concentration_range_tested": [],
            "functional_groups": ["carbon nanomaterial"],
            "primary_mechanisms": ["Dynamic electrostatic shielding", "HER suppression"],
            "mechanism_summary": "Promotes uniform electric field and reduces Zn2+ nucleation overpotential.",
            "battery_context": {
                "baseline_electrolyte": "Unspecified / Not reported",
                "solvent": "H₂O"
            },
            "confidence_score": 0.95,
            "extraction_evidence": "developed a hybrid electrolyte with graphene oxide as an additive"
        }
    ]

    adapter = TypeAdapter(List[ExtractedAdditive])
    valid_adds = adapter.validate_python(null_conc_payload)
    assert len(valid_adds) == 1
    assert valid_adds[0].name == "Graphene oxide"
    assert valid_adds[0].optimal_concentration is None
    assert valid_adds[0].smiles is None


def test_rsc_paper_detection_and_localization():
    """Verify Royal Society of Chemistry detection logic and localization keys across all languages."""
    from localization import get_text as tr

    rsc_doi = "10.1039/D1TA01234A"
    non_rsc_doi = "10.1021/acs.est.1c02345"

    assert rsc_doi.lower().startswith("10.1039/")
    assert not non_rsc_doi.lower().startswith("10.1039/")

    for lang in ["en", "fr", "zh"]:
        badge = tr("badge_rsc_partial_abstract", lang=lang)
        notice = tr("rsc_abstract_notice", lang=lang)
        unspec_conc = tr("card_reported_conc_unspecified", lang=lang)
        err_msg = tr("gemini_api_error_warning", lang=lang)

        assert badge and not badge.startswith("[")
        assert notice and not notice.startswith("[")
        assert unspec_conc and not unspec_conc.startswith("[")
        assert err_msg and not err_msg.startswith("[")


def test_extract_batch_from_papers_stop_check():
    """Verify that extract_batch_from_papers honors the stop_check callback and stops gracefully."""
    extractor = PaperExtractor(api_key=None)
    mock_papers = [
        PaperMetadata(paper_id=f"p_{i}", title=f"Paper {i}", abstract=f"Abstract for paper {i} testing additive.")
        for i in range(10)
    ]

    # Stop after the second batch (i.e., after 4 papers when batch_size=2)
    batch_count = 0
    def stop_flag():
        nonlocal batch_count
        batch_count += 1
        return batch_count > 2

    results = extractor.extract_batch_from_papers(
        mock_papers,
        batch_size=2,
        stop_check=stop_flag
    )

    # Only 4 papers should have been processed (2 batches of 2)
    assert len(results) == 4
    assert [r["paper"].paper_id for r in results] == ["p_0", "p_1", "p_2", "p_3"]


def test_extract_batch_from_papers_transient_503_retry():
    """Verify that transient 503/high-demand spikes trigger exponential backoff retry and succeed."""
    extractor = PaperExtractor(api_key="AIzaSyTestKeyDummy")
    mock_client = MagicMock()
    extractor._genai_client = mock_client

    mock_papers = [
        PaperMetadata(paper_id="p_0", title="Graphene oxide additive in ZIB", abstract="Using graphene oxide in zinc ion battery.")
    ]

    valid_json_resp = json.dumps([
        {
            "paper_index": 0,
            "has_additive": True,
            "additives": [
                {
                    "name": "Graphene oxide",
                    "iupac_name": None,
                    "smiles": None,
                    "optimal_concentration": None,
                    "concentration_range_tested": [],
                    "functional_groups": ["carbon nanomaterial"],
                    "primary_mechanisms": ["Dynamic electrostatic shielding"],
                    "mechanism_summary": "Shields zinc surface.",
                    "battery_context": {
                        "baseline_electrolyte": "Unspecified / Not reported",
                        "solvent": "H₂O"
                    },
                    "confidence_score": 0.95,
                    "extraction_evidence": "graphene oxide in zinc ion battery"
                }
            ]
        }
    ])

    mock_success_response = MagicMock()
    mock_success_response.text = valid_json_resp

    # Attempt 1: 503 error, Attempt 2: Success
    mock_client.models.generate_content.side_effect = [
        Exception("503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand. Spikes in demand'}}"),
        mock_success_response
    ]

    with patch("time.sleep", return_value=None) as mock_sleep:
        results = extractor.extract_batch_from_papers(
            mock_papers,
            batch_size=5
        )

    assert len(results) == 1
    assert results[0]["additive"].name == "Graphene oxide"
    assert results[0]["additive"].optimal_concentration is None
    # Verify retry was attempted and sleep was called with backoff
    assert mock_client.models.generate_content.call_count == 2
    mock_sleep.assert_called_once_with(1.5)


def test_batch_extraction_zero_fabrication_for_paper_without_additive():
    """Verify that when an LLM returns has_additive: false, it does NOT fabricate Aqueous ZIB Additive or run regex fallback."""
    extractor = PaperExtractor(api_key="AIzaSyTestKeyDummy")
    mock_client = MagicMock()
    extractor._genai_client = mock_client

    mock_papers = [
        PaperMetadata(
            paper_id="paper_cathode_study",
            title="MnO2 Cathode Architecture for Aqueous Zinc Batteries",
            abstract="We develop a hybrid electrolyte with enhanced compatibility for MnO2 cathode without testing any additive."
        )
    ]

    mock_response = MagicMock()
    mock_response.text = json.dumps([
        {
            "paper_index": 0,
            "has_additive": False,
            "additives": []
        }
    ])
    mock_client.models.generate_content.return_value = mock_response

    results = extractor.extract_batch_from_papers(mock_papers, batch_size=5)

    assert len(results) == 1
    # Additive must be None, NOT "Aqueous ZIB Additive" and NOT "Electrolyte"
    assert results[0]["additive"] is None


def test_batch_extraction_rejects_generic_electrolyte_noun():
    """Verify that generic nouns like 'Electrolyte', 'Additive', 'Solution' are rejected as additive names."""
    extractor = PaperExtractor(api_key="AIzaSyTestKeyDummy")
    mock_client = MagicMock()
    extractor._genai_client = mock_client

    mock_papers = [
        PaperMetadata(
            paper_id="paper_generic_electrolyte",
            title="Study of Electrolyte Properties in ZIB",
            abstract="Study of aqueous electrolyte."
        )
    ]

    mock_response = MagicMock()
    mock_response.text = json.dumps([
        {
            "paper_index": 0,
            "has_additive": True,
            "additives": [
                {
                    "name": "Electrolyte",
                    "iupac_name": None,
                    "smiles": None,
                    "optimal_concentration": None,
                    "concentration_range_tested": [],
                    "functional_groups": ["electrolyte"],
                    "primary_mechanisms": [],
                    "mechanism_summary": "General electrolyte",
                    "battery_context": {"baseline_electrolyte": "Unspecified / Not reported", "solvent": "H2O"},
                    "confidence_score": 0.5,
                    "extraction_evidence": "electrolyte"
                }
            ]
        }
    ])
    mock_client.models.generate_content.return_value = mock_response

    results = extractor.extract_batch_from_papers(mock_papers, batch_size=5)

    assert len(results) == 1
    assert results[0]["additive"] is None


def test_generate_mined_literature_csv():
    from export_engine import generate_mined_literature_csv
    from models import ExtractedAdditive, Concentration, ZincIonBatteryContext

    mock_rec = {
        "paper": {
            "title": "Trehalose Additive for Aqueous Zinc Batteries",
            "authors": ["Alice Smith", "Bob Jones"],
            "year": 2024,
            "venue": "Energy Storage Materials",
            "doi": "10.1016/j.ensm.2024.01.001",
            "open_access_pdf_url": "https://example.com/paper.pdf",
            "citation_count": 42,
            "abstract": "We investigate trehalose as an electrolyte additive in 2 M ZnSO4.",
        },
        "additive": ExtractedAdditive(
            name="Trehalose",
            smiles="C(C1C(C(C(C(O1)OC2C(C(C(C(CO)O2)O)O)O)O)O)O)O",
            optimal_concentration=Concentration(value=10.0, unit="mM", raw_text="10 mM"),
            battery_context=ZincIonBatteryContext(baseline_electrolyte="2 M ZnSO4", solvent="H2O"),
            primary_mechanisms=["Zn2+ solvation sheath modulation", "HER suppression"],
            mechanism_summary="Modulates solvation sheath and suppresses hydrogen evolution.",
            confidence_score=0.96,
            extraction_evidence="Optimal concentration of 10 mM trehalose enables reversible plating."
        )
    }

    csv_output = generate_mined_literature_csv([mock_rec])
    assert "Paper Title" in csv_output
    assert "Trehalose Additive for Aqueous Zinc Batteries" in csv_output
    assert "Trehalose" in csv_output
    assert "10.1016/j.ensm.2024.01.001" in csv_output
    assert "2 M ZnSO4" in csv_output
    assert "10" in csv_output


def test_generate_mined_literature_xlsx():
    import zipfile
    import io
    from export_engine import generate_mined_literature_xlsx
    from models import ExtractedAdditive, Concentration, ZincIonBatteryContext

    mock_rec = {
        "paper": {
            "title": "TEABr Stabilized Zinc Anodes",
            "authors": ["Mark Cui"],
            "year": 2025,
            "venue": "Advanced Materials",
            "doi": "10.1002/adma.202500001",
            "citation_count": 15,
            "abstract": "Tetraethylammonium bromide stabilizes the zinc anode interface.",
        },
        "additive": ExtractedAdditive(
            name="Tetraethylammonium bromide",
            smiles="CC[N+](CC)(CC)CC.[Br-]",
            optimal_concentration=Concentration(value=5.0, unit="mM", raw_text="5 mM"),
            battery_context=ZincIonBatteryContext(baseline_electrolyte="1 M Zn(OTf)2", solvent="H2O"),
            primary_mechanisms=["Electrostatic shielding"],
            mechanism_summary="Forms protective electrostatic layer suppressing dendrites.",
            confidence_score=0.98,
            extraction_evidence="5 mM TEABr extends symmetric cell lifespan to 2000 hours."
        )
    }

    xlsx_bytes = generate_mined_literature_xlsx([mock_rec])
    assert len(xlsx_bytes) > 0
    with zipfile.ZipFile(io.BytesIO(xlsx_bytes), "r") as z:
        names = z.namelist()
        assert "[Content_Types].xml" in names
        assert "xl/workbook.xml" in names
        assert "xl/worksheets/sheet1.xml" in names
        assert "xl/worksheets/sheet2.xml" in names


def test_literature_deduplication_logic():
    import re

    existing_papers = [
        {"title": "Trehalose Electrolyte Additive for Zinc Batteries", "doi": "10.1016/j.ensm.2024.001"},
        {"title": "Sorbitol Regulation in Aqueous Zinc Cells", "doi": ""},
    ]
    seen_dois = {p["doi"].lower().strip() for p in existing_papers if p.get("doi")}
    seen_titles = {re.sub(r'[^a-zA-Z0-9]', '', p["title"].lower()) for p in existing_papers}

    incoming_papers = [
        {"title": "Trehalose Electrolyte Additive for Zinc Batteries", "doi": "10.1016/j.ensm.2024.001"}, # Exact duplicate DOI
        {"title": "sorbitol regulation in aqueous zinc cells!", "doi": "10.1016/j.ensm.2024.999"},       # Duplicate title
        {"title": "Thiourea Additive for Uniform Plating", "doi": "10.1021/acsenergylett.4c0001"},        # New paper
    ]

    new_to_extract = []
    for p in incoming_papers:
        doi_k = (p.get("doi") or "").lower().strip()
        title_k = re.sub(r'[^a-zA-Z0-9]', '', p["title"].lower())
        if doi_k and doi_k in seen_dois:
            continue
        if title_k and title_k in seen_titles:
            continue
        new_to_extract.append(p)
        if doi_k:
            seen_dois.add(doi_k)
        if title_k:
            seen_titles.add(title_k)

    assert len(new_to_extract) == 1
    assert new_to_extract[0]["title"] == "Thiourea Additive for Uniform Plating"












