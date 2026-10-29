"""
Literature Discovery and Extraction Engine.
Module 1: Materials Acceleration Platform (MAP) for Zinc-Ion Batteries (ZIB).

Designed for Self-Driving Laboratories (SDLs) to:
1. Query open-access literature via Semantic Scholar API
2. Extract text from academic PDFs
3. Extract tested electrolyte additives, optimal concentrations, mechanisms, and metrics via LLM
4. Generate structurally/chemically similar candidate molecules using RDKit for robotic screening campaigns.
"""

from __future__ import annotations

import os
import sys
import re
import json
import logging
import argparse
import textwrap
from typing import List, Optional, Dict, Any, Tuple

from models import (
    PaperMetadata,
    ExtractedAdditive,
    CandidateMolecule,
    ExtractionResult,
    Concentration,
    ZincIonBatteryContext,
    PerformanceMetrics,
    CyclingCondition
)
try:
    from discovery.similarity import RDKitSimilarityEngine
    from discovery.search import SemanticScholarClient
    from discovery.extractor import PaperExtractor, GENERIC_NON_ADDITIVES, clean_llm_json_output
    from discovery.formula_utils import get_chemical_formula, format_chem_formula, normalize_condition_key, convert_concentration_to_mM
except ImportError:
    from similarity import RDKitSimilarityEngine
    from search import SemanticScholarClient
    from extractor import PaperExtractor, GENERIC_NON_ADDITIVES, clean_llm_json_output
    from formula_utils import get_chemical_formula, format_chem_formula, normalize_condition_key, convert_concentration_to_mM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LiteratureDiscoveryEngine")


def parse_llm_response(raw_response: str) -> Dict[str, Any]:
    """
    Safely parse raw LLM output into a dictionary, stripping markdown code fences.
    """
    cleaned = clean_llm_json_output(raw_response)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        return {"additives": [], "summary": f"Failed to parse JSON response: {e}"}


def extract_additive_info(raw_response: str) -> Dict[str, Any]:
    """Alias for parse_llm_response."""
    return parse_llm_response(raw_response)


class LiteratureDiscoveryEngine:
    """
    Orchestrates the entire discovery, extraction, and candidate recommendation pipeline.
    """

    def parse_llm_response(self, raw_response: str) -> Dict[str, Any]:
        """Safely parse raw LLM output into a dictionary, stripping markdown code fences."""
        return parse_llm_response(raw_response)

    def extract_additive_info(self, raw_response: str) -> Dict[str, Any]:
        """Alias for parse_llm_response."""
        return parse_llm_response(raw_response)

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        semantic_scholar_api_key: Optional[str] = None,
        candidate_pool: Optional[List[Dict[str, str]]] = None,
        llm_model: str = "gemini-3.5-flash-lite"
    ):
        self.search_client = SemanticScholarClient(api_key=semantic_scholar_api_key)
        self.similarity_engine = RDKitSimilarityEngine(candidate_pool=candidate_pool)
        self.extractor = PaperExtractor(api_key=gemini_api_key, model_name=llm_model)

    def search_literature(
        self,
        query: str = '"zinc-ion battery" "electrolyte additive"',
        limit: int = 5,
        year: Optional[str] = None,
        sort: str = "relevance",
        batch_size: int = 25,
        pause_seconds: float = 0.3,
        batch_callback: Optional[Any] = None
    ) -> List[PaperMetadata]:
        """Search academic literature for papers on zinc-ion battery additives."""
        logger.info(f"Querying literature with: '{query}' (limit={limit}, sort={sort})")
        papers = self.search_client.search_papers(
            query=query,
            limit=limit,
            year=year,
            sort=sort,
            batch_size=batch_size,
            pause_seconds=pause_seconds,
            batch_callback=batch_callback
        )
        logger.info(f"Discovered {len(papers)} papers from academic search.")
        return papers

    def generate_two_tier_candidates(
        self,
        additives: List[ExtractedAdditive],
        threshold: float = 0.15,
        top_n_analogues_per_scaffold: int = 4,
        use_pubchem: bool = True
    ) -> Tuple[List[CandidateMolecule], List[CandidateMolecule]]:
        """
        Partition additive knowledge into two strictly separated tiers:
        Tier 1 (📚 Mined Literature Additives): All unique validated extracted additives from the literature.
               Never truncated to 8. Includes reported concentrations, baseline salts, mechanisms,
               chemical formulas, and Delaney ESOL limits.
        Tier 2 (✨ AI-Suggested Untested Additives): Strictly distinct candidate molecules generated via
               RDKit Morgan fingerprints (2048-bit, radius 2) and live PubChem similarity matching anchored
               to the mined literature scaffolds. Excludes any reported literature entity.
        """
        from discovery.similarity import calculate_delaney_esol, assess_electrolyte_compatibility
        from rdkit import Chem

        tier1_candidates: List[CandidateMolecule] = []
        mined_canon_smiles = set()
        mined_names_lower = set()
        mined_condition_keys = set()

        for item in additives:
            if item is None:
                continue

            if isinstance(item, CandidateMolecule):
                name_clean = item.name.strip()
                name_lower = name_clean.lower()
                conc_val = item.reported_concentration if item.reported_concentration is not None else 15.0
                salt_val = item.parent_baseline_electrolyte or "2 M ZnSO4"
                cond_key = normalize_condition_key(name_clean, conc_val, salt_val)
                if cond_key in mined_condition_keys:
                    continue
                tier1_candidates.append(item)
                mined_condition_keys.add(cond_key)
                mined_names_lower.add(name_lower)
                if item.smiles:
                    mined_canon_smiles.add(item.smiles)
                continue

            if isinstance(item, str):
                name_clean = item.strip()
                if not name_clean or name_clean.lower() in GENERIC_NON_ADDITIVES:
                    continue
                name_lower = name_clean.lower()
                cond_key = normalize_condition_key(name_clean, 15.0, "2 M ZnSO4")
                if cond_key in mined_condition_keys:
                    continue
                smi = self.similarity_engine.resolve_chemical_name_to_smiles(name_clean)
                mol = Chem.MolFromSmiles(smi) if smi else None
                canon_smi = Chem.MolToSmiles(mol) if mol else (smi or "")
                desc = self.similarity_engine.calculate_descriptors(canon_smi) if canon_smi else None
                esol = calculate_delaney_esol(mol) if mol else None
                formula_str = get_chemical_formula(canon_smi or smi, name_clean)
                cand1 = CandidateMolecule(
                    name=name_clean,
                    smiles=canon_smi or smi or "",
                    similarity_score=1.0,
                    similarity_metric="Direct Literature Finding",
                    molecular_weight=desc["molecular_weight"] if desc else 0.0,
                    logp=desc["logp"] if desc else 0.0,
                    logs=esol,
                    tpsa=desc["tpsa"] if desc else 0.0,
                    hbd=desc["hbd"] if desc else 0,
                    hba=desc["hba"] if desc else 0,
                    rotatable_bonds=desc["rotatable_bonds"] if desc else 0,
                    rationale=f"Direct literature finding: {name_clean}.",
                    sdl_formulation_notes="Imported / mined study.",
                    parent_baseline_electrolyte="Unspecified / Not reported",
                    electrolyte_compatibility_note="Baseline unassessed",
                    electrolyte_compatibility_breakdown={},
                    chemical_formula=formula_str,
                    reported_concentration=None,
                    reported_concentration_unit=None,
                    reported_mechanisms=[],
                    mechanism_summary="Unspecified / Not reported",
                    is_literature_additive=True
                )
                tier1_candidates.append(cand1)
                mined_condition_keys.add(cond_key)
                mined_names_lower.add(name_lower)
                if canon_smi:
                    mined_canon_smiles.add(canon_smi)
                continue

            additive = item
            name_clean = getattr(additive, "name", "").strip()
            if not name_clean or name_clean.lower() in GENERIC_NON_ADDITIVES:
                continue
            name_lower = name_clean.lower()

            bc = getattr(additive, "battery_context", None)
            base_salt = bc.baseline_electrolyte if (bc and getattr(bc, "baseline_electrolyte", None)) else getattr(additive, "parent_baseline_electrolyte", "Unspecified / Not reported")
            opt_c = getattr(additive, "optimal_concentration", None)
            raw_conc_val = float(opt_c.value) if (opt_c and getattr(opt_c, "value", None) is not None) else getattr(additive, "reported_concentration", None)
            raw_conc_unit = getattr(opt_c, "unit", None) if opt_c else getattr(additive, "reported_concentration_unit", "mM")
            raw_conc_text = getattr(opt_c, "raw_text", None) if opt_c else None

            smi = getattr(additive, "smiles", None)
            if not smi:
                smi = self.similarity_engine.resolve_chemical_name_to_smiles(name_clean)

            mol = Chem.MolFromSmiles(smi) if smi else None
            canon_smi = Chem.MolToSmiles(mol) if mol else (smi or "")

            desc = self.similarity_engine.calculate_descriptors(canon_smi) if canon_smi else None
            mw_val = desc["molecular_weight"] if desc else 0.0
            conc_val, conc_unit = convert_concentration_to_mM(raw_conc_val, raw_conc_unit, mw=mw_val, raw_text=raw_conc_text)

            cond_key = normalize_condition_key(name_clean, conc_val, base_salt)
            if cond_key in mined_condition_keys:
                continue

            esol = calculate_delaney_esol(mol) if mol else None
            compat_note, compat_breakdown = assess_electrolyte_compatibility(
                mol=mol,
                baseline_electrolyte=base_salt
            ) if mol else ("Baseline unassessed", {})

            if opt_c and getattr(opt_c, "raw_text", None):
                conc_label = f" ({opt_c.raw_text})"
            elif conc_val is not None:
                conc_label = f" ({conc_val:.1f} {conc_unit})"
            else:
                conc_label = ""
            formula_str = get_chemical_formula(canon_smi or smi, name_clean)

            cand1 = CandidateMolecule(
                name=name_clean,
                smiles=(smi if mol else None) or canon_smi or (smi or ""),
                similarity_score=1.0,
                similarity_metric="Direct Literature Finding",
                molecular_weight=mw_val,
                logp=desc["logp"] if desc else 0.0,
                logs=esol,
                tpsa=desc["tpsa"] if desc else 0.0,
                hbd=desc["hbd"] if desc else 0,
                hba=desc["hba"] if desc else 0,
                rotatable_bonds=desc["rotatable_bonds"] if desc else 0,
                rationale=f"Direct literature finding: {name_clean}{conc_label} in {base_salt}.",
                sdl_formulation_notes="Mined from peer-reviewed ZIB study.",
                parent_baseline_electrolyte=base_salt,
                electrolyte_compatibility_note=compat_note,
                electrolyte_compatibility_breakdown=compat_breakdown,
                chemical_formula=formula_str,
                reported_concentration=conc_val,
                reported_concentration_unit=conc_unit if conc_val is not None else None,
                reported_mechanisms=getattr(additive, "primary_mechanisms", None) or getattr(additive, "reported_mechanisms", None) or [],
                mechanism_summary=getattr(additive, "mechanism_summary", "") or "Unspecified / Not reported",
                is_literature_additive=True
            )
            tier1_candidates.append(cand1)
            mined_condition_keys.add(cond_key)
            mined_names_lower.add(name_lower)
            if canon_smi:
                mined_canon_smiles.add(canon_smi)

        # Generate Tier 2 AI-suggested novel analogues anchored to the mined literature scaffolds
        tier2_dict: Dict[str, CandidateMolecule] = {}

        for t1_cand in tier1_candidates:
            if not t1_cand.smiles:
                continue
            similar_hits = self.similarity_engine.find_similar_candidates(
                query_smiles=t1_cand.smiles,
                top_n=top_n_analogues_per_scaffold,
                threshold=threshold,
                baseline_electrolyte=t1_cand.parent_baseline_electrolyte,
                use_pubchem=use_pubchem,
                parent_name=t1_cand.name,
            )
            for hit in similar_hits:
                hit_name_lower = hit.name.strip().lower()
                # Strictly exclude any reported literature entity
                if hit_name_lower in mined_names_lower:
                    continue

                hit_mol = Chem.MolFromSmiles(hit.smiles) if hit.smiles else None
                hit_canon = Chem.MolToSmiles(hit_mol) if hit_mol else hit.smiles
                if not hit_canon or hit_canon in mined_canon_smiles:
                    continue

                if hit.similarity_score >= 1.0 or "Direct Literature Finding" in hit.similarity_metric:
                    continue

                # Ensure chemical formula is set and marked as AI untested analogue
                hit.chemical_formula = get_chemical_formula(hit_canon, hit.name)
                hit.is_literature_additive = False
                hit.parent_baseline_electrolyte = t1_cand.parent_baseline_electrolyte
                hit.parent_scaffold_name = t1_cand.name
                hit.parent_scaffold_smiles = t1_cand.smiles
                hit.parent_scaffold_similarity = hit.similarity_score

                if hit_canon not in tier2_dict or hit.similarity_score > tier2_dict[hit_canon].similarity_score:
                    tier2_dict[hit_canon] = hit

        tier2_candidates = list(tier2_dict.values())
        tier2_candidates.sort(key=lambda c: (c.similarity_score, -(c.logs or -99)), reverse=True)

        return tier1_candidates, tier2_candidates

    def recommend_candidates_from_papers(
        self,
        papers: List[PaperMetadata],
        top_n: Optional[int] = None,
        threshold: float = 0.15,
        use_pubchem: bool = True,
        extracted_additives: Optional[List[ExtractedAdditive]] = None
    ) -> List[CandidateMolecule]:
        """
        Extract reported additives from a batch of papers and recommend a ranked pool
        of candidate molecules (both the reported entities and structurally similar analogues).
        """
        if extracted_additives is not None:
            return self.recommend_candidates_from_additives(
                additives=extracted_additives,
                top_n=top_n,
                threshold=threshold,
                use_pubchem=use_pubchem
            )

        batch_records = self.extractor.extract_batch_from_papers(papers)
        additives_found = [rec["additive"] for rec in batch_records if rec.get("additive")]
        return self.recommend_candidates_from_additives(
            additives=additives_found,
            top_n=top_n,
            threshold=threshold,
            use_pubchem=use_pubchem
        )

    def recommend_candidates_from_additives(
        self,
        additives: List[ExtractedAdditive],
        top_n: Optional[int] = None,
        threshold: float = 0.15,
        source_label: Optional[str] = None,
        use_pubchem: bool = True
    ) -> List[CandidateMolecule]:
        """
        Extract reported additives and recommend a ranked pool of candidate molecules
        (both direct findings and structurally similar analogues computed via RDKit Morgan fingerprints).
        """
        all_candidates: List[CandidateMolecule] = []
        seen_smiles = set()

        valid_additives = [
            a for a in additives
            if a is not None and getattr(a, "name", None) and a.name.lower().strip() not in GENERIC_NON_ADDITIVES
        ]

        for additive in valid_additives:
            smi = additive.smiles
            if not smi:
                smi = self.similarity_engine.resolve_chemical_name_to_smiles(additive.name)
            if smi and smi not in seen_smiles:
                desc = self.similarity_engine.calculate_descriptors(smi)
                if desc:
                    from discovery.similarity import calculate_delaney_esol, assess_electrolyte_compatibility
                    from rdkit import Chem
                    mol = Chem.MolFromSmiles(smi)
                    esol = calculate_delaney_esol(mol) if mol else -2.0
                    base_salt = additive.battery_context.baseline_electrolyte if (additive.battery_context and additive.battery_context.baseline_electrolyte) else "Unspecified / Not reported"
                    compat_note, compat_breakdown = assess_electrolyte_compatibility(
                        mol=mol,
                        baseline_electrolyte=base_salt
                    )
                    opt_c_rec = additive.optimal_concentration
                    if opt_c_rec and opt_c_rec.value is not None:
                        conc_val, conc_unit = convert_concentration_to_mM(
                            opt_c_rec.value,
                            opt_c_rec.unit,
                            mw=desc["molecular_weight"],
                            raw_text=opt_c_rec.raw_text,
                        )
                        conc_label = f" ({opt_c_rec.raw_text})" if opt_c_rec.raw_text else f" ({conc_val:.1f} {conc_unit})"
                    else:
                        conc_val = None
                        conc_unit = None
                        conc_label = ""
                    formula_str = get_chemical_formula(smi, additive.name)
                    cand = CandidateMolecule(
                        name=additive.name,
                        smiles=smi,
                        similarity_score=1.0,
                        similarity_metric="Direct Literature Finding",
                        molecular_weight=desc["molecular_weight"],
                        logp=desc["logp"],
                        logs=esol,
                        tpsa=desc["tpsa"],
                        hbd=desc["hbd"],
                        hba=desc["hba"],
                        rotatable_bonds=desc["rotatable_bonds"],
                        rationale=source_label or f"Direct literature finding: {additive.name}{conc_label} in {base_salt}.",
                        sdl_formulation_notes="Mined from peer-reviewed ZIB study.",
                        parent_baseline_electrolyte=base_salt,
                        electrolyte_compatibility_note=compat_note,
                        electrolyte_compatibility_breakdown=compat_breakdown,
                        chemical_formula=formula_str,
                        reported_concentration=conc_val,
                        reported_concentration_unit=conc_unit,
                        reported_mechanisms=additive.primary_mechanisms or [],
                        mechanism_summary=additive.mechanism_summary or "",
                        is_literature_additive=True,
                    )
                    all_candidates.append(cand)
                    seen_smiles.add(smi)

            sim_n = top_n if (top_n is not None and top_n > 0) else 5
            similar = self.recommend_candidates_for_additive(additive, top_n=sim_n, threshold=threshold, use_pubchem=use_pubchem)
            for cand in similar:
                if cand.smiles not in seen_smiles:
                    cand.chemical_formula = get_chemical_formula(cand.smiles, cand.name)
                    all_candidates.append(cand)
                    seen_smiles.add(cand.smiles)

        all_candidates.sort(key=lambda c: (c.similarity_score, -(c.logs or -99)), reverse=True)
        if top_n is not None:
            return all_candidates[:top_n]
        return all_candidates

    def recommend_candidates_for_additive(
        self,
        additive: ExtractedAdditive,
        top_n: int = 5,
        threshold: float = 0.15,
        use_pubchem: bool = True
    ) -> List[CandidateMolecule]:
        """
        Use RDKit cheminformatics to find candidate molecules similar to the extracted additive.
        If SMILES is absent, attempts to match known common names in the curated pool or PubChem.
        """
        query_smiles = additive.smiles

        # Fallback name-matching or PubChem resolution if LLM did not provide SMILES
        if not query_smiles:
            query_smiles = self.similarity_engine.resolve_chemical_name_to_smiles(additive.name)

        if not query_smiles:
            logger.warning(f"No valid SMILES found for additive '{additive.name}'; cannot compute RDKit similarity.")
            return []

        baseline = None
        if additive.battery_context and additive.battery_context.baseline_electrolyte:
            baseline = additive.battery_context.baseline_electrolyte

        candidates = self.similarity_engine.find_similar_candidates(
            query_smiles=query_smiles,
            top_n=top_n,
            threshold=threshold,
            baseline_electrolyte=baseline,
            use_pubchem=use_pubchem,
            parent_name=additive.name,
        )
        for cand in candidates:
            cand.parent_scaffold_name = additive.name
            cand.parent_scaffold_smiles = query_smiles
            cand.parent_scaffold_similarity = cand.similarity_score
        return candidates

    def process_paper_text(
        self,
        paper_metadata: PaperMetadata,
        text: str,
        top_n_candidates: int = 5
    ) -> ExtractionResult:
        """
        Process raw manuscript text: extract additive details and suggest similar candidates.
        """
        logger.info(f"Extracting additive knowledge from text for paper: {paper_metadata.title[:60]}...")
        additives, summary = self.extractor.extract_from_text(text)

        all_candidates: List[CandidateMolecule] = []
        seen_smiles = set()

        for additive in additives:
            candidates = self.recommend_candidates_for_additive(additive, top_n=top_n_candidates)
            for cand in candidates:
                if cand.smiles not in seen_smiles:
                    all_candidates.append(cand)
                    seen_smiles.add(cand.smiles)

        return ExtractionResult(
            paper_metadata=paper_metadata,
            additives=additives,
            candidate_molecules=all_candidates,
            extraction_summary=summary,
            source_type="text"
        )

    def process_paper_pdf(
        self,
        pdf_path: Any,
        paper_metadata: Optional[PaperMetadata] = None,
        max_pages: Optional[int] = None,
        top_n_candidates: int = 5
    ) -> ExtractionResult:
        """
        Process a local PDF file, path, or stream: extract text, parse additives via LLM, and recommend candidate molecules.
        """
        if isinstance(pdf_path, (str, os.PathLike)):
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF not found: {pdf_path}")
            pdf_title = os.path.basename(str(pdf_path))
            pdf_id = os.path.splitext(pdf_title)[0]
        else:
            pdf_title = getattr(pdf_path, "name", "uploaded_paper.pdf")
            pdf_id = os.path.splitext(pdf_title)[0]

        logger.info(f"Reading PDF from: {pdf_title}")
        text = self.extractor.extract_text_from_pdf(pdf_path, max_pages=max_pages)

        if not paper_metadata:
            paper_metadata = PaperMetadata(
                paper_id=pdf_id,
                title=pdf_title,
                abstract=text[:500] if text else "Extracted from local PDF file."
            )

        result = self.process_paper_text(
            paper_metadata=paper_metadata,
            text=text,
            top_n_candidates=top_n_candidates
        )
        result.source_type = "pdf"
        return result

    def process_multiple_pdfs(
        self,
        pdf_paths: List[Any],
        max_pages: Optional[int] = None,
        top_n_candidates: int = 5
    ) -> List[ExtractionResult]:
        """
        Process multiple local PDF files, directories, or uploaded streams in batch:
        extracts text, parses additives via LLM, and recommends candidate molecules for each.
        """
        resolved: List[Any] = []
        for p in pdf_paths:
            if isinstance(p, (str, os.PathLike)):
                if os.path.isdir(p):
                    for f in sorted(os.listdir(p)):
                        if f.lower().endswith(".pdf"):
                            resolved.append(os.path.join(p, f))
                elif os.path.isfile(p):
                    resolved.append(p)
                else:
                    logger.warning(f"PDF path not found or invalid: {p}")
            else:
                # Stream, BytesIO, or UploadedFile
                resolved.append(p)

        results: List[ExtractionResult] = []
        for pdf_file in resolved:
            f_display = os.path.basename(str(pdf_file)) if isinstance(pdf_file, (str, os.PathLike)) else getattr(pdf_file, "name", "stream.pdf")
            logger.info(f"Processing PDF batch item: {f_display}")
            res = self.process_paper_pdf(
                pdf_path=pdf_file,
                max_pages=max_pages,
                top_n_candidates=top_n_candidates
            )
            results.append(res)
        return results

    def run_demo_pipeline(self) -> ExtractionResult:
        """
        Demonstration pipeline simulating end-to-end literature extraction
        and candidate generation for a high-impact ZIB paper.
        """
        demo_paper = PaperMetadata(
            paper_id="demo_zib_2024",
            title="Polyol-Mediated Solvation Sheath Regulation for Highly Reversible Zinc Metal Anodes",
            authors=["Jian Zhang", "Wei Sun", "Linda Chen"],
            year=2024,
            venue="Nature Energy",
            doi="10.1038/s41560-024-00000-0",
            abstract=(
                "Aqueous zinc batteries are hindered by hydrogen evolution and dendrites. "
                "Here, we reveal that 15 mM D-sorbitol effectively modifies the Zn²⁺ solvation structure, "
                "yielding 2500 hours of dendrite-free Zn plating/stripping at 2 mA cm-2."
            )
        )

        demo_manuscript_text = (
            "Aqueous zinc-ion batteries (ZIBs) represent a promising grid-scale energy storage technology, "
            "yet suffer from parasitic hydrogen evolution reaction (HER) and uncontrollable dendritic growth. "
            "Here, we introduce D-sorbitol as an electrolyte additive into a standard 2 M ZnSO4 aqueous electrolyte. "
            "Through systematic concentration screening (5 mM, 10 mM, 15 mM, 30 mM), we determined that 15 mM D-sorbitol "
            "exhibits optimal performance. The rich hydroxyl groups (-OH) of D-sorbitol reconstruct the Zn²⁺ solvation sheath "
            "by partially displacing coordinated active H2O molecules, elevating the HER overpotential and guiding preferred (002) "
            "crystallographic Zn deposition. Consequently, the Zn//Zn symmetric cell demonstrated exceptional cycling stability "
            "exceeding 2500 hours at 2 mA cm-2 and 1 mAh cm-2 with an overpotential of only 28 mV. In Zn//Cu asymmetric cells, "
            "an average Coulombic efficiency of 99.7% was maintained over 1000 cycles. Full cells paired with a V2O5 cathode "
            "retained 92.4% capacity after 800 cycles."
        )

        mock_llm_payload = {
            "additives": [
                {
                    "name": "D-Sorbitol",
                    "iupac_name": "(2R,3R,4R,5S)-hexane-1,2,3,4,5,6-hexol",
                    "smiles": "C(C(C(C(C(CO)O)O)O)O)O",
                    "optimal_concentration": {
                        "value": 15.0,
                        "unit": "mM",
                        "raw_text": "15 mM"
                    },
                    "concentration_range_tested": [
                        {"value": 5.0, "unit": "mM", "raw_text": "5 mM"},
                        {"value": 10.0, "unit": "mM", "raw_text": "10 mM"},
                        {"value": 15.0, "unit": "mM", "raw_text": "15 mM"},
                        {"value": 30.0, "unit": "mM", "raw_text": "30 mM"}
                    ],
                    "functional_groups": ["hydroxyl", "polyol"],
                    "primary_mechanisms": [
                        "Zn²⁺ solvation sheath modulation",
                        "HER suppression",
                        "Preferred (002) texture deposition",
                        "Dendrite inhibition"
                    ],
                    "mechanism_summary": "Displaces active water from Zn²⁺ inner solvation sheath and promotes (002) plane texturing.",
                    "battery_context": {
                        "baseline_electrolyte": "2 M ZnSO4",
                        "solvent": "H2O",
                        "electrolyte_pH": 4.0,
                        "anode_material": "Zn foil",
                        "cathode_material": "V2O5",
                        "cell_format": "Symmetric Zn//Zn & Zn//V2O5 full cell",
                        "temperature_celsius": 25.0
                    },
                    "performance_metrics": {
                        "symmetric_cell_performance": [
                            {
                                "current_density_mA_cm2": 2.0,
                                "areal_capacity_mAh_cm2": 1.0,
                                "lifespan_hours": 2500.0
                            }
                        ],
                        "coulombic_efficiency_percent": 99.7,
                        "capacity_retention_percent": 92.4,
                        "full_cell_cycle_count": 800,
                        "overpotential_mV": 28.0
                    },
                    "confidence_score": 0.98,
                    "extraction_evidence": "Through systematic concentration screening, we determined that 15 mM D-sorbitol exhibits optimal performance."
                }
            ],
            "summary": "15 mM D-sorbitol in 2 M ZnSO4 restructures Zn²⁺ solvation, enabling >2500 h symmetric cell life."
        }

        # If GEMINI_API_KEY is not configured or in demo mode, use high-fidelity domain parser directly
        if not self.extractor._genai_client:
            logger.info("Using domain reference mock for demonstration mode (No GEMINI_API_KEY required).")
            # Directly parse into models
            additives, _ = self.extractor.extract_from_text("")
            # Populate additives from payload
            from unittest.mock import patch
            with patch.object(self.extractor, "_call_llm_structured", return_value=mock_llm_payload):
                return self.process_paper_text(demo_paper, demo_manuscript_text)
        else:
            return self.process_paper_text(demo_paper, demo_manuscript_text)


def format_cli_output(result: ExtractionResult):
    """Render beautiful terminal output for MAP researchers and SDL engineers."""
    print("=" * 80)
    print("MATERIALS ACCELERATION PLATFORM (MAP) - MODULE 1: LITERATURE EXTRACTION")
    print("=" * 80)
    p = result.paper_metadata
    print(f"\n[PAPER METADATA]")
    print(f"  Title:   {p.title}")
    print(f"  Authors: {', '.join(p.authors) if p.authors else 'N/A'}")
    print(f"  Venue:   {p.venue or 'N/A'} ({p.year or 'N/A'})")
    print(f"  DOI:     {p.doi or 'N/A'}")
    print(f"  Source:  {result.source_type.upper()}")

    print(f"\n[EXTRACTED ADDITIVES] (Found: {len(result.additives)})")
    for idx, add in enumerate(result.additives, 1):
        print(f"\n  --- Additive #{idx}: {add.name} ---")
        print(f"    SMILES:          {add.smiles or 'N/A'}")
        print(f"    Optimal Conc:    {add.optimal_concentration.value} {add.optimal_concentration.unit}")
        print(f"    Base Electrolyte:{add.battery_context.baseline_electrolyte} in {add.battery_context.solvent}")
        print(f"    Mechanisms:      {', '.join(add.primary_mechanisms)}")
        print(f"    Mechanism Note:  {add.mechanism_summary}")
        if add.performance_metrics:
            m = add.performance_metrics
            if m.symmetric_cell_performance:
                perf = m.symmetric_cell_performance[0]
                print(f"    Lifespan:        {perf.lifespan_hours or 'N/A'} h (at {perf.current_density_mA_cm2 or 'N/A'} mA/cm2)")
            print(f"    Coulombic Eff:   {m.coulombic_efficiency_percent or 'N/A'}%")
            print(f"    Overpotential:   {m.overpotential_mV or 'N/A'} mV")
        print(f"    Confidence:      {add.confidence_score * 100:.1f}%")
        print(f"    Evidence Quote:  \"{add.extraction_evidence}\"")

    print(f"\n[RDKIT SIMILAR CANDIDATE RECOMMENDATIONS FOR SDL SCREENING] (Count: {len(result.candidate_molecules)})")
    for idx, cand in enumerate(result.candidate_molecules, 1):
        print(f"\n  Candidate #{idx}: {cand.name}")
        print(f"    SMILES:          {cand.smiles}")
        print(f"    Tanimoto Sim:    {cand.similarity_score:.4f} ({cand.similarity_metric})")
        print(f"    MW / LogP / TPSA:{cand.molecular_weight} g/mol | LogP: {cand.logp} | TPSA: {cand.tpsa} A^2")
        print(f"    HBD / HBA / RotB:{cand.hbd} donors | {cand.hba} acceptors | {cand.rotatable_bonds} rotatable")
        print(f"    SDL Rationale:   {cand.rationale}")
        print(f"    Formulation Note:{cand.sdl_formulation_notes}")
        if cand.parent_baseline_electrolyte:
            print(f"    Parent Baseline: {cand.parent_baseline_electrolyte}")
        if cand.electrolyte_compatibility_note:
            print(f"    Compatibility:   {cand.electrolyte_compatibility_note}")
        if cand.electrolyte_compatibility_breakdown:
            print("    Multi-Electrolyte Compatibility Breakdown:")
            for salt, detail in cand.electrolyte_compatibility_breakdown.items():
                print(f"      * {salt:<13} [{detail.status:<18} | {detail.expected_effectiveness} Eff]: {detail.notes}")

    print("\n" + "=" * 80)
    print("STATUS: Machine-actionable JSON payload ready for SDL Formulation Robot.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Zinc-Ion Battery Literature Discovery & Extraction Engine (MAP Module 1)"
    )
    parser.add_argument("--demo", action="store_true", help="Run demonstration pipeline with sample high-impact paper")
    parser.add_argument("--search", type=str, help="Search query for Semantic Scholar API")
    parser.add_argument("--limit", type=int, default=3, help="Max papers to search")
    parser.add_argument("--export-abstracts", type=str, help="Export searched paper titles and abstracts to a specified JSON file path")
    parser.add_argument("--pdf", nargs="+", help="Path(s) to one or more PDF files or directories for batch text extraction & LLM parsing")
    parser.add_argument("--json", action="store_true", help="Output raw JSON payload")
    args = parser.parse_args()

    engine = LiteratureDiscoveryEngine()

    if args.demo or (len(sys.argv) == 1):
        result = engine.run_demo_pipeline()
        if args.json:
            print(result.model_dump_json(indent=2))
        else:
            format_cli_output(result)
        return

    if args.search:
        papers = engine.search_literature(query=args.search, limit=args.limit)
        
        if args.export_abstracts:
            export_data = [{"title": p.title, "abstract": p.abstract} for p in papers]
            try:
                with open(args.export_abstracts, "w") as f:
                    json.dump(export_data, f, indent=2)
                print(f"\nSuccessfully exported {len(papers)} titles and abstracts to {args.export_abstracts}")
            except Exception as e:
                logger.error(f"Failed to export abstracts to {args.export_abstracts}: {e}")

        if args.json:
            print(json.dumps([p.model_dump() for p in papers], indent=2))
        else:
            print(f"\nDiscovered {len(papers)} papers on Semantic Scholar:\n")
            for i, p in enumerate(papers, 1):
                print(f"{i}. [{p.year or 'N/A'}] {p.title}")
                print(f"   Authors: {', '.join(p.authors[:3]) if p.authors else 'N/A'}")
                print(f"   DOI: {p.doi or 'N/A'} | OpenAccess PDF: {p.open_access_pdf_url or 'None'}")
                
                abstract_text = p.abstract if p.abstract else "No abstract available."
                wrapped_abstract = textwrap.fill(
                    abstract_text, 
                    width=100, 
                    initial_indent="   Abstract: ", 
                    subsequent_indent="             "
                )
                print(f"{wrapped_abstract}\n")
        return

    if args.pdf:
        results = engine.process_multiple_pdfs(args.pdf)
        if not results:
            print("No valid PDF files found to process.")
            return

        if len(results) == 1 and len(args.pdf) == 1 and not os.path.isdir(args.pdf[0]):
            if args.json:
                print(results[0].model_dump_json(indent=2))
            else:
                format_cli_output(results[0])
        else:
            if args.json:
                print(json.dumps([r.model_dump() for r in results], indent=2))
            else:
                for idx, r in enumerate(results, 1):
                    print(f"\n{'#' * 80}")
                    print(f"# BATCH PDF INGESTION [{idx}/{len(results)}]: {r.paper_metadata.title}")
                    print(f"{'#' * 80}")
                    format_cli_output(r)
        return


if __name__ == "__main__":
    main()