"""
Autonomous Materials Acceleration Platform (MAP) Orchestrator for Zinc-Ion Batteries (ZIB).
Executes end-to-end sequential workflow:
Phase 1 (Discovery & Literature Mining) ➔
Phase 2 (Electrolyte Formulation & Robotic Dispensing) ➔
Phase 3 (Electrochemical Analysis & Adaptive Reporting)
"""

import os
import sys
import json
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
import numpy as np

from models import (
    ExtractionResult,
    CandidateMolecule,
    ExtractedAdditive,
    Concentration,
    ZincIonBatteryContext,
    ConcentrationAnchor,
    DoEPlan,
    DispensingSchedule,
    ElectrochemicalAnalysisReport,
    CampaignResult,
    FormulatedCellResult,
    PaperMetadata,
)
from optimization.concentration_anchor import ConcentrationAnchorEngine, parse_salt_string
from optimization.doe_engine import DoEEngine
from optimization.dispenser import RoboticDispenserEngine
from analysis.analyzer_cli import run_analysis
from discovery.literature_engine import LiteratureDiscoveryEngine
from discovery.similarity import CURATED_ZIB_ADDITIVES

logger = logging.getLogger("MAPOrchestrator")


def create_benchmark_cycler_file(
    filepath: Union[str, Path],
    duration_hours: float = 100.0,
    short_at_hour: Optional[float] = 60.0,
    plateau_voltage_v: float = 0.035,
    current_ma: float = 1.13097,
    point_interval_s: float = 60.0,
    plating_stripping_period_s: float = 7200.0,
) -> Path:
    """
    Generate synthetic Zn||Zn symmetric cell cycling data in standard Landt/Arbin CSV format.

    Args:
        filepath: Destination path for the CSV file.
        duration_hours: Total nominal testing duration in hours.
        short_at_hour: Hour at which dendrite short-circuit flatline commences.
        plateau_voltage_v: Baseline stripping/plating plateau voltage magnitude (V).
        current_ma: Applied cycling current (mA). Standard ~1.131 mA for 1 mA/cm2 on 12mm disc.
        point_interval_s: Sampling interval in seconds (default 60s).
        plating_stripping_period_s: Full cycle period in seconds (default 7200s = 1h strip, 1h plate).

    Returns:
        Path to generated CSV file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    total_seconds = int(duration_hours * 3600.0)
    times = np.arange(0, total_seconds, point_interval_s, dtype=float)
    half_period_s = plating_stripping_period_s / 2.0

    # Phase: True for stripping (+current), False for plating (-current)
    phase = (times % plating_stripping_period_s) < half_period_s

    current = np.where(phase, current_ma, -current_ma)
    voltage = np.where(phase, plateau_voltage_v, -plateau_voltage_v)

    # Slight random potential fluctuations (+/- 0.5 mV) using a local deterministic RNG per sample
    import zlib
    seed_key = f"{path.name}_{duration_hours:.2f}_{short_at_hour or 0.0:.2f}_{plateau_voltage_v:.4f}"
    rng = np.random.default_rng(zlib.adler32(seed_key.encode("utf-8")))
    noise = rng.normal(0, 0.0005, size=len(times))
    voltage += noise

    if short_at_hour is not None:
        short_s = short_at_hour * 3600.0
        short_mask = times >= short_s
        # Short circuit drops voltage to near zero (< 2 mV)
        voltage[short_mask] = 0.001 * np.where(phase[short_mask], 1.0, -1.0) + noise[short_mask] * 0.1

    # Cumulative capacity in mAh
    capacity_step = (np.abs(current) * point_interval_s) / 3600.0
    cumulative_capacity = np.cumsum(capacity_step)

    lines = ["Time(s),Voltage(V),Current(mA),Capacity(mAh)\n"]
    for t, v, i, c in zip(times, voltage, current, cumulative_capacity):
        lines.append(f"{t:.1f},{v:.6f},{i:.4f},{c:.4f}\n")

    path.write_text("".join(lines), encoding="utf-8")
    return path


class MAPOrchestrator:
    """
    End-to-End Orchestrator for the autonomous Materials Acceleration Platform (MAP).
    Seamlessly links Discovery ➔ Formulation ➔ Characterization.
    """

    def __init__(self, output_dir: Union[str, Path] = "./campaign_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.anchor_engine = ConcentrationAnchorEngine()
        self.doe_engine = DoEEngine()
        self.dispenser_engine = RoboticDispenserEngine()

    def run_discovery(
        self,
        query: Optional[str] = None,
        from_extraction: Optional[Union[str, Path]] = None,
        candidate_pool: Optional[List[Dict[str, str]]] = None,
        limit: int = 5,
    ) -> ExtractionResult:
        """
        Execute Phase 1 discovery or ingest an existing extraction JSON file.

        Args:
            query: Semantic literature search query.
            from_extraction: Path to pre-existing ExtractionResult JSON file.
            candidate_pool: Optional custom candidate molecule pool.
            limit: Number of literature papers to query if searching online.

        Returns:
            ExtractionResult instance.
        """
        if from_extraction:
            ext_path = Path(from_extraction)
            if not ext_path.is_file():
                raise FileNotFoundError(f"Extraction file not found: {ext_path}")
            with open(ext_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ExtractionResult.model_validate(data)

        if not query:
            raise ValueError("Must provide either 'query' or 'from_extraction' to run discovery.")

        logger.info(f"Running autonomous literature discovery for query: '{query}'")
        engine = LiteratureDiscoveryEngine(candidate_pool=candidate_pool)
        papers = engine.search_literature(query=query, limit=limit)

        if papers:
            paper = papers[0]
            text = f"{paper.title}\n{paper.abstract or ''}"
            try:
                res = engine.process_paper_text(paper_metadata=paper, text=text)
                if res.additives or res.candidate_molecules:
                    return res
            except Exception as e:
                logger.warning(f"Paper extraction encountered error: {e}; falling back to curated pool.")

        # Fallback to curated candidates
        logger.info("Semantic search returned no extractable additives; using curated ZIB additive pool.")
        q_lower = query.lower()
        matched = [
            m for m in CURATED_ZIB_ADDITIVES
            if q_lower in m["name"].lower() or any(w in m["name"].lower() for w in q_lower.split())
        ]
        if not matched:
            matched = CURATED_ZIB_ADDITIVES[:3]

        candidates = []
        for item in matched:
            cand = CandidateMolecule(
                name=item["name"],
                smiles=item["smiles"],
                similarity_score=1.0,
                molecular_weight=180.0,
                logp=-1.5,
                tpsa=100.0,
                hbd=4,
                hba=4,
                rotatable_bonds=2,
                rationale=f"Selected from curated pool matching '{query}'",
                sdl_formulation_notes="Recommended for aqueous ZIB electrolyte formulation",
                parent_baseline_electrolyte="2 M ZnSO4",
            )
            candidates.append(cand)

        first = matched[0]
        additives = [
            ExtractedAdditive(
                name=first["name"],
                smiles=first["smiles"],
                optimal_concentration=Concentration(value=15.0, unit="mM", raw_text="15 mM"),
                concentration_range_tested=[Concentration(value=15.0, unit="mM", raw_text="15 mM")],
                functional_groups=["polyol" if "ol" in first["name"].lower() else "general"],
                primary_mechanisms=["Solvation modulation", "HER suppression"],
                mechanism_summary="Interfacial water displacement and zinc dendrite suppression.",
                battery_context=ZincIonBatteryContext(
                    baseline_electrolyte="2 M ZnSO4",
                    solvent="H2O",
                    anode_material="Zn foil",
                    cell_format="Symmetric",
                ),
                confidence_score=0.9,
                extraction_evidence="Curated additive benchmark baseline.",
            )
        ]
        return ExtractionResult(
            paper_metadata=PaperMetadata(
                paper_id="curated_zib_database",
                title="Curated Aqueous Zinc-Ion Battery Additives Database",
                year=2024,
            ),
            additives=additives,
            candidate_molecules=candidates,
            extraction_summary=f"Discovered {len(additives)} additives and {len(candidates)} candidates matching '{query}'.",
            source_type="curated_database",
        )

    def run_formulation(
        self,
        extraction: Union[ExtractionResult, CandidateMolecule, ExtractedAdditive, Dict[str, Any]],
        target_salt: str = "Zn(OTf)2",
        method: str = "lhs",
        num_samples: int = 9,
        rack_format: str = "24-vial rack",
        total_volume_uL: Optional[float] = None,
        additive_stock_conc_mM: Optional[float] = None,
        salt_stock_conc_M: Optional[float] = None,
        published_conc_override: Optional[float] = None,
        parent_electrolyte_override: Optional[str] = None,
        additive_name_override: Optional[str] = None,
        smiles_override: Optional[str] = None,
        seed: int = 42,
        deck_slot: int = 1,
        aspiration_rate_ul_s: Optional[float] = None,
        dispense_rate_ul_s: Optional[float] = None,
    ) -> Tuple[DoEPlan, DispensingSchedule]:
        """
        Execute Phase 2: salt-aware anchoring, DoE matrix, and robotic dispensing schedule.

        Args:
            extraction: ExtractionResult, CandidateMolecule, or ExtractedAdditive.
            target_salt: Target electrolyte salt (e.g., 'Zn(OTf)2', 'ZnSO4', '5 M ZnCl2').
            method: DoE methodology ('lhs' or 'ccd').
            num_samples: Number of formulation samples.
            rack_format: Robotic deck format ('24-vial rack' or '96-well plate').
            total_volume_uL: Dispense volume per vial (default: 2000 uL for vial, 200 uL for plate).
            additive_stock_conc_mM: Additive stock solution concentration in mM.
            salt_stock_conc_M: Salt stock solution concentration in M.
            published_conc_override: Optional concentration override in mM.
            parent_electrolyte_override: Optional parent electrolyte string override.
            additive_name_override: Optional additive chemical name override.
            smiles_override: Optional SMILES string override.
            seed: Random seed for DoE sampling.
            deck_slot: Assigned OT-2 deck slot (default 1).
            aspiration_rate_ul_s: Optional custom aspiration rate in uL/s.
            dispense_rate_ul_s: Optional custom dispense rate in uL/s.

        Returns:
            Tuple of (DoEPlan, DispensingSchedule).
        """
        additive_name = ""
        smiles = None
        published_conc = 15.0
        parent_electrolyte = "2 M ZnSO4"

        if isinstance(extraction, ExtractionResult):
            if not extraction.additives:
                if extraction.candidate_molecules:
                    cand = extraction.candidate_molecules[0]
                    additive_name = cand.name
                    smiles = cand.smiles
                    parent_electrolyte = cand.parent_baseline_electrolyte or "2 M ZnSO4"
                else:
                    raise ValueError("ExtractionResult contains no additives or candidate molecules.")
            else:
                target_add = extraction.additives[0]
                additive_name = target_add.name
                smiles = target_add.smiles
                published_conc = target_add.optimal_concentration.value if target_add.optimal_concentration is not None else 15.0
                parent_electrolyte = target_add.battery_context.baseline_electrolyte if target_add.battery_context is not None else "2 M ZnSO4"
        elif isinstance(extraction, ExtractedAdditive):
            additive_name = extraction.name
            smiles = extraction.smiles
            published_conc = extraction.optimal_concentration.value if extraction.optimal_concentration is not None else 15.0
            parent_electrolyte = extraction.battery_context.baseline_electrolyte if extraction.battery_context is not None else "2 M ZnSO4"
        elif isinstance(extraction, CandidateMolecule):
            additive_name = extraction.name
            smiles = extraction.smiles
            parent_electrolyte = extraction.parent_baseline_electrolyte or "2 M ZnSO4"
            published_conc = 15.0
        elif isinstance(extraction, dict):
            additive_name = extraction.get("name") or extraction.get("additive_name", "Unknown Additive")
            smiles = extraction.get("smiles")
            opt_c = extraction.get("optimal_concentration")
            if isinstance(opt_c, dict) and "value" in opt_c:
                published_conc = float(opt_c["value"])
            elif "published_concentration" in extraction:
                published_conc = float(extraction["published_concentration"])
            b_ctx = extraction.get("battery_context")
            if isinstance(b_ctx, dict) and "baseline_electrolyte" in b_ctx:
                parent_electrolyte = b_ctx["baseline_electrolyte"]
            elif "parent_electrolyte" in extraction:
                parent_electrolyte = extraction["parent_electrolyte"]

        if additive_name_override is not None:
            additive_name = str(additive_name_override)
        if smiles_override is not None:
            smiles = str(smiles_override)
        if published_conc_override is not None:
            published_conc = float(published_conc_override)
        if parent_electrolyte_override is not None:
            parent_electrolyte = str(parent_electrolyte_override)

        # 1. Salt-Aware Concentration Anchoring & Delaney ESOL Capping
        anchor = self.anchor_engine.calculate_anchor(
            additive_name=additive_name,
            published_conc_mM=published_conc,
            parent_electrolyte=parent_electrolyte,
            target_electrolyte=target_salt,
            smiles=smiles,
        )

        # 2. Multi-Factor DoE Formulation Matrix (LHS / CCD)
        plan = self.doe_engine.create_plan(
            anchor=anchor,
            method=method,
            num_samples=num_samples,
            seed=seed,
        )

        # 3. Robotic Liquid Handling Schedule
        schedule = self.dispenser_engine.generate_schedule(
            plan=plan,
            rack_format=rack_format,
            total_volume_uL=total_volume_uL,
            additive_stock_conc_mM=additive_stock_conc_mM,
            salt_stock_conc_M=salt_stock_conc_M,
            deck_slot=deck_slot,
            aspiration_rate_ul_s=aspiration_rate_ul_s,
            dispense_rate_ul_s=dispense_rate_ul_s,
        )

        return plan, schedule

    def run_analysis(
        self,
        data_source: Union[str, Path],
        technique: str = "auto",
        area_cm2: float = 1.13097,
        sample_id: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        active_mass_mg: Optional[float] = None,
        drop_rest_min: Optional[float] = None,
        no_html: bool = False,
    ) -> ElectrochemicalAnalysisReport:
        """
        Execute Phase 3: ingest cycler data, normalize, analyze failure modes, and generate reports.

        Args:
            data_source: Path to raw cycler file (.csv, .mpt, .xlsx, .mpr).
            technique: 'auto' or explicit technique ('galvanostatic_symmetric', etc.).
            area_cm2: Electrode working area in cm2.
            sample_id: Custom sample identifier (default: file stem).
            output_dir: Target output directory for reports and dashboards.
            active_mass_mg: Active material mass in mg (for full cells).
            drop_rest_min: Rest duration threshold in minutes to strip.
            no_html: If True, skips HTML dashboard rendering.

        Returns:
            ElectrochemicalAnalysisReport instance.
        """
        dest_dir = Path(output_dir) if output_dir else self.output_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        report = run_analysis(
            filepath=data_source,
            technique_str=technique,
            area_cm2=area_cm2,
            active_mass_mg=active_mass_mg,
            drop_rest_min=drop_rest_min,
            output_dir=dest_dir,
            no_html=no_html,
        )

        if sample_id:
            report.sample_id = sample_id

        return report

    def run_full_campaign(
        self,
        query: Optional[str] = None,
        from_extraction: Optional[Union[str, Path]] = None,
        additive_name: Optional[str] = None,
        published_conc_mM: Optional[float] = None,
        smiles: Optional[str] = None,
        parent_electrolyte: str = "2 M ZnSO4",
        target_salt: str = "Zn(OTf)2",
        doe_method: str = "lhs",
        num_samples: int = 9,
        rack_format: str = "24-vial rack",
        total_volume_uL: Optional[float] = None,
        cycler_data: Optional[Union[str, Path]] = None,
        use_benchmark: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
        seed: int = 42,
    ) -> CampaignResult:
        """
        Execute full end-to-end MAP campaign across all three phases:
        Phase 1 (Discovery) ➔ Phase 2 (Formulation & Dispensing) ➔ Phase 3 (Analysis & Reporting).

        Args:
            query: Search query for Phase 1.
            from_extraction: Pre-existing extraction JSON path.
            additive_name: Direct manual additive name.
            published_conc_mM: Direct manual published concentration (mM).
            smiles: Direct manual SMILES string.
            parent_electrolyte: Parent electrolyte string.
            target_salt: Target electrolyte salt system.
            doe_method: 'lhs' or 'ccd'.
            num_samples: Number of formulation candidates.
            rack_format: '24-vial rack' or '96-well plate'.
            total_volume_uL: Target volume per vial in uL.
            cycler_data: Optional experimental cycler file or folder.
            use_benchmark: If True, simulates benchmark cycling runs for each formulated point.
            output_dir: Destination folder for all artifacts.
            seed: Random seed for DoE generation.

        Returns:
            CampaignResult instance.
        """
        target_dir = Path(output_dir) if output_dir else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        campaign_id = f"CAMP_{target_salt}_{uuid.uuid4().hex[:6]}"
        logger.info(f"Starting End-to-End MAP Campaign: {campaign_id}")

        # -------------------------------------------------------------
        # Phase 1: Discovery & Literature Mining
        # -------------------------------------------------------------
        extraction = None
        discovery_summary = None

        if from_extraction:
            extraction = self.run_discovery(from_extraction=from_extraction)
            discovery_summary = {
                "source": str(from_extraction),
                "title": extraction.paper_metadata.title,
                "additives_found": len(extraction.additives),
                "candidates_found": len(extraction.candidate_molecules),
            }
            if additive_name:
                discovery_summary["override_additive_name"] = additive_name
            if published_conc_mM is not None:
                discovery_summary["override_published_conc_mM"] = published_conc_mM
        elif query:
            extraction = self.run_discovery(query=query)
            discovery_summary = {
                "query": query,
                "title": extraction.paper_metadata.title,
                "additives_found": len(extraction.additives),
                "candidates_found": len(extraction.candidate_molecules),
            }
        elif additive_name and published_conc_mM is not None:
            # Synthetic manual extraction structure
            extraction = {
                "name": additive_name,
                "published_concentration": published_conc_mM,
                "parent_electrolyte": parent_electrolyte,
                "smiles": smiles,
            }
            discovery_summary = {
                "manual_input": True,
                "additive_name": additive_name,
                "published_concentration_mM": published_conc_mM,
                "parent_electrolyte": parent_electrolyte,
            }
        else:
            raise ValueError(
                "Either provide extraction data (via 'from_extraction' or 'query') "
                "or both 'additive_name' and 'published_conc_mM'."
            )

        # -------------------------------------------------------------
        # Phase 2: Formulation Optimization & Robotic Dispensing
        # -------------------------------------------------------------
        logger.info("Executing Phase 2: Design of Experiments & Dispensing Schedule Generator")
        has_user_override = (additive_name is not None) or (published_conc_mM is not None) or (smiles is not None)
        plan, schedule = self.run_formulation(
            extraction=extraction,
            target_salt=target_salt,
            method=doe_method,
            num_samples=num_samples,
            rack_format=rack_format,
            total_volume_uL=total_volume_uL,
            published_conc_override=published_conc_mM,
            parent_electrolyte_override=parent_electrolyte if has_user_override else None,
            additive_name_override=additive_name,
            smiles_override=smiles,
            seed=seed,
        )

        # Write Phase 2 artifacts
        plan_path = target_dir / "doe_plan.json"
        plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")

        schedule_path = target_dir / "dispensing_schedule.json"
        schedule_path.write_text(schedule.model_dump_json(indent=2), encoding="utf-8")

        # -------------------------------------------------------------
        # Phase 3: Characterization, Windowing & Aggregation
        # -------------------------------------------------------------
        logger.info("Executing Phase 3: Electrochemical Characterization & Ingestion")
        cell_results: List[FormulatedCellResult] = []

        cycler_dir = target_dir / "cycler_data"
        cycler_dir.mkdir(parents=True, exist_ok=True)

        for i, vial in enumerate(schedule.vials):
            form_pt = plan.formulations[i] if i < len(plan.formulations) else None
            sample_id = form_pt.sample_id if form_pt else vial.vial_id
            point_type = form_pt.point_type if form_pt else "center"
            temp_c = form_pt.temperature_C if form_pt else None
            vial_id = vial.vial_id
            report: Optional[ElectrochemicalAnalysisReport] = None
            performance_score: Optional[float] = None

            cycler_file = None
            if cycler_data:
                cd_path = Path(cycler_data)
                if cd_path.is_file():
                    cycler_file = cd_path
                elif cd_path.is_dir():
                    candidate_files = list(cd_path.glob(f"*{sample_id}*")) or list(cd_path.glob(f"*{vial_id}*"))
                    if candidate_files:
                        cycler_file = candidate_files[0]

            if not cycler_file and use_benchmark:
                # Simulate realistic variation across formulation points:
                # Optimal anchor formulation achieves longer active lifespan (e.g. 100-140 h)
                # Far-off concentrations experience earlier dendrite short-circuits (e.g. 30-70 h)
                anchor_conc = plan.anchor.shifted_anchor_conc
                conc_diff_ratio = abs(vial.additive_conc_mM - anchor_conc) / (anchor_conc if anchor_conc > 0 else 1.0)
                simulated_lifespan = max(20.0, min(150.0, 120.0 - conc_diff_ratio * 40.0 + (i % 3) * 5.0))
                short_hour = simulated_lifespan * 0.85

                cycler_file = cycler_dir / f"{sample_id}_{vial_id}_benchmark_cycling.csv"
                create_benchmark_cycler_file(
                    filepath=cycler_file,
                    duration_hours=simulated_lifespan,
                    short_at_hour=short_hour,
                    plateau_voltage_v=0.030 + conc_diff_ratio * 0.01,
                )

            if cycler_file and cycler_file.exists():
                try:
                    report = self.run_analysis(
                        data_source=cycler_file,
                        technique="galvanostatic_symmetric",
                        area_cm2=1.13097,
                        sample_id=sample_id,
                        output_dir=target_dir,
                    )
                    # Performance score: prioritize longer healthy active lifespan and lower overpotential
                    if report.symmetric_cell:
                        active_h = report.symmetric_cell.active_cycling_metrics.get("active_lifespan_hours", 0.0) if report.symmetric_cell.active_cycling_metrics else report.symmetric_cell.total_lifespan_hours
                        overpot_mv = report.symmetric_cell.active_cycling_metrics.get("mean_overpotential_mv", 40.0) if report.symmetric_cell.active_cycling_metrics else report.symmetric_cell.mean_overpotential_mv
                        performance_score = round(active_h / (1.0 + overpot_mv / 50.0), 3)

                    # Enrich report summary and metadata with Phase 2 tracking context
                    report.summary_metrics.update({
                        "sample_id": sample_id,
                        "vial_id": vial.vial_id,
                        "point_type": point_type,
                        "additive_conc_mM": vial.additive_conc_mM,
                        "salt_conc_M": vial.salt_conc_M,
                        "rack_format": vial.rack_format,
                        "well_coordinate": f"{vial.row}{vial.col}",
                        "slot": vial.slot,
                        "aspiration_rate_ul_s": vial.aspiration_rate_ul_s,
                        "dispense_rate_ul_s": vial.dispense_rate_ul_s,
                    })
                    if report.processed_data:
                        report.processed_data.metadata.update({
                            "sample_id": sample_id,
                            "vial_id": vial.vial_id,
                            "point_type": point_type,
                            "additive_conc_mM": vial.additive_conc_mM,
                            "salt_conc_M": vial.salt_conc_M,
                        })
                except Exception as e:
                    logger.error(f"Failed to analyze cycler file '{cycler_file}': {e}")

            cell_results.append(
                FormulatedCellResult(
                    sample_id=sample_id,
                    vial_id=vial.vial_id,
                    point_type=point_type,
                    rack_format=vial.rack_format,
                    row=vial.row,
                    col=vial.col,
                    slot=vial.slot,
                    aspiration_rate_ul_s=vial.aspiration_rate_ul_s,
                    dispense_rate_ul_s=vial.dispense_rate_ul_s,
                    additive_conc_mM=vial.additive_conc_mM,
                    salt_conc_M=vial.salt_conc_M,
                    target_volume_uL=vial.target_volume_uL,
                    temperature_C=temp_c,
                    analysis_report=report,
                    performance_score=performance_score,
                    status="completed" if report is not None else "pending_cycling",
                    metadata={
                        "additive_stock_uL": vial.additive_stock_uL,
                        "salt_stock_uL": vial.salt_stock_uL,
                        "water_uL": vial.water_uL,
                        "warnings": vial.warnings,
                    },
                )
            )

        pareto_candidates, best_cand = self.compute_pareto_frontier(cell_results)

        campaign = CampaignResult(
            campaign_id=campaign_id,
            target_salt=target_salt,
            discovery_summary=discovery_summary,
            formulation_plan=plan,
            dispensing_schedule=schedule,
            cell_results=cell_results,
            pareto_optimal_candidates=pareto_candidates,
            best_candidate=best_cand,
            output_dir=str(target_dir),
            created_at=datetime.now(timezone.utc).isoformat(),
            iteration_round=0,
            total_iterations=1,
            history=[{
                "iteration_round": 0,
                "new_samples": len(cell_results),
                "total_cells": len(cell_results),
                "best_sample_id": best_cand.sample_id if best_cand else None,
                "best_score": best_cand.performance_score if best_cand else None,
            }],
        )

        summary_path = target_dir / "campaign_summary.json"
        summary_path.write_text(campaign.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"Campaign execution completed. Summary saved to: {summary_path}")

        return campaign

    def compute_pareto_frontier(
        self,
        cells: List[FormulatedCellResult],
    ) -> Tuple[List[FormulatedCellResult], Optional[FormulatedCellResult]]:
        """
        Identify multi-objective Pareto-optimal candidates and champion cell.
        Objectives: Maximize active lifespan (h), Minimize mean overpotential (mV).
        """
        completed_cells = [c for c in cells if c.performance_score is not None]
        for c in cells:
            c.is_pareto_optimal = False

        if not completed_cells:
            return [], None

        pareto_candidates: List[FormulatedCellResult] = []
        for c1 in completed_cells:
            is_dominated = False
            c1_symm = c1.analysis_report.symmetric_cell if c1.analysis_report else None
            c1_act = (
                c1_symm.active_cycling_metrics.get("active_lifespan_hours", 0.0)
                if (c1_symm and c1_symm.active_cycling_metrics)
                else (c1_symm.total_lifespan_hours if c1_symm else 0.0)
            )
            c1_overpot = (
                c1_symm.active_cycling_metrics.get("mean_overpotential_mv", 999.0)
                if (c1_symm and c1_symm.active_cycling_metrics)
                else (c1_symm.mean_overpotential_mv if c1_symm else 999.0)
            )

            for c2 in completed_cells:
                if c1 == c2:
                    continue
                c2_symm = c2.analysis_report.symmetric_cell if c2.analysis_report else None
                c2_act = (
                    c2_symm.active_cycling_metrics.get("active_lifespan_hours", 0.0)
                    if (c2_symm and c2_symm.active_cycling_metrics)
                    else (c2_symm.total_lifespan_hours if c2_symm else 0.0)
                )
                c2_overpot = (
                    c2_symm.active_cycling_metrics.get("mean_overpotential_mv", 999.0)
                    if (c2_symm and c2_symm.active_cycling_metrics)
                    else (c2_symm.mean_overpotential_mv if c2_symm else 999.0)
                )

                # c2 strictly dominates c1 if c2 has >= lifespan and <= overpotential, and strictly better in at least one
                if (c2_act >= c1_act and c2_overpot <= c1_overpot and
                    (c2_act > c1_act or c2_overpot < c1_overpot)):
                    is_dominated = True
                    break

            if not is_dominated:
                c1.is_pareto_optimal = True
                pareto_candidates.append(c1)

        frontier = pareto_candidates if pareto_candidates else completed_cells
        best_cand = max(frontier, key=lambda c: c.performance_score or 0.0)
        return pareto_candidates, best_cand

    def run_adaptive_iteration(
        self,
        campaign: CampaignResult,
        num_samples: int = 3,
        cycler_data: Optional[Union[str, Path]] = None,
        use_benchmark: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
        seed: int = 42,
        rack_format: Optional[str] = None,
        total_volume_uL: Optional[float] = None,
        al_strategy: str = "balanced",
    ) -> CampaignResult:
        """
        Execute an adaptive Phase 3 ➔ Phase 2 active learning sequential DoE iteration.

        Ingests prior experimental results from the campaign, models the response surface,
        detects performance gradients, generates the next batch of candidate formulations
        respecting Delaney ESOL solubility limits, schedules robotic dispensing with well
        offset continuity, runs Phase 3 characterization, and updates the global Pareto frontier.

        Args:
            campaign: Existing CampaignResult instance to update and advance.
            num_samples: Number of adaptive formulations to propose and evaluate.
            cycler_data: Optional experimental cycler file or folder for new formulations.
            use_benchmark: If True, simulates benchmark cycling runs for each new formulation point.
            output_dir: Optional override output directory.
            seed: Random seed for DoE sampling.
            rack_format: Optional rack format override.
            total_volume_uL: Optional total volume override.
            al_strategy: Active learning strategy ('fastest'/'exploration', 'refine'/'exploitation', 'balanced').

        Returns:
            Updated CampaignResult containing aggregated formulations, vials, and global Pareto frontier.
        """
        target_dir = Path(output_dir) if output_dir else Path(campaign.output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        next_round = max([c.iteration_round for c in campaign.cell_results], default=0) + 1
        logger.info(f"Executing Adaptive Iteration Round {next_round} for campaign {campaign.campaign_id}")

        # 1. Phase 2 Adaptive DoE Plan
        adaptive_plan = self.doe_engine.generate_adaptive_plan(
            anchor=campaign.formulation_plan.anchor,
            prior_results=campaign.cell_results,
            num_samples=num_samples,
            iteration_round=next_round,
            seed=seed,
            start_sample_index=len(campaign.cell_results) + 1,
            al_strategy=al_strategy,
        )

        plan_path = target_dir / f"doe_plan_round_{next_round}.json"
        plan_path.write_text(adaptive_plan.model_dump_json(indent=2), encoding="utf-8")

        # 2. Robotic Dispensing Schedule with rack continuity
        rf = rack_format or campaign.dispensing_schedule.rack_format
        tot_vol = total_volume_uL or campaign.dispensing_schedule.total_volume_per_vial_uL
        existing_vials = campaign.dispensing_schedule.vials

        schedule = self.dispenser_engine.generate_schedule(
            plan=adaptive_plan,
            rack_format=rf,
            total_volume_uL=tot_vol,
            additive_stock_conc_mM=campaign.dispensing_schedule.additive_stock_conc_mM,
            salt_stock_conc_M=campaign.dispensing_schedule.salt_stock_conc_M,
            initial_vial_offset=len(existing_vials),
            existing_vials=existing_vials,
            iteration_round=next_round,
        )

        sched_path = target_dir / f"dispensing_schedule_round_{next_round}.json"
        sched_path.write_text(schedule.model_dump_json(indent=2), encoding="utf-8")

        # 3. Phase 3 Characterization & Evaluation
        cycler_dir = target_dir / "cycler_data"
        cycler_dir.mkdir(parents=True, exist_ok=True)

        new_cell_results: List[FormulatedCellResult] = []

        # Find previous best candidate to anchor simulated benchmark progression
        prev_best = campaign.best_candidate
        target_optimal_conc = prev_best.additive_conc_mM if prev_best else adaptive_plan.anchor.shifted_anchor_conc
        if prev_best and adaptive_plan.anchor.shifted_anchor_conc > 0:
            target_optimal_conc = max(prev_best.additive_conc_mM, max(f.additive_conc_mM for f in adaptive_plan.formulations))

        for i, vial in enumerate(schedule.vials):
            form_pt = adaptive_plan.formulations[i] if i < len(adaptive_plan.formulations) else None
            sample_id = form_pt.sample_id if form_pt else vial.vial_id
            point_type = form_pt.point_type if form_pt else "adaptive"
            temp_c = form_pt.temperature_C if form_pt else None
            vial_id = vial.vial_id
            report: Optional[ElectrochemicalAnalysisReport] = None
            performance_score: Optional[float] = None

            cycler_file = None
            if cycler_data:
                cd_path = Path(cycler_data)
                if cd_path.is_file():
                    cycler_file = cd_path
                elif cd_path.is_dir():
                    candidate_files = list(cd_path.glob(f"*{sample_id}*")) or list(cd_path.glob(f"*{vial_id}*"))
                    if candidate_files:
                        cycler_file = candidate_files[0]

            if not cycler_file and use_benchmark:
                conc_diff = abs(vial.additive_conc_mM - target_optimal_conc)
                norm_diff = conc_diff / (target_optimal_conc if target_optimal_conc > 0 else 1.0)
                simulated_lifespan = max(25.0, min(160.0, 135.0 - norm_diff * 45.0 + (i % 2) * 5.0))
                short_hour = simulated_lifespan * 0.88
                plateau_voltage_v = 0.028 + norm_diff * 0.008

                cycler_file = cycler_dir / f"{sample_id}_{vial_id}_benchmark_cycling.csv"
                create_benchmark_cycler_file(
                    filepath=cycler_file,
                    duration_hours=simulated_lifespan,
                    short_at_hour=short_hour,
                    plateau_voltage_v=plateau_voltage_v,
                )

            if cycler_file and cycler_file.exists():
                try:
                    report = self.run_analysis(
                        data_source=cycler_file,
                        technique="galvanostatic_symmetric",
                        area_cm2=1.13097,
                        sample_id=sample_id,
                        output_dir=target_dir,
                    )
                    if report.symmetric_cell:
                        symm = report.symmetric_cell
                        active_h = symm.active_cycling_metrics.get("active_lifespan_hours", 0.0) if symm.active_cycling_metrics else symm.total_lifespan_hours
                        overpot_mv = symm.active_cycling_metrics.get("mean_overpotential_mv", 40.0) if symm.active_cycling_metrics else symm.mean_overpotential_mv
                        performance_score = round(active_h / (1.0 + overpot_mv / 50.0), 3)

                    report.summary_metrics.update({
                        "sample_id": sample_id,
                        "vial_id": vial.vial_id,
                        "point_type": point_type,
                        "additive_conc_mM": vial.additive_conc_mM,
                        "salt_conc_M": vial.salt_conc_M,
                        "rack_format": vial.rack_format,
                        "well_coordinate": f"{vial.row}{vial.col}",
                        "slot": vial.slot,
                        "aspiration_rate_ul_s": vial.aspiration_rate_ul_s,
                        "dispense_rate_ul_s": vial.dispense_rate_ul_s,
                        "iteration_round": next_round,
                    })
                    if report.processed_data:
                        report.processed_data.metadata.update({
                            "sample_id": sample_id,
                            "vial_id": vial.vial_id,
                            "point_type": point_type,
                            "additive_conc_mM": vial.additive_conc_mM,
                            "salt_conc_M": vial.salt_conc_M,
                            "iteration_round": next_round,
                        })
                except Exception as e:
                    logger.error(f"Failed to analyze cycler file '{cycler_file}': {e}")

            new_cell_results.append(
                FormulatedCellResult(
                    sample_id=sample_id,
                    vial_id=vial.vial_id,
                    point_type=point_type,
                    rack_format=vial.rack_format,
                    row=vial.row,
                    col=vial.col,
                    slot=vial.slot,
                    aspiration_rate_ul_s=vial.aspiration_rate_ul_s,
                    dispense_rate_ul_s=vial.dispense_rate_ul_s,
                    additive_conc_mM=vial.additive_conc_mM,
                    salt_conc_M=vial.salt_conc_M,
                    target_volume_uL=vial.target_volume_uL,
                    temperature_C=temp_c,
                    analysis_report=report,
                    performance_score=performance_score,
                    status="completed" if report is not None else "pending_cycling",
                    iteration_round=next_round,
                    metadata={
                        "additive_stock_uL": vial.additive_stock_uL,
                        "salt_stock_uL": vial.salt_stock_uL,
                        "water_uL": vial.water_uL,
                        "warnings": vial.warnings,
                    },
                )
            )

        # 4. Aggregate across all rounds and recompute global Pareto frontier
        campaign.cell_results.extend(new_cell_results)
        campaign.dispensing_schedule.vials.extend(schedule.vials)
        campaign.dispensing_schedule.total_vials = len(campaign.dispensing_schedule.vials)
        campaign.formulation_plan = adaptive_plan
        campaign.iteration_round = next_round
        campaign.total_iterations = next_round + 1

        pareto_candidates, best_cand = self.compute_pareto_frontier(campaign.cell_results)
        campaign.pareto_optimal_candidates = pareto_candidates
        campaign.best_candidate = best_cand

        campaign.history.append({
            "iteration_round": next_round,
            "new_samples": len(new_cell_results),
            "total_cells": len(campaign.cell_results),
            "best_sample_id": best_cand.sample_id if best_cand else None,
            "best_score": best_cand.performance_score if best_cand else None,
        })

        summary_path = target_dir / "campaign_summary.json"
        summary_path.write_text(campaign.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"Adaptive iteration {next_round} completed. Summary saved to: {summary_path}")

        return campaign

    def run_closed_loop_campaign(
        self,
        query: Optional[str] = None,
        from_extraction: Optional[Union[str, Path]] = None,
        additive_name: Optional[str] = None,
        published_conc_mM: Optional[float] = None,
        smiles: Optional[str] = None,
        parent_electrolyte: str = "2 M ZnSO4",
        target_salt: str = "Zn(OTf)2",
        doe_method: str = "lhs",
        initial_samples: int = 3,
        max_iterations: int = 2,
        samples_per_iteration: int = 3,
        rack_format: str = "24-vial rack",
        total_volume_uL: Optional[float] = None,
        cycler_data: Optional[Union[str, Path]] = None,
        use_benchmark: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
        seed: int = 42,
        al_strategy: str = "balanced",
    ) -> CampaignResult:
        """
        Execute an autonomous closed-loop campaign across multiple iterations:
        Round 0 (Initial DoE & Phase 3 Evaluation) ➔ Feedback ➔ Round 1+ (Adaptive Sequential DoE)
        ➔ Global Pareto Frontier Update.

        Args:
            query: Phase 1 search query.
            from_extraction: Pre-existing extraction JSON path.
            additive_name: Direct manual additive chemical name.
            published_conc_mM: Direct manual published concentration (mM).
            smiles: Direct manual SMILES string.
            parent_electrolyte: Baseline parent electrolyte string.
            target_salt: Target electrolyte salt system.
            doe_method: Initial DoE methodology ('lhs' or 'ccd').
            initial_samples: Number of formulations in Round 0.
            max_iterations: Total number of iterations (Round 0, 1, ..., max_iterations - 1).
            samples_per_iteration: Formulations to evaluate in each adaptive round.
            rack_format: Robotic deck format ('24-vial rack' or '96-well plate').
            total_volume_uL: Volume per vial in uL.
            cycler_data: Optional cycler file or folder.
            use_benchmark: If True, simulates cycling for all formulated points.
            output_dir: Output artifacts directory.
            seed: Random seed.
            al_strategy: Active learning strategy ('fastest'/'exploration', 'refine'/'exploitation', 'balanced').

        Returns:
            Final CampaignResult containing aggregated results and global champion candidate.
        """
        logger.info(
            f"Launching Closed-Loop Active Learning Campaign: "
            f"Target {target_salt}, {max_iterations} iterations, {initial_samples} initial + {samples_per_iteration} per adaptive round"
        )

        # 1. Round 0 execution
        campaign = self.run_full_campaign(
            query=query,
            from_extraction=from_extraction,
            additive_name=additive_name,
            published_conc_mM=published_conc_mM,
            smiles=smiles,
            parent_electrolyte=parent_electrolyte,
            target_salt=target_salt,
            doe_method=doe_method,
            num_samples=initial_samples,
            rack_format=rack_format,
            total_volume_uL=total_volume_uL,
            cycler_data=cycler_data,
            use_benchmark=use_benchmark,
            output_dir=output_dir,
            seed=seed,
        )

        # Ensure Round 0 tagging
        campaign.iteration_round = 0
        campaign.total_iterations = 1
        campaign.history = [{
            "iteration_round": 0,
            "new_samples": len(campaign.cell_results),
            "total_cells": len(campaign.cell_results),
            "best_sample_id": campaign.best_candidate.sample_id if campaign.best_candidate else None,
            "best_score": campaign.best_candidate.performance_score if campaign.best_candidate else None,
        }]
        for cell in campaign.cell_results:
            cell.iteration_round = 0

        # 2. Sequential adaptive iterations
        for round_idx in range(1, max_iterations):
            logger.info(f"Closed-Loop Feedback: Triggering adaptive round {round_idx} / {max_iterations - 1}")
            campaign = self.run_adaptive_iteration(
                campaign=campaign,
                num_samples=samples_per_iteration,
                cycler_data=cycler_data,
                use_benchmark=use_benchmark,
                output_dir=output_dir,
                seed=seed + round_idx * 17,
                rack_format=rack_format,
                total_volume_uL=total_volume_uL,
                al_strategy=al_strategy,
            )

        return campaign


def sync_custom_formulation_pipeline(
    additive_name: str,
    published_conc_mM: float,
    parent_electrolyte: str,
    target_electrolyte: str,
    smiles: Optional[str] = None,
    doe_method: str = "lhs",
    num_samples: int = 5,
    seed: int = 42,
    rack_format: str = "24-vial rack",
    total_volume_uL: float = 2000.0,
    session_state: Optional[Any] = None,
    update_custom: bool = True,
) -> Tuple[ConcentrationAnchor, DoEPlan, DispensingSchedule]:
    """
    Synchronously compute the ConcentrationAnchor, DoEPlan, and OT-2 DispensingSchedule
    for a given additive and electrolyte pair, and update Streamlit session state.

    Args:
        additive_name: Name of the selected electrolyte additive.
        published_conc_mM: Reported baseline concentration in mM.
        parent_electrolyte: Literature parent baseline electrolyte string (e.g., "2 M ZnSO4").
        target_electrolyte: Target baseline salt or electrolyte string (e.g., "ZnSO4").
        smiles: Optional SMILES string of the additive.
        doe_method: Experimental design sampling method ("lhs", "sobol", "grid", "bayesian").
        num_samples: Number of experimental points to generate.
        seed: Random seed for reproducible sampling.
        rack_format: Labware rack format ("24-vial rack" or "96-well plate").
        total_volume_uL: Total dispense volume per sample in microliters.
        session_state: Optional mapping/session_state object to update (defaults to st.session_state if available).
        update_custom: Whether to also update custom_formulation_plan and custom_dispensing_schedule.

    Returns:
        Tuple of (ConcentrationAnchor, DoEPlan, DispensingSchedule).
    """
    anchor = ConcentrationAnchorEngine().calculate_anchor(
        additive_name=additive_name,
        published_conc_mM=float(published_conc_mM),
        parent_electrolyte=parent_electrolyte,
        target_electrolyte=target_electrolyte,
        smiles=smiles,
    )
    plan = DoEEngine().create_plan(
        anchor=anchor,
        method=doe_method,
        num_samples=num_samples,
        seed=seed,
    )
    schedule = RoboticDispenserEngine().generate_schedule(
        plan=plan,
        rack_format=rack_format,
        total_volume_uL=total_volume_uL,
    )

    if session_state is None:
        try:
            import streamlit as st
            session_state = st.session_state
        except Exception:
            session_state = None

    if session_state is not None:
        session_state["extracted_formulation_plan"] = plan
        session_state["extracted_dispensing_schedule"] = schedule
        if update_custom:
            session_state["custom_formulation_plan"] = plan
            session_state["custom_dispensing_schedule"] = schedule

    return anchor, plan, schedule

