"""
Data schemas for the Zinc-Ion Battery Literature Extraction pipeline.
Defines the Pydantic models for structured LLM extraction and SDL APIs.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ElectrolyteCompatibilityDetail(BaseModel):
    salt: str = Field(description="Electrolyte salt system (e.g., ZnSO4, Zn(OTf)2, Zn(ClO4)2, Zn(TFSI)2, ZnCl2, Zn(CH3COO)2)")
    status: str = Field(description="Compatibility status (e.g., 'Validated', 'Highly Compatible', 'Compatible', 'Caution / Conditional', 'Incompatible')")
    expected_effectiveness: str = Field(description="Expected electrochemical effectiveness / performance level (e.g., 'High', 'Moderate', 'Conditional')")
    notes: str = Field(description="Scientific rationale and mechanistic guidance for this chemistry")

# Alias for convenience
ElectrolyteCompatibility = ElectrolyteCompatibilityDetail

class PaperMetadata(BaseModel):
    paper_id: str
    title: str
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    open_access_pdf_url: Optional[str] = None

class Concentration(BaseModel):
    value: Optional[float] = Field(default=None, description="Numerical value of the concentration")
    unit: Optional[str] = Field(default=None, description="Unit of measurement (e.g., mM, M, wt%, vol%)")
    raw_text: Optional[str] = Field(default=None, description="The exact text as it appeared in the paper")

class ZincIonBatteryContext(BaseModel):
    baseline_electrolyte: str = Field(description="Baseline salt and concentration (e.g., 2 M ZnSO4)")
    solvent: str = Field(default="H2O", description="Primary solvent (e.g., H2O)")
    electrolyte_pH: Optional[float] = Field(
        default=None, 
        description="pH of the electrolyte, critical for HER severity and zinc salt precipitation."
    )
    anode_material: str = Field(default="Zn foil", description="Material used for anode (e.g., Zn foil)")
    cathode_material: Optional[str] = Field(None, description="Material used for cathode (e.g., V2O5, MnO2)")
    cell_format: str = Field(default="Symmetric", description="Type of cell assembled (e.g., Symmetric, Asymmetric, Full cell)")
    temperature_celsius: Optional[float] = Field(None, description="Testing temperature in Celsius")
    separator_material: Optional[str] = Field(
        default=None, 
        description="Separator type used (e.g., Glass fiber GF/A, Celgard 3501, Filter paper). Critical for avoiding confounding morphological effects."
    )

class CyclingCondition(BaseModel):
    current_density_mA_cm2: float = Field(description="Applied current density in mA/cm2")
    areal_capacity_mAh_cm2: float = Field(description="Areal capacity in mAh/cm2")
    lifespan_hours: float = Field(description="Total lifespan achieved in hours under these specific conditions")

class PerformanceMetrics(BaseModel):
    symmetric_cell_performance: Optional[List[CyclingCondition]] = Field(
        default=None, 
        description="List of cycling conditions and their respective achieved lifespans."
    )
    coulombic_efficiency_percent: Optional[float] = None
    capacity_retention_percent: Optional[float] = None
    full_cell_cycle_count: Optional[int] = None
    overpotential_mV: Optional[float] = None

class ExtractedAdditive(BaseModel):
    name: str = Field(description="Common name of the electrolyte additive")
    iupac_name: Optional[str] = Field(None, description="Formal IUPAC name if available")
    smiles: Optional[str] = Field(None, description="SMILES string representation of the additive")
    optimal_concentration: Optional[Concentration] = Field(default=None, description="Reported optimal additive concentration")
    concentration_range_tested: List[Concentration] = Field(default_factory=list)
    functional_groups: List[str] = Field(default_factory=list, description="Key functional groups (e.g., hydroxyl, amine)")
    primary_mechanisms: List[str] = Field(default_factory=list, description="How the additive works (e.g., HER suppression)")
    mechanism_summary: str = Field(default="", description="A brief summary of the mechanistic action")
    battery_context: Optional[ZincIonBatteryContext] = None
    performance_metrics: Optional[PerformanceMetrics] = None
    confidence_score: float = Field(default=1.0, description="Confidence in extraction accuracy (0.0 to 1.0)")
    extraction_evidence: str = Field(default="", description="Direct quote from text supporting optimal concentration")

class CandidateMolecule(BaseModel):
    name: str
    smiles: str
    similarity_score: float
    similarity_metric: str = "Tanimoto (Morgan)"
    molecular_weight: float
    logp: float
    logs: Optional[float] = Field(None, description="Estimated intrinsic aqueous solubility (ESOL)")
    tpsa: float
    hbd: int
    hba: int
    rotatable_bonds: int
    rationale: str
    sdl_formulation_notes: str = Field(default="", description="Aqueous solubility limits and liquid handling alerts")
    parent_baseline_electrolyte: Optional[str] = Field(
        default=None, 
        description="The baseline electrolyte salt and concentration used in the parent study (e.g., 2 M ZnSO4)."
    )
    electrolyte_compatibility_note: Optional[str] = Field(
        default=None, 
        description="Electrochemical compatibility guidance across alternative electrolyte chemistries."
    )
    electrolyte_compatibility_breakdown: Optional[Dict[str, ElectrolyteCompatibilityDetail]] = Field(
        default=None, 
        description="Per-salt compatibility and effectiveness evaluation across all major aqueous ZIB electrolyte chemistries."
    )
    chemical_formula: Optional[str] = Field(default=None, description="Conventional chemical formula with Unicode subscripts")
    reported_concentration: Optional[float] = Field(default=None, description="Reported optimal additive concentration in mM")
    reported_concentration_unit: Optional[str] = Field(default="mM", description="Unit of reported concentration")
    reported_mechanisms: Optional[List[str]] = Field(default=None, description="Primary mechanisms reported in literature")
    mechanism_summary: Optional[str] = Field(default=None, description="Summary of mechanistic action")
    source_paper_title: Optional[str] = Field(default=None, description="Title of parent literature publication")
    is_literature_additive: bool = Field(default=False, description="True if direct literature finding; False if AI-suggested novel analogue")
    citation_count: int = Field(default=1, description="Number of literature citations / occurrences")
    parent_scaffold_name: Optional[str] = Field(default=None, description="Name of the parent literature scaffold this analogue is anchored to")
    parent_scaffold_smiles: Optional[str] = Field(default=None, description="SMILES of the parent literature scaffold")
    parent_scaffold_similarity: Optional[float] = Field(default=None, description="Tanimoto similarity score to the parent literature scaffold")
    ai_rationale: Optional[str] = Field(default=None, description="Dedicated chemoinformatic AI suggestion rationale")
    functional_motif_rationale: Optional[str] = Field(default=None, description="Mechanistic rationale for functional groups in Zn2+ solvation/interphase")
    physical_hypothesis: Optional[str] = Field(default=None, description="Physical hypothesis regarding interfacial adsorption and solubility")

class ExtractionResult(BaseModel):
    paper_metadata: PaperMetadata
    additives: List[ExtractedAdditive]
    candidate_molecules: List[CandidateMolecule]
    extraction_summary: str
    source_type: str


# =====================================================================
# Phase 2: Autonomous Design of Experiments (DoE) & Robotic Dispensing
# =====================================================================

class ConcentrationAnchor(BaseModel):
    additive_name: str
    smiles: Optional[str] = None
    published_concentration: float
    published_unit: str = "mM"
    parent_electrolyte: str
    parent_salt: str
    parent_molarity: float
    target_electrolyte: str
    target_electrolyte_salt: str
    target_salt_molarity: float
    scaling_factor: float
    shifted_anchor_conc: float
    anchor_unit: str = "mM"
    min_conc: float
    max_conc: float
    solubility_limit_mM: Optional[float] = None
    solubility_ceiling_mM: Optional[float] = None
    is_capped_by_solubility: bool = False
    electrolyte_adjustment_rationale: str
    functional_groups: List[str] = Field(default_factory=list)
    is_baseline_unspecified: bool = Field(default=False, description="True if concentration or baseline salt was unspecified in literature/file upload")
    reported_in_file: bool = Field(default=True, description="True if concentration and baseline salt were explicitly reported in uploaded file")


class DoEParameter(BaseModel):
    name: str
    display_name: str
    unit: str
    min_val: float
    max_val: float
    baseline_val: float


class FormulationPoint(BaseModel):
    sample_id: str
    additive_conc_mM: float
    salt_conc_M: float
    temperature_C: Optional[float] = None
    point_type: str = "center"
    coordinate: Optional[str] = None


class DoEPlan(BaseModel):
    plan_id: str
    additive_name: str
    target_electrolyte_salt: str
    target_salt_default_molarity: float
    target_salt_molarity_range: List[float]
    method: str
    anchor: ConcentrationAnchor
    parameters: List[DoEParameter]
    formulations: List[FormulationPoint]
    electrolyte_adjustment_rationale: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    iteration_round: int = 0
    prior_observations: List[FormulatedCellResult] = Field(default_factory=list)


class RoboticVial(BaseModel):
    vial_id: str
    rack_format: str
    row: str
    col: int
    additive_conc_mM: float
    salt_conc_M: float
    target_volume_uL: float
    additive_stock_uL: float
    salt_stock_uL: float
    water_uL: float
    warnings: List[str] = Field(default_factory=list)
    slot: int = 1
    aspiration_rate_ul_s: float = 50.0
    dispense_rate_ul_s: float = 50.0


class DispensingSchedule(BaseModel):
    schedule_id: str
    target_electrolyte_salt: str
    electrolyte_adjustment_rationale: str
    rack_format: str
    total_vials: int
    total_volume_per_vial_uL: float
    additive_stock_conc_mM: float
    salt_stock_conc_M: float
    salt_stock_name: str
    vials: List[RoboticVial]
    warnings: List[str] = Field(default_factory=list)
    ot2_deck_slot: int = 1
    default_aspiration_rate_ul_s: float = 50.0
    default_dispense_rate_ul_s: float = 50.0
    created_at: Optional[str] = None
    iteration_round: int = 0


# =====================================================================
# Phase 3: Autonomous Electrochemical Characterization & Failure Analysis
# =====================================================================

from enum import Enum


class ElectrochemicalTechnique(str, Enum):
    GALVANOSTATIC_SYMMETRIC = "galvanostatic_symmetric"
    GALVANOSTATIC_ASYMMETRIC = "galvanostatic_asymmetric"
    GALVANOSTATIC_FULL_CELL = "galvanostatic_full_cell"
    LSV = "lsv"
    EIS = "eis"
    CV = "cv"
    UNKNOWN = "unknown"


class ElectrodeGeometry(BaseModel):
    area_cm2: float = 1.13097
    diameter_mm: Optional[float] = None
    active_mass_mg: Optional[float] = None
    anode_material: Optional[str] = "Zn foil"
    cathode_material: Optional[str] = None


class CyclerRawData(BaseModel):
    time_s: List[float] = Field(default_factory=list)
    voltage_v: List[float] = Field(default_factory=list)
    current_ma: List[float] = Field(default_factory=list)
    step_index: Optional[List[int]] = None
    cycle_number: Optional[List[int]] = None
    capacity_charge_mah: Optional[List[float]] = None
    capacity_discharge_mah: Optional[List[float]] = None
    frequency_hz: Optional[List[float]] = None
    z_real_ohm: Optional[List[float]] = None
    z_imag_ohm: Optional[List[float]] = None
    coulombic_efficiency: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NormalizedElectrochemicalData(BaseModel):
    time_s: List[float] = Field(default_factory=list)
    display_time: List[float] = Field(default_factory=list)
    time_unit: str = "h"
    voltage_v: List[float] = Field(default_factory=list)
    current_ma: List[float] = Field(default_factory=list)
    current_density_ma_cm2: List[float] = Field(default_factory=list)
    capacity_mah: List[float] = Field(default_factory=list)
    areal_capacity_mah_cm2: List[float] = Field(default_factory=list)
    specific_capacity_mah_g: Optional[List[float]] = None
    dq_dv: Optional[List[float]] = None
    step_index: Optional[List[int]] = None
    cycle_number: Optional[List[int]] = None
    capacity_charge_mah: Optional[List[float]] = None
    capacity_discharge_mah: Optional[List[float]] = None
    coulombic_efficiency: Optional[List[float]] = None
    rest_trimmed_seconds: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SymmetricCellCycleMetric(BaseModel):
    cycle_number: int
    stripping_plateau_v: float
    plating_plateau_v: float
    overpotential_mv: float
    voltage_hysteresis_mv: float
    areal_capacity_mah_cm2: float


class ActiveCyclingMetrics(BaseModel):
    mean_overpotential_mv: float
    mean_hysteresis_mv: float
    active_lifespan_hours: float
    active_cycles: int
    data_points: int
    performance_score: Optional[float] = None


class TotalTestSummary(BaseModel):
    nominal_duration_hours: float
    total_data_points: int
    failure_mode: str = "none"
    failure_time_hours: Optional[float] = None
    failure_cycle: Optional[int] = None
    short_circuit_detected: bool = False
    diagnostic_message: Optional[str] = None


class SymmetricCellAnalysis(BaseModel):
    technique: str = "galvanostatic_symmetric"
    total_cycles: int
    total_lifespan_hours: float
    mean_overpotential_mv: float
    mean_hysteresis_mv: float
    short_circuit_detected: bool = False
    soft_short_detected: bool = False
    failure_cycle: Optional[int] = None
    failure_time_hours: Optional[float] = None
    failure_mode: str = "none"
    cycle_metrics: List[SymmetricCellCycleMetric] = Field(default_factory=list)
    active_cycling_metrics: Optional[Dict[str, Any]] = None
    total_test_summary: Optional[Dict[str, Any]] = None


class AsymmetricHalfCellCycleMetric(BaseModel):
    cycle_number: int
    plating_capacity_mah_cm2: float
    stripping_capacity_mah_cm2: float
    coulombic_efficiency_percent: float
    nucleation_overpotential_mv: float
    plateau_overpotential_mv: float


class AsymmetricHalfCellAnalysis(BaseModel):
    technique: str = "galvanostatic_asymmetric"
    total_cycles: int
    average_ce_percent: float
    final_ce_percent: float
    cumulative_capacity_mah_cm2: float
    mean_nucleation_overpotential_mv: float
    cycle_metrics: List[AsymmetricHalfCellCycleMetric] = Field(default_factory=list)


class FullCellAnalysis(BaseModel):
    technique: str = "galvanostatic_full_cell"
    initial_discharge_capacity_mah_g: float
    final_discharge_capacity_mah_g: float
    capacity_retention_percent: float
    average_coulombic_efficiency_percent: float
    energy_efficiency_percent: Optional[float] = None
    average_energy_efficiency_percent: Optional[float] = None
    total_cycles: int
    cycle_numbers: List[int] = Field(default_factory=list)
    discharge_capacities_mah_g: List[float] = Field(default_factory=list)
    coulombic_efficiencies: List[float] = Field(default_factory=list)
    energy_efficiencies: List[float] = Field(default_factory=list)


class LSVAnalysis(BaseModel):
    technique: str = "lsv"
    her_onset_potential_v: Optional[float] = None
    her_current_density_threshold_ma_cm2: float = 1.0
    corrosion_potential_ecorr_v: Optional[float] = None
    corrosion_current_jcorr_ma_cm2: Optional[float] = None
    tafel_slope_cathodic_mv_per_dec: Optional[float] = None
    tafel_slope_anodic_mv_per_dec: Optional[float] = None
    electrochemical_stability_window_v: Optional[float] = None


class EISAnalysis(BaseModel):
    technique: str = "eis"
    solution_resistance_rs_ohm: Optional[float] = None
    charge_transfer_resistance_rct_ohm: Optional[float] = None
    characteristic_frequency_hz: Optional[float] = None
    equivalent_circuit: str = "R(RQ)"


class CVAnalysis(BaseModel):
    technique: str = "cv"
    anodic_peak_potential_epa_v: Optional[float] = None
    cathodic_peak_potential_epc_v: Optional[float] = None
    anodic_peak_current_ipa_ma: Optional[float] = None
    cathodic_peak_current_ipc_ma: Optional[float] = None
    peak_separation_delta_ep_mv: Optional[float] = None
    current_ratio_ipa_ipc: Optional[float] = None
    plating_onset_potential_v: Optional[float] = None
    stripping_onset_potential_v: Optional[float] = None
    nucleation_overpotential_mv: Optional[float] = None


class ElectrochemicalAnalysisReport(BaseModel):
    sample_id: str
    file_source: str
    technique: str
    geometry: ElectrodeGeometry
    processed_data: NormalizedElectrochemicalData
    symmetric_cell: Optional[SymmetricCellAnalysis] = None
    asymmetric_cell: Optional[AsymmetricHalfCellAnalysis] = None
    full_cell: Optional[FullCellAnalysis] = None
    lsv: Optional[LSVAnalysis] = None
    eis: Optional[EISAnalysis] = None
    cv: Optional[CVAnalysis] = None
    summary_metrics: Dict[str, Any] = Field(default_factory=dict)
    active_cycling_metrics: Optional[Dict[str, Any]] = None
    total_test_summary: Optional[Dict[str, Any]] = None
    visualization_html_path: Optional[str] = None
    created_at: Optional[str] = None



# =====================================================================
# End-to-End Campaign & Formulated Cell Results
# =====================================================================

class FormulatedCellResult(BaseModel):
    sample_id: str
    vial_id: str
    point_type: str = "center"
    rack_format: str = "24-vial rack"
    row: Optional[str] = None
    col: Optional[int] = None
    slot: int = 1
    aspiration_rate_ul_s: float = 50.0
    dispense_rate_ul_s: float = 50.0
    additive_conc_mM: float
    salt_conc_M: float
    target_volume_uL: Optional[float] = None
    temperature_C: Optional[float] = None
    analysis_report: Optional[ElectrochemicalAnalysisReport] = None
    performance_score: Optional[float] = None
    is_pareto_optimal: bool = False
    status: str = "pending_cycling"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    iteration_round: int = 0


class CampaignResult(BaseModel):
    campaign_id: str
    target_salt: str
    discovery_summary: Dict[str, Any] = Field(default_factory=dict)
    formulation_plan: DoEPlan
    dispensing_schedule: DispensingSchedule
    cell_results: List[FormulatedCellResult] = Field(default_factory=list)
    pareto_optimal_candidates: List[FormulatedCellResult] = Field(default_factory=list)
    best_candidate: Optional[FormulatedCellResult] = None
    output_dir: str
    created_at: Optional[str] = None
    iteration_round: int = 0
    total_iterations: int = 1
    history: List[Dict[str, Any]] = Field(default_factory=list)


DoEPlan.model_rebuild()
CampaignResult.model_rebuild()

