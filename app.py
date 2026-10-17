"""
Autonomous Materials Acceleration Platform (MAP) for Aqueous Zinc-Ion Batteries (ZIB).
Interactive Streamlit Dashboard (app.py)

Author & Lead Developer: Mark Cui
AI Research & Coding Partner: Gemini 3.8 Flash
Development Platform: Google Antigravity & Visual Studio Code
Core Technologies: RDKit, Opentrons OT-2, Plotly, Streamlit, Pydantic v2, Pytest (120/120 Passing Tests)
"""

import os
import sys
import re
import json
import time
import logging
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem, rdMolDescriptors, Lipinski

logger = logging.getLogger(__name__)

# Project modules & schemas
from models import (
    CampaignResult,
    FormulatedCellResult,
    ExtractionResult,
    ExtractedAdditive,
    CandidateMolecule,
    Concentration,
    ZincIonBatteryContext,
    ConcentrationAnchor,
    DoEPlan,
    DispensingSchedule,
    ElectrochemicalAnalysisReport,
    ElectrochemicalTechnique,
    ElectrodeGeometry,
    NormalizedElectrochemicalData,
    CyclerRawData,
    SymmetricCellAnalysis,
    AsymmetricHalfCellAnalysis,
    FullCellAnalysis,
    LSVAnalysis,
    EISAnalysis,
    CVAnalysis,
    PaperMetadata,
)
from map_orchestrator import (
    MAPOrchestrator,
    create_benchmark_cycler_file,
    sync_custom_formulation_pipeline,
)
from optimization.concentration_anchor import (
    ConcentrationAnchorEngine,
    CANONICAL_ELECTROLYTE_PROFILES,
    ELECTROLYTE_SCALING_RULES,
)
from optimization.doe_engine import DoEEngine
from optimization.dispenser import RoboticDispenserEngine
from discovery.similarity import (
    CURATED_ZIB_ADDITIVES,
    RDKitSimilarityEngine,
    SUPPORTED_SALTS,
    calculate_delaney_esol,
    detect_functional_groups,
    assess_electrolyte_compatibility,
    generate_chemoinformatic_rationale,
)
from discovery.search import SemanticScholarClient, clean_scientific_text, OFFLINE_SAMPLE_PAPERS
from discovery.extractor import PaperExtractor
from discovery.literature_engine import LiteratureDiscoveryEngine
from discovery.additive_importer import (
    parse_additive_file,
    deduplicate_and_resolve_additives,
    generate_template_csv,
    generate_template_xlsx,
)
from discovery.pubchem_client import is_valid_smiles, canonicalize_smiles
from analysis.analyzer_cli import run_analysis
from analysis.processing.math_engine import MathEngine
from analysis.visualization.plot_engine import ElectrochemicalPlotEngine
from analysis.parsers.template_generators import (
    generate_template_biologic_mpt,
    generate_template_landt_csv,
    generate_template_arbin_xlsx,
)
from localization import get_text
from export_engine import (
    generate_dispensing_schedule_xlsx,
    generate_dispensing_schedule_docx,
    generate_dispensing_schedule_txt,
    generate_doe_plan_xlsx,
    generate_doe_plan_docx,
    generate_doe_plan_txt,
    generate_campaign_summary_xlsx,
    generate_campaign_summary_docx,
    generate_campaign_summary_txt,
    generate_electrochemical_dataset_csv,
    generate_electrochemical_dataset_xlsx,
    generate_mined_literature_csv,
    generate_mined_literature_xlsx,
)
from ui.styles import CUSTOM_CSS, inject_global_styles
from ui.i18n import (
    tr,
    format_failure_mode,
    SALT_DISPLAY_MAPS,
    SALT_DISPLAY_MAP,
    FG_DISPLAY_MAP,
    format_fg_label,
    get_candidate_provenance,
)
from ui.components import (
    get_all_additives,
    get_similarity_engine,
    cached_generate_template_csv,
    cached_generate_template_xlsx,
    get_curated_molecules_df,
    render_rdkit_mol_image,
    compute_similarity_ranking,
    _apply_card_candidate,
    is_formulation_reminder_hidden,
    _render_post_extraction_reminder_dialog,
    _render_upload_popover,
    _merge_candidates_into_session,
    _render_candidate_chemical_dossier,
    render_two_tier_card_deck,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Global Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Autonomous ZIB Materials Acceleration Platform (MAP)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_styles()


# -----------------------------------------------------------------------------
# Cached Utility Functions & Formula Re-exports
# -----------------------------------------------------------------------------
from discovery.formula_utils import (
    CURATED_FORMULAS,
    PERIODIC_TABLE_ELEMENTS,
    KNOWN_FORMULA_ABBREVIATIONS,
    INORGANIC_METALS,
    INORGANIC_ELEMENT_PRIORITY,
    SUB_TO_DIGIT,
    DIGIT_TO_SUB,
    is_chemical_formula,
    mol_to_conventional_formula,
    normalize_formula_to_conventional,
    get_chemical_formula,
    format_chem_formula,
    normalize_condition_key,
)


def _hash_doe_plan(plan: DoEPlan) -> str:
    return plan.model_dump_json()


def _hash_dispensing_schedule(schedule: DispensingSchedule) -> str:
    return schedule.model_dump_json()


def _hash_analysis_report(report: ElectrochemicalAnalysisReport) -> Tuple[Any, ...]:
    d = report.processed_data
    v_len = len(d.voltage_v) if (d and d.voltage_v) else 0
    v_first = d.voltage_v[0] if v_len > 0 else None
    v_last = d.voltage_v[-1] if v_len > 0 else None
    return (
        "v2_html_toolbar",
        report.sample_id,
        report.file_source,
        str(report.technique),
        report.created_at,
        v_len,
        v_first,
        v_last,
        json.dumps(report.summary_metrics, sort_keys=True, default=str),
    )


def _hash_campaign_result(campaign: CampaignResult) -> Tuple[Any, ...]:
    return (
        campaign.campaign_id,
        campaign.created_at,
        campaign.target_salt,
        campaign.total_iterations,
        len(campaign.cell_results),
        campaign.formulation_plan.model_dump_json(),
        campaign.dispensing_schedule.model_dump_json(),
        tuple((c.sample_id, c.performance_score, c.status) for c in campaign.cell_results),
    )


EXPORT_HASH_FUNCS = {
    DoEPlan: _hash_doe_plan,
    DispensingSchedule: _hash_dispensing_schedule,
    ElectrochemicalAnalysisReport: _hash_analysis_report,
    CampaignResult: _hash_campaign_result,
}


@st.cache_data
def cached_generate_template_biologic_mpt() -> bytes:
    return generate_template_biologic_mpt()


@st.cache_data
def cached_generate_template_landt_csv() -> bytes:
    return generate_template_landt_csv()


@st.cache_data
def cached_generate_template_arbin_xlsx() -> bytes:
    return generate_template_arbin_xlsx()


@st.cache_data(hash_funcs=EXPORT_HASH_FUNCS)
def cached_generate_dispensing_schedule_xlsx(schedule: DispensingSchedule) -> bytes:
    return generate_dispensing_schedule_xlsx(schedule)


@st.cache_data(hash_funcs=EXPORT_HASH_FUNCS)
def cached_generate_dispensing_schedule_docx(schedule: DispensingSchedule) -> bytes:
    return generate_dispensing_schedule_docx(schedule)


@st.cache_data(hash_funcs=EXPORT_HASH_FUNCS)
def cached_generate_dispensing_schedule_txt(schedule: DispensingSchedule) -> str:
    return generate_dispensing_schedule_txt(schedule)


@st.cache_data(hash_funcs=EXPORT_HASH_FUNCS)
def cached_dispensing_schedule_json(schedule: DispensingSchedule) -> str:
    return schedule.model_dump_json(indent=2)


@st.cache_data(hash_funcs=EXPORT_HASH_FUNCS)
def cached_generate_doe_plan_xlsx(plan: DoEPlan) -> bytes:
    return generate_doe_plan_xlsx(plan)


@st.cache_data(hash_funcs=EXPORT_HASH_FUNCS)
def cached_generate_doe_plan_docx(plan: DoEPlan) -> bytes:
    return generate_doe_plan_docx(plan)


@st.cache_data(hash_funcs=EXPORT_HASH_FUNCS)
def cached_generate_doe_plan_txt(plan: DoEPlan) -> str:
    return generate_doe_plan_txt(plan)


@st.cache_data(hash_funcs=EXPORT_HASH_FUNCS)
def cached_doe_plan_json(plan: DoEPlan) -> str:
    return plan.model_dump_json(indent=2)


@st.cache_data(hash_funcs=EXPORT_HASH_FUNCS)
def cached_generate_electrochemical_dataset_xlsx(report: ElectrochemicalAnalysisReport) -> bytes:
    return generate_electrochemical_dataset_xlsx(report)


@st.cache_data(hash_funcs=EXPORT_HASH_FUNCS)
def cached_generate_electrochemical_dataset_csv(report: ElectrochemicalAnalysisReport) -> bytes:
    return generate_electrochemical_dataset_csv(report)


@st.cache_data(hash_funcs=EXPORT_HASH_FUNCS)
def cached_generate_campaign_summary_xlsx(campaign: CampaignResult) -> bytes:
    return generate_campaign_summary_xlsx(campaign)


@st.cache_data(hash_funcs=EXPORT_HASH_FUNCS)
def cached_generate_campaign_summary_docx(campaign: CampaignResult) -> bytes:
    return generate_campaign_summary_docx(campaign)


@st.cache_data(hash_funcs=EXPORT_HASH_FUNCS)
def cached_generate_campaign_summary_txt(campaign: CampaignResult) -> str:
    return generate_campaign_summary_txt(campaign)


@st.cache_data(hash_funcs=EXPORT_HASH_FUNCS)
def cached_campaign_summary_json(campaign: CampaignResult) -> str:
    return campaign.model_dump_json(indent=2)


@st.cache_data(hash_funcs=EXPORT_HASH_FUNCS)
def cached_render_analysis_report_html(report: ElectrochemicalAnalysisReport) -> str:
    plot_engine = ElectrochemicalPlotEngine(output_dir="./reports")
    report_html_path = plot_engine.render_report(report)
    return report_html_path.read_text(encoding="utf-8")


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "app_language" not in st.session_state:
    st.session_state.app_language = "en"

if "campaign_result" not in st.session_state:
    st.session_state.campaign_result = None

if "benchmark_analysis_report" not in st.session_state:
    st.session_state.benchmark_analysis_report = None

if "custom_analysis_report" not in st.session_state:
    st.session_state.custom_analysis_report = None


# -----------------------------------------------------------------------------
# Sidebar: Configuration, Automation Parameters & Project Credits
# -----------------------------------------------------------------------------
with st.sidebar:
    # Responsive Vector Logo (renders crisp vector in both light and dark mode)
    logo_svg_file = Path(__file__).resolve().parent / "assets" / "logo.svg"
    logo_png_file = Path(__file__).resolve().parent / "assets" / "logo.png"
    if logo_svg_file.exists():
        svg_code = logo_svg_file.read_text(encoding="utf-8")
        svg_clean = "".join(line.strip() for line in svg_code.splitlines())
        st.markdown(f'<div class="sidebar-logo-container">{svg_clean}</div>', unsafe_allow_html=True)
    elif logo_png_file.exists():
        st.image(str(logo_png_file), use_container_width=True)
    else:
        st.markdown("## ⚡ **ZIB · MAP**")

    # Language Switcher (Native English, French, Chinese)
    lang_options = ["en", "fr", "zh"]
    lang_names = {
        "en": "English",
        "fr": "Français (French)",
        "zh": "中文 (Chinese)",
    }
    cur_lang = st.session_state.get("app_language", "en")
    cur_idx = lang_options.index(cur_lang) if cur_lang in lang_options else 0
    selected_lang = st.selectbox(
        "🌐 Language / Langue / 语言",
        lang_options,
        index=cur_idx,
        format_func=lambda x: lang_names.get(x, x),
        key="app_language",
        help="Select platform interface language (English, Français, 中文). Chemical formulas and scientific terms remain standard."
    )
    cur_lang = selected_lang
    prev_rendered_lang = st.session_state.get("_last_rendered_app_language")
    lang_changed = (prev_rendered_lang is not None and prev_rendered_lang != cur_lang)
    st.session_state["_last_rendered_app_language"] = cur_lang
    st.session_state["_lang_changed_this_run"] = lang_changed
    if lang_changed:
        for _loc_wkey in (
            "app_operating_mode",
            "lit_search_sort_select",
            "lit_search_limit_select",
            "step2_target_electrolyte_select",
            "step3_technique_selector",
            "step3_view_mode_radio",
            "cv_cyc_filter",
            "step2_tier1_additive_dropdown",
            "step2_tier1_bottom_dropdown",
            "tab2_suggested_additives_dropdown",
            "step2_tier2_bottom_dropdown",
            "upload_mode_radio_t1",
            "upload_mode_radio_t1_empty",
        ):
            if _loc_wkey in st.session_state:
                st.session_state[_loc_wkey] = st.session_state[_loc_wkey]

    st.markdown(f"### **{tr('sidebar_control_center')}**")
    st.caption(tr("sidebar_caption"))

    st.markdown("---")
    st.markdown(f"#### **{tr('sidebar_exp_setup')}**")

    mode_options = ["demo", "custom"]
    mode_labels = {
        "demo": tr("mode_demo"),
        "custom": tr("mode_custom"),
    }
    if "app_operating_mode" not in st.session_state or st.session_state.get("app_operating_mode") not in mode_options:
        st.session_state["app_operating_mode"] = "custom"
    app_mode_key = st.radio(
        tr("sidebar_operating_mode"),
        mode_options,
        format_func=lambda x: mode_labels.get(x, x),
        help=tr("mode_demo_help"),
        key="app_operating_mode",
    )
    is_demo = (app_mode_key == "demo")

    if is_demo:
        salt_options = list(CANONICAL_ELECTROLYTE_PROFILES.keys())
        cur_sb_salt = st.session_state.get("selected_target_salt", "Zn(OTf)2")
        cur_sb_idx = salt_options.index(cur_sb_salt) if cur_sb_salt in salt_options else 1
        selected_target_salt = st.selectbox(
            tr("target_electrolyte"),
            salt_options,
            index=cur_sb_idx,
            format_func=lambda s: SALT_DISPLAY_MAPS.get(cur_lang, SALT_DISPLAY_MAP).get(s, format_chem_formula(s)),
            help=tr("target_electrolyte_help"),
        )
        st.session_state["selected_target_salt"] = selected_target_salt

        all_additives = get_all_additives()
        additive_pool_names = [a["name"] for a in all_additives]

        cur_sel_name = st.session_state.get("selected_additive_name", additive_pool_names[0])
        cur_sel_idx = additive_pool_names.index(cur_sel_name) if cur_sel_name in additive_pool_names else 0

        selected_additive_name = st.selectbox(
            tr("additive_to_test"),
            additive_pool_names,
            index=cur_sel_idx,
            help=tr("additive_to_test_help"),
        )
        st.session_state["selected_additive_name"] = selected_additive_name

        matching_additive = next((a for a in all_additives if a["name"] == selected_additive_name), all_additives[0])
        selected_smiles = matching_additive["smiles"]

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            default_conc = float(st.session_state.get("published_conc_input", 15.0))
            published_conc_input = st.number_input(
                tr("reported_conc"),
                min_value=0.5,
                max_value=1000.0,
                value=default_conc,
                step=1.0,
                help=tr("reported_conc_help"),
            )
            st.session_state["published_conc_input"] = published_conc_input
        with col_s2:
            default_parent = str(st.session_state.get("parent_electrolyte_input", "2 M ZnSO₄"))
            parent_electrolyte_input = st.text_input(
                tr("baseline_electrolyte"),
                value=default_parent,
                help=tr("baseline_electrolyte_help"),
            )
            st.session_state["parent_electrolyte_input"] = parent_electrolyte_input

        st.markdown(f"##### **{tr('opt_header')}**")
        closed_loop_iterations = st.slider(
            tr("opt_rounds"),
            min_value=1,
            max_value=4,
            value=int(st.session_state.get("closed_loop_iterations", 2)),
            step=1,
            help=tr("opt_rounds_help"),
        )
        st.session_state["closed_loop_iterations"] = closed_loop_iterations

        doe_labels = {"lhs": tr("doe_lhs"), "ccd": tr("doe_ccd")}
        doe_method = st.selectbox(
            tr("doe_method"),
            ["lhs", "ccd"],
            index=0 if st.session_state.get("doe_method", "lhs") == "lhs" else 1,
            format_func=lambda x: doe_labels.get(x, x),
            help=tr("doe_method_help"),
        )
        st.session_state["doe_method"] = doe_method

        num_doe_samples = st.slider(
            tr("batch1_samples"),
            min_value=3,
            max_value=12,
            value=int(st.session_state.get("num_doe_samples", 3)),
            step=1,
            help=tr("batch1_samples_help"),
        )
        st.session_state["num_doe_samples"] = num_doe_samples

        if closed_loop_iterations >= 2:
            adaptive_samples_per_round = st.slider(
                tr("followup_samples"),
                min_value=1,
                max_value=6,
                value=int(st.session_state.get("adaptive_samples_per_round", 2)),
                step=1,
                help=tr("followup_samples_help"),
            )
            st.session_state["adaptive_samples_per_round"] = adaptive_samples_per_round
            al_strategy_keys = ["fastest", "refine", "balanced"]
            al_strategy_map = {
                "fastest": tr("al_opt_fastest"),
                "refine": tr("al_opt_refine"),
                "balanced": tr("al_opt_balanced"),
            }
            cur_al_code = st.session_state.get("al_strategy_code", "fastest")
            cur_al_idx = al_strategy_keys.index(cur_al_code) if cur_al_code in al_strategy_keys else 0
            al_strategy_code = st.selectbox(
                tr("al_strategy"),
                al_strategy_keys,
                index=cur_al_idx,
                format_func=lambda k: al_strategy_map.get(k, k),
                help=tr("al_strategy_help"),
            )
            st.session_state["al_strategy_code"] = al_strategy_code
            al_strategy = al_strategy_map[al_strategy_code]
        else:
            adaptive_samples_per_round = 2
            al_strategy_code = "fastest"
            al_strategy = tr("al_opt_fastest")
            st.caption(tr("al_disabled_caption"))

        rack_options = ["24-vial rack", "96-well plate"]
        rack_labels = {"24-vial rack": tr("rack_24"), "96-well plate": tr("rack_96")}
        rack_format = st.selectbox(
            tr("rack_format"),
            rack_options,
            index=0 if st.session_state.get("rack_format", "24-vial rack") == "24-vial rack" else 1,
            format_func=lambda x: rack_labels.get(x, x),
            help=tr("rack_format_help"),
        )
        st.session_state["rack_format"] = rack_format

        total_vol_ul = st.number_input(
            tr("target_volume"),
            min_value=50.0,
            max_value=10000.0,
            value=float(st.session_state.get("total_vol_ul", 2000.0 if rack_format == "24-vial rack" else 200.0)),
            step=100.0,
            help=tr("target_volume_help"),
        )
        st.session_state["total_vol_ul"] = total_vol_ul
    else:
        # Custom Experiment Mode: clean defaults from session state (no preloaded literature additive)
        selected_target_salt = st.session_state.get("selected_target_salt", "Zn(OTf)2")
        custom_name_default = st.session_state.get("custom_additive_name", "")
        custom_smiles_default = st.session_state.get("custom_smiles", "")
        selected_additive_name = custom_name_default
        selected_smiles = custom_smiles_default
        published_conc_input = float(st.session_state.get("published_conc_input", 15.0))
        parent_electrolyte_input = str(st.session_state.get("parent_electrolyte_input", "2 M ZnSO₄"))
        closed_loop_iterations = int(st.session_state.get("closed_loop_iterations", 2))
        doe_method = str(st.session_state.get("doe_method", "lhs"))
        num_doe_samples = int(st.session_state.get("num_doe_samples", 3))
        adaptive_samples_per_round = int(st.session_state.get("adaptive_samples_per_round", 2))
        al_strategy_code = str(st.session_state.get("al_strategy_code", "fastest"))
        al_strategy = tr("al_opt_fastest")
        rack_format = str(st.session_state.get("rack_format", "24-vial rack"))
        total_vol_ul = float(st.session_state.get("total_vol_ul", 2000.0))

    st.markdown("---")

    # Project Credits & Tech Stack
    st.markdown(f'<div class="sidebar-section-title">👨‍🔬 {tr("credits_header")}</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sidebar-credits-box">
        <div style="margin-bottom: 5px;"><strong>{tr("author_label")}</strong>: <strong>Mark Cui</strong></div>
        <div style="margin-bottom: 6px;">
            <div><strong>{tr("partner_label")}</strong>:</div>
            <div style="padding-left: 10px; line-height: 1.45; margin-top: 2px;">
                <div>• <strong>Gemini 3.8 Flash</strong></div>
                <div>• <strong>Gemini 3.1 Pro</strong></div>
                <div>• <strong>Gemini 4 Argon</strong></div>
            </div>
        </div>
        <div style="margin-bottom: 10px;"><strong>{tr("platform_label")}</strong>: Google Antigravity & VS Code</div>
        <div style="font-weight: 600; margin-bottom: 6px; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.4px;">{tr("core_tech_label")}</div>
        <div class="tech-icon-grid">
            <div class="tech-pill" title="RDKit Chemoinformatics & Delaney ESOL">
                <svg viewBox="0 0 24 24" width="16" height="16" style="min-width: 16px; margin-right: 5px;"><path fill="#00A896" d="M19 19a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V9a2 2 0 0 1 .59-1.41L9 4.17V3a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v1.17l3.41 3.42A2 2 0 0 1 19 9v10zm-9-13.59L6.59 8.83A.5.5 0 0 0 6.41 9.17V19a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1V9.17a.5.5 0 0 0-.18-.34L14 5.41V4h-4v1.41z"/></svg>
                <span>RDKit</span>
            </div>
            <div class="tech-pill" title="Opentrons OT-2 Liquid Handling Automation">
                <svg viewBox="0 0 24 24" width="16" height="16" style="min-width: 16px; margin-right: 5px;"><path fill="#02C39A" d="M12 2a2 2 0 0 1 2 2v2h4a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4V4a2 2 0 0 1 2-2zm-1 6H6v4h5V8zm7 0h-5v4h5V8zM6 14v4h5v-4H6zm7 0v4h5v-4h-5z"/></svg>
                <span>OT-2</span>
            </div>
            <div class="tech-pill" title="Google Gemini LLM Literature & PDF Extraction">
                <svg viewBox="0 0 24 24" width="16" height="16" style="min-width: 16px; margin-right: 5px;"><path fill="#8E75B2" d="M12 2C12 7.52 16.48 12 22 12C16.48 12 12 16.48 12 22C12 16.48 7.52 12 2 12C7.52 12 12 7.52 12 2Z"/></svg>
                <span>Gemini AI</span>
            </div>
            <div class="tech-pill" title="NCBI PubChem REST PUG Structure & Similarity API">
                <svg viewBox="0 0 24 24" width="16" height="16" style="min-width: 16px; margin-right: 5px;"><path fill="#2B7BBA" d="M12 2l8.66 5v10L12 22l-8.66-5V7L12 2zm0 2.31L5.34 8.15v7.7L12 19.69l6.66-3.84v-7.7L12 4.31zm0 2.69a5 5 0 1 1 0 10 5 5 0 0 1 0-10zm0 2a3 3 0 1 0 0 6 3 3 0 0 0 0-6z"/></svg>
                <span>PubChem</span>
            </div>
            <div class="tech-pill" title="Plotly Interactive Visualizations">
                <svg viewBox="0 0 24 24" width="16" height="16" style="min-width: 16px; margin-right: 5px;"><path fill="#636EFA" d="M3 3v18h18v-2H5V3H3zm4 14h2v-5H7v5zm4 0h2V7h-2v10zm4 0h2v-8h-2v8z"/></svg>
                <span>Plotly</span>
            </div>
            <div class="tech-pill" title="Pydantic v2 Schema Validation">
                <svg viewBox="0 0 24 24" width="16" height="16" style="min-width: 16px; margin-right: 5px;"><path fill="#E92063" d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm-2 16l-4-4 1.41-1.41L10 14.17l6.59-6.59L18 9l-8 8z"/></svg>
                <span>Pydantic v2</span>
            </div>
            <div class="tech-pill" title="NumPy & Pandas Electrochemical Signal Processing">
                <svg viewBox="0 0 24 24" width="16" height="16" style="min-width: 16px; margin-right: 5px;"><path fill="#4DABCF" d="M4 4h6v6H4V4zm10 0h6v6h-6V4zM4 14h6v6H4v-6zm10 0h6v6h-6v-6z"/></svg>
                <span>NumPy / Pandas</span>
            </div>
            <div class="tech-pill" title="Streamlit Dashboard Framework">
                <svg viewBox="7 4 19 11" width="18" height="11" style="min-width: 18px; margin-right: 5px;">
                    <path fill="#FF4B4B" d="M16.8193 10.7723L13.7268 9.19202L8.35854 6.44935C8.35363 6.44461 8.34383 6.44461 8.33893 6.44461C8.14292 6.35461 7.94201 6.54409 8.01552 6.73356L10.7508 13.4775L10.7513 13.4789C10.7543 13.4856 10.7567 13.4922 10.7597 13.4988C10.8719 13.7504 11.1169 13.9057 11.3835 13.9668C11.406 13.9716 11.4221 13.9759 11.4492 13.9811C11.4761 13.9869 11.5138 13.9948 11.5462 13.9972C11.5515 13.9976 11.5564 13.9976 11.5618 13.9981H11.5658C11.5697 13.9986 11.5736 13.9986 11.5775 13.9991H11.5829C11.5863 13.9995 11.5903 13.9995 11.5937 13.9995H11.6001C11.604 14 11.6079 14 11.6118 14H22.0352C22.0768 14 22.1175 13.9981 22.1567 13.9943C22.1694 13.9929 22.1817 13.9915 22.1939 13.9901C22.1954 13.9896 22.1974 13.9896 22.1988 13.9891C22.2072 13.9882 22.2155 13.9867 22.2238 13.9853C22.2361 13.9839 22.2483 13.9815 22.2606 13.9792C22.2851 13.9739 22.2962 13.9702 22.3295 13.9591C22.3627 13.9481 22.4179 13.9287 22.4525 13.9126C22.487 13.8965 22.511 13.8811 22.5399 13.8636C22.5757 13.8418 22.6097 13.8195 22.6442 13.7945C22.6591 13.7835 22.6693 13.7764 22.68 13.7665L22.6742 13.7632L16.8193 10.7723Z"/>
                    <path fill="#BD1E2C" d="M25.2891 6.4494H25.2842L19.9139 9.19208L22.8992 13.4899L25.6277 6.73362V6.72414C25.6963 6.52519 25.4856 6.34519 25.2891 6.4494"/>
                    <path fill="#FF7C7C" d="M17.1055 5.14529C16.9683 4.95155 16.6694 4.95155 16.5371 5.14529L13.7268 9.19205L16.8194 10.7723L22.6801 13.7664C22.7169 13.7316 22.7464 13.6977 22.7776 13.6618C22.8217 13.6092 22.8629 13.5529 22.8991 13.4899L19.9139 9.19205L17.1055 5.14529Z"/>
                </svg>
                <span>Streamlit</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main Header & Dashboard Navigation
# -----------------------------------------------------------------------------
st.title(tr("app_title"))
st.markdown(
    f'<p class="hero-sub">{tr("app_hero_sub")}</p>',
    unsafe_allow_html=True,
)

if is_demo:
    tabs = st.tabs([
        tr("tab_run"),
        tr("tab_step1"),
        tr("tab_step2"),
        tr("tab_step3"),
        tr("tab_summary"),
    ])
    tab_run = tabs[0]
    tab_step1 = tabs[1]
    tab_step2 = tabs[2]
    tab_step3 = tabs[3]
    tab_summary = tabs[4]
else:
    tabs = st.tabs([
        tr("tab_step1_custom"),
        tr("tab_step2_custom"),
        tr("tab_step3_custom"),
    ])
    tab_run = None
    tab_step1 = tabs[0]
    tab_step2 = tabs[1]
    tab_step3 = tabs[2]
    tab_summary = None

# Client-side persistence check for map_hide_formulation_reminder
st.components.v1.html(
    """
    <script>
    (function() {
        try {
            const hide = window.parent.localStorage.getItem('map_hide_formulation_reminder');
            if (hide === 'true') {
                if (!window.parent.document.cookie.includes('map_hide_formulation_reminder=true')) {
                    window.parent.document.cookie = 'map_hide_formulation_reminder=true; path=/; max-age=31536000; SameSite=Lax';
                }
            }
        } catch(e) {}
    })();
    </script>
    """,
    height=0,
    width=0,
)

if st.session_state.pop("sync_hide_formulation_reminder_storage", False):
    st.components.v1.html(
        """
        <script>
        (function() {
            try {
                window.parent.localStorage.setItem('map_hide_formulation_reminder', 'true');
                window.parent.document.cookie = 'map_hide_formulation_reminder=true; path=/; max-age=31536000; SameSite=Lax';
            } catch(e) {}
        })();
        </script>
        """,
        height=0,
        width=0,
    )

if st.session_state.pop("trigger_switch_to_formulation_tab", False):
    target_tab_idx = 2 if is_demo else 1
    st.components.v1.html(
        f"""
        <script>
        (function() {{
            const parentDoc = window.parent.document;
            function switchToFormulationTab() {{
                const mainTabs = parentDoc.querySelector('div[data-testid="stTabs"]');
                if (!mainTabs) return false;
                const tabs = mainTabs.querySelectorAll('button[role="tab"], button[data-baseweb="tab"]');
                if (!tabs || tabs.length === 0) return false;
                
                for (let i = 0; i < tabs.length; i++) {{
                    const text = (tabs[i].textContent || "").toLowerCase();
                    if (text.includes("formulation") || text.includes("配方")) {{
                        tabs[i].click();
                        return true;
                    }}
                }}
                
                if (tabs[{target_tab_idx}]) {{
                    tabs[{target_tab_idx}].click();
                    return true;
                }}
                return false;
            }}
            
            if (!switchToFormulationTab()) {{
                setTimeout(switchToFormulationTab, 50);
                setTimeout(switchToFormulationTab, 150);
                setTimeout(switchToFormulationTab, 350);
            }}
        }})();
        </script>
        """,
        height=0,
        width=0,
    )



# =============================================================================
# TAB 1: AUTOMATED EXPERIMENT RUN
# =============================================================================
if is_demo and tab_run is not None:
    with tab_run:
        st.markdown(f"### **{tr('tab1_title')}**")
        st.markdown(tr("tab1_desc"))

        col_btn, col_info = st.columns([1.5, 2.5])
        with col_btn:
            run_campaign_btn = st.button(tr("btn_run_campaign"), type="primary", use_container_width=True)
        with col_info:
            round_info = f"Closed-Loop: {closed_loop_iterations} Rounds ({num_doe_samples} in Batch 1 + {adaptive_samples_per_round} in follow-up rounds)" if closed_loop_iterations >= 2 else f"Single Pass: {num_doe_samples} points"
            st.caption(f"**Experiment Setup:** {selected_additive_name} ({published_conc_input} mM in {format_chem_formula(parent_electrolyte_input)}) ➔ Transferred to **{format_chem_formula(selected_target_salt)}** using {doe_method.upper()} | {round_info}.")

        status_container = st.container()

        if run_campaign_btn:
            with status_container:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f"#### **{tr('tracker_title')}**")
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                orchestrator = MAPOrchestrator(output_dir="./campaign_results/demo_campaign")

                if closed_loop_iterations >= 2:
                    # Step 1: Round 0 Discovery
                    status_text.markdown("🔄 **[Round 0 - Step 1/3] Discovery & Solubility Check:** Ingesting literature baseline & calculating Delaney ESOL 80% saturation safety cap...")
                    progress_bar.progress(0.15)

                    # Step 2: Round 0 Formulation
                    status_text.markdown(f"🔄 **[Round 0 - Step 2/3] Formulation Matrix:** Scaling concentration to {format_chem_formula(selected_target_salt)}, generating {num_doe_samples} initial formulations (Vials A1–A{num_doe_samples})...")
                    progress_bar.progress(0.35)

                    # Step 3: Round 0 Evaluation
                    status_text.markdown("🔄 **[Round 0 - Step 3/3] Battery Cycling Evaluation:** Cycling Round 0 cells & evaluating two-stage performance metrics (stable cycling vs short-circuit)...")
                    progress_bar.progress(0.55)

                    # Step 4: Round 1 Response Surface & Active Learning
                    start_w = num_doe_samples + 1
                    end_w = num_doe_samples + adaptive_samples_per_round
                    status_text.markdown(f"🔄 **[Round 1 - Active Learning] Surface Modeling:** Modeling performance gradients ({al_strategy}), allocating follow-up Vials A{start_w}–A{end_w} with continuous rack coordinates...")
                    progress_bar.progress(0.75)

                    # Step 5: Round 1 Adaptive Evaluation
                    status_text.markdown("🔄 **[Round 1 - Step 3/3] Follow-up Cycling:** Evaluating Round 1 candidates & identifying top-performing formulations (Pareto frontier)...")
                    progress_bar.progress(0.90)

                    campaign_res = orchestrator.run_closed_loop_campaign(
                        from_extraction="samples/test_extraction.json" if is_demo else None,
                        additive_name=selected_additive_name or None,
                        published_conc_mM=published_conc_input,
                        smiles=selected_smiles or None,
                        parent_electrolyte=parent_electrolyte_input,
                        target_salt=selected_target_salt,
                        doe_method=doe_method,
                        initial_samples=num_doe_samples,
                        max_iterations=closed_loop_iterations,
                        samples_per_iteration=adaptive_samples_per_round,
                        rack_format=rack_format,
                        total_volume_uL=total_vol_ul,
                        use_benchmark=True,
                        seed=42,
                        al_strategy=al_strategy_code,
                    )
                else:
                    status_text.markdown("🔄 **[Step 1/3] Discovery & Solubility Check:** Ingesting literature baseline & calculating Delaney ESOL 80% saturation safety cap...")
                    progress_bar.progress(0.25)

                    status_text.markdown(f"🔄 **[Step 2/3] Formulation Matrix:** Scaling concentration to {format_chem_formula(selected_target_salt)}, computing {num_doe_samples}-point {doe_method.upper()} matrix & Opentrons OT-2 layout...")
                    progress_bar.progress(0.60)

                    status_text.markdown("🔄 **[Step 3/3] Battery Cycling Evaluation:** Ingesting cycling runs, computing two-stage performance metrics & Pareto frontier...")
                    progress_bar.progress(0.85)

                    campaign_res = orchestrator.run_full_campaign(
                        from_extraction="samples/test_extraction.json" if is_demo else None,
                        additive_name=selected_additive_name or None,
                        published_conc_mM=published_conc_input,
                        smiles=selected_smiles or None,
                        parent_electrolyte=parent_electrolyte_input,
                        target_salt=selected_target_salt,
                        doe_method=doe_method,
                        num_samples=num_doe_samples,
                        rack_format=rack_format,
                        total_volume_uL=total_vol_ul,
                        use_benchmark=True,
                        seed=42,
                    )

                st.session_state.campaign_result = campaign_res
                progress_bar.progress(1.0)
                status_text.markdown(tr("opt_completed"))
                st.markdown('</div>', unsafe_allow_html=True)

        # If campaign has run (or load default pre-run)
        campaign: Optional[CampaignResult] = st.session_state.campaign_result

        if campaign is None:
            st.markdown(f'<div class="callout-box">{tr("campaign_callout")}</div>', unsafe_allow_html=True)
            with st.spinner("Pre-loading closed-loop active learning baseline..."):
                orch = MAPOrchestrator(output_dir="./campaign_results/demo_campaign")
                if closed_loop_iterations >= 2:
                    campaign = orch.run_closed_loop_campaign(
                        from_extraction="samples/test_extraction.json",
                        additive_name=selected_additive_name or None,
                        published_conc_mM=published_conc_input,
                        smiles=selected_smiles or None,
                        parent_electrolyte=parent_electrolyte_input,
                        target_salt=selected_target_salt,
                        doe_method=doe_method,
                        initial_samples=num_doe_samples,
                        max_iterations=closed_loop_iterations,
                        samples_per_iteration=adaptive_samples_per_round,
                        rack_format=rack_format,
                        total_volume_uL=total_vol_ul,
                        use_benchmark=True,
                        seed=42,
                        al_strategy=al_strategy_code,
                    )
                else:
                    campaign = orch.run_full_campaign(
                        from_extraction="samples/test_extraction.json",
                        additive_name=selected_additive_name or None,
                        published_conc_mM=published_conc_input,
                        smiles=selected_smiles or None,
                        parent_electrolyte=parent_electrolyte_input,
                        target_salt=selected_target_salt,
                        doe_method=doe_method,
                        num_samples=num_doe_samples,
                        rack_format=rack_format,
                        total_volume_uL=total_vol_ul,
                        use_benchmark=True,
                        seed=42,
                    )
                st.session_state.campaign_result = campaign

        if campaign:
            best: Optional[FormulatedCellResult] = campaign.best_candidate

            st.markdown(f'<div class="section-title">{tr("top_formulation_title")}</div>', unsafe_allow_html=True)

            if best and best.analysis_report and best.analysis_report.symmetric_cell:
                symm = best.analysis_report.symmetric_cell
                active_m = symm.active_cycling_metrics or {}
                total_m = symm.total_test_summary or {}

                active_life = active_m.get("active_lifespan_hours", symm.total_lifespan_hours)
                active_overpot = active_m.get("mean_overpotential_mv", symm.mean_overpotential_mv)
                active_cyc = active_m.get("active_cycles", symm.total_cycles)
                fail_mode = format_failure_mode(total_m.get("failure_mode", symm.failure_mode))
                best_round = getattr(best, "iteration_round", 0)

                st.markdown(f"""
                <div class="metric-card-hero">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <div>
                            <h3 class="hero-card-title">Sample {best.sample_id} ({best.vial_id}) — {tr('top_performer')}</h3>
                            <div class="hero-card-subtitle">
                                Discovered in <strong>Round {best_round}</strong> • ⭐ <strong>Best Balanced Formulation (Optimal Lifespan & Efficiency)</strong> • Evaluated during stable cycling (before failure)
                            </div>
                        </div>
                        <span class="badge-tag badge-gold" style="font-size: 1.0rem; padding: 8px 16px;">{tr('overall_score')}: {best.performance_score:.2f}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px;">
                        <div>
                            <div class="hero-stat-label">{tr('optimal_additive_conc')}</div>
                            <div class="hero-stat-val-teal">{best.additive_conc_mM:.1f} mM</div>
                            <div class="hero-stat-sub">{campaign.formulation_plan.additive_name}</div>
                        </div>
                        <div>
                            <div class="hero-stat-label">{tr('optimal_salt_conc')}</div>
                            <div class="hero-stat-val-blue">{best.salt_conc_M:.2f} M</div>
                            <div class="hero-stat-sub">{format_chem_formula(campaign.target_salt)}</div>
                        </div>
                        <div>
                            <div class="hero-stat-label">{tr('stable_cycling_lifespan')}</div>
                            <div class="hero-stat-val-green">{active_life:.1f} h</div>
                            <div class="hero-stat-sub">{active_cyc} cycles (Failure: {fail_mode})</div>
                        </div>
                        <div>
                            <div class="hero-stat-label">{tr('mean_overpotential')}</div>
                            <div class="hero-stat-val-amber">{active_overpot:.1f} mV</div>
                            <div class="hero-stat-sub">{tr('polarization_note')}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Candidates Performance Matrix & Pareto Visualizer
            st.markdown(f'<div class="section-title">{tr("evaluated_formulations_title")}</div>', unsafe_allow_html=True)

            cell_rows = []
            for cell in campaign.cell_results:
                rep = cell.analysis_report
                symm = rep.symmetric_cell if rep else None
                act = symm.active_cycling_metrics if symm else {}
                tot = symm.total_test_summary if symm else {}
                is_pareto = getattr(cell, "is_pareto_optimal", False)
                rnd = getattr(cell, "iteration_round", 0)
                cell_rows.append({
                    "Sample ID": cell.sample_id,
                    "Vial": cell.vial_id,
                    "Optimization Round": f"Round {rnd}",
                    "Frontier Status": tr("frontier_top") if is_pareto else tr("frontier_dominated"),
                    "Additive Conc (mM)": cell.additive_conc_mM,
                    "Salt Conc (M)": cell.salt_conc_M,
                    "Stable Lifespan (h)": act.get("active_lifespan_hours", symm.total_lifespan_hours if symm else 0.0),
                    "Mean Overpot (mV)": act.get("mean_overpotential_mv", symm.mean_overpotential_mv if symm else 0.0),
                    "Failure Mode": format_failure_mode(tot.get("failure_mode", symm.failure_mode if symm else "unknown")),
                    "Performance Score": cell.performance_score or 0.0,
                    "Status": cell.status,
                })
            df_cells = pd.DataFrame(cell_rows)

            col_tbl, col_chart = st.columns([1.5, 1.5])
            with col_tbl:
                max_score = float(df_cells["Performance Score"].max()) if not df_cells.empty and df_cells["Performance Score"].max() > 0 else 100.0
                st.dataframe(
                    df_cells,
                    column_config={
                        "Performance Score": st.column_config.ProgressColumn(
                            "Performance Score",
                            help="Scoring metric: Stable Lifespan / (1 + η / 50)",
                            format="%.2f",
                            min_value=0.0,
                            max_value=max_score,
                        ),
                        "Stable Lifespan (h)": st.column_config.NumberColumn("Stable Lifespan (h)", format="%.1f h"),
                        "Mean Overpot (mV)": st.column_config.NumberColumn("Mean Overpot (mV)", format="%.1f mV"),
                        "Frontier Status": st.column_config.TextColumn("Frontier Status", help="Top-performing formulations balancing lifespan and overpotential"),
                    },
                    hide_index=True,
                    use_container_width=True,
                    height=340,
                )

            with col_chart:
                # Interactive 2D Scatter / Bubble Plot showing the Top Performing Formulations (Pareto Frontier)
                fig_pareto = go.Figure()

                round_colors = {"Round 0": "#1f77b4", "Round 1": "#2ca02c", "Round 2": "#ff7f0e", "Round 3": "#9467bd"}
                for r_name in df_cells["Optimization Round"].unique():
                    sub_df = df_cells[df_cells["Optimization Round"] == r_name]
                    fig_pareto.add_trace(go.Scatter(
                        x=sub_df["Stable Lifespan (h)"],
                        y=sub_df["Mean Overpot (mV)"],
                        mode="markers+text",
                        name=r_name,
                        text=sub_df["Sample ID"],
                        textposition="top center",
                        marker=dict(
                            size=np.clip(sub_df["Performance Score"] / max_score * 22 + 10, 10, 32),
                            color=round_colors.get(r_name, "#17becf"),
                            opacity=0.85,
                            line=dict(width=1.5, color="#2b2d42"),
                        ),
                        customdata=np.stack((
                            sub_df["Sample ID"],
                            sub_df["Vial"],
                            sub_df["Additive Conc (mM)"],
                            sub_df["Salt Conc (M)"],
                            sub_df["Performance Score"],
                            sub_df["Frontier Status"],
                            sub_df["Failure Mode"],
                        ), axis=-1),
                        hovertemplate=(
                            "<b>Sample %{customdata[0]} (%{customdata[1]})</b><br>"
                            "Round: " + r_name + "<br>"
                            "Status: %{customdata[5]}<br>"
                            "Stable Lifespan: %{x:.1f} h<br>"
                            "Mean Overpotential: %{y:.1f} mV<br>"
                            "Performance Score: %{customdata[4]:.2f}<br>"
                            "Additive Conc: %{customdata[2]:.1f} mM<br>"
                            "Salt Conc: %{customdata[3]:.2f} M<br>"
                            "Failure Mode: %{customdata[6]}<extra></extra>"
                        ),
                    ))

                # Overlay Pareto Frontier envelope
                pareto_sub = df_cells[df_cells["Frontier Status"] == tr("frontier_top")].sort_values(by="Stable Lifespan (h)")
                if not pareto_sub.empty:
                    fig_pareto.add_trace(go.Scatter(
                        x=pareto_sub["Stable Lifespan (h)"],
                        y=pareto_sub["Mean Overpot (mV)"],
                        mode="lines+markers",
                        name=tr("frontier_top"),
                        line=dict(color="#f59f00", width=2.5, dash="dot"),
                        marker=dict(symbol="star", size=14, color="#f59f00", line=dict(width=1.5, color="#b25e00")),
                        hoverinfo="skip",
                    ))

                fig_pareto.update_layout(
                    title=tr("pareto_chart_title"),
                    xaxis_title=tr("pareto_x_axis"),
                    yaxis_title=tr("pareto_y_axis"),
                    template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig_pareto, use_container_width=True)


# =============================================================================
# TAB 2: STEP 1 DISCOVERY & SOLUBILITY LIMITS
# =============================================================================
with tab_step1:
    st.markdown(f"### **{tr('tab2_title' if is_demo else 'tab2_title_custom')}**")
    st.markdown(tr("tab2_desc"))

    # Literature Input Sub-Tabs: Online Academic Search & Batch Multi-PDF Ingestion
    lit_input_tabs = st.tabs([
        f"🌐 {tr('subtab_online_search')}",
        f"📄 {tr('subtab_batch_pdf')}",
    ])

    gemini_key_ver = st.session_state.get("gemini_key_version", 0)
    gemini_model_options_map = {
        tr("gemini_model_35_flash_lite"): "gemini-3.5-flash-lite",
        tr("gemini_model_31_flash_lite"): "gemini-3.1-flash-lite",
        tr("gemini_model_38_flash"): "gemini-3.8-flash",
        tr("gemini_model_31_pro"): "gemini-3.1-pro",
        tr("gemini_model_4_argon"): "gemini-4-argon",
        tr("gemini_model_custom_option"): "custom",
    }
    model_id_to_label = {v: k for k, v in gemini_model_options_map.items()}
    display_model_options = list(gemini_model_options_map.keys())

    online_model_sel_key = f"online_search_gemini_model_select_{gemini_key_ver}"
    batch_model_sel_key = f"batch_pdf_gemini_model_select_{gemini_key_ver}"
    online_custom_model_key = f"online_search_gemini_custom_model_input_{gemini_key_ver}"
    batch_custom_model_key = f"batch_pdf_gemini_custom_model_input_{gemini_key_ver}"

    # Synchronize model selection across subtabs BEFORE widgets are instantiated
    cur_model = st.session_state.get("gemini_selected_model", "gemini-3.5-flash-lite")
    last_synced = st.session_state.get("_last_synced_gemini_model")
    last_synced_lang = st.session_state.get("_last_synced_gemini_lang")

    if last_synced != cur_model or last_synced_lang != cur_lang:
        sync_lbl = model_id_to_label.get(cur_model, tr("gemini_model_custom_option"))
        st.session_state[online_model_sel_key] = sync_lbl
        st.session_state[batch_model_sel_key] = sync_lbl
        st.session_state["_prev_online_model_lbl"] = sync_lbl
        st.session_state["_prev_batch_model_lbl"] = sync_lbl
        st.session_state["_last_synced_gemini_model"] = cur_model
        st.session_state["_last_synced_gemini_lang"] = cur_lang
    else:
        cur_expected_lbl = model_id_to_label.get(cur_model, tr("gemini_model_custom_option"))
        online_lbl = st.session_state.get(online_model_sel_key)
        batch_lbl = st.session_state.get(batch_model_sel_key)
        prev_online = st.session_state.get("_prev_online_model_lbl", cur_expected_lbl)
        prev_batch = st.session_state.get("_prev_batch_model_lbl", cur_expected_lbl)

        if online_lbl and online_lbl != prev_online and online_lbl in gemini_model_options_map:
            new_id = gemini_model_options_map[online_lbl]
            st.session_state[batch_model_sel_key] = online_lbl
            st.session_state["_prev_online_model_lbl"] = online_lbl
            st.session_state["_prev_batch_model_lbl"] = online_lbl
            if new_id != "custom":
                st.session_state["gemini_selected_model"] = new_id
                st.session_state["_last_synced_gemini_model"] = new_id
            else:
                cust_val = st.session_state.get(online_custom_model_key, "").strip()
                if cust_val:
                    st.session_state["gemini_selected_model"] = cust_val
                    st.session_state["_last_synced_gemini_model"] = cust_val
        elif batch_lbl and batch_lbl != prev_batch and batch_lbl in gemini_model_options_map:
            new_id = gemini_model_options_map[batch_lbl]
            st.session_state[online_model_sel_key] = batch_lbl
            st.session_state["_prev_online_model_lbl"] = batch_lbl
            st.session_state["_prev_batch_model_lbl"] = batch_lbl
            if new_id != "custom":
                st.session_state["gemini_selected_model"] = new_id
                st.session_state["_last_synced_gemini_model"] = new_id
            else:
                cust_val = st.session_state.get(batch_custom_model_key, "").strip()
                if cust_val:
                    st.session_state["gemini_selected_model"] = cust_val
                    st.session_state["_last_synced_gemini_model"] = cust_val

        online_custom = st.session_state.get(online_custom_model_key, "").strip()
        batch_custom = st.session_state.get(batch_custom_model_key, "").strip()
        prev_custom = st.session_state.get("_prev_custom_model_val", "")
        if online_custom and online_custom != prev_custom:
            st.session_state["gemini_selected_model"] = online_custom
            st.session_state[batch_custom_model_key] = online_custom
            st.session_state["_prev_custom_model_val"] = online_custom
            st.session_state["_last_synced_gemini_model"] = online_custom
        elif batch_custom and batch_custom != prev_custom:
            st.session_state["gemini_selected_model"] = batch_custom
            st.session_state[online_custom_model_key] = batch_custom
            st.session_state["_prev_custom_model_val"] = batch_custom
            st.session_state["_last_synced_gemini_model"] = batch_custom

    active_m = st.session_state.get("gemini_selected_model", "gemini-3.5-flash-lite")
    active_lbl = model_id_to_label.get(active_m, tr("gemini_model_custom_option"))
    if online_model_sel_key not in st.session_state:
        st.session_state[online_model_sel_key] = active_lbl
        st.session_state["_prev_online_model_lbl"] = active_lbl
    if batch_model_sel_key not in st.session_state:
        st.session_state[batch_model_sel_key] = active_lbl
        st.session_state["_prev_batch_model_lbl"] = active_lbl
    st.session_state["_last_synced_gemini_model"] = active_m

    with lit_input_tabs[0]:
        # -------------------------------------------------------------------------
        # 1. Google Scholar-Style Academic Literature Search (Titles & Abstracts)
        # -------------------------------------------------------------------------
        st.markdown(f'<div class="section-title">📚 {tr("sec_literature_search_title")}</div>', unsafe_allow_html=True)
        st.caption(tr("sec_literature_search_desc"))
        st.info(tr("online_search_opt_notice"))

        cur_active_gemini_model = st.session_state.get("gemini_selected_model", "gemini-3.5-flash-lite")
        has_online_gemini = bool(st.session_state.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY"))
        if has_online_gemini:
            st.markdown(
                f'<div style="font-size: 0.88rem; font-weight: 600; padding: 6px 12px; background: rgba(46, 204, 113, 0.12); border-left: 4px solid #2ecc71; border-radius: 4px; margin-bottom: 8px;">'
                f'{tr("engine_status_gemini_active", model=cur_active_gemini_model)}'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="font-size: 0.88rem; font-weight: 600; padding: 6px 12px; background: rgba(241, 196, 15, 0.12); border-left: 4px solid #f1c40f; border-radius: 4px; margin-bottom: 8px;">'
                f'{tr("engine_status_offline_heuristic")}'
                f'</div>',
                unsafe_allow_html=True
            )

        # Search Bar, Sort, Depth Limit & Gemini Key Controls
        def _trigger_lit_search():
            # In automated test runners (pytest / AppTest), .input().run() simulates typing + Enter
            if "pytest" in sys.modules or "streamlit.testing.v1" in sys.modules:
                st.session_state["trigger_literature_search"] = True

        # Client-side listener to ensure pressing Enter inside the search input triggers the search button,
        # while blurring/tabbing or clicking other dropdowns (e.g. Number of Literature, Sort By) does NOT trigger search.
        st.components.v1.html(
            """
            <script>
            (function() {
                const parentDoc = window.parent.document;
                function attachSearchEnter() {
                    const input = parentDoc.querySelector('.st-key-lit_search_query_input input');
                    if (input && !input.dataset.searchEnterBound) {
                        input.dataset.searchEnterBound = "true";
                        input.addEventListener('keydown', function(e) {
                            if (e.key === 'Enter') {
                                e.preventDefault();
                                const btn = parentDoc.querySelector('.st-key-lit_search_submit_btn button');
                                if (btn) {
                                    btn.click();
                                }
                            }
                        });
                    }
                }
                attachSearchEnter();
                if (!window.parent._litSearchObserverAttached) {
                    window.parent._litSearchObserverAttached = true;
                    const observer = new MutationObserver(attachSearchEnter);
                    observer.observe(parentDoc.body, { childList: true, subtree: true });
                }
            })();
            </script>
            """,
            height=0,
            width=0,
        )

        # Step 1 Online Literature Search Action Bar:
        # Row 1: Field Labels and Options (Keywords, Append Checkbox, Sort By, Number of Literature)
        with st.container(key="lit_search_label_row"):
            col_lbl_kw, col_lbl_append, col_lbl_sort, col_lbl_limit, _, _ = st.columns(
                [0.55, 1.30, 0.65, 0.85, 1.35, 0.9], vertical_alignment="center"
            )
            with col_lbl_kw:
                st.markdown(
                    f"<div class='lit-search-label'>{tr('search_literature_label')}</div>",
                    unsafe_allow_html=True,
                )
            with col_lbl_append:
                append_mode_active = st.checkbox(
                    tr("append_to_dataset_label"),
                    value=st.session_state.get("lit_search_append_mode", False),
                    key="lit_search_append_checkbox",
                    help="Keep previously mined literature papers and append newly discovered papers to your cumulative dataset.",
                )
                st.session_state["lit_search_append_mode"] = append_mode_active
            with col_lbl_sort:
                st.markdown(
                    f"<div class='lit-search-label'>{tr('search_sort_label')}</div>",
                    unsafe_allow_html=True,
                )
            with col_lbl_limit:
                st.markdown(
                    f"<div class='lit-search-label'>{tr('search_limit_label')}</div>",
                    unsafe_allow_html=True,
                )

        # Row 2: Search Input Box, Dropdowns, Gemini Key Popover, Search Button
        col_search_box, col_sort, col_limit, col_gemini, col_btn = st.columns([1.85, 0.65, 0.85, 1.35, 0.9], vertical_alignment="center")
        with col_search_box:
            default_q = st.session_state.get("literature_search_query", "")
            search_query_input = st.text_input(
                tr("search_literature_label"),
                value=default_q,
                placeholder=tr("search_literature_placeholder"),
                key="lit_search_query_input",
                on_change=_trigger_lit_search,
                label_visibility="collapsed",
            )
        with col_sort:
            sort_opts = ["relevance", "date", "citations"]
            sort_lbls = {
                "relevance": tr("sort_relevance"),
                "date": tr("sort_date"),
                "citations": tr("sort_citations"),
            }
            search_sort = st.selectbox(
                tr("search_sort_label"),
                options=sort_opts,
                format_func=lambda s: sort_lbls.get(s, s),
                index=0,
                key="lit_search_sort_select",
                label_visibility="collapsed",
            )
        with col_limit:
            search_depth_options = [25, 50, 100, 150]
            depth_labels = {
                25: tr("depth_tier_25"),
                50: tr("depth_tier_50"),
                100: tr("depth_tier_100"),
                150: tr("depth_tier_150"),
            }
            search_limit = st.selectbox(
                tr("search_limit_label"),
                options=search_depth_options,
                format_func=lambda d: depth_labels.get(d, f"{d} Papers") if isinstance(d, int) else d,
                index=0,  # 25 Papers (⚡ Fast) — Default
                key="lit_search_limit_select",
                label_visibility="collapsed",
            )
        with col_gemini:
            has_online_gemini = bool(st.session_state.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY"))
            popover_label = f"🔑 {tr('gemini_api_key_label')} (✅ {tr('gemini_badge_active')})" if has_online_gemini else f"🔑 {tr('gemini_api_key_label')}"
            with st.popover(popover_label, use_container_width=True, key="online_search_gemini_popover", help=tr("gemini_popover_button_help")):
                if has_online_gemini:
                    active_pop_model = st.session_state.get("gemini_selected_model", "gemini-3.5-flash-lite")
                    st.markdown(
                        f'<div style="font-size: 0.88rem; font-weight: 600; padding: 8px 12px; background: rgba(46, 204, 113, 0.15); border-left: 4px solid #2ecc71; border-radius: 4px; margin-bottom: 8px; color: inherit;">'
                        f'{tr("gemini_popover_active_banner", model=active_pop_model)}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div style="font-size: 0.84rem; padding: 6px 10px; background: rgba(241, 196, 15, 0.12); border-left: 4px solid #f1c40f; border-radius: 4px; margin-bottom: 8px; color: inherit;">'
                        f'{tr("gemini_popover_offline_banner")}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                st.caption(tr("gemini_api_key_help"))
                st.markdown(f'<a href="https://aistudio.google.com/" target="_blank" style="font-size: 0.84rem; text-decoration: none; color: #1e88e5; display: inline-block; margin-bottom: 8px;">🔗 {tr("get_free_gemini_key_link")} ↗</a>', unsafe_allow_html=True)
                gemini_key_ver = st.session_state.get("gemini_key_version", 0)
                online_k_key = f"online_search_gemini_api_key_input_{gemini_key_ver}" if gemini_key_ver > 0 else "online_search_gemini_api_key_input"
                online_k_val = st.text_input(
                    tr("gemini_api_key_label"),
                    value=st.session_state.get("gemini_api_key", ""),
                    type="password",
                    key=online_k_key,
                    label_visibility="collapsed"
                )
                if online_k_val and online_k_val.strip() and online_k_val.strip() != st.session_state.get("gemini_api_key", ""):
                    st.session_state["gemini_api_key"] = online_k_val.strip()

                # Gemini Model Selection Dropdown
                cur_sel_model = st.session_state.get("gemini_selected_model", "gemini-3.5-flash-lite")
                cur_label = model_id_to_label.get(cur_sel_model, tr("gemini_model_custom_option"))
                sel_idx = display_model_options.index(cur_label) if cur_label in display_model_options else 0
                online_model_sel_key = f"online_search_gemini_model_select_{gemini_key_ver}"

                chosen_label = st.selectbox(
                    tr("gemini_model_select_label"),
                    options=display_model_options,
                    index=0 if online_model_sel_key in st.session_state else sel_idx,
                    key=online_model_sel_key,
                )
                chosen_opt_id = gemini_model_options_map.get(chosen_label, "gemini-3.5-flash-lite")

                if chosen_opt_id == "custom":
                    online_custom_model_key = f"online_search_gemini_custom_model_input_{gemini_key_ver}"
                    custom_model_input = st.text_input(
                        tr("gemini_model_custom_label"),
                        value=cur_sel_model if cur_sel_model not in gemini_model_options_map.values() else "",
                        placeholder=tr("gemini_model_custom_placeholder"),
                        key=online_custom_model_key,
                    ).strip()
                    if custom_model_input:
                        st.session_state["gemini_selected_model"] = custom_model_input
                        st.session_state["_last_synced_gemini_model"] = custom_model_input

                st.markdown(tr("gemini_quota_card_html"), unsafe_allow_html=True)
                st.caption(f"🔒 {tr('gemini_security_notice')}")
                col_k_save, col_k_clear = st.columns(2)
                with col_k_save:
                    if st.button(f"💾 {tr('btn_save_key')}", key="btn_save_online_gemini_key", use_container_width=True):
                        if online_k_val and online_k_val.strip():
                            stripped_k = online_k_val.strip()
                            if not (stripped_k.startswith("AIzaSy") and len(stripped_k) >= 30):
                                st.warning(tr("invalid_gemini_key_format_error"))
                            st.session_state["gemini_api_key"] = stripped_k
                            st.session_state["_close_gemini_popover"] = True
                            st.toast(tr("key_saved_toast"), icon="🔑")
                        else:
                            st.session_state["gemini_api_key"] = ""
                        st.rerun()
                with col_k_clear:
                    if st.button(f"🗑️ {tr('btn_clear_key')}", key="btn_clear_online_gemini_key", use_container_width=True):
                        st.session_state["gemini_api_key"] = ""
                        st.session_state["gemini_key_version"] = gemini_key_ver + 1
                        st.toast(tr("key_cleared_toast"), icon="🗑️")
                        st.rerun()
        with col_btn:
            search_clicked = st.button(
                f"🔍 {tr('btn_search_literature')}",
                type="primary",
                use_container_width=True,
                key="lit_search_submit_btn",
            )

        if search_limit in [100, 150]:
            st.info(tr("deep_consensus_notice"))

        if is_demo:
            def _apply_quick_search_chip(query_str: str):
                st.session_state["literature_search_query"] = query_str
                st.session_state["lit_search_query_input"] = query_str
                st.session_state["trigger_literature_search"] = True
                st.session_state["lit_search_current_page"] = 1

            st.markdown(f'<div style="font-size: 0.88rem; font-weight: 600; margin: 8px 0 4px 0; color: inherit;">💡 {tr("quick_search_chips_label")}</div>', unsafe_allow_html=True)
            col_c1, col_c2, col_c3, col_c4, col_c5 = st.columns(5)
            with col_c1:
                st.button(
                    tr("chip_trehalose"),
                    key="btn_chip_trehalose",
                    use_container_width=True,
                    on_click=_apply_quick_search_chip,
                    args=("trehalose",),
                )
            with col_c2:
                st.button(
                    tr("chip_teabr"),
                    key="btn_chip_teabr",
                    use_container_width=True,
                    on_click=_apply_quick_search_chip,
                    args=("tetraethylammonium bromide",),
                )
            with col_c3:
                st.button(
                    tr("chip_sorbitol"),
                    key="btn_chip_sorbitol",
                    use_container_width=True,
                    on_click=_apply_quick_search_chip,
                    args=("sorbitol",),
                )
            with col_c4:
                st.button(
                    tr("chip_thiourea"),
                    key="btn_chip_thiourea",
                    use_container_width=True,
                    on_click=_apply_quick_search_chip,
                    args=("thiourea",),
                )
            with col_c5:
                st.button(
                    tr("chip_glucose"),
                    key="btn_chip_glucose",
                    use_container_width=True,
                    on_click=_apply_quick_search_chip,
                    args=("glucose",),
                )

        is_fetch_next = st.session_state.pop("trigger_literature_fetch_next", False)
        search_triggered = search_clicked or is_fetch_next or st.session_state.pop("trigger_literature_search", False)

        if search_triggered:
            st.session_state.pop("last_extracted_notice", None)
            st.session_state.pop("bridged_additive_name", None)
            st.session_state["show_formulation_reminder_modal"] = False

            if is_fetch_next:
                q_text = (st.session_state.get("literature_search_query") or search_query_input or "").strip()
                is_appending = True
                query_offset = st.session_state.get("lit_search_offset", len(st.session_state.get("literature_search_results", [])))
            else:
                q_text = (search_query_input or "").strip()
                st.session_state["literature_search_query"] = q_text
                st.session_state["lit_search_current_page"] = 1
                is_appending = append_mode_active
                query_offset = 0
                if not is_appending:
                    st.session_state["lit_search_offset"] = 0

            existing_papers = list(st.session_state.get("literature_search_results", [])) if is_appending else []
            existing_records = list(st.session_state.get("literature_mined_records", [])) if is_appending else []

            if q_text:
                active_gemini_key = st.session_state.get("gemini_api_key") or (online_k_val.strip() if 'online_k_val' in locals() and online_k_val else None) or os.environ.get("GEMINI_API_KEY")
                client = SemanticScholarClient()
                with st.status(tr("mining_status_in_progress"), expanded=True) as status_box:
                    col_stat_info, col_stat_stop = status_box.columns([3.2, 1.2], vertical_alignment="center")
                    with col_stat_info:
                        offset_label = f" (Offset: {query_offset})" if query_offset > 0 else ""
                        st.write(f"🔍 **{tr('mining_start_query')}:** `{q_text}` ({search_limit} {tr('search_limit_papers_label')}{offset_label})")
                    with col_stat_stop:
                        stop_mining_btn = st.button(
                            f"⏹️ {tr('btn_stop_mining')}",
                            key="btn_stop_literature_mining",
                            use_container_width=True,
                            help=tr("btn_stop_mining_help")
                        )
                        if stop_mining_btn:
                            st.session_state["literature_mining_stopped_by_user"] = True

                    lit_progress_bar = status_box.progress(0.05, text=f"📥 {tr('mining_stage_retrieval')}")

                    def _stream_milestone(batch_idx: int, total_batches: int, batch_papers: List[PaperMetadata], accum: List[PaperMetadata]):
                        p_start = (batch_idx - 1) * 25 + 1
                        p_end = p_start + len(batch_papers) - 1
                        pct = min(0.20, 0.05 + 0.15 * (len(accum) / max(search_limit, 1)))
                        lit_progress_bar.progress(pct, text=f"📥 {tr('batch_progress_msg', batch=batch_idx, total=total_batches, start=p_start, end=p_end)} ({len(accum)}/{search_limit})")
                        status_box.write(f"📥 **{tr('batch_progress_msg', batch=batch_idx, total=total_batches, start=p_start, end=p_end)}** ({len(accum)}/{search_limit})")

                    status_box.write(f"📥 **{tr('mining_stage_retrieval')}**...")
                    found = client.search_papers(
                        query=q_text,
                        limit=search_limit,
                        offset=query_offset,
                        sort=search_sort,
                        batch_size=25,
                        pause_seconds=0.3,
                        batch_callback=_stream_milestone
                    )
                    lit_progress_bar.progress(0.20, text=f"🤖 {tr('mining_stage_extraction')}")

                    # Deduplicate against existing accumulated papers by DOI and normalized title
                    seen_dois = {getattr(p, "doi", "").lower().strip() for p in existing_papers if getattr(p, "doi", None)}
                    seen_titles = {re.sub(r'[^a-zA-Z0-9]', '', (getattr(p, "title", "") or "").lower()) for p in existing_papers}
                    papers_to_extract = []
                    for p in found:
                        doi_k = (getattr(p, "doi", "") or "").lower().strip()
                        title_k = re.sub(r'[^a-zA-Z0-9]', '', (getattr(p, "title", "") or "").lower())
                        if doi_k and doi_k in seen_dois:
                            continue
                        if title_k and title_k in seen_titles:
                            continue
                        papers_to_extract.append(p)
                        if doi_k:
                            seen_dois.add(doi_k)
                        if title_k:
                            seen_titles.add(title_k)

                    active_model = st.session_state.get("gemini_selected_model", "gemini-3.5-flash-lite")
                    extractor = PaperExtractor(api_key=active_gemini_key, model_name=active_model)

                    def _stream_extract(b_idx: int, total_b: int, b_papers: List[Any], accum_res: List[Any]):
                        p_start = (b_idx - 1) * 5 + 1
                        p_end = min(len(papers_to_extract), p_start + len(b_papers) - 1)
                        st.session_state["literature_mined_partial"] = list(accum_res)
                        st.session_state["literature_found_partial"] = papers_to_extract[:len(accum_res)]
                        tot_ext = max(len(papers_to_extract), 1)
                        pct = min(0.68, 0.20 + 0.48 * (len(accum_res) / tot_ext))
                        if active_gemini_key:
                            b_msg = f"🤖 {tr('mining_status_gemini_batch', batch=b_idx, total=total_b, start=p_start, end=p_end)} ({len(accum_res)}/{len(papers_to_extract)})"
                        else:
                            b_msg = f"⚙️ {tr('mining_status_extracting')} (Papers {p_start}–{p_end} / {len(papers_to_extract)})..."
                        lit_progress_bar.progress(pct, text=b_msg)
                        status_box.write(b_msg)

                    if active_gemini_key:
                        status_box.write(f"🤖 **{tr('mining_stage_extraction')}**...")
                    else:
                        status_box.write(f"⚙️ **{tr('mining_status_extracting')}**...")

                    def _check_user_stop():
                        return bool(st.session_state.get("literature_mining_stopped_by_user", False))

                    mined_records = extractor.extract_batch_from_papers(
                        papers=papers_to_extract,
                        batch_size=5,
                        batch_callback=_stream_extract,
                        stop_check=_check_user_stop
                    )

                    if active_gemini_key and getattr(extractor, "last_batch_error", None):
                        clean_err = str(extractor.last_batch_error)[:120]
                        status_box.write(tr("gemini_api_error_warning", error=clean_err))

                    was_stopped = bool(st.session_state.pop("literature_mining_stopped_by_user", False))

                    # Combine with existing records if appending
                    combined_records = list(existing_records) + mined_records
                    combined_found = list(existing_papers) + (papers_to_extract[:len(mined_records)])

                    # Extract unique additives across all records
                    all_unique_additives = []
                    seen_additive_names = set()
                    for rec in combined_records:
                        prim = rec.get("additive")
                        if prim and getattr(prim, "name", None):
                            nm_low = prim.name.lower().strip()
                            if nm_low not in seen_additive_names:
                                seen_additive_names.add(nm_low)
                                all_unique_additives.append(prim)

                    if combined_records:
                        # Stage 3: PubChem & RDKit Descriptors
                        lit_progress_bar.progress(0.72, text=f"🧪 {tr('mining_stage_pubchem')} ({len(all_unique_additives)} {tr('unique_additives_label')})...")
                        status_box.write(f"🧪 **{tr('mining_stage_pubchem')}** ({len(all_unique_additives)} {tr('unique_additives_label')})...")
                        lit_engine = LiteratureDiscoveryEngine(gemini_api_key=active_gemini_key, llm_model=active_model)

                        # Stage 4: Consensus & AI Analogue Generation
                        lit_progress_bar.progress(0.85, text=f"🧠 {tr('mining_stage_consensus')}...")
                        status_box.write(f"🧠 **{tr('mining_stage_consensus')}**...")
                        t1_cands, t2_cands = lit_engine.generate_two_tier_candidates(all_unique_additives, use_pubchem=True)
                        for tc in t1_cands:
                            tc.similarity_metric = "Direct Literature Extraction"
                            matched_title = next((getattr(r.get("paper"), "title", None) or (r.get("paper", {}).get("title") if isinstance(r.get("paper"), dict) else "Mined Literature Paper") for r in combined_records if r.get("additive") and getattr(r["additive"], "name", "").lower() == tc.name.lower()), "Mined Literature Paper")
                            tc.rationale = f"Extracted from literature paper: {matched_title}"
                        _merge_candidates_into_session(t1_cands, t2_cands, prompt_on_conflict=not is_fetch_next)

                        # Stage 5: Finalizing, Pre-compiling Dossiers & Rendering
                        lit_progress_bar.progress(0.92, text=f"📊 {tr('mining_stage_finalizing')}...")
                        status_box.write(f"📊 **{tr('mining_stage_finalizing')}**...")
                        st.session_state["extracted_additives"] = all_unique_additives
                        st.session_state["literature_search_results"] = combined_found
                        st.session_state["literature_mined_records"] = combined_records
                        st.session_state["lit_search_offset"] = query_offset + len(found)

                        # Pre-cache CSV and XLSX downloads inside the active status block so user sees progress
                        cached_csv = generate_mined_literature_csv(combined_records)
                        cached_xlsx = generate_mined_literature_xlsx(combined_records)
                        st.session_state["cached_mined_lit_csv"] = cached_csv
                        st.session_state["cached_mined_lit_xlsx"] = cached_xlsx
                        st.session_state["cached_mined_lit_len"] = len(combined_records)

                        lit_progress_bar.progress(1.0, text=tr("mining_reassuring_render_notice"))

                        if was_stopped:
                            status_box.update(label=tr("mining_stopped_by_user_notice", count=len(mined_records)), state="complete", expanded=True)
                            st.warning(f"⏹️ {tr('mining_stopped_by_user_notice', count=len(mined_records))}")
                        else:
                            if is_fetch_next:
                                toast_msg = tr("mining_fetch_next_success", new_count=len(mined_records), total_papers=len(combined_found), total_additives=len(all_unique_additives))
                                st.toast(toast_msg, icon="📚")
                            else:
                                if len(all_unique_additives) >= 1:
                                    first_add = all_unique_additives[0]
                                    opt_c = getattr(first_add, "optimal_concentration", None)
                                    ext_conc = float(opt_c.value) if (opt_c and getattr(opt_c, "value", None) is not None) else 10.0
                                    toast_conc = f"{ext_conc:.1f} mM" if (opt_c and getattr(opt_c, "value", None) is not None) else f"{ext_conc:.1f} mM (default)"
                                    bc = getattr(first_add, "battery_context", None)
                                    ext_salt = (bc.baseline_electrolyte if (bc and getattr(bc, "baseline_electrolyte", None)) else "2 M ZnSO₄")
                                    st.session_state["last_extracted_notice"] = tr(
                                        "extraction_success_toast",
                                        name=first_add.name,
                                        conc=toast_conc,
                                        parent=format_chem_formula(ext_salt),
                                    )
                                    if not is_formulation_reminder_hidden():
                                        st.session_state["show_formulation_reminder_modal"] = True
                            status_box.update(label=f"✅ {tr('mining_status_complete', count=len(combined_records))} ({len(all_unique_additives)} {tr('unique_additives_label')})", state="complete", expanded=True)
                    else:
                        st.session_state["literature_search_results"] = []
                        st.session_state["literature_mined_records"] = []
                        if was_stopped:
                            status_box.update(label=tr("mining_stopped_no_papers_notice"), state="error", expanded=True)
                            st.warning(f"⏹️ {tr('mining_stopped_no_papers_notice')}")
                        else:
                            status_box.update(label=f"ℹ️ {tr('mining_stopped_no_papers_notice')}", state="complete", expanded=True)
            else:
                if not is_appending:
                    st.session_state["literature_search_results"] = []
                    st.session_state["literature_mined_records"] = []
                st.warning(tr("search_query_empty_warning"))

        # Check if previous execution was stopped by user via rerun
        if st.session_state.pop("literature_mining_stopped_by_user", False):
            partial_mined = st.session_state.get("literature_mined_partial", [])
            partial_found = st.session_state.get("literature_found_partial", [])
            if not partial_mined and "literature_search_results" in st.session_state and st.session_state["literature_search_results"]:
                off_ext = PaperExtractor(api_key=None)
                partial_found = st.session_state["literature_search_results"]
                partial_mined = off_ext.extract_batch_from_papers(partial_found)
                st.session_state["literature_mined_partial"] = partial_mined
                st.session_state["literature_found_partial"] = partial_found

            if partial_mined:
                st.session_state["literature_search_results"] = partial_found
                st.session_state["literature_mined_records"] = partial_mined
                u_adds = []
                s_n = set()
                for r in partial_mined:
                    pa = r.get("additive")
                    if pa and getattr(pa, "name", None):
                        nl = pa.name.lower().strip()
                        if nl not in s_n:
                            s_n.add(nl)
                            u_adds.append(pa)
                eng = LiteratureDiscoveryEngine(
                    gemini_api_key=st.session_state.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY"),
                    llm_model=st.session_state.get("gemini_selected_model", "gemini-3.5-flash-lite")
                )
                t1_cands, t2_cands = eng.generate_two_tier_candidates(u_adds, use_pubchem=True)
                for tc in t1_cands:
                    tc.similarity_metric = "Direct Literature Extraction"
                    matched_title = next((getattr(r.get("paper"), "title", None) or (r.get("paper", {}).get("title") if isinstance(r.get("paper"), dict) else "Mined Literature Paper") for r in partial_mined if r.get("additive") and getattr(r["additive"], "name", "").lower() == tc.name.lower()), "Mined Literature Paper")
                    tc.rationale = f"Extracted from literature paper: {matched_title}"
                _merge_candidates_into_session(t1_cands, t2_cands)
                st.session_state["extracted_additives"] = u_adds
                st.warning(f"⏹️ {tr('mining_stopped_by_user_notice', count=len(partial_mined))}")
            else:
                st.warning(f"⏹️ {tr('mining_stopped_no_papers_notice')}")

        # Transient toast notification if an additive was just extracted & transferred
        if "last_extracted_notice" in st.session_state:
            notice_text = st.session_state.pop("last_extracted_notice")
            st.toast(notice_text, icon="🧪")

        discovered_papers: List[PaperMetadata] = st.session_state.get("literature_search_results", [])
        mined_records: List[Dict[str, Any]] = st.session_state.get("literature_mined_records", [])

        if "literature_search_results" in st.session_state and not discovered_papers:
            query_val = st.session_state.get("literature_search_query", "").strip()
            if query_val:
                st.info(tr("no_papers_found_for_query", query=query_val))

        # Ensure mined_records has complete parity with discovered_papers
        if discovered_papers and len(mined_records) != len(discovered_papers):
            active_gemini_key = st.session_state.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY")
            active_model = st.session_state.get("gemini_selected_model", "gemini-3.5-flash-lite")
            extractor = PaperExtractor(api_key=active_gemini_key, model_name=active_model)
            mined_records = extractor.extract_batch_from_papers(discovered_papers, batch_size=5)
            st.session_state["literature_mined_records"] = mined_records

        # UI Pagination (10 Cards per Page)
        total_items = len(discovered_papers)
        if total_items > 0:
            PAGE_SIZE = 10
            total_pages = max(1, (total_items + PAGE_SIZE - 1) // PAGE_SIZE)
            current_page = st.session_state.get("lit_search_current_page", 1)
            if current_page > total_pages:
                current_page = total_pages
                st.session_state["lit_search_current_page"] = current_page
            if current_page < 1:
                current_page = 1
                st.session_state["lit_search_current_page"] = current_page

            start_idx = (current_page - 1) * PAGE_SIZE
            end_idx = min(total_items, current_page * PAGE_SIZE)

            st.markdown(
                f'<div style="margin: 4px 0 12px 0; font-size: 0.88rem; color: inherit;">'
                f'ℹ️ {tr("results_count_found", count=total_items)}'
                f'</div>',
                unsafe_allow_html=True
            )

            # Dataset Action Bar: Download CSV, Download XLSX, Fetch Next, Reset Dataset
            col_d1, col_d2, col_d3, col_d4 = st.columns([1.3, 1.3, 1.7, 0.9], vertical_alignment="center")
            with col_d1:
                cached_len = st.session_state.get("cached_mined_lit_len", -1)
                if cached_len == len(mined_records) and "cached_mined_lit_csv" in st.session_state:
                    csv_bytes = st.session_state["cached_mined_lit_csv"]
                else:
                    csv_bytes = generate_mined_literature_csv(mined_records)
                    st.session_state["cached_mined_lit_csv"] = csv_bytes
                    st.session_state["cached_mined_lit_len"] = len(mined_records)
                q_slug = re.sub(r'[^a-zA-Z0-9]+', '_', st.session_state.get("literature_search_query", "Literature")).strip('_') or "Literature"
                st.download_button(
                    f"📥 {tr('btn_download_lit_csv')}",
                    data=csv_bytes,
                    file_name=f"MAP_Literature_{q_slug}_{total_items}Papers.csv",
                    mime="text/csv",
                    key="btn_dl_mined_lit_csv",
                    use_container_width=True,
                )
            with col_d2:
                if cached_len == len(mined_records) and "cached_mined_lit_xlsx" in st.session_state:
                    xlsx_bytes = st.session_state["cached_mined_lit_xlsx"]
                else:
                    xlsx_bytes = generate_mined_literature_xlsx(mined_records)
                    st.session_state["cached_mined_lit_xlsx"] = xlsx_bytes
                    st.session_state["cached_mined_lit_len"] = len(mined_records)
                st.download_button(
                    f"📊 {tr('btn_download_lit_xlsx')}",
                    data=xlsx_bytes,
                    file_name=f"MAP_Literature_{q_slug}_{total_items}Papers.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="btn_dl_mined_lit_xlsx",
                    use_container_width=True,
                )
            with col_d3:
                fetch_cnt = search_limit if isinstance(search_limit, int) else 25
                if st.button(
                    f"➕ {tr('btn_fetch_next_papers', count=fetch_cnt)}",
                    key="btn_fetch_next_papers_top",
                    help=tr("btn_fetch_next_help"),
                    use_container_width=True,
                ):
                    st.session_state["trigger_literature_fetch_next"] = True
                    st.rerun()
            with col_d4:
                if st.button(f"🗑️ {tr('btn_clear_lit_dataset')}", key="btn_clear_lit_dataset_action", help=tr("btn_clear_lit_dataset_help"), use_container_width=True):
                    st.session_state["literature_search_results"] = []
                    st.session_state["literature_mined_records"] = []
                    st.session_state["lit_search_offset"] = 0
                    st.toast(tr("mining_dataset_reset_toast"), icon="🗑️")
                    st.rerun()

            st.markdown('<div style="margin-bottom: 8px;"></div>', unsafe_allow_html=True)

            # Navigation Controls (Top)
            col_prev, col_info, col_next = st.columns([1.2, 2.6, 1.2], vertical_alignment="center")
            with col_prev:
                if st.button(f"◀ {tr('pagination_prev')}", key="lit_prev_page_top", disabled=(current_page <= 1), use_container_width=True):
                    st.session_state["lit_search_current_page"] = current_page - 1
                    st.rerun()
            with col_info:
                st.markdown(
                    f'<div style="text-align: center; font-weight: 600; font-size: 0.92rem; color: inherit; padding: 4px 0;">'
                    f'{tr("pagination_page_info", page=current_page, total_pages=total_pages, start=start_idx + 1, end=end_idx, total_items=total_items)}'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with col_next:
                if st.button(f"{tr('pagination_next')} ▶", key="lit_next_page_top", disabled=(current_page >= total_pages), use_container_width=True):
                    st.session_state["lit_search_current_page"] = current_page + 1
                    st.rerun()

            st.markdown('<div style="margin-bottom: 12px;"></div>', unsafe_allow_html=True)

            # Render 10 Cards for Current Page
            for idx in range(start_idx, end_idx):
                item_rec = mined_records[idx]
                paper = item_rec["paper"]
                ext_item = item_rec["additive"]

                authors_str = ", ".join(paper.authors) if paper.authors else "Unknown Authors"
                venue_str = paper.venue or "Peer-Reviewed Academic Publication"
                year_str = str(paper.year) if paper.year else "Recent"
                is_oa = bool(paper.open_access_pdf_url)
                doi_val = paper.doi or ""
                citation_cnt = getattr(paper, "citation_count", None)

                # Detect Royal Society of Chemistry (RSC) papers (DOI prefix 10.1039 or RSC venue)
                is_rsc = (doi_val.lower().startswith("10.1039/") or 
                          "royal society of chemistry" in venue_str.lower() or 
                          "rsc" in venue_str.lower().split())

                oa_badge = f'<span class="badge-tag badge-green">{tr("badge_open_access")}</span>' if is_oa else f'<span class="badge-tag badge-amber">{tr("badge_subscription")}</span>'
                doi_badge = f'<a href="https://doi.org/{doi_val}" target="_blank" style="text-decoration: none;"><span class="badge-tag badge-teal">DOI: {doi_val} ↗</span></a>' if doi_val else ''
                cite_badge = f'<span class="badge-tag badge-blue">Cited by {citation_cnt}</span>' if citation_cnt is not None else ''
                rsc_badge = f'<span class="badge-tag badge-amber" title="{tr("rsc_abstract_notice")}">⚠️ {tr("badge_rsc_partial_abstract")}</span>' if is_rsc else ''

                badges_list = [b for b in [oa_badge, doi_badge, cite_badge, rsc_badge] if b]
                badges_html = " ".join(badges_list)

                clean_title = clean_scientific_text(paper.title)
                clean_abstract = clean_scientific_text(paper.abstract) if paper.abstract else tr("no_abstract_available")

                if ext_item is not None:
                    ext_name = ext_item.name
                    ext_smiles = ext_item.smiles
                    has_valid_additive = True
                    ext_formula = format_chem_formula(get_chemical_formula(ext_smiles, ext_name)) if (ext_smiles or ext_name) else ""
                    ext_formula_display = f' <span style="font-family: monospace; font-size: 0.88rem; color: #1f77b4;">({ext_formula})</span>' if ext_formula else ""

                    if ext_item.optimal_concentration and ext_item.optimal_concentration.raw_text:
                        ext_conc_str = clean_scientific_text(ext_item.optimal_concentration.raw_text)
                    elif ext_item.optimal_concentration and ext_item.optimal_concentration.value is not None:
                        ext_conc_str = f"{ext_item.optimal_concentration.value:g} {ext_item.optimal_concentration.unit or 'mM'}"
                    else:
                        ext_conc_str = f'<span class="badge-tag badge-amber">⚠️ {tr("card_reported_conc_unspecified")}</span>'

                    raw_salt = ext_item.battery_context.baseline_electrolyte if (ext_item.battery_context and ext_item.battery_context.baseline_electrolyte) else "Unspecified / Not reported"
                    is_unspecified_salt = (not raw_salt or "unspecified" in str(raw_salt).lower() or "not reported" in str(raw_salt).lower())
                    if is_unspecified_salt:
                        ext_salt_display = f'<span class="badge-tag badge-amber">⚠️ {tr("baseline_salt_unspecified")}</span>'
                    else:
                        ext_salt_display = format_chem_formula(raw_salt)
                    ext_mech_str = clean_scientific_text(", ".join(ext_item.primary_mechanisms) if ext_item.primary_mechanisms else ext_item.mechanism_summary)
                else:
                    has_valid_additive = False
                    ext_name = f'<span class="badge-tag badge-amber">⚠️ {tr("additive_unspecified_in_abstract")}</span>'
                    ext_formula_display = ""
                    ext_conc_str = f'<span class="badge-tag badge-amber">⚠️ {tr("card_reported_conc_unspecified")}</span>'
                    salt_m = re.search(r'(\d+(?:\.\d+)?\s*M)?\s*(ZnSO4|ZnSO₄|Zn\(OTf\)2|Zn\(OTf\)₂|Zn\(TFSI\)2|Zn\(TFSI\)₂|ZnCl2|ZnCl₂|Zn\(ClO4\)2|Zn\(ClO₄\)₂|Zn\(CH3COO\)2|Zn\(CH₃COO\)₂|Zn\(OAc\)2)', clean_abstract, re.IGNORECASE)
                    if salt_m:
                        conc_prefix = salt_m.group(1) or ""
                        ext_salt_display = format_chem_formula(f"{conc_prefix} {salt_m.group(2)}".strip())
                    else:
                        ext_salt_display = f'<span class="badge-tag badge-amber">⚠️ {tr("baseline_salt_unspecified")}</span>'
                    ext_mech_str = tr("card_mechanism_none_or_unspecified")

                card_html = (
                    f'<div class="literature-paper-card">'
                    f'<div class="paper-card-title">{clean_title}</div>'
                    f'<div class="paper-card-meta">'
                    f'<strong>{authors_str}</strong> — <em>{venue_str}</em> ({year_str})'
                    f'</div>'
                    f'<div style="margin-bottom: 8px;">{badges_html}</div>'
                    f'<div class="paper-analysis-box" style="margin-top: 6px; padding: 10px 14px;">'
                    f'<div style="font-size: 0.98rem; font-weight: 700; margin-bottom: 4px; color: inherit;">'
                    f'🧪 {tr("card_extracted_additive_label")}: <strong>{ext_name}</strong>{ext_formula_display}'
                    f'</div>'
                    f'<div style="font-size: 0.86rem; color: inherit; margin-bottom: 3px;">'
                    f'• <strong>{tr("card_reported_conc_label")}:</strong> {ext_conc_str} | <strong>{tr("card_baseline_salt_label")}:</strong> {ext_salt_display}'
                    f'</div>'
                    f'<div style="font-size: 0.86rem; color: inherit;">'
                    f'• <strong>{tr("card_mechanism_label")}:</strong> {ext_mech_str}'
                    f'</div>'
                    f'</div>'
                    f'</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)

                with st.expander(f"📄 {tr('view_abstract_label')} — {clean_title[:65]}...", expanded=False):
                    if is_rsc:
                        st.markdown(
                            f'<div style="background-color: rgba(255, 193, 7, 0.12); border-left: 4px solid #ffc107; padding: 8px 12px; margin-bottom: 10px; border-radius: 4px; font-size: 0.86rem; color: inherit;">'
                            f'⚠️ <strong>{tr("badge_rsc_partial_abstract")}:</strong> {tr("rsc_abstract_notice")}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    st.markdown(f'<div class="paper-abstract-box">{clean_abstract}</div>', unsafe_allow_html=True)

                st.markdown('<div style="margin-bottom: 12px;"></div>', unsafe_allow_html=True)

            # Navigation Controls (Bottom, if total_pages > 1)
            if total_pages > 1:
                st.markdown('<div style="margin-top: 14px;"></div>', unsafe_allow_html=True)
                col_prev_b, col_info_b, col_next_b = st.columns([1.2, 2.6, 1.2], vertical_alignment="center")
                with col_prev_b:
                    if st.button(f"◀ {tr('pagination_prev')}", key="lit_prev_page_bot", disabled=(current_page <= 1), use_container_width=True):
                        st.session_state["lit_search_current_page"] = current_page - 1
                        st.rerun()
                with col_info_b:
                    st.markdown(
                        f'<div style="text-align: center; font-weight: 600; font-size: 0.92rem; color: inherit; padding: 4px 0;">'
                        f'{tr("pagination_page_info", page=current_page, total_pages=total_pages, start=start_idx + 1, end=end_idx, total_items=total_items)}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                with col_next_b:
                    if st.button(f"{tr('pagination_next')} ▶", key="lit_next_page_bot", disabled=(current_page >= total_pages), use_container_width=True):
                        st.session_state["lit_search_current_page"] = current_page + 1
                        st.rerun()

            st.markdown('<div style="margin-top: 14px;"></div>', unsafe_allow_html=True)
            col_bot_f1, col_bot_f2 = st.columns([3, 1], vertical_alignment="center")
            with col_bot_f1:
                fetch_cnt = search_limit if isinstance(search_limit, int) else 25
                if st.button(
                    f"➕ {tr('btn_fetch_next_papers', count=fetch_cnt)}",
                    key="btn_fetch_next_papers_bot",
                    help=tr("btn_fetch_next_help"),
                    use_container_width=True,
                ):
                    st.session_state["trigger_literature_fetch_next"] = True
                    st.rerun()
            with col_bot_f2:
                if st.button(f"🗑️ {tr('btn_clear_lit_dataset')}", key="btn_clear_lit_dataset_bot", help=tr("btn_clear_lit_dataset_help"), use_container_width=True):
                    st.session_state["literature_search_results"] = []
                    st.session_state["literature_mined_records"] = []
                    st.session_state["lit_search_offset"] = 0
                    st.toast(tr("mining_dataset_reset_toast"), icon="🗑️")
                    st.rerun()

    with lit_input_tabs[1]:
        # -------------------------------------------------------------------------
        # 2. Batch Multi-PDF Document Ingestion & Chemical Entity Extraction
        # -------------------------------------------------------------------------
        st.markdown(f'<div class="section-title">📄 {tr("sec_batch_pdf_title")}</div>', unsafe_allow_html=True)
        st.caption(tr("sec_batch_pdf_desc"))
        st.info(tr("batch_pdf_online_opt_notice"))

        cur_active_gemini_model = st.session_state.get("gemini_selected_model", "gemini-3.5-flash-lite")
        has_gemini_k = bool(st.session_state.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY"))
        if has_gemini_k:
            st.markdown(
                f'<div style="font-size: 0.88rem; font-weight: 600; padding: 6px 12px; background: rgba(46, 204, 113, 0.12); border-left: 4px solid #2ecc71; border-radius: 4px; margin-bottom: 8px;">'
                f'{tr("engine_status_gemini_active", model=cur_active_gemini_model)}'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="font-size: 0.88rem; font-weight: 600; padding: 6px 12px; background: rgba(241, 196, 15, 0.12); border-left: 4px solid #f1c40f; border-radius: 4px; margin-bottom: 8px;">'
                f'{tr("engine_status_offline_heuristic")}'
                f'</div>',
                unsafe_allow_html=True
            )

        uploader_version = st.session_state.get("batch_pdf_uploader_version", 0)
        uploader_key = "upload_batch_pdf" if uploader_version == 0 else f"upload_batch_pdf_{uploader_version}"

        uploaded_pdfs = st.file_uploader(
            tr("upload_batch_pdf_label"),
            type=["pdf"],
            accept_multiple_files=True,
            key=uploader_key,
            help=tr("upload_batch_pdf_help"),
        )
        st.caption(tr("pdf_batch_guardrail_tip"))

        if uploaded_pdfs and len(uploaded_pdfs) > 25:
            st.warning(tr("pdf_batch_limit_warning", count=len(uploaded_pdfs)))

        if is_demo:
            load_sample_pdf_btn = st.button(
                f"📑 {tr('btn_load_sample_pdf')}",
                key="btn_load_sample_pdf",
                use_container_width=True,
                help=tr("btn_load_sample_pdf_help"),
            )
        else:
            load_sample_pdf_btn = False

        col_act1, col_act2, col_act3 = st.columns([1.3, 1.0, 1.0], vertical_alignment="center")
        with col_act1:
            batch_process_btn = st.button(
                f"⚡ {tr('btn_process_batch_pdfs')}",
                key="btn_process_batch_pdfs",
                type="primary",
                use_container_width=True,
            )
        with col_act2:
            clear_pdfs_btn = st.button(
                f"🗑️ {tr('btn_clear_uploaded_pdfs')}",
                key="btn_clear_uploaded_pdfs",
                use_container_width=True,
                help=tr("btn_clear_uploaded_pdfs_help"),
            )
        with col_act3:
            has_batch_gemini = bool(st.session_state.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY"))
            batch_popover_label = f"🔑 {tr('gemini_api_key_label')} (✅ {tr('gemini_badge_active')})" if has_batch_gemini else f"🔑 {tr('gemini_api_key_label')}"
            with st.popover(batch_popover_label, use_container_width=True, key="batch_pdf_gemini_popover", help=tr("gemini_popover_button_help")):
                if has_batch_gemini:
                    active_pop_model = st.session_state.get("gemini_selected_model", "gemini-3.5-flash-lite")
                    st.markdown(
                        f'<div style="font-size: 0.88rem; font-weight: 600; padding: 8px 12px; background: rgba(46, 204, 113, 0.15); border-left: 4px solid #2ecc71; border-radius: 4px; margin-bottom: 8px; color: inherit;">'
                        f'{tr("gemini_popover_active_banner", model=active_pop_model)}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div style="font-size: 0.84rem; padding: 6px 10px; background: rgba(241, 196, 15, 0.12); border-left: 4px solid #f1c40f; border-radius: 4px; margin-bottom: 8px; color: inherit;">'
                        f'{tr("gemini_popover_offline_banner")}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                st.caption(tr("gemini_api_key_help"))
                st.markdown(f'<a href="https://aistudio.google.com/" target="_blank" style="font-size: 0.84rem; text-decoration: none; color: #1e88e5; display: inline-block; margin-bottom: 8px;">🔗 {tr("get_free_gemini_key_link")} ↗</a>', unsafe_allow_html=True)
                gemini_key_ver = st.session_state.get("gemini_key_version", 0)
                batch_k_key = f"batch_pdf_gemini_api_key_input_{gemini_key_ver}" if gemini_key_ver > 0 else "batch_pdf_gemini_api_key_input"
                batch_k_val = st.text_input(
                    tr("gemini_api_key_label"),
                    value=st.session_state.get("gemini_api_key", ""),
                    type="password",
                    key=batch_k_key,
                    label_visibility="collapsed"
                )
                if batch_k_val and batch_k_val.strip() and batch_k_val.strip() != st.session_state.get("gemini_api_key", ""):
                    st.session_state["gemini_api_key"] = batch_k_val.strip()

                # Gemini Model Selection Dropdown
                cur_sel_model = st.session_state.get("gemini_selected_model", "gemini-3.5-flash-lite")
                cur_label = model_id_to_label.get(cur_sel_model, tr("gemini_model_custom_option"))
                sel_idx = display_model_options.index(cur_label) if cur_label in display_model_options else 0
                batch_model_sel_key = f"batch_pdf_gemini_model_select_{gemini_key_ver}"

                chosen_label = st.selectbox(
                    tr("gemini_model_select_label"),
                    options=display_model_options,
                    index=0 if batch_model_sel_key in st.session_state else sel_idx,
                    key=batch_model_sel_key,
                )
                chosen_opt_id = gemini_model_options_map.get(chosen_label, "gemini-3.5-flash-lite")

                if chosen_opt_id == "custom":
                    batch_custom_model_key = f"batch_pdf_gemini_custom_model_input_{gemini_key_ver}"
                    custom_model_input = st.text_input(
                        tr("gemini_model_custom_label"),
                        value=cur_sel_model if cur_sel_model not in gemini_model_options_map.values() else "",
                        placeholder=tr("gemini_model_custom_placeholder"),
                        key=batch_custom_model_key,
                    ).strip()
                    if custom_model_input:
                        st.session_state["gemini_selected_model"] = custom_model_input
                        st.session_state["_last_synced_gemini_model"] = custom_model_input

                st.markdown(tr("gemini_quota_card_html"), unsafe_allow_html=True)
                st.caption(f"🔒 {tr('gemini_security_notice')}")
                col_bk_save, col_bk_clear = st.columns(2)
                with col_bk_save:
                    if st.button(f"💾 {tr('btn_save_key')}", key="btn_save_batch_gemini_key", use_container_width=True):
                        if batch_k_val and batch_k_val.strip():
                            stripped_bk = batch_k_val.strip()
                            if not (stripped_bk.startswith("AIzaSy") and len(stripped_bk) >= 30):
                                st.warning(tr("invalid_gemini_key_format_error"))
                            st.session_state["gemini_api_key"] = stripped_bk
                            st.session_state["_close_gemini_popover"] = True
                            st.toast(tr("key_saved_toast"), icon="🔑")
                        else:
                            st.session_state["gemini_api_key"] = ""
                        st.rerun()
                with col_bk_clear:
                    if st.button(f"🗑️ {tr('btn_clear_key')}", key="btn_clear_batch_gemini_key", use_container_width=True):
                        st.session_state["gemini_api_key"] = ""
                        st.session_state["gemini_key_version"] = gemini_key_ver + 1
                        st.toast(tr("key_cleared_toast"), icon="🗑️")
                        st.rerun()

        if clear_pdfs_btn:
            st.session_state["batch_pdf_uploader_version"] = uploader_version + 1
            st.session_state["batch_pdf_extractions"] = []
            if st.session_state.get("suggested_source") == "pdf_database":
                st.session_state["mined_literature_additives"] = []
                st.session_state["extracted_additives"] = []
                st.session_state["ai_suggested_additives"] = []
                st.session_state["suggested_candidates"] = []
                st.session_state["suggested_source"] = "none"
            st.session_state["batch_pdf_cleared_toast"] = True
            st.rerun()

        if st.session_state.pop("batch_pdf_cleared_toast", False):
            st.success(f"🗑️ {tr('batch_pdf_cleared_notice')}")

        trigger_batch_process = batch_process_btn or load_sample_pdf_btn
        if trigger_batch_process:
            st.session_state.pop("bridged_additive_name", None)
            st.session_state["show_formulation_reminder_modal"] = False
            files_to_process = []
            if load_sample_pdf_btn:
                sample_pdf_file = Path(__file__).resolve().parent / "samples" / "sample_zib_paper.pdf"
                if sample_pdf_file.exists():
                    import io
                    sample_bytes = sample_pdf_file.read_bytes()
                    sample_stream = io.BytesIO(sample_bytes)
                    sample_stream.name = "sample_zib_paper.pdf"
                    files_to_process = [sample_stream]
            elif uploaded_pdfs:
                files_to_process = list(uploaded_pdfs)

            if files_to_process:
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                batch_results = []
                extracted_additives_list = []
                active_model = st.session_state.get("gemini_selected_model", "gemini-3.5-flash-lite")
                active_batch_gemini_key = st.session_state.get("gemini_api_key") or (batch_k_val.strip() if 'batch_k_val' in locals() and batch_k_val else None) or os.environ.get("GEMINI_API_KEY")
                extractor = PaperExtractor(api_key=active_batch_gemini_key, model_name=active_model)
                lit_engine = LiteratureDiscoveryEngine(gemini_api_key=active_batch_gemini_key, llm_model=active_model)

                for i, pdf_item in enumerate(files_to_process):
                    fname = getattr(pdf_item, "name", f"document_{i+1}.pdf")
                    status_text.markdown(f"🔄 {tr('batch_pdf_progress_text', filename=fname, current=i+1, total=len(files_to_process))}")
                    pct_step = 0.05 + 0.65 * (i / len(files_to_process))
                    progress_bar.progress(pct_step, text=f"🔄 {tr('batch_pdf_progress_text', filename=fname, current=i+1, total=len(files_to_process))}")

                    raw_text = extractor.extract_text_from_pdf(pdf_item)
                    additives, summary = extractor.extract_from_text(raw_text)

                    if not additives:
                        batch_results.append({
                            "filename": fname,
                            "additive": None,
                            "suggested_analogue": None,
                            "summary": summary or tr("no_additive_identified_desc")
                        })
                    else:
                        for add in additives:
                            analogues = lit_engine.recommend_candidates_for_additive(add, top_n=1)
                            top_analogue = analogues[0] if analogues else None
                            batch_results.append({
                                "filename": fname,
                                "additive": add,
                                "suggested_analogue": top_analogue,
                                "summary": summary
                            })
                            extracted_additives_list.append(add)

                    progress_bar.progress(0.05 + 0.65 * ((i + 1) / len(files_to_process)))

                if extracted_additives_list:
                    progress_bar.progress(0.75, text=f"🧪 {tr('mining_stage_pubchem')}...")
                    u_pdf = []
                    s_pdf = set()
                    for a in extracted_additives_list:
                        if a and getattr(a, "name", None):
                            nl = a.name.lower().strip()
                            if nl not in s_pdf:
                                s_pdf.add(nl)
                                u_pdf.append(a)
                    progress_bar.progress(0.85, text=f"🧠 {tr('mining_stage_consensus')}...")
                    t1_cands, t2_cands = lit_engine.generate_two_tier_candidates(u_pdf, use_pubchem=True)
                    for tc in t1_cands:
                        tc.similarity_metric = "Batch PDF Ingestion"
                        matched_fname = next((item["filename"] for item in batch_results if item.get("additive") and item["additive"].name.lower() == tc.name.lower()), "Batch Ingested PDF")
                        tc.rationale = f"Extracted from batch literature PDF: {matched_fname}"
                    _merge_candidates_into_session(t1_cands, t2_cands)
                    st.session_state["extracted_additives"] = u_pdf
                    st.session_state["suggested_source"] = "pdf_database"
                    if len(u_pdf) >= 1 and not is_formulation_reminder_hidden():
                        st.session_state["show_formulation_reminder_modal"] = True
                else:
                    st.session_state["extracted_additives"] = []
                    st.session_state["suggested_source"] = "none"

                progress_bar.progress(1.0, text=tr("batch_pdf_success_notice", docs_count=len(files_to_process), additives_count=len(extracted_additives_list)))
                st.session_state["batch_pdf_extractions"] = batch_results
                status_text.markdown(f"✅ {tr('batch_pdf_success_notice', docs_count=len(files_to_process), additives_count=len(extracted_additives_list))}")

        batch_saved = st.session_state.get("batch_pdf_extractions", [])
        if batch_saved:
            st.markdown(f'<div class="section-title">{tr("batch_summary_title")}</div>', unsafe_allow_html=True)
            st.caption(tr("batch_summary_caption"))

            # Interactive DataFrame summary table
            summary_table_rows = []
            for item in batch_saved:
                add = item["additive"]
                sugg = item.get("suggested_analogue")
                if add is not None:
                    formula = get_chemical_formula(add.smiles or "", add.name)
                    if add.optimal_concentration and add.optimal_concentration.raw_text:
                        conc_str = clean_scientific_text(add.optimal_concentration.raw_text)
                    elif add.optimal_concentration and add.optimal_concentration.value is not None:
                        conc_str = f"{add.optimal_concentration.value:g} {add.optimal_concentration.unit or 'mM'}"
                    else:
                        conc_str = tr("card_reported_conc_unspecified")

                    raw_salt = add.battery_context.baseline_electrolyte if (add.battery_context and add.battery_context.baseline_electrolyte) else "Unspecified / Not reported"
                    is_unspec_salt = (not raw_salt or "unspecified" in str(raw_salt).lower() or "not reported" in str(raw_salt).lower())
                    base_salt = tr("baseline_salt_unspecified") if is_unspec_salt else raw_salt
                    mechs_str = ", ".join(add.primary_mechanisms) if add.primary_mechanisms else (add.mechanism_summary or "")
                    sugg_source = "PubChem" if (sugg and "PubChem" in sugg.similarity_metric) else "Curated"
                    sugg_str = f"{sugg.name} ({get_chemical_formula(sugg.smiles, sugg.name)}) [Sim: {sugg.similarity_score:.2f} | {sugg_source}]" if sugg else "—"
                    summary_table_rows.append({
                        tr("col_pdf_filename"): item["filename"],
                        tr("col_pdf_additive"): add.name,
                        tr("col_pdf_formula"): format_chem_formula(formula),
                        tr("col_pdf_concentration"): conc_str,
                        tr("col_pdf_baseline_salt"): format_chem_formula(base_salt),
                        tr("col_pdf_mechanism"): format_chem_formula(mechs_str),
                        tr("col_pdf_suggested_additive"): sugg_str,
                    })
                else:
                    summary_table_rows.append({
                        tr("col_pdf_filename"): item["filename"],
                        tr("col_pdf_additive"): f"⚠️ {tr('no_additive_identified_title')}",
                        tr("col_pdf_formula"): "—",
                        tr("col_pdf_concentration"): "—",
                        tr("col_pdf_baseline_salt"): "—",
                        tr("col_pdf_mechanism"): "—",
                        tr("col_pdf_suggested_additive"): "—",
                    })

            st.dataframe(
                pd.DataFrame(summary_table_rows),
                column_config={
                    tr("col_pdf_filename"): st.column_config.TextColumn(tr("col_pdf_filename"), width="medium"),
                    tr("col_pdf_additive"): st.column_config.TextColumn(tr("col_pdf_additive"), width="medium"),
                    tr("col_pdf_formula"): st.column_config.TextColumn(tr("col_pdf_formula"), width="small"),
                    tr("col_pdf_concentration"): st.column_config.TextColumn(tr("col_pdf_concentration"), width="small"),
                    tr("col_pdf_baseline_salt"): st.column_config.TextColumn(tr("col_pdf_baseline_salt"), width="small"),
                    tr("col_pdf_mechanism"): st.column_config.TextColumn(tr("col_pdf_mechanism"), width="medium"),
                    tr("col_pdf_suggested_additive"): st.column_config.TextColumn(tr("col_pdf_suggested_additive"), width="medium"),
                },
                hide_index=True,
                use_container_width=True,
                height=min(240, 45 + len(summary_table_rows) * 35),
            )

            col_b_dl1, col_b_dl2 = st.columns(2)
            batch_cache_key = tuple((item.get("filename", ""), getattr(item.get("additive"), "name", "")) for item in batch_saved)
            if st.session_state.get("cached_batch_pdf_key") == batch_cache_key and "cached_batch_pdf_csv" in st.session_state and "cached_batch_pdf_xlsx" in st.session_state:
                b_csv_bytes = st.session_state["cached_batch_pdf_csv"]
                b_xlsx_bytes = st.session_state["cached_batch_pdf_xlsx"]
            else:
                b_csv_bytes = generate_mined_literature_csv(batch_saved)
                b_xlsx_bytes = generate_mined_literature_xlsx(batch_saved)
                st.session_state["cached_batch_pdf_csv"] = b_csv_bytes
                st.session_state["cached_batch_pdf_xlsx"] = b_xlsx_bytes
                st.session_state["cached_batch_pdf_key"] = batch_cache_key
            with col_b_dl1:
                st.download_button(
                    f"📥 {tr('btn_download_batch_pdf_csv')}",
                    data=b_csv_bytes,
                    file_name=f"MAP_Batch_PDF_Summary_{len(batch_saved)}Docs.csv",
                    mime="text/csv",
                    key="btn_dl_batch_pdf_csv",
                    use_container_width=True,
                )
            with col_b_dl2:
                st.download_button(
                    f"📊 {tr('btn_download_batch_pdf_xlsx')}",
                    data=b_xlsx_bytes,
                    file_name=f"MAP_Batch_PDF_Dossier_{len(batch_saved)}Docs.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="btn_dl_batch_pdf_xlsx",
                    use_container_width=True,
                )

            # Detailed Cards with Direct "Apply as Active Additive" Action Buttons
            for b_idx, item in enumerate(batch_saved):
                add = item["additive"]
                fname = item["filename"]
                sugg = item.get("suggested_analogue")

                if add is None:
                    card_html = (
                        f'<div class="callout-box" style="border-left-color: #f7b731; margin-bottom: 12px;">'
                        f'<div style="font-size: 0.95rem; font-weight: 700; color: inherit; margin-bottom: 4px;">'
                        f'📄 <em>{fname}</em> ➔ ⚠️ {tr("no_additive_identified_title")}'
                        f'</div>'
                        f'<div style="font-size: 0.85rem; color: inherit;">'
                        f'{item.get("summary", tr("no_additive_identified_desc"))}'
                        f'</div>'
                        f'</div>'
                    )
                    st.markdown(card_html, unsafe_allow_html=True)
                    continue

                formula = format_chem_formula(get_chemical_formula(add.smiles or "", add.name))
                formula_disp = f" ({formula})" if formula else ""
                if add.optimal_concentration and add.optimal_concentration.raw_text:
                    conc_str = clean_scientific_text(add.optimal_concentration.raw_text)
                elif add.optimal_concentration and add.optimal_concentration.value is not None:
                    conc_str = f"{add.optimal_concentration.value:g} {add.optimal_concentration.unit or 'mM'}"
                else:
                    conc_str = f'<span class="badge-tag badge-amber">⚠️ {tr("card_reported_conc_unspecified")}</span>'

                raw_salt = add.battery_context.baseline_electrolyte if (add.battery_context and add.battery_context.baseline_electrolyte) else "Unspecified / Not reported"
                is_unspec_salt = (not raw_salt or "unspecified" in str(raw_salt).lower() or "not reported" in str(raw_salt).lower())
                if is_unspec_salt:
                    salt_display = f'<span class="badge-tag badge-amber">⚠️ {tr("baseline_salt_unspecified")}</span>'
                else:
                    salt_display = format_chem_formula(raw_salt)

                mechs_str = format_chem_formula(clean_scientific_text(", ".join(add.primary_mechanisms) if add.primary_mechanisms else add.mechanism_summary))
                evidence = clean_scientific_text(add.extraction_evidence or add.mechanism_summary or item.get("summary", ""))

                sugg_section_html = ""
                if sugg:
                    sugg_formula = format_chem_formula(get_chemical_formula(sugg.smiles, sugg.name))
                    source_badge = '<span class="badge-tag badge-teal">PubChem Live</span>' if 'PubChem' in sugg.similarity_metric else '<span class="badge-tag badge-teal">Curated Database</span>'
                    sugg_section_html = (
                        f'<div style="margin-top: 8px; padding-top: 6px; border-top: 1px dashed rgba(128,128,128,0.3); font-size: 0.88rem;">'
                        f'✨ <strong>{tr("suggested_analogue_card_label")}:</strong> <strong>{sugg.name}</strong> ({sugg_formula}) {source_badge} '
                        f'— <span class="badge-tag badge-purple">Tanimoto: {sugg.similarity_score:.2f}</span> '
                        f'<span class="badge-tag badge-blue">Delaney ESOL LogS: {sugg.logs:.2f}</span>'
                        f'<div style="font-size: 0.80rem; margin-top: 2px;"><em>{clean_scientific_text(sugg.rationale or "")}</em></div>'
                        f'</div>'
                    )

                card_html = (
                    f'<div class="paper-analysis-box">'
                    f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">'
                    f'<div style="font-size: 1.0rem; font-weight: 700; color: inherit;">'
                    f'📄 <em>{fname}</em> ➔ <strong>{add.name}</strong>{formula_disp}'
                    f'</div>'
                    f'<span class="badge-tag badge-teal">{salt_display}</span>'
                    f'</div>'
                    f'<div style="font-size: 0.88rem; color: inherit; margin-bottom: 4px;">'
                    f'• <strong>{tr("extracted_optimal_conc")}:</strong> {conc_str} | <strong>{tr("extracted_baseline_salt")}:</strong> {salt_display}'
                    f'</div>'
                    f'<div style="font-size: 0.88rem; color: inherit; margin-bottom: 6px;">'
                    f'• <strong>{tr("extracted_mechanisms")}:</strong> {mechs_str}'
                    f'</div>'
                    f'<div style="font-size: 0.82rem; color: inherit;">'
                    f'<em>{evidence}</em>'
                    f'</div>'
                    f'{sugg_section_html}'
                    f'</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)

                col_b_act1, col_b_act2 = st.columns(2)
                with col_b_act1:
                    if st.button(
                        f"🚀 {tr('btn_apply_pdf_additive')}: {add.name}",
                        key=f"btn_apply_pdf_{b_idx}",
                        type="primary",
                        use_container_width=True,
                        help=tr("btn_apply_pdf_additive_help")
                    ):
                        ext_name = add.name
                        ext_conc = float(add.optimal_concentration.value) if (add.optimal_concentration and add.optimal_concentration.value is not None) else 15.0
                        raw_salt_apply = add.battery_context.baseline_electrolyte if (add.battery_context and add.battery_context.baseline_electrolyte) else "Unspecified / Not reported"
                        is_unspec_apply = (not raw_salt_apply or "unspecified" in str(raw_salt_apply).lower() or "not reported" in str(raw_salt_apply).lower())
                        ext_salt = selected_target_salt if is_unspec_apply else format_chem_formula(raw_salt_apply)
                        ext_smiles = add.smiles
                        if not ext_smiles:
                            matching = next((a["smiles"] for a in CURATED_ZIB_ADDITIVES if a["name"].lower() == ext_name.lower()), None)
                            ext_smiles = matching or "CC[N+](CC)(CC)CC.[Br-]"

                        custom_list = st.session_state.get("custom_additives", [])
                        if not any(a["name"].lower() == ext_name.lower() for a in CURATED_ZIB_ADDITIVES + custom_list):
                            custom_list.append({"name": ext_name, "smiles": ext_smiles})
                            st.session_state["custom_additives"] = custom_list

                        existing_t1 = st.session_state.get("mined_literature_additives", [])
                        found_t1 = next((c for c in existing_t1 if c.name.lower() == ext_name.lower()), None)
                        if not found_t1:
                            sim_engine = get_similarity_engine()
                            desc = sim_engine.calculate_descriptors(ext_smiles) if ext_smiles else None
                            esol = desc["logs"] if desc and "logs" in desc else None
                            new_t1 = CandidateMolecule(
                                name=ext_name,
                                smiles=ext_smiles,
                                similarity_score=1.0,
                                similarity_metric="Batch PDF Ingestion",
                                molecular_weight=desc["molecular_weight"] if desc else 0.0,
                                logp=desc["logp"] if desc else 0.0,
                                logs=esol,
                                tpsa=desc["tpsa"] if desc else 0.0,
                                hbd=desc["hbd"] if desc else 0,
                                hba=desc["hba"] if desc else 0,
                                rotatable_bonds=desc["rotatable_bonds"] if desc else 0,
                                reported_concentration=ext_conc,
                                reported_concentration_unit="mM",
                                parent_baseline_electrolyte=ext_salt,
                                is_literature_additive=True,
                                rationale=f"Extracted from batch literature PDF: {fname}",
                            )
                            existing_t1.insert(0, new_t1)
                            st.session_state["mined_literature_additives"] = existing_t1
                        else:
                            found_t1.similarity_metric = "Batch PDF Ingestion"
                            found_t1.rationale = f"Extracted from batch literature PDF: {fname}"
                            if ext_conc:
                                found_t1.reported_concentration = ext_conc
                            if ext_salt:
                                found_t1.parent_baseline_electrolyte = ext_salt

                        st.session_state["tab2_mol_select"] = ext_name
                        st.session_state["selected_additive_name"] = ext_name
                        st.session_state["selected_tier1_name"] = ext_name
                        for t1_i, c in enumerate(st.session_state.get("mined_literature_additives", [])):
                            if c.name.lower() == ext_name.lower():
                                st.session_state["last_tier1_choice"] = t1_i
                                st.session_state["last_bottom_tier1_choice"] = t1_i
                                st.session_state["pending_tier1_choice"] = t1_i
                                st.session_state["pending_tier1_bottom_choice"] = t1_i
                                st.session_state["step2_tier1_current_page"] = (t1_i // 6) + 1
                                break
                        st.session_state["custom_additive_name"] = ext_name
                        st.session_state["custom_smiles"] = ext_smiles
                        st.session_state["tab2_custom_additive_name_input"] = ext_name
                        st.session_state["tab2_custom_smiles_input"] = ext_smiles
                        st.session_state["last_suggested_choice"] = st.session_state.get("selected_suggested_idx", 0)
                        st.session_state["published_conc_input"] = ext_conc
                        st.session_state["parent_electrolyte_input"] = ext_salt

                        sync_custom_formulation_pipeline(
                            additive_name=ext_name,
                            published_conc_mM=ext_conc,
                            parent_electrolyte=ext_salt,
                            target_electrolyte=selected_target_salt,
                            smiles=ext_smiles,
                            doe_method=doe_method,
                            num_samples=num_doe_samples,
                            seed=42,
                            rack_format=rack_format,
                            total_volume_uL=total_vol_ul,
                            session_state=st.session_state,
                        )
                        st.session_state["bridged_additive_name"] = ext_name
                        toast_pdf_conc = f"{ext_conc:.1f} mM" if (add.optimal_concentration and add.optimal_concentration.value is not None) else f"{ext_conc:.1f} mM (default)"
                        st.session_state["last_extracted_notice"] = tr("extraction_success_toast", name=ext_name, conc=toast_pdf_conc, parent=format_chem_formula(ext_salt))
                        st.rerun()

                with col_b_act2:
                    if sugg:
                        if st.button(
                            f"✨ {tr('btn_apply_suggested_pdf_additive', name=sugg.name)}",
                            key=f"btn_apply_sugg_{b_idx}",
                            use_container_width=True,
                            help=tr("btn_apply_suggested_pdf_additive_help")
                        ):
                            custom_list = st.session_state.get("custom_additives", [])
                            if not any(a["name"].lower() == sugg.name.lower() for a in CURATED_ZIB_ADDITIVES + custom_list):
                                custom_list.append({"name": sugg.name, "smiles": sugg.smiles})
                                st.session_state["custom_additives"] = custom_list

                            sugg_conc = float(add.optimal_concentration.value) if (add.optimal_concentration and add.optimal_concentration.value is not None) else 15.0
                            raw_salt_sugg = add.battery_context.baseline_electrolyte if (add.battery_context and add.battery_context.baseline_electrolyte) else "Unspecified / Not reported"
                            is_unspec_sugg = (not raw_salt_sugg or "unspecified" in str(raw_salt_sugg).lower() or "not reported" in str(raw_salt_sugg).lower())
                            sugg_salt = selected_target_salt if is_unspec_sugg else format_chem_formula(raw_salt_sugg)

                            existing_t2 = st.session_state.get("ai_suggested_additives", [])
                            found_t2 = next((c for c in existing_t2 if c.name.lower() == sugg.name.lower()), None)
                            if not found_t2:
                                existing_t2.insert(0, sugg)
                                st.session_state["ai_suggested_additives"] = existing_t2
                                st.session_state["suggested_candidates"] = existing_t2

                            st.session_state["tab2_mol_select"] = sugg.name
                            st.session_state["selected_additive_name"] = sugg.name
                            st.session_state["selected_tier2_name"] = sugg.name
                            for t2_i, c in enumerate(st.session_state.get("ai_suggested_additives", [])):
                                if c.name.lower() == sugg.name.lower():
                                    st.session_state["last_suggested_choice"] = t2_i
                                    st.session_state["last_bottom_tier2_choice"] = t2_i
                                    st.session_state["pending_tier2_choice"] = t2_i
                                    st.session_state["pending_tier2_bottom_choice"] = t2_i
                                    st.session_state["step2_tier2_current_page"] = (t2_i // 6) + 1
                                    break
                            st.session_state["custom_additive_name"] = sugg.name
                            st.session_state["custom_smiles"] = sugg.smiles
                            st.session_state["published_conc_input"] = sugg_conc
                            st.session_state["parent_electrolyte_input"] = sugg_salt

                            new_anchor, _, _ = sync_custom_formulation_pipeline(
                                additive_name=sugg.name,
                                published_conc_mM=sugg_conc,
                                parent_electrolyte=sugg_salt,
                                target_electrolyte=selected_target_salt,
                                smiles=sugg.smiles,
                                doe_method=doe_method,
                                num_samples=num_doe_samples,
                                seed=42,
                                rack_format=rack_format,
                                total_volume_uL=total_vol_ul,
                                session_state=st.session_state,
                            )
                            st.session_state["bridged_additive_name"] = sugg.name
                            hard_ceil = new_anchor.solubility_ceiling_mM or 100.0
                            st.session_state["last_extracted_notice"] = tr("suggested_additive_applied_toast", name=sugg.name, ceiling=hard_ceil)
                            st.rerun()
        else:
            st.info(tr("no_pdf_uploaded_tip"))

    if "bridged_additive_name" in st.session_state and st.session_state["bridged_additive_name"]:
        b_name = st.session_state.get("bridged_additive_name")
        st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
        col_br1, col_br2 = st.columns([0.88, 0.12], vertical_alignment="center")
        with col_br1:
            st.info(f"✨ **{b_name}** {tr('bridged_to_step2_notice')}")
        with col_br2:
            st.markdown('<div style="height: 4px;"></div>', unsafe_allow_html=True)
            if st.button(f"✕ {tr('btn_dismiss_notice')}", key="btn_dismiss_bridged_notice", use_container_width=True):
                st.session_state.pop("bridged_additive_name", None)
                st.rerun()

    # Post-Extraction Reminder Modal
    if st.session_state.get("show_formulation_reminder_modal") and (
        not is_formulation_reminder_hidden() or st.session_state.get("has_merge_conflict_prompt")
    ):
        _render_post_extraction_reminder_dialog(is_demo_mode=is_demo)




# =============================================================================
# TAB 3: STEP 2 FORMULATION & ROBOTIC DISPENSING
# =============================================================================
with tab_step2:
    if not is_demo:
        st.markdown(f"### **{tr('tab3_title_custom')}**")
        st.markdown(tr("tab3_desc_custom"))
    else:
        st.markdown(f"### **{tr('tab3_title')}**")
        st.markdown(tr("tab3_desc"))

        # Step 2 Educational Explanation Guide (Demo Mode Only)
        with st.expander(f"💡 {tr('step2_demo_guide_title')}", expanded=True):
            st.markdown(
                f"""
                <p style="font-size: 0.95rem; margin-bottom: 12px;">{tr('step2_demo_guide_intro')}</p>
                <div style="display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); grid-auto-rows: 1fr; gap: 10px 16px; align-items: stretch;">
                    <div style="background: rgba(31, 119, 180, 0.06); border-left: 3px solid #1f77b4; padding: 10px 14px; border-radius: 4px; height: 100%; box-sizing: border-box;">
                        <strong style="color: #1f77b4;">{tr('step2_demo_guide_step1_title')}</strong><br>
                        <span style="font-size: 0.88rem;">{tr('step2_demo_guide_step1_desc')}</span>
                    </div>
                    <div style="background: rgba(44, 160, 44, 0.06); border-left: 3px solid #2ca02c; padding: 10px 14px; border-radius: 4px; height: 100%; box-sizing: border-box;">
                        <strong style="color: #2ca02c;">{tr('step2_demo_guide_step3_title')}</strong><br>
                        <span style="font-size: 0.88rem;">{tr('step2_demo_guide_step3_desc')}</span>
                    </div>
                    <div style="background: rgba(230, 57, 70, 0.06); border-left: 3px solid #e63946; padding: 10px 14px; border-radius: 4px; height: 100%; box-sizing: border-box;">
                        <strong style="color: #e63946;">{tr('step2_demo_guide_step2_title')}</strong><br>
                        <span style="font-size: 0.88rem;">{tr('step2_demo_guide_step2_desc')}</span>
                    </div>
                    <div style="background: rgba(255, 127, 14, 0.06); border-left: 3px solid #ff7f0e; padding: 10px 14px; border-radius: 4px; height: 100%; box-sizing: border-box;">
                        <strong style="color: #ff7f0e;">{tr('step2_demo_guide_step4_title')}</strong><br>
                        <span style="font-size: 0.88rem;">{tr('step2_demo_guide_step4_desc')}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    all_additives = get_all_additives()
    mol_options = [a["name"] for a in all_additives]
    salt_options = list(CANONICAL_ELECTROLYTE_PROFILES.keys())

    if is_demo:
        st.markdown(f'<div style="font-size: 0.88rem; font-weight: 600; margin: 8px 0 6px 0; color: inherit;">💡 {tr("preloaded_examples_label")}</div>', unsafe_allow_html=True)
        col_ex1, col_ex2, col_ex3, col_ex4, col_ex5 = st.columns(5)
        with col_ex1:
            if st.button("D-Sorbitol", key="btn_demo_ex_sorbitol", use_container_width=True):
                st.session_state["tab2_mol_select"] = "D-Sorbitol"
                st.session_state["selected_additive_name"] = "D-Sorbitol"
                st.rerun()
        with col_ex2:
            if st.button("Trehalose", key="btn_demo_ex_trehalose", use_container_width=True):
                st.session_state["tab2_mol_select"] = "Trehalose"
                st.session_state["selected_additive_name"] = "Trehalose"
                st.rerun()
        with col_ex3:
            if st.button("TEABr", key="btn_demo_ex_teabr", use_container_width=True):
                st.session_state["tab2_mol_select"] = "Tetraethylammonium bromide"
                st.session_state["selected_additive_name"] = "Tetraethylammonium bromide"
                st.rerun()
        with col_ex4:
            if st.button("Thiourea", key="btn_demo_ex_thiourea", use_container_width=True):
                st.session_state["tab2_mol_select"] = "Thiourea"
                st.session_state["selected_additive_name"] = "Thiourea"
                st.rerun()
        with col_ex5:
            if st.button("Glucose", key="btn_demo_ex_glucose", use_container_width=True):
                st.session_state["tab2_mol_select"] = "Glucose"
                st.session_state["selected_additive_name"] = "Glucose"
                st.rerun()

        selected_target_salt = st.session_state.get("step2_target_electrolyte_select") or st.session_state.get("selected_target_salt", "Zn(OTf)2")
        st.session_state["selected_target_salt"] = selected_target_salt

        cur_tab2_mol = st.session_state.get("tab2_mol_select", mol_options[0])
        cur_tab2_idx = mol_options.index(cur_tab2_mol) if cur_tab2_mol in mol_options else 0
        selected_mol_name = st.selectbox(
            tr("select_additive_inspect"),
            mol_options,
            index=cur_tab2_idx,
        )
        prev_inspect_mol = st.session_state.get("_last_inspect_mol_select")
        st.session_state["_last_inspect_mol_select"] = selected_mol_name
        if prev_inspect_mol is None or selected_mol_name != prev_inspect_mol:
            st.session_state["tab2_mol_select"] = selected_mol_name
            st.session_state["selected_additive_name"] = selected_mol_name
            if prev_inspect_mol is not None and selected_mol_name != prev_inspect_mol:
                for t1_i, c in enumerate(st.session_state.get("mined_literature_additives", [])):
                    if c.name.lower() == selected_mol_name.lower():
                        st.session_state["selected_tier1_name"] = c.name
                        st.session_state["last_tier1_choice"] = t1_i
                        st.session_state["last_bottom_tier1_choice"] = t1_i
                        st.session_state["pending_tier1_choice"] = t1_i
                        st.session_state["pending_tier1_bottom_choice"] = t1_i
                        st.session_state["step2_tier1_current_page"] = (t1_i // 6) + 1
                        break
                for t2_i, c in enumerate(st.session_state.get("ai_suggested_additives", [])):
                    if c.name.lower() == selected_mol_name.lower():
                        st.session_state["selected_tier2_name"] = c.name
                        st.session_state["last_suggested_choice"] = t2_i
                        st.session_state["last_bottom_tier2_choice"] = t2_i
                        st.session_state["pending_tier2_choice"] = t2_i
                        st.session_state["pending_tier2_bottom_choice"] = t2_i
                        st.session_state["step2_tier2_current_page"] = (t2_i // 6) + 1
                        break

        active_mol_name = st.session_state.get("selected_additive_name") or selected_mol_name
        target_info = next((a for a in all_additives if a["name"].lower() == active_mol_name.lower()), all_additives[0])
        target_name = target_info["name"]
        target_smiles = target_info["smiles"]
        target_formula = get_chemical_formula(target_smiles, target_name)

        render_two_tier_card_deck(
            selected_target_salt=selected_target_salt,
            doe_method=doe_method,
            num_doe_samples=num_doe_samples,
            rack_format=rack_format,
            total_vol_ul=total_vol_ul,
            published_conc_input=published_conc_input,
            parent_electrolyte_input=parent_electrolyte_input,
            is_demo=True,
        )
        if st.session_state.get("selected_additive_name") and st.session_state.get("selected_additive_name") != target_name:
            target_name = st.session_state["selected_additive_name"]
            all_additives = get_all_additives()
            target_info = next((a for a in all_additives if a["name"].lower() == target_name.lower()), None)
            target_smiles = target_info["smiles"] if target_info else st.session_state.get("custom_smiles", "")
            target_formula = get_chemical_formula(target_smiles, target_name) if target_smiles else ""
    else:
        # Custom Experiment Mode
        st.markdown(f"""
        <div class="callout-box" style="border-left-color: #02c39a; margin-bottom: 12px;">
            {tr("custom_mode_banner")}
        </div>
        """, unsafe_allow_html=True)

        selected_target_salt = st.session_state.get("step2_target_electrolyte_select") or st.session_state.get("selected_target_salt", "Zn(OTf)2")
        st.session_state["selected_target_salt"] = selected_target_salt

        render_two_tier_card_deck(
            selected_target_salt=selected_target_salt,
            doe_method=doe_method,
            num_doe_samples=num_doe_samples,
            rack_format=rack_format,
            total_vol_ul=total_vol_ul,
            published_conc_input=published_conc_input,
            parent_electrolyte_input=parent_electrolyte_input,
            is_demo=False,
        )

        tier1_present = bool(st.session_state.get("mined_literature_additives", []))
        tier2_present = bool(st.session_state.get("ai_suggested_additives", []) or st.session_state.get("suggested_candidates", []))

        if tier1_present or tier2_present:
            with st.expander(f"✏️ {tr('manual_custom_additive_override_label')}", expanded=False):
                c_col1, c_col2 = st.columns(2)
                with c_col1:
                    custom_name_val = st.text_input(
                        tr("custom_additive_name_label"),
                        value=st.session_state.get("custom_additive_name", selected_additive_name if is_demo else ""),
                        key="tab2_custom_additive_name_input",
                        placeholder=tr("custom_additive_name_placeholder"),
                        help=tr("custom_additive_name_help"),
                    ).strip()
                    target_name = custom_name_val if custom_name_val else ""
                with c_col2:
                    custom_smiles_val = st.text_input(
                        tr("custom_smiles_label"),
                        value=st.session_state.get("custom_smiles", selected_smiles if is_demo else ""),
                        key="tab2_custom_smiles_input",
                        placeholder=tr("custom_smiles_placeholder"),
                        help=tr("custom_smiles_help"),
                    ).strip()
                    target_smiles = custom_smiles_val if custom_smiles_val else ""
        else:
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                custom_name_val = st.text_input(
                    tr("custom_additive_name_label"),
                    value=st.session_state.get("custom_additive_name", selected_additive_name if is_demo else ""),
                    key="tab2_custom_additive_name_input",
                    placeholder=tr("custom_additive_name_placeholder"),
                    help=tr("custom_additive_name_help"),
                ).strip()
                target_name = custom_name_val if custom_name_val else ""
            with c_col2:
                custom_smiles_val = st.text_input(
                    tr("custom_smiles_label"),
                    value=st.session_state.get("custom_smiles", selected_smiles if is_demo else ""),
                    key="tab2_custom_smiles_input",
                    placeholder=tr("custom_smiles_placeholder"),
                    help=tr("custom_smiles_help"),
                ).strip()
                target_smiles = custom_smiles_val if custom_smiles_val else ""


        if target_name:
            st.session_state["custom_additive_name"] = target_name
            st.session_state["selected_additive_name"] = target_name
            st.session_state["tab2_mol_select"] = target_name
        if target_smiles:
            st.session_state["custom_smiles"] = target_smiles
        target_formula = get_chemical_formula(target_smiles, target_name) if target_smiles else ""

    tier1_cands_present = bool(st.session_state.get("mined_literature_additives", []))
    tier2_cands_present = bool(st.session_state.get("ai_suggested_additives", []) or st.session_state.get("suggested_candidates", []))

    if target_name and not tier1_cands_present and not tier2_cands_present:
        _render_candidate_chemical_dossier(
            cand_name=target_name,
            cand_smiles=target_smiles or "",
            is_tier2=False,
            parent_electrolyte_input=parent_electrolyte_input,
            cur_lang=cur_lang,
            key_prefix="standalone",
        )
    elif not target_name and not tier1_cands_present and not tier2_cands_present:
        st.info(tr("custom_mode_step1_empty_notice"))

    if is_demo and target_smiles:
        st.markdown(f'<div class="section-title">{tr("sec_additive_pool")}</div>', unsafe_allow_html=True)
        df_sim = compute_similarity_ranking(target_smiles)
        st.dataframe(
            df_sim,
            column_config={
                "Candidate Molecule": st.column_config.TextColumn("Candidate Molecule", width="medium"),
                "Chemical Formula": st.column_config.TextColumn("Chemical Formula", width="medium"),
                "Tanimoto Similarity": st.column_config.ProgressColumn(
                    "Tanimoto Similarity",
                    format="%.3f",
                    min_value=0.0,
                    max_value=1.0,
                ),
                "Aqueous Saturation (mM)": st.column_config.NumberColumn(
                    "Aqueous Saturation (mM)",
                    format="%.1f mM",
                ),
                "80% Ceiling (mM)": st.column_config.NumberColumn(
                    "80% Ceiling (mM)",
                    format="%.1f mM",
                ),
                "MW (g/mol)": st.column_config.NumberColumn("MW (g/mol)", format="%.2f"),
                "cLogP": st.column_config.NumberColumn("cLogP", format="%.2f"),
                "Primary Functional Group": st.column_config.TextColumn("Primary Functional Group", width="medium"),
            },
            hide_index=True,
            use_container_width=True,
            height=280,
        )

    # Use campaign's existing formulation plan and schedule if available
    campaign_res = st.session_state.get("custom_campaign_result") if not is_demo else st.session_state.campaign_result

    if not is_demo:
        # Custom Experiment Mode: only display if a custom plan has been generated/transferred
        plan = st.session_state.get("custom_formulation_plan") or st.session_state.get("extracted_formulation_plan")
        schedule = st.session_state.get("custom_dispensing_schedule") or st.session_state.get("extracted_dispensing_schedule")
        anchor = plan.anchor if plan else None
    else:
        if st.session_state.get("extracted_formulation_plan"):
            plan = st.session_state["extracted_formulation_plan"]
            schedule = st.session_state["extracted_dispensing_schedule"]
            anchor = plan.anchor
        elif campaign_res and campaign_res.formulation_plan:
            plan = campaign_res.formulation_plan
            schedule = campaign_res.dispensing_schedule
            anchor = plan.anchor
            st.session_state["extracted_formulation_plan"] = plan
            st.session_state["extracted_dispensing_schedule"] = schedule
        else:
            plan = None
            schedule = None
            anchor = None

    # Immediate cross-salt and molecule synchronization
    if target_name:
        if plan is None or anchor is None or anchor.additive_name != target_name or anchor.target_electrolyte_salt != selected_target_salt or anchor.smiles != (target_smiles or ""):
            anchor, plan, schedule = sync_custom_formulation_pipeline(
                additive_name=target_name,
                published_conc_mM=published_conc_input,
                parent_electrolyte=parent_electrolyte_input,
                target_electrolyte=selected_target_salt,
                smiles=target_smiles,
                doe_method=doe_method,
                num_samples=num_doe_samples,
                seed=42,
                rack_format=rack_format,
                total_volume_uL=total_vol_ul,
                session_state=st.session_state,
                update_custom=(not is_demo),
            )
    elif plan is not None and anchor is not None:
        if anchor.target_electrolyte_salt != selected_target_salt:
            anchor, plan, schedule = sync_custom_formulation_pipeline(
                additive_name=anchor.additive_name,
                published_conc_mM=anchor.published_concentration,
                parent_electrolyte=anchor.parent_electrolyte,
                target_electrolyte=selected_target_salt,
                smiles=anchor.smiles,
                doe_method=doe_method,
                num_samples=num_doe_samples,
                seed=42,
                rack_format=rack_format,
                total_volume_uL=total_vol_ul,
                session_state=st.session_state,
            )

    if plan is None or schedule is None or anchor is None:
        st.info(tr("custom_mode_step2_empty_notice"))
    else:
        # Prominent Consolidated Active Formulation Context Banner (Additive & Electrolytes)
        chem_formula = format_chem_formula(get_chemical_formula(anchor.smiles or "", anchor.additive_name))
        formula_html = f' <span style="font-size: 0.95rem; font-weight: 600;">({chem_formula})</span>' if chem_formula else ""
        parent_elyte_clean = format_chem_formula(anchor.parent_electrolyte or f"{anchor.parent_molarity:.1f} M {anchor.parent_salt}")
        target_elyte_clean = format_chem_formula(anchor.target_electrolyte or f"{anchor.target_salt_molarity:.1f} M {anchor.target_electrolyte_salt}")

        banner_smiles_html = (
            f'<div style="font-size: 0.82rem; margin-top: 4px; word-break: break-all; color: inherit;">'
            f'<strong>{tr("active_smiles_label")}:</strong> <code style="font-size: 0.80rem; background: rgba(128, 128, 128, 0.1); padding: 2px 6px; border-radius: 4px;">{anchor.smiles}</code>'
            f'</div>'
        ) if anchor.smiles else ""

        st.markdown(f'<div class="section-title" style="margin-bottom: 8px;">{tr("sec_active_additive_overview")} &amp; {tr("active_formulation_label")}</div>', unsafe_allow_html=True)

        with st.container(border=True):
            col_b_add, col_b_dropdown, col_b_parent, col_b_arrow, col_b_target = st.columns([1.30, 1.15, 1.10, 0.12, 1.25], vertical_alignment="center")
            with col_b_add:
                st.markdown(
                    f"""
                    <div style="padding: 2px 0;">
                        <div style="font-size: 1.18rem; font-weight: 700;">
                            🧪 <strong>{anchor.additive_name}</strong>{formula_html}
                        </div>
                        {banner_smiles_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_b_dropdown:
                st.markdown(
                    f'<div style="font-size: 0.72rem; text-transform: uppercase; font-weight: 700; color: #2ca02c; margin-bottom: 3px;">🎯 {tr("target_electrolyte")}</div>',
                    unsafe_allow_html=True,
                )
                cur_step2_salt = st.session_state.get("step2_target_electrolyte_select") or st.session_state.get("selected_target_salt", anchor.target_electrolyte_salt or "Zn(OTf)2")
                cur_step2_idx = salt_options.index(cur_step2_salt) if cur_step2_salt in salt_options else 1
                step2_target_salt = st.selectbox(
                    f"🎯 {tr('target_electrolyte')}",
                    salt_options,
                    index=0 if "step2_target_electrolyte_select" in st.session_state else cur_step2_idx,
                    format_func=lambda s: SALT_DISPLAY_MAPS.get(cur_lang, SALT_DISPLAY_MAP).get(s, format_chem_formula(s)),
                    help=tr("target_electrolyte_step2_help"),
                    key="step2_target_electrolyte_select",
                    label_visibility="collapsed",
                )
                if step2_target_salt != st.session_state.get("selected_target_salt") or step2_target_salt != anchor.target_electrolyte_salt:
                    st.session_state["selected_target_salt"] = step2_target_salt
                    st.rerun()

            with col_b_parent:
                st.markdown(
                    f"""
                    <div style="background: rgba(31, 119, 180, 0.1); border: 1px solid rgba(31, 119, 180, 0.3); border-radius: 6px; padding: 8px 12px; text-align: right; min-height: 52px; display: flex; flex-direction: column; justify-content: center;">
                        <div style="font-size: 0.72rem; text-transform: uppercase; font-weight: 700; color: #1f77b4;">{tr("parent_electrolyte_label")}</div>
                        <div style="font-size: 0.95rem; font-weight: 700; color: #1f77b4;">{parent_elyte_clean} <span style="font-size: 0.80rem; font-weight: 600; color: #1f77b4;">({anchor.published_concentration:.1f} mM)</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_b_arrow:
                st.markdown(
                    '<div style="font-size: 1.3rem; text-align: center; color: #1f77b4; font-weight: bold;">➔</div>',
                    unsafe_allow_html=True,
                )
            with col_b_target:
                st.markdown(
                    f"""
                    <div style="background: rgba(44, 160, 44, 0.1); border: 1px solid rgba(44, 160, 44, 0.3); border-radius: 6px; padding: 8px 12px; text-align: left; min-height: 52px; display: flex; flex-direction: column; justify-content: center;">
                        <div style="font-size: 0.72rem; text-transform: uppercase; font-weight: 700; color: #2ca02c;">{tr("target_electrolyte_label")}</div>
                        <div style="font-size: 0.95rem; font-weight: 700; color: #2ca02c;">{target_elyte_clean} <span style="font-size: 0.80rem; font-weight: 600; color: #2ca02c;">(Factor: {anchor.scaling_factor:.2f}x)</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Determine if concentration and salt were explicitly reported or unspecified/assumed
        is_baseline_unspecified = False

        if getattr(anchor, "is_baseline_unspecified", False) or not getattr(anchor, "reported_in_file", True):
            is_baseline_unspecified = True

        tier1_cands = st.session_state.get("mined_literature_additives", [])
        active_c = next((c for c in tier1_cands if c.name and c.name.strip().lower() == anchor.additive_name.strip().lower()), None)
        if active_c is not None:
            if active_c.reported_concentration is None:
                is_baseline_unspecified = True
            raw_s = active_c.parent_baseline_electrolyte
            if not raw_s or "unspecified" in str(raw_s).lower() or "not reported" in str(raw_s).lower():
                is_baseline_unspecified = True

        batch_items = (st.session_state.get("batch_pdf_extractions", []) or []) + (st.session_state.get("batch_extracted_additives", []) or [])
        for item in batch_items:
            add = item.get("additive")
            if add and add.name and add.name.strip().lower() == anchor.additive_name.strip().lower():
                if add.optimal_concentration is None or getattr(add.optimal_concentration, "value", None) is None:
                    is_baseline_unspecified = True
                bc = getattr(add, "battery_context", None)
                raw_s = bc.baseline_electrolyte if bc else None
                if not raw_s or "unspecified" in str(raw_s).lower() or "not reported" in str(raw_s).lower():
                    is_baseline_unspecified = True
                break

        tier2_cands = st.session_state.get("ai_suggested_additives", []) or st.session_state.get("suggested_candidates", [])
        active_t2 = next((c for c in tier2_cands if c.name and c.name.strip().lower() == anchor.additive_name.strip().lower()), None)
        if active_t2 is not None:
            raw_s = active_t2.parent_baseline_electrolyte
            if not raw_s or "unspecified" in str(raw_s).lower() or "not reported" in str(raw_s).lower():
                is_baseline_unspecified = True

        if st.session_state.get("is_active_baseline_unspecified", False):
            is_baseline_unspecified = True

        col_anc1, col_anc2, col_anc3, col_anc4 = st.columns(4)
        with col_anc1:
            if is_baseline_unspecified:
                card_label = f"{tr('metric_assumed_baseline')}: {anchor.published_concentration:.1f} mM ({tr('standard_ref_label', salt=parent_elyte_clean)})"
                st.metric(
                    card_label,
                    f"{anchor.published_concentration:.1f} mM",
                    f"(Standard {parent_elyte_clean} Reference)",
                    help=f"{tr('metric_assumed_baseline')}: {anchor.published_concentration:.1f} mM ({tr('standard_ref_label', salt=parent_elyte_clean)})"
                )
                st.markdown(
                    f'<div style="margin-top: 4px;"><span class="badge-tag badge-amber" style="font-size: 0.76rem; font-weight: 600;">⚠️ {tr("unspecified_in_upload_tag")}</span></div>',
                    unsafe_allow_html=True
                )
            else:
                card_label = f"{tr('metric_lit_baseline')}: {anchor.published_concentration:.1f} mM {tr('in_salt_label', salt=parent_elyte_clean)}"
                st.metric(
                    card_label,
                    f"{anchor.published_concentration:.1f} mM",
                    f"in {parent_elyte_clean}",
                    help=f"{tr('metric_lit_baseline')}: {anchor.published_concentration:.1f} mM in {parent_elyte_clean}"
                )
        with col_anc2:
            st.metric(tr("metric_scaling_factor"), f"{anchor.scaling_factor:.2f}x", f"Shift to {target_elyte_clean}")
        with col_anc3:
            st.metric(tr("metric_target_ref_conc"), f"{anchor.shifted_anchor_conc:.2f} mM", f"Range: [{anchor.min_conc:.1f}, {anchor.max_conc:.1f}] mM")
        with col_anc4:
            st.metric(tr("metric_safety_cap"), f"{anchor.solubility_ceiling_mM:.1f} mM" if anchor.solubility_ceiling_mM else "None", "Capped: " + str(anchor.is_capped_by_solubility))

        st.markdown(f'<div class="callout-box"><strong>{tr("mechanistic_scaling_label")}</strong> {format_chem_formula(anchor.electrolyte_adjustment_rationale)}</div>', unsafe_allow_html=True)

        doe_container_height = 540
        if is_demo:
            col_doe_plot, col_vial_table = st.columns([1.4, 1.6])
        else:
            col_doe_plot, col_plan_table = st.columns([1.25, 1.75])

        with col_doe_plot:
            with st.container(height=doe_container_height, border=False):
                st.markdown(f'<div class="section-title">{tr("sec_doe_space")}</div>', unsafe_allow_html=True)

                if campaign_res and campaign_res.cell_results:
                    pts = []
                    for c in campaign_res.cell_results:
                        rnd = getattr(c, "iteration_round", 0)
                        pts.append({
                            "Sample ID": c.sample_id,
                            f"{anchor.additive_name} Conc (mM)": c.additive_conc_mM,
                            f"{format_chem_formula(anchor.target_electrolyte_salt)} Salt Conc (M)": c.salt_conc_M,
                            "Optimization Round": f"Round {rnd} ({'Baseline DoE' if rnd == 0 else 'Adaptive DoE'})",
                            "Point Type": getattr(c, "point_type", "lhs"),
                            "Vial": c.vial_id,
                        })
                    pts_df = pd.DataFrame(pts)
                    fig_doe = px.scatter(
                        pts_df,
                        x=f"{anchor.additive_name} Conc (mM)",
                        y=f"{format_chem_formula(anchor.target_electrolyte_salt)} Salt Conc (M)",
                        color="Optimization Round",
                        color_discrete_map={
                            "Round 0 (Baseline DoE)": "#1f77b4",
                            "Round 1 (Adaptive DoE)": "#2ca02c",
                            "Round 2 (Adaptive DoE)": "#ff7f0e",
                        },
                        text="Sample ID",
                        hover_data=["Vial", f"{anchor.additive_name} Conc (mM)", f"{format_chem_formula(anchor.target_electrolyte_salt)} Salt Conc (M)", "Point Type"],
                        title=f"DoE Multi-Round Formulation Space: {anchor.additive_name} in {target_elyte_clean} ({len(pts_df)} Formulations Evaluated)",
                    )
                else:
                    pts_df = pd.DataFrame([
                        {"Sample ID": f.sample_id, f"{anchor.additive_name} Conc (mM)": f.additive_conc_mM, f"{format_chem_formula(anchor.target_electrolyte_salt)} Salt Conc (M)": f.salt_conc_M}
                        for f in plan.formulations
                    ])
                    fig_doe = px.scatter(
                        pts_df,
                        x=f"{anchor.additive_name} Conc (mM)",
                        y=f"{format_chem_formula(anchor.target_electrolyte_salt)} Salt Conc (M)",
                        text="Sample ID",
                        hover_data=[f"{anchor.additive_name} Conc (mM)", f"{format_chem_formula(anchor.target_electrolyte_salt)} Salt Conc (M)"],
                        title=f"{plan.method.upper()} Formulation Matrix: {anchor.additive_name} in {target_elyte_clean} ({len(pts_df)} Points)",
                    )

                # Add Anchor Point Marker
                fig_doe.add_trace(go.Scatter(
                    x=[anchor.shifted_anchor_conc],
                    y=[CANONICAL_ELECTROLYTE_PROFILES[selected_target_salt]["default_molarity"]],
                    mode="markers+text",
                    name="Reference Center",
                    text=["REFERENCE"],
                    textposition="top center",
                    marker=dict(symbol="star", size=16, color="#d62728", line=dict(width=1.5, color="black")),
                ))

                # Add Delaney ESOL Saturation Ceiling vertical line
                if anchor.solubility_ceiling_mM:
                    fig_doe.add_vline(
                        x=anchor.solubility_ceiling_mM,
                        line_width=2,
                        line_dash="dash",
                        line_color="#e63946",
                        annotation_text=f"ESOL 80% Safety Cap ({anchor.solubility_ceiling_mM:.1f} mM)",
                        annotation_position="top left",
                        annotation_font_color="#e63946",
                    )

                fig_doe.update_traces(textposition="bottom center")
                fig_doe.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20), height=465)
                st.plotly_chart(fig_doe, use_container_width=True)

        if is_demo:
            with col_vial_table:
                with st.container(height=doe_container_height, border=False):
                    st.markdown(f'<div class="section-title">{tr("sec_ot2_protocol")}</div>', unsafe_allow_html=True)
                    vial_records = []
                    for v in schedule.vials:
                        vial_records.append({
                            "Vial": v.vial_id,
                            "Row/Col": f"{v.row}{v.col}",
                            "Additive Stock (µL)": v.additive_stock_uL,
                            "Salt Stock (µL)": v.salt_stock_uL,
                            "Water Diluent (µL)": v.water_uL,
                            "Asp Rate (µL/s)": v.aspiration_rate_ul_s,
                            "Disp Rate (µL/s)": v.dispense_rate_ul_s,
                            "Warnings": len(v.warnings),
                        })
                    st.dataframe(pd.DataFrame(vial_records), hide_index=True, use_container_width=True, height=465)

            # Interactive Opentrons OT-2 24-vial rack / 96-well plate layout heatmap
            st.markdown(f'<div class="section-title">{tr("sec_deck_layout")}</div>', unsafe_allow_html=True)

            if schedule.rack_format == "24-vial rack":
                grid_rows = ["A", "B", "C", "D"]
                grid_cols = [1, 2, 3, 4, 5, 6]
            else:
                grid_rows = ["A", "B", "C", "D", "E", "F", "G", "H"]
                grid_cols = list(range(1, 13))

            # Map vials by coordinate
            vial_map = {(v.row, v.col): v for v in schedule.vials}

            # Associate sample IDs and iteration rounds with vials
            cell_by_vial = {}
            if campaign_res and campaign_res.cell_results:
                for c in campaign_res.cell_results:
                    cell_by_vial[c.vial_id] = c

            z_vals = []
            text_vals = []
            custom_vals = []

            for r in grid_rows:
                z_row = []
                t_row = []
                c_row = []
                for c in grid_cols:
                    if (r, c) in vial_map:
                        v = vial_map[(r, c)]
                        cell_item = cell_by_vial.get(v.vial_id)
                        s_id = cell_item.sample_id if cell_item else v.vial_id
                        r_num = getattr(cell_item, "iteration_round", getattr(schedule, "iteration_round", 0)) if cell_item else getattr(schedule, "iteration_round", 0)

                        z_row.append(v.additive_stock_uL)
                        t_row.append(f"<b>{v.vial_id}</b><br>R{r_num}<br><b>{v.additive_stock_uL:.0f} µL</b>")
                        c_row.append([
                            v.vial_id,
                            s_id,
                            f"Round {r_num}",
                            v.additive_conc_mM,
                            v.salt_conc_M,
                            v.additive_stock_uL,
                            v.salt_stock_uL,
                            v.water_uL,
                            v.aspiration_rate_ul_s,
                            v.dispense_rate_ul_s,
                            "; ".join(v.warnings) if v.warnings else "None",
                        ])
                    else:
                        z_row.append(0.0)
                        t_row.append("—")
                        c_row.append([
                            "Unassigned",
                            "-",
                            "Unassigned",
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            schedule.default_aspiration_rate_ul_s,
                            schedule.default_dispense_rate_ul_s,
                            "None",
                        ])
                z_vals.append(z_row)
                text_vals.append(t_row)
                custom_vals.append(c_row)

            fig_heatmap = go.Figure(data=go.Heatmap(
                z=z_vals,
                x=[f"Col {c}" for c in grid_cols],
                y=grid_rows,
                text=text_vals,
                texttemplate="%{text}",
                textfont={"size": 14, "family": "system-ui, -apple-system, sans-serif"},
                colorscale="Tealgrn",
                hoverongaps=False,
                customdata=custom_vals,
                hovertemplate=(
                    "<b>Vial %{customdata[0]} (%{y}%{x})</b><br>"
                    "Sample ID: %{customdata[1]} (%{customdata[2]})<br>"
                    "Additive Conc: %{customdata[3]:.1f} mM<br>"
                    "Salt Conc: %{customdata[4]:.2f} M<br>"
                    "Additive Stock: %{customdata[5]:.1f} µL<br>"
                    "Salt Stock: %{customdata[6]:.1f} µL<br>"
                    "Diluent Water: %{customdata[7]:.1f} µL<br>"
                    "Aspiration Rate: %{customdata[8]:.1f} µL/s<br>"
                    "Dispense Rate: %{customdata[9]:.1f} µL/s<br>"
                    "Warnings: %{customdata[10]}<extra></extra>"
                ),
                colorbar=dict(title="Additive Stock (µL)"),
            ))
            fig_heatmap.update_layout(
                title=dict(
                    text=f"<b>OT-2 Deck Coordinate Layout ({schedule.rack_format} — Slot {schedule.ot2_deck_slot}) — Multi-Round Continuity</b>",
                    font=dict(size=15),
                ),
                yaxis=dict(
                    autorange="reversed",
                    tickfont=dict(size=13),
                    title=dict(text="Row", font=dict(size=13)),
                ),
                xaxis=dict(
                    tickfont=dict(size=13),
                    title=dict(text="Column", font=dict(size=13)),
                ),
                height=390 if len(grid_rows) == 4 else 520,
                margin=dict(l=30, r=20, t=50, b=30),
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

            # Liquid Handling Automation Callouts & Viscosity Adaptation
            st.markdown(f'<div class="section-title">{tr("sec_liquid_handling")}</div>', unsafe_allow_html=True)
            c_rob1, c_rob2, c_rob3, c_rob4 = st.columns(4)
            with c_rob1:
                st.metric("OT-2 Deck Slot", f"Slot {schedule.ot2_deck_slot}")
            with c_rob2:
                st.metric("Aspiration Rate", f"{schedule.default_aspiration_rate_ul_s:.1f} µL/s")
            with c_rob3:
                st.metric("Dispense Rate", f"{schedule.default_dispense_rate_ul_s:.1f} µL/s")
            with c_rob4:
                is_visc = (schedule.default_aspiration_rate_ul_s < 40.0 or selected_target_salt in ("ZnCl2", "Zn(TFSI)2"))
                st.metric("Viscosity Throttle", "50% Slow Rate (25 µL/s)" if is_visc else "Standard (50 µL/s)")

            if is_visc:
                st.info(tr("viscosity_throttle_info"))

            if schedule.warnings:
                for w in schedule.warnings:
                    st.warning(f"⚠️ {w}")
            else:
                st.success(tr("pipetting_success"))
        else:
            with col_plan_table:
                with st.container(height=doe_container_height, border=False):
                    st.markdown(f'<div class="section-title">{tr("sec_doe_formulation_table")}</div>', unsafe_allow_html=True)
                    doe_records = []
                    for f in plan.formulations:
                        rec = {
                            tr("col_sample_id"): f.sample_id,
                            f"{anchor.additive_name} (mM)": f.additive_conc_mM,
                            f"{format_chem_formula(anchor.target_electrolyte_salt)} (M)": f.salt_conc_M,
                            tr("col_point_type"): f.point_type.upper(),
                        }
                        if f.temperature_C is not None:
                            rec[tr("col_temperature")] = f"{f.temperature_C:.1f} °C"
                        doe_records.append(rec)
                    calc_table_height = max(135, min(210, 38 + len(doe_records) * 35))
                    st.dataframe(pd.DataFrame(doe_records), hide_index=True, use_container_width=True, height=calc_table_height)

                    cur_lang = st.session_state.get("app_language", "en")

                    # 1. Anchor & Scaling Description
                    if anchor.scaling_factor != 1.0:
                        if cur_lang == "fr":
                            anchor_desc = f"Centré à <strong>{anchor.shifted_anchor_conc:.1f} mM</strong> dans {target_elyte_clean} (obtenu par mise à l'échelle de {anchor.published_concentration:.1f} mM dans {parent_elyte_clean} par un facteur <strong>{anchor.scaling_factor:.2f}×</strong> selon la permittivité diélectrique, la viscosité et le nombre donneur)."
                        elif cur_lang == "zh":
                            anchor_desc = f"配方中心锚定于 {target_elyte_clean} 中的 <strong>{anchor.shifted_anchor_conc:.1f} mM</strong>（基于介电常数、动力学粘度与给体数 DN 差异，由 {parent_elyte_clean} 中的文献基准 {anchor.published_concentration:.1f} mM 经 <strong>{anchor.scaling_factor:.2f}×</strong> 比例换算外推）。"
                        else:
                            anchor_desc = f"Centered at <strong>{anchor.shifted_anchor_conc:.1f} mM</strong> in {target_elyte_clean} (scaled from {anchor.published_concentration:.1f} mM in {parent_elyte_clean} by <strong>{anchor.scaling_factor:.2f}×</strong> based on dielectric constant, viscosity, and donor number ratios)."
                    else:
                        if cur_lang == "fr":
                            anchor_desc = f"Centré à <strong>{anchor.shifted_anchor_conc:.1f} mM</strong> dans {target_elyte_clean} (conforme à la référence {parent_elyte_clean} grâce aux énergies de solvatation équivalentes)."
                        elif cur_lang == "zh":
                            anchor_desc = f"配方中心锚定于 {target_elyte_clean} 中的 <strong>{anchor.shifted_anchor_conc:.1f} mM</strong>（与 {parent_elyte_clean} 文献基准一致，两体系溶剂化热力学能量相当）。"
                        else:
                            anchor_desc = f"Centered at <strong>{anchor.shifted_anchor_conc:.1f} mM</strong> in {target_elyte_clean} (matching literature baseline in {parent_elyte_clean} due to equivalent solvation energetics)."

                    # 2. Solubility Protection Description
                    ceil_val = anchor.solubility_ceiling_mM or 0.0
                    if cur_lang == "fr":
                        solubility_desc = f"Plage de concentration en additif : [<strong>{anchor.min_conc:.1f}</strong>, <strong>{anchor.max_conc:.1f} mM</strong>]"
                        if anchor.is_capped_by_solubility:
                            solubility_desc += f", strictement bridée par le <strong>plafond de sécurité Delaney ESOL 80% ({ceil_val:.1f} mM)</strong> afin d'éliminer la sursaturation locale, le relargage et la nucléation de dendrites."
                        else:
                            solubility_desc += f", restant sous le seuil théorique de saturation Delaney ESOL ({ceil_val:.1f} mM) pour garantir une parfaite dissolution."
                    elif cur_lang == "zh":
                        solubility_desc = f"添加剂浓度探索区间为 [<strong>{anchor.min_conc:.1f}</strong>, <strong>{anchor.max_conc:.1f} mM</strong>]"
                        if anchor.is_capped_by_solubility:
                            solubility_desc += f"，受 <strong>Delaney ESOL 80% 溶解度安全上限 ({ceil_val:.1f} mM)</strong> 刚性约束，彻底防止高离子强度下的局部过饱和、盐析效应与析出诱发枝晶。"
                        else:
                            solubility_desc += f"，远低于 Delaney ESOL 水相饱和溶解度阈值 ({ceil_val:.1f} mM)，确保完全分子级均相溶解。"
                    else:
                        solubility_desc = f"Additive concentrations range across [<strong>{anchor.min_conc:.1f}</strong>, <strong>{anchor.max_conc:.1f} mM</strong>]"
                        if anchor.is_capped_by_solubility:
                            solubility_desc += f", capped by the <strong>Delaney ESOL 80% safety cap ({ceil_val:.1f} mM)</strong> to prevent localized supersaturation, salting-out, and precipitate-nucleated dendrites."
                        else:
                            solubility_desc += f", safely below the Delaney ESOL aqueous saturation threshold ({ceil_val:.1f} mM) to ensure complete molecular dissolution."

                    # 3. Sampling Method Description
                    sampling_desc = tr("doe_rationale_lhs_desc") if plan.method.lower() == "lhs" else tr("doe_rationale_ccd_desc")

                    # 4. Salt Molarity Window Description
                    salt_min = plan.target_salt_molarity_range[0] if plan.target_salt_molarity_range else 1.0
                    salt_max = plan.target_salt_molarity_range[1] if plan.target_salt_molarity_range else 3.0
                    salt_base = plan.target_salt_default_molarity
                    if cur_lang == "fr":
                        salt_desc = f"La molarité du sel varie entre [<strong>{salt_min:.1f} M</strong> et <strong>{salt_max:.1f} M</strong>] autour du niveau de référence ({salt_base:.1f} M) pour évaluer l'équilibre entre réduction de l'activité de l'eau libre et cinétique de désolvatation de l'additif."
                    elif cur_lang == "zh":
                        salt_desc = f"主盐摩尔浓度在基准 ({salt_base:.1f} M) 周围的 [<strong>{salt_min:.1f} M</strong>, <strong>{salt_max:.1f} M</strong>] 窗口内变化，旨在评估自由水分子活度抑制程度与添加剂脱溶剂化动力学之间的协同竞争关系。"
                    else:
                        salt_desc = f"Salt molarity spans [<strong>{salt_min:.1f} M</strong>, <strong>{salt_max:.1f} M</strong>] around baseline ({salt_base:.1f} M) to evaluate the interplay between free water activity suppression and additive desolvation kinetics."

                    st.markdown(
                        f"""
                        <div style="margin-top: 8px; padding: 10px 13px; background: rgba(128, 128, 128, 0.05); border: 1px solid rgba(128, 128, 128, 0.18); border-radius: 8px; font-size: 0.84rem; line-height: 1.5; color: inherit;">
                            <div style="font-weight: 700; font-size: 0.88rem; margin-bottom: 6px; display: flex; align-items: center; justify-content: space-between;">
                                <span>💡 {tr("sec_doe_rationale_title")}</span>
                                <span class="badge-tag badge-blue" style="font-size: 0.70rem; font-weight: 600; text-transform: uppercase;">{plan.method.upper()} Design</span>
                            </div>
                            <div style="margin-bottom: 5px;">
                                <strong>🎯 {tr("doe_rationale_anchor_label")}:</strong> {anchor_desc}
                            </div>
                            <div style="margin-bottom: 5px;">
                                <strong>🛡️ {tr("doe_rationale_solubility_label")}:</strong> {solubility_desc}
                            </div>
                            <div style="margin-bottom: 5px;">
                                <strong>📊 {tr("doe_rationale_sampling_label")} ({plan.method.upper()}):</strong> {sampling_desc}
                            </div>
                            <div>
                                <strong>⚡ {tr("doe_rationale_salt_label")}:</strong> {salt_desc}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        # Download Buttons for Editable Formats (Excel, Word, Text) and Machine Data (JSON)
        with st.expander(f"📥 {tr('sec_download_protocol_plan')}", expanded=False):
            if is_demo:
                c_dl1, c_dl2 = st.columns(2)
                with c_dl1:
                    st.markdown(f"**{tr('sec_export_protocol')}**")
                    st.download_button(
                        tr("btn_dl_dispensing_xlsx"),
                        data=cached_generate_dispensing_schedule_xlsx(schedule),
                        file_name="Opentrons_OT2_Robotic_Dispensing_Schedule.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="step2_btn_dl_dispensing_xlsx",
                        help="Formatted Microsoft Excel workbook with well coordinates, liquid volumes, and robotic pipetting parameters.",
                    )
                    st.download_button(
                        tr("btn_dl_dispensing_docx"),
                        data=cached_generate_dispensing_schedule_docx(schedule),
                        file_name="Opentrons_OT2_Robotic_Dispensing_Schedule.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True,
                        key="step2_btn_dl_dispensing_docx",
                        help="Formal Microsoft Word protocol document editable for standard laboratory operation.",
                    )
                    st.download_button(
                        tr("btn_dl_dispensing_txt"),
                        data=cached_generate_dispensing_schedule_txt(schedule),
                        file_name="Opentrons_OT2_Robotic_Dispensing_Schedule.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key="step2_btn_dl_dispensing_txt",
                        help="Human-readable plain text protocol format editable in Notepad, Microsoft Word, or any text editor.",
                    )
                    st.download_button(
                        tr("btn_dl_dispensing"),
                        data=cached_dispensing_schedule_json(schedule),
                        file_name="Opentrons_OT2_Robotic_Dispensing_Schedule.json",
                        mime="application/json",
                        use_container_width=True,
                        key="step2_btn_dl_dispensing_json",
                        help="Standard JSON file for automated OT-2 robot execution.",
                    )
                with c_dl2:
                    st.markdown(f"**{tr('sec_export_formulation')}**")
                    st.download_button(
                        tr("btn_dl_doe_xlsx"),
                        data=cached_generate_doe_plan_xlsx(plan),
                        file_name="Electrolyte_Formulation_DoE_Plan.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="step2_btn_dl_doe_xlsx",
                        help="Formatted Microsoft Excel workbook with formulation matrix and concentration parameters.",
                    )
                    st.download_button(
                        tr("btn_dl_doe_docx"),
                        data=cached_generate_doe_plan_docx(plan),
                        file_name="Electrolyte_Formulation_DoE_Plan.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True,
                        key="step2_btn_dl_doe_docx",
                        help="Formal Microsoft Word experimental design document with Delaney ESOL limits and scaling rationale.",
                    )
                    st.download_button(
                        tr("btn_dl_doe_txt"),
                        data=cached_generate_doe_plan_txt(plan),
                        file_name="Electrolyte_Formulation_DoE_Plan.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key="step2_btn_dl_doe_txt",
                        help="Human-readable plain text formulation table editable in Notepad, Microsoft Word, or any text editor.",
                    )
                    st.download_button(
                        tr("btn_dl_doe"),
                        data=cached_doe_plan_json(plan),
                        file_name="Electrolyte_Formulation_DoE_Plan.json",
                        mime="application/json",
                        use_container_width=True,
                        key="step2_btn_dl_doe_json",
                        help="Standard JSON file for optimization algorithms and pipeline reproduction.",
                    )
            else:
                st.markdown(f"**{tr('sec_export_formulation')}**")
                c_dl_c1, c_dl_c2 = st.columns(2)
                with c_dl_c1:
                    st.download_button(
                        tr("btn_dl_doe_xlsx"),
                        data=cached_generate_doe_plan_xlsx(plan),
                        file_name="Electrolyte_Formulation_DoE_Plan.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="step2_btn_dl_doe_xlsx",
                        help="Formatted Microsoft Excel workbook with formulation matrix and concentration parameters.",
                    )
                    st.download_button(
                        tr("btn_dl_doe_docx"),
                        data=cached_generate_doe_plan_docx(plan),
                        file_name="Electrolyte_Formulation_DoE_Plan.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True,
                        key="step2_btn_dl_doe_docx",
                        help="Formal Microsoft Word experimental design document with Delaney ESOL limits and scaling rationale.",
                    )
                with c_dl_c2:
                    st.download_button(
                        tr("btn_dl_doe_txt"),
                        data=cached_generate_doe_plan_txt(plan),
                        file_name="Electrolyte_Formulation_DoE_Plan.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key="step2_btn_dl_doe_txt",
                        help="Human-readable plain text formulation table editable in Notepad, Microsoft Word, or any text editor.",
                    )
                    st.download_button(
                        tr("btn_dl_doe"),
                        data=cached_doe_plan_json(plan),
                        file_name="Electrolyte_Formulation_DoE_Plan.json",
                        mime="application/json",
                        use_container_width=True,
                        key="step2_btn_dl_doe_json",
                        help="Standard JSON file for optimization algorithms and pipeline reproduction.",
                    )


# =============================================================================
# TAB 4: STEP 3 ELECTROCHEMICAL ANALYSIS
# =============================================================================
with tab_step3:
    st.markdown(f"### **{tr('tab4_title' if is_demo else 'tab4_title_custom')}**")
    st.markdown(tr("tab4_desc"))

    # Style Step 3 technique selector and file uploader to align perfectly and have identical height
    st.markdown("""
    <style>
    /* Equalize label container heights and alignment */
    .st-key-uploader_step3_cycler_file label[data-testid="stWidgetLabel"],
    .st-key-step3_technique_selector label[data-testid="stWidgetLabel"],
    div[data-testid="stColumn"]:has(.st-key-uploader_step3_cycler_file) label[data-testid="stWidgetLabel"],
    div[data-testid="stColumn"]:has(.step3-technique-col) label[data-testid="stWidgetLabel"] {
        min-height: 28px !important;
        display: flex !important;
        align-items: center !important;
        margin-bottom: 4px !important;
    }

    /* Equalize heights and vertical thickness: Option A (compact 42px sleek layout) */
    .st-key-uploader_step3_cycler_file [data-testid="stFileUploaderDropzone"],
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"],
    div[data-testid="stColumn"]:has(.st-key-uploader_step3_cycler_file) [data-testid="stFileUploaderDropzone"],
    div[data-testid="stColumn"]:has(div[data-testid="stFileUploader"]) [data-testid="stFileUploaderDropzone"] {
        min-height: 42px !important;
        height: 42px !important;
        max-height: 42px !important;
        padding: 4px 12px !important;
        box-sizing: border-box !important;
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: space-between !important;
        gap: 8px !important;
        overflow: hidden !important;
    }

    /* Compact the Browse files button inside the dropzone */
    .st-key-uploader_step3_cycler_file [data-testid="stFileUploaderDropzone"] button,
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button {
        min-height: 32px !important;
        height: 32px !important;
        padding: 2px 10px !important;
        font-size: 0.85rem !important;
        line-height: 1.1 !important;
    }

    /* Compact the dropzone instructions text to single line */
    .st-key-uploader_step3_cycler_file [data-testid="stFileUploaderDropzoneInstructions"],
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        gap: 6px !important;
        font-size: 0.82rem !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    .st-key-step3_technique_selector div[data-baseweb="select"],
    .st-key-step3_technique_selector div[data-baseweb="select"] > div,
    div[data-testid="stColumn"]:has(.step3-technique-col) div[data-baseweb="select"],
    div[data-testid="stColumn"]:has(.step3-technique-col) div[data-baseweb="select"] > div {
        min-height: 42px !important;
        height: 42px !important;
        max-height: 42px !important;
        box-sizing: border-box !important;
        display: flex !important;
        align-items: center !important;
        border-radius: 8px !important;
        font-size: 0.98rem !important;
    }

    .st-key-step3_technique_selector div[data-baseweb="select"] > div > div,
    div[data-testid="stColumn"]:has(.step3-technique-col) div[data-baseweb="select"] > div > div {
        display: flex !important;
        align-items: center !important;
    }
    </style>
    """, unsafe_allow_html=True)

    col_u1, col_u2 = st.columns([2.2, 1.2], vertical_alignment="top")
    with col_u1:
        uploaded_cycler_file = st.file_uploader(
            tr("upload_cycler_label"),
            type=["mpr", "mpt", "csv", "xlsx", "xls", "dta", "txt", "dat"],
            help="Upload raw cycling files from BioLogic (.mpr, .mpt), Landt, Arbin, Gamry (.dta), or Neware.",
            key="uploader_step3_cycler_file",
        )
    with col_u2:
        technique_keys = [
            "auto",
            "galvanostatic_symmetric",
            "galvanostatic_asymmetric",
            "galvanostatic_full_cell",
            "cv",
            "lsv",
            "eis",
        ]
        technique_labels = {
            "auto": tr("technique_auto"),
            "galvanostatic_symmetric": tr("technique_symmetric"),
            "galvanostatic_asymmetric": tr("technique_asymmetric"),
            "galvanostatic_full_cell": tr("technique_full_cell"),
            "cv": tr("technique_cv"),
            "lsv": tr("technique_lsv"),
            "eis": tr("technique_eis"),
        }
        selected_technique = st.selectbox(
            tr("technique_selector_label"),
            options=technique_keys,
            format_func=lambda k: technique_labels.get(k, k),
            key="step3_technique_selector",
            help="Auto-detects technique from voltage/current profiles, or select an explicit technique to force specific analysis.",
        )
        st.markdown('<div class="step3-technique-col" style="display:none;"></div>', unsafe_allow_html=True)

    # Ingestion handler
    active_report: Optional[ElectrochemicalAnalysisReport] = None

    if uploaded_cycler_file is not None:
        file_bytes = uploaded_cycler_file.getvalue()
        file_suffix = Path(uploaded_cycler_file.name).suffix
        need_reparse = (
            st.session_state.get("uploaded_cycler_name") != uploaded_cycler_file.name
            or st.session_state.get("last_selected_technique") != selected_technique
            or st.session_state.get("custom_analysis_report") is None
        )
        if need_reparse:
            with tempfile.NamedTemporaryFile(suffix=file_suffix, delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                active_report = run_analysis(tmp_path, technique_str=selected_technique)
                st.session_state.custom_analysis_report = active_report
                st.session_state.uploaded_cycler_name = uploaded_cycler_file.name
                st.session_state.last_selected_technique = selected_technique
                st.success(f"Successfully parsed and analyzed {uploaded_cycler_file.name}!")
            except Exception as e:
                st.error(f"Error parsing uploaded file: {e}")
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        else:
            active_report = st.session_state.get("custom_analysis_report")

    # Report resolution
    if is_demo:
        if active_report is None:
            if st.session_state.get("custom_analysis_report"):
                active_report = st.session_state.custom_analysis_report
            elif st.session_state.get("benchmark_analysis_report"):
                active_report = st.session_state.benchmark_analysis_report
            else:
                with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    create_benchmark_cycler_file(tmp_path, duration_hours=100.0, short_at_hour=60.0, plateau_voltage_v=0.035)
                    active_report = run_analysis(tmp_path, technique_str="galvanostatic_symmetric")
                    st.session_state.benchmark_analysis_report = active_report
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
    else:
        # Custom Experiment Mode: only display if a custom report exists
        if active_report is None:
            active_report = st.session_state.get("custom_analysis_report")

    if active_report is None:
        st.info(tr("custom_mode_step3_empty_notice"))
    else:
        tech_str = active_report.technique.value if hasattr(active_report.technique, "value") else str(active_report.technique)
        tech_display_name = technique_labels.get(tech_str, tech_str.replace("_", " ").title())
        pts_count = len(active_report.processed_data.voltage_v) if active_report.processed_data else 0
        max_dur = max(active_report.processed_data.display_time, default=0.0) if active_report.processed_data else 0.0
        time_unit = active_report.processed_data.time_unit if active_report.processed_data else "h"

        st.markdown(f"""
        <div class="tech-file-banner">
            <span style="font-weight: 600; color: #0284c7;">📊 {tech_display_name}</span>
            <span class="banner-sep">|</span>
            <span class="banner-text">File: <code class="banner-code">{active_report.file_source}</code></span>
            <span class="banner-sep">|</span>
            <span class="banner-text">Data Points: <b class="banner-metric">{pts_count:,}</b></span>
            <span class="banner-sep">|</span>
            <span class="banner-text">Duration: <b class="banner-metric">{max_dur:.1f} {time_unit}</b></span>
        </div>
        """, unsafe_allow_html=True)

        if active_report.symmetric_cell:
            symm = active_report.symmetric_cell
            d = active_report.processed_data
            act_m = symm.active_cycling_metrics or {}
            tot_m = symm.total_test_summary or {}

            st.markdown(f'<div class="section-title">{tr("sec_two_stage_metrics")}</div>', unsafe_allow_html=True)
            col_m1, col_m2 = st.columns(2)

            with col_m1:
                st.markdown(f"""
                <div class="metric-card stage1-card" style="margin-bottom: 12px;">
                    <h4 class="stage1-title">{tr('stage1_header')}</h4>
                    <p class="stage-desc">{tr('stage1_desc')}</p>
                </div>
                """, unsafe_allow_html=True)
                t1_c1, t1_c2 = st.columns(2)
                with t1_c1:
                    st.metric("Stable Lifespan", f"{act_m.get('active_lifespan_hours', symm.total_lifespan_hours):.1f} h")
                    st.metric("Active Cycles", f"{act_m.get('active_cycles', symm.total_cycles)}")
                with t1_c2:
                    st.metric("Mean Overpotential (η)", f"{act_m.get('mean_overpotential_mv', symm.mean_overpotential_mv):.1f} mV")
                    st.metric("Mean Hysteresis (ΔV)", f"{act_m.get('mean_hysteresis_mv', symm.mean_hysteresis_mv):.1f} mV")

            with col_m2:
                st.markdown(f"""
                <div class="metric-card stage2-card" style="margin-bottom: 12px;">
                    <h4 class="stage2-title">{tr('stage2_header')}</h4>
                    <p class="stage-desc">{tr('stage2_desc')}</p>
                </div>
                """, unsafe_allow_html=True)
                t2_c1, t2_c2 = st.columns(2)
                with t2_c1:
                    st.metric("Total Test Duration", f"{tot_m.get('nominal_duration_hours', symm.total_lifespan_hours):.1f} h")
                    st.metric("Failure Mode", format_failure_mode(tot_m.get('failure_mode', symm.failure_mode)))
                with t2_c2:
                    fail_t = tot_m.get("failure_time_hours", symm.failure_time_hours)
                    st.metric("Time of Failure", f"{fail_t:.1f} h" if fail_t else "None")
                    st.metric("Short-Circuit Detected", f"{tot_m.get('short_circuit_detected', symm.short_circuit_detected)}")

            # Plotly Dual-Range Interactive Charts
            st.markdown(f'<div class="section-title">{tr("sec_cycling_profiles")}</div>', unsafe_allow_html=True)

            if is_demo:
                col_bench_act, col_bench_choice = st.columns([1, 2.2], vertical_alignment="center")
                with col_bench_act:
                    load_benchmark_btn = st.button(
                        tr("btn_load_benchmark"),
                        key="btn_load_benchmark_step3",
                        use_container_width=True,
                    )
                with col_bench_choice:
                    benchmark_choice = st.radio(
                        "Benchmark Scenario",
                        ["Stable Reversible Cycling (Benchmark)", "Dendrite Short-Circuit Failure at 60 h"],
                        index=1,
                        horizontal=True,
                        label_visibility="collapsed",
                        key="benchmark_scenario_step3",
                    )

                if load_benchmark_btn:
                    if "60 h" in benchmark_choice:
                        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                            tmp_path = tmp.name
                        try:
                            create_benchmark_cycler_file(tmp_path, duration_hours=100.0, short_at_hour=60.0, plateau_voltage_v=0.035)
                            new_rep = run_analysis(tmp_path, technique_str="galvanostatic_symmetric")
                            st.session_state.benchmark_analysis_report = new_rep
                            st.session_state.custom_analysis_report = None
                        finally:
                            Path(tmp_path).unlink(missing_ok=True)
                    else:
                        benchmark_csv = Path(__file__).resolve().parent / "samples" / "benchmark_zn_symmetric_cell.csv"
                        if benchmark_csv.exists():
                            new_rep = run_analysis(str(benchmark_csv), technique_str="galvanostatic_symmetric")
                            st.session_state.benchmark_analysis_report = new_rep
                            st.session_state.custom_analysis_report = None
                    st.rerun()

            view_mode_keys = ["stable", "full", "zoom"]
            view_mode_map = {
                "stable": tr("view_stable_cycles"),
                "full": tr("view_full_test"),
                "zoom": tr("view_zoom_24h"),
            }
            view_mode = st.radio(
                tr("view_window_label"),
                view_mode_keys,
                format_func=lambda k: view_mode_map.get(k, k),
                horizontal=True,
                index=0
            )

            t_fail = symm.failure_time_hours
            max_t = float(max(d.display_time)) if d.display_time else 100.0

            if view_mode == "stable" and t_fail:
                x_range = [0, t_fail * 1.05]
            elif view_mode == "zoom":
                x_range = [0, min(24.0, max_t)]
            else:
                x_range = [0, max_t]

            # Voltage Profile Figure
            fig_v = go.Figure()
            fig_v.add_trace(go.Scatter(
                x=d.display_time,
                y=d.voltage_v,
                mode="lines",
                name="Cell Potential (V)",
                line=dict(color="#1f77b4", width=1.5),
            ))

            if t_fail:
                fig_v.add_vline(
                    x=t_fail,
                    line_width=2,
                    line_dash="dash",
                    line_color="#d62728",
                    annotation_text=f"Failure: {t_fail:.1f} h ({format_failure_mode(symm.failure_mode)})",
                    annotation_position="top left",
                    annotation_font_color="#d62728",
                )

            fig_v.update_layout(
                title=f"{tr('chart_voltage_title')} ({active_report.file_source})",
                xaxis_title="Time (hours)",
                yaxis_title=tr("chart_voltage_y"),
                xaxis=dict(range=x_range, rangeslider=dict(visible=True)),
                template="plotly_white",
                height=380,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(
                fig_v,
                use_container_width=True,
                config={"toImageButtonOptions": {"format": "svg", "scale": 3, "filename": f"{active_report.sample_id}_symmetric_voltage_profile"}},
            )

            if symm.cycle_metrics:
                cyc_df = pd.DataFrame([m.model_dump() for m in symm.cycle_metrics])
                fig_cyc = go.Figure()
                fig_cyc.add_trace(go.Scatter(
                    x=cyc_df["cycle_number"],
                    y=cyc_df["overpotential_mv"],
                    mode="lines+markers",
                    name="Overpotential η (mV)",
                    line=dict(color="#d62728", width=2),
                ))
                fig_cyc.add_trace(go.Scatter(
                    x=cyc_df["cycle_number"],
                    y=cyc_df["voltage_hysteresis_mv"],
                    mode="lines+markers",
                    name="Hysteresis ΔV (mV)",
                    line=dict(color="#ff7f0e", width=1.5, dash="dash"),
                ))
                if symm.failure_cycle:
                    fig_cyc.add_vline(
                        x=symm.failure_cycle,
                        line_width=2,
                        line_dash="dash",
                        line_color="#d62728",
                        annotation_text=f"Failure Cycle {symm.failure_cycle}",
                        annotation_position="top left",
                    )
                fig_cyc.update_layout(
                    title=tr("chart_overpotential_title"),
                    xaxis_title="Cycle Number",
                    yaxis_title=tr("chart_overpotential_y"),
                    template="plotly_white",
                    height=320,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(
                    fig_cyc,
                    use_container_width=True,
                    config={"toImageButtonOptions": {"format": "svg", "scale": 3, "filename": f"{active_report.sample_id}_symmetric_cycle_metrics"}},
                )
            pass

        elif active_report.asymmetric_cell:
            asymm = active_report.asymmetric_cell
            d = active_report.processed_data

            st.markdown(f'<div class="section-title">{tr("sec_asymmetric_metrics")}</div>', unsafe_allow_html=True)
            col_as1, col_as2, col_as3, col_as4 = st.columns(4)
            with col_as1:
                avg_ce_str = f"{asymm.average_ce_percent:.1f} %" if asymm.average_ce_percent is not None else "N/A"
                st.metric(tr("asymm_avg_ce"), avg_ce_str)
            with col_as2:
                final_ce_str = f"{asymm.final_ce_percent:.1f} %" if asymm.final_ce_percent is not None else "N/A"
                st.metric(tr("asymm_final_ce"), final_ce_str)
            with col_as3:
                cum_cap_str = f"{asymm.cumulative_capacity_mah_cm2:.2f} mAh/cm²" if asymm.cumulative_capacity_mah_cm2 is not None else "N/A"
                st.metric(tr("asymm_cum_cap"), cum_cap_str)
            with col_as4:
                nuc_over_str = f"{asymm.mean_nucleation_overpotential_mv:.1f} mV" if asymm.mean_nucleation_overpotential_mv is not None else "N/A"
                st.metric(tr("asymm_nuc_overpot"), nuc_over_str)

            fig_v = go.Figure()
            fig_v.add_trace(go.Scatter(
                x=d.display_time,
                y=d.voltage_v,
                mode="lines",
                name="Cell Potential (V)",
                line=dict(color="#1f77b4", width=1.5),
            ))
            fig_v.update_layout(
                title=f"{tr('chart_voltage_title')} ({active_report.file_source})",
                xaxis_title=f"Time ({d.time_unit})",
                yaxis_title=tr("chart_voltage_y"),
                template="plotly_white",
                height=380,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(rangeslider=dict(visible=True)),
            )
            st.plotly_chart(
                fig_v,
                use_container_width=True,
                config={"toImageButtonOptions": {"format": "svg", "scale": 3, "filename": f"{active_report.sample_id}_asymmetric_voltage_profile"}},
            )

            if asymm.cycle_metrics:
                as_df = pd.DataFrame([m.model_dump() for m in asymm.cycle_metrics])
                fig_as = go.Figure()
                fig_as.add_trace(go.Scatter(
                    x=as_df["cycle_number"],
                    y=as_df["coulombic_efficiency_percent"],
                    mode="lines+markers",
                    name="Coulombic Efficiency (%)",
                    line=dict(color="#2ca02c", width=2),
                ))
                fig_as.update_layout(
                    title="Coulombic Efficiency vs Cycle Number",
                    xaxis_title="Cycle Number",
                    yaxis_title="Coulombic Efficiency (%)",
                    template="plotly_white",
                    height=320,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(
                    fig_as,
                    use_container_width=True,
                    config={"toImageButtonOptions": {"format": "svg", "scale": 3, "filename": f"{active_report.sample_id}_asymmetric_ce_profile"}},
                )

            pass

        elif active_report.full_cell:
            fc = active_report.full_cell
            d = active_report.processed_data

            st.markdown(f'<div class="section-title">{tr("sec_full_cell_metrics")}</div>', unsafe_allow_html=True)
            col_fc1, col_fc2, col_fc3, col_fc4, col_fc5 = st.columns(5)
            with col_fc1:
                ret_str = f"{fc.capacity_retention_percent:.1f} %" if fc.capacity_retention_percent is not None else "N/A"
                st.metric(tr("fc_retention"), ret_str)
            with col_fc2:
                init_str = f"{fc.initial_discharge_capacity_mah_g:.1f} mAh/g" if fc.initial_discharge_capacity_mah_g is not None else "N/A"
                st.metric(tr("fc_init_cap"), init_str)
            with col_fc3:
                fin_str = f"{fc.final_discharge_capacity_mah_g:.1f} mAh/g" if fc.final_discharge_capacity_mah_g is not None else "N/A"
                st.metric(tr("fc_final_cap"), fin_str)
            with col_fc4:
                ce_str = f"{fc.average_coulombic_efficiency_percent:.2f} %" if fc.average_coulombic_efficiency_percent is not None else "N/A"
                st.metric(tr("fc_avg_ce"), ce_str)
            with col_fc5:
                ee_str = f"{fc.average_energy_efficiency_percent:.1f} %" if fc.average_energy_efficiency_percent is not None else "N/A"
                st.metric(tr("fc_energy_eff"), ee_str)

            is_summary = d.metadata.get("is_cycle_summary", False)
            if not is_summary and len(d.voltage_v) > len(fc.cycle_numbers):
                fig_v = go.Figure()
                fig_v.add_trace(go.Scatter(
                    x=d.display_time,
                    y=d.voltage_v,
                    mode="lines",
                    name="Full Cell Voltage (V)",
                    line=dict(color="#1f77b4", width=1.5),
                ))
                fig_v.update_layout(
                    title=f"Full Cell Galvanostatic Voltage vs Time ({active_report.file_source})",
                    xaxis_title=f"Time ({d.time_unit})",
                    yaxis_title="Cell Voltage (V)",
                    template="plotly_white",
                    height=380,
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis=dict(rangeslider=dict(visible=True)),
                )
                st.plotly_chart(
                    fig_v,
                    use_container_width=True,
                    config={"toImageButtonOptions": {"format": "svg", "scale": 3, "filename": f"{active_report.sample_id}_full_cell_voltage_profile"}},
                )

            if fc.cycle_numbers and len(fc.cycle_numbers) >= 1:
                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(
                    x=fc.cycle_numbers,
                    y=fc.discharge_capacities_mah_g,
                    mode="lines+markers",
                    name=tr("fc_trace_capacity"),
                    marker=dict(symbol="circle", size=6, color="#1f77b4"),
                    line=dict(color="#1f77b4", width=2),
                    yaxis="y1",
                ))
                fig_fc.add_trace(go.Scatter(
                    x=fc.cycle_numbers,
                    y=fc.coulombic_efficiencies,
                    mode="lines+markers",
                    name=tr("fc_trace_ce"),
                    marker=dict(symbol="diamond", size=6, color="#ff7f0e"),
                    line=dict(color="#ff7f0e", width=1.5, dash="dot"),
                    yaxis="y2",
                ))

                min_ce = min(fc.coulombic_efficiencies) if fc.coulombic_efficiencies else 80.0
                max_ce = max(fc.coulombic_efficiencies) if fc.coulombic_efficiencies else 100.0
                y2_min = max(0.0, min(80.0, min_ce - 5.0))
                y2_max = min(150.0, max(105.0, max_ce + 5.0))

                fig_fc.update_layout(
                    title=f"{tr('fc_chart_title')} ({active_report.file_source})",
                    xaxis=dict(title=tr("fc_xaxis_cycle"), showgrid=True),
                    yaxis=dict(
                        title=tr("fc_yaxis_cap"),
                        title_font=dict(color="#1f77b4"),
                        tickfont=dict(color="#1f77b4"),
                        side="left",
                        showgrid=True,
                    ),
                    yaxis2=dict(
                        title=tr("fc_yaxis_ce"),
                        title_font=dict(color="#ff7f0e"),
                        tickfont=dict(color="#ff7f0e"),
                        overlaying="y",
                        side="right",
                        showgrid=False,
                        range=[y2_min, y2_max],
                    ),
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    template="plotly_white",
                    height=420,
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(
                    fig_fc,
                    use_container_width=True,
                    config={"toImageButtonOptions": {"format": "svg", "scale": 3, "filename": f"{active_report.sample_id}_full_cell_capacity_ce"}},
                )
            pass

        elif active_report.cv:
            cv = active_report.cv
            d = active_report.processed_data

            st.markdown(f'<div class="section-title">{tr("sec_cv_metrics")}</div>', unsafe_allow_html=True)
            col_cv1, col_cv2, col_cv3, col_cv4, col_cv5 = st.columns(5)
            with col_cv1:
                epa_str = f"{cv.anodic_peak_potential_epa_v:.3f} V" if cv.anodic_peak_potential_epa_v is not None else "N/A"
                st.metric(tr("cv_epa"), epa_str)
            with col_cv2:
                epc_str = f"{cv.cathodic_peak_potential_epc_v:.3f} V" if cv.cathodic_peak_potential_epc_v is not None else "N/A"
                st.metric(tr("cv_epc"), epc_str)
            with col_cv3:
                dep_str = f"{cv.peak_separation_delta_ep_mv:.1f} mV" if cv.peak_separation_delta_ep_mv is not None else "N/A"
                st.metric(tr("cv_delta_ep"), dep_str)
            with col_cv4:
                ratio_str = f"{cv.current_ratio_ipa_ipc:.2f}" if cv.current_ratio_ipa_ipc is not None else "N/A"
                st.metric(tr("cv_ratio"), ratio_str)
            with col_cv5:
                on_plate = f"{cv.plating_onset_potential_v:.3f} V" if cv.plating_onset_potential_v is not None else "N/A"
                st.metric(tr("cv_plating_onset"), on_plate)

            has_cyc = d.cycle_number is not None and len(d.cycle_number) == len(d.voltage_v)
            unique_cycs = sorted(list(set(d.cycle_number))) if has_cyc else []

            if len(unique_cycs) > 1:
                cyc_opts = ["all"] + [str(c) for c in unique_cycs[:20]]
                cyc_labels = {"all": tr("view_full_test"), **{str(c): f"Cycle {c}" for c in unique_cycs[:20]}}
                sel_cyc = st.selectbox(
                    "Cycle Filter",
                    options=cyc_opts,
                    format_func=lambda k: cyc_labels.get(k, k),
                    key="cv_cyc_filter",
                )
            else:
                sel_cyc = "all"

            fig_cv = go.Figure()
            if has_cyc and sel_cyc != "all":
                target_c = int(sel_cyc)
                mask = [c == target_c for c in d.cycle_number]
                sub_v = [v for v, m in zip(d.voltage_v, mask) if m]
                sub_i = [i for i, m in zip(d.current_ma, mask) if m]
                fig_cv.add_trace(go.Scatter(
                    x=sub_v,
                    y=sub_i,
                    mode="lines",
                    name=f"Cycle {target_c}",
                    line=dict(color="#1f77b4", width=2),
                ))
            elif has_cyc and len(unique_cycs) > 1:
                colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
                for c_idx, c_num in enumerate(unique_cycs[:10]):
                    mask = [c == c_num for c in d.cycle_number]
                    sub_v = [v for v, m in zip(d.voltage_v, mask) if m]
                    sub_i = [i for i, m in zip(d.current_ma, mask) if m]
                    col = colors[c_idx % len(colors)]
                    fig_cv.add_trace(go.Scatter(
                        x=sub_v,
                        y=sub_i,
                        mode="lines",
                        name=f"Cycle {c_num}",
                        line=dict(color=col, width=1.8),
                    ))
            else:
                fig_cv.add_trace(go.Scatter(
                    x=d.voltage_v,
                    y=d.current_ma,
                    mode="lines",
                    name=tr("cv_trace_name"),
                    line=dict(color="#1f77b4", width=2),
                ))

            if cv.anodic_peak_potential_epa_v is not None and cv.anodic_peak_current_ipa_ma is not None:
                fig_cv.add_trace(go.Scatter(
                    x=[cv.anodic_peak_potential_epa_v],
                    y=[cv.anodic_peak_current_ipa_ma],
                    mode="markers+text",
                    name=f"E_pa ({cv.anodic_peak_potential_epa_v:.3f} V)",
                    text=["E_pa"],
                    textposition="top center",
                    marker=dict(symbol="triangle-up", size=13, color="#d62728", line=dict(width=1, color="black")),
                ))

            if cv.cathodic_peak_potential_epc_v is not None and cv.cathodic_peak_current_ipc_ma is not None:
                fig_cv.add_trace(go.Scatter(
                    x=[cv.cathodic_peak_potential_epc_v],
                    y=[cv.cathodic_peak_current_ipc_ma],
                    mode="markers+text",
                    name=f"E_pc ({cv.cathodic_peak_potential_epc_v:.3f} V)",
                    text=["E_pc"],
                    textposition="bottom center",
                    marker=dict(symbol="triangle-down", size=13, color="#2ca02c", line=dict(width=1, color="black")),
                ))

            if cv.plating_onset_potential_v is not None:
                fig_cv.add_vline(
                    x=cv.plating_onset_potential_v,
                    line_dash="dot",
                    line_color="#7f7f7f",
                    annotation_text=f"Plating Onset: {cv.plating_onset_potential_v:.3f} V",
                    annotation_position="bottom left",
                )

            if cv.stripping_onset_potential_v is not None:
                fig_cv.add_vline(
                    x=cv.stripping_onset_potential_v,
                    line_dash="dot",
                    line_color="#ff7f0e",
                    annotation_text=f"Stripping Onset: {cv.stripping_onset_potential_v:.3f} V",
                    annotation_position="top right",
                )

            fig_cv.update_layout(
                title=f"{tr('chart_cv_title')} ({active_report.file_source})",
                xaxis_title=tr("chart_cv_x"),
                yaxis_title=tr("chart_cv_y"),
                template="plotly_white",
                height=420,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(
                fig_cv,
                use_container_width=True,
                config={"toImageButtonOptions": {"format": "svg", "scale": 3, "filename": f"{active_report.sample_id}_cyclic_voltammetry"}},
            )

        elif active_report.lsv:
            lsv = active_report.lsv
            d = active_report.processed_data

            st.markdown(f'<div class="section-title">{tr("sec_lsv_metrics")}</div>', unsafe_allow_html=True)
            col_lsv1, col_lsv2, col_lsv3, col_lsv4, col_lsv5 = st.columns(5)
            with col_lsv1:
                ecorr_str = f"{lsv.corrosion_potential_ecorr_v:.3f} V" if lsv.corrosion_potential_ecorr_v is not None else "N/A"
                st.metric(tr("lsv_ecorr"), ecorr_str)
            with col_lsv2:
                jcorr_str = f"{lsv.corrosion_current_jcorr_ma_cm2:.4f} mA/cm²" if lsv.corrosion_current_jcorr_ma_cm2 is not None else "N/A"
                st.metric(tr("lsv_jcorr"), jcorr_str)
            with col_lsv3:
                betac_str = f"{lsv.tafel_slope_cathodic_mv_per_dec:.1f} mV/dec" if lsv.tafel_slope_cathodic_mv_per_dec is not None else "N/A"
                st.metric(tr("lsv_beta_c"), betac_str)
            with col_lsv4:
                betaa_str = f"{lsv.tafel_slope_anodic_mv_per_dec:.1f} mV/dec" if lsv.tafel_slope_anodic_mv_per_dec is not None else "N/A"
                st.metric(tr("lsv_beta_a"), betaa_str)
            with col_lsv5:
                her_str = f"{lsv.her_onset_potential_v:.3f} V" if lsv.her_onset_potential_v is not None else "N/A"
                st.metric(tr("lsv_her_onset"), her_str)

            # Isolate active linear sweep from pre-sweep settling / OCP points
            v_arr = np.array(d.voltage_v, dtype=float)
            j_arr = np.array(d.current_density_ma_cm2, dtype=float)
            if len(v_arr) > 10:
                if v_arr[-1] >= v_arr[0]:
                    start_idx = int(np.argmin(v_arr[: min(50, len(v_arr))]))
                else:
                    start_idx = int(np.argmax(v_arr[: min(50, len(v_arr))]))
                v_clean = v_arr[start_idx:].tolist()
                j_clean = j_arr[start_idx:].tolist()
            else:
                v_clean = d.voltage_v
                j_clean = d.current_density_ma_cm2

            col_plot1, col_plot2 = st.columns(2)

            with col_plot1:
                fig_lsv = go.Figure()
                fig_lsv.add_trace(go.Scatter(
                    x=v_clean,
                    y=j_clean,
                    mode="lines",
                    name="j (mA/cm²)",
                    line=dict(color="#d62728", width=2),
                ))
                if lsv.corrosion_potential_ecorr_v is not None:
                    fig_lsv.add_vline(
                        x=lsv.corrosion_potential_ecorr_v,
                        line_dash="dash",
                        line_color="#1f77b4",
                        annotation_text=f"E_corr: {lsv.corrosion_potential_ecorr_v:.3f} V",
                        annotation_position="top left",
                    )
                if lsv.her_onset_potential_v is not None:
                    fig_lsv.add_vline(
                        x=lsv.her_onset_potential_v,
                        line_dash="dot",
                        line_color="#9467bd",
                        annotation_text=f"HER: {lsv.her_onset_potential_v:.3f} V",
                        annotation_position="bottom right",
                    )
                fig_lsv.update_layout(
                    title=tr("chart_lsv_title"),
                    xaxis_title=tr("chart_cv_x"),
                    yaxis_title=tr("chart_lsv_y"),
                    template="plotly_white",
                    height=380,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(
                    fig_lsv,
                    use_container_width=True,
                    config={"toImageButtonOptions": {"format": "svg", "scale": 3, "filename": f"{active_report.sample_id}_lsv_polarization"}},
                )

            with col_plot2:
                j_abs_arr = np.abs(np.array(j_clean, dtype=float))
                safe_j = np.where(j_abs_arr < 1e-8, 1e-8, j_abs_arr)
                log_j = np.log10(safe_j)

                fig_tafel = go.Figure()
                fig_tafel.add_trace(go.Scatter(
                    x=v_clean,
                    y=log_j.tolist(),
                    mode="lines",
                    name="log₁₀|j|",
                    line=dict(color="#9467bd", width=2),
                ))
                if lsv.corrosion_potential_ecorr_v is not None:
                    fig_tafel.add_vline(
                        x=lsv.corrosion_potential_ecorr_v,
                        line_dash="dash",
                        line_color="#1f77b4",
                        annotation_text=f"E_corr: {lsv.corrosion_potential_ecorr_v:.3f} V",
                        annotation_position="top left",
                    )
                fig_tafel.update_layout(
                    title=tr("chart_tafel_title"),
                    xaxis_title=tr("chart_cv_x"),
                    yaxis_title=tr("chart_tafel_y"),
                    template="plotly_white",
                    height=380,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(
                    fig_tafel,
                    use_container_width=True,
                    config={"toImageButtonOptions": {"format": "svg", "scale": 3, "filename": f"{active_report.sample_id}_tafel_slope"}},
                )

        elif active_report.eis:
            eis = active_report.eis
            d = active_report.processed_data

            st.markdown(f'<div class="section-title">{tr("sec_eis_metrics")}</div>', unsafe_allow_html=True)
            col_eis1, col_eis2, col_eis3, col_eis4 = st.columns(4)
            with col_eis1:
                rs_str = f"{eis.solution_resistance_rs_ohm:.2f} Ω" if eis.solution_resistance_rs_ohm is not None else "N/A"
                st.metric(tr("eis_rs"), rs_str)
            with col_eis2:
                rct_str = f"{eis.charge_transfer_resistance_rct_ohm:.2f} Ω" if eis.charge_transfer_resistance_rct_ohm is not None else "N/A"
                st.metric(tr("eis_rct"), rct_str)
            with col_eis3:
                f_str = f"{eis.characteristic_frequency_hz:.1f} Hz" if eis.characteristic_frequency_hz is not None else "N/A"
                st.metric(tr("eis_f_apex"), f_str)
            with col_eis4:
                ckt_str = eis.equivalent_circuit or "R_s + R_ct // CPE"
                st.metric(tr("eis_circuit"), ckt_str)

            zr = (d.metadata.get("z_real_ohm") if (hasattr(d, "metadata") and d.metadata) else None) or []
            zi = (d.metadata.get("z_imag_ohm") if (hasattr(d, "metadata") and d.metadata) else None) or []
            neg_zi = ([-z if np.mean(zi) < 0 else z for z in zi]) if len(zi) > 0 else []

            fig_nyquist = go.Figure()
            if zr and neg_zi:
                fig_nyquist.add_trace(go.Scatter(
                    x=zr,
                    y=neg_zi,
                    mode="lines+markers",
                    name="Nyquist (-Z'' vs Z')",
                    line=dict(color="#1f77b4", width=2),
                    marker=dict(size=7, color="#1f77b4"),
                ))

                if eis.solution_resistance_rs_ohm is not None:
                    fig_nyquist.add_annotation(
                        x=eis.solution_resistance_rs_ohm,
                        y=0,
                        text=f"R_s ({eis.solution_resistance_rs_ohm:.1f} Ω)",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="#2ca02c",
                        ax=0,
                        ay=-30,
                        font=dict(color="#2ca02c", size=11),
                    )

                if eis.charge_transfer_resistance_rct_ohm is not None and neg_zi:
                    apex_idx = int(np.argmax(neg_zi))
                    apex_x = zr[apex_idx] if apex_idx < len(zr) else 0.0
                    apex_y = neg_zi[apex_idx]
                    freq_text = f"f_apex: {eis.characteristic_frequency_hz:.1f} Hz" if eis.characteristic_frequency_hz else "Apex"
                    fig_nyquist.add_annotation(
                        x=apex_x,
                        y=apex_y,
                        text=freq_text,
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="#d62728",
                        ax=30,
                        ay=-25,
                        font=dict(color="#d62728", size=11),
                    )
            else:
                st.warning("Raw impedance spectrum data (z_real_ohm, z_imag_ohm) is missing from report metadata.")
                fig_nyquist.add_annotation(
                    text="⚠️ Raw impedance spectrum data (z_real_ohm / z_imag_ohm) not available",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(color="#d97706", size=14),
                )

            fig_nyquist.update_layout(
                title=f"{tr('chart_nyquist_title')} ({active_report.file_source})",
                xaxis_title=tr("chart_nyquist_x"),
                yaxis=dict(
                    title=tr("chart_nyquist_y"),
                    scaleanchor="x",
                    scaleratio=1,
                ),
                template="plotly_white",
                height=480,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(
                fig_nyquist,
                use_container_width=True,
                config={"toImageButtonOptions": {"format": "svg", "scale": 3, "filename": f"{active_report.sample_id}_eis_nyquist"}},
            )

        else:
            d = active_report.processed_data
            v_arr = np.array(d.voltage_v) if d.voltage_v else np.array([0.0])

            st.markdown(f'<div class="section-title">{tr("sec_general_cycling_metrics")}</div>', unsafe_allow_html=True)
            col_g1, col_g2, col_g3, col_g4 = st.columns(4)
            with col_g1:
                st.metric(tr("general_duration"), f"{max(d.display_time, default=0.0):.1f} {d.time_unit}")
            with col_g2:
                st.metric(tr("general_data_points"), f"{len(d.voltage_v):,}")
            with col_g3:
                st.metric(tr("general_max_v"), f"{float(np.max(v_arr)):.4f} V")
            with col_g4:
                st.metric(tr("general_min_v"), f"{float(np.min(v_arr)):.4f} V")

            fig_gen = go.Figure()
            fig_gen.add_trace(go.Scatter(
                x=d.display_time,
                y=d.voltage_v,
                mode="lines",
                name="Potential (V)",
                line=dict(color="#1f77b4", width=1.5),
            ))
            fig_gen.update_layout(
                title=f"Electrochemical Potential vs Time ({active_report.file_source})",
                xaxis_title=f"Time ({d.time_unit})",
                yaxis_title="Cell Potential vs Zn/Zn²⁺ (V)",
                template="plotly_white",
                height=380,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(rangeslider=dict(visible=True)),
            )
            st.plotly_chart(
                fig_gen,
                use_container_width=True,
                config={"toImageButtonOptions": {"format": "svg", "scale": 3, "filename": f"{active_report.sample_id}_general_potential"}},
            )

            if len(d.current_ma) > 0 and any(abs(x) > 0.001 for x in d.current_ma):
                fig_i = go.Figure()
                fig_i.add_trace(go.Scatter(
                    x=d.display_time,
                    y=d.current_ma,
                    mode="lines",
                    name="Current (mA)",
                    line=dict(color="#ff7f0e", width=1.5),
                ))
                fig_i.update_layout(
                    title=f"Applied Current vs Time ({active_report.file_source})",
                    xaxis_title=f"Time ({d.time_unit})",
                    yaxis_title="Current (mA)",
                    template="plotly_white",
                    height=280,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(
                    fig_i,
                    use_container_width=True,
                    config={"toImageButtonOptions": {"format": "svg", "scale": 3, "filename": f"{active_report.sample_id}_general_current"}},
                )

        # Clean Dataset Exporters (OriginLab / GraphPad Prism)
        st.markdown(f'<div class="section-title">{tr("sec_dataset_export")}</div>', unsafe_allow_html=True)
        col_dl_ds1, col_dl_ds2 = st.columns(2)
        with col_dl_ds1:
            st.download_button(
                tr("btn_dl_dataset_xlsx"),
                data=cached_generate_electrochemical_dataset_xlsx(active_report),
                file_name=f"{active_report.sample_id}_Clean_Dataset.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="step3_btn_dl_dataset_xlsx",
                help=tr("help_dl_dataset_xlsx"),
            )
        with col_dl_ds2:
            st.download_button(
                tr("btn_dl_dataset_csv"),
                data=cached_generate_electrochemical_dataset_csv(active_report),
                file_name=f"{active_report.sample_id}_Clean_Timeseries.csv",
                mime="text/csv",
                use_container_width=True,
                key="step3_btn_dl_dataset_csv",
                help=tr("help_dl_dataset_csv"),
            )


# =============================================================================
# TAB 5: EXPERIMENT SUMMARY & EXPORT
# =============================================================================
if tab_summary is not None:
    with tab_summary:
        st.markdown(f"### **{tr('tab5_title')}**")
        st.markdown(tr("tab5_desc"))

        if is_demo:
            campaign: Optional[CampaignResult] = st.session_state.campaign_result
        else:
            campaign: Optional[CampaignResult] = st.session_state.get("custom_campaign_result")

        if campaign:
            best = campaign.best_candidate
            best_report_html = cached_render_analysis_report_html(best.analysis_report) if (best and best.analysis_report) else None
            summary_col1, summary_col2 = st.columns(2)

            with summary_col1:
                st.markdown(f"#### **{tr('sec_campaign_metadata')}**")
                st.json({
                    "experiment_id": campaign.campaign_id,
                    "target_salt": format_chem_formula(campaign.target_salt),
                    "created_at": campaign.created_at,
                    "total_iterations": campaign.total_iterations,
                    "candidate_count": len(campaign.cell_results),
                    "optimal_balance_candidates_count": len(campaign.pareto_optimal_candidates),
                    "best_sample_id": best.sample_id if best else "N/A",
                    "best_performance_score": best.performance_score if best else 0.0,
                })

            with summary_col2:
                st.markdown(f"#### **{tr('sec_audit_trail')}**")
                st.markdown(f"""
                - **Reported Literature Baseline:** {campaign.formulation_plan.additive_name} in {format_chem_formula(campaign.formulation_plan.anchor.parent_salt)}
                - **Target Electrolyte Salt:** {format_chem_formula(campaign.target_salt)}
                - **Concentration Scaling Factor:** {campaign.formulation_plan.anchor.scaling_factor:.2f}x
                - **Optimization Rounds Executed:** {campaign.total_iterations} Round(s)
                - **Total Formulations Evaluated:** {len(campaign.cell_results)} Candidates
                - **Robotic Rack Format:** {campaign.dispensing_schedule.rack_format} ({len(campaign.dispensing_schedule.vials)} vials allocated)
                - **Deck Flow Rates:** {campaign.dispensing_schedule.default_aspiration_rate_ul_s} µL/s aspiration, {campaign.dispensing_schedule.default_dispense_rate_ul_s} µL/s dispense
                - **Delaney ESOL Capping:** {'Active (Enforced 80% Saturation Ceiling)' if campaign.formulation_plan.anchor.is_capped_by_solubility else 'Uncapped (High Solubility Margin)'}
                """)

            # Display Multi-Round Campaign History Table
            if campaign.history:
                st.markdown(f"#### **{tr('sec_seq_history')}**")
                hist_df = pd.DataFrame(campaign.history)
                st.dataframe(
                    hist_df,
                    column_config={
                        "iteration_round": st.column_config.NumberColumn("Iteration Round"),
                        "new_samples": st.column_config.NumberColumn("New Samples"),
                        "total_cells": st.column_config.NumberColumn("Total Cumulative Cells"),
                        "best_sample_id": st.column_config.TextColumn("Champion Sample"),
                        "best_score": st.column_config.NumberColumn("Champion Score", format="%.2f"),
                    },
                    hide_index=True,
                    use_container_width=True,
                )

            with st.expander(f"📥 {tr('sec_download_center')}", expanded=False):
                dl_c1, dl_c2, dl_c3, dl_c4 = st.columns(4)

                out_dir = Path(campaign.output_dir)

                with dl_c1:
                    st.markdown(f"**{tr('sec_export_campaign')}**")
                    st.download_button(
                        tr("btn_dl_campaign_xlsx"),
                        data=cached_generate_campaign_summary_xlsx(campaign),
                        file_name="Autonomous_Battery_Optimization_Laboratory_Experiment_Dossier.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="tab5_btn_dl_campaign_xlsx",
                        help=tr("help_dl_campaign_xlsx"),
                    )
                    st.download_button(
                        tr("btn_dl_campaign_docx"),
                        data=cached_generate_campaign_summary_docx(campaign),
                        file_name="Autonomous_Battery_Optimization_Laboratory_Experiment_Dossier.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True,
                        key="tab5_btn_dl_campaign_docx",
                        help=tr("help_dl_campaign_docx"),
                    )
                    st.download_button(
                        tr("btn_dl_campaign_txt"),
                        data=cached_generate_campaign_summary_txt(campaign),
                        file_name="Autonomous_Battery_Optimization_Laboratory_Experiment_Dossier.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key="tab5_btn_dl_campaign_txt",
                        help=tr("help_dl_campaign_txt"),
                    )
                    st.download_button(
                        tr("btn_dl_campaign_summary"),
                        data=cached_campaign_summary_json(campaign),
                        file_name="Autonomous_Battery_Optimization_Experiment_Summary.json",
                        mime="application/json",
                        use_container_width=True,
                        key="tab5_btn_dl_campaign_json",
                        help=tr("help_dl_campaign_json"),
                    )

                with dl_c2:
                    st.markdown(f"**{tr('sec_export_protocol')}**")
                    st.download_button(
                        tr("btn_dl_dispensing_xlsx"),
                        data=cached_generate_dispensing_schedule_xlsx(campaign.dispensing_schedule),
                        file_name="Opentrons_OT2_Robotic_Dispensing_Schedule.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="tab5_btn_dl_dispensing_xlsx",
                        help="Microsoft Excel spreadsheet of Opentrons OT-2 robotic dispensing schedule.",
                    )
                    st.download_button(
                        tr("btn_dl_dispensing_docx"),
                        data=cached_generate_dispensing_schedule_docx(campaign.dispensing_schedule),
                        file_name="Opentrons_OT2_Robotic_Dispensing_Schedule.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True,
                        key="tab5_btn_dl_dispensing_docx",
                        help="Microsoft Word lab protocol for Opentrons OT-2 robot execution.",
                    )
                    st.download_button(
                        tr("btn_dl_dispensing_txt"),
                        data=cached_generate_dispensing_schedule_txt(campaign.dispensing_schedule),
                        file_name="Opentrons_OT2_Robotic_Dispensing_Schedule.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key="tab5_btn_dl_dispensing_txt",
                        help="Plain text robotic dispensing schedule.",
                    )
                    st.download_button(
                        tr("btn_dl_dispensing"),
                        data=cached_dispensing_schedule_json(campaign.dispensing_schedule),
                        file_name="Opentrons_OT2_Robotic_Dispensing_Schedule.json",
                        mime="application/json",
                        use_container_width=True,
                        key="tab5_btn_dl_dispensing_json",
                        help="Standard JSON file for automated OT-2 robot execution.",
                    )
                    # Check for round 1 schedule
                    r1_sched = out_dir / "dispensing_schedule_round_1.json"
                    if r1_sched.exists():
                        st.download_button(
                            "💾 Round 1 Schedule (.json)",
                            data=r1_sched.read_text(encoding="utf-8"),
                            file_name="Opentrons_OT2_Robotic_Dispensing_Schedule_Round_1.json",
                            mime="application/json",
                            use_container_width=True,
                            key="tab5_btn_dl_dispensing_r1_json",
                        )

                with dl_c3:
                    st.markdown(f"**{tr('sec_export_formulation')}**")
                    st.download_button(
                        tr("btn_dl_doe_xlsx"),
                        data=cached_generate_doe_plan_xlsx(campaign.formulation_plan),
                        file_name="Electrolyte_Formulation_DoE_Plan.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="tab5_btn_dl_doe_xlsx",
                        help="Microsoft Excel spreadsheet of formulation DoE design points.",
                    )
                    st.download_button(
                        tr("btn_dl_doe_docx"),
                        data=cached_generate_doe_plan_docx(campaign.formulation_plan),
                        file_name="Electrolyte_Formulation_DoE_Plan.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True,
                        key="tab5_btn_dl_doe_docx",
                        help="Microsoft Word document of formulation DoE plan.",
                    )
                    st.download_button(
                        tr("btn_dl_doe_txt"),
                        data=cached_generate_doe_plan_txt(campaign.formulation_plan),
                        file_name="Electrolyte_Formulation_DoE_Plan.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key="tab5_btn_dl_doe_txt",
                        help="Plain text formulation DoE plan.",
                    )
                    st.download_button(
                        tr("btn_dl_doe"),
                        data=cached_doe_plan_json(campaign.formulation_plan),
                        file_name="Electrolyte_Formulation_DoE_Plan.json",
                        mime="application/json",
                        use_container_width=True,
                        key="tab5_btn_dl_doe_json",
                        help="Machine JSON data of formulation DoE plan.",
                    )
                    # Check for round 1 plan
                    r1_plan = out_dir / "doe_plan_round_1.json"
                    if r1_plan.exists():
                        st.download_button(
                            "💾 Round 1 Plan (.json)",
                            data=r1_plan.read_text(encoding="utf-8"),
                            file_name="Electrolyte_Formulation_DoE_Plan_Round_1.json",
                            mime="application/json",
                            use_container_width=True,
                            key="tab5_btn_dl_doe_r1_json",
                        )

                with dl_c4:
                    st.markdown(f"**{tr('sec_export_interactive')}**")
                    if best and best.analysis_report and best_report_html:
                        st.download_button(
                            tr("btn_dl_html_dashboard"),
                            data=best_report_html,
                            file_name="Zinc_Battery_Electrochemical_Analysis_Dashboard.html",
                            mime="text/html",
                            use_container_width=True,
                            key="tab5_btn_dl_html_dashboard",
                            help="Interactive Plotly HTML standalone dashboard viewable in any web browser.",
                        )
                        st.download_button(
                            tr("btn_dl_dataset_xlsx"),
                            data=cached_generate_electrochemical_dataset_xlsx(best.analysis_report),
                            file_name=f"{best.sample_id}_Champion_Clean_Dataset.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            key="tab5_btn_dl_dataset_xlsx",
                            help=tr("help_dl_dataset_xlsx"),
                        )
                        st.download_button(
                            tr("btn_dl_dataset_csv"),
                            data=cached_generate_electrochemical_dataset_csv(best.analysis_report),
                            file_name=f"{best.sample_id}_Champion_Clean_Timeseries.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="tab5_btn_dl_dataset_csv",
                            help=tr("help_dl_dataset_csv"),
                        )

            # Embedded standalone HTML preview
            st.markdown(f'<div class="section-title">{tr("sec_html_preview")}</div>', unsafe_allow_html=True)
            if best and best.analysis_report and best_report_html:
                st.components.v1.html(best_report_html, height=650, scrolling=True)
        else:
            if is_demo:
                st.info("Run an automated experiment in Tab 1 to generate executive reports and download artifacts.")
            else:
                st.info(tr("custom_mode_summary_empty_notice"))