"""
UI localization helpers and display maps for the ZIB MAP Streamlit Dashboard.
"""

from typing import Optional, Any, Tuple
import streamlit as st
from localization import get_text

SALT_DISPLAY_MAPS = {
    "en": {
        "ZnSO4": "ZnSO₄ — Zinc Sulfate (Standard Aqueous Baseline)",
        "Zn(OTf)2": "Zn(OTf)₂ — Zinc Triflate (Fluorinated Interphase)",
        "Zn(ClO4)2": "Zn(ClO₄)₂ — Zinc Perchlorate (Fast Desolvation)",
        "Zn(TFSI)2": "Zn(TFSI)₂ — Zinc TFSI (Water-in-Salt / High Voltage)",
        "ZnCl2": "ZnCl₂ — Zinc Chloride (High Concentration Hydrate)",
        "Zn(CH3COO)2": "Zn(CH₃COO)₂ — Zinc Acetate (Mild Buffer)",
    },
    "fr": {
        "ZnSO4": "ZnSO₄ — Sulfate de Zinc (Référence Aqueuse Standard)",
        "Zn(OTf)2": "Zn(OTf)₂ — Triflate de Zinc (Interphase Fluorée)",
        "Zn(ClO4)2": "Zn(ClO₄)₂ — Perchlorate de Zinc (Désolvatation Rapide)",
        "Zn(TFSI)2": "Zn(TFSI)₂ — TFSI de Zinc (Water-in-Salt / Haute Tension)",
        "ZnCl2": "ZnCl₂ — Chlorure de Zinc (Hydrate Haute Concentration)",
        "Zn(CH3COO)2": "Zn(CH₃COO)₂ — Acétate de Zinc (Tampon Doux)",
    },
    "zh": {
        "ZnSO4": "ZnSO₄ — 硫酸锌 (标准水系基准)",
        "Zn(OTf)2": "Zn(OTf)₂ — 三氟甲磺酸锌 (含氟界面相)",
        "Zn(ClO4)2": "Zn(ClO₄)₂ — 高氯酸锌 (快速去溶剂化)",
        "Zn(TFSI)2": "Zn(TFSI)₂ — 双三氟甲磺酰亚胺锌 (高浓度盐包水/高电压)",
        "ZnCl2": "ZnCl₂ — 氯化锌 (高浓度水合物)",
        "Zn(CH3COO)2": "Zn(CH₃COO)₂ — 乙酸锌 (温和缓冲体系)",
    },
}
SALT_DISPLAY_MAP = SALT_DISPLAY_MAPS["en"]

FG_DISPLAY_MAP = {
    "polyol": "Polyol",
    "amide_urea": "Amide / Urea",
    "thiocarbonyl": "Thiocarbonyl",
    "quaternary_ammonium": "Quaternary Ammonium",
    "sulfoxide": "Sulfoxide",
    "ether_polyether": "Ether / Polyether",
    "carboxylate": "Carboxylate",
    "general": "General Organic Motif",
}


def tr(key: str, **kwargs) -> str:
    """Retrieve localized text for the active UI language."""
    current_lang = st.session_state.get("app_language", "en")
    return get_text(key, lang=current_lang, **kwargs)


def format_failure_mode(mode: Optional[str]) -> str:
    """Format internal failure mode identifier to clean, human-readable scientific phrasing."""
    if not mode or mode in ("none", "None", ""):
        return tr("failure_none")
    key = f"failure_{mode}"
    localized = tr(key)
    if localized != key:
        return localized
    clean = str(mode).replace("_", " ").strip().title()
    if "Dendritic" in clean and "Circuit" not in clean:
        clean = "Dendritic Short Circuit"
    return clean


def format_fg_label(fg_key: str, lang: str = "en") -> str:
    """Format functional group key into a localized display label."""
    loc_key = f"fg_name_{fg_key}"
    translated = get_text(loc_key, lang=lang)
    if translated and translated != loc_key:
        return translated
    return FG_DISPLAY_MAP.get(fg_key, fg_key.replace("_", " ").title())


def get_candidate_provenance(cand: Any, lang: str = "en") -> Tuple[str, str, str, str]:
    """
    Determine candidate provenance origin:
    - User Uploaded ('upload')
    - Batch Ingested PDF ('batch_pdf')
    - Online Mined Paper ('online_mined')
    - Curated Library ('curated')
    Returns: (provenance_key, badge_label, badge_class, icon)
    """
    metric = (getattr(cand, "similarity_metric", "") or "").lower()
    rat = (getattr(cand, "rationale", "") or "").lower()

    if "user uploaded" in metric or "upload" in metric or "user uploaded" in rat or "uploaded additive" in rat:
        return "uploaded", get_text("provenance_uploaded", lang=lang), "badge-teal", "📤"
    elif "batch" in metric or "batch literature pdf" in rat or "batch pdf" in rat or "batch pdf ingestion" in metric:
        return "batch_pdf", get_text("provenance_batch_pdf", lang=lang), "badge-amber", "📑"
    elif "literature paper" in rat or "direct literature extraction" in metric or "online" in metric:
        return "online_mined", get_text("provenance_online_mined", lang=lang), "badge-purple", "🌐"
    else:
        return "curated", get_text("provenance_curated", lang=lang), "badge-blue", "📚"

