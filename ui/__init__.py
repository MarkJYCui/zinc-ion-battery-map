"""
UI Package for the ZIB Materials Acceleration Platform (MAP) Streamlit Dashboard.
"""

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

__all__ = [
    "CUSTOM_CSS",
    "inject_global_styles",
    "tr",
    "format_failure_mode",
    "SALT_DISPLAY_MAPS",
    "SALT_DISPLAY_MAP",
    "FG_DISPLAY_MAP",
    "format_fg_label",
    "get_candidate_provenance",
    "get_all_additives",
    "get_similarity_engine",
    "cached_generate_template_csv",
    "cached_generate_template_xlsx",
    "get_curated_molecules_df",
    "render_rdkit_mol_image",
    "compute_similarity_ranking",
    "_apply_card_candidate",
    "is_formulation_reminder_hidden",
    "_render_post_extraction_reminder_dialog",
    "_render_upload_popover",
    "_merge_candidates_into_session",
    "_render_candidate_chemical_dossier",
    "render_two_tier_card_deck",
]

