"""
Reusable Streamlit UI components, molecular rendering, upload/reminder dialogs,
and Two-Tier Candidate Card Deck for the ZIB MAP Dashboard.
"""

import re
import logging
from typing import Optional, List, Dict, Any

import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Lipinski
try:
    from rdkit.Chem import Draw
except ImportError:
    Draw = None


from models import CandidateMolecule
from map_orchestrator import sync_custom_formulation_pipeline
from discovery.similarity import (
    CURATED_ZIB_ADDITIVES,
    RDKitSimilarityEngine,
    calculate_delaney_esol,
    is_esol_applicable,
    detect_functional_groups,
    assess_electrolyte_compatibility,
    generate_chemoinformatic_rationale,
)
from discovery.search import clean_scientific_text, OFFLINE_SAMPLE_PAPERS
from discovery.extractor import PaperExtractor
from discovery.literature_engine import LiteratureDiscoveryEngine
from discovery.additive_importer import (
    parse_additive_file,
    deduplicate_and_resolve_additives,
    generate_template_csv,
    generate_template_xlsx,
)
from discovery.pubchem_client import is_valid_smiles, canonicalize_smiles
from discovery.formula_utils import (
    get_chemical_formula,
    format_chem_formula,
    normalize_condition_key,
    convert_concentration_to_mM,
    format_solubility_conc,
)
from ui.i18n import (
    tr,
    format_fg_label,
    get_candidate_provenance,
)

logger = logging.getLogger(__name__)


def get_all_additives() -> List[Dict[str, str]]:
    """Return all available additive candidates including dynamically extracted ones."""
    pool = list(CURATED_ZIB_ADDITIVES)
    custom = st.session_state.get("custom_additives", []) if hasattr(st, "session_state") else []
    for c in custom:
        if not any(a["name"].lower() == c["name"].lower() for a in pool):
            pool.append(c)
    return pool


@st.cache_resource
def get_similarity_engine() -> RDKitSimilarityEngine:
    """Return a cached singleton instance of RDKitSimilarityEngine with precomputed pool descriptors."""
    return RDKitSimilarityEngine()


@st.cache_data
def cached_generate_template_csv() -> bytes:
    return generate_template_csv()


@st.cache_data
def cached_generate_template_xlsx() -> bytes:
    return generate_template_xlsx()


@st.cache_data
def get_curated_molecules_df() -> pd.DataFrame:
    rows = []
    sim_engine = get_similarity_engine()
    for item in sim_engine.pool_mols:
        desc = item["descriptors"]
        fg = item["functional_groups"]
        esol = desc["logs"]
        esol_ok, _ = is_esol_applicable(item["mol"], esol)
        sat_mm = round((10 ** esol) * 1000.0, 1) if esol_ok else None
        ceiling_mm = round(0.80 * sat_mm, 1) if sat_mm is not None else None
        chem_formula = get_chemical_formula(item["smiles"], item["name"])
        rows.append({
            "Name": item["name"],
            "Chemical Formula": chem_formula,
            "ESOL LogS": esol if esol_ok else None,
            "Saturation (mM)": sat_mm,
            "80% Ceiling (mM)": ceiling_mm,
            "MW (g/mol)": desc["molecular_weight"] if desc else 0.0,
            "cLogP": desc["logp"] if desc else 0.0,
            "TPSA (Å²)": desc["tpsa"] if desc else 0.0,
            "HBD": desc["hbd"] if desc else 0,
            "HBA": desc["hba"] if desc else 0,
            "Rotatable Bonds": desc["rotatable_bonds"] if desc else 0,
            "Functional Groups": ", ".join(fg) if fg else "general",
        })
    return pd.DataFrame(rows)


@st.cache_data
def render_rdkit_mol_image(smiles: str, width: int = 420, height: int = 240):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Draw.MolToImage(mol, size=(width, height))
    except Exception:
        return None


@st.cache_data
def compute_similarity_ranking(target_smiles: str) -> pd.DataFrame:
    sim_engine = get_similarity_engine()
    target_mol = Chem.MolFromSmiles(target_smiles)
    if target_mol is None:
        return pd.DataFrame()
    
    target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2, nBits=2048)
    records = []
    for item in sim_engine.pool_mols:
        cand_fp = item["fp"]
        sim = round(float(AllChem.DataStructs.TanimotoSimilarity(target_fp, cand_fp)), 4)
        desc = item["descriptors"]
        esol = desc["logs"]
        esol_ok, _ = is_esol_applicable(item["mol"], esol)
        sat_mm = round((10 ** esol) * 1000.0, 1) if esol_ok else None
        ceiling_mm = round(0.80 * sat_mm, 1) if sat_mm is not None else None
        fg = item["functional_groups"]
        chem_formula = get_chemical_formula(item["smiles"], item["name"])
        
        records.append({
            "Candidate Molecule": item["name"],
            "Tanimoto Similarity": sim,
            "Chemical Formula": chem_formula,
            "Delaney ESOL (LogS)": esol if esol_ok else None,
            "Aqueous Saturation (mM)": sat_mm,
            "80% Ceiling (mM)": ceiling_mm,
            "MW (g/mol)": desc["molecular_weight"] if desc else 0.0,
            "cLogP": desc["logp"] if desc else 0.0,
            "Primary Functional Group": fg[0] if fg else "general",
        })
    df = pd.DataFrame(records)
    return df.sort_values(by="Tanimoto Similarity", ascending=False).reset_index(drop=True)


def _apply_card_candidate(
    cand: CandidateMolecule,
    selected_target_salt: str,
    doe_method: str = "lhs",
    num_doe_samples: int = 5,
    rack_format: str = "24-vial rack",
    total_vol_ul: float = 2000.0,
    published_conc_input: float = 15.0,
    parent_electrolyte_input: str = "2 M ZnSO4",
    show_toast: bool = True,
):
    """Synchronize an additive candidate with Step 2 formulation, DoE plan, and OT-2 schedule."""
    c_name = cand.name
    c_smiles = cand.smiles
    if not c_smiles:
        matching = next((a["smiles"] for a in CURATED_ZIB_ADDITIVES if a["name"].lower() == c_name.lower()), None)
        if not matching:
            clean_base = re.sub(r'\s*\([^)]*\)\s*$', '', c_name).strip().lower()
            matching = next((a["smiles"] for a in CURATED_ZIB_ADDITIVES if a["name"].lower() == clean_base), None)
        c_smiles = matching or ""

    custom_list = st.session_state.get("custom_additives", [])
    if not any(a["name"].lower() == c_name.lower() for a in CURATED_ZIB_ADDITIVES + custom_list):
        custom_list.append({"name": c_name, "smiles": c_smiles})
        st.session_state["custom_additives"] = custom_list

    st.session_state["tab2_mol_select"] = c_name
    st.session_state["_last_inspect_mol_select"] = c_name
    st.session_state["_last_inspect_mol_widget"] = c_name
    st.session_state["_last_sidebar_additive_widget"] = c_name
    st.session_state["selected_additive_name"] = c_name
    st.session_state["custom_additive_name"] = c_name
    st.session_state["custom_smiles"] = c_smiles
    st.session_state["tab2_custom_additive_name_input"] = c_name
    st.session_state["tab2_custom_smiles_input"] = c_smiles

    if cand.reported_concentration is not None and cand.reported_concentration > 0:
        conv_c, _ = convert_concentration_to_mM(
            cand.reported_concentration,
            cand.reported_concentration_unit,
            mw=cand.molecular_weight,
        )
        c_conc = float(conv_c) if conv_c is not None else float(cand.reported_concentration)
    else:
        c_conc = published_conc_input

    raw_parent = cand.parent_baseline_electrolyte or parent_electrolyte_input
    is_unspec_parent = (not raw_parent or "unspecified" in str(raw_parent).lower() or "not reported" in str(raw_parent).lower())
    c_salt = selected_target_salt if is_unspec_parent else format_chem_formula(raw_parent)
    st.session_state["is_active_baseline_unspecified"] = (
        cand.reported_concentration is None or is_unspec_parent
    )

    st.session_state["published_conc_input"] = float(c_conc)
    st.session_state["parent_electrolyte_input"] = c_salt

    new_anchor, _, _ = sync_custom_formulation_pipeline(
        additive_name=c_name,
        published_conc_mM=float(c_conc),
        parent_electrolyte=c_salt,
        target_electrolyte=selected_target_salt,
        smiles=c_smiles,
        doe_method=doe_method,
        num_samples=num_doe_samples,
        seed=42,
        rack_format=rack_format,
        total_volume_uL=total_vol_ul,
        session_state=st.session_state,
    )
    st.session_state["show_formulation_reminder_modal"] = False
    if show_toast:
        hard_ceil = new_anchor.solubility_ceiling_mM or 100.0
        st.session_state["last_extracted_notice"] = tr("suggested_additive_applied_toast", name=c_name, ceiling=hard_ceil)
    st.rerun()


def is_formulation_reminder_hidden() -> bool:
    """Return True if user previously opted out via 'Do not show this reminder again'."""
    if st.session_state.get("hide_formulation_reminder", False):
        return True
    try:
        if hasattr(st, "context") and hasattr(st.context, "cookies"):
            if st.context.cookies.get("map_hide_formulation_reminder") == "true":
                st.session_state["hide_formulation_reminder"] = True
                return True
    except Exception:
        pass
    return False


def _apply_modal_merge_choice(selected_mode: Optional[str] = None):
    """Apply the user's Append vs Replace selection from the post-extraction modal."""
    if not st.session_state.get("has_merge_conflict_prompt"):
        return
    mode = selected_mode or st.session_state.get("modal_extraction_merge_mode_radio")
    new_t1 = st.session_state.get("_pending_extracted_t1_cands", [])
    new_t2 = st.session_state.get("_pending_extracted_t2_cands", [])
    merged_t1 = st.session_state.get("_merged_extracted_t1_cands", [])
    merged_t2 = st.session_state.get("_merged_extracted_t2_cands", [])

    if mode == tr("upload_mode_replace"):
        st.session_state["mined_literature_additives"] = list(new_t1)
        st.session_state["ai_suggested_additives"] = list(new_t2)
        st.session_state["suggested_candidates"] = list(new_t2)
        if new_t1:
            first_c = new_t1[0]
            st.session_state["selected_tier1_name"] = first_c.name
            st.session_state["selected_additive_name"] = first_c.name
            st.session_state["pending_tier1_choice"] = 0
            st.session_state["pending_tier1_bottom_choice"] = 0
            st.session_state["last_tier1_choice"] = 0
            st.session_state["last_bottom_tier1_choice"] = 0
            st.session_state["step2_tier1_current_page"] = 1
            st.session_state["step2_tier2_current_page"] = 1
            st.session_state.pop("step2_tier1_additive_dropdown", None)
            st.session_state.pop("step2_tier1_bottom_dropdown", None)
    else:
        st.session_state["mined_literature_additives"] = list(merged_t1)
        st.session_state["ai_suggested_additives"] = list(merged_t2)
        st.session_state["suggested_candidates"] = list(merged_t2)


def _render_post_extraction_reminder_dialog(is_demo_mode: bool = True):
    """Render the post-extraction reminder modal using native @st.dialog."""
    has_conflict = bool(st.session_state.get("has_merge_conflict_prompt"))
    reminder_hidden = is_formulation_reminder_hidden()
    if reminder_hidden and not has_conflict:
        return

    def _on_modal_dismiss():
        _apply_modal_merge_choice()
        if st.session_state.get("chk_dont_show_formulation_reminder"):
            st.session_state["hide_formulation_reminder"] = True
            st.session_state["sync_hide_formulation_reminder_storage"] = True
        st.session_state["has_merge_conflict_prompt"] = False
        st.session_state["show_formulation_reminder_modal"] = False

    dialog_title = tr("modal_merge_conflict_title") if has_conflict else tr("modal_extraction_complete_title")

    @st.dialog(title=dialog_title, width="medium", on_dismiss=_on_modal_dismiss)
    def _dialog_content():
        # Immediate client-side check for localStorage preference:
        # Only auto-close when there is no merge conflict and the user previously hid the workflow tip.
        if not has_conflict:
            st.components.v1.html(
                """
                <script>
                (function() {
                    try {
                        if (window.parent.localStorage.getItem('map_hide_formulation_reminder') === 'true') {
                            const dlg = window.parent.document.querySelector('div[role="dialog"]');
                            if (dlg) {
                                const closeBtn = dlg.querySelector('button[aria-label="Close"]');
                                if (closeBtn) closeBtn.click();
                            }
                        }
                    } catch(e) {}
                })();
                </script>
                """,
                height=0,
                width=0,
            )

        dont_show = False
        if has_conflict:
            prior_t1 = st.session_state.get("_prior_existing_t1_cands", [])
            new_t1 = st.session_state.get("_pending_extracted_t1_cands", [])
            st.info(
                tr(
                    "modal_merge_conflict_notice",
                    existing_count=len(prior_t1),
                    new_count=len(new_t1),
                )
            )
            merge_choice = st.radio(
                tr("modal_merge_mode_label"),
                options=[tr("upload_mode_append"), tr("upload_mode_replace")],
                index=0,
                horizontal=True,
                key="modal_extraction_merge_mode_radio",
            )
            _apply_modal_merge_choice(merge_choice)
            st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)
        else:
            with st.container(key="modal_reminder_body_slot"):
                st.markdown(tr("modal_extraction_complete_body"))
                st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)
            dont_show = st.checkbox(
                tr("modal_dont_show_again"),
                key="chk_dont_show_formulation_reminder",
                help="Do not show this workflow tip again on this browser",
            )
            st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)

        confirm_btn_label = tr("modal_btn_confirm_merge") if has_conflict else tr("modal_btn_got_it")
        col_got_it, col_go_form = st.columns([1.0, 1.4], vertical_alignment="center")
        with col_got_it:
            if st.button(confirm_btn_label, key="btn_reminder_modal_got_it", use_container_width=True):
                _apply_modal_merge_choice()
                if dont_show:
                    st.session_state["hide_formulation_reminder"] = True
                    st.session_state["sync_hide_formulation_reminder_storage"] = True
                st.session_state["has_merge_conflict_prompt"] = False
                st.session_state["show_formulation_reminder_modal"] = False
                st.rerun()

        with col_go_form:
            if st.button(tr("modal_btn_go_formulation"), type="primary", key="btn_reminder_modal_go_formulation", use_container_width=True):
                _apply_modal_merge_choice()
                if dont_show:
                    st.session_state["hide_formulation_reminder"] = True
                    st.session_state["sync_hide_formulation_reminder_storage"] = True
                st.session_state["has_merge_conflict_prompt"] = False
                st.session_state["show_formulation_reminder_modal"] = False
                st.session_state["trigger_switch_to_formulation_tab"] = True
                st.rerun()

    _dialog_content()


def _render_upload_popover(key_suffix: str = "t1", use_container_width: bool = True):
    """Render the Standalone Custom Additive List Upload popover with multi-format parsing and template downloads."""
    with st.popover(
        f"📤 {tr('btn_upload_additives')}",
        use_container_width=use_container_width,
        key=f"upload_additives_popover_{key_suffix}",
        help="Import up to 5,000 additives from .csv, .xlsx, .xls, .txt, or .json with auto-deduplication"
    ):
        st.markdown(f'<div style="font-size: 0.96rem; font-weight: 700; margin-bottom: 2px;">📤 {tr("upload_additives_title")}</div>', unsafe_allow_html=True)
        st.caption(tr("upload_additives_desc"))

        st.info(f"⚡ {tr('upload_capacity_notice')}")

        st.markdown(
            f'<div style="font-size: 0.85rem; padding: 6px 10px; background: rgba(128, 128, 128, 0.08); border-radius: 6px; margin-bottom: 10px;">'
            f'{tr("upload_format_guidance")}'
            f'</div>',
            unsafe_allow_html=True
        )

        col_t_csv, col_t_xlsx = st.columns(2)
        with col_t_csv:
            st.download_button(
                f"📥 {tr('dl_csv_template')}",
                data=cached_generate_template_csv(),
                file_name="zib_additives_template.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"btn_dl_csv_template_{key_suffix}"
            )
        with col_t_xlsx:
            st.download_button(
                f"📥 {tr('dl_xlsx_template')}",
                data=cached_generate_template_xlsx(),
                file_name="zib_additives_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key=f"btn_dl_xlsx_template_{key_suffix}"
            )

        uploaded_file = st.file_uploader(
            tr("upload_file_prompt"),
            type=["csv", "xlsx", "xls", "txt", "json"],
            key=f"uploader_step2_additives_{key_suffix}"
        )

        up_mode_labels = {
            "append": tr("upload_mode_append"),
            "replace": tr("upload_mode_replace"),
        }
        up_mode = st.radio(
            "Import Mode",
            ["append", "replace"],
            index=0,
            format_func=lambda x: up_mode_labels.get(x, x),
            horizontal=True,
            label_visibility="collapsed",
            key=f"upload_mode_radio_{key_suffix}"
        )

        resolve_pubchem = st.checkbox(
            tr("resolve_pubchem_checkbox"),
            value=True,
            help=tr("resolve_pubchem_help"),
            key=f"chk_resolve_pubchem_upload_{key_suffix}"
        )

        if uploaded_file is not None:
            f_bytes = uploaded_file.getvalue()
            raw_recs, notices = parse_additive_file(f_bytes, uploaded_file.name)
            if notices:
                for n in notices:
                    st.caption(f"ℹ️ {n}")

            if raw_recs:
                st.success(f"✅ Found {len(raw_recs)} records in file.")
                if st.button(f"🚀 {tr('btn_import_additives')}", type="primary", use_container_width=True, key=f"btn_commit_upload_additives_{key_suffix}"):
                    progress_box = st.empty()
                    progress_bar = progress_box.progress(0.0, text=tr("import_progress_init"))

                    def _upload_progress_cb(cur: int, total: int, name: str):
                        pct = min(1.0, max(0.0, cur / max(total, 1)))
                        progress_bar.progress(pct, text=tr("import_progress_step", cur=cur, total=total, name=name))

                    with st.spinner(tr("import_progress_init")):
                        imported_cands, stats = deduplicate_and_resolve_additives(
                            raw_recs,
                            max_limit=5000,
                            resolve_pubchem=resolve_pubchem,
                            progress_callback=_upload_progress_cb
                        )
                    progress_box.empty()

                    if up_mode in ("replace", tr("upload_mode_replace")):
                        st.session_state["mined_literature_additives"] = imported_cands
                    else:
                        existing = st.session_state.get("mined_literature_additives", [])
                        ex_keys = {
                            normalize_condition_key(c.name, c.reported_concentration, c.parent_baseline_electrolyte): c
                            for c in existing
                        }
                        combined = list(existing)
                        for ic in imported_cands:
                            ckey = normalize_condition_key(ic.name, ic.reported_concentration, ic.parent_baseline_electrolyte)
                            if ckey in ex_keys:
                                ex_cand = ex_keys[ckey]
                                ex_cand.citation_count += getattr(ic, "citation_count", 1)
                                if "citation(s)" in ex_cand.rationale:
                                    ex_cand.rationale = re.sub(
                                        r'Reported in \d+ citation\(s\)',
                                        f'Reported in {ex_cand.citation_count} citation(s)',
                                        ex_cand.rationale
                                    )
                            else:
                                combined.append(ic)
                                ex_keys[ckey] = ic
                        st.session_state["mined_literature_additives"] = combined

                    new_t2 = []
                    try:
                        lit_eng = LiteratureDiscoveryEngine()
                        all_t1 = st.session_state["mined_literature_additives"]
                        _, new_t2 = lit_eng.generate_two_tier_candidates(all_t1[:15], use_pubchem=False)
                        if new_t2:
                            st.session_state["ai_suggested_additives"] = new_t2
                            st.session_state["suggested_candidates"] = new_t2
                    except Exception as e:
                        logger.debug(f"Tier 2 regeneration note: {e}")

                    if imported_cands:
                        first_c = imported_cands[0]
                        st.session_state["selected_tier1_name"] = first_c.name
                        if new_t2:
                            st.session_state["selected_tier2_name"] = new_t2[0].name
                        t1_idx = next(
                            (i for i, c in enumerate(st.session_state.get("mined_literature_additives", [])) if c.name.lower() == first_c.name.lower()),
                            0,
                        )
                        st.session_state["last_tier1_choice"] = t1_idx
                        st.session_state["last_bottom_tier1_choice"] = t1_idx
                        st.session_state["pending_tier1_choice"] = t1_idx
                        st.session_state["pending_tier1_bottom_choice"] = t1_idx
                        st.session_state["step2_tier1_current_page"] = (t1_idx // 6) + 1
                        st.session_state["last_suggested_choice"] = 0
                        st.session_state["last_bottom_tier2_choice"] = 0
                        st.session_state["pending_tier2_choice"] = 0
                        st.session_state["pending_tier2_bottom_choice"] = 0
                        st.session_state["step2_tier2_current_page"] = 1
                        _apply_card_candidate(
                            cand=first_c,
                            selected_target_salt=st.session_state.get("selected_target_salt", "Zn(OTf)2"),
                            doe_method=st.session_state.get("doe_method", "lhs"),
                            num_doe_samples=st.session_state.get("num_doe_samples", 5),
                            rack_format=st.session_state.get("rack_format", "24-vial rack"),
                            total_vol_ul=st.session_state.get("total_vol_ul", 2000.0),
                            published_conc_input=float(first_c.reported_concentration) if (first_c.reported_concentration is not None and first_c.reported_concentration > 0) else 15.0,
                            parent_electrolyte_input=first_c.parent_baseline_electrolyte if (first_c.parent_baseline_electrolyte and "unspecified" not in str(first_c.parent_baseline_electrolyte).lower() and "not reported" not in str(first_c.parent_baseline_electrolyte).lower()) else "2 M ZnSO4",
                        )
                    else:
                        st.session_state["last_extracted_notice"] = tr("toast_imported_success", count=0)
                        st.rerun()
            else:
                st.warning("No valid additive records found. Please ensure the file has an 'Additive Name' column or uses our template.")


def _merge_candidates_into_session(t1_new: List[Any], t2_new: List[Any], prompt_on_conflict: bool = True):
    """
    Merge discovered or extracted literature/AI candidates into session state.
    When distinct existing additives are already present in Step 2 and prompt_on_conflict is True,
    records both merged (Append) and isolated (Replace) candidate sets and prompts the user
    in the post-extraction modal whether to Append or Replace.
    """
    if st.session_state.get("has_merge_conflict_prompt") and st.session_state.get("_prior_existing_t1_cands"):
        existing_t1 = list(st.session_state.get("_prior_existing_t1_cands", []))
        existing_t2 = list(st.session_state.get("_prior_existing_t2_cands", []))
    else:
        existing_t1 = list(st.session_state.get("mined_literature_additives", []))
        existing_t2 = list(st.session_state.get("ai_suggested_additives", []))

    new_keys = {
        normalize_condition_key(tc.name, tc.reported_concentration, tc.parent_baseline_electrolyte)
        for tc in t1_new
    }
    ex_keys = {
        normalize_condition_key(c.name, c.reported_concentration, c.parent_baseline_electrolyte): c
        for c in existing_t1
    }
    has_distinct_prior = any(k not in new_keys for k in ex_keys)

    if existing_t1:
        merged_t1 = list(existing_t1)
        for tc in t1_new:
            ckey = normalize_condition_key(tc.name, tc.reported_concentration, tc.parent_baseline_electrolyte)
            if ckey not in ex_keys:
                merged_t1.insert(0, tc)
                ex_keys[ckey] = tc
            else:
                existing_item = ex_keys[ckey]
                metric_tc = getattr(tc, "similarity_metric", "") or ""
                if metric_tc in ("Direct Literature Extraction", "Batch PDF Ingestion"):
                    existing_item.similarity_metric = metric_tc
                    existing_item.rationale = tc.rationale
        st.session_state["mined_literature_additives"] = merged_t1
    else:
        merged_t1 = list(t1_new)
        st.session_state["mined_literature_additives"] = merged_t1

    if existing_t2:
        ex_t2_names = {c.name.lower() for c in existing_t2}
        merged_t2 = list(existing_t2)
        for tc in t2_new:
            if tc.name.lower() not in ex_t2_names:
                merged_t2.insert(0, tc)
                ex_t2_names.add(tc.name.lower())
        st.session_state["ai_suggested_additives"] = merged_t2
        st.session_state["suggested_candidates"] = merged_t2
    else:
        merged_t2 = list(t2_new)
        st.session_state["ai_suggested_additives"] = merged_t2
        st.session_state["suggested_candidates"] = merged_t2

    if has_distinct_prior and t1_new and prompt_on_conflict:
        st.session_state["_prior_existing_t1_cands"] = existing_t1
        st.session_state["_prior_existing_t2_cands"] = existing_t2
        st.session_state["_pending_extracted_t1_cands"] = list(t1_new)
        st.session_state["_pending_extracted_t2_cands"] = list(t2_new)
        st.session_state["_merged_extracted_t1_cands"] = list(merged_t1)
        st.session_state["_merged_extracted_t2_cands"] = list(merged_t2)
        st.session_state["has_merge_conflict_prompt"] = True
        st.session_state["show_formulation_reminder_modal"] = True
        st.session_state.pop("modal_extraction_merge_mode_radio", None)
    else:
        st.session_state["has_merge_conflict_prompt"] = False


def _render_candidate_chemical_dossier(
    cand_name: str,
    cand_smiles: str,
    is_tier2: bool = False,
    cand_obj: Optional[CandidateMolecule] = None,
    parent_electrolyte_input: str = "2 M ZnSO4",
    cur_lang: str = "en",
    key_prefix: str = "cand",
):
    """Render 2D Structure, compact RDKit descriptors, AI Rationale card, and electrolyte compatibility."""
    if not cand_name and not cand_smiles:
        st.info(tr("custom_mode_step1_empty_notice"))
        return

    cand_formula = get_chemical_formula(cand_smiles, cand_name) if cand_smiles else ""

    if not cand_smiles:
        mol_obj = None
        img = None
        fg_list = []
        esol_val = None
        theor_sat_mm = None
        hard_ceiling_mm = None
        esol_ok = False
        esol_reason = "Valid SMILES required"
        desc = None
    else:
        mol_obj = Chem.MolFromSmiles(cand_smiles)
        if mol_obj is None:
            st.warning(tr("invalid_smiles_warning"))
            img = None
            fg_list = []
            esol_val = None
            theor_sat_mm = None
            hard_ceiling_mm = None
            esol_ok = False
            esol_reason = "Valid SMILES required"
            desc = None
        else:
            img = render_rdkit_mol_image(cand_smiles, width=380, height=200)
            sim_engine = get_similarity_engine()
            desc = sim_engine.calculate_descriptors(cand_smiles)
            fg_list = detect_functional_groups(mol_obj)
            if cand_obj and cand_obj.logs is not None and cand_obj.logs > -10:
                raw_esol = cand_obj.logs
            elif desc and "logs" in desc:
                raw_esol = desc["logs"]
            else:
                raw_esol = calculate_delaney_esol(mol_obj)
            esol_ok, esol_reason = is_esol_applicable(mol_obj, raw_esol)
            if esol_ok:
                esol_val = raw_esol
                theor_sat_mm = round((10 ** esol_val) * 1000.0, 1)
                hard_ceiling_mm = round(0.80 * theor_sat_mm, 1)
            else:
                esol_val = None
                theor_sat_mm = None
                hard_ceiling_mm = None

    dossier_container_height = 880
    col_mol1, col_mol2 = st.columns([1.25, 1.75])
    with col_mol1:
        with st.container(height=dossier_container_height, border=False):
            st.markdown(f'<div class="section-title">{tr("sec_structure_descriptors")}</div>', unsafe_allow_html=True)
            if img:
                st.image(img, caption=f"{cand_name} — Chemical Formula: {format_chem_formula(cand_formula)}", use_container_width=True)
            elif not cand_smiles:
                st.info(f"🧪 **{cand_name}** — {tr('no_structure_provided_notice')}")

            if fg_list:
                fg_badges = "".join([
                    f'<span class="badge-tag badge-teal" style="font-size: 0.90rem; font-weight: 600; padding: 4px 10px; margin-right: 6px; margin-bottom: 4px; display: inline-block;">{format_fg_label(fg, cur_lang)}</span>'
                    for fg in fg_list
                ])
                st.markdown(
                    f"""
                    <div style="margin: 6px 0 10px 0; padding: 8px 12px; background: rgba(128, 128, 128, 0.05); border: 1px solid rgba(128, 128, 128, 0.15); border-radius: 6px; font-size: 0.92rem; line-height: 1.5; color: inherit;">
                        <div style="font-weight: 700; margin-bottom: 4px; font-size: 0.90rem; text-transform: uppercase; letter-spacing: 0.3px;">{tr("functional_groups_label")}</div>
                        <div>{fg_badges}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            if desc:
                st.markdown(
                    f"""
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-bottom: 8px;">
                        <div style="background: rgba(128, 128, 128, 0.06); border: 1px solid rgba(128, 128, 128, 0.15); border-radius: 5px; padding: 4px 8px;">
                            <div style="font-size: 0.68rem; font-weight: 600; text-transform: uppercase; color: inherit;">{tr("metric_mw_label")}</div>
                            <div style="font-size: 0.85rem; font-weight: 700; color: inherit;">{desc['molecular_weight']:.2f} <span style="font-size: 0.70rem; font-weight: normal;">g/mol</span></div>
                        </div>
                        <div style="background: rgba(128, 128, 128, 0.06); border: 1px solid rgba(128, 128, 128, 0.15); border-radius: 5px; padding: 4px 8px;">
                            <div style="font-size: 0.68rem; font-weight: 600; text-transform: uppercase; color: inherit;">{tr("metric_clogp_label")}</div>
                            <div style="font-size: 0.85rem; font-weight: 700; color: inherit;">{desc['logp']:.2f}</div>
                        </div>
                        <div style="background: rgba(128, 128, 128, 0.06); border: 1px solid rgba(128, 128, 128, 0.15); border-radius: 5px; padding: 4px 8px;">
                            <div style="font-size: 0.68rem; font-weight: 600; text-transform: uppercase; color: inherit;">{tr("metric_rot_bonds_label")}</div>
                            <div style="font-size: 0.85rem; font-weight: 700; color: inherit;">{desc['rotatable_bonds']}</div>
                        </div>
                        <div style="background: rgba(128, 128, 128, 0.06); border: 1px solid rgba(128, 128, 128, 0.15); border-radius: 5px; padding: 4px 8px;">
                            <div style="font-size: 0.68rem; font-weight: 600; text-transform: uppercase; color: inherit;">{tr("metric_tpsa_label")}</div>
                            <div style="font-size: 0.85rem; font-weight: 700; color: inherit;">{desc['tpsa']:.1f} <span style="font-size: 0.70rem; font-weight: normal;">Å²</span></div>
                        </div>
                        <div style="background: rgba(128, 128, 128, 0.06); border: 1px solid rgba(128, 128, 128, 0.15); border-radius: 5px; padding: 4px 8px;">
                            <div style="font-size: 0.68rem; font-weight: 600; text-transform: uppercase; color: inherit;">{tr("metric_hbd_label")}</div>
                            <div style="font-size: 0.85rem; font-weight: 700; color: inherit;">{desc.get('hbd', 0)}</div>
                        </div>
                        <div style="background: rgba(128, 128, 128, 0.06); border: 1px solid rgba(128, 128, 128, 0.15); border-radius: 5px; padding: 4px 8px;">
                            <div style="font-size: 0.68rem; font-weight: 600; text-transform: uppercase; color: inherit;">{tr("metric_hba_label")}</div>
                            <div style="font-size: 0.85rem; font-weight: 700; color: inherit;">{desc.get('hba', 0)}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Dedicated AI Suggestion Rationale Card (Tier 2 AI Analogues only)
            is_cand_lit = getattr(cand_obj, "is_literature_additive", False) if cand_obj else False
            custom_input_name = st.session_state.get("tab2_custom_additive_name_input", "").strip().lower()

            t2_pool = st.session_state.get("ai_suggested_additives", []) or st.session_state.get("suggested_candidates", [])
            t2_names = [c.name.lower() for c in t2_pool if not getattr(c, "is_literature_additive", False)]

            cand_low = cand_name.lower() if cand_name else ""
            if key_prefix == "tier2":
                t1_pool = st.session_state.get("mined_literature_additives", [])
                t1_names = {c.name.lower() for c in t1_pool}
                if custom_input_name and (custom_input_name in t1_names or (t1_pool and custom_input_name in {a["name"].lower() for a in CURATED_ZIB_ADDITIVES})):
                    # When custom_input_name holds a Tier 1 literature additive from the other tab,
                    # the Tier 2 tab autonomously displays its selected AI analogue's rationale
                    is_active_t2 = (cand_low in t2_names)
                elif custom_input_name and not t1_names and (custom_input_name not in t2_names):
                    # Standalone custom input test scenario where no literature candidates are loaded
                    # and user typed a non-Tier-2 additive name into custom input
                    is_active_t2 = False
                else:
                    is_active_t2 = (cand_low in t2_names)
            else:
                active_name = custom_input_name or cand_low
                is_active_t2 = (active_name in t2_names) if active_name else (is_tier2 and not is_cand_lit)

            if is_tier2 and cand_name and not is_cand_lit and is_active_t2:
                parent_name = getattr(cand_obj, "parent_scaffold_name", None) if cand_obj else None
                if not parent_name:
                    t1_pool = st.session_state.get("mined_literature_additives", [])
                    if t1_pool:
                        sim_engine = get_similarity_engine()
                        best_t1 = max(t1_pool, key=lambda t1: sim_engine.compute_tanimoto(t1.smiles, cand_smiles) if (t1.smiles and cand_smiles) else 0.0)
                        parent_name = best_t1.name
                    else:
                        parent_name = "Literature Scaffold"

                sim_score = getattr(cand_obj, "similarity_score", 0.85) if cand_obj else 0.85
                sim_score = sim_score or 0.85
                rationale_dict = generate_chemoinformatic_rationale(
                    cand_name=cand_name,
                    cand_mol=mol_obj,
                    parent_name=parent_name,
                    similarity_score=sim_score,
                    logs=esol_val,
                    lang=cur_lang,
                )

                st.markdown(
                    f"""
                    <div style="background: rgba(155, 89, 182, 0.06); border: 1px solid rgba(155, 89, 182, 0.28); border-left: 4px solid #9b59b6; border-radius: 7px; padding: 16px 18px; margin-top: 10px; margin-bottom: 8px;">
                        <div style="font-size: 1.05rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.3px; color: #8e44ad; margin-bottom: 12px; display: flex; align-items: center; gap: 6px;">
                            <span>💡</span> <span>{tr("ai_suggestion_rationale_title")}</span>
                        </div>
                        <div style="background: rgba(155, 89, 182, 0.04); border: 1px solid rgba(155, 89, 182, 0.15); border-radius: 6px; padding: 12px 16px; margin-bottom: 11px; font-size: 0.96rem; line-height: 1.60; color: inherit;">
                            • <strong>{tr("rationale_parent_anchor_label")}:</strong> {rationale_dict["parent_anchor"]}
                        </div>
                        <div style="background: rgba(155, 89, 182, 0.04); border: 1px solid rgba(155, 89, 182, 0.15); border-radius: 6px; padding: 12px 16px; margin-bottom: 11px; font-size: 0.96rem; line-height: 1.60; color: inherit;">
                            • <strong>{tr("rationale_functional_motifs_label")}:</strong> {rationale_dict["functional_motifs"]}
                        </div>
                        <div style="background: rgba(155, 89, 182, 0.04); border: 1px solid rgba(155, 89, 182, 0.15); border-radius: 6px; padding: 12px 16px; margin-bottom: 0px; font-size: 0.96rem; line-height: 1.60; color: inherit;">
                            • <strong>{tr("rationale_physical_hypothesis_label")}:</strong> {rationale_dict["physical_hypothesis"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with col_mol2:
        with st.container(height=dossier_container_height, border=False):
            st.markdown(f'<div class="section-title">{tr("multi_electrolyte_comp")}</div>', unsafe_allow_html=True)
            base_salt_comp = parent_electrolyte_input
            note, comp_dict = assess_electrolyte_compatibility(mol_obj, baseline_electrolyte=base_salt_comp)

            st.caption(tr("compat_table_tip"))
            comp_rows = []
            for salt, detail in comp_dict.items():
                comp_rows.append({
                    tr("salt_system_col"): format_chem_formula(salt),
                    tr("compat_status_col"): detail.status,
                    tr("exp_eff_col"): detail.expected_effectiveness,
                    tr("mech_rationale_col"): format_chem_formula(detail.notes),
                })
            st.dataframe(
                pd.DataFrame(comp_rows),
                column_config={
                    tr("salt_system_col"): st.column_config.TextColumn(tr("salt_system_col"), width="small"),
                    tr("compat_status_col"): st.column_config.TextColumn(tr("compat_status_col"), width="medium"),
                    tr("exp_eff_col"): st.column_config.TextColumn(tr("exp_eff_col"), width="small"),
                    tr("mech_rationale_col"): st.column_config.TextColumn(tr("mech_rationale_col"), width="large", help="Double-click to expand cell"),
                },
                hide_index=True,
                use_container_width=True,
                key=f"df_compat_{key_prefix}",
            )

            with st.expander(f"🔬 {tr('detailed_rationale_expander')}", expanded=False):
                for salt, detail in comp_dict.items():
                    salt_fmt = format_chem_formula(salt)
                    status_badge = "badge-teal" if detail.status in ("Validated", "Highly Compatible") else ("badge-amber" if "Caution" in detail.status else "badge-blue")
                    st.markdown(
                        f"""
                        <div style="margin-bottom: 8px; padding: 10px 14px; border-radius: 7px; border: 1px solid rgba(128, 128, 128, 0.2); background: rgba(128, 128, 128, 0.06); color: inherit;">
                            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                                <strong style="font-size: 0.95rem; margin-right: 8px; color: inherit;">{salt_fmt}</strong>
                                <span class="badge-tag {status_badge}">{detail.status}</span>
                                <span class="badge-tag badge-blue" style="margin-left: 4px;">{detail.expected_effectiveness}</span>
                            </div>
                            <div style="font-size: 0.90rem; line-height: 1.50; color: inherit;">
                                {format_chem_formula(detail.notes)}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    # Dedicated Delaney ESOL Physical Saturation Guardrail Card
    esol_body_str = tr("esol_callout_body", ceiling=hard_ceiling_mm if hard_ceiling_mm is not None else 0.0)
    if hard_ceiling_mm is None:
        esol_body_str = esol_body_str.replace("(0.0 mM)", "(N/A)")
    raw_esol_text = f"{tr('esol_callout_title')} {esol_body_str}"
    clean_esol_tooltip = re.sub(r'</?(?:strong|em|b|i)\b[^>]*>', '', raw_esol_text, flags=re.IGNORECASE)
    clean_esol_tooltip = re.sub(r'(?:<br\s*/?>\s*){2,}', ' __ESOL_PARA__ ', clean_esol_tooltip, flags=re.IGNORECASE)
    clean_esol_tooltip = re.sub(r'<br\s*/?>', ' __ESOL_BR__ ', clean_esol_tooltip, flags=re.IGNORECASE)
    clean_esol_tooltip = re.sub(r'<[^>]+>', '', clean_esol_tooltip)
    clean_esol_tooltip = re.sub(r'\s+', ' ', clean_esol_tooltip).strip()
    clean_esol_tooltip = re.sub(r'\s+([.,;:!?])', r'\1', clean_esol_tooltip)
    clean_esol_tooltip = (
        clean_esol_tooltip.replace('&', '&amp;')
        .replace('"', '&quot;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace(' __ESOL_PARA__ ', '&#10;&#10;')
        .replace('__ESOL_PARA__', '&#10;&#10;')
        .replace(' __ESOL_BR__ ', '&#10;')
        .replace('__ESOL_BR__', '&#10;')
    )

    st.markdown(
        f'<div class="section-title" style="display: flex; align-items: center; gap: 8px; margin-top: 10px;">'
        f'<span>{tr("sec_solubility_protection")}</span>'
        f'<span title="{clean_esol_tooltip}" style="cursor: help; font-size: 0.85rem; font-weight: normal; color: #1f77b4;" aria-label="Delaney ESOL explanation">(?)</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    def _remove_tier1_candidate_at(rem_idx: Optional[int]):
        t1_list = list(st.session_state.get("mined_literature_additives", []))
        if not t1_list:
            return
        if rem_idx is None or not (0 <= rem_idx < len(t1_list)):
            rem_idx = 0
        removed_cand = t1_list.pop(rem_idx)
        st.session_state["mined_literature_additives"] = t1_list

        ext_adds = st.session_state.get("extracted_additives", [])
        if isinstance(ext_adds, list) and removed_cand and removed_cand.name:
            st.session_state["extracted_additives"] = [
                a for a in ext_adds if str(a).strip().lower() != removed_cand.name.strip().lower()
            ]

        if t1_list:
            new_idx = min(rem_idx, len(t1_list) - 1)
            next_c = t1_list[new_idx]
            st.session_state["pending_tier1_choice"] = new_idx
            st.session_state["pending_tier1_bottom_choice"] = new_idx
            st.session_state["last_tier1_choice"] = new_idx
            st.session_state["last_bottom_tier1_choice"] = new_idx
            st.session_state["step2_tier1_current_page"] = (new_idx // 6) + 1
            st.session_state["selected_tier1_name"] = next_c.name
            st.session_state["selected_additive_name"] = next_c.name
            _apply_card_candidate(
                cand=next_c,
                selected_target_salt=st.session_state.get("selected_target_salt", "Zn(OTf)2"),
                doe_method=st.session_state.get("doe_method", "lhs"),
                num_doe_samples=st.session_state.get("num_doe_samples", 5),
                rack_format=st.session_state.get("rack_format", "24-vial rack"),
                total_vol_ul=st.session_state.get("total_vol_ul", 2000.0),
                published_conc_input=st.session_state.get("published_conc_input", 15.0),
                parent_electrolyte_input=next_c.parent_baseline_electrolyte or parent_electrolyte_input,
            )
        else:
            for k in (
                "pending_tier1_choice",
                "pending_tier1_bottom_choice",
                "last_tier1_choice",
                "last_bottom_tier1_choice",
                "selected_tier1_name",
                "step2_tier1_current_page",
            ):
                st.session_state.pop(k, None)
            t2_list = st.session_state.get("ai_suggested_additives", []) or st.session_state.get("suggested_candidates", [])
            if t2_list:
                next_t2 = t2_list[0]
                st.session_state["selected_tier2_name"] = next_t2.name
                _apply_card_candidate(
                    cand=next_t2,
                    selected_target_salt=st.session_state.get("selected_target_salt", "Zn(OTf)2"),
                    doe_method=st.session_state.get("doe_method", "lhs"),
                    num_doe_samples=st.session_state.get("num_doe_samples", 5),
                    rack_format=st.session_state.get("rack_format", "24-vial rack"),
                    total_vol_ul=st.session_state.get("total_vol_ul", 2000.0),
                    published_conc_input=st.session_state.get("published_conc_input", 15.0),
                    parent_electrolyte_input=next_t2.parent_baseline_electrolyte or parent_electrolyte_input,
                )
            else:
                st.session_state["selected_additive_name"] = ""
                st.session_state["custom_additive_name"] = ""
                st.session_state["custom_smiles"] = ""
                st.session_state["tab2_custom_additive_name_input"] = ""
                st.session_state["tab2_custom_smiles_input"] = ""
                st.session_state.pop("custom_formulation_plan", None)
                st.session_state.pop("custom_dispensing_schedule", None)
                st.session_state.pop("extracted_formulation_plan", None)
                st.session_state.pop("extracted_dispensing_schedule", None)
                st.rerun()

    def _remove_tier2_candidate_at(rem_idx: Optional[int]):
        t2_list = list(st.session_state.get("ai_suggested_additives", []) or st.session_state.get("suggested_candidates", []))
        if not t2_list:
            return
        if rem_idx is None or not (0 <= rem_idx < len(t2_list)):
            rem_idx = 0
        t2_list.pop(rem_idx)
        st.session_state["ai_suggested_additives"] = t2_list
        st.session_state["suggested_candidates"] = t2_list

        if t2_list:
            new_idx = min(rem_idx, len(t2_list) - 1)
            next_c = t2_list[new_idx]
            st.session_state["pending_tier2_choice"] = new_idx
            st.session_state["pending_tier2_bottom_choice"] = new_idx
            st.session_state["selected_suggested_idx"] = new_idx
            st.session_state["last_suggested_choice"] = new_idx
            st.session_state["last_bottom_tier2_choice"] = new_idx
            st.session_state["step2_tier2_current_page"] = (new_idx // 6) + 1
            st.session_state["selected_tier2_name"] = next_c.name
            _apply_card_candidate(
                cand=next_c,
                selected_target_salt=st.session_state.get("selected_target_salt", "Zn(OTf)2"),
                doe_method=st.session_state.get("doe_method", "lhs"),
                num_doe_samples=st.session_state.get("num_doe_samples", 5),
                rack_format=st.session_state.get("rack_format", "24-vial rack"),
                total_vol_ul=st.session_state.get("total_vol_ul", 2000.0),
                published_conc_input=st.session_state.get("published_conc_input", 15.0),
                parent_electrolyte_input=next_c.parent_baseline_electrolyte or parent_electrolyte_input,
            )
        else:
            for k in (
                "pending_tier2_choice",
                "pending_tier2_bottom_choice",
                "selected_suggested_idx",
                "last_suggested_choice",
                "last_bottom_tier2_choice",
                "selected_tier2_name",
                "step2_tier2_current_page",
            ):
                st.session_state.pop(k, None)
            st.rerun()

    # Synchronized live additive selector inside the Aqueous Solubility Protection section
    if not is_tier2 and key_prefix == "tier1":
        tier1_cands = st.session_state.get("mined_literature_additives", [])
        if tier1_cands:
            cur_active_t1 = st.session_state.get("selected_tier1_name", "") or cand_name
            cur_b1_idx = 0
            if cur_active_t1:
                for idx, c in enumerate(tier1_cands):
                    if c.name.lower() == cur_active_t1.lower():
                        cur_b1_idx = idx
                        break

            if "pending_tier1_bottom_choice" in st.session_state:
                st.session_state["step2_tier1_bottom_dropdown"] = st.session_state.pop("pending_tier1_bottom_choice")
            elif "step2_tier1_bottom_dropdown" not in st.session_state:
                st.session_state["step2_tier1_bottom_dropdown"] = cur_b1_idx
            elif isinstance(st.session_state["step2_tier1_bottom_dropdown"], int) and not (0 <= st.session_state["step2_tier1_bottom_dropdown"] < len(tier1_cands)):
                st.session_state["step2_tier1_bottom_dropdown"] = cur_b1_idx
            elif st.session_state.get("_lang_changed_this_run"):
                st.session_state["step2_tier1_bottom_dropdown"] = st.session_state["step2_tier1_bottom_dropdown"]

            _unspec_conc_bot = tr("card_reported_conc_unspecified")
            _unspec_salt_bot = tr("baseline_salt_unspecified")

            def _fmt_t1_bot(idx: int) -> str:
                c = tier1_cands[idx]
                f_str = format_chem_formula(c.chemical_formula or get_chemical_formula(c.smiles, c.name))
                f_disp = f" ({f_str})" if f_str else ""
                c_disp = f"{c.reported_concentration:g} {c.reported_concentration_unit or 'mM'}" if c.reported_concentration is not None else _unspec_conc_bot
                raw_s = c.parent_baseline_electrolyte
                salt_disp = _unspec_salt_bot if (not raw_s or "unspecified" in str(raw_s).lower() or "not reported" in str(raw_s).lower()) else format_chem_formula(raw_s)
                _, p_label, _, p_icon = get_candidate_provenance(c, lang=cur_lang)
                return f"{c.name}{f_disp} — {c_disp} [{salt_disp}] [{p_icon} {p_label}]"

            bot_t1_choice = st.selectbox(
                f"🎯 {tr('select_tier1_dropdown_label')}",
                options=range(len(tier1_cands)),
                format_func=_fmt_t1_bot,
                index=None,
                key="step2_tier1_bottom_dropdown",
                help="Select a literature-tested additive to formulate and analyze"
            )
            prev_bot_t1 = st.session_state.get("last_bottom_tier1_choice", cur_b1_idx)
            if st.session_state.pop("_tier1_top_changed_this_run", False):
                st.session_state["last_bottom_tier1_choice"] = cur_b1_idx
            elif bot_t1_choice is None:
                _remove_tier1_candidate_at(prev_bot_t1)
            elif bot_t1_choice != prev_bot_t1 and "last_bottom_tier1_choice" in st.session_state:
                st.session_state["active_step2_tier_tab"] = "tier1"
                st.session_state["last_bottom_tier1_choice"] = bot_t1_choice
                st.session_state["selected_tier1_name"] = tier1_cands[bot_t1_choice].name
                st.session_state["selected_additive_name"] = tier1_cands[bot_t1_choice].name
                st.session_state["last_tier1_choice"] = bot_t1_choice
                st.session_state["pending_tier1_choice"] = bot_t1_choice
                st.session_state["step2_tier1_current_page"] = (bot_t1_choice // 6) + 1
                _apply_card_candidate(
                    cand=tier1_cands[bot_t1_choice],
                    selected_target_salt=st.session_state.get("selected_target_salt", "Zn(OTf)2"),
                    doe_method=st.session_state.get("doe_method", "lhs"),
                    num_doe_samples=st.session_state.get("num_doe_samples", 5),
                    rack_format=st.session_state.get("rack_format", "24-vial rack"),
                    total_vol_ul=st.session_state.get("total_vol_ul", 2000.0),
                    published_conc_input=st.session_state.get("published_conc_input", 15.0),
                    parent_electrolyte_input=tier1_cands[bot_t1_choice].parent_baseline_electrolyte or parent_electrolyte_input,
                )
            else:
                st.session_state["last_bottom_tier1_choice"] = bot_t1_choice

    elif is_tier2 and key_prefix == "tier2":
        tier2_cands = st.session_state.get("ai_suggested_additives", []) or st.session_state.get("suggested_candidates", [])
        if tier2_cands:
            cur_active_t2 = st.session_state.get("selected_tier2_name", "") or cand_name
            cur_b2_idx = 0
            if cur_active_t2:
                for idx, c in enumerate(tier2_cands):
                    if c.name.lower() == cur_active_t2.lower():
                        cur_b2_idx = idx
                        break

            if "pending_tier2_bottom_choice" in st.session_state:
                st.session_state["step2_tier2_bottom_dropdown"] = st.session_state.pop("pending_tier2_bottom_choice")
            elif "step2_tier2_bottom_dropdown" not in st.session_state:
                st.session_state["step2_tier2_bottom_dropdown"] = cur_b2_idx
            elif isinstance(st.session_state["step2_tier2_bottom_dropdown"], int) and not (0 <= st.session_state["step2_tier2_bottom_dropdown"] < len(tier2_cands)):
                st.session_state["step2_tier2_bottom_dropdown"] = cur_b2_idx
            elif st.session_state.get("_lang_changed_this_run"):
                st.session_state["step2_tier2_bottom_dropdown"] = st.session_state["step2_tier2_bottom_dropdown"]

            def _fmt_t2_bot(idx: int) -> str:
                c = tier2_cands[idx]
                f_str = format_chem_formula(c.chemical_formula or get_chemical_formula(c.smiles, c.name))
                f_disp = f" ({f_str})" if f_str else ""
                sim = c.similarity_score if c.similarity_score is not None else 0.5
                met = c.similarity_metric or "Tanimoto"
                return f"{c.name}{f_disp} — Sim: {sim:.2f} [{met}]"

            bot_t2_choice = st.selectbox(
                f"✨ {tr('select_tier2_dropdown_label')}",
                options=range(len(tier2_cands)),
                format_func=_fmt_t2_bot,
                index=None,
                key="step2_tier2_bottom_dropdown",
                help="Select an AI-suggested novel analogue to formulate and analyze"
            )
            prev_bot_t2 = st.session_state.get("last_bottom_tier2_choice", cur_b2_idx)
            if bot_t2_choice is None:
                _remove_tier2_candidate_at(prev_bot_t2)
            elif bot_t2_choice != prev_bot_t2 and "last_bottom_tier2_choice" in st.session_state:
                st.session_state["active_step2_tier_tab"] = "tier2"
                st.session_state["last_bottom_tier2_choice"] = bot_t2_choice
                st.session_state["selected_tier2_name"] = tier2_cands[bot_t2_choice].name
                st.session_state["selected_suggested_idx"] = bot_t2_choice
                st.session_state["last_suggested_choice"] = bot_t2_choice
                st.session_state["pending_tier2_choice"] = bot_t2_choice
                st.session_state["step2_tier2_current_page"] = (bot_t2_choice // 6) + 1
                st.session_state["is_active_baseline_unspecified"] = (
                    tier2_cands[bot_t2_choice].reported_concentration is None or
                    not tier2_cands[bot_t2_choice].parent_baseline_electrolyte or
                    "unspecified" in str(tier2_cands[bot_t2_choice].parent_baseline_electrolyte).lower() or
                    "not reported" in str(tier2_cands[bot_t2_choice].parent_baseline_electrolyte).lower()
                )
                _apply_card_candidate(
                    cand=tier2_cands[bot_t2_choice],
                    selected_target_salt=st.session_state.get("selected_target_salt", "Zn(OTf)2"),
                    doe_method=st.session_state.get("doe_method", "lhs"),
                    num_doe_samples=st.session_state.get("num_doe_samples", 5),
                    rack_format=st.session_state.get("rack_format", "24-vial rack"),
                    total_vol_ul=st.session_state.get("total_vol_ul", 2000.0),
                    published_conc_input=st.session_state.get("published_conc_input", 15.0),
                    parent_electrolyte_input=tier2_cands[bot_t2_choice].parent_baseline_electrolyte or parent_electrolyte_input,
                )
            else:
                st.session_state["last_bottom_tier2_choice"] = bot_t2_choice

    g_c1, g_c2, g_c3, g_c4 = st.columns(4)
    if esol_ok and esol_val is not None and theor_sat_mm is not None and hard_ceiling_mm is not None:
        with g_c1:
            st.metric(tr("metric_esol_logs"), f"{esol_val:.2f} LogS", tr("metric_intrinsic_solubility"))
        with g_c2:
            st.metric(tr("metric_theor_sat"), format_solubility_conc(theor_sat_mm), tr("metric_formula_math"))
        with g_c3:
            st.metric(tr("metric_safety_cap_max"), format_solubility_conc(hard_ceiling_mm), tr("metric_calc_safety"))
        with g_c4:
            st.metric(tr("metric_precip_protection"), tr("metric_active_enforced"), tr("metric_deterministic_cap"))
    else:
        with g_c1:
            st.metric(tr("metric_esol_logs"), "N/A", esol_reason)
        with g_c2:
            st.metric(tr("metric_theor_sat"), "N/A", esol_reason)
        with g_c3:
            st.metric(tr("metric_safety_cap_max"), "N/A", esol_reason)
        with g_c4:
            st.metric(tr("metric_precip_protection"), "N/A", esol_reason)


def render_two_tier_card_deck(
    selected_target_salt: str,
    doe_method: str = "lhs",
    num_doe_samples: int = 5,
    rack_format: str = "24-vial rack",
    total_vol_ul: float = 2000.0,
    published_conc_input: float = 15.0,
    parent_electrolyte_input: str = "2 M ZnSO4",
    is_demo: bool = False,
    cur_lang: Optional[str] = None,
):
    """Render Two-Tier Paginated Card Deck in Step 2 separating mined literature additives and AI analogues."""
    cur_lang = cur_lang or st.session_state.get("app_language", "en")
    tier1_candidates = st.session_state.get("mined_literature_additives", [])
    tier2_candidates = st.session_state.get("ai_suggested_additives", [])
    if not tier2_candidates and "suggested_candidates" in st.session_state:
        tier2_candidates = st.session_state.get("suggested_candidates", [])

    def _remove_top_tier1_candidate(rem_idx: Optional[int]):
        t1_list = list(st.session_state.get("mined_literature_additives", []))
        if not t1_list:
            return
        if rem_idx is None or not (0 <= rem_idx < len(t1_list)):
            rem_idx = 0
        removed_cand = t1_list.pop(rem_idx)
        st.session_state["mined_literature_additives"] = t1_list

        ext_adds = st.session_state.get("extracted_additives", [])
        if isinstance(ext_adds, list) and removed_cand and removed_cand.name:
            st.session_state["extracted_additives"] = [
                a for a in ext_adds if str(a).strip().lower() != removed_cand.name.strip().lower()
            ]

        if t1_list:
            new_idx = min(rem_idx, len(t1_list) - 1)
            next_c = t1_list[new_idx]
            st.session_state["pending_tier1_choice"] = new_idx
            st.session_state["pending_tier1_bottom_choice"] = new_idx
            st.session_state["last_tier1_choice"] = new_idx
            st.session_state["last_bottom_tier1_choice"] = new_idx
            st.session_state["step2_tier1_current_page"] = (new_idx // 6) + 1
            st.session_state["selected_tier1_name"] = next_c.name
            st.session_state["selected_additive_name"] = next_c.name
            _apply_card_candidate(
                cand=next_c,
                selected_target_salt=selected_target_salt,
                doe_method=doe_method,
                num_doe_samples=num_doe_samples,
                rack_format=rack_format,
                total_vol_ul=total_vol_ul,
                published_conc_input=published_conc_input,
                parent_electrolyte_input=next_c.parent_baseline_electrolyte or parent_electrolyte_input,
            )
        else:
            for k in (
                "pending_tier1_choice",
                "pending_tier1_bottom_choice",
                "last_tier1_choice",
                "last_bottom_tier1_choice",
                "selected_tier1_name",
                "step2_tier1_current_page",
            ):
                st.session_state.pop(k, None)
            if tier2_candidates:
                next_t2 = tier2_candidates[0]
                st.session_state["selected_tier2_name"] = next_t2.name
                _apply_card_candidate(
                    cand=next_t2,
                    selected_target_salt=selected_target_salt,
                    doe_method=doe_method,
                    num_doe_samples=num_doe_samples,
                    rack_format=rack_format,
                    total_vol_ul=total_vol_ul,
                    published_conc_input=published_conc_input,
                    parent_electrolyte_input=next_t2.parent_baseline_electrolyte or parent_electrolyte_input,
                )
            else:
                if not is_demo:
                    st.session_state["selected_additive_name"] = ""
                    st.session_state["custom_additive_name"] = ""
                    st.session_state["custom_smiles"] = ""
                    st.session_state["tab2_custom_additive_name_input"] = ""
                    st.session_state["tab2_custom_smiles_input"] = ""
                    st.session_state.pop("custom_formulation_plan", None)
                    st.session_state.pop("custom_dispensing_schedule", None)
                st.session_state.pop("extracted_formulation_plan", None)
                st.session_state.pop("extracted_dispensing_schedule", None)
                st.rerun()

    def _remove_top_tier2_candidate(rem_idx: Optional[int]):
        t2_list = list(st.session_state.get("ai_suggested_additives", []) or st.session_state.get("suggested_candidates", []))
        if not t2_list:
            return
        if rem_idx is None or not (0 <= rem_idx < len(t2_list)):
            rem_idx = 0
        t2_list.pop(rem_idx)
        st.session_state["ai_suggested_additives"] = t2_list
        st.session_state["suggested_candidates"] = t2_list

        if t2_list:
            new_idx = min(rem_idx, len(t2_list) - 1)
            next_c = t2_list[new_idx]
            st.session_state["pending_tier2_choice"] = new_idx
            st.session_state["pending_tier2_bottom_choice"] = new_idx
            st.session_state["selected_suggested_idx"] = new_idx
            st.session_state["last_suggested_choice"] = new_idx
            st.session_state["last_bottom_tier2_choice"] = new_idx
            st.session_state["step2_tier2_current_page"] = (new_idx // 6) + 1
            st.session_state["selected_tier2_name"] = next_c.name
            _apply_card_candidate(
                cand=next_c,
                selected_target_salt=selected_target_salt,
                doe_method=doe_method,
                num_doe_samples=num_doe_samples,
                rack_format=rack_format,
                total_vol_ul=total_vol_ul,
                published_conc_input=published_conc_input,
                parent_electrolyte_input=next_c.parent_baseline_electrolyte or parent_electrolyte_input,
            )
        else:
            for k in (
                "pending_tier2_choice",
                "pending_tier2_bottom_choice",
                "selected_suggested_idx",
                "last_suggested_choice",
                "last_bottom_tier2_choice",
                "selected_tier2_name",
                "step2_tier2_current_page",
            ):
                st.session_state.pop(k, None)
            st.rerun()

    st.markdown(f'<div class="section-title">🧪 {tr("sec_step2_additive_selection")}</div>', unsafe_allow_html=True)
    st.caption(tr("sec_step2_additive_selection_desc"))

    t1_tab_label = f"📚 {tr('tier1_title')} ({len(tier1_candidates)})"
    t2_tab_label = f"✨ {tr('tier2_title')} ({len(tier2_candidates)})"

    _sync_t1_clicked = st.button("Sync Tier 1", key="btn_sync_tier1_active_tab") if tier1_candidates else False
    _sync_t2_clicked = st.button("Sync Tier 2", key="btn_sync_tier2_active_tab") if tier2_candidates else False

    cur_tabs_widget_val = st.session_state.get("step2_tier_tabs")
    prev_tabs_widget_val = st.session_state.get("_last_step2_tier_tabs_val")
    user_clicked_tier_tab = (
        cur_tabs_widget_val in (t1_tab_label, t2_tab_label)
        and prev_tabs_widget_val in (t1_tab_label, t2_tab_label)
        and cur_tabs_widget_val != prev_tabs_widget_val
    )

    if _sync_t1_clicked:
        st.session_state["active_step2_tier_tab"] = "tier1"
        st.session_state["step2_tier_tabs"] = t1_tab_label
    elif _sync_t2_clicked:
        st.session_state["active_step2_tier_tab"] = "tier2"
        st.session_state["step2_tier_tabs"] = t2_tab_label
    elif user_clicked_tier_tab:
        st.session_state["active_step2_tier_tab"] = "tier2" if cur_tabs_widget_val == t2_tab_label else "tier1"
    else:
        st.session_state["step2_tier_tabs"] = (
            t2_tab_label if st.session_state.get("active_step2_tier_tab", "tier1") == "tier2" else t1_tab_label
        )

    with st.container(key="step2_two_tier_tabs_container"):
        try:
            tier_tabs = st.tabs(
                [t1_tab_label, t2_tab_label],
                key="step2_tier_tabs",
                on_change="rerun",
            )
        except TypeError:
            tier_tabs = st.tabs([t1_tab_label, t2_tab_label])

    st.session_state["_last_step2_tier_tabs_val"] = st.session_state.get("step2_tier_tabs")
    cur_tier_tab = st.session_state.get("active_step2_tier_tab", "tier1")
    _switched_to_t1 = _sync_t1_clicked or (user_clicked_tier_tab and cur_tier_tab == "tier1")
    _switched_to_t2 = _sync_t2_clicked or (user_clicked_tier_tab and cur_tier_tab == "tier2")

    # -------------------------------------------------------------------------
    # Tier 1: Mined Literature Additives (Tested Chemistries)
    # -------------------------------------------------------------------------
    with tier_tabs[0]:
        st.caption(tr("tier1_caption"))
        if not tier1_candidates:
            st.session_state.pop("step2_tier1_additive_dropdown", None)
            st.session_state.pop("step2_tier1_bottom_dropdown", None)
            st.info(tr("no_tier1_additives_info"))
            if is_demo:
                col_b1, col_b2 = st.columns([1.5, 2.5])
                with col_b1:
                    if st.button(f"⚡ {tr('btn_load_sample_candidates')}", key="btn_load_sample_candidates_t1", use_container_width=True):
                        lit_eng = LiteratureDiscoveryEngine()
                        ext = PaperExtractor(api_key=None)
                        sample_recs = ext.extract_batch_from_papers(OFFLINE_SAMPLE_PAPERS)
                        sample_adds = [r["additive"] for r in sample_recs if r.get("additive")]
                        t1_c, t2_c = lit_eng.generate_two_tier_candidates(sample_adds, use_pubchem=False)
                        st.session_state["mined_literature_additives"] = t1_c
                        st.session_state["extracted_additives"] = sample_adds
                        st.session_state["ai_suggested_additives"] = t2_c
                        st.session_state["suggested_candidates"] = t2_c
                        st.rerun()
                with col_b2:
                    if st.button(f"💡 {tr('suggested_example_banner')}", key="btn_preview_sample_suggestions_t1", use_container_width=True):
                        lit_eng = LiteratureDiscoveryEngine()
                        ext = PaperExtractor(api_key=None)
                        sample_recs = ext.extract_batch_from_papers(OFFLINE_SAMPLE_PAPERS[:3])
                        sample_adds = [r["additive"] for r in sample_recs if r.get("additive")]
                        t1_c, t2_c = lit_eng.generate_two_tier_candidates(sample_adds, use_pubchem=False)
                        st.session_state["mined_literature_additives"] = t1_c
                        st.session_state["extracted_additives"] = sample_adds
                        st.session_state["ai_suggested_additives"] = t2_c
                        st.session_state["suggested_candidates"] = t2_c
                        st.rerun()
            _render_upload_popover(key_suffix="t1_empty", use_container_width=False)
        else:
            # 1. Quick Dropdown Menu (Prominent & Visible by Default at Top)
            cur_active_name = st.session_state.get("selected_tier1_name") or st.session_state.get("selected_additive_name", "")
            cur_t1_idx = 0
            if cur_active_name:
                for idx, c in enumerate(tier1_candidates):
                    if c.name.lower() == cur_active_name.lower():
                        cur_t1_idx = idx
                        break

            _unspec_conc_top = tr("card_reported_conc_unspecified")
            _unspec_salt_top = tr("baseline_salt_unspecified")

            def _format_tier1_option(idx: int) -> str:
                c = tier1_candidates[idx]
                f_str = format_chem_formula(c.chemical_formula or get_chemical_formula(c.smiles, c.name))
                f_disp = f" ({f_str})" if f_str else ""
                if c.reported_concentration is not None:
                    c_disp = f"{c.reported_concentration:g} {c.reported_concentration_unit or 'mM'}"
                else:
                    c_disp = _unspec_conc_top
                raw_s = c.parent_baseline_electrolyte
                if not raw_s or "unspecified" in str(raw_s).lower() or "not reported" in str(raw_s).lower():
                    salt_disp = _unspec_salt_top
                else:
                    salt_disp = format_chem_formula(raw_s)
                _, p_label, _, p_icon = get_candidate_provenance(c, lang=cur_lang)
                return f"{c.name}{f_disp} — {c_disp} [{salt_disp}] [{p_icon} {p_label}]"

            if "pending_tier1_choice" in st.session_state:
                st.session_state["step2_tier1_additive_dropdown"] = st.session_state.pop("pending_tier1_choice")
            elif "step2_tier1_additive_dropdown" not in st.session_state:
                st.session_state["step2_tier1_additive_dropdown"] = cur_t1_idx
            elif isinstance(st.session_state["step2_tier1_additive_dropdown"], int) and not (0 <= st.session_state["step2_tier1_additive_dropdown"] < len(tier1_candidates)):
                st.session_state["step2_tier1_additive_dropdown"] = cur_t1_idx
            elif st.session_state.get("_lang_changed_this_run"):
                st.session_state["step2_tier1_additive_dropdown"] = st.session_state["step2_tier1_additive_dropdown"]

            col_t1_sel, col_t1_edit, col_t1_up = st.columns([3.1, 0.95, 0.95], vertical_alignment="bottom")
            with col_t1_sel:
                t1_choice = st.selectbox(
                    tr("select_tier1_dropdown_label"),
                    range(len(tier1_candidates)),
                    format_func=_format_tier1_option,
                    index=None,
                    key="step2_tier1_additive_dropdown",
                    help="Select a literature additive for formulation planning"
                )

            if t1_choice is None:
                _remove_top_tier1_candidate(st.session_state.get("last_tier1_choice", cur_t1_idx))
                return

            # Reactive sync on dropdown change or sub-tab switch
            prev_t1_choice = st.session_state.get("last_tier1_choice")
            t1_changed = (prev_t1_choice is not None and prev_t1_choice != t1_choice)
            picked_c = tier1_candidates[t1_choice]
            active_add_lower = str(st.session_state.get("selected_additive_name") or "").strip().lower()
            sel_t2_lower = str(st.session_state.get("selected_tier2_name") or "").strip().lower()
            t2_names_lower = {c.name.strip().lower() for c in tier2_candidates if c.name}
            t1_needs_active_sync = (
                st.session_state.get("active_step2_tier_tab", "tier1") == "tier1"
                and (
                    prev_t1_choice is None
                    or not active_add_lower
                    or active_add_lower in t2_names_lower
                    or (sel_t2_lower and active_add_lower == sel_t2_lower and picked_c.name.strip().lower() != sel_t2_lower)
                )
            )
            if not _switched_to_t2 and (
                t1_changed
                or _switched_to_t1
                or t1_needs_active_sync
            ):
                if t1_changed:
                    st.session_state["_tier1_top_changed_this_run"] = True
                st.session_state["active_step2_tier_tab"] = "tier1"
                if t1_changed or "step2_tier1_current_page" not in st.session_state:
                    st.session_state["step2_tier1_current_page"] = (t1_choice // 6) + 1
                st.session_state["last_tier1_choice"] = t1_choice
                st.session_state["last_bottom_tier1_choice"] = t1_choice
                st.session_state["pending_tier1_bottom_choice"] = t1_choice
                c_smi = picked_c.smiles
                if not c_smi:
                    clean_base = re.sub(r'\s*\([^)]*\)\s*$', '', picked_c.name).strip().lower()
                    matching = next((a["smiles"] for a in CURATED_ZIB_ADDITIVES if a["name"].lower() == picked_c.name.lower() or a["name"].lower() == clean_base), None)
                    c_smi = matching or ""
                custom_list = st.session_state.get("custom_additives", [])
                if not any(a["name"].lower() == picked_c.name.lower() for a in CURATED_ZIB_ADDITIVES + custom_list):
                    custom_list.append({"name": picked_c.name, "smiles": c_smi})
                    st.session_state["custom_additives"] = custom_list
                st.session_state["selected_additive_name"] = picked_c.name
                st.session_state["selected_tier1_name"] = picked_c.name
                st.session_state["custom_additive_name"] = picked_c.name
                st.session_state["custom_smiles"] = c_smi
                st.session_state["tab2_mol_select"] = picked_c.name
                st.session_state["_last_inspect_mol_select"] = picked_c.name
                st.session_state["_last_inspect_mol_widget"] = picked_c.name
                st.session_state["_last_sidebar_additive_widget"] = picked_c.name
                st.session_state["tab2_custom_additive_name_input"] = picked_c.name
                st.session_state["tab2_custom_smiles_input"] = c_smi
                if picked_c.reported_concentration is not None:
                    conv_c, _ = convert_concentration_to_mM(
                        picked_c.reported_concentration,
                        picked_c.reported_concentration_unit,
                        mw=picked_c.molecular_weight,
                    )
                    st.session_state["published_conc_input"] = float(conv_c) if conv_c is not None else float(picked_c.reported_concentration)
                else:
                    st.session_state["published_conc_input"] = 15.0
                is_unspec_salt_t1 = (
                    not picked_c.parent_baseline_electrolyte or
                    "unspecified" in str(picked_c.parent_baseline_electrolyte).lower() or
                    "not reported" in str(picked_c.parent_baseline_electrolyte).lower()
                )
                if not is_unspec_salt_t1:
                    st.session_state["parent_electrolyte_input"] = picked_c.parent_baseline_electrolyte
                else:
                    st.session_state["parent_electrolyte_input"] = "2 M ZnSO4"
                st.session_state["is_active_baseline_unspecified"] = (
                    picked_c.reported_concentration is None or is_unspec_salt_t1
                )

                sync_custom_formulation_pipeline(
                    additive_name=picked_c.name,
                    published_conc_mM=st.session_state.get("published_conc_input", 15.0),
                    parent_electrolyte=st.session_state.get("parent_electrolyte_input", "2 M ZnSO4"),
                    target_electrolyte=selected_target_salt,
                    smiles=c_smi,
                    doe_method=doe_method,
                    num_samples=num_doe_samples,
                    seed=42,
                    rack_format=rack_format,
                    total_volume_uL=total_vol_ul,
                    session_state=st.session_state,
                )
                if t1_changed or _switched_to_t1:
                    st.rerun()
            else:
                st.session_state["last_tier1_choice"] = t1_choice
                if tier1_candidates:
                    st.session_state["selected_tier1_name"] = tier1_candidates[t1_choice].name

            # ✏️ Edit Additive Popover
            with col_t1_edit:
                picked_cand_t1 = tier1_candidates[t1_choice]
                with st.popover(f"✏️ {tr('btn_edit_additive')}", use_container_width=True):
                    st.markdown(f'<div style="font-size: 0.96rem; font-weight: 700; margin-bottom: 2px;">✏️ {tr("edit_additive_title")}</div>', unsafe_allow_html=True)
                    st.caption(tr("edit_additive_desc"))

                    edit_name = st.text_input(tr("edit_additive_name_label"), value=picked_cand_t1.name, key=f"edit_name_in_{t1_choice}")

                    cur_salt_val = picked_cand_t1.parent_baseline_electrolyte or "2 M ZnSO4"
                    if "unspecified" in str(cur_salt_val).lower() or "not reported" in str(cur_salt_val).lower():
                        cur_salt_val = "2 M ZnSO4"

                    edit_salt = st.text_input(
                        tr("edit_baseline_salt_label"),
                        value=cur_salt_val,
                        help="e.g. 2 M ZnSO4, 1 M Zn(OTf)2, 3 M Zn(OTf)2, 0.5 M Zn(ClO4)2",
                        key=f"edit_salt_in_{t1_choice}"
                    )

                    if picked_cand_t1.reported_concentration is not None:
                        _conv_ed, _ = convert_concentration_to_mM(
                            picked_cand_t1.reported_concentration,
                            picked_cand_t1.reported_concentration_unit,
                            mw=picked_cand_t1.molecular_weight,
                        )
                        raw_edit_c = float(_conv_ed) if _conv_ed is not None else float(picked_cand_t1.reported_concentration)
                    else:
                        raw_edit_c = 15.0
                    cur_conc_val = min(max(raw_edit_c, 0.01), 10000.0)
                    edit_conc = st.number_input(
                        tr("edit_concentration_label"),
                        value=cur_conc_val,
                        min_value=0.01,
                        max_value=10000.0,
                        step=5.0,
                        key=f"edit_conc_in_{t1_choice}"
                    )

                    edit_smiles = st.text_input(
                        tr("edit_smiles_label"),
                        value=picked_cand_t1.smiles or "",
                        key=f"edit_smiles_in_{t1_choice}"
                    )

                    edit_formula = st.text_input(
                        tr("edit_formula_label"),
                        value=picked_cand_t1.chemical_formula or "",
                        key=f"edit_formula_in_{t1_choice}"
                    )

                    cur_mechs = ", ".join(picked_cand_t1.reported_mechanisms) if picked_cand_t1.reported_mechanisms else (picked_cand_t1.mechanism_summary or "")
                    if "unspecified" in cur_mechs.lower() or "not reported" in cur_mechs.lower():
                        cur_mechs = ""
                    edit_mechs = st.text_input(
                        tr("edit_mechanisms_label"),
                        value=cur_mechs,
                        placeholder="e.g. Solvation sheath modulation, Dendrite suppression",
                        key=f"edit_mechs_in_{t1_choice}"
                    )

                    if st.button(f"💾 {tr('btn_save_changes')}", key=f"btn_save_edit_{t1_choice}", type="primary", use_container_width=True):
                        updated_c = tier1_candidates[t1_choice]
                        updated_c.name = edit_name.strip() or updated_c.name
                        updated_c.parent_baseline_electrolyte = edit_salt.strip() or "2 M ZnSO4"
                        updated_c.reported_concentration = float(edit_conc)
                        updated_c.reported_concentration_unit = "mM"
                        if edit_smiles.strip() and is_valid_smiles(edit_smiles.strip()):
                            can_smi = canonicalize_smiles(edit_smiles.strip())
                            if can_smi:
                                updated_c.smiles = can_smi
                                mol = Chem.MolFromSmiles(can_smi)
                                if mol:
                                    updated_c.molecular_weight = round(float(Descriptors.MolWt(mol)), 2)
                                    updated_c.logp = round(float(Descriptors.MolLogP(mol)), 2)
                                    updated_c.logs = calculate_delaney_esol(mol)
                                    updated_c.tpsa = round(float(Descriptors.TPSA(mol)), 2)
                                    updated_c.hbd = int(Lipinski.NumHDonors(mol))
                                    updated_c.hba = int(Lipinski.NumHAcceptors(mol))
                                    updated_c.rotatable_bonds = int(Lipinski.NumRotatableBonds(mol))

                        if edit_formula.strip():
                            updated_c.chemical_formula = edit_formula.strip()
                        elif updated_c.smiles:
                            updated_c.chemical_formula = get_chemical_formula(updated_c.smiles, updated_c.name)

                        if edit_mechs.strip():
                            new_m_list = [m.strip() for m in re.split(r'[,;]', edit_mechs) if m.strip()]
                            updated_c.reported_mechanisms = new_m_list
                            updated_c.mechanism_summary = ", ".join(new_m_list)
                        else:
                            updated_c.reported_mechanisms = []
                            updated_c.mechanism_summary = "Unspecified / Not reported"

                        tier1_candidates[t1_choice] = updated_c
                        st.session_state["mined_literature_additives"] = tier1_candidates

                        st.session_state["selected_additive_name"] = updated_c.name
                        st.session_state["custom_additive_name"] = updated_c.name
                        st.session_state["custom_smiles"] = updated_c.smiles
                        st.session_state["tab2_mol_select"] = updated_c.name
                        st.session_state["_last_inspect_mol_select"] = updated_c.name
                        st.session_state["tab2_custom_additive_name_input"] = updated_c.name
                        st.session_state["tab2_custom_smiles_input"] = updated_c.smiles
                        st.session_state["published_conc_input"] = float(updated_c.reported_concentration)
                        st.session_state["parent_electrolyte_input"] = updated_c.parent_baseline_electrolyte

                        sync_custom_formulation_pipeline(
                            additive_name=updated_c.name,
                            published_conc_mM=updated_c.reported_concentration,
                            parent_electrolyte=updated_c.parent_baseline_electrolyte,
                            target_electrolyte=selected_target_salt,
                            smiles=updated_c.smiles,
                            doe_method=doe_method,
                            num_samples=num_doe_samples,
                            seed=42,
                            rack_format=rack_format,
                            total_volume_uL=total_vol_ul,
                            session_state=st.session_state,
                        )
                        st.session_state["last_extracted_notice"] = tr("toast_additive_updated", name=updated_c.name, salt=updated_c.parent_baseline_electrolyte, conc=updated_c.reported_concentration)
                        st.rerun()

            # 📤 Upload Additives Popover
            with col_t1_up:
                _render_upload_popover(key_suffix="t1", use_container_width=True)

            picked_cand_t1 = tier1_candidates[t1_choice]
            with st.container(key="tier1_exploratory_notice_slot"):
                if picked_cand_t1.reported_concentration is None:
                    st.caption(f"💡 {tr('exploratory_anchor_notice', conc=15.0)}")

            # 2. Detailed Card Deck (Collapsed by default in an Expander, stays open across dropdown changes)
            with st.expander(
                f"🃏 {tr('view_card_deck_label')} ({len(tier1_candidates)} {tr('candidates_label')})",
                expanded=False,
                key="step2_tier1_card_deck_expander",
            ):
                PAGE_SIZE = 6
                total_t1 = len(tier1_candidates)
                total_pages_t1 = max(1, (total_t1 + PAGE_SIZE - 1) // PAGE_SIZE)
                curr_page_t1 = st.session_state.get("step2_tier1_current_page", 1)
                if curr_page_t1 > total_pages_t1:
                    curr_page_t1 = total_pages_t1
                    st.session_state["step2_tier1_current_page"] = curr_page_t1
                if curr_page_t1 < 1:
                    curr_page_t1 = 1
                    st.session_state["step2_tier1_current_page"] = curr_page_t1

                start_t1 = (curr_page_t1 - 1) * PAGE_SIZE
                end_t1 = min(total_t1, curr_page_t1 * PAGE_SIZE)

                col_p, col_i, col_n = st.columns([1.2, 2.6, 1.2], vertical_alignment="center")
                with col_p:
                    if st.button(f"◀ {tr('pagination_prev')}", key="tier1_prev_page", disabled=(curr_page_t1 <= 1), use_container_width=True):
                        st.session_state["step2_tier1_current_page"] = curr_page_t1 - 1
                        st.rerun()
                with col_i:
                    st.markdown(
                        f'<div style="text-align: center; font-weight: 600; font-size: 0.90rem; color: inherit; padding: 4px 0;">'
                        f'{tr("pagination_page_info", page=curr_page_t1, total_pages=total_pages_t1, start=start_t1 + 1, end=end_t1, total_items=total_t1)}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                with col_n:
                    if st.button(f"{tr('pagination_next')} ▶", key="tier1_next_page", disabled=(curr_page_t1 >= total_pages_t1), use_container_width=True):
                        st.session_state["step2_tier1_current_page"] = curr_page_t1 + 1
                        st.rerun()

                st.markdown('<div style="margin-bottom: 8px;"></div>', unsafe_allow_html=True)
                col_grid1 = st.columns(2)
                for i, idx in enumerate(range(start_t1, end_t1)):
                    cand = tier1_candidates[idx]
                    with col_grid1[i % 2]:
                        cand_name = cand.name
                        cand_smiles = cand.smiles
                        cand_formula = cand.chemical_formula or get_chemical_formula(cand_smiles, cand_name)
                        fmt_formula = format_chem_formula(cand_formula) if cand_formula else ""
                        formula_disp = f' <span style="font-family: monospace; font-size: 0.88rem; color: #1f77b4;">({fmt_formula})</span>' if fmt_formula else ""
                        cur_active_name = st.session_state.get("selected_additive_name", "")
                        is_active = bool(cand_name and cur_active_name and cand_name.lower() == cur_active_name.lower())
                        cand_has_esol, _ = is_esol_applicable(cand_smiles, cand.logs)
                        if cand_has_esol and cand.logs is not None:
                            theor_sat_mm = round((10 ** cand.logs) * 1000.0, 1)
                            safety_cap_mm = round(0.80 * theor_sat_mm, 1)
                            esol_disp = format_solubility_conc(theor_sat_mm)
                            safety_disp = f'<span style="color: #2ca02c; font-weight: 600;">{format_solubility_conc(safety_cap_mm)}</span>'
                        else:
                            esol_disp = "<em>N/A</em>"
                            safety_disp = "<em>N/A</em>"

                        if cand.reported_concentration is not None:
                            conc_disp = f'<span style="font-weight: 600;">{cand.reported_concentration:g} {cand.reported_concentration_unit or "mM"}</span>'
                        else:
                            conc_disp = f'<span class="badge-tag badge-amber">⚠️ {tr("card_reported_conc_unspecified")}</span>'

                        base_salt = cand.parent_baseline_electrolyte
                        is_unspec_salt = (not base_salt or "unspecified" in str(base_salt).lower() or "not reported" in str(base_salt).lower())
                        if is_unspec_salt:
                            base_salt_display = f'<span class="badge-tag badge-amber">⚠️ {tr("baseline_salt_unspecified")}</span>'
                        else:
                            base_salt_display = f'<span style="font-weight: 600;">{format_chem_formula(base_salt)}</span>'

                        raw_m = ", ".join(cand.reported_mechanisms) if cand.reported_mechanisms else (cand.mechanism_summary or "")
                        if not raw_m or "unspecified" in raw_m.lower() or "not reported" in raw_m.lower():
                            mechs_disp = f'<em>{tr("card_mechanism_none_or_unspecified")}</em>'
                        else:
                            mechs_disp = f'<em>{clean_scientific_text(raw_m)}</em>'

                        _, p_label, p_badge, p_icon = get_candidate_provenance(cand, lang=cur_lang)
                        prov_badge = f'<span class="badge-tag {p_badge}">{p_icon} {p_label}</span>'
                        active_badge = f'<span class="badge-tag badge-green">✅ {tr("card_active_badge")}</span>' if is_active else f'<span class="badge-tag badge-teal">📚 {tr("tier_tested_badge")}</span>'

                        card_html = f"""
                        <div class="paper-analysis-box candidate-card-deck-box" style="border-left: 4px solid {'#2ecc71' if is_active else '#1f77b4'};">
                            <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; margin-bottom: 4px;">
                                <div class="candidate-card-clamp" style="font-size: 1.0rem; font-weight: 700; color: inherit; min-width: 0; flex: 1 1 auto; overflow-wrap: break-word;">
                                    🧪 <strong>{cand_name}</strong>{formula_disp}
                                </div>
                                <div style="display: flex; gap: 4px; align-items: center; flex-shrink: 0; white-space: nowrap;">
                                    {prov_badge}
                                    {active_badge}
                                </div>
                            </div>
                            <div style="font-size: 0.86rem; color: inherit; margin-bottom: 2px;">
                                • <strong>{tr('card_reported_conc')}:</strong> {conc_disp} | <strong>{tr('card_baseline_salt')}:</strong> {base_salt_display}
                            </div>
                            <div style="font-size: 0.86rem; color: inherit; margin-bottom: 2px;">
                                • <strong>{tr('card_delaney_esol')}:</strong> {esol_disp} | <strong>{tr('card_safety_cap')}:</strong> {safety_disp}
                            </div>
                            <div class="candidate-card-clamp" style="font-size: 0.82rem; color: inherit; margin-bottom: 2px;">
                                • <strong>{tr('card_mechanisms')}:</strong> {mechs_disp}
                            </div>
                            <div class="candidate-card-clamp" style="font-size: 0.82rem; color: inherit;">
                                • <strong>{tr('card_provenance_label')}:</strong> {prov_badge} — <em>{clean_scientific_text(cand.rationale)}</em>
                            </div>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)

                        if st.button(
                            f"🧪 {tr('btn_use_this_additive')}: {cand_name}",
                            key=f"btn_formulate_t1_{idx}",
                            type="primary" if not is_active else "secondary",
                            use_container_width=True,
                            help=f"Select {cand_name} for formulation planning"
                        ):
                            st.session_state["active_step2_tier_tab"] = "tier1"
                            st.session_state["selected_tier1_name"] = cand.name
                            st.session_state["selected_additive_name"] = cand.name
                            st.session_state["last_tier1_choice"] = idx
                            st.session_state["last_bottom_tier1_choice"] = idx
                            st.session_state["pending_tier1_choice"] = idx
                            st.session_state["pending_tier1_bottom_choice"] = idx
                            st.session_state["step2_tier1_current_page"] = (idx // 6) + 1
                            _apply_card_candidate(
                                cand=cand,
                                selected_target_salt=selected_target_salt,
                                doe_method=doe_method,
                                num_doe_samples=num_doe_samples,
                                rack_format=rack_format,
                                total_vol_ul=total_vol_ul,
                                published_conc_input=published_conc_input,
                                parent_electrolyte_input=parent_electrolyte_input,
                            )
                            st.rerun()

        # Candidate Chemical Dossier & Analysis for Selected Tier 1 Additive
        if tier1_candidates:
            picked_cand_t1 = tier1_candidates[t1_choice]
            _render_candidate_chemical_dossier(
                cand_name=picked_cand_t1.name,
                cand_smiles=picked_cand_t1.smiles or "",
                is_tier2=False,
                cand_obj=picked_cand_t1,
                parent_electrolyte_input=picked_cand_t1.parent_baseline_electrolyte or parent_electrolyte_input,
                cur_lang=cur_lang,
                key_prefix="tier1",
            )

    with tier_tabs[1]:
        is_pdf_db = (st.session_state.get("suggested_source") == "pdf_database")
        if is_pdf_db:
            st.markdown(f'<div style="font-size: 0.88rem; font-weight: 600; margin: 4px 0 2px 0; color: inherit;">{tr("sec_suggested_additives_db")}</div>', unsafe_allow_html=True)
            st.caption(tr("suggested_additive_notice_db"))
        else:
            st.caption(tr("tier2_caption"))
        if not tier2_candidates:
            st.session_state.pop("tab2_suggested_additives_dropdown", None)
            st.session_state.pop("step2_tier2_bottom_dropdown", None)
            st.info(tr("no_tier2_additives_info"))
            st.caption(f"💡 {tr('tier2_guidance_upload_in_tier1')}")
        else:
            # 1. Quick Dropdown Menu (Prominent & Visible by Default at Top - Streamlined without redundant button)
            cur_active_t2 = st.session_state.get("selected_tier2_name", "")
            cur_t2_idx = 0
            if cur_active_t2:
                for idx, c in enumerate(tier2_candidates):
                    if c.name.lower() == cur_active_t2.lower():
                        cur_t2_idx = idx
                        break
            elif st.session_state.get("selected_suggested_idx", 0) < len(tier2_candidates):
                cur_t2_idx = st.session_state.get("selected_suggested_idx", 0)

            def _format_tier2_option(idx: int) -> str:
                c = tier2_candidates[idx]
                f_str = format_chem_formula(c.chemical_formula or get_chemical_formula(c.smiles, c.name))
                f_disp = f" ({f_str})" if f_str else ""
                sim = c.similarity_score if c.similarity_score is not None else 0.5
                met = c.similarity_metric or "Tanimoto"
                return f"{c.name}{f_disp} — Sim: {sim:.2f} [{met}]"

            if "pending_tier2_choice" in st.session_state:
                st.session_state["tab2_suggested_additives_dropdown"] = st.session_state.pop("pending_tier2_choice")
            elif "tab2_suggested_additives_dropdown" not in st.session_state:
                st.session_state["tab2_suggested_additives_dropdown"] = cur_t2_idx
            elif isinstance(st.session_state["tab2_suggested_additives_dropdown"], int) and not (0 <= st.session_state["tab2_suggested_additives_dropdown"] < len(tier2_candidates)):
                st.session_state["tab2_suggested_additives_dropdown"] = cur_t2_idx
            elif st.session_state.get("_lang_changed_this_run"):
                st.session_state["tab2_suggested_additives_dropdown"] = st.session_state["tab2_suggested_additives_dropdown"]

            t2_choice = st.selectbox(
                tr("select_tier2_dropdown_label"),
                range(len(tier2_candidates)),
                format_func=_format_tier2_option,
                index=None,
                key="tab2_suggested_additives_dropdown",
                help="Select an AI-suggested similar molecule for formulation planning"
            )

            if t2_choice is None:
                _remove_top_tier2_candidate(st.session_state.get("last_suggested_choice", cur_t2_idx))
                return

            prev_t2_choice = st.session_state.get("last_suggested_choice", cur_t2_idx)
            picked_cand = tier2_candidates[t2_choice]
            t2_changed = (prev_t2_choice != t2_choice and "last_suggested_choice" in st.session_state)
            active_add_lower_t2 = str(st.session_state.get("selected_additive_name") or "").strip().lower()
            t2_needs_active_sync = (
                st.session_state.get("active_step2_tier_tab") == "tier2"
                and (
                    "last_suggested_choice" not in st.session_state
                    or not active_add_lower_t2
                    or active_add_lower_t2 != picked_cand.name.strip().lower()
                )
            )
            if not _switched_to_t1 and (t2_changed or _switched_to_t2 or t2_needs_active_sync):
                st.session_state["active_step2_tier_tab"] = "tier2"
                st.session_state["last_suggested_choice"] = t2_choice
                st.session_state["pending_tier2_bottom_choice"] = t2_choice
                st.session_state["selected_suggested_idx"] = t2_choice
                st.session_state["step2_tier2_current_page"] = (t2_choice // 6) + 1
                st.session_state["selected_tier2_name"] = picked_cand.name
                st.session_state["is_active_baseline_unspecified"] = (
                    picked_cand.reported_concentration is None or
                    not picked_cand.parent_baseline_electrolyte or
                    "unspecified" in str(picked_cand.parent_baseline_electrolyte).lower() or
                    "not reported" in str(picked_cand.parent_baseline_electrolyte).lower()
                )
                _apply_card_candidate(
                    cand=picked_cand,
                    selected_target_salt=selected_target_salt,
                    doe_method=doe_method,
                    num_doe_samples=num_doe_samples,
                    rack_format=rack_format,
                    total_vol_ul=total_vol_ul,
                    published_conc_input=published_conc_input,
                    parent_electrolyte_input=parent_electrolyte_input,
                    show_toast=not (_switched_to_t2 or t2_needs_active_sync),
                )
            else:
                st.session_state["last_suggested_choice"] = t2_choice
                if tier2_candidates:
                    st.session_state["selected_tier2_name"] = tier2_candidates[t2_choice].name

            # 2. Detailed Card Deck (Collapsed by default in an Expander, stays open across dropdown changes)
            with st.expander(
                f"🃏 {tr('view_card_deck_label')} ({len(tier2_candidates)} {tr('candidates_label')})",
                expanded=False,
                key="step2_tier2_card_deck_expander",
            ):
                PAGE_SIZE = 6
                total_t2 = len(tier2_candidates)
                total_pages_t2 = max(1, (total_t2 + PAGE_SIZE - 1) // PAGE_SIZE)
                curr_page_t2 = st.session_state.get("step2_tier2_current_page", 1)
                if curr_page_t2 > total_pages_t2:
                    curr_page_t2 = total_pages_t2
                    st.session_state["step2_tier2_current_page"] = curr_page_t2
                if curr_page_t2 < 1:
                    curr_page_t2 = 1
                    st.session_state["step2_tier2_current_page"] = curr_page_t2

                start_t2 = (curr_page_t2 - 1) * PAGE_SIZE
                end_t2 = min(total_t2, curr_page_t2 * PAGE_SIZE)

                col_p2, col_i2, col_n2 = st.columns([1.2, 2.6, 1.2], vertical_alignment="center")
                with col_p2:
                    if st.button(f"◀ {tr('pagination_prev')}", key="tier2_prev_page", disabled=(curr_page_t2 <= 1), use_container_width=True):
                        st.session_state["step2_tier2_current_page"] = curr_page_t2 - 1
                        st.rerun()
                with col_i2:
                    st.markdown(
                        f'<div style="text-align: center; font-weight: 600; font-size: 0.90rem; color: inherit; padding: 4px 0;">'
                        f'{tr("pagination_page_info", page=curr_page_t2, total_pages=total_pages_t2, start=start_t2 + 1, end=end_t2, total_items=total_t2)}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                with col_n2:
                    if st.button(f"{tr('pagination_next')} ▶", key="tier2_next_page", disabled=(curr_page_t2 >= total_pages_t2), use_container_width=True):
                        st.session_state["step2_tier2_current_page"] = curr_page_t2 + 1
                        st.rerun()

                st.markdown('<div style="margin-bottom: 8px;"></div>', unsafe_allow_html=True)
                col_grid2 = st.columns(2)
                for i, idx in enumerate(range(start_t2, end_t2)):
                    cand = tier2_candidates[idx]
                    with col_grid2[i % 2]:
                        cand_name = cand.name
                        cand_smiles = cand.smiles
                        cand_formula = cand.chemical_formula or get_chemical_formula(cand_smiles, cand_name)
                        fmt_formula = format_chem_formula(cand_formula) if cand_formula else ""
                        formula_disp = f' <span style="font-family: monospace; font-size: 0.88rem; color: #1f77b4;">({fmt_formula})</span>' if fmt_formula else ""
                        cur_active_name = st.session_state.get("selected_additive_name", "")
                        is_active = bool(cand_name and cur_active_name and cand_name.lower() == cur_active_name.lower())
                        cand_has_esol, _ = is_esol_applicable(cand_smiles, cand.logs)
                        if cand_has_esol and cand.logs is not None:
                            theor_sat_mm = round((10 ** cand.logs) * 1000.0, 1)
                            safety_cap_mm = round(0.80 * theor_sat_mm, 1)
                            esol_t2_disp = format_solubility_conc(theor_sat_mm)
                            safety_t2_disp = f'<span style="color: #2ca02c; font-weight: 600;">{format_solubility_conc(safety_cap_mm)}</span>'
                        else:
                            esol_t2_disp = "<em>N/A</em>"
                            safety_t2_disp = "<em>N/A</em>"
                        sim_score = cand.similarity_score if cand.similarity_score is not None else 0.5
                        source_lbl = "PubChem Live Search" if "PubChem" in cand.similarity_metric else "Curated Scaffolds"
                        compat_note = cand.electrolyte_compatibility_note or "Compatible across zinc salt systems"

                        active_badge = f'<span class="badge-tag badge-green">✅ {tr("card_active_badge")}</span>' if is_active else f'<span class="badge-tag badge-purple">✨ {tr("tier_untested_badge")}</span>'

                        card_html = f"""
                        <div class="paper-analysis-box candidate-card-deck-box" style="border-left: 4px solid {'#2ecc71' if is_active else '#9b59b6'};">
                            <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; margin-bottom: 4px;">
                                <div class="candidate-card-clamp" style="font-size: 1.0rem; font-weight: 700; color: inherit; min-width: 0; flex: 1 1 auto; overflow-wrap: break-word;">
                                    🧪 <strong>{cand_name}</strong>{formula_disp}
                                </div>
                                <div style="display: flex; gap: 4px; align-items: center; flex-shrink: 0; white-space: nowrap;">
                                    {active_badge}
                                </div>
                            </div>
                            <div style="font-size: 0.86rem; color: inherit; margin-bottom: 2px;">
                                • <strong>{tr('card_tanimoto_similarity')}:</strong> <span class="badge-tag badge-purple">{sim_score:.2f}</span> | <strong>{tr('card_source')}:</strong> <span class="badge-tag badge-blue">{source_lbl}</span>
                            </div>
                            <div style="font-size: 0.86rem; color: inherit; margin-bottom: 2px;">
                                • <strong>{tr('card_delaney_esol')}:</strong> {esol_t2_disp} | <strong>{tr('card_safety_cap')}:</strong> {safety_t2_disp}
                            </div>
                            <div class="candidate-card-clamp" style="font-size: 0.82rem; color: inherit;">
                                • <strong>{tr('card_compat')}:</strong> <em>{clean_scientific_text(compat_note)}</em>
                            </div>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)

                        if st.button(
                            f"🧪 {tr('btn_use_this_additive')}: {cand_name}",
                            key=f"btn_formulate_t2_{idx}",
                            type="primary" if not is_active else "secondary",
                            use_container_width=True,
                            help=f"Select {cand_name} for formulation planning"
                        ):
                            st.session_state["active_step2_tier_tab"] = "tier2"
                            st.session_state["selected_tier2_name"] = cand.name
                            st.session_state["selected_suggested_idx"] = idx
                            st.session_state["last_suggested_choice"] = idx
                            st.session_state["last_bottom_tier2_choice"] = idx
                            st.session_state["pending_tier2_choice"] = idx
                            st.session_state["pending_tier2_bottom_choice"] = idx
                            st.session_state["step2_tier2_current_page"] = (idx // 6) + 1
                            st.session_state["is_active_baseline_unspecified"] = (
                                not cand.parent_baseline_electrolyte or
                                "unspecified" in str(cand.parent_baseline_electrolyte).lower() or
                                "not reported" in str(cand.parent_baseline_electrolyte).lower()
                            )
                            _apply_card_candidate(
                                cand=cand,
                                selected_target_salt=selected_target_salt,
                                doe_method=doe_method,
                                num_doe_samples=num_doe_samples,
                                rack_format=rack_format,
                                total_vol_ul=total_vol_ul,
                                published_conc_input=published_conc_input,
                                parent_electrolyte_input=parent_electrolyte_input,
                            )
                            st.rerun()

        # Candidate Chemical Dossier & Analysis for Selected Tier 2 Additive
        if tier2_candidates:
            picked_cand_t2 = tier2_candidates[t2_choice]
            _render_candidate_chemical_dossier(
                cand_name=picked_cand_t2.name,
                cand_smiles=picked_cand_t2.smiles or "",
                is_tier2=True,
                cand_obj=picked_cand_t2,
                parent_electrolyte_input=picked_cand_t2.parent_baseline_electrolyte or parent_electrolyte_input,
                cur_lang=cur_lang,
                key_prefix="tier2",
            )

    # Emit client-side sub-tab synchronization and expander state persistence AFTER both tier tabs
    # have updated st.session_state so _mapServerTierState always reflects the latest server state.
    _active_tier_tab_state = st.session_state.get("active_step2_tier_tab", "tier1")
    _cur_active_add_name = (st.session_state.get("selected_additive_name") or "").strip().lower()
    _cur_t1_sel_name = (st.session_state.get("selected_tier1_name") or "").strip().lower()
    if not _cur_t1_sel_name and tier1_candidates:
        _cur_t1_sel_name = tier1_candidates[0].name.strip().lower()
    _cur_t2_sel_name = (st.session_state.get("selected_tier2_name") or "").strip().lower()
    if not _cur_t2_sel_name and tier2_candidates:
        _cur_t2_sel_name = tier2_candidates[0].name.strip().lower()

    _sync_marker = (
        f"{st.session_state.get('last_tier1_choice')}_"
        f"{st.session_state.get('step2_tier1_current_page', 1)}_"
        f"{st.session_state.get('last_suggested_choice')}_"
        f"{st.session_state.get('step2_tier2_current_page', 1)}_"
        f"{len(tier1_candidates)}_{len(tier2_candidates)}_"
        f"{_active_tier_tab_state}_{_cur_active_add_name}_{_cur_t1_sel_name}_{_cur_t2_sel_name}"
    )
    st.components.v1.html(
        f"""
        <!-- card-deck-sync: {_sync_marker} -->
        <script>
        (function() {{
            const parentWin = window.parent;
            const parentDoc = parentWin.document;
            if (!parentWin._mapCardDeckOpenState) {{
                parentWin._mapCardDeckOpenState = {{ tier1: false, tier2: false }};
            }}
            parentWin._mapServerTierState = {{
                activeTab: "{_active_tier_tab_state}",
                activeAdditive: {_cur_active_add_name!r},
                tier1Name: {_cur_t1_sel_name!r},
                tier2Name: {_cur_t2_sel_name!r}
            }};

            function syncCardDeckExpanders() {{
                if (parentWin._mapTierTabsClickListener) {{
                    parentDoc.removeEventListener('click', parentWin._mapTierTabsClickListener, true);
                    parentWin._mapTierTabsClickListener = null;
                }}
                const slots = [
                    {{ key: 'tier1', selector: '.st-key-step2_tier1_card_deck_expander details' }},
                    {{ key: 'tier2', selector: '.st-key-step2_tier2_card_deck_expander details' }}
                ];
                slots.forEach(({{ key, selector }}) => {{
                    const details = parentDoc.querySelector(selector);
                    if (!details) return;
                    const summary = details.querySelector('summary');
                    if (!summary) return;

                    if (!details.dataset.mapExpanderBound) {{
                        details.dataset.mapExpanderBound = "true";

                        summary.addEventListener('pointerdown', function() {{
                            details.dataset.userToggling = "true";
                        }}, true);
                        summary.addEventListener('keydown', function(e) {{
                            if (e.key === 'Enter' || e.key === ' ') {{
                                details.dataset.userToggling = "true";
                            }}
                        }}, true);

                        details.addEventListener('toggle', function() {{
                            if (details.dataset.restoringOpen === "true") {{
                                details.dataset.restoringOpen = "false";
                                return;
                            }}
                            if (details.dataset.userToggling === "true") {{
                                details.dataset.userToggling = "false";
                                parentWin._mapCardDeckOpenState[key] = details.open;
                            }} else if (!details.open && parentWin._mapCardDeckOpenState[key]) {{
                                details.dataset.restoringOpen = "true";
                                summary.click();
                            }}
                        }});
                    }}

                    if (parentWin._mapCardDeckOpenState[key] && !details.open) {{
                        details.dataset.restoringOpen = "true";
                        summary.click();
                    }}
                }});
            }}

            parentWin._syncCardDeckExpanders = syncCardDeckExpanders;
            syncCardDeckExpanders();
            setTimeout(syncCardDeckExpanders, 50);
            setTimeout(syncCardDeckExpanders, 150);

            if (parentWin._mapCardDeckObserver) {{
                try {{ parentWin._mapCardDeckObserver.disconnect(); }} catch (e) {{}}
            }}
            const observer = new MutationObserver(function() {{
                if (parentWin._syncCardDeckExpanders) {{
                    parentWin._syncCardDeckExpanders();
                }}
            }});
            observer.observe(parentDoc.body, {{ childList: true, subtree: true, attributes: true, attributeFilter: ['open', 'aria-selected'] }});
            parentWin._mapCardDeckObserver = observer;
        }})();
        </script>
        """,
        height=0,
        width=0,
    )

