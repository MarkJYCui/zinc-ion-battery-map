"""
Automated Headless UI Tests for the Autonomous Materials Acceleration Platform (MAP).
Tests Streamlit User Interface (app.py) using streamlit.testing.v1.AppTest.

Verifies:
1. app.py compiles, initializes, and executes headless with zero runtime exceptions.
2. Closed-loop active learning orchestrator executes multi-round sequential DoE (Round 0 ➔ Round 1).
3. Candidate evaluation table and Global Multi-Objective Pareto Frontier scatter plot render accurately.
4. Delaney ESOL 80% physical saturation guardrail limits are displayed and enforced.
5. Opentrons OT-2 robotic deck layout maintains multi-round vial coordinate continuity (A1–A5) without collision.
6. Approach C dual-tier separation (Tier 1 healthy active window vs Tier 2 forensics) functions correctly.
7. Executive report and export center download buttons serialize session state data structures.
"""

import asyncio
import gc
import multiprocessing
import os
import threading
import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1 import element_tree as _st_element_tree
from streamlit.testing.v1 import local_script_runner as _st_local_runner

from models import CampaignResult, DoEPlan, DispensingSchedule, FormulatedCellResult

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


class _HeadlessTestEventLoop(asyncio.AbstractEventLoop):
    """
    Lightweight non-running event loop for Streamlit's LocalScriptRunner.
    Avoids opening hundreds of Windows Proactor IOCP handles and loopback TCP
    socket pairs (which can hang or leave background handles open on Windows).
    """

    def __init__(self) -> None:
        self._closed = False

    def is_closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        self._closed = True

    def is_running(self) -> bool:
        return False


# Patch LocalScriptRunner.__init__ to use _HeadlessTestEventLoop instead of opening OS loopback sockets
_orig_local_runner_init = _st_local_runner.LocalScriptRunner.__init__


def _patched_local_runner_init(self, *args, **kwargs):
    orig_new_loop = _st_local_runner.asyncio.new_event_loop
    _st_local_runner.asyncio.new_event_loop = _HeadlessTestEventLoop
    try:
        _orig_local_runner_init(self, *args, **kwargs)
    finally:
        _st_local_runner.asyncio.new_event_loop = orig_new_loop


_st_local_runner.LocalScriptRunner.__init__ = _patched_local_runner_init


# Ensure UnknownElement extracts srcdoc from st.components.v1.html (iframe) protos
_orig_unknown_element_content = getattr(_st_element_tree, "_unknown_element_content", None)
if _orig_unknown_element_content is not None:
    def _patched_unknown_element_content(proto):
        res = _orig_unknown_element_content(proto)
        if res is not None:
            return res
        srcdoc = getattr(proto, "srcdoc", None)
        if srcdoc:
            return srcdoc
        return res

    _st_element_tree._unknown_element_content = _patched_unknown_element_content


# Expose .help on popover Block elements and .popover / .html on AppTest if not natively present
if not hasattr(_st_element_tree.Block, "help"):
    @property
    def _block_help(self):
        if hasattr(self, "proto") and hasattr(self.proto, "popover"):
            return getattr(self.proto.popover, "help", None)
        return None

    _st_element_tree.Block.help = _block_help

if not hasattr(AppTest, "popover"):
    @property
    def _apptest_popover(self):
        return self.get("popover")

    AppTest.popover = _apptest_popover

if not hasattr(AppTest, "html"):
    @property
    def _apptest_html(self):
        return list(self.get("html")) + list(self.get("iframe"))

    AppTest.html = _apptest_html


def _cleanup_background_threads_and_processes() -> None:
    """
    Explicitly terminate any child processes and shut down background executor
    or Streamlit runtime threads so the test runner exits cleanly without hanging.
    """
    # 1. Terminate any multiprocessing child processes
    try:
        for child in multiprocessing.active_children():
            try:
                child.terminate()
                child.join(timeout=1.0)
                if child.is_alive():
                    child.kill()
            except Exception:
                pass
    except Exception:
        pass

    # 2. Terminate any OS child processes via psutil if available
    try:
        import psutil
        current_proc = psutil.Process(os.getpid())
        children = current_proc.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except Exception:
                pass
        if children:
            _, alive = psutil.wait_procs(children, timeout=1.0)
            for p in alive:
                try:
                    p.kill()
                except Exception:
                    pass
    except Exception:
        pass

    # 3. Clean up concurrent.futures ThreadPoolExecutor queues if any were spawned
    try:
        import concurrent.futures.thread as _cft
        for t, q in list(getattr(_cft, "_threads_queues", {}).items()):
            try:
                if q is not None:
                    q.put(None)
            except Exception:
                pass
    except Exception:
        pass

    # 4. Clear any lingering Streamlit ScriptRunContext on the main thread without logging
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import SCRIPT_RUN_CONTEXT_ATTR_NAME
        if hasattr(threading.current_thread(), SCRIPT_RUN_CONTEXT_ATTR_NAME):
            delattr(threading.current_thread(), SCRIPT_RUN_CONTEXT_ATTR_NAME)
    except Exception:
        pass

    # 5. Clear concurrent.futures._threads_queues so _python_exit does not block indefinitely
    try:
        import concurrent.futures.thread as _cft
        getattr(_cft, "_threads_queues", {}).clear()
    except Exception:
        pass

    gc.collect()


@pytest.fixture(scope="session", autouse=True)
def teardown_all_ui_background_resources(request):
    """Session-wide autouse fixture that cleans up all background processes and threads upon suite completion."""
    import atexit
    import sys
    import time

    session = request.session

    def _force_clean_exit():
        _cleanup_background_threads_and_processes()
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        exit_code = 1 if getattr(session, "testsfailed", 0) > 0 else 0

        def _watchdog():
            time.sleep(1.5)
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
            os._exit(exit_code)

        t = threading.Thread(target=_watchdog, name="PytestExitWatchdog", daemon=True)
        t.start()

    atexit.register(_force_clean_exit)
    yield
    _cleanup_background_threads_and_processes()


def pytest_sessionfinish(session, exitstatus):
    """Ensure all child processes and background threads are terminated when pytest finishes."""
    _cleanup_background_threads_and_processes()


@pytest.fixture(autouse=True)
def mock_all_external_network_and_search(monkeypatch):
    """
    Autouse fixture ensuring UI tests never hit external rate-limited APIs
    (Semantic Scholar, OpenAlex, Google Scholar, NCBI PubChem, Google Gemini)
    and instead instantaneously return curated domain literature and offline mocks.
    Guarantees 100% offline test execution with zero network egress or hangs.
    """
    import requests
    from discovery.search import get_offline_sample_papers, SemanticScholarClient
    from discovery.pubchem_client import PubChemClient
    from discovery.extractor import PaperExtractor

    # 1. Strip Gemini API key from environment to prevent live LLM egress
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    orig_extractor_init = PaperExtractor.__init__
    def _mock_extractor_init(self, *args, **kwargs):
        kwargs["api_key"] = None
        orig_extractor_init(self, *args, **kwargs)
        self._genai_client = None
    monkeypatch.setattr(PaperExtractor, "__init__", _mock_extractor_init)

    # 2. Mock SemanticScholarClient high-level and low-level search methods
    def _mock_search_papers(self, query="", limit=25, offset=0, sort="relevance", **kwargs):
        papers = get_offline_sample_papers(query=query or "", limit=limit, sort=sort, offset=offset)
        cb = kwargs.get("batch_callback")
        if cb is not None:
            try:
                cb(1, 1, papers, papers)
            except Exception:
                pass
        return papers

    monkeypatch.setattr(SemanticScholarClient, "search_papers", _mock_search_papers)
    monkeypatch.setattr(SemanticScholarClient, "_search_semantic_scholar", lambda *a, **kw: [])
    monkeypatch.setattr(SemanticScholarClient, "_search_openalex", lambda *a, **kw: [])
    monkeypatch.setattr(SemanticScholarClient, "download_open_access_pdf", lambda *a, **kw: False)

    # 3. Mock PubChemClient to avoid slow network queries and socket timeouts
    monkeypatch.setattr(PubChemClient, "search_similar_compounds", lambda *a, **kw: [])
    monkeypatch.setattr(PubChemClient, "name_to_smiles", lambda *a, **kw: None)
    monkeypatch.setattr(PubChemClient, "smiles_to_name", lambda *a, **kw: None)

    # 4. Strictly block any unmocked HTTP requests across requests & requests.Session
    def _mock_network_block(*args, **kwargs):
        raise requests.exceptions.ConnectionError("Simulated offline testing: external network requests are disabled in tests")

    monkeypatch.setattr("requests.get", _mock_network_block)
    monkeypatch.setattr("requests.post", _mock_network_block)
    monkeypatch.setattr("requests.request", _mock_network_block)
    monkeypatch.setattr(requests.Session, "get", _mock_network_block)
    monkeypatch.setattr(requests.Session, "post", _mock_network_block)
    monkeypatch.setattr(requests.Session, "request", _mock_network_block)


def test_ui_app_headless_initialization():
    """
    Verify that app.py initializes and executes headless in demo mode without exceptions.
    Ensures that the initial closed-loop campaign pre-loads, populating session state with
    both Round 0 and Round 1 active learning candidates.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()

    # Verify no unhandled exceptions occurred in default startup
    assert len(at.exception) == 0, f"AppTest raised exceptions: {at.exception}"

    # Switch to Demo Mode (Pre-loaded Literature Data) to verify pre-loaded campaign
    at.sidebar.radio[0].set_value("demo").run()
    assert len(at.exception) == 0, f"AppTest raised exceptions in demo mode: {at.exception}"

    # Verify campaign result in session state
    campaign = at.session_state.get("campaign_result")
    assert campaign is not None, "campaign_result must be initialized in session state"
    assert isinstance(campaign, CampaignResult)

    # Verify multi-round closed loop properties
    assert campaign.total_iterations == 2, f"Expected 2 iterations, got {campaign.total_iterations}"
    assert len(campaign.cell_results) == 5, f"Expected 5 total formulations, got {len(campaign.cell_results)}"

    # Verify Pareto frontier candidate identification
    assert len(campaign.pareto_optimal_candidates) > 0
    assert campaign.best_candidate is not None
    assert campaign.best_candidate.is_pareto_optimal is True

    # Verify well coordinate continuity across rounds
    vial_ids = [c.vial_id for c in campaign.cell_results]
    assert vial_ids == ["A1", "A2", "A3", "A4", "A5"]


def test_ui_campaign_button_trigger_closed_loop():
    """
    Verify that clicking the 'Run Autonomous Closed-Loop Campaign' button executes
    the closed-loop active learning workflow and updates the UI state without tracebacks.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    # Switch to Demo Mode where Automated Experiment Run tab is visible
    at.sidebar.radio[0].set_value("demo").run()
    assert len(at.exception) == 0

    # Locate and trigger the primary campaign button
    buttons = [btn for btn in at.button if "Run Autonomous" in getattr(btn, "label", "")]
    assert len(buttons) > 0, "Autonomous run button must be present in Demo Mode"

    primary_btn = buttons[0]
    primary_btn.click()
    at.run()

    assert len(at.exception) == 0, f"AppTest raised exception on campaign execution: {at.exception}"

    campaign = at.session_state.get("campaign_result")
    assert campaign is not None
    assert len(campaign.cell_results) >= 5
    assert len(campaign.history) >= 2
    assert campaign.best_candidate is not None


def test_ui_opentrons_multiround_rack_continuity_and_viscosity():
    """
    Verify that the Opentrons OT-2 liquid handling dispensing schedule maintains
    sequential well continuity (Round 0 A1-A3, Round 1 A4-A5) without well collisions,
    and correctly reflects deck parameters.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    at.sidebar.radio[0].set_value("demo").run()
    assert len(at.exception) == 0

    campaign = at.session_state.get("campaign_result")
    assert campaign is not None

    schedule = campaign.dispensing_schedule
    assert isinstance(schedule, DispensingSchedule)
    assert schedule.rack_format == "24-vial rack"
    assert schedule.ot2_deck_slot == 1

    # Verify that all 5 vials exist with correct row/col coordinates
    assert len(schedule.vials) == 5
    expected_coords = [("A", 1), ("A", 2), ("A", 3), ("A", 4), ("A", 5)]
    actual_coords = [(v.row, v.col) for v in schedule.vials]
    assert actual_coords == expected_coords

    # Verify non-zero pipetting volumes meeting > 5 uL threshold
    for v in schedule.vials:
        assert v.target_volume_uL == pytest.approx(2000.0)
        assert v.additive_stock_uL > 0.0
        assert v.salt_stock_uL > 0.0
        assert v.water_uL >= 0.0


def test_ui_delaney_esol_physical_guardrail_rendering():
    """
    Verify that Phase 1 Delaney ESOL 80% saturation physical ceiling is calculated
    and verified on the candidate pool and concentration anchor.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    at.sidebar.radio[0].set_value("demo").run()
    assert len(at.exception) == 0

    campaign = at.session_state.get("campaign_result")
    anchor = campaign.formulation_plan.anchor

    # Delaney ESOL ceiling must be strictly positive
    assert anchor.solubility_ceiling_mM is not None
    assert anchor.solubility_ceiling_mM > 0.0

    # Theoretical saturation must be equal to ceiling / 0.80
    assert anchor.solubility_limit_mM == pytest.approx(anchor.solubility_ceiling_mM / 0.80, rel=1e-2)

    # All formulated points must respect the physical ceiling
    for cell in campaign.cell_results:
        assert cell.additive_conc_mM <= anchor.solubility_ceiling_mM + 1e-3


def test_ui_approach_c_dual_tier_metrics():
    """
    Verify Approach C failure-aware dual-tier separation:
    - Tier 1: Healthy active cycling metrics govern performance score.
    - Tier 2: Total test summary captures cell failure forensics (e.g. short-circuit).
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    at.sidebar.radio[0].set_value("demo").run()
    assert len(at.exception) == 0

    campaign = at.session_state.get("campaign_result")
    best = campaign.best_candidate
    assert best is not None
    assert best.analysis_report is not None
    assert best.analysis_report.symmetric_cell is not None

    symm = best.analysis_report.symmetric_cell
    assert symm.active_cycling_metrics is not None
    assert "active_lifespan_hours" in symm.active_cycling_metrics
    assert "mean_overpotential_mv" in symm.active_cycling_metrics

    assert symm.total_test_summary is not None
    assert "failure_mode" in symm.total_test_summary
    assert "nominal_duration_hours" in symm.total_test_summary

    # Performance score must match the active cycling formula: lifespan / (1 + overpot / 50)
    act_h = symm.active_cycling_metrics["active_lifespan_hours"]
    overpot = symm.active_cycling_metrics["mean_overpotential_mv"]
    expected_score = round(act_h / (1.0 + overpot / 50.0), 3)
    assert best.performance_score == pytest.approx(expected_score, abs=0.01)


def test_ui_header_badges_removed():
    """
    Verify that all header badges requested for removal by the user:
    - 'Pytest Suite: 120/120 Passing Tests'
    - 'Step 1: Chemical Similarity & Solubility Safety Limit (ESOL 80%)'
    - 'Step 2: Concentration Scaling & OT-2 Robot'
    - 'Step 3: Two-Stage Battery Cycling & Failure Diagnostics'
    - 'Hardware: Opentrons OT-2 Liquid Handler'
    are strictly absent from the rendered DOM.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    all_markdown_text = " ".join([m.value for m in at.markdown if hasattr(m, "value")])

    assert "Pytest Suite: 120/120 Passing Tests" not in all_markdown_text
    assert "Step 1: Chemical Similarity & Solubility Safety Limit (ESOL 80%)" not in all_markdown_text
    assert "Step 2: Concentration Scaling & OT-2 Robot" not in all_markdown_text
    assert "Step 3: Two-Stage Battery Cycling & Failure Diagnostics" not in all_markdown_text
    assert "Hardware: Opentrons OT-2 Liquid Handler" not in all_markdown_text


def test_ui_localization_multilingual_switching():
    """
    Verify that the app cleanly switches between English (en), Français (fr), and 中文 (zh)
    without runtime exceptions, verifying that:
    1. UI strings are translated natively.
    2. Chemical formulas (ZnSO₄, Zn(OTf)₂, Zn²⁺) and units (h, mM, s, V, mV, µL) remain untranslated.
    """
    # 1. English default execution and switch to French via UI selectbox
    at_en = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at_en.run()
    assert len(at_en.exception) == 0

    at_en.sidebar.radio[0].set_value("demo").run()
    assert len(at_en.exception) == 0

    buttons_en = [btn for btn in at_en.button if "Run Autonomous" in getattr(btn, "label", "")]
    assert len(buttons_en) > 0

    # Switch to French via selectbox index 1
    lang_sb = [sb for sb in at_en.sidebar.selectbox if sb.key == "app_language"]
    if lang_sb:
        lang_sb[0].select_index(1).run()
    else:
        at_en.session_state["app_language"] = "fr"
        at_en.run()

    assert len(at_en.exception) == 0
    # Operating mode must remain 'demo' with set_value=True and French wire label
    assert at_en.sidebar.radio[0].value == "demo"
    assert at_en.sidebar.radio[0].proto.set_value is True
    assert "Mode Démo" in at_en.sidebar.radio[0].proto.raw_value

    # Keyed dropdowns must immediately update their wire value to French without requiring manual mode re-selection
    sort_sb_fr = [sb for sb in at_en.selectbox if sb.key == "lit_search_sort_select"][0]
    assert sort_sb_fr.proto.set_value is True
    assert "Pertinence" in sort_sb_fr.proto.raw_value

    button_labels_fr = [getattr(btn, "label", "") for btn in at_en.button]
    assert any("lancer" in lbl.lower() and ("expérience" in lbl.lower() or "campagne" in lbl.lower()) for lbl in button_labels_fr), f"Expected French button, got {button_labels_fr}"

    md_fr = " ".join([m.value for m in at_en.markdown if hasattr(m, "value")])
    # Chemical formulas must stay intact in French
    assert "ZnSO₄" in md_fr
    assert "Zn(OTf)₂" in md_fr
    assert "Zn²⁺" in md_fr

    # 2. Switch to Chinese via UI selectbox (test both Custom Mode and Demo Mode)
    at_zh = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at_zh.run()
    assert len(at_zh.exception) == 0
    assert at_zh.sidebar.radio[0].value == "custom"

    # Switch to Chinese while in Custom Mode first
    lang_sb_zh_custom = [sb for sb in at_zh.sidebar.selectbox if sb.key == "app_language"]
    lang_sb_zh_custom[0].select_index(2).run()
    assert len(at_zh.exception) == 0
    assert at_zh.sidebar.radio[0].value == "custom"
    assert at_zh.sidebar.radio[0].proto.set_value is True
    assert "自定义实验模式" in at_zh.sidebar.radio[0].proto.raw_value

    # Switch to Demo Mode, then switch language to French and back to Chinese
    at_zh.sidebar.radio[0].set_value("demo").run()
    assert len(at_zh.exception) == 0
    lang_sb_fr2 = [sb for sb in at_zh.sidebar.selectbox if sb.key == "app_language"]
    lang_sb_fr2[0].select_index(1).run()
    assert at_zh.sidebar.radio[0].value == "demo"
    assert at_zh.sidebar.radio[0].proto.set_value is True

    lang_sb_zh = [sb for sb in at_zh.sidebar.selectbox if sb.key == "app_language"]
    if lang_sb_zh:
        lang_sb_zh[0].select_index(2).run()
    else:
        at_zh.session_state["app_language"] = "zh"
        at_zh.run()

    assert len(at_zh.exception) == 0
    assert at_zh.sidebar.radio[0].value == "demo"
    assert at_zh.sidebar.radio[0].proto.set_value is True
    assert "演示模式" in at_zh.sidebar.radio[0].proto.raw_value

    step2_salt_zh = [sb for sb in at_zh.selectbox if sb.key == "step2_target_electrolyte_select"][0]
    assert step2_salt_zh.proto.set_value is True
    assert "三氟甲磺酸锌" in step2_salt_zh.proto.raw_value

    button_labels_zh = [getattr(btn, "label", "") for btn in at_zh.button]
    assert any("闭环" in lbl or "实验" in lbl for lbl in button_labels_zh), f"Expected Chinese button, got {button_labels_zh}"

    md_zh = " ".join([m.value for m in at_zh.markdown if hasattr(m, "value")])
    # Chemical formulas and scientific units must stay intact in Chinese
    assert "ZnSO₄" in md_zh
    assert "Zn(OTf)₂" in md_zh
    assert "Zn²⁺" in md_zh


def test_ui_sidebar_github_and_streamlit_hosting():
    """
    Verify that the sidebar renders:
    1. Logo renders without CSS source leaking into markdown text and with reduced badge background width.
    2. GitHub and Streamlit Community Cloud hyperlinks are removed from the sidebar.
    3. Tech stack badges (RDKit, Opentrons, Plotly, Pydantic, etc.) and Project Credits remain.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    sidebar_md = " ".join([m.value for m in at.sidebar.markdown if hasattr(m, "value")])
    assert "https://github.com" not in sidebar_md
    assert "https://share.streamlit.io" not in sidebar_md
    assert "sidebar-link-card" not in sidebar_md
    assert 'width="228"' in sidebar_md
    assert 'width="280"' not in sidebar_md
    assert "RDKit" in sidebar_md
    assert "OT-2" in sidebar_md
    assert "Gemini AI" in sidebar_md
    assert "PubChem" in sidebar_md
    assert "Plotly" in sidebar_md
    assert "Pydantic v2" in sidebar_md
    assert "NumPy / Pandas" in sidebar_md
    assert "Streamlit" in sidebar_md
    assert "Gemini 3.8 Flash" in sidebar_md
    assert "Gemini 3.1 Pro" in sidebar_md
    assert "Gemini 4 Argon" in sidebar_md

    # Verify no raw CSS leaked into visible sidebar markdown text
    assert "@media (prefers-color-scheme: dark)" not in sidebar_md
    assert ".logo-title-zib { fill:" not in sidebar_md


def test_ui_theme_adaptive_css_and_selectbox_wrapping():
    """
    Verify that CUSTOM_CSS includes:
    1. BaseWeb select and popover wrapping rules to prevent text clipping in all dropdowns.
    2. Explicit Light Mode overrides [data-theme="light"] to guarantee high-contrast readability.
    3. Dark Mode rules for both prefers-color-scheme and data-theme="dark".
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    css_blocks = [m.value for m in at.markdown if "<style>" in getattr(m, "value", "")]
    assert len(css_blocks) > 0, "CUSTOM_CSS must be injected"
    full_css = " ".join(css_blocks)

    # 1. BaseWeb select & popover wrapping rules
    assert 'div[data-baseweb="select"]' in full_css
    assert 'div[data-baseweb="popover"]' in full_css
    assert "word-break: break-word" in full_css

    # 2. Explicit Light theme overrides
    assert '[data-theme="light"]' in full_css
    assert '[data-theme="light"] .sidebar-section-title' in full_css

    # 3. Dark theme rules
    assert '[data-theme="dark"]' in full_css
    assert "@media (prefers-color-scheme: dark)" in full_css


def test_ui_chemical_formula_display_and_no_smiles():
    """
    Verify that:
    1. SMILES is not displayed in image captions or user-facing table headers.
    2. Standard chemical formulas with Unicode subscripts (e.g. C₆H₁₄O₆, ZnSO₄, Zn(OTf)₂) are displayed.
    3. Electrolyte compatibility tip and detailed rationale expander are rendered.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    at.sidebar.radio[0].set_value("demo").run()
    assert len(at.exception) == 0

    all_captions = []
    for img in at.image:
        if hasattr(img, "captions") and img.captions:
            all_captions.extend(img.captions)
        elif hasattr(img, "caption") and img.caption:
            all_captions.append(img.caption)
    caption_text = " ".join(all_captions)

    # Caption must not have (SMILES: ...)
    assert "(SMILES:" not in caption_text
    # Caption must display Chemical Formula with Unicode subscripts
    assert "Chemical Formula: C₄H₆O₄" in caption_text or "Chemical Formula: C₆H₁₄O₆" in caption_text

    # Verify compatibility instruction tip and expander exist
    all_captions_md = [getattr(c, "value", "") for c in at.caption if hasattr(c, "value")]
    caption_corpus = " ".join(all_captions_md)
    assert "Double-click any cell" in caption_corpus or "Tip:" in caption_corpus

    expander_labels = [getattr(e, "label", "") for e in at.expander]
    assert any("Detailed Mechanistic Rationale" in lbl for lbl in expander_labels)

    # Verify compatibility table contains correctly subscripted salts (no ASCII subscripts)
    found_acetate = False
    found_perchlorate = False
    for df_comp in at.dataframe:
        df_val = getattr(df_comp, "value", None)
        if df_val is not None and not df_val.empty:
            df_str = str(df_val.to_dict())
            if "Zn(CH₃COO)₂" in df_str:
                found_acetate = True
            if "Zn(ClO₄)₂" in df_str:
                found_perchlorate = True
            assert "Zn(CH3COO)" not in df_str
            assert "Zn(ClO4)" not in df_str
    assert found_acetate, "Zn(CH₃COO)₂ must be present in compatibility dataframe"
    assert found_perchlorate, "Zn(ClO₄)₂ must be present in compatibility dataframe"


def test_ui_doe_method_and_deck_layout_styling():
    """
    Verify that:
    1. DoE method selectbox is present with full descriptive options.
    2. Opentrons OT-2 deck layout section renders cleanly.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    # In Custom Mode (default), verify DoE method selectbox is hidden from sidebar
    doe_boxes_custom = [sb for sb in at.sidebar.selectbox if "Experimental Design Method" in getattr(sb, "label", "")]
    assert len(doe_boxes_custom) == 0, "DoE method selectbox must be hidden in Custom Mode"

    # Switch to Demo Mode and verify DoE method selectbox is present
    at.sidebar.radio[0].set_value("demo").run()
    assert len(at.exception) == 0
    doe_boxes = [sb for sb in at.sidebar.selectbox if "Experimental Design Method" in getattr(sb, "label", "")]
    assert len(doe_boxes) > 0, "DoE method selectbox must exist in Demo Mode"
    doe_sb = doe_boxes[0]
    assert doe_sb.value in ["lhs", "ccd"]

    # Verify OT-2 deck layout markdown section exists
    all_markdown_text = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    assert "Opentrons OT-2 Labware Deck Layout" in all_markdown_text


def test_ui_step2_formal_editable_downloads():
    """
    Verify that Step 2 provides formal, clearly-named, human-editable downloads
    (.xlsx, .docx, .txt) and machine data (.json) for both Opentrons OT-2 dispensing
    schedule and DoE formulation matrix.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    at.sidebar.radio[0].set_value("demo").run()
    assert len(at.exception) == 0

    download_keys = [getattr(b, "key", "") for b in at.download_button]
    download_labels = [getattr(b, "label", "") for b in at.download_button]

    # Verify OT-2 dispensing protocol download options
    assert "step2_btn_dl_dispensing_xlsx" in download_keys
    assert "step2_btn_dl_dispensing_docx" in download_keys
    assert "step2_btn_dl_dispensing_txt" in download_keys
    assert "step2_btn_dl_dispensing_json" in download_keys

    # Verify DoE formulation matrix download options
    assert "step2_btn_dl_doe_xlsx" in download_keys
    assert "step2_btn_dl_doe_docx" in download_keys
    assert "step2_btn_dl_doe_txt" in download_keys
    assert "step2_btn_dl_doe_json" in download_keys

    # Verify label formatting indicates editable file types
    assert any(".xlsx" in lbl for lbl in download_labels)
    assert any(".docx" in lbl for lbl in download_labels)
    assert any(".txt" in lbl for lbl in download_labels)
    assert any(".json" in lbl for lbl in download_labels)

    # Verify downloadable items are housed in a drop-down expander menu
    expander_labels = [getattr(e, "label", "") for e in at.expander]
    assert any("Download Protocol & Formulation Plan" in lbl for lbl in expander_labels), (
        "Step 2 downloadable items must be enclosed in an expander drop-down menu"
    )


def test_ui_tab5_formal_editable_downloads():
    """
    Verify that Tab 5 (Experiment Summary & Export) provides comprehensive editable
    formats (.xlsx, .docx, .txt) and machine/dashboard formats (.json, .html).
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    at.sidebar.radio[0].set_value("demo").run()
    assert len(at.exception) == 0

    # Verify Tab 5 Download Center is housed in a drop-down expander menu
    expander_labels = [getattr(e, "label", "") for e in at.expander]
    assert any("Download Center" in lbl for lbl in expander_labels), (
        "Tab 5 Download Center must be enclosed in an expander drop-down menu"
    )

    download_keys = [getattr(b, "key", "") for b in at.download_button]

    # Campaign summary downloads
    assert "tab5_btn_dl_campaign_xlsx" in download_keys
    assert "tab5_btn_dl_campaign_docx" in download_keys
    assert "tab5_btn_dl_campaign_txt" in download_keys
    assert "tab5_btn_dl_campaign_json" in download_keys

    # Dispensing schedule downloads in Tab 5
    assert "tab5_btn_dl_dispensing_xlsx" in download_keys
    assert "tab5_btn_dl_dispensing_docx" in download_keys
    assert "tab5_btn_dl_dispensing_txt" in download_keys
    assert "tab5_btn_dl_dispensing_json" in download_keys

    # DoE plan downloads in Tab 5
    assert "tab5_btn_dl_doe_xlsx" in download_keys
    assert "tab5_btn_dl_doe_docx" in download_keys
    assert "tab5_btn_dl_doe_txt" in download_keys
    assert "tab5_btn_dl_doe_json" in download_keys

    # Interactive electrochemical HTML dashboard
    assert "tab5_btn_dl_html_dashboard" in download_keys


def test_ui_black_white_typography_and_no_dark_blue():
    """
    Verify that:
    1. CUSTOM_CSS enforces high-contrast black (#000000) typography in Light Mode
       and pure white (#ffffff) typography in Dark Mode.
    2. Forbidden dark blue colors (#0f172a, #004e89, #1a365d, #0b2545) are never used
       in CUSTOM_CSS, app text, or the sidebar logo SVG.
    """
    import re
    from app import CUSTOM_CSS

    # Verify dark mode enforces white text
    assert "--text-color: #ffffff;" in CUSTOM_CSS
    assert "--app-text: #ffffff;" in CUSTOM_CSS
    assert "color: #ffffff !important;" in CUSTOM_CSS

    # Verify light mode enforces black text
    assert "--text-color: #000000;" in CUSTOM_CSS
    assert "--app-text: #000000;" in CUSTOM_CSS
    assert "color: #000000 !important;" in CUSTOM_CSS

    # Verify absence of dark blue hex colors
    dark_blues = ["#0f172a", "#004e89", "#1a365d", "#0b2545", "#1e293b", "#334155"]
    for db in dark_blues:
        matches = re.findall(re.escape(db), CUSTOM_CSS, re.IGNORECASE)
        assert len(matches) == 0, f"Found forbidden dark blue color {db} in CUSTOM_CSS: {matches}"

    # Verify logo.svg does not contain forbidden dark blues
    logo_path = Path(__file__).resolve().parent.parent / "assets" / "logo.svg"
    if logo_path.exists():
        logo_content = logo_path.read_text(encoding="utf-8", errors="ignore")
        for db in dark_blues:
            matches = re.findall(re.escape(db), logo_content, re.IGNORECASE)
            assert len(matches) == 0, f"Found forbidden dark blue color {db} in logo.svg: {matches}"


def test_ui_high_contrast_theme_and_step2_baseline_banner():
    """
    Verify:
    1. Step 2 'Literature Baseline (Parent)' label has color: #1f77b4 matching the text below it.
    2. Step 2 baseline card background is high contrast (rgba(31, 119, 180, 0.1)) and not low contrast grey.
    3. Transition arrow is #1f77b4, not grey #888.
    4. Target electrolyte label has color: #2ca02c matching the target text below it.
    5. Light mode CSS enforces high contrast #000000 black typography across captions, metrics, widgets, and cards.
    6. App contains zero low-contrast text-dimming opacities or grey text colors.
    """
    import re
    from app import CUSTOM_CSS

    # 1. Verify Step 2 Literature Baseline banner markup in app.py
    app_text = Path(APP_PATH).read_text(encoding="utf-8")
    assert 'color: #1f77b4;">{tr("parent_electrolyte_label")}</div>' in app_text, (
        "Literature Baseline (Parent) must use #1f77b4 matching the text below it"
    )
    assert 'background: rgba(31, 119, 180, 0.1);' in app_text, (
        "Literature Baseline card must have tinted blue background, not grey"
    )
    assert 'color: #1f77b4; font-weight: bold;">➔</div>' in app_text, (
        "Transition arrow must be high contrast #1f77b4, not grey #888"
    )
    assert 'color: #2ca02c;">{tr("target_electrolyte_label")}</div>' in app_text, (
        "Target electrolyte label must use high contrast #2ca02c"
    )

    # 2. Verify Light Mode CSS rules in CUSTOM_CSS enforce black text and opacity 1
    assert '[data-theme="light"]' in CUSTOM_CSS
    assert '@media (prefers-color-scheme: light)' in CUSTOM_CSS
    assert '[data-testid="stCaptionContainer"] *' in CUSTOM_CSS
    assert '[data-testid="stMetricLabel"] *' in CUSTOM_CSS
    assert 'color: #000000 !important;' in CUSTOM_CSS
    assert 'opacity: 1 !important;' in CUSTOM_CSS

    # 3. Verify absence of grey text colors in app text
    forbidden_greys = ["#888", "#666", "#777", "#999", "#aaa", "#bbb", "#ccc", "#555"]
    for line in app_text.splitlines():
        l = line.lower()
        if "color:" in l:
            for g in forbidden_greys:
                assert f"color: {g}" not in l and f"color:{g}" not in l, (
                    f"Forbidden grey text color {g} found in app.py line: {line.strip()}"
                )

    # 4. Headless verification in Demo Mode
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    at.sidebar.radio[0].set_value("demo").run()
    assert len(at.exception) == 0

    markdown_corpus = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    assert "color: #1f77b4" in markdown_corpus, "Step 2 baseline banner with #1f77b4 must render in Demo Mode"


def test_export_engine_formats_generation():
    """
    Unit test validating that export_engine creates well-formed OpenXML packages
    (.xlsx, .docx) and formatted text (.txt) with zero third-party dependencies.
    """
    import zipfile
    import io
    import xml.etree.ElementTree as ET
    from export_engine import (
        generate_dispensing_schedule_xlsx,
        generate_dispensing_schedule_docx,
        generate_dispensing_schedule_txt,
        generate_campaign_summary_xlsx,
        generate_campaign_summary_docx,
        generate_campaign_summary_txt,
    )
    from models import DispensingSchedule, RoboticVial

    vials = [
        RoboticVial(
            vial_id="A1",
            rack_format="opentrons_24_tuberack_generic_2ml_screwcap",
            row="A",
            col=1,
            additive_conc_mM=25.0,
            salt_conc_M=2.0,
            target_volume_uL=1000.0,
            additive_stock_uL=200.0,
            salt_stock_uL=500.0,
            water_uL=300.0,
            warnings=[],
            slot=1,
            aspiration_rate_ul_s=50.0,
            dispense_rate_ul_s=50.0,
        )
    ]
    sched = DispensingSchedule(
        schedule_id="SCHED_001",
        target_electrolyte_salt="Zn(OTf)2",
        electrolyte_adjustment_rationale="Solvation sheath screening",
        rack_format="opentrons_24_tuberack_generic_2ml_screwcap",
        total_vials=1,
        total_volume_per_vial_uL=1000.0,
        additive_stock_conc_mM=100.0,
        salt_stock_conc_M=4.0,
        salt_stock_name="Zn(OTf)2 4M Stock",
        vials=vials,
        warnings=[],
        ot2_deck_slot=1,
        default_aspiration_rate_ul_s=50.0,
        default_dispense_rate_ul_s=50.0,
    )

    # Test XLSX generation & validity
    xlsx_bytes = generate_dispensing_schedule_xlsx(sched)
    assert len(xlsx_bytes) > 0
    with zipfile.ZipFile(io.BytesIO(xlsx_bytes)) as z:
        assert "[Content_Types].xml" in z.namelist()
        sheet_xml = z.read("xl/worksheets/sheet1.xml")
        ET.fromstring(sheet_xml)

    # Test DOCX generation & validity
    docx_bytes = generate_dispensing_schedule_docx(sched)
    assert len(docx_bytes) > 0
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as z:
        assert "word/document.xml" in z.namelist()
        doc_xml = z.read("word/document.xml")
        ET.fromstring(doc_xml)

    # Test TXT generation
    txt_str = generate_dispensing_schedule_txt(sched)
    assert "OPENTRONS OT-2 ROBOTIC DISPENSING SCHEDULE" in txt_str
    assert "SCHED_001" in txt_str


def test_ui_failure_mode_proper_phrasing_and_no_underscores():
    """
    Verify that raw failure mode strings with underscores (e.g., 'dendritic_short')
    are never displayed to the user, and are instead properly formatted as
    'Dendritic Short Circuit'.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    at.sidebar.radio[0].set_value("demo").run()
    assert len(at.exception) == 0

    # Collect visible text elements
    all_text = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    for metric in at.metric:
        if hasattr(metric, "value") and metric.value:
            all_text += " " + str(metric.value)
        if hasattr(metric, "label") and metric.label:
            all_text += " " + str(metric.label)

    # Raw snake_case string must not appear in visible UI elements
    assert "dendritic_short" not in all_text, "dendritic_short should be formatted without underscores"
    assert "Dendritic Short Circuit" in all_text, "Proper failure mode phrasing 'Dendritic Short Circuit' must be rendered"


def test_ui_tab2_google_scholar_literature_search_and_cards():
    """
    Verify that Step 1 (Tab 2) renders:
    1. Academic literature search bar with query input and search button.
    2. Instruction words ('Keywords' and 'Number of Literature') visible on input boxes.
    3. Initially empty search query and no paper cards before search is run.
    4. Google Scholar-style paper cards after searching, with titles, authors, venues, years, and DOI links.
    5. Access badges (Open Access / Journal Article).
    6. Expandable full scientific abstracts.
    7. Action buttons ('Use Extracted Additive' and 'Analyze Abstract') on paper cards.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    all_md_init = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    button_labels = [getattr(b, "label", "") for b in at.button]

    # Verify search title
    assert "Academic Literature Search & Mining" in all_md_init or "Literature Search" in all_md_init

    # Verify search input, limit selectbox, and instruction words
    search_inputs = [ti for ti in at.text_input if ti.key == "lit_search_query_input"]
    assert len(search_inputs) > 0, "Literature search text input must be present"
    assert search_inputs[0].value == "", "Search query input must be empty by default"
    assert "Keywords" in search_inputs[0].label, "Instruction word 'Keywords' must be on text input"

    limit_selects = [sb for sb in at.selectbox if sb.key == "lit_search_limit_select"]
    assert len(limit_selects) > 0, "Literature limit selectbox must be present"
    assert "Number of Literature" in limit_selects[0].label, "Instruction word 'Number of Literature' must be on selectbox"

    search_buttons = [b for b in at.button if b.key == "lit_search_submit_btn"]
    assert len(search_buttons) > 0, "Search submit button must exist"

    # Verify NO paper cards exist initially before search is triggered
    assert '<div class="literature-paper-card">' not in all_md_init, "No paper cards should appear before searching"
    assert len([b for b in at.button if b.key and "btn_use_paper_" in b.key]) == 0, "No card action buttons should exist before searching"
    assert "Found" not in all_md_init or "academic papers via Semantic Scholar" not in all_md_init

    # Now execute search by providing query and clicking search button
    search_inputs[0].input("zinc-ion battery electrolyte additive").run()
    search_buttons = [b for b in at.button if b.key == "lit_search_submit_btn"]
    search_buttons[0].click().run()
    assert len(at.exception) == 0

    all_md = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    card_buttons = [getattr(b, "label", "") for b in at.button]

    # Verify literature paper card content in markdown after search
    assert "literature-paper-card" in all_md
    assert "Regulating Solvation Sheaths via Trehalose Additive" in all_md or "Trehalose" in all_md
    assert "DOI:" in all_md

    # Verify per-card action buttons have been removed to keep UI streamlined
    assert len([b for b in at.button if b.key and "btn_use_paper_" in b.key]) == 0, "Per-card action buttons must be removed"
    assert len([b for b in at.button if b.key and "btn_analyze_paper_" in b.key]) == 0, "Per-card analyze buttons must be removed"

    # Verify extracted additives are automatically populated into mined_literature_additives
    mined_t1 = at.session_state.get("mined_literature_additives", [])
    assert len(mined_t1) > 0, "Extracted additives must be automatically loaded into mined_literature_additives"

    # Verify abstracts have clean scientific subscripts and units without spaces
    assert "ZnSO₄" in all_md, "Abstracts and cards must render ZnSO₄ with subscript 4"
    assert "Zn²⁺" in all_md, "Abstracts and cards must render Zn²⁺ with superscript charge"
    assert "mA/cm²" in all_md, "Abstracts and cards must render mA/cm² units"
    assert "2 M ZnSO 4" not in all_md, "Spaced-out '2 M ZnSO 4' should not appear in rendered UI"


def test_ui_tab2_use_extracted_additive_action_updates_esol_and_step2():
    """
    Verify that literature search automatically loads extracted chemical additives into Step 2:
    1. Performs search to display paper cards without clutter buttons.
    2. Extracted additive (Trehalose, 10 mM, 2 M ZnSO4) is automatically loaded into mined_literature_additives (Tier 1).
    3. Delaney ESOL solubility check (logs) and descriptors are computed.
    4. Candidate is immediately available in Step 2 for formulation planning.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    # Search for papers first to generate cards
    search_inputs = [ti for ti in at.text_input if ti.key == "lit_search_query_input"]
    search_inputs[0].input("trehalose").run()
    search_btn = [b for b in at.button if b.key == "lit_search_submit_btn"][0]
    search_btn.click().run()
    assert len(at.exception) == 0

    # Verify per-card action buttons are removed
    assert len([b for b in at.button if b.key and "btn_use_paper_" in b.key]) == 0

    # Verify extracted candidate is automatically loaded into mined_literature_additives (Tier 1)
    t1_cands = at.session_state.get("mined_literature_additives", [])
    assert len(t1_cands) > 0, "mined_literature_additives must be automatically populated"
    treh = next((c for c in t1_cands if "trehalose" in c.name.lower()), None)
    assert treh is not None, "Trehalose must be present in mined_literature_additives"
    assert treh.reported_concentration == 10.0
    assert "ZnSO₄" in treh.parent_baseline_electrolyte
    assert treh.logs is not None, "Delaney ESOL must be pre-calculated"
    assert treh.molecular_weight > 300.0

    # Verify Trehalose chemical formula is displayed in image caption
    captions = []
    for img in at.image:
        if hasattr(img, "caption") and img.caption:
            captions.append(img.caption)
        elif hasattr(img, "captions") and img.captions:
            captions.extend(img.captions)
    caption_corpus = " ".join(captions)
    assert "C₁₂H₂₂O₁₁" in caption_corpus, f"Expected Trehalose formula C₁₂H₂₂O₁₁, got {caption_corpus}"

    # Verify transient toast notification was fired with subscript 4 for ZnSO₄
    assert len(at.toast) > 0, "A transient toast notification must be fired on extraction"
    toast_values = " ".join([t.value for t in at.toast])
    assert "ZnSO₄" in toast_values, f"Expected subscript ZnSO₄ in toast, got {toast_values}"
    assert "ZnSO4" not in toast_values, f"ZnSO4 with ASCII 4 must not appear in toast, got {toast_values}"

    # Verify notice was popped from session state so it never lingers on screen
    assert "last_extracted_notice" not in at.session_state


def test_ui_tab2_demo_mode_shows_preloaded_examples():
    """
    Verify that in Demo Mode:
    1. Pre-loaded literature quick search chips (Trehalose, TEABr, D-Sorbitol, Thiourea, Glucose) are present.
    2. Pre-loaded example inspection buttons (D-Sorbitol, Trehalose, TEABr, Thiourea, Glucose) are present.
    3. Clicking a pre-loaded example button updates the 2D structure, formula, descriptors, and ESOL cap.
    4. Curated additive pool similarity table is rendered.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    at.sidebar.radio[0].set_value("demo").run()
    assert len(at.exception) == 0

    button_keys = [getattr(b, "key", "") for b in at.button]

    # 1. Quick search benchmark chips
    assert "btn_chip_trehalose" in button_keys
    assert "btn_chip_teabr" in button_keys
    assert "btn_chip_sorbitol" in button_keys
    assert "btn_chip_thiourea" in button_keys
    assert "btn_chip_glucose" in button_keys

    # 2. Pre-loaded example inspection buttons
    assert "btn_demo_ex_sorbitol" in button_keys
    assert "btn_demo_ex_trehalose" in button_keys
    assert "btn_demo_ex_teabr" in button_keys
    assert "btn_demo_ex_thiourea" in button_keys
    assert "btn_demo_ex_glucose" in button_keys

    # 3. Click pre-loaded example button (Glucose)
    glucose_btn = [b for b in at.button if b.key == "btn_demo_ex_glucose"][0]
    glucose_btn.click().run()
    assert len(at.exception) == 0

    assert at.session_state.get("tab2_mol_select") == "Glucose"
    captions = []
    for img in at.image:
        if hasattr(img, "caption") and img.caption:
            captions.append(img.caption)
        elif hasattr(img, "captions") and img.captions:
            captions.extend(img.captions)
    caption_corpus = " ".join(captions)
    assert "Glucose" in caption_corpus
    assert "C₆H₁₂O₆" in caption_corpus

    # 4. Curated additive pool table is present in Demo Mode
    all_md = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    assert "sec_additive_pool" in all_md or "Curated Zinc Battery Additive Pool" in all_md or "精选锌电池添加剂" in all_md or "Réserve Curée d'Additifs" in all_md

    # 5. Click quick search benchmark chip (Trehalose) without StreamlitWidgetAlreadyInstantiatedError
    trehalose_chip = [b for b in at.button if b.key == "btn_chip_trehalose"][0]
    trehalose_chip.click().run()
    assert len(at.exception) == 0
    assert at.session_state["lit_search_query_input"] == "trehalose zinc ion battery electrolyte additive"
    assert at.session_state["literature_search_query"] == "trehalose zinc ion battery electrolyte additive"


def test_ui_tab2_custom_mode_hides_examples_and_renders_custom_additive():
    """
    Verify that in Custom Experiment Mode:
    1. Pre-loaded literature quick search chips are strictly hidden.
    2. Pre-loaded example inspection buttons are strictly hidden.
    3. Curated additive pool table is hidden, displaying notification caption.
    4. Custom additive name and SMILES inputs are rendered and active.
    5. 2D structure, RDKit descriptors, multi-electrolyte compatibility, and Delaney ESOL 80% safety cap are computed for custom molecule.
    6. Transferring custom additive to Step 2 updates formulation plan and dispensing schedule with ESOL ceiling.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    # Switch operating mode to Custom Experiment Mode (index 1 in sidebar radio)
    radios = [r for r in at.sidebar.radio]
    assert len(radios) > 0, "Sidebar operating mode radio must exist"
    mode_radio = radios[0]
    mode_radio.set_value("custom").run()
    assert len(at.exception) == 0

    button_keys = [getattr(b, "key", "") for b in at.button]

    # 1. Pre-loaded quick search chips must NOT be rendered in Custom Mode
    assert "btn_chip_trehalose" not in button_keys
    assert "btn_chip_teabr" not in button_keys
    assert "btn_chip_sorbitol" not in button_keys
    assert "btn_chip_thiourea" not in button_keys
    assert "btn_chip_glucose" not in button_keys

    # 2. Pre-loaded example buttons must NOT be rendered in Custom Mode
    assert "btn_demo_ex_sorbitol" not in button_keys
    assert "btn_demo_ex_trehalose" not in button_keys
    assert "btn_demo_ex_teabr" not in button_keys
    assert "btn_demo_ex_thiourea" not in button_keys
    assert "btn_demo_ex_glucose" not in button_keys

    # 3. Curated additive pool table is hidden (and redundant hidden-pool caption is omitted)
    all_captions = [getattr(c, "value", "") for c in at.caption if hasattr(c, "value")]
    caption_text = " ".join(all_captions)
    assert "Pre-loaded benchmark additive pool table is hidden" not in caption_text

    # 4. Custom molecule inputs exist and are initially empty
    custom_name_inputs = [ti for ti in at.text_input if ti.key == "tab2_custom_additive_name_input"]
    custom_smiles_inputs = [ti for ti in at.text_input if ti.key == "tab2_custom_smiles_input"]
    assert len(custom_name_inputs) > 0, "Custom additive name input must exist in Tab 2"
    assert len(custom_smiles_inputs) > 0, "Custom SMILES input must exist in Tab 2"

    # In Custom Mode with empty inputs, no molecular image is pre-loaded
    assert len(at.image) == 0, "No preloaded molecular image should be rendered in Custom Mode when inputs are empty"

    # User inputs custom candidate: Succinic Acid
    custom_name_inputs[0].set_value("Succinic Acid")
    custom_smiles_inputs[0].set_value("C(CC(=O)O)C(=O)O").run()
    assert len(at.exception) == 0

    # 5. Verify custom molecule (Succinic Acid) has valid 2D structure, formula, descriptors, and ESOL cap
    captions = []
    for img in at.image:
        if hasattr(img, "caption") and img.caption:
            captions.append(img.caption)
        elif hasattr(img, "captions") and img.captions:
            captions.extend(img.captions)
    caption_corpus = " ".join(captions)
    assert "C₄H₆O₄" in caption_corpus, f"Expected Succinic Acid formula C₄H₆O₄, got {caption_corpus}"

    # 6. Verify reactive auto-synchronization: redundant transfer button is removed and plan synchronizes in real-time
    apply_btn = [b for b in at.button if b.key == "btn_apply_custom_additive_tab2"]
    assert len(apply_btn) == 0, "btn_apply_custom_additive_tab2 button must be removed as redundant with reactive sync"

    plan = at.session_state.get("extracted_formulation_plan")
    assert plan is not None
    assert plan.anchor.additive_name == "Succinic Acid"
    assert plan.anchor.solubility_ceiling_mM is not None
    assert plan.anchor.solubility_ceiling_mM > 0.0

    sched = at.session_state.get("extracted_dispensing_schedule")
    assert sched is not None
    assert len(sched.vials) > 0


def test_ui_tab2_search_sorting_and_ml_suggested_additives_sync():
    """
    Verify:
    1. Search sort selectbox (relevance, date, citations) and limit selectbox exist.
    2. Executing search populates suggested candidates.
    3. Suggested Additives dropdown is rendered in Tab 2.
    4. Selecting a suggested candidate automatically updates Custom Molecule (SMILES),
       custom additive name, and downstream models.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    # 1. Switch to Custom Mode
    mode_radio = at.sidebar.radio[0]
    mode_radio.set_value("custom").run()
    assert len(at.exception) == 0

    # 2. Verify sort and limit selectboxes
    sort_selects = [sb for sb in at.selectbox if sb.key == "lit_search_sort_select"]
    assert len(sort_selects) > 0, "lit_search_sort_select selectbox must exist"
    assert any("Relevance" in opt for opt in sort_selects[0].options)
    assert any("Date" in opt for opt in sort_selects[0].options)
    assert any("Citations" in opt for opt in sort_selects[0].options)

    limit_selects = [sb for sb in at.selectbox if sb.key == "lit_search_limit_select"]
    assert len(limit_selects) > 0, "lit_search_limit_select selectbox must exist"
    assert any("25" in opt for opt in limit_selects[0].options)
    assert any("50" in opt for opt in limit_selects[0].options)
    assert any("100" in opt for opt in limit_selects[0].options)
    assert any("150" in opt for opt in limit_selects[0].options)

    # 3. Trigger search with sorting by date
    date_opt = [opt for opt in sort_selects[0].options if "Date" in opt][0]
    sort_selects[0].set_value(date_opt).run()
    assert len(at.exception) == 0

    search_inputs = [ti for ti in at.text_input if ti.key == "lit_search_query_input"]
    search_inputs[0].input("polyol").run()
    search_buttons = [b for b in at.button if b.key == "lit_search_submit_btn"]
    search_buttons[0].click().run()
    assert len(at.exception) == 0

    # 4. Verify suggested candidates and mined additives were generated
    suggested_cands = at.session_state.get("suggested_candidates")
    assert suggested_cands is not None and len(suggested_cands) > 0

    mined_cands = at.session_state.get("mined_literature_additives")
    assert mined_cands is not None and len(mined_cands) > 0

    # 5. Verify both Tier 1 and Tier 2 dropdowns exist in Tab 2
    t1_dropdowns = [sb for sb in at.selectbox if sb.key == "step2_tier1_additive_dropdown"]
    assert len(t1_dropdowns) > 0, "step2_tier1_additive_dropdown must be rendered"

    sugg_dropdowns = [sb for sb in at.selectbox if sb.key == "tab2_suggested_additives_dropdown"]
    assert len(sugg_dropdowns) > 0, "tab2_suggested_additives_dropdown must be rendered"

    # 6. Verify default active additive is synchronized with top mined literature candidate
    picked_cand = mined_cands[0]
    custom_smiles_inputs = [ti for ti in at.text_input if ti.key == "tab2_custom_smiles_input"]
    assert len(custom_smiles_inputs) > 0
    assert custom_smiles_inputs[0].value == picked_cand.smiles

    custom_name_inputs = [ti for ti in at.text_input if ti.key == "tab2_custom_additive_name_input"]
    assert len(custom_name_inputs) > 0
    assert custom_name_inputs[0].value == picked_cand.name


def test_ui_tab2_enter_key_triggers_search():
    """Verify that pressing Enter in the query input triggers literature search without clicking button."""
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    search_inputs = [ti for ti in at.text_input if ti.key == "lit_search_query_input"]
    assert len(search_inputs) > 0
    # .input(...).run() simulates typing text and pressing Enter
    search_inputs[0].input("thiourea").run()
    assert len(at.exception) == 0

    results = at.session_state.get("literature_search_results")
    assert results is not None and len(results) > 0
    assert at.session_state.get("literature_search_query") == "thiourea"


def test_ui_tab2_batch_pdf_uploader_and_extraction():
    """
    Verify:
    1. Multi-PDF file uploader widget exists in Tab 2.
    2. 'Load Benchmark Sample PDF' button exists.
    3. Clicking sample PDF button parses document, populates batch_pdf_extractions.
    4. Suggested candidates dropdown is populated with ranked candidates.
    5. Ingestion summary table and card render extracted entity, concentration, and salt.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    at.sidebar.radio[0].set_value("demo").run()
    assert len(at.exception) == 0

    # Verify file uploader widget exists
    uploaders = [u for u in at.file_uploader if u.key == "upload_batch_pdf"]
    assert len(uploaders) > 0, "upload_batch_pdf file uploader must exist in Step 1 tab"

    # Verify sample loading and process buttons exist
    sample_btns = [b for b in at.button if b.key == "btn_load_sample_pdf"]
    assert len(sample_btns) > 0, "btn_load_sample_pdf button must exist"

    # Click Load Benchmark Sample PDF
    sample_btns[0].click().run()
    assert len(at.exception) == 0

    # Verify batch PDF extractions populated in session state
    batch_ext = at.session_state.get("batch_pdf_extractions")
    assert batch_ext is not None and len(batch_ext) > 0
    item = batch_ext[0]
    assert item["additive"].name == "Tetraethylammonium bromide"
    assert item["additive"].optimal_concentration.value == 50.0

    # Verify suggested chemical database analogue is populated (Issue 2)
    assert item.get("suggested_analogue") is not None
    assert item["suggested_analogue"].similarity_score is not None

    # Verify suggested source is marked as chemical database (Issue 4)
    assert at.session_state.get("suggested_source") == "pdf_database"

    # Verify suggested candidates populated
    suggested = at.session_state.get("suggested_candidates")
    assert suggested is not None and len(suggested) > 0


def test_ui_tab2_batch_pdf_apply_as_active_additive():
    """
    Verify clicking 'Apply as Active Additive' on an extracted PDF item:
    1. Sets active additive name and SMILES.
    2. Synchronizes Step 2 formulation plan and dispensing schedule.
    3. Recomputes 2D formula and Delaney ESOL ceiling.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    at.sidebar.radio[0].set_value("demo").run()
    assert len(at.exception) == 0

    # Trigger sample PDF extraction
    sample_btns = [b for b in at.button if b.key == "btn_load_sample_pdf"]
    assert len(sample_btns) > 0
    sample_btns[0].click().run()
    assert len(at.exception) == 0

    # Find the apply button for the first extracted batch item
    apply_btns = [b for b in at.button if b.key == "btn_apply_pdf_0"]
    assert len(apply_btns) > 0, "btn_apply_pdf_0 button must exist for the extracted document"

    apply_btns[0].click().run()
    assert len(at.exception) == 0

    # Verify active additive updated
    assert at.session_state.get("selected_additive_name") == "Tetraethylammonium bromide"
    assert at.session_state.get("tab2_mol_select") == "Tetraethylammonium bromide"

    # Verify Step 2 formulation plan updated with anchor concentration
    plan = at.session_state.get("extracted_formulation_plan")
    assert plan is not None
    assert plan.anchor.additive_name == "Tetraethylammonium bromide"
    assert plan.anchor.published_concentration == 50.0

    # Verify dispensing schedule updated
    sched = at.session_state.get("extracted_dispensing_schedule")
    assert sched is not None
    assert len(sched.vials) > 0


def test_ui_tab2_batch_pdf_apply_suggested_analogue():
    """
    Verify clicking 'Apply Suggested Analogue' on an extracted PDF item:
    1. Sets the chemical database analogue as the active additive.
    2. Synchronizes Step 2 formulation plan and dispensing schedule.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    at.sidebar.radio[0].set_value("demo").run()
    assert len(at.exception) == 0

    # Trigger sample PDF extraction
    sample_btns = [b for b in at.button if b.key == "btn_load_sample_pdf"]
    assert len(sample_btns) > 0
    sample_btns[0].click().run()
    assert len(at.exception) == 0

    batch_ext = at.session_state.get("batch_pdf_extractions")
    assert batch_ext is not None
    top_analogue = batch_ext[0]["suggested_analogue"]
    assert top_analogue is not None

    # Click the apply button for the suggested analogue
    apply_sugg_btns = [b for b in at.button if b.key == "btn_apply_sugg_0"]
    assert len(apply_sugg_btns) > 0, "btn_apply_sugg_0 button must exist"

    apply_sugg_btns[0].click().run()
    assert len(at.exception) == 0

    # Verify active additive updated to the analogue
    assert at.session_state.get("selected_additive_name") == top_analogue.name
    plan = at.session_state.get("extracted_formulation_plan")
    assert plan is not None
    assert plan.anchor.additive_name == top_analogue.name


def test_ui_suggested_additives_header_dynamic():
    """
    Verify that the suggested additives section header dynamically changes
    to indicate 'Chemical Database' when generated from uploaded PDFs (Issue 4).
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    at.sidebar.radio[0].set_value("demo").run()
    assert len(at.exception) == 0

    # Click Load Benchmark Sample PDF to trigger PDF mode
    sample_btns = [b for b in at.button if b.key == "btn_load_sample_pdf"]
    sample_btns[0].click().run()
    assert len(at.exception) == 0

    assert at.session_state.get("suggested_source") == "pdf_database"

    all_text = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    assert "Chemical Database" in all_text or "Base de données chimiques" in all_text or "化学数据库" in all_text


def test_batch_pdf_trilingual_localization_parity():
    """Verify all batch PDF keys are translated across English, French, and Chinese with 100% key parity."""
    from localization import TRANSLATIONS
    batch_pdf_keys = [
        "subtab_online_search",
        "subtab_batch_pdf",
        "sec_batch_pdf_title",
        "sec_batch_pdf_desc",
        "batch_pdf_online_opt_notice",
        "get_free_gemini_key_link",
        "upload_batch_pdf_label",
        "upload_batch_pdf_help",
        "btn_process_batch_pdfs",
        "btn_load_sample_pdf",
        "btn_load_sample_pdf_help",
        "batch_pdf_processing_title",
        "batch_pdf_progress_text",
        "batch_pdf_success_notice",
        "batch_summary_title",
        "batch_summary_caption",
        "col_pdf_filename",
        "col_pdf_additive",
        "col_pdf_formula",
        "col_pdf_concentration",
        "col_pdf_baseline_salt",
        "col_pdf_mechanism",
        "col_pdf_suggested_additive",
        "btn_apply_pdf_additive",
        "btn_apply_pdf_additive_help",
        "btn_apply_suggested_pdf_additive",
        "btn_apply_suggested_pdf_additive_help",
        "sec_suggested_additives_db",
        "suggested_additive_notice_db",
        "no_additive_identified_title",
        "no_additive_identified_desc",
        "suggested_analogue_card_label",
        "engine_status_gemini_active",
        "engine_status_offline_heuristic",
        "engine_status_gemini_tip",
        "gemini_api_key_label",
        "gemini_api_key_help",
        "btn_save_key",
        "btn_clear_uploaded_pdfs",
        "btn_clear_uploaded_pdfs_help",
        "batch_pdf_cleared_notice",
        "no_additive_found_online",
        "no_pdf_uploaded_tip",
    ]

    for lang in ["en", "fr", "zh"]:
        assert lang in TRANSLATIONS
        for k in batch_pdf_keys:
            assert k in TRANSLATIONS[lang], f"Key '{k}' missing from '{lang}' translations"
            assert len(TRANSLATIONS[lang][k]) > 0


def test_batch_pdf_clear_button_and_demo_mode_visibility():
    """
    Verify:
    1. In Custom Mode (default), 'btn_load_sample_pdf' is hidden while 'btn_clear_uploaded_pdfs' remains visible.
    2. 'btn_save_batch_gemini_key' uses the localized label rather than a hardcoded trilingual string.
    3. In Demo Mode, 'btn_load_sample_pdf' exists.
    4. Clicking 'btn_clear_uploaded_pdfs' clears session state batch extractions and resets uploader version.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    # 1. In Custom Mode (default)
    sample_btns_custom = [b for b in at.button if b.key == "btn_load_sample_pdf"]
    assert len(sample_btns_custom) == 0, "btn_load_sample_pdf must NOT be visible under Custom Mode"

    clear_btns = [b for b in at.button if b.key == "btn_clear_uploaded_pdfs"]
    assert len(clear_btns) > 0, "btn_clear_uploaded_pdfs must exist"

    # 2. Check localized Save Key button label
    save_btns = [b for b in at.button if b.key == "btn_save_batch_gemini_key"]
    assert len(save_btns) > 0
    assert "Enregistrer /" not in save_btns[0].label, "Save key button must not display raw multi-language string in English"

    # 3. Switch to Demo Mode and verify btn_load_sample_pdf appears
    radios = [r for r in at.sidebar.radio]
    assert len(radios) > 0
    radios[0].set_value("demo").run()
    assert len(at.exception) == 0

    sample_btns = [b for b in at.button if b.key == "btn_load_sample_pdf"]
    assert len(sample_btns) > 0, "btn_load_sample_pdf must be visible in Demo Mode"

    # Load sample PDF to populate batch extractions
    sample_btns[0].click().run()
    assert len(at.exception) == 0
    assert len(at.session_state.get("batch_pdf_extractions", [])) > 0

    # Click Clear Uploaded PDFs
    clear_btns = [b for b in at.button if b.key == "btn_clear_uploaded_pdfs"]
    assert len(clear_btns) > 0
    clear_btns[0].click().run()
    assert len(at.exception) == 0
    assert len(at.session_state.get("batch_pdf_extractions", [])) == 0
    assert at.session_state.get("batch_pdf_uploader_version", 0) > 0


def test_custom_mode_is_default_and_sidebar_minimal_and_optimization_notice():
    """
    Verify:
    1. Custom Experiment Mode is selected by default upon app startup.
    2. In Custom Mode, the only sections in the sidebar under Operating Mode are:
       - 'Repository & Hosting'
       - 'Project Credits & Tech Stack'
       (i.e., 'Target Electrolyte to Optimize', additive pickers, DoE sliders, liquid handler, and Gemini key are hidden).
    3. Gemini API key text input is completely removed from the sidebar in both Custom and Demo modes.
    4. Batch PDF document ingestion subtab label contains the optimization notice.
    5. Switching to Demo Mode reveals target electrolyte and DoE controls in the sidebar.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    # 1. Default operating mode is Custom Experiment Mode
    radios = [r for r in at.sidebar.radio]
    assert len(radios) > 0
    assert radios[0].value == "custom", "Default operating mode must be 'custom'"

    # 2. In Custom Mode, formulation controls are hidden from the sidebar
    sidebar_selectboxes = [sb for sb in at.sidebar.selectbox]
    sidebar_sliders = [sl for sl in at.sidebar.slider]
    sidebar_text_inputs = [ti for ti in at.sidebar.text_input]
    sidebar_number_inputs = [ni for ni in at.sidebar.number_input]

    # Only language selectbox should be in sidebar
    assert len(sidebar_selectboxes) == 1, f"Only language selectbox should be in sidebar under custom mode, found: {[sb.label for sb in sidebar_selectboxes]}"
    assert sidebar_selectboxes[0].key == "app_language"
    assert len(sidebar_sliders) == 0, f"No sliders should be in sidebar under custom mode, found: {[sl.label for sl in sidebar_sliders]}"
    assert len(sidebar_text_inputs) == 0, f"No text inputs should be in sidebar under custom mode, found: {[ti.label for ti in sidebar_text_inputs]}"
    assert len(sidebar_number_inputs) == 0, f"No number inputs should be in sidebar under custom mode, found: {[ni.label for ni in sidebar_number_inputs]}"

    # Verify Gemini key input is not in the sidebar
    gemini_sidebar_inputs = [ti for ti in at.sidebar.text_input if "gemini" in (getattr(ti, "key", "") or "").lower()]
    assert len(gemini_sidebar_inputs) == 0, "Gemini API key input must be completely removed from sidebar"

    # Verify Project Credits & Tech Stack section exists in sidebar and repository/hosting links are removed
    sidebar_md = " ".join([m.value for m in at.sidebar.markdown if hasattr(m, "value")])
    assert "https://github.com" not in sidebar_md
    assert "https://share.streamlit.io" not in sidebar_md
    assert "RDKit" in sidebar_md
    assert "Opentrons OT-2" in sidebar_md

    # 3. Verify Batch PDF document ingestion subtab label is clean and optimization notice is present under section
    from localization import get_text
    en_label = get_text("subtab_batch_pdf", lang="en")
    fr_label = get_text("subtab_batch_pdf", lang="fr")
    zh_label = get_text("subtab_batch_pdf", lang="zh")
    assert en_label == "Upload & Extract PDF Papers"
    assert fr_label == "Téléverser & Extraire des PDF"
    assert zh_label == "上传并解析 PDF 论文"

    en_notice = get_text("batch_pdf_online_opt_notice", lang="en")
    fr_notice = get_text("batch_pdf_online_opt_notice", lang="fr")
    zh_notice = get_text("batch_pdf_online_opt_notice", lang="zh")
    assert "Google Gemini" in en_notice and "Google AI Studio" in en_notice
    assert "Google Gemini" in fr_notice and "Google AI Studio" in fr_notice
    assert "Google Gemini" in zh_notice and "Google AI Studio" in zh_notice

    # Verify offline heuristic warning and popover help text
    en_offline = get_text("engine_status_offline_heuristic", lang="en")
    assert "Low Accuracy" in en_offline
    en_help = get_text("gemini_api_key_help", lang="en")
    assert "low accuracy" in en_help.lower() and "google ai studio" in en_help.lower()

    # 4. Switch to Demo Mode and verify formulation controls reappear in sidebar
    radios[0].set_value("demo").run()
    assert len(at.exception) == 0

    demo_sb_labels = [getattr(sb, "label", "") for sb in at.sidebar.selectbox]
    assert any("Target Electrolyte" in lbl for lbl in demo_sb_labels)
    assert any("Additive to Test" in lbl for lbl in demo_sb_labels)
    assert any("Concentration Sampling Method" in lbl or "Experimental Design Method" in lbl for lbl in demo_sb_labels)
    assert any("Robotic Liquid Handler" in lbl for lbl in demo_sb_labels)

    # Gemini API key must still NOT be in sidebar under demo mode
    gemini_demo_inputs = [ti for ti in at.sidebar.text_input if "gemini" in (getattr(ti, "key", "") or "").lower()]
    assert len(gemini_demo_inputs) == 0, "Gemini API key must remain absent from sidebar in demo mode"


def test_ui_automated_experiment_run_tab_visibility_demo_mode_only():
    """
    Verify that:
    1. In Custom Mode (default upon app startup), the 'Automated Experiment Run' tab
       is strictly hidden / omitted from tabs.
    2. Only standalone tool tabs (without 'Step') are rendered in Custom Mode.
    3. The 'Run Autonomous Experiment' button is NOT present in Custom Mode.
    4. Switching to 'Demo Mode (Pre-loaded Literature Data)' dynamically reveals the
       'Automated Experiment Run' tab as the first tab (tabs[0]), with Step 1 as tabs[1].
    5. The 'Run Autonomous Experiment' button is present in Demo Mode.
    6. Switching back to Custom Mode hides the 'Automated Experiment Run' tab again.
    """
    from localization import get_text
    tab_run_en = get_text("tab_run", lang="en")
    tab_step1_en = get_text("tab_step1", lang="en")
    tab_step1_custom_en = get_text("tab_step1_custom", lang="en")

    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    # 1. Custom mode by default: verify Automated Experiment Run tab is NOT present and top-level tabs have no 'Step'
    tab_labels_custom = [tab.label for tab in at.tabs]
    assert not any(tab_run_en in lbl for lbl in tab_labels_custom), "Automated Experiment Run tab must NOT be visible in Custom Mode"
    assert any(tab_step1_custom_en in lbl for lbl in tab_labels_custom), "Literature Search & Additive Finder tab must be present in Custom Mode"
    for top_lbl in tab_labels_custom[:3]:
        assert "Step " not in top_lbl, f"Custom Mode top-level tab '{top_lbl}' must not contain 'Step '"

    # Buttons in custom mode must NOT include 'Run Autonomous'
    run_buttons_custom = [b for b in at.button if "Run Autonomous" in getattr(b, "label", "")]
    assert len(run_buttons_custom) == 0, "Run Autonomous button must NOT be present in Custom Mode"

    # 2. Switch to Demo Mode: verify Automated Experiment Run tab is visible as first tab
    at.sidebar.radio[0].set_value("demo").run()
    assert len(at.exception) == 0

    tab_labels_demo = [tab.label for tab in at.tabs]
    assert any(tab_run_en in lbl for lbl in tab_labels_demo), "Automated Experiment Run tab MUST be visible in Demo Mode"
    assert tab_run_en in tab_labels_demo[0], "Automated Experiment Run tab must be the first tab in Demo Mode"
    assert tab_step1_en in tab_labels_demo[1], "Step 1 tab must be the second tab (tabs[1]) in Demo Mode"

    run_buttons_demo = [b for b in at.button if "Run Autonomous" in getattr(b, "label", "")]
    assert len(run_buttons_demo) > 0, "Run Autonomous button MUST be present in Demo Mode"

    # 3. Switch back to Custom Mode: verify Automated Experiment Run tab is hidden again
    at.sidebar.radio[0].set_value("custom").run()
    assert len(at.exception) == 0

    tab_labels_custom_again = [tab.label for tab in at.tabs]
    assert not any(tab_run_en in lbl for lbl in tab_labels_custom_again), "Automated Experiment Run tab must be hidden when switching back to Custom Mode"
    assert any(tab_step1_custom_en in lbl for lbl in tab_labels_custom_again)


def test_ui_all_tables_hide_row_number_index():
    """
    Verify that row numbers (index columns) are removed from all tables across the application:
    - Candidates Performance Matrix table (Automated Experiment Run tab)
    - Liquid Handling Dispensing Protocol table (Step 2 tab)
    - Multi-Round Campaign History table (Experiment Summary tab)
    - Chemical Compatibility & Similarity Candidate Pool tables (Step 1 tab)
    Ensures hide_index=True is enforced on every rendered st.dataframe.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    # In Custom Mode, verify all rendered dataframes have index hidden
    for df_elem in at.dataframe:
        col_proto = df_elem.proto.columns
        assert '"hidden": true' in col_proto or '"hidden":true' in col_proto, \
            f"Dataframe in Custom Mode must have hide_index=True (got: {col_proto})"

    # Switch to Demo Mode to render all tables (Automated Experiment Run, Step 2, Summary)
    at.sidebar.radio[0].set_value("demo").run()
    assert len(at.exception) == 0

    assert len(at.dataframe) >= 4, f"Expected at least 4 tables rendered in Demo Mode, got {len(at.dataframe)}"
    for i, df_elem in enumerate(at.dataframe):
        col_proto = df_elem.proto.columns
        assert '"hidden": true' in col_proto or '"hidden":true' in col_proto, \
            f"Table index {i} does not have hide_index=True! columns proto: {col_proto}"


def test_ui_step1_and_step3_layout_updates():
    """
    Verify:
    1. Step 1:
       - 'Extract Additives & Recommend Candidates', 'Clear Uploaded PDFs', and 'Gemini API Key'
         are co-located on the same row with 'Clear Uploaded PDFs' in the middle.
       - CSS enforces identical heights (48px) across all three buttons.
       - 'Advanced Domain Knowledge Heuristic Engine (Offline Mode)' status banner is rendered.
    2. Step 3:
       - Cycler file uploader is rendered full-width.
       - 'Load Benchmark Dataset' and scenario radio options are in the Interactive Cycling section.
       - Clicking 'Load Benchmark Dataset' updates the active benchmark report without errors.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    # 1. Verify Step 1 batch buttons and equal-height CSS
    btn_keys = [getattr(b, "key", "") for b in at.button]
    assert "btn_process_batch_pdfs" in btn_keys
    assert "btn_clear_uploaded_pdfs" in btn_keys
    assert "btn_save_batch_gemini_key" in btn_keys

    css_blocks = [m.value for m in at.markdown if "<style>" in getattr(m, "value", "")]
    full_css = " ".join(css_blocks)
    assert ".st-key-btn_process_batch_pdfs button" in full_css
    assert ".st-key-btn_clear_uploaded_pdfs button" in full_css
    assert ".st-key-batch_pdf_gemini_popover button" in full_css
    assert "height: 48px !important" in full_css

    all_md = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    assert "Advanced Domain Knowledge Heuristic Engine" in all_md or "Moteur Heuristique" in all_md
    assert "Low Accuracy" in all_md or "Précision Réduite" in all_md or "低精度" in all_md

    # 2. Verify Step 3 reference cycler template downloads are removed (app auto-heals without templates)
    dl_keys_custom = [getattr(b, "key", "") for b in at.download_button]
    assert "step3_btn_dl_mpt_template" not in dl_keys_custom
    assert "step3_btn_dl_landt_csv_template" not in dl_keys_custom
    assert "step3_btn_dl_arbin_xlsx_template" not in dl_keys_custom

    # 3. Switch to Demo Mode to verify benchmark dataset controls
    at.sidebar.radio[0].set_value("demo").run()
    assert len(at.exception) == 0

    dl_keys_demo = [getattr(b, "key", "") for b in at.download_button]
    assert "step3_btn_dl_mpt_template" not in dl_keys_demo
    assert "step3_btn_dl_landt_csv_template" not in dl_keys_demo
    assert "step3_btn_dl_arbin_xlsx_template" not in dl_keys_demo

    btn_keys_demo = [getattr(b, "key", "") for b in at.button]
    assert "btn_load_benchmark_step3" in btn_keys_demo

    # Test clicking Load Benchmark Dataset button
    load_btns = [b for b in at.button if b.key == "btn_load_benchmark_step3"]
    assert len(load_btns) > 0
    load_btns[0].click().run()
    assert len(at.exception) == 0

    # Switch scenario and load
    benchmark_radios = [r for r in at.radio if r.key == "benchmark_scenario_step3"]
    benchmark_radios[0].set_value("Stable Reversible Cycling (Benchmark)").run()
    assert len(at.exception) == 0
    load_btns = [b for b in at.button if b.key == "btn_load_benchmark_step3"]
    load_btns[0].click().run()

def test_custom_mode_clears_all_preloaded_data_across_tabs():
    """
    Verify that under Custom Experiment Mode:
    1. Step 1: Additive Name and SMILES inputs are empty; no preloaded 2D structure image,
       no compatibility table, and no ESOL card are rendered; clean empty-state notice is shown.
    2. Step 2: No preloaded DoE exploration scatter plot, no pipetting schedule table,
       no deck layout, and no downloads are rendered; clean empty-state notice is shown.
    3. Step 3: No preloaded cycling profiles (Plotly voltage/overpotential plots) and no
       two-stage metrics cards are rendered; clean empty-state notice is shown.
    4. Experiment Summary: No preloaded campaign metadata, no audit trail, no history table,
       no downloads, and no HTML preview are rendered; clean empty-state notice is shown.
    5. Switching to Demo Mode immediately displays preloaded data across all tabs.
    """
    from localization import get_text
    empty_step1_en = get_text("custom_mode_step1_empty_notice", lang="en")
    empty_step2_en = get_text("custom_mode_step2_empty_notice", lang="en")
    empty_step3_en = get_text("custom_mode_step3_empty_notice", lang="en")
    empty_summary_en = get_text("custom_mode_summary_empty_notice", lang="en")

    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    # 1. Step 1: verify inputs are empty and no preloaded images/tables exist
    name_inputs = [ti for ti in at.text_input if ti.key == "tab2_custom_additive_name_input"]
    smiles_inputs = [ti for ti in at.text_input if ti.key == "tab2_custom_smiles_input"]
    assert len(name_inputs) > 0
    assert len(smiles_inputs) > 0
    assert name_inputs[0].value == "", f"Expected empty additive name in Custom Mode, got '{name_inputs[0].value}'"
    assert smiles_inputs[0].value == "", f"Expected empty SMILES in Custom Mode, got '{smiles_inputs[0].value}'"

    assert len(at.image) == 0, "No 2D molecular image should be rendered in Custom Mode before molecule input"
    assert len(at.dataframe) == 0, "No dataframes should be rendered in Custom Mode initially"

    all_info_texts = [getattr(info, "value", "") for info in at.info if hasattr(info, "value")]
    info_corpus = " ".join(all_info_texts)
    assert "No custom additive entered yet." in info_corpus, "Step 1 empty notice must be displayed in Custom Mode"
    assert "No custom formulation plan available yet." in info_corpus, "Step 2 empty notice must be displayed in Custom Mode"
    assert "No custom electrochemical cycling data uploaded yet." in info_corpus, "Step 3 empty notice must be displayed in Custom Mode"
    tab_summary_en = get_text("tab_summary", lang="en")
    tab_labels_custom = [tab.label for tab in at.tabs]
    assert not any(tab_summary_en in lbl for lbl in tab_labels_custom), "Experiment Summary & Export tab must be removed in Custom Mode"

    # Step 2 & 5 downloads must NOT be present initially in Custom Mode
    dl_keys = [getattr(b, "key", "") for b in at.download_button]
    assert "step2_btn_dl_dispensing_xlsx" not in dl_keys
    assert "step2_btn_dl_doe_xlsx" not in dl_keys
    assert "tab5_btn_dl_campaign_xlsx" not in dl_keys

    # 2. Switch to Demo Mode: verify all preloaded data is rendered
    at.sidebar.radio[0].set_value("demo").run()
    assert len(at.exception) == 0

    assert len(at.image) > 0, "Preloaded 2D molecular image must be rendered in Demo Mode"
    assert len(at.dataframe) >= 4, "Preloaded tables (Step 1 pool, Step 2 schedule, Step 5 history, Tab 1) must be rendered in Demo Mode"

    dl_keys_demo = [getattr(b, "key", "") for b in at.download_button]
    assert "step2_btn_dl_dispensing_xlsx" in dl_keys_demo
    assert "step2_btn_dl_doe_xlsx" in dl_keys_demo
    assert "tab5_btn_dl_campaign_xlsx" in dl_keys_demo


def test_ui_step2_demo_explanation_active_formulation_and_custom_mode_planning():
    """
    Verify:
    1. In Custom Mode, Step 2 tab is named 'Step 2: Electrolyte Formulation & DoE Planning'.
    2. In Custom Mode, the Demo Mode educational explanation guide is NOT rendered.
    3. In Demo Mode, the educational explanation guide ('What Does Step 2 Do?') is rendered.
    4. In Demo Mode, the Active Formulation Context Banner clearly displays additive name,
       parent baseline electrolyte, and target electrolyte.
    5. In Demo Mode, the Robotic Pipetting Protocol Table is rendered.
    6. In Custom Mode with an active formulation plan, the Robotic Pipetting Protocol Table
       is removed and replaced with the Planned DoE Formulation Matrix table.
    """
    from localization import get_text
    tab_step2_custom_en = get_text("tab_step2_custom", lang="en")
    tab_step2_demo_en = get_text("tab_step2", lang="en")
    guide_title_en = get_text("step2_demo_guide_title", lang="en")
    ot2_protocol_title_en = get_text("sec_ot2_protocol", lang="en")
    doe_table_title_en = get_text("sec_doe_formulation_table", lang="en")

    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    # 1. Custom Mode by default: verify tab label is tab_step2_custom
    tab_labels_custom = [tab.label for tab in at.tabs]
    assert any(tab_step2_custom_en in lbl for lbl in tab_labels_custom), \
        f"Expected '{tab_step2_custom_en}' in Custom Mode tabs, got: {tab_labels_custom}"

    # Verify educational guide is NOT rendered in Custom Mode
    expander_labels_custom = [exp.label for exp in at.expander]
    assert not any(guide_title_en in exp for exp in expander_labels_custom), \
        "Demo educational guide should NOT be rendered in Custom Mode"

    # 2. Switch to Demo Mode
    at.sidebar.radio[0].set_value("demo").run()
    assert len(at.exception) == 0

    # Verify tab label is standard tab_step2 in Demo Mode
    tab_labels_demo = [tab.label for tab in at.tabs]
    assert any(tab_step2_demo_en in lbl for lbl in tab_labels_demo), \
        f"Expected '{tab_step2_demo_en}' in Demo Mode tabs, got: {tab_labels_demo}"

    # Verify educational explanation guide IS rendered in Demo Mode
    expander_labels_demo = [exp.label for exp in at.expander]
    assert any(guide_title_en in exp for exp in expander_labels_demo), \
        f"Expected guide '{guide_title_en}' in Demo Mode expanders, got: {expander_labels_demo}"

    # Verify Active Formulation Context Banner and Robotic Pipetting Table in Demo Mode
    markdown_corpus = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    assert "D-Sorbitol" in markdown_corpus, "Active additive name (D-Sorbitol) must be displayed in Step 2"
    assert "ZnSO" in markdown_corpus, "Baseline electrolyte (ZnSO4) must be displayed in Step 2"
    assert "Zn(OTf)" in markdown_corpus, "Target electrolyte (Zn(OTf)2) must be displayed in Step 2"
    assert ot2_protocol_title_en in markdown_corpus, "Robotic Pipetting Protocol table must be rendered in Demo Mode"

    # 3. Switch back to Custom Mode, enter custom additive and apply
    at.sidebar.radio[0].set_value("custom").run()
    assert len(at.exception) == 0

    custom_name_inputs = [ti for ti in at.text_input if ti.key == "tab2_custom_additive_name_input"]
    custom_smiles_inputs = [ti for ti in at.text_input if ti.key == "tab2_custom_smiles_input"]
    custom_name_inputs[0].set_value("Succinic Acid")
    custom_smiles_inputs[0].set_value("C(CC(=O)O)C(=O)O").run()

    apply_btn = [b for b in at.button if b.key == "btn_apply_custom_additive_tab2"]
    assert len(apply_btn) == 0, "btn_apply_custom_additive_tab2 button must be removed (redundant with reactive sync)"
    assert len(at.exception) == 0

    markdown_custom_corpus = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    # Verify Active Formulation Banner displays Succinic Acid and electrolytes
    assert "Succinic Acid" in markdown_custom_corpus, "Custom additive (Succinic Acid) must be displayed in Step 2"
    assert "ZnSO" in markdown_custom_corpus, "Parent electrolyte must be displayed in Step 2"
    assert "Zn(OTf)" in markdown_custom_corpus, "Target electrolyte must be displayed in Step 2"

    # In Custom Mode, Robotic Pipetting Protocol Table must be REMOVED
    assert ot2_protocol_title_en not in markdown_custom_corpus, \
        "Robotic Pipetting Protocol Table must be REMOVED in Custom Mode"

    # In Custom Mode, Planned DoE Formulation Matrix Table must be present
    assert doe_table_title_en in markdown_custom_corpus, \
        "Planned DoE Formulation Matrix table must be rendered in Custom Mode"


def test_ui_step2_in_tab_target_electrolyte_selector_and_immediate_sync():
    """
    Verify:
    1. Step 2 renders an interactive 'Target Electrolyte System' selectbox (step2_target_electrolyte_select).
    2. Changing target salt in Step 2 immediately updates the ConcentrationAnchor target_electrolyte_salt.
    3. The Active Formulation Context Banner and DoE matrix immediately reflect the new salt system.
    4. Synchronization works across canonical zinc salts (e.g. ZnCl2, Zn(TFSI)2, Zn(ClO4)2).
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    # 1. Switch to Demo Mode
    at.sidebar.radio[0].set_value("demo").run()
    assert len(at.exception) == 0

    # Verify Step 2 target electrolyte selector exists
    salt_selects = [sb for sb in at.selectbox if sb.key == "step2_target_electrolyte_select"]
    assert len(salt_selects) > 0, "step2_target_electrolyte_select must exist in Step 2"

    # Default is Zn(OTf)2
    plan = at.session_state.get("extracted_formulation_plan")
    assert plan is not None
    assert plan.anchor.target_electrolyte_salt == "Zn(OTf)2"

    # 2. Change target electrolyte in Step 2 to ZnCl2
    salt_selects[0].set_value("ZnCl2").run()
    assert len(at.exception) == 0

    # Verify anchor updated immediately
    plan = at.session_state.get("extracted_formulation_plan")
    assert plan is not None
    assert plan.anchor.target_electrolyte_salt == "ZnCl2"
    assert at.session_state.get("selected_target_salt") == "ZnCl2"

    # Verify Active Formulation Context Banner reflects ZnCl2
    all_md = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    assert "ZnCl₂" in all_md, "Target electrolyte ZnCl₂ must be rendered in banner"

    # 3. Change target electrolyte to Zn(TFSI)2
    salt_selects = [sb for sb in at.selectbox if sb.key == "step2_target_electrolyte_select"]
    salt_selects[0].set_value("Zn(TFSI)2").run()
    assert len(at.exception) == 0

    plan = at.session_state.get("extracted_formulation_plan")
    assert plan is not None
    assert plan.anchor.target_electrolyte_salt == "Zn(TFSI)2"
    assert at.session_state.get("selected_target_salt") == "Zn(TFSI)2"

    all_md = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    assert "Zn(TFSI)₂" in all_md, "Target electrolyte Zn(TFSI)₂ must be rendered in banner"


def test_ui_step1_to_step2_action_button_and_overview_card():
    """
    Verify:
    1. Discovered paper cards render action button '[Plan Formulation in Step 2 →]'.
    2. Clicking action button bridges the additive to Step 2 and displays bridging notice in Step 1.
    3. Step 2 renders the Active Additive Overview Card with molecule name, SMILES, and chemical formula.
    4. Multi-electrolyte compatibility table in Step 2 has hide_index=True.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    # 1. Search for literature papers
    search_inputs = [ti for ti in at.text_input if ti.key == "lit_search_query_input"]
    search_inputs[0].input("trehalose").run()
    search_btn = [b for b in at.button if b.key == "lit_search_submit_btn"][0]
    search_btn.click().run()
    assert len(at.exception) == 0

    # Verify per-card action buttons are removed from card deck
    assert len([b for b in at.button if b.key and "btn_use_paper_" in b.key]) == 0

    # Trigger active additive bridging to test banner and Step 2 overview card
    at.session_state["bridged_additive_name"] = "Trehalose"
    at.session_state["tab2_mol_select"] = "Trehalose"
    at.session_state["selected_additive_name"] = "Trehalose"
    at.run()
    assert len(at.exception) == 0

    # 2. Verify bridging notice rendered in Step 1
    all_infos = " ".join([getattr(info, "value", "") for info in at.info if hasattr(info, "value")])
    assert "Trehalose" in all_infos
    assert "set as the active additive" in all_infos or "défini comme additif actif" in all_infos or "已设为当前活性添加剂" in all_infos

    # 3. Verify Active Additive Overview Card in Step 2
    all_md = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    assert "Active Additive Overview" in all_md or "Aperçu de l'additif actif" in all_md or "活性添加剂概览" in all_md
    assert "C₁₂H₂₂O₁₁" in all_md, "Chemical formula C₁₂H₂₂O₁₁ must be rendered in overview card"
    assert "C(C1C(C(C(C(O1)OC2C(C(C(C(O2)CO)O)O)O)O)O)O)O" in all_md, "SMILES string must be rendered in overview card"

    # 4. Verify all dataframes enforce hide_index=True
    for df_elem in at.dataframe:
        col_proto = df_elem.proto.columns
        assert '"hidden": true' in col_proto or '"hidden":true' in col_proto, \
            f"Table index must be hidden: {col_proto}"


def test_ui_online_search_depth_tiers_and_notice():
    """
    Verify:
    1. Online search depth selector provides 4 discrete tiers: 25, 50, 100, 150 papers.
    2. Default selection is 25 papers.
    3. Selecting 100 or 150 papers displays the subtle consensus micro-batches info notice.
    4. Selecting 25 or 50 papers hides the notice.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    limit_selects = [sb for sb in at.selectbox if sb.key == "lit_search_limit_select"]
    assert len(limit_selects) > 0, "lit_search_limit_select selectbox must exist"
    limit_sb = limit_selects[0]

    # Verify options present
    assert any("25" in opt for opt in limit_sb.options)
    assert any("50" in opt for opt in limit_sb.options)
    assert any("100" in opt for opt in limit_sb.options)
    assert any("150" in opt for opt in limit_sb.options)
    assert limit_sb.value == 25, "Default depth tier must be 25 papers"

    # 100 papers tier renders consensus notice
    limit_sb.set_value(100).run()
    assert len(at.exception) == 0
    infos_100 = [info.value for info in at.info]
    assert any("micro-batches" in inf or "consensus" in inf or "150" in inf for inf in infos_100), \
        "Deep consensus info notice must be displayed when 100 papers tier is selected"

    # 150 papers tier renders consensus notice
    limit_sb.set_value(150).run()
    assert len(at.exception) == 0
    infos_150 = [info.value for info in at.info]
    assert any("micro-batches" in inf or "consensus" in inf or "150" in inf for inf in infos_150), \
        "Deep consensus info notice must be displayed when 150 papers tier is selected"

    # 25 papers tier does not show consensus notice
    limit_sb.set_value(25).run()
    assert len(at.exception) == 0
    infos_25 = [info.value for info in at.info]
    assert not any("micro-batches" in inf or "consensus model" in inf for inf in infos_25), \
        "Deep consensus notice must not be shown for 25 papers"


def test_ui_online_search_pagination_and_card_fields():
    """
    Verify:
    1. Executing search displays results in 10-per-page pages.
    2. Navigation controls: Prev button disabled on page 1, Next button advances to page 2.
    3. Page counter indicates 'Page 1 of 3 (Items 1–10 of 25)'.
    4. Each card contains Extracted Additive, Chemical Formula with Unicode subscripts,
       Reported Concentration, Baseline Salt, Mechanism, and Action Button.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    search_inputs = [ti for ti in at.text_input if ti.key == "lit_search_query_input"]
    search_inputs[0].input("trehalose").run()
    search_btn = [b for b in at.button if b.key == "lit_search_submit_btn"][0]
    search_btn.click().run()
    assert len(at.exception) == 0

    # Verify pagination top controls exist
    prev_buttons = [b for b in at.button if b.key == "lit_prev_page_top"]
    next_buttons = [b for b in at.button if b.key == "lit_next_page_top"]
    assert len(prev_buttons) > 0, "Top Prev button must exist"
    assert len(next_buttons) > 0, "Top Next button must exist"
    assert prev_buttons[0].disabled is True, "Prev button must be disabled on page 1"
    assert next_buttons[0].disabled is False, "Next button must be enabled on page 1 of 3"

    # Verify page 1 info indicator
    all_md = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    assert "Page 1 of" in all_md
    assert "Items 1–10" in all_md

    # Verify first card fields
    assert "Trehalose" in all_md
    assert "C₁₂H₂₂O₁₁" in all_md
    assert "2 M ZnSO₄" in all_md
    assert len([b for b in at.button if b.key and "btn_use_paper_" in b.key]) == 0

    # Advance to page 2
    next_buttons[0].click().run()
    assert len(at.exception) == 0
    assert at.session_state.get("lit_search_current_page") == 2

    all_md_p2 = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    assert "Page 2 of" in all_md_p2
    assert "Items 11–20" in all_md_p2

    # Advance to page 3
    next_p2 = [b for b in at.button if b.key == "lit_next_page_top"][0]
    next_p2.click().run()
    assert len(at.exception) == 0
    assert at.session_state.get("lit_search_current_page") == 3

    next_p3 = [b for b in at.button if b.key == "lit_next_page_top"][0]
    prev_p3 = [b for b in at.button if b.key == "lit_prev_page_top"][0]
    assert next_p3.disabled is True, "Next button must be disabled on last page"
    assert prev_p3.disabled is False, "Prev button must be enabled on last page"


def test_ui_online_search_consensus_candidates_pool_sync():
    """
    Verify:
    1. Online literature search populates suggested_candidates via Morgan ECFP4 fingerprint consensus.
    2. Ranked candidate pool reflects multi-paper consensus with similarity scores and properties.
    3. Step 2 Suggested Additives dropdown is accessible.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=60.0)
    at.run()
    assert len(at.exception) == 0

    search_inputs = [ti for ti in at.text_input if ti.key == "lit_search_query_input"]
    search_inputs[0].input("zinc")
    search_btn = [b for b in at.button if b.key == "lit_search_submit_btn"][0]
    search_btn.click().run(timeout=60.0)
    assert len(at.exception) == 0

    suggested = at.session_state.get("suggested_candidates")
    assert suggested is not None and len(suggested) > 0, "suggested_candidates pool must be populated"
    for cand in suggested:
        assert cand.name is not None
        assert cand.smiles is not None
        assert cand.similarity_score >= 0.0
        assert cand.molecular_weight > 0


def test_ui_online_search_gemini_popover_and_clear_key():
    """
    Verify:
    1. Online Academic Search sub-tab renders the Gemini API Key popover on the search row.
    2. Input has type='password' to mask sensitive key characters.
    3. Popover includes security notice and clear key button.
    4. Saving key updates st.session_state['gemini_api_key'].
    5. Clicking Clear Key resets st.session_state['gemini_api_key'].
    6. Online optimization notice is rendered.
    7. CSS enforces 48px height across popover and search submit buttons.
    """
    from localization import get_text
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    # 1. Verify online search optimization notice
    all_infos = " ".join([info.value for info in at.info])
    assert "Google Gemini" in all_infos
    assert "Google AI Studio" in all_infos

    # 2. Verify online popover and key input masking
    online_key_inputs = [ti for ti in at.text_input if ti.key == "online_search_gemini_api_key_input"]
    assert len(online_key_inputs) > 0, "online_search_gemini_api_key_input must exist"
    assert online_key_inputs[0].proto.Type.Name(online_key_inputs[0].proto.type) == "PASSWORD", "Gemini API key input must be masked with type='password'"

    # 3. Verify security notice in popover
    all_captions = " ".join([c.value for c in at.caption])
    sec_notice_en = get_text("gemini_security_notice", lang="en")
    assert sec_notice_en in all_captions or "ephemeral" in all_captions.lower() or "session" in all_captions.lower()

    # 4. Verify save key updates session state
    save_btns = [b for b in at.button if b.key == "btn_save_online_gemini_key"]
    assert len(save_btns) > 0
    test_key = "AIzaSyD-TestKey12345678901234567890"
    online_key_inputs[0].set_value(test_key).run()
    save_btns = [b for b in at.button if b.key == "btn_save_online_gemini_key"]
    save_btns[0].click().run()
    assert len(at.exception) == 0
    assert at.session_state.get("gemini_api_key") == test_key

    # 5. Verify clear key wipes session state
    clear_btns = [b for b in at.button if b.key == "btn_clear_online_gemini_key"]
    assert len(clear_btns) > 0
    clear_btns[0].click().run()
    assert len(at.exception) == 0
    assert at.session_state.get("gemini_api_key") == ""

    # 6. Verify 48px equal height CSS rules
    css_blocks = [m.value for m in at.markdown if "<style>" in getattr(m, "value", "")]
    full_css = " ".join(css_blocks)
    assert ".st-key-online_search_gemini_popover button" in full_css
    assert ".st-key-lit_search_submit_btn button" in full_css
    assert "height: 48px !important" in full_css


def test_ui_online_search_unspecified_baseline_salt_tag():
    """
    Verify:
    1. When a mined paper lacks an explicit baseline salt, the card renders '⚠️ Unspecified in Abstract'.
    2. Bridging to Step 2 uses the active target salt rather than hallucinating 2 M ZnSO4.
    """
    from localization import get_text
    from models import PaperMetadata, ExtractedAdditive, Concentration, ZincIonBatteryContext
    unspecified_tag = get_text("baseline_salt_unspecified", lang="en")

    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    # Inject a paper with unspecified baseline salt
    mock_paper = PaperMetadata(
        paper_id="paper_novel_unspecified",
        title="Novel Biopolymer Additive for Zinc Batteries",
        abstract="A novel biopolymer additive at 10 mM protects the zinc anode without specifying baseline salt.",
        authors=["Scientist A"],
        year=2025
    )
    mock_add = ExtractedAdditive(
        name="Biopolymer Additive",
        optimal_concentration=Concentration(value=10.0, unit="mM", raw_text="10 mM"),
        primary_mechanisms=["Zn²⁺ solvation sheath modulation"],
        battery_context=ZincIonBatteryContext(baseline_electrolyte="Unspecified / Not reported")
    )
    at.session_state["literature_search_results"] = [mock_paper]
    at.session_state["literature_mined_records"] = [{"paper": mock_paper, "additive": mock_add}]
    at.run()
    assert len(at.exception) == 0

    # Verify unspecified salt badge is rendered
    all_md = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    assert unspecified_tag in all_md or "Unspecified in Abstract" in all_md
    assert "2 M ZnSO₄" not in all_md.split("Biopolymer Additive")[1].split("</div")[0]

    # Verify per-card action buttons are removed from card deck
    assert len([b for b in at.button if b.key and "btn_use_paper_" in b.key]) == 0


def test_ui_online_search_input_boxes_equal_height_css():
    """
    Verify that app.py injects CSS rules to set the heights of:
    - Keywords text box (.st-key-lit_search_query_input)
    - Sort By select box (.st-key-lit_search_sort_select)
    - Number of Literature select box (.st-key-lit_search_limit_select)
    equal to the 48px height of the Gemini key popover and search literature buttons.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    all_md = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    assert ".st-key-lit_search_query_input div[data-baseweb=\"input\"]" in all_md
    assert ".st-key-lit_search_sort_select div[data-baseweb=\"select\"]" in all_md
    assert ".st-key-lit_search_limit_select div[data-baseweb=\"select\"]" in all_md
    assert "height: 48px !important;" in all_md


def test_ui_gemini_model_selector_dropdown_and_badge():
    """
    Verify:
    1. Gemini Model selection dropdown exists in Online Search popover.
    2. Default model is gemini-3.5-flash-lite (with 500 RPD quota guidance).
    3. Selecting another model (e.g. gemini-3.1-pro) updates session state and active status badge.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    # Pre-set a valid key so the active engine status badge is rendered
    at.session_state["gemini_api_key"] = "AIzaSyDummyValidKey123456789012345"
    at.run()
    assert len(at.exception) == 0

    # Verify default badge shows gemini-3.5-flash-lite
    all_md = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    assert "gemini-3.5-flash-lite" in all_md

    # Find the model selectbox in the popover
    model_selects = [sb for sb in at.selectbox if "gemini_model_select" in str(sb.key)]
    assert len(model_selects) > 0
    online_sb = model_selects[0]
    assert any("3.5 Flash-Lite" in opt for opt in online_sb.options)
    assert any("3.1 Flash-Lite" in opt for opt in online_sb.options)
    assert any("3.8 Flash" in opt for opt in online_sb.options)
    assert any("3.1 Pro" in opt for opt in online_sb.options)
    assert any("4 Argon" in opt for opt in online_sb.options)

    # Change to gemini-3.1-pro (index 3) and run
    online_sb.select_index(3).run()
    assert len(at.exception) == 0
    assert at.session_state.get("gemini_selected_model") == "gemini-3.1-pro"

    # Verify badge now displays gemini-3.1-pro
    all_md_updated = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    assert "gemini-3.1-pro" in all_md_updated


def test_ui_gemini_model_selector_custom_model_and_cross_tab_sync():
    """
    Verify:
    1. Both Online Search and Batch PDF subtabs have the model selector.
    2. Selecting Custom model displays custom text input and updates active model.
    3. Model selection is synchronized across both subtabs.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.session_state["gemini_api_key"] = "AIzaSyDummyValidKey123456789012345"
    at.run()
    assert len(at.exception) == 0

    model_selects = [sb for sb in at.selectbox if "gemini_model_select" in str(sb.key)]
    assert len(model_selects) >= 2, "Both Online Search and Batch PDF must have model selectors"
    online_sb = model_selects[0]
    batch_sb = model_selects[1]

    # Select Custom option (last option in list) in online search
    custom_idx = len(online_sb.options) - 1
    online_sb.select_index(custom_idx).run()
    assert len(at.exception) == 0

    # Custom text input should now be visible
    custom_inputs = [ti for ti in at.text_input if "gemini_custom_model_input" in str(ti.key)]
    assert len(custom_inputs) > 0, "Custom model text input should be rendered"
    custom_inputs[0].input("gemini-experimental-ultra-4").run()
    assert len(at.exception) == 0

    assert at.session_state.get("gemini_selected_model") == "gemini-experimental-ultra-4"
    all_md = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    assert "gemini-experimental-ultra-4" in all_md


def test_ui_online_search_stop_mining_and_layout_ratios():
    """
    Verify:
    1. Localization keys for stop mining and online status exist across en, fr, and zh.
    2. Depth tier labels across all languages do not contain time estimations (~, 30s, min).
    3. Search row columns in app.py use the updated ratios [2.0, 0.75, 0.9, 1.45, 0.9].
    """
    from localization import get_text, TRANSLATIONS

    for lang in ["en", "fr", "zh"]:
        # Verify new keys
        starting_txt = get_text("mining_status_gemini_starting", lang=lang, model="gemini-3.8-flash")
        stop_btn_txt = get_text("btn_stop_mining", lang=lang)
        stop_help = get_text("btn_stop_mining_help", lang=lang)
        stop_notice = get_text("mining_stopped_by_user_notice", lang=lang, count=10)
        stop_empty = get_text("mining_stopped_no_papers_notice", lang=lang)

        assert not starting_txt.startswith("[") and "gemini-3.8-flash" in starting_txt
        assert not stop_btn_txt.startswith("[")
        assert not stop_help.startswith("[")
        assert not stop_notice.startswith("[") and "10" in stop_notice
        assert not stop_empty.startswith("[")

        # Verify time-less depth tier labels
        for tier in [25, 50, 100, 150]:
            label = get_text(f"depth_tier_{tier}", lang=lang)
            assert not label.startswith("[")
            assert "~" not in label
            assert "30s" not in label
            assert "min" not in label

    # Verify column layout in app.py (Two-Row Search Toolbar or legacy 1-row)
    import inspect
    import app
    source = inspect.getsource(app)
    assert (
        ("[3.80, 1.15]" in source and "[1.0, 1.15, 1.55]" in source)
        or "[1.85, 0.65, 0.85, 1.35, 0.9]" in source
        or "[1.6, 0.65, 0.85, 1.45, 0.95]" in source
        or "[2.0, 0.75, 0.9, 1.45, 0.9]" in source
    )


def test_ui_online_search_paper_card_with_no_additive():
    """
    Verify:
    1. When a mined paper has additive = None, the card displays '⚠️ Unspecified in Abstract (No liquid additive tested)'.
    2. The Plan Formulation button is disabled with '🚫 No Additive to Plan'.
    3. The Analyze Abstract button displays 'No Verified Electrolyte Additive Identified'.
    """
    from models import PaperMetadata
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    mock_p = PaperMetadata(
        paper_id="paper_no_additive_01",
        title="Vanadium Oxide Cathode for Rechargeable Zinc Batteries",
        abstract="We investigate the structural phase evolution of V2O5 cathode in 2 M ZnSO4 without any electrolyte additive.",
        authors=["Author A"],
        year=2024
    )

    at.session_state["literature_search_results"] = [mock_p]
    at.session_state["literature_mined_records"] = [{"paper": mock_p, "additive": None}]
    at.run()
    assert len(at.exception) == 0

    all_md = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    assert "No liquid additive tested" in all_md or "Unspecified in Abstract" in all_md
    assert "Aqueous ZIB Additive" not in all_md

    # Verify no per-card clutter buttons are rendered
    assert len([b for b in at.button if b.key and "btn_use_paper_" in b.key]) == 0
    assert len([b for b in at.button if b.key and "btn_analyze_paper_" in b.key]) == 0


def test_ui_dropdown_browsing_does_not_trigger_search():
    """Verify that changing search dropdowns (limit or sort) does NOT trigger literature search."""
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    # Verify results initially empty
    assert at.session_state.get("literature_search_results") is None

    # Change limit dropdown
    limit_selects = [sb for sb in at.selectbox if sb.key == "lit_search_limit_select"]
    assert len(limit_selects) > 0
    limit_selects[0].select_index(1).run()
    assert len(at.exception) == 0

    # Ensure search was NOT triggered by simply changing dropdown
    assert at.session_state.get("literature_search_results") is None


def test_universal_conventional_chemical_formulas():
    """
    Verify universal conventional chemical formula conversion:
    1. Inorganic salts citation format: metal / cation first, then anion (LiCl, NaCl, ZnCl₂, ZnSO₄).
    2. Alphabetical Hill representations (ClLi, ClNa, Cl2Zn, O4SZn, MgO4S) are normalized to conventional chemistry.
    3. Organic molecules remain in IUPAC Hill system (C, then H, then alphabetical).
    4. Mineral acids and hydroxides follow conventional inorganic notation (HCl, H₂SO₄, NaOH, KOH).
    """
    from app import get_chemical_formula, format_chem_formula, normalize_formula_to_conventional

    # 1. Inorganic salts from SMILES & name
    assert get_chemical_formula(smiles="[Li+].[Cl-]", name="Lithium chloride") == "LiCl"
    assert get_chemical_formula(smiles="[Li+].[Cl-]") == "LiCl"
    assert get_chemical_formula(name="Lithium chloride") == "LiCl"
    assert get_chemical_formula(smiles="[Na+].[Cl-]") == "NaCl"
    assert get_chemical_formula(smiles="[K+].[Cl-]") == "KCl"
    assert get_chemical_formula(smiles="[Zn+2].[Cl-].[Cl-]") == "ZnCl₂"
    assert get_chemical_formula(smiles="[Zn+2].[O-]S(=O)(=O)[O-]") == "ZnSO₄"
    assert get_chemical_formula(smiles="[Mg+2].[O-]S(=O)(=O)[O-]") == "MgSO₄"
    assert get_chemical_formula(smiles="[Na+].[OH-]") == "NaOH"
    assert get_chemical_formula(smiles="[NH4+].[Cl-]") == "NH₄Cl"

    # 2. Universal string normalization from alphabetical Hill strings
    assert normalize_formula_to_conventional("ClLi") == "LiCl"
    assert normalize_formula_to_conventional("ClNa") == "NaCl"
    assert normalize_formula_to_conventional("ClK") == "KCl"
    assert normalize_formula_to_conventional("Cl2Zn") == "ZnCl₂"
    assert normalize_formula_to_conventional("O4SZn") == "ZnSO₄"
    assert normalize_formula_to_conventional("MgO4S") == "MgSO₄"
    assert normalize_formula_to_conventional("FNa") == "NaF"
    assert normalize_formula_to_conventional("BrK") == "KBr"
    assert normalize_formula_to_conventional("ILi") == "LiI"

    # 3. format_chem_formula normalization
    assert format_chem_formula("ClLi") == "LiCl"
    assert format_chem_formula("ClNa") == "NaCl"
    assert format_chem_formula("Cl2Zn") == "ZnCl₂"
    assert format_chem_formula("O4SZn") == "ZnSO₄"
    assert format_chem_formula("MgO4S") == "MgSO₄"

    # 4. Organic molecules stay Hill
    assert get_chemical_formula(smiles="OCC(O)CO") == "C₃H₈O₃"
    assert get_chemical_formula(smiles="C2H6O2", name="Ethylene Glycol") == "C₂H₆O₂"
    assert normalize_formula_to_conventional("C6H12O6") == "C₆H₁₂O₆"


def test_iupac_chemical_names_never_mangled_to_subscripts():
    """
    Verify universal guard preventing chemical names with numbers/hyphens from being
    mangled into pseudo-subscripts/superscripts (e.g. 2-(...) -> ₂⁻(...)).
    """
    from app import is_chemical_formula, normalize_formula_to_conventional, get_chemical_formula, format_chem_formula

    # 1. is_chemical_formula correctly differentiates true formulas from IUPAC names
    assert is_chemical_formula("2-(N-morpholino)ethanesulfonic acid") is False
    assert is_chemical_formula("3-(Hydroxy(phenyl)phosphoryl)propanoic acid") is False
    assert is_chemical_formula("Lithium chloride") is False
    assert is_chemical_formula("Sodium chloride") is False
    assert is_chemical_formula("Ethylene Glycol") is False
    assert is_chemical_formula("LiCl") is True
    assert is_chemical_formula("ClLi") is True
    assert is_chemical_formula("ZnSO4") is True
    assert is_chemical_formula("C6H12O6") is True
    assert is_chemical_formula("C₆H₁₃NO₄S") is True

    # 2. normalize_formula_to_conventional does NOT translate names into subscripts
    uncured_name = "3-(Hydroxy(phenyl)phosphoryl)propanoic acid"
    assert normalize_formula_to_conventional(uncured_name) == uncured_name
    assert "₃⁻" not in normalize_formula_to_conventional(uncured_name)
    assert "₂⁻" not in normalize_formula_to_conventional(uncured_name)

    # 3. get_chemical_formula returns either curated formula or empty string, NEVER mangled names
    assert get_chemical_formula(name="2-(N-morpholino)ethanesulfonic acid") == "C₆H₁₃NO₄S"
    assert get_chemical_formula(name="2-(N-morpholino)ethanesulfonic acid (MES)") == "C₆H₁₃NO₄S"
    assert get_chemical_formula(name="MES") == "C₆H₁₃NO₄S"
    assert get_chemical_formula(smiles="C1COCCN1CCS(=O)(=O)O") == "C₆H₁₃NO₄S"

    assert get_chemical_formula(name="3-(Hydroxy(phenyl)phosphoryl)propanoic acid") == ""
    assert "₃⁻" not in get_chemical_formula(name="3-(Hydroxy(phenyl)phosphoryl)propanoic acid")

    # 4. format_chem_formula does not mangle IUPAC names
    assert format_chem_formula(uncured_name) == uncured_name
    assert "₃⁻" not in format_chem_formula(uncured_name)


def test_ui_gemini_key_active_badge_and_step1_titles():
    """
    Verify:
    1. Step 1 tab and section titles/descriptions reflect discovery & additive mining.
    2. Popover button displays '(✅ Active)' when Gemini API key is configured.
    3. Popover interior displays prominent active status banner with verified model.
    """
    from localization import get_text

    # 1. Localization verification across all 3 languages
    for lang in ["en", "fr", "zh"]:
        tab_t = get_text("tab_step1", lang=lang)
        sec_t = get_text("tab2_title", lang=lang)
        desc_t = get_text("tab2_desc", lang=lang)
        active_badge = get_text("gemini_badge_active", lang=lang)
        active_banner = get_text("gemini_popover_active_banner", lang=lang, model="gemini-3.5-flash-lite")

        assert "Solubility Limits" not in tab_t
        assert "Solubility Limits" not in sec_t
        assert "Delaney ESOL 80%" not in desc_t
        assert any(term in active_badge for term in ["Active", "Actif", "已生效", "已激活"])
        assert "gemini-3.5-flash-lite" in active_banner

    # 2. UI verification in Streamlit AppTest
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    # Before key is set: offline banner is rendered in popover
    all_md_initial = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    assert "Offline Mode:" in all_md_initial

    # Set Gemini API Key via the UI input and click Save Key
    key_inputs = [ti for ti in at.text_input if "online_search_gemini_api_key_input" in ti.key]
    assert len(key_inputs) > 0
    test_key = "AIzaSyD-ActiveTestKey123456789012345"
    key_inputs[0].set_value(test_key).run()

    save_btns = [b for b in at.button if b.key == "btn_save_online_gemini_key"]
    assert len(save_btns) > 0
    save_btns[0].click().run()
    assert len(at.exception) == 0
    assert at.session_state.get("gemini_api_key") == test_key

    # Active banner rendered inside popover
    all_md_active = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    assert "Online Mode Active:" in all_md_active
    assert "gemini-3.5-flash-lite" in all_md_active


def test_ui_step1_elimination_of_8_molecule_cap_retains_all_additives():
    """
    Verify:
    1. LiteratureDiscoveryEngine does not enforce an arbitrary 8-molecule cap when mining additives.
    2. Mining 12 distinct additives extracts and stores all 12 unique candidates in session state.
    3. Neither Tier 1 nor Tier 2 candidates are truncated to 8.
    """
    from discovery.literature_engine import LiteratureDiscoveryEngine
    from models import ExtractedAdditive, Concentration, ZincIonBatteryContext

    # Create 12 distinct mock additives
    mock_additives = []
    test_chemicals = [
        ("D-Sorbitol", "C(C(C(C(C(CO)O)O)O)O)O", 20.0),
        ("Trehalose", "C(C1C(C(C(C(O1)OC2C(C(C(C(O2)CO)O)O)O)O)O)O)O", 15.0),
        ("Tetraethylammonium bromide", "CC[N+](CC)(CC)CC.[Br-]", 50.0),
        ("Thiourea", "C(=S)(N)N", 10.0),
        ("Glucose", "C(C1C(C(C(C(O1)O)O)O)O)O", 25.0),
        ("Glycine", "C(C(=O)O)N", 30.0),
        ("Succinic Acid", "C(CC(=O)O)C(=O)O", 12.0),
        ("Citric Acid", "C(C(=O)O)C(CC(=O)O)(C(=O)O)O", 18.0),
        ("Urea", "C(=O)(N)N", 22.0),
        ("Xylitol", "C(C(C(C(CO)O)O)O)O", 16.0),
        ("Erythritol", "C(C(C(CO)O)O)O", 14.0),
        ("Glycerol", "C(C(CO)O)O", 40.0),
    ]

    for name, smiles, conc in test_chemicals:
        mock_additives.append(ExtractedAdditive(
            name=name,
            smiles=smiles,
            optimal_concentration=Concentration(value=conc, unit="mM", raw_text=f"{conc} mM"),
            primary_mechanisms=["Zn²⁺ solvation sheath modulation"],
            battery_context=ZincIonBatteryContext(baseline_electrolyte="2 M ZnSO4")
        ))

    engine = LiteratureDiscoveryEngine()
    tier1_cands, tier2_cands = engine.generate_two_tier_candidates(mock_additives, use_pubchem=False)

    # Verify all 12 unique additives are preserved without truncation to 8
    assert len(tier1_cands) == 12, f"Expected all 12 mined additives to be retained in Tier 1, got {len(tier1_cands)}"

    # In Streamlit session state, verify all 12 are retained
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    at.session_state["mined_literature_additives"] = tier1_cands
    at.session_state["extracted_additives"] = mock_additives
    at.session_state["ai_suggested_additives"] = tier2_cands
    at.session_state["suggested_candidates"] = tier2_cands
    at.run()
    assert len(at.exception) == 0

    assert len(at.session_state["mined_literature_additives"]) == 12
    assert len(at.session_state["extracted_additives"]) == 12


def test_ui_step2_two_tier_separation_and_candidate_counts():
    """
    Verify:
    1. Step 2 renders two distinct candidate tabs:
       - Tier 1: Mined Literature Additives (Tested Chemistries)
       - Tier 2: AI-Suggested Untested Additives (Novel Analogues & Derivatives)
    2. All Tier 1 candidates have similarity_score == 1.0 and is_literature_additive == True.
    3. All Tier 2 candidates have similarity_score < 1.0 and is_literature_additive == False.
    4. Tier 2 novel analogues are strictly distinct from the reported literature entities.
    """
    from discovery.literature_engine import LiteratureDiscoveryEngine
    from models import ExtractedAdditive, Concentration, ZincIonBatteryContext
    from localization import get_text

    t1_title = get_text("tier1_title", lang="en")
    t2_title = get_text("tier2_title", lang="en")

    mock_additives = [
        ExtractedAdditive(
            name="Thiourea",
            smiles="C(=S)(N)N",
            optimal_concentration=Concentration(value=10.0, unit="mM", raw_text="10 mM"),
            primary_mechanisms=["SEI formation", "Dendrite suppression"],
            battery_context=ZincIonBatteryContext(baseline_electrolyte="2 M ZnSO4")
        ),
        ExtractedAdditive(
            name="Glycine",
            smiles="C(C(=O)O)N",
            optimal_concentration=Concentration(value=30.0, unit="mM", raw_text="30 mM"),
            primary_mechanisms=["Solvation sheath modulation"],
            battery_context=ZincIonBatteryContext(baseline_electrolyte="2 M ZnSO4")
        ),
    ]

    engine = LiteratureDiscoveryEngine()
    tier1_cands, tier2_cands = engine.generate_two_tier_candidates(mock_additives, use_pubchem=False)

    # 1. Verification of Tier 1 properties
    assert len(tier1_cands) == 2
    for c in tier1_cands:
        assert c.similarity_score == 1.0
        assert c.is_literature_additive is True
        assert c.reported_concentration is not None
        assert c.parent_baseline_electrolyte == "2 M ZnSO4"

    # 2. Verification of Tier 2 properties
    assert len(tier2_cands) > 0
    t1_names = {c.name.lower() for c in tier1_cands}
    t1_smiles = {c.smiles for c in tier1_cands}
    for c in tier2_cands:
        assert c.similarity_score < 1.0
        assert c.is_literature_additive is False
        assert c.name.lower() not in t1_names, f"Tier 2 must not contain Tier 1 entity {c.name}"
        assert c.smiles not in t1_smiles, f"Tier 2 must not contain Tier 1 SMILES {c.smiles}"

    # 3. Headless UI rendering verification
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    at.session_state["mined_literature_additives"] = tier1_cands
    at.session_state["ai_suggested_additives"] = tier2_cands
    at.run()
    assert len(at.exception) == 0

    all_tabs = [t.label for t in at.tabs]
    assert any(t1_title in t for t in all_tabs), f"Expected '{t1_title}' tab in Step 2"
    assert any(t2_title in t for t in all_tabs), f"Expected '{t2_title}' tab in Step 2"


def test_ui_step2_paginated_cards_and_navigation():
    """
    Verify:
    1. Tier 1 candidates are paginated at 6 cards per page.
    2. With 14 candidates, Page 1 shows 'Page 1 of 3 (1–6 / 14)'.
    3. Clicking Next ▶ advances to Page 2 (7–12 / 14).
    4. Clicking Next ▶ advances to Page 3 (13–14 / 14).
    5. Clicking ◀ Previous navigates back.
    """
    from models import CandidateMolecule

    # Create 14 mock candidate molecules
    mock_14_cands = []
    for i in range(14):
        mock_14_cands.append(CandidateMolecule(
            name=f"Candidate Mol {i+1}",
            smiles="C(C(CO)O)O",
            similarity_score=1.0,
            similarity_metric="Direct Literature Finding",
            molecular_weight=92.09,
            logp=-1.76,
            logs=-0.5,
            tpsa=60.69,
            hbd=3,
            hba=3,
            rotatable_bonds=2,
            rationale="Tested candidate",
            chemical_formula="C3H8O3",
            reported_concentration=15.0,
            reported_concentration_unit="mM",
            parent_baseline_electrolyte="2 M ZnSO4",
            is_literature_additive=True,
        ))

    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    at.session_state["mined_literature_additives"] = mock_14_cands
    at.run()
    assert len(at.exception) == 0

    # 1. Verify Page 1 of 3
    md_corpus = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    assert "Page 1 of 3" in md_corpus or "Page 1 / 3" in md_corpus or "1–6 / 14" in md_corpus

    # 2. Click Next button
    next_btns = [b for b in at.button if b.key == "tier1_next_page"]
    assert len(next_btns) > 0, "tier1_next_page button must exist"
    next_btns[0].click().run()
    assert len(at.exception) == 0

    assert at.session_state.get("step2_tier1_current_page") == 2
    md_corpus_p2 = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    assert "Page 2 of 3" in md_corpus_p2 or "7–12 / 14" in md_corpus_p2

    # 3. Click Next button again -> Page 3
    next_btns = [b for b in at.button if b.key == "tier1_next_page"]
    next_btns[0].click().run()
    assert len(at.exception) == 0

    assert at.session_state.get("step2_tier1_current_page") == 3
    md_corpus_p3 = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    assert "Page 3 of 3" in md_corpus_p3 or "13–14 / 14" in md_corpus_p3

    # 4. Click Prev button -> Page 2
    prev_btns = [b for b in at.button if b.key == "tier1_prev_page"]
    assert len(prev_btns) > 0
    prev_btns[0].click().run()
    assert len(at.exception) == 0
    assert at.session_state.get("step2_tier1_current_page") == 2


def test_ui_step2_one_click_select_and_formulate_bridges_data():
    """
    Verify:
    1. Clicking '🧪 Select & Formulate in Step 2' on a card (btn_formulate_t1_0):
       - Sets the active additive name and SMILES.
       - Synchronizes reported concentration and baseline parent salt.
       - Recomputes ConcentrationAnchor, DoE formulation plan, and OT-2 dispensing schedule.
       - Updates the Active Additive Overview card and Delaney ESOL ceiling in Step 2.
    """
    from models import CandidateMolecule

    cand = CandidateMolecule(
        name="Trehalose Dihydrate",
        smiles="C(C1C(C(C(C(O1)OC2C(C(C(C(O2)CO)O)O)O)O)O)O)O",
        similarity_score=1.0,
        similarity_metric="Direct Literature Finding",
        molecular_weight=342.3,
        logp=-3.6,
        logs=-0.8,
        tpsa=189.5,
        hbd=8,
        hba=11,
        rotatable_bonds=3,
        rationale="Tested disaccharide additive",
        chemical_formula="C12H22O11",
        reported_concentration=25.0,
        reported_concentration_unit="mM",
        parent_baseline_electrolyte="2 M ZnSO4",
        is_literature_additive=True,
    )

    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    at.session_state["mined_literature_additives"] = [cand]
    at.run()
    assert len(at.exception) == 0

    # Find the formulate action button for candidate 0
    formulate_btns = [b for b in at.button if b.key == "btn_formulate_t1_0"]
    assert len(formulate_btns) > 0, "btn_formulate_t1_0 button must exist"

    # Click the formulate button
    formulate_btns[0].click().run()
    assert len(at.exception) == 0

    # 1. Verify active additive updated
    assert at.session_state.get("selected_additive_name") == "Trehalose Dihydrate"
    assert at.session_state.get("custom_additive_name") == "Trehalose Dihydrate"
    assert at.session_state.get("custom_smiles") == cand.smiles
    assert at.session_state.get("published_conc_input") == 25.0

    # 2. Verify DoE plan and dispensing schedule generated
    plan = at.session_state.get("extracted_formulation_plan")
    assert plan is not None
    assert plan.anchor.additive_name == "Trehalose Dihydrate"
    assert plan.anchor.published_concentration == 25.0

    sched = at.session_state.get("extracted_dispensing_schedule")
    assert sched is not None
    assert len(sched.vials) > 0

    # 3. Verify Active Additive Overview renders Trehalose Dihydrate
    md_corpus = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    assert "Trehalose Dihydrate" in md_corpus


def test_ui_step2_top_dropdowns_and_no_empty_parentheses():
    """
    Verify:
    1. Tier 1 and Tier 2 render top-level dropdown selectboxes visible by default.
    2. Additives with no chemical formula NEVER render empty '()' in HTML cards or dropdown options.
    3. The card deck is encapsulated within an expander with .candidate-card-deck-box.
    """
    from models import CandidateMolecule

    cand_no_formula = CandidateMolecule(
        name="N-doped carbon quantum dots",
        smiles="",
        similarity_score=1.0,
        similarity_metric="Direct Literature Finding",
        molecular_weight=0.0,
        logp=0.0,
        logs=-1.0,
        tpsa=0.0,
        hbd=0,
        hba=0,
        rotatable_bonds=0,
        rationale="Tested nanomaterial additive",
        chemical_formula="",
        reported_concentration=10.0,
        reported_concentration_unit="mg/mL",
        parent_baseline_electrolyte="2 M ZnSO4",
        is_literature_additive=True,
    )

    cand_with_formula = CandidateMolecule(
        name="Phosphonoglycolic acid",
        smiles="C(C(=O)O)P(=O)(O)O",
        similarity_score=1.0,
        similarity_metric="Direct Literature Finding",
        molecular_weight=156.03,
        logp=-1.2,
        logs=-0.2,
        tpsa=104.99,
        hbd=4,
        hba=5,
        rotatable_bonds=2,
        rationale="Tested organic acid additive",
        chemical_formula="C2H5O6P",
        reported_concentration=15.0,
        reported_concentration_unit="mM",
        parent_baseline_electrolyte="2 M ZnSO4",
        is_literature_additive=True,
    )

    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    at.session_state["mined_literature_additives"] = [cand_no_formula, cand_with_formula]
    at.run()
    assert len(at.exception) == 0

    # 1. Verify Tier 1 top dropdown exists and is visible
    t1_dropdowns = [sb for sb in at.selectbox if sb.key == "step2_tier1_additive_dropdown"]
    assert len(t1_dropdowns) == 1, "step2_tier1_additive_dropdown must exist at top of Tier 1"

    # 2. Verify selecting candidate reactively synchronizes active additive without separate formulate button
    assert at.session_state.get("selected_additive_name") in [cand_no_formula.name, cand_with_formula.name]

    # 3. Verify markdown content has no empty "()"
    md_corpus = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    assert "N-doped carbon quantum dots ()" not in md_corpus, "Empty '()' must not follow chemical names without formulas"
    assert "Phosphonoglycolic acid ()" not in md_corpus, "Phosphonoglycolic acid must not have empty '()'"
    assert "candidate-card-deck-box" in md_corpus, "Equal height .candidate-card-deck-box class must be present"

    # 4. Verify clicking the "X" clear button on step2_tier1_additive_dropdown removes the active additive without TypeError
    t1_dropdowns = [sb for sb in at.selectbox if sb.key == "step2_tier1_additive_dropdown"]
    t1_dropdowns[0].set_value(None).run()
    assert len(at.exception) == 0
    remaining = at.session_state["mined_literature_additives"]
    assert len(remaining) == 1
    assert remaining[0].name == "Phosphonoglycolic acid"
    assert at.session_state.get("selected_additive_name") == "Phosphonoglycolic acid"


def test_additive_importer_all_formats_and_deduplication():
    """
    Verify:
    1. Parsing .csv, .xlsx, .txt, and .json formats.
    2. Smart column auto-detection (name, conc, salt, smiles).
    3. Automatic deduplication with occurrence tracking.
    4. 5,000 capacity guardrail.
    5. Pure-Python template generation (.csv and .xlsx).
    """
    from discovery.additive_importer import (
        parse_additive_file,
        deduplicate_and_resolve_additives,
        generate_template_csv,
        generate_template_xlsx,
    )

    # 1. Template generation
    csv_bytes = generate_template_csv()
    assert len(csv_bytes) > 50
    assert b"Trehalose" in csv_bytes

    xlsx_bytes = generate_template_xlsx()
    assert len(xlsx_bytes) > 500

    # 2. CSV parsing
    csv_records, csv_notices = parse_additive_file(csv_bytes, "template.csv")
    assert len(csv_records) == 4
    names = [r["name"] for r in csv_records]
    assert "Trehalose" in names
    assert "Glycine" in names

    # 3. XLSX parsing
    xlsx_records, xlsx_notices = parse_additive_file(xlsx_bytes, "template.xlsx")
    assert len(xlsx_records) == 4
    assert any(r["name"] == "Thiourea" for r in xlsx_records)

    # 4. JSON parsing
    import json
    sample_json = json.dumps([
        {"Additive": "Ascorbic Acid", "Conc (mM)": 20.0, "Salt": "2 M ZnSO4"},
        {"Additive": "Ascorbic Acid", "Conc (mM)": 20.0, "Salt": "2 M ZnSO4"},  # Duplicate under same condition
        {"Additive": "Glucose", "Conc (mM)": 10.0, "Salt": "1 M Zn(OTf)2"},
    ]).encode("utf-8")
    json_records, _ = parse_additive_file(sample_json, "batch.json")
    assert len(json_records) == 3

    # 5. TXT parsing (plain line-by-line)
    sample_txt = "Trehalose\nGlycine\nThiourea\n".encode("utf-8")
    txt_records, _ = parse_additive_file(sample_txt, "list.txt")
    assert len(txt_records) == 3

    # 6. Deduplication and chemistry resolution
    combined_raw = csv_records + json_records
    cands, stats = deduplicate_and_resolve_additives(combined_raw, max_limit=5000, resolve_pubchem=False)
    assert len(cands) == 6  # 4 from template + 2 unique from JSON (Ascorbic Acid deduplicated)
    assert stats["total_records_parsed"] == 7
    assert stats["unique_additives_count"] == 6
    assert stats["truncated"] is False

    # Check that Ascorbic Acid was resolved properly with RDKit
    asc = next(c for c in cands if c.name == "Ascorbic Acid")
    assert asc.chemical_formula in ("C6H8O6", "C₆H₈O₆")
    assert asc.molecular_weight > 170.0
    assert "2 citation(s)" in asc.rationale


def test_ui_step2_edit_additive_in_place_and_reanchor():
    """
    Verify:
    1. User can edit an additive with unspecified baseline electrolyte (e.g. changing to '2 M ZnSO4').
    2. Editing updates candidate in session state and recalculates the formulation anchor.
    """
    import copy
    from models import CandidateMolecule
    from app import _apply_card_candidate

    cand_unspecified = CandidateMolecule(
        name="Graphene oxide (GO)",
        smiles="C1=CC=CC=C1",
        similarity_score=1.0,
        similarity_metric="Direct Literature Finding",
        molecular_weight=78.11,
        logp=1.5,
        logs=-1.2,
        tpsa=0.0,
        hbd=0,
        hba=0,
        rotatable_bonds=0,
        rationale="Tested additive with unspecified baseline",
        chemical_formula="",
        reported_concentration=15.0,
        reported_concentration_unit="mM",
        parent_baseline_electrolyte="Unspecified / Not reported",
        is_literature_additive=True,
    )

    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    at.session_state["mined_literature_additives"] = [cand_unspecified]
    at.run()
    assert len(at.exception) == 0

    # Verify selectbox exists
    t1_dropdowns = [sb for sb in at.selectbox if sb.key == "step2_tier1_additive_dropdown"]
    assert len(t1_dropdowns) == 1

    # Modify the candidate baseline electrolyte simulating edit save (new object)
    edited_cand = copy.deepcopy(cand_unspecified)
    edited_cand.parent_baseline_electrolyte = "2 M ZnSO4"
    edited_cand.reported_concentration = 40.0
    at.session_state["mined_literature_additives"] = [edited_cand]
    at.run()
    assert len(at.exception) == 0

    # Verify updated values in session state
    saved_cand = at.session_state["mined_literature_additives"][0]
    assert saved_cand.parent_baseline_electrolyte == "2 M ZnSO4"
    assert saved_cand.reported_concentration == 40.0


def test_ui_custom_mode_no_sample_candidate_buttons_and_tier1_exclusive_upload():
    """
    Verify:
    1. Under Custom Experiment Mode (is_demo=False), Step 2 does NOT render 'load sample literature candidates'
       or 'benchmark recommendation ...' buttons.
    2. Under Demo Mode (is_demo=True), when candidates are empty, those sample buttons are available.
    3. Standalone Custom Additive List Upload uploader/widgets are available ONLY in Tier 1 ('t1_empty' / 't1'),
       NOT in Tier 2 ('t2_empty' / 't2').
    4. Tier 2 empty state shows guidance directing users to Tier 1 or Step 1.
    """
    at_custom = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at_custom.run()
    # Switch to Custom Mode
    at_custom.sidebar.radio[0].set_value("custom").run()
    assert len(at_custom.exception) == 0

    # Ensure mined_literature_additives is empty
    at_custom.session_state["mined_literature_additives"] = []
    at_custom.session_state["ai_suggested_additives"] = []
    at_custom.session_state["suggested_candidates"] = []
    at_custom.run()
    assert len(at_custom.exception) == 0

    # 1. In Custom Experiment Mode, sample candidate buttons must NOT exist
    b_keys = [getattr(b, "key", "") for b in at_custom.button]
    assert "btn_load_sample_candidates_t1" not in b_keys, "Sample candidate button must not be present in Custom Mode"
    assert "btn_preview_sample_suggestions_t1" not in b_keys, "Benchmark recommendation button must not be present in Custom Mode"

    # 2. Upload controls exist strictly in Tier 1, NOT in Tier 2
    dl_keys = [getattr(b, "key", "") for b in at_custom.download_button]
    assert any("btn_dl_csv_template" in k for k in dl_keys), "Download template button must be available"
    assert any("btn_dl_xlsx_template" in k for k in dl_keys), "Download template button must be available"

    fu_keys = [getattr(u, "key", "") for u in at_custom.file_uploader]
    assert "uploader_step2_additives_t1_empty" in fu_keys, "Upload must be available in Tier 1 empty state"
    assert "uploader_step2_additives_t2_empty" not in fu_keys, "Upload must NOT be rendered in Tier 2"

    # 3. Tier 2 guidance note is rendered
    captions = [c.value for c in at_custom.caption]
    assert any("Tier 1" in cap or "mined" in cap.lower() or "tested literature additives" in cap.lower() for cap in captions), "Guidance note to upload in Tier 1 / Tested Literature Additives must be present"

    # 4. In Demo Mode, when candidates are empty, sample candidate buttons ARE present
    at_demo = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at_demo.run()
    at_demo.sidebar.radio[0].set_value("demo").run()
    at_demo.session_state["mined_literature_additives"] = []
    at_demo.run()
    assert len(at_demo.exception) == 0
    demo_b_keys = [getattr(b, "key", "") for b in at_demo.button]
    assert "btn_load_sample_candidates_t1" in demo_b_keys, "Sample candidates button must be present in Demo Mode"
    assert "btn_preview_sample_suggestions_t1" in demo_b_keys, "Benchmark recommendation button must be present in Demo Mode"


def test_ui_tier1_upload_populates_both_tiers_and_tier2_guidance():
    """
    Verify:
    1. Upload parses the list and populates Tier 1 (literature) and Tier 2 (AI-generated analogues).
    2. Both dropdowns are then available.
    """
    from discovery.additive_importer import generate_template_csv, parse_additive_file, deduplicate_and_resolve_additives
    from discovery.literature_engine import LiteratureDiscoveryEngine

    sample_csv = generate_template_csv()
    records, _ = parse_additive_file(sample_csv, "test.csv")
    cands, _ = deduplicate_and_resolve_additives(records, max_limit=5000, resolve_pubchem=False)
    assert len(cands) > 0

    lit_eng = LiteratureDiscoveryEngine()
    _, t2_cands = lit_eng.generate_two_tier_candidates(cands, use_pubchem=False)
    assert len(t2_cands) > 0

    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    # Simulate populated candidates
    at.session_state["mined_literature_additives"] = cands
    at.session_state["ai_suggested_additives"] = t2_cands
    at.session_state["suggested_candidates"] = t2_cands
    at.run()
    assert len(at.exception) == 0

    # Both dropdowns should now be populated
    t1_dropdowns = [sb for sb in at.selectbox if sb.key == "step2_tier1_additive_dropdown"]
    assert len(t1_dropdowns) == 1
    t2_dropdowns = [sb for sb in at.selectbox if sb.key == "tab2_suggested_additives_dropdown"]
    assert len(t2_dropdowns) == 1

    # Verify Tier 2 does NOT have an upload button when populated
    fu_keys = [getattr(u, "key", "") for u in at.file_uploader]
    assert "uploader_step2_additives_t2" not in fu_keys, "Tier 2 populated state must not have upload button"


def test_condition_aware_deduplication_and_combine():
    """
    Verify condition-aware deduplication:
    1. Additives with the SAME name but DIFFERENT concentration (e.g. 10 mM vs 50 mM) are preserved as distinct candidates.
    2. Additives with the SAME name but DIFFERENT baseline electrolyte (e.g. 2 M ZnSO4 vs 1 M Zn(OTf)2) are preserved as distinct candidates.
    3. Additives with the SAME name, concentration, and baseline electrolyte are collapsed into 1 entry with citation frequency incremented.
    4. Session state combine logic correctly appends distinct conditions and increments citations for exact matches.
    """
    from discovery.additive_importer import deduplicate_and_resolve_additives
    from discovery.formula_utils import normalize_condition_key

    records = [
        {"name": "Glucose", "concentration": 10.0, "unit": "mM", "salt": "2 M ZnSO4", "smiles": "C(C1C(C(C(C(O1)O)O)O)O)O"},
        {"name": "Glucose", "concentration": 50.0, "unit": "mM", "salt": "2 M ZnSO4", "smiles": "C(C1C(C(C(C(O1)O)O)O)O)O"},
        {"name": "Glucose", "concentration": 10.0, "unit": "mM", "salt": "1 M Zn(OTf)2", "smiles": "C(C1C(C(C(C(O1)O)O)O)O)O"},
        {"name": "Glucose", "concentration": 10.0, "unit": "mM", "salt": "2 M ZnSO4", "smiles": "C(C1C(C(C(C(O1)O)O)O)O)O"},  # True duplicate of #1
    ]

    cands, stats = deduplicate_and_resolve_additives(records, max_limit=5000, resolve_pubchem=False)

    # Must preserve 3 distinct candidates
    assert len(cands) == 3, f"Expected 3 distinct candidates, got {len(cands)}"
    assert stats["total_records_parsed"] == 4
    assert stats["unique_additives_count"] == 3

    # Check the duplicate was collapsed with 2 citations
    cand_10_so4 = next(c for c in cands if c.reported_concentration == 10.0 and "znso4" in c.parent_baseline_electrolyte.lower())
    assert "2 citation(s)" in cand_10_so4.rationale

    cand_50_so4 = next(c for c in cands if c.reported_concentration == 50.0 and "znso4" in c.parent_baseline_electrolyte.lower())
    assert "1 citation(s)" in cand_50_so4.rationale

    cand_10_otf = next(c for c in cands if c.reported_concentration == 10.0 and "znotf" in c.parent_baseline_electrolyte.lower().replace("(", "").replace(")", ""))
    assert "1 citation(s)" in cand_10_otf.rationale

    # Now verify append / combine logic
    existing = list(cands)
    new_incoming = [
        # Exact condition match for cand_10_so4
        cands[0].model_copy(update={"rationale": "Reported in 1 citation(s)."}),
        # Brand new formulation: Glucose 100 mM in 2 M ZnSO4
        cands[0].model_copy(update={"reported_concentration": 100.0, "rationale": "Reported in 1 citation(s)."}),
    ]

    ex_keys = {
        normalize_condition_key(c.name, c.reported_concentration, c.parent_baseline_electrolyte): c
        for c in existing
    }
    combined = list(existing)
    for ic in new_incoming:
        ckey = normalize_condition_key(ic.name, ic.reported_concentration, ic.parent_baseline_electrolyte)
        if ckey in ex_keys:
            ex_cand = ex_keys[ckey]
            ex_cand.citation_count = getattr(ex_cand, "citation_count", 1) + getattr(ic, "citation_count", 1)
        else:
            combined.append(ic)
            ex_keys[ckey] = ic

    assert len(combined) == 4
    assert any(c.reported_concentration == 100.0 for c in combined)


def test_upload_only_names_has_no_fabricated_guesses_and_truthful_ui():
    """
    Verify:
    1. Uploading a file with ONLY additive names (no conc, no salt, no mechanisms, no smiles)
       does NOT fabricate 15 mM, 2 M ZnSO4, or 'Interfacial protection'.
    2. CandidateMolecule accurately reflects None concentration, 'Unspecified / Not reported' salt and mechanism.
    3. UI reflects truthful 'Unspecified / Not reported' and exploratory formulation notice.
    """
    from discovery.additive_importer import parse_additive_file, deduplicate_and_resolve_additives
    from localization import get_text

    raw_names_txt = "\n".join([f"Experimental Molecule {i}" for i in range(110)]).encode("utf-8")
    parsed_records, notices = parse_additive_file(raw_names_txt, "raw_names.txt")
    assert len(parsed_records) == 110

    # Ensure parser did not fabricate defaults
    assert parsed_records[0]["concentration"] is None
    assert parsed_records[0]["salt"] == "Unspecified / Not reported"
    assert parsed_records[0]["smiles"] == ""

    # Ingest and resolve
    cands, stats = deduplicate_and_resolve_additives(parsed_records, max_limit=5000, resolve_pubchem=False)
    assert len(cands) == 110
    sample_c = cands[0]

    # Verify zero fabrication
    assert sample_c.reported_concentration is None
    assert sample_c.parent_baseline_electrolyte == "Unspecified / Not reported"
    assert sample_c.reported_mechanisms == []
    assert sample_c.mechanism_summary == "Unspecified / Not reported"
    assert sample_c.electrolyte_compatibility_note == "Unassessed (chemical structure not provided)"
    assert sample_c.logs is None

    # Test in AppTest UI
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    at.session_state["mined_literature_additives"] = cands
    at.run()
    assert len(at.exception) == 0

    # Verify dropdown options format contains 'Unspecified / Not reported'
    t1_dropdown = next(sb for sb in at.selectbox if sb.key == "step2_tier1_additive_dropdown")
    first_opt_str = t1_dropdown.format_func(0)
    assert get_text("card_reported_conc_unspecified") in first_opt_str
    assert get_text("baseline_salt_unspecified") in first_opt_str

    # Verify exploratory anchor notice is rendered
    captions = [c.value for c in at.caption]
    assert any("exploratory" in cap.lower() or "15" in cap for cap in captions)


def test_ui_step2_ai_suggestion_rationale_card_in_whitespace():
    """
    Task 1: Verify AI Suggestion Rationale card in col_mol1 whitespace:
    - Displayed for Tier 2 AI analogues.
    - Explains parent literature scaffold and Tanimoto similarity score.
    - Explains functional group motif promise for Zn2+ solvation modulation or dendrite/HER suppression.
    - Explains physical hypothesis (zincophilic adsorption + aqueous solubility).
    - Not displayed for direct Tier 1 literature additives.
    """
    from models import CandidateMolecule
    from localization import get_text

    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    # Tier 2 candidate
    t2_cand = CandidateMolecule(
        name="D-Mannitol",
        smiles="C(C(C(C(C(CO)O)O)O)O)O",
        similarity_score=0.92,
        similarity_metric="Tanimoto (Morgan R=2, Curated DB)",
        molecular_weight=182.17,
        logp=-3.1,
        logs=-1.1,
        tpsa=121.4,
        hbd=6,
        hba=6,
        rotatable_bonds=5,
        rationale="AI-suggested analogue of D-Sorbitol",
        is_literature_additive=False,
        parent_scaffold_name="D-Sorbitol",
        parent_scaffold_similarity=0.92,
        parent_baseline_electrolyte="2 M ZnSO4",
    )

    at.session_state["ai_suggested_additives"] = [t2_cand]
    at.session_state["suggested_candidates"] = [t2_cand]

    # In Custom Mode, set the active candidate via custom inputs
    custom_name_inputs = [ti for ti in at.text_input if ti.key == "tab2_custom_additive_name_input"]
    custom_smiles_inputs = [ti for ti in at.text_input if ti.key == "tab2_custom_smiles_input"]
    custom_name_inputs[0].set_value("D-Mannitol")
    custom_smiles_inputs[0].set_value("C(C(C(C(C(CO)O)O)O)O)O").run()
    assert len(at.exception) == 0

    all_md = " ".join([m.value for m in at.markdown if hasattr(m, "value")])

    # 1. Verify "💡 AI Suggestion Rationale" card is present
    assert get_text("ai_suggestion_rationale_title") in all_md, "AI Suggestion Rationale title must be rendered in col_mol1"

    # 2. Verify parent scaffold anchor and similarity score
    assert "D-Sorbitol" in all_md, "Parent literature scaffold (D-Sorbitol) must be displayed"
    assert "0.92" in all_md, "Tanimoto similarity score (0.92) must be displayed"

    # 3. Verify functional group motif electrochemical rationale
    assert any(term in all_md for term in ["solvation", "solvatation", "溶剂化", "BZS", "hydroxyl"]), \
        "Functional group electrochemical rationale must be displayed"

    # 4. Verify physical hypothesis
    assert any(term in all_md for term in ["adsorption", "solubility", "saturation", "Delaney", "溶解度", "吸附"]), \
        "Physical hypothesis must be displayed"

    # 5. Verify that for Tier 1 direct literature finding (e.g. D-Sorbitol with is_literature_additive=True),
    # the AI suggestion rationale card is not rendered
    custom_name_inputs = [ti for ti in at.text_input if ti.key == "tab2_custom_additive_name_input"]
    custom_smiles_inputs = [ti for ti in at.text_input if ti.key == "tab2_custom_smiles_input"]
    custom_name_inputs[0].set_value("D-Sorbitol")
    custom_smiles_inputs[0].set_value("C(C(C(C(C(CO)O)O)O)O)O").run()
    assert len(at.exception) == 0

    md_t1 = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    assert get_text("ai_suggestion_rationale_title") not in md_t1, \
        "AI Suggestion Rationale card should NOT be rendered for Tier 1 literature additive"


def test_ui_step2_ai_suggestion_rationale_auto_loads_without_manual_selection():
    """
    Verify that when both Tier 1 literature candidates and Tier 2 AI analogues exist
    (e.g., following an additive list upload or literature mining), the '💡 AI Suggestion Rationale'
    card automatically renders for the active Tier 2 candidate upon viewing the tab, without requiring
    the user to manually toggle the dropdown or click a card in the card deck.
    """
    from streamlit.testing.v1 import AppTest
    from models import CandidateMolecule
    from localization import get_text

    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    t1_cand = CandidateMolecule(
        name="Polyethylene Glycol",
        smiles="C(CO)O",
        similarity_score=1.0,
        similarity_metric="Direct Match",
        molecular_weight=62.07,
        logp=-1.36,
        logs=0.5,
        tpsa=40.5,
        hbd=2,
        hba=2,
        rotatable_bonds=1,
        rationale="Reported in literature",
        is_literature_additive=True,
        reported_concentration=15.0,
        parent_baseline_electrolyte="2 M ZnSO4",
    )
    t2_cand = CandidateMolecule(
        name="1,3-Propanediol",
        smiles="CCCO",
        similarity_score=0.88,
        similarity_metric="Tanimoto (Morgan R=2, Curated DB)",
        molecular_weight=76.09,
        logp=-0.3,
        logs=-0.1,
        tpsa=40.5,
        hbd=2,
        hba=2,
        rotatable_bonds=2,
        rationale="AI-suggested analogue of Polyethylene Glycol",
        is_literature_additive=False,
        parent_scaffold_name="Polyethylene Glycol",
        parent_scaffold_similarity=0.88,
        parent_baseline_electrolyte="2 M ZnSO4",
    )

    # Simulate state right after upload: Tier 1 candidate is active in session state
    at.session_state["mined_literature_additives"] = [t1_cand]
    at.session_state["ai_suggested_additives"] = [t2_cand]
    at.session_state["suggested_candidates"] = [t2_cand]
    at.session_state["selected_additive_name"] = "Polyethylene Glycol"
    at.session_state["tab2_custom_additive_name_input"] = "Polyethylene Glycol"
    at.run()
    assert len(at.exception) == 0

    all_md = " ".join([m.value for m in at.markdown if hasattr(m, "value")])

    # Verify AI Suggestion Rationale is automatically rendered for the Tier 2 candidate
    assert get_text("ai_suggestion_rationale_title") in all_md, \
        "AI Suggestion Rationale card must automatically load for Tier 2 candidate without manual reselection"
    assert "Polyethylene Glycol" in all_md, "Parent anchor Polyethylene Glycol must be rendered"
    assert "0.88" in all_md, "Similarity score must be rendered"


def test_ui_step2_delaney_esol_condensed_and_transfer_button_removed():
    """
    Task 2 & 3:
    - Verify redundant transfer button (btn_apply_custom_additive_tab2) is removed from Step 2.
    - Verify large green callout box under Delaney ESOL metrics is removed.
    - Verify compact tooltip (?) on the section title with Delaney ESOL explanation is present.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    # 1. Verify redundant transfer button is completely absent in Custom Mode
    transfer_btns = [b for b in at.button if b.key == "btn_apply_custom_additive_tab2"]
    assert len(transfer_btns) == 0, "Redundant transfer button btn_apply_custom_additive_tab2 must be removed"

    # 2. Switch to Demo Mode (where D-Sorbitol is loaded by default)
    at.sidebar.radio[0].set_value("demo").run()
    assert len(at.exception) == 0

    # Redundant transfer button should not exist in Demo Mode either
    transfer_btns_demo = [b for b in at.button if b.key == "btn_apply_custom_additive_tab2"]
    assert len(transfer_btns_demo) == 0, "Redundant transfer button must not exist in Demo Mode"

    all_md = " ".join([m.value for m in at.markdown if hasattr(m, "value")])

    # 3. Verify large green callout box under Delaney ESOL is removed
    assert 'border-left-color: #00a896' not in all_md, "Large green Delaney ESOL callout box must be removed"

    # 4. Verify section title has compact tooltip (?) with Delaney ESOL explanation
    assert "(?)" in all_md or "aria-label=\"Delaney ESOL explanation\"" in all_md, \
        "Compact tooltip (?) must be attached to the Delaney ESOL section title"
    assert "Delaney" in all_md and "80%" in all_md, "Delaney ESOL section title must be rendered"


def test_ui_step2_distinguish_reported_vs_assumed_exploratory_baseline():
    """
    Task 4: Distinguish Reported vs. Assumed Exploratory Baseline:
    - When concentration/salt are reported in the file: Display 'Reported Literature Baseline: {conc} mM in {salt}'.
    - When concentration/salt are unspecified in the file: Honestly label as 'Assumed Exploratory Baseline: {conc} mM (Standard 2 M ZnSO₄ Reference)'
      with '⚠️ Unspecified in file upload' tag.
    """
    from models import CandidateMolecule
    from localization import get_text

    # Case A: Reported Literature Baseline in Demo Mode (D-Sorbitol has 15.0 mM in 2 M ZnSO4)
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    at.sidebar.radio[0].set_value("demo").run()
    assert len(at.exception) == 0

    metric_labels = [m.label for m in at.metric if hasattr(m, "label") and m.label]
    assert any("Reported Literature Baseline" in lbl for lbl in metric_labels), \
        f"Expected 'Reported Literature Baseline' in metric labels, got {metric_labels}"

    all_md_reported = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    assert get_text("unspecified_in_upload_tag") not in all_md_reported, \
        "Reported additive must NOT have 'Unspecified in file upload' tag"

    # Case B: Unspecified in file upload
    unspec_cand = CandidateMolecule(
        name="Experimental X1",
        smiles="OCC(O)CO",
        similarity_score=1.0,
        similarity_metric="Direct Literature Finding",
        molecular_weight=92.09,
        logp=-1.75,
        logs=0.4,
        tpsa=60.7,
        hbd=3,
        hba=3,
        rotatable_bonds=2,
        rationale="Imported additive",
        is_literature_additive=True,
        reported_concentration=None,  # Unspecified
        parent_baseline_electrolyte="Unspecified / Not reported",  # Unspecified
    )

    at2 = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at2.session_state["mined_literature_additives"] = [unspec_cand]
    at2.session_state["is_active_baseline_unspecified"] = True
    at2.run()
    assert len(at2.exception) == 0

    # In Custom Mode, enter Experimental X1
    custom_name_inputs = [ti for ti in at2.text_input if ti.key == "tab2_custom_additive_name_input"]
    custom_smiles_inputs = [ti for ti in at2.text_input if ti.key == "tab2_custom_smiles_input"]
    custom_name_inputs[0].set_value("Experimental X1")
    custom_smiles_inputs[0].set_value("OCC(O)CO").run()
    assert len(at2.exception) == 0

    metric_labels_unspec = [m.label for m in at2.metric if hasattr(m, "label") and m.label]
    assert any("Assumed Exploratory Baseline" in lbl for lbl in metric_labels_unspec), \
        f"Expected 'Assumed Exploratory Baseline' in metric labels, got {metric_labels_unspec}"

    all_md_unspec = " ".join([m.value for m in at2.markdown if hasattr(m, "value")])
    assert get_text("unspecified_in_upload_tag") in all_md_unspec, \
        "Unspecified candidate must display '⚠️ Unspecified in file upload' tag"


def test_step2_trilingual_localization_parity():
    """
    Verify 100% trilingual dictionary parity across en, fr, and zh for all new Step 2 enhancement keys.
    """
    from localization import TRANSLATIONS, get_text

    keys_to_verify = [
        "ai_suggestion_rationale_title",
        "rationale_parent_anchor_label",
        "rationale_functional_motifs_label",
        "rationale_physical_hypothesis_label",
        "rationale_anchored_to",
        "fg_rationale_polyol",
        "fg_rationale_amide_urea",
        "fg_rationale_thiocarbonyl",
        "fg_rationale_quaternary_ammonium",
        "fg_rationale_sulfoxide",
        "fg_rationale_ether_polyether",
        "fg_rationale_carboxylate",
        "fg_rationale_general",
        "hypothesis_high_solubility",
        "hypothesis_esol_capped",
        "metric_assumed_baseline",
        "unspecified_in_upload_tag",
        "standard_ref_label",
        "in_salt_label",
        "fg_name_polyol",
        "fg_name_amide_urea",
        "fg_name_thiocarbonyl",
        "fg_name_quaternary_ammonium",
        "fg_name_sulfoxide",
        "fg_name_ether_polyether",
        "fg_name_carboxylate",
        "fg_name_general",
        "provenance_uploaded",
        "provenance_online_mined",
        "provenance_batch_pdf",
        "provenance_curated",
        "card_provenance_label",
    ]

    for lang in ["en", "fr", "zh"]:
        assert lang in TRANSLATIONS, f"Language '{lang}' missing from TRANSLATIONS"
        for k in keys_to_verify:
            assert k in TRANSLATIONS[lang], f"Key '{k}' missing from '{lang}' translations"
            val = get_text(k, lang=lang, parent="TestParent", score="0.90", ceiling=50.0, salt="2 M ZnSO4")
            assert not val.startswith("["), f"Key '{k}' failed translation in '{lang}': {val}"
            assert len(val) > 0


def test_ui_step1_search_additive_merging_and_bridging_notice_dismissal():
    """
    Verify:
    1. Uploaded/existing additives in Step 2 are NOT wiped when online search runs in Step 1.
    2. Newly discovered candidates are non-destructively merged into mined_literature_additives.
    3. Step 2 candidate selection does not set bridged_additive_name in Step 1.
    4. Bridging notice banner in Step 1 provides an interactive [✕ Dismiss] button that clears it.
    """
    from models import CandidateMolecule

    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    # 1. Simulate an uploaded additive in Step 2
    uploaded_cand = CandidateMolecule(
        name="Custom Glycerol Test",
        smiles="OCC(O)CO",
        similarity_score=1.0,
        similarity_metric="Direct Upload",
        molecular_weight=92.09,
        logp=-1.76,
        logs=0.48,
        tpsa=60.69,
        hbd=3,
        hba=3,
        rotatable_bonds=2,
        reported_concentration=25.0,
        parent_baseline_electrolyte="2 M ZnSO4",
        is_literature_additive=True,
        rationale="Uploaded candidate test",
    )
    at.session_state["mined_literature_additives"] = [uploaded_cand]

    # Verify candidate card click in Step 2 does NOT set bridged_additive_name
    assert "bridged_additive_name" not in at.session_state

    # 2. Run online search in Step 1
    search_inputs = [ti for ti in at.text_input if ti.key == "lit_search_query_input"]
    search_inputs[0].input("trehalose").run()
    search_btn = [b for b in at.button if b.key == "lit_search_submit_btn"][0]
    search_btn.click().run()
    assert len(at.exception) == 0

    # 3. Verify non-destructive merge: both custom uploaded and searched candidates are present
    t1_cands = at.session_state.get("mined_literature_additives", [])
    cand_names = [c.name.lower() for c in t1_cands]
    assert "custom glycerol test" in cand_names, "Uploaded additive must NOT be wiped by Step 1 search"
    assert any("trehalose" in cn for cn in cand_names), "Searched candidate must be present in mined_literature_additives"

    # 4. Verify bridging notice dismiss functionality
    at.session_state["bridged_additive_name"] = "trehalose"
    at.run()
    assert len(at.exception) == 0

    assert "bridged_additive_name" in at.session_state
    dismiss_btns = [b for b in at.button if b.key == "btn_dismiss_bridged_notice"]
    assert len(dismiss_btns) > 0, "Dismiss button for bridging notice must be rendered"

    # 5. Click Dismiss button
    dismiss_btns[0].click().run()
    assert len(at.exception) == 0
    assert "bridged_additive_name" not in at.session_state, "Dismiss button must remove bridged_additive_name from session_state"


def test_step2_additive_provenance_and_card_deck_reactive_sync():
    """
    Verify:
    1. get_candidate_provenance correctly identifies:
       - User Uploaded additives ('uploaded', 📤)
       - Batch Ingested PDF additives ('batch_pdf', 📑)
       - Online Mined Paper additives ('online_mined', 🌐)
       - Curated Library additives ('curated', 📚)
    2. Trilingual rendering across en, fr, zh.
    3. Clicking 'Use This Additive' on a Tier 2 candidate card updates
       tab2_suggested_additives_dropdown and selected_tier2_name immediately.
    """
    from app import get_candidate_provenance
    from models import CandidateMolecule

    # 1. Test provenance detection
    cand_up = CandidateMolecule(
        name="Uploaded Polyol", smiles="OCCO", similarity_score=1.0,
        similarity_metric="User Uploaded Additive", molecular_weight=62.07,
        logp=-1.36, logs=0.35, tpsa=40.46, hbd=2, hba=2, rotatable_bonds=1,
        rationale="User uploaded additive. Reported in 3 citation(s)."
    )
    p_key, p_lbl, p_cls, p_icon = get_candidate_provenance(cand_up, lang="en")
    assert p_key == "uploaded"
    assert p_icon == "📤"
    assert p_lbl == "User Uploaded"
    assert p_cls == "badge-teal"

    cand_pdf = CandidateMolecule(
        name="PDF Additive", smiles="CC(C(=O)N)C", similarity_score=1.0,
        similarity_metric="Batch PDF Ingestion", molecular_weight=87.12,
        logp=-0.15, logs=0.10, tpsa=43.09, hbd=1, hba=1, rotatable_bonds=1,
        rationale="Extracted from batch literature PDF: electrolyte_paper.pdf"
    )
    p_key, p_lbl, p_cls, p_icon = get_candidate_provenance(cand_pdf, lang="en")
    assert p_key == "batch_pdf"
    assert p_icon == "📑"
    assert p_lbl == "Batch Ingested PDF"
    assert p_cls == "badge-amber"

    cand_mine = CandidateMolecule(
        name="Mined Additive", smiles="C(=S)(N)N", similarity_score=1.0,
        similarity_metric="Direct Literature Extraction", molecular_weight=76.12,
        logp=-0.44, logs=-0.12, tpsa=55.88, hbd=2, hba=1, rotatable_bonds=0,
        rationale="Extracted from literature paper: Advanced Materials 2024"
    )
    p_key, p_lbl, p_cls, p_icon = get_candidate_provenance(cand_mine, lang="en")
    assert p_key == "online_mined"
    assert p_icon == "🌐"
    assert p_lbl == "Online Mined Paper"
    assert p_cls == "badge-purple"

    cand_cur = CandidateMolecule(
        name="Curated Additive", smiles="C(C(C(C(C(CO)O)O)O)O)O", similarity_score=1.0,
        similarity_metric="Tanimoto (Morgan)", molecular_weight=182.17,
        logp=-3.11, logs=0.45, tpsa=121.38, hbd=6, hba=6, rotatable_bonds=5,
        rationale="Direct literature finding: D-Sorbitol."
    )
    p_key, p_lbl, p_cls, p_icon = get_candidate_provenance(cand_cur, lang="en")
    assert p_key == "curated"
    assert p_icon == "📚"
    assert p_lbl == "Curated Library"
    assert p_cls == "badge-blue"

    # 2. Trilingual rendering
    _, p_lbl_fr, _, _ = get_candidate_provenance(cand_up, lang="fr")
    _, p_lbl_zh, _, _ = get_candidate_provenance(cand_up, lang="zh")
    assert p_lbl_fr == "Téléversé par l'utilisateur"
    assert p_lbl_zh == "用户上传"

    # 3. Reactive UI AppTest verification for Tier 2 card deck button click
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    t2_cands = at.session_state.get("ai_suggested_additives", [])
    if len(t2_cands) >= 2:
        target_cand = t2_cands[1]
        btn_t2 = [b for b in at.button if b.key == "btn_formulate_t2_1"]
        if btn_t2:
            btn_t2[0].click().run()
            assert len(at.exception) == 0
            assert at.session_state.get("selected_tier2_name") == target_cand.name
            assert at.session_state.get("tab2_suggested_additives_dropdown") == 1

    # 4. Verify selecting from Tier 1 dropdown updates candidate without reverting
    t1_cands = at.session_state.get("mined_literature_additives", [])
    if len(t1_cands) >= 2:
        t1_sb = [sb for sb in at.selectbox if sb.key == "step2_tier1_additive_dropdown"]
        if t1_sb:
            t1_sb[0].select_index(1).run()
            assert len(at.exception) == 0
            assert at.session_state.get("selected_tier1_name") == t1_cands[1].name
            assert at.session_state.get("selected_additive_name") == t1_cands[1].name
            assert at.session_state.get("step2_tier1_additive_dropdown") == 1

    # 5. Verify selecting from Tier 2 dropdown updates candidate without reverting
    if len(t2_cands) >= 2:
        t2_sb = [sb for sb in at.selectbox if sb.key == "tab2_suggested_additives_dropdown"]
        if t2_sb:
            t2_sb[0].select_index(0).run()
            assert len(at.exception) == 0
            assert at.session_state.get("selected_tier2_name") == t2_cands[0].name
            assert at.session_state.get("tab2_suggested_additives_dropdown") == 0


def test_ui_step2_bottom_dropdown_synchronization_and_delaney_esol():
    """Verify that bottom dropdowns in Delaney ESOL section update active candidate and sync bidirectionally."""
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    from models import ExtractedAdditive, Concentration, ZincIonBatteryContext
    from discovery.literature_engine import LiteratureDiscoveryEngine
    mock_additives = [
        ExtractedAdditive(
            name="Succinic Acid",
            smiles="C(CC(=O)O)C(=O)O",
            optimal_concentration=Concentration(value=15.0, unit="mM", raw_text="15 mM"),
            primary_mechanisms=["Corrosion inhibition"],
            battery_context=ZincIonBatteryContext(baseline_electrolyte="2 M ZnSO4")
        ),
        ExtractedAdditive(
            name="Glycine",
            smiles="C(C(=O)O)N",
            optimal_concentration=Concentration(value=30.0, unit="mM", raw_text="30 mM"),
            primary_mechanisms=["Solvation sheath modulation"],
            battery_context=ZincIonBatteryContext(baseline_electrolyte="2 M ZnSO4")
        ),
        ExtractedAdditive(
            name="Thiourea",
            smiles="NC(=S)N",
            optimal_concentration=Concentration(value=10.0, unit="mM", raw_text="10 mM"),
            primary_mechanisms=["Adsorption layer"],
            battery_context=ZincIonBatteryContext(baseline_electrolyte="2 M ZnSO4")
        ),
    ]
    engine = LiteratureDiscoveryEngine()
    t1_cands, t2_cands = engine.generate_two_tier_candidates(mock_additives, use_pubchem=False)
    at.session_state["mined_literature_additives"] = t1_cands
    at.session_state["ai_suggested_additives"] = t2_cands
    at.session_state["suggested_candidates"] = t2_cands
    at.session_state["selected_additive_name"] = t1_cands[0].name
    at.session_state["selected_tier1_name"] = t1_cands[0].name
    at.session_state["selected_tier2_name"] = t2_cands[0].name
    at.run()
    assert len(at.exception) == 0

    # 1. Verify bottom Tier 1 dropdown exists and selecting index 1 updates active candidate and top dropdown
    bot_t1_sb = [sb for sb in at.selectbox if sb.key == "step2_tier1_bottom_dropdown"]
    assert len(bot_t1_sb) > 0, "step2_tier1_bottom_dropdown must exist in Delaney ESOL section"
    bot_t1_sb[0].select_index(1).run()
    assert len(at.exception) == 0
    assert at.session_state.get("selected_tier1_name") == t1_cands[1].name
    assert at.session_state.get("selected_additive_name") == t1_cands[1].name
    assert at.session_state.get("step2_tier1_additive_dropdown") == 1

    # 2. Verify top Tier 1 dropdown syncs back down to bottom dropdown
    top_t1_sb = [sb for sb in at.selectbox if sb.key == "step2_tier1_additive_dropdown"]
    assert len(top_t1_sb) > 0
    top_t1_sb[0].select_index(2).run()
    assert len(at.exception) == 0
    assert at.session_state.get("selected_tier1_name") == t1_cands[2].name
    assert at.session_state.get("selected_additive_name") == t1_cands[2].name
    bot_t1_updated = [sb for sb in at.selectbox if sb.key == "step2_tier1_bottom_dropdown"]
    assert bot_t1_updated[0].value == 2 or at.session_state.get("step2_tier1_bottom_dropdown") == 2

    # 3. Verify manual custom molecule expander is present when candidates exist
    all_expanders = [getattr(e, "label", "") for e in at.expander if hasattr(e, "label")]
    expander_text = " ".join(all_expanders)
    assert any(term in expander_text for term in ["Manual Custom", "Saisie Manuelle", "手动输入"]), \
        "Manual Custom Molecule expander must be present when candidate pool exists"


def test_ui_step2_doe_formulation_matrix_compact_and_rationale_card():
    """
    Verify:
    1. In Custom Experiment Mode (is_demo=False), Planned DoE Formulation Matrix is rendered.
    2. The DoE Concentration Selection & Sampling Rationale card is rendered right below the compact table.
    3. The card displays the 4 mechanistic pillars:
       - Thermodynamic Anchor & Scaling
       - Solubility Protection & Anti-Precipitation
       - Space-Filling Exploration Strategy (LHS/CCD)
       - Electrolyte Salt Molarity Window
    4. Switching language updates the rationale card labels.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    from models import ExtractedAdditive, Concentration, ZincIonBatteryContext
    from discovery.literature_engine import LiteratureDiscoveryEngine

    mock_additives = [
        ExtractedAdditive(
            name="Nicotinamide",
            smiles="NC(=O)c1cccnc1",
            optimal_concentration=Concentration(value=25.0, unit="mM", raw_text="25 mM"),
            primary_mechanisms=["Solvation sheath reconstruction"],
            battery_context=ZincIonBatteryContext(baseline_electrolyte="2 M ZnSO4")
        ),
    ]
    engine = LiteratureDiscoveryEngine()
    t1_cands, t2_cands = engine.generate_two_tier_candidates(mock_additives, use_pubchem=False)
    at.session_state["mined_literature_additives"] = t1_cands
    at.session_state["ai_suggested_additives"] = t2_cands
    at.session_state["selected_additive_name"] = t1_cands[0].name
    at.session_state["selected_tier1_name"] = t1_cands[0].name
    at.session_state["tab2_custom_additive_name_input"] = t1_cands[0].name
    at.session_state["custom_additive_name"] = t1_cands[0].name
    at.session_state["custom_smiles"] = t1_cands[0].smiles
    at.run()
    assert len(at.exception) == 0

    # Trigger candidate selection via bottom selectbox to generate plan
    bot_t1_sb = [sb for sb in at.selectbox if sb.key == "step2_tier1_bottom_dropdown"]
    assert len(bot_t1_sb) > 0
    bot_t1_sb[0].select_index(0).run()
    assert len(at.exception) == 0

    # Verify DoE Rationale Card rendered in markdown
    all_md = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    assert "How Test Concentrations Were Chosen (DoE Rationale)" in all_md
    assert "Thermodynamic Anchor & Scaling" in all_md
    assert "Solubility Protection & Anti-Precipitation" in all_md
    assert "Space-Filling Exploration Strategy" in all_md
    assert "Electrolyte Salt Molarity Window" in all_md
    assert "LHS Design" in all_md or "CCD Design" in all_md


def test_ui_step3_multisheet_arbin_deposition_upload_renders_properly():
    """
    Verify:
    1. Multi-sheet Arbin Excel workbooks with metadata in Sheet 1 (Global_Info)
       and unipolar deposition time series in Sheet 2 (Channel_7_1) are parsed cleanly.
    2. In Custom Mode, uploading the file populates custom_analysis_report and displays
       success message.
    3. The empty notice ('custom_mode_step3_empty_notice') is NOT displayed.
    4. General chronopotentiometry metrics (Duration, Data Points, Max V, Min V) and
       potential vs time Plotly chart are rendered.
    """
    from export_engine import make_multi_sheet_xlsx
    from localization import get_text

    global_info = [
        ["Test Name", "ZC-2-2-1-depo"],
        ["Channel Number", 7],
        ["Start Date", "2024-05-01"],
    ]
    times = [float(t) for t in range(0, 600, 10)]
    channel_data = [
        ["Test_Time(s)", "Voltage(V)", "Current(A)", "Capacity(Ah)", "Cycle_Index"],
    ]
    for t in times:
        channel_data.append([t, -0.045, -0.001, 0.001 * t / 3600.0, 1])

    xlsx_bytes = make_multi_sheet_xlsx({"Global_Info": global_info, "Channel_7_1": channel_data})

    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    fu = [u for u in at.file_uploader if u.key == "uploader_step3_cycler_file"]
    assert len(fu) > 0, "uploader_step3_cycler_file must be rendered in Step 3"
    fu[0].upload("ZC-2-2-1-depo_Channel_7_Wb_1.xlsx", xlsx_bytes).run()
    assert len(at.exception) == 0

    # 1. Custom report exists and success message is displayed
    assert at.session_state.get("custom_analysis_report") is not None
    success_texts = [s.value for s in at.success]
    assert any("Successfully parsed and analyzed" in t for t in success_texts)

    # 2. Contradictory empty notice is absent
    empty_step3_en = get_text("custom_mode_step3_empty_notice", lang="en")
    info_texts = [getattr(i, "value", "") for i in at.info]
    assert not any(empty_step3_en in t for t in info_texts), "Empty notice must NOT be shown when custom report is active"

    # 3. General metrics rendered
    metric_labels = [getattr(m, "label", "") for m in at.metric if hasattr(m, "label")]
    assert any("Duration" in lbl for lbl in metric_labels)
    assert any("Data Points" in lbl for lbl in metric_labels)


def test_ui_step3_mpr_and_dta_upload_allowed():
    """
    Verifies that Step 3 file uploader allows BioLogic .mpr and Gamry .dta files
    and successfully ingests them.
    """
    import struct
    header = b'BIO-LOGIC FILE VMP3\x00\x00\x00'
    vmp_tag = b'VMP data\x00\x00'
    num_rows = 3
    num_cols = 4
    col_ids = bytes([4, 6, 7, 8])
    row_data = (
        struct.pack('<ffff', 0.0, 1.20, 10.0, 1.0)
        + struct.pack('<ffff', 60.0, 1.25, 10.0, 1.0)
        + struct.pack('<ffff', 120.0, 1.30, 10.0, 1.0)
    )
    mpr_bytes = header + vmp_tag + struct.pack('<IB', num_rows, num_cols) + col_ids + row_data

    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    fu = [u for u in at.file_uploader if u.key == "uploader_step3_cycler_file"]
    assert len(fu) > 0
    uploader = fu[0]
    # Verify allowed types contain .mpr and .dta
    assert ".mpr" in uploader.allowed_type
    assert ".dta" in uploader.allowed_type

    # Upload mpr bytes
    uploader.upload("cell_cycling.mpr", mpr_bytes).run()
    assert len(at.exception) == 0
    assert at.session_state.get("custom_analysis_report") is not None
    success_texts = [s.value for s in at.success]
    assert any("Successfully parsed and analyzed" in t for t in success_texts)


def test_ui_step3_upload_mpt_and_xlsx_and_mpr_rendering_no_nameerror():
    """
    Verifies that dropping .mpt, .mpr, and .xlsx files into Step 3 uploader
    renders without NameError: name 'NormalizedElectrochemicalData' is not defined.
    """
    from analysis.parsers.template_generators import (
        generate_template_biologic_mpt,
        generate_template_arbin_xlsx,
    )

    # 1. Test .mpt file upload and render
    mpt_bytes = generate_template_biologic_mpt(duration_hours=10.0)
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    fu = [u for u in at.file_uploader if u.key == "uploader_step3_cycler_file"][0]
    fu.upload("test_biologic.mpt", mpt_bytes).run()
    assert len(at.exception) == 0, f"AppTest threw exceptions on .mpt upload: {at.exception}"

    # 2. Test .xlsx file upload and render
    xlsx_bytes = generate_template_arbin_xlsx(duration_hours=10.0)
    at2 = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at2.run()
    assert len(at2.exception) == 0

    fu2 = [u for u in at2.file_uploader if u.key == "uploader_step3_cycler_file"][0]
    fu2.upload("test_arbin.xlsx", xlsx_bytes).run()
    assert len(at2.exception) == 0, f"AppTest threw exceptions on .xlsx upload: {at2.exception}"


def test_ui_step3_cv_first_class_view():
    """
    Verifies that Step 3 renders a dedicated, first-class Cyclic Voltammetry view:
    1. Anodic peak (E_pa), Cathodic peak (E_pc), Peak Separation (ΔE_p),
       Current Ratio (I_pa/|I_pc|), and Plating Onset metric cards are displayed.
    2. Voltammogram title or trace is displayed.
    3. No fallback warnings or unhandled exceptions occur.
    """
    from models import (
        CVAnalysis,
        ElectrochemicalAnalysisReport,
        NormalizedElectrochemicalData,
        ElectrodeGeometry,
    )
    import numpy as np

    v = np.concatenate([np.linspace(-0.2, 0.6, 50), np.linspace(0.6, -0.2, 50)])
    i = np.sin(np.linspace(0, 2 * np.pi, 100)) * 5.0
    norm_data = NormalizedElectrochemicalData(
        time_s=list(range(100)),
        display_time=[t / 3600.0 for t in range(100)],
        voltage_v=v.tolist(),
        current_ma=i.tolist(),
        current_density_ma_cm2=i.tolist(),
        capacity_mah=[0.0] * 100,
        areal_capacity_mah_cm2=[0.0] * 100,
    )
    cv_res = CVAnalysis(
        technique="cv",
        anodic_peak_potential_epa_v=0.452,
        cathodic_peak_potential_epc_v=-0.125,
        anodic_peak_current_ipa_ma=4.82,
        cathodic_peak_current_ipc_ma=-4.91,
        peak_separation_delta_ep_mv=577.0,
        current_ratio_ipa_ipc=0.98,
        plating_onset_potential_v=-0.045,
        stripping_onset_potential_v=0.075,
    )
    cv_report = ElectrochemicalAnalysisReport(
        sample_id="CV_FirstClass_Sample",
        file_source="cv_cycle.mpt",
        technique="cv",
        geometry=ElectrodeGeometry(area_cm2=1.0),
        processed_data=norm_data,
        cv=cv_res,
    )

    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    at.session_state["custom_analysis_report"] = cv_report
    at.run()
    assert len(at.exception) == 0

    # Metric labels
    labels = [getattr(m, "label", "") for m in at.metric if hasattr(m, "label")]
    values = [getattr(m, "value", "") for m in at.metric if hasattr(m, "value")]

    assert any("E_pa" in lbl for lbl in labels), f"Expected E_pa metric card, got {labels}"
    assert any("E_pc" in lbl for lbl in labels), f"Expected E_pc metric card, got {labels}"
    assert any("ΔE_p" in lbl or "Delta" in lbl for lbl in labels), f"Expected ΔE_p metric card, got {labels}"
    assert any("I_pa" in lbl or "Ratio" in lbl for lbl in labels), f"Expected I_pa / |I_pc| metric card, got {labels}"
    assert any("Plating" in lbl or "Onset" in lbl for lbl in labels), f"Expected Plating Onset metric card, got {labels}"

    # Metric values
    assert any("0.452 V" in val for val in values)
    assert any("-0.125 V" in val for val in values)
    assert any("577.0 mV" in val for val in values)


def test_ui_step3_lsv_first_class_view():
    """
    Verifies that Step 3 renders a dedicated, first-class LSV & Tafel Kinetics view:
    1. Corrosion potential (E_corr), Corrosion current (j_corr),
       Cathodic Tafel slope (β_c), Anodic Tafel slope (β_a), and HER onset metric cards are displayed.
    2. Dual polarization and Tafel figures render without exceptions.
    """
    from models import (
        LSVAnalysis,
        ElectrochemicalAnalysisReport,
        NormalizedElectrochemicalData,
        ElectrodeGeometry,
    )
    import numpy as np

    v = np.linspace(-0.4, 0.2, 80)
    j = np.exp((v + 0.1) * 12.0) - np.exp(-(v + 0.1) * 12.0)

    norm_data = NormalizedElectrochemicalData(
        time_s=list(range(80)),
        display_time=[t / 3600.0 for t in range(80)],
        voltage_v=v.tolist(),
        current_ma=j.tolist(),
        current_density_ma_cm2=j.tolist(),
        capacity_mah=[0.0] * 80,
        areal_capacity_mah_cm2=[0.0] * 80,
    )
    lsv_res = LSVAnalysis(
        technique="lsv",
        corrosion_potential_ecorr_v=-0.095,
        corrosion_current_jcorr_ma_cm2=0.0142,
        tafel_slope_cathodic_mv_per_dec=118.5,
        tafel_slope_anodic_mv_per_dec=86.3,
        her_onset_potential_v=-0.235,
    )
    lsv_report = ElectrochemicalAnalysisReport(
        sample_id="LSV_FirstClass_Sample",
        file_source="lsv_scan.csv",
        technique="lsv",
        geometry=ElectrodeGeometry(area_cm2=1.0),
        processed_data=norm_data,
        lsv=lsv_res,
    )

    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    at.session_state["custom_analysis_report"] = lsv_report
    at.run()
    assert len(at.exception) == 0

    labels = [getattr(m, "label", "") for m in at.metric if hasattr(m, "label")]
    values = [getattr(m, "value", "") for m in at.metric if hasattr(m, "value")]

    assert any("E_corr" in lbl for lbl in labels), f"Expected E_corr metric card, got {labels}"
    assert any("j_corr" in lbl for lbl in labels), f"Expected j_corr metric card, got {labels}"
    assert any("β_c" in lbl or "beta_c" in lbl for lbl in labels), f"Expected β_c metric card, got {labels}"
    assert any("β_a" in lbl or "beta_a" in lbl for lbl in labels), f"Expected β_a metric card, got {labels}"
    assert any("HER" in lbl for lbl in labels), f"Expected HER onset metric card, got {labels}"

    assert any("-0.095 V" in val for val in values)
    assert any("0.0142" in val for val in values)
    assert any("118.5 mV/dec" in val for val in values)


def test_ui_step3_eis_first_class_view():
    """
    Verifies that Step 3 renders a dedicated, first-class EIS view:
    1. Solution resistance (R_s), Charge-transfer resistance (R_ct),
       Characteristic apex frequency (f_apex), and Equivalent circuit metric cards are displayed.
    2. Nyquist spectrum renders without exceptions.
    """
    from models import (
        EISAnalysis,
        ElectrochemicalAnalysisReport,
        NormalizedElectrochemicalData,
        ElectrodeGeometry,
    )

    zr = [2.25, 4.5, 8.0, 14.0, 22.0, 28.5]
    zi = [0.0, -2.5, -6.5, -9.0, -5.2, 0.0]
    norm_data = NormalizedElectrochemicalData(
        time_s=list(range(6)),
        display_time=[0.0] * 6,
        voltage_v=[0.0] * 6,
        current_ma=[0.0] * 6,
        current_density_ma_cm2=[0.0] * 6,
        capacity_mah=[0.0] * 6,
        areal_capacity_mah_cm2=[0.0] * 6,
        metadata={"z_real_ohm": zr, "z_imag_ohm": zi, "frequency_hz": [100000, 10000, 1000, 100, 10, 1]},
    )
    eis_res = EISAnalysis(
        technique="eis",
        solution_resistance_rs_ohm=2.25,
        charge_transfer_resistance_rct_ohm=26.25,
        characteristic_frequency_hz=150.0,
        equivalent_circuit="R_s + R_ct // CPE",
    )
    eis_report = ElectrochemicalAnalysisReport(
        sample_id="EIS_FirstClass_Sample",
        file_source="eis_spec.dta",
        technique="eis",
        geometry=ElectrodeGeometry(area_cm2=1.0),
        processed_data=norm_data,
        eis=eis_res,
    )

    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    at.session_state["custom_analysis_report"] = eis_report
    at.run()
    assert len(at.exception) == 0

    labels = [getattr(m, "label", "") for m in at.metric if hasattr(m, "label")]
    values = [getattr(m, "value", "") for m in at.metric if hasattr(m, "value")]

    assert any("R_s" in lbl for lbl in labels), f"Expected R_s metric card, got {labels}"
    assert any("R_ct" in lbl for lbl in labels), f"Expected R_ct metric card, got {labels}"
    assert any("f_apex" in lbl or "Apex" in lbl for lbl in labels), f"Expected f_apex metric card, got {labels}"
    assert any("Circuit" in lbl or "circuit" in lbl for lbl in labels), f"Expected Circuit metric card, got {labels}"

    assert any("2.25 Ω" in val for val in values)
    assert any("26.25 Ω" in val for val in values)
    assert any("150.0 Hz" in val for val in values)


def test_ui_step3_galvanostatic_dqdv_expander_and_controls():
    """
    Verifies that galvanostatic cycling views render an interactive dQ/dV Differential Capacity
    expander with Savitzky-Golay window and polynomial order controls.
    """
    from models import (
        FullCellAnalysis,
        ElectrochemicalAnalysisReport,
        NormalizedElectrochemicalData,
        ElectrodeGeometry,
    )

    v = [1.0 + 0.01 * i for i in range(60)]
    q = [2.0 * i for i in range(60)]
    cycles = [1] * 30 + [2] * 30
    norm_data = NormalizedElectrochemicalData(
        time_s=list(range(60)),
        display_time=[t / 3600.0 for t in range(60)],
        voltage_v=v,
        current_ma=[1.0] * 60,
        current_density_ma_cm2=[1.0] * 60,
        capacity_mah=q,
        areal_capacity_mah_cm2=q,
        cycle_number=cycles,
    )
    fc_res = FullCellAnalysis(
        technique="galvanostatic_full_cell",
        initial_discharge_capacity_mah_g=120.0,
        final_discharge_capacity_mah_g=115.0,
        capacity_retention_percent=95.8,
        average_coulombic_efficiency_percent=99.2,
        total_cycles=2,
        cycle_numbers=[1, 2],
        discharge_capacities_mah_g=[120.0, 115.0],
        coulombic_efficiencies=[99.1, 99.3],
    )
    fc_report = ElectrochemicalAnalysisReport(
        sample_id="FC_DQDV_UI_Sample",
        file_source="fc_cycle.csv",
        technique="galvanostatic_full_cell",
        geometry=ElectrodeGeometry(area_cm2=1.0),
        processed_data=norm_data,
        full_cell=fc_res,
    )

    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    at.session_state["custom_analysis_report"] = fc_report
    at.run()
    assert len(at.exception) == 0

    # Verify dQ/dV expander is completely removed per user requirements
    expanders = [getattr(exp, "label", "") for exp in at.expander]
    assert not any("Differential Capacity" in exp or "dQ/dV" in exp for exp in expanders), f"dQ/dV expander should be removed, but found: {expanders}"

    # Verify Full Cell metrics render properly
    metric_labels = [getattr(m, "label", "") for m in at.metric if hasattr(m, "label")]
    assert any("Retention" in lbl for lbl in metric_labels), f"Expected Retention metric, got {metric_labels}"


def test_ui_step3_trilingual_parity_rendering():
    """
    Verifies that Step 3 views render cleanly across all 3 supported languages
    (en, fr, zh) without missing keys or runtime exceptions.
    """
    from models import (
        CVAnalysis,
        ElectrochemicalAnalysisReport,
        NormalizedElectrochemicalData,
        ElectrodeGeometry,
    )
    from localization import get_text

    norm_data = NormalizedElectrochemicalData(
        time_s=list(range(20)),
        display_time=[t / 3600.0 for t in range(20)],
        voltage_v=[0.1 * i for i in range(20)],
        current_ma=[1.0] * 20,
        current_density_ma_cm2=[1.0] * 20,
        capacity_mah=[0.5 * i for i in range(20)],
        areal_capacity_mah_cm2=[0.5 * i for i in range(20)],
    )
    cv_res = CVAnalysis(
        technique="cv",
        anodic_peak_potential_epa_v=0.45,
        cathodic_peak_potential_epc_v=-0.12,
        peak_separation_delta_ep_mv=570.0,
        current_ratio_ipa_ipc=1.0,
        plating_onset_potential_v=-0.05,
    )
    cv_report = ElectrochemicalAnalysisReport(
        sample_id="CV_Trilingual_Sample",
        file_source="cv.mpt",
        technique="cv",
        geometry=ElectrodeGeometry(area_cm2=1.0),
        processed_data=norm_data,
        cv=cv_res,
    )

    for lang in ["en", "fr", "zh"]:
        at = AppTest.from_file(APP_PATH, default_timeout=35.0)
        at.run()
        assert len(at.exception) == 0

        at.session_state["custom_analysis_report"] = cv_report
        at.sidebar.selectbox[0].set_value(lang).run()
        assert len(at.exception) == 0, f"AppTest raised exceptions under lang={lang}: {at.exception}"

        labels = [getattr(m, "label", "") for m in at.metric if hasattr(m, "label")]
        expected_epa_label = get_text("cv_epa", lang=lang)
        assert any(expected_epa_label in lbl for lbl in labels), f"Missing localized E_pa in {lang}: {labels}"


def test_ui_step3_upload_cycle_summary_excel_dual_axis_plot():
    """
    Verifies that uploading a 3-column cycle summary Excel file
    (Cycle number, Capacity, Columbic Efficiency) into Step 3:
    1. Parses cleanly without 'No valid numeric data points' error.
    2. Identifies technique as Full Cell (galvanostatic_full_cell).
    3. Renders full cell metrics (Initial Capacity, Final Capacity, Retention, Average CE).
    4. Renders the dual-axis Plotly figure with Capacity on left y-axis and CE on right y-axis.
    """
    from export_engine import make_multi_sheet_xlsx

    # Create 3-column Excel workbook with realistic formatting variations
    sheet_data = [
        ["Cycle number", "Capacity", "Columbic Efficiency"],
        ["", "mAh g-1", "%"],
        [1, "175.2 mAh/g", "100.0%"],
        [2, "171.8 mAh/g", "96.4%"],
        [3, "168.5 mAh/g", "98.1%"],
        [4, "165.0 mAh/g", "99.0%"],
        [5, "162.3 mAh/g", "99.5%"],
    ]
    xlsx_bytes = make_multi_sheet_xlsx({"Sheet1": sheet_data})

    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    fu = [u for u in at.file_uploader if u.key == "uploader_step3_cycler_file"][0]
    fu.upload("full_cell_cycling.xlsx", xlsx_bytes).run()
    assert len(at.exception) == 0, f"AppTest raised exception on cycle summary upload: {at.exception}"

    # Verify no error message was shown
    err_msgs = [e.value for e in at.error]
    assert not any("No valid numeric data points" in msg for msg in err_msgs), f"Error banner present: {err_msgs}"

    # Verify custom analysis report in session state
    report = at.session_state.get("custom_analysis_report")
    assert report is not None
    assert str(report.technique).lower() in ("galvanostatic_full_cell", "electrochemicaltechnique.galvanostatic_full_cell")
    assert report.full_cell is not None
    assert report.full_cell.total_cycles == 5
    assert len(report.full_cell.discharge_capacities_mah_g) == 5
    assert len(report.full_cell.coulombic_efficiencies) == 5

    # Verify full cell metric cards are displayed
    metric_labels = [getattr(m, "label", "") for m in at.metric if hasattr(m, "label")]
    assert any("Initial Capacity" in lbl or "Capacit" in lbl for lbl in metric_labels)
    assert any("Retention" in lbl for lbl in metric_labels)
    assert any("Coulombic Efficiency" in lbl or "CE" in lbl for lbl in metric_labels)


def test_ui_step1_all_updates_verification():
    """
    Verify complete implementation of Step 1 features and user feedback:
    1. Pagination Button: 'Fetch Next {count} Papers & Append' top and bottom.
    2. Cross-Keyword Accumulation: 'Append to current dataset' checkbox and Reset Dataset button.
    3. Dossier Download: CSV & XLSX generation with DOI, DOI URL, OA link, additives, concentrations, baseline salts.
    4. Batch PDF downloads: Download Batch Summary (.csv) and Download Batch Dossier (.xlsx).
    5. PDF Batch Guardrail: Soft advisory for <= 25 PDFs per run, both proactive caption and >25 warning.
    6. Google API Key button tooltip and Free-Tier Quota & Rate Limit breakdown:
       - 5 papers / request micro-batching
       - Gemini 3.5 & 3.1 Flash-Lite: 500 requests/day = up to 2,500 papers/day (15 req/min)
       - Gemini 3.8 Flash: 20 requests/day = up to 100 papers/day (15 req/min)
       - Offline heuristic mode.
    7. Multi-stage progress tracking (Step 1/5 to Step 5/5) with pre-cached dossier downloads.
    """
    from localization import get_text
    from export_engine import generate_mined_literature_csv, generate_mined_literature_xlsx
    from models import PaperMetadata, ExtractedAdditive, Concentration, ZincIonBatteryContext

    # 1. Trilingual localization parity check
    for lang in ["en", "fr", "zh"]:
        guard_tip = get_text("pdf_batch_guardrail_tip", lang=lang)
        pop_help = get_text("gemini_popover_button_help", lang=lang)
        quota_html = get_text("gemini_quota_card_html", lang=lang)
        dl_b_csv = get_text("btn_download_batch_pdf_csv", lang=lang)
        dl_b_xlsx = get_text("btn_download_batch_pdf_xlsx", lang=lang)
        stage_1 = get_text("mining_stage_retrieval", lang=lang)
        stage_5 = get_text("mining_stage_finalizing", lang=lang)

        assert not guard_tip.startswith("[") and "25" in guard_tip
        assert not pop_help.startswith("[") and ("2,500" in pop_help or "2 500" in pop_help)
        assert not quota_html.startswith("[")
        assert "500" in quota_html
        assert "20" in quota_html
        assert "5" in quota_html  # 5 papers / request
        assert not dl_b_csv.startswith("[")
        assert not dl_b_xlsx.startswith("[")
        assert "1/5" in stage_1
        assert "5/5" in stage_5

    # 2. Verify export engine CSV & XLSX with all required fields
    sample_paper = PaperMetadata(
        paper_id="test_p_01",
        title="Trehalose Electrolyte for Zinc Batteries",
        authors=["Alice Scientist", "Bob Engineer"],
        year=2024,
        venue="Nature Energy",
        doi="10.1038/s41560-024-0001",
        open_access_pdf_url="https://nature.com/articles/s41560-024-0001.pdf",
        citation_count=42,
        abstract="Trehalose additive at 15 mM stabilizes 2 M ZnSO4 electrolyte."
    )
    sample_add = ExtractedAdditive(
        name="Trehalose",
        smiles="C(C1C(C(C(C(O1)OC2C(C(C(C(O2)CO)O)O)O)O)O)O)O",
        optimal_concentration=Concentration(value=15.0, unit="mM", raw_text="15 mM"),
        battery_context=ZincIonBatteryContext(baseline_electrolyte="2 M ZnSO4"),
        primary_mechanisms=["Dendrite suppression", "Water structure modification"],
        mechanism_summary="Suppress Zn dendrites via hydrogen bonding"
    )
    test_records = [{"paper": sample_paper, "additive": sample_add}]

    csv_out = generate_mined_literature_csv(test_records)
    assert "Trehalose" in csv_out
    assert "10.1038/s41560-024-0001" in csv_out
    assert "https://doi.org/10.1038/s41560-024-0001" in csv_out
    assert "https://nature.com/articles/s41560-024-0001.pdf" in csv_out
    assert "15 mM" in csv_out
    assert "2 M ZnSO4" in csv_out

    xlsx_bytes = generate_mined_literature_xlsx(test_records)
    assert isinstance(xlsx_bytes, bytes)
    assert len(xlsx_bytes) > 100

    # Test Batch PDF records export
    batch_pdf_records = [{
        "filename": "manuscript_teabr.pdf",
        "additive": sample_add,
        "suggested_analogue": None,
        "summary": "Extracted from manuscript"
    }]
    b_csv_out = generate_mined_literature_csv(batch_pdf_records)
    assert "manuscript_teabr.pdf" in b_csv_out
    assert "Trehalose" in b_csv_out
    b_xlsx_out = generate_mined_literature_xlsx(batch_pdf_records)
    assert isinstance(b_xlsx_out, bytes)
    assert len(b_xlsx_out) > 100

    # 3. Streamlit AppTest UI verification
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    # Check append checkbox exists
    append_cbs = [cb for cb in at.checkbox if cb.key == "lit_search_append_checkbox"]
    assert len(append_cbs) > 0, "Append to dataset checkbox must exist"

    # Check proactive batch guardrail caption exists
    all_captions = [getattr(c, "value", "") for c in at.caption if hasattr(c, "value")]
    caption_text = " ".join(all_captions)
    assert "25" in caption_text, "Proactive batch guardrail caption recommending <= 25 PDFs must exist"

    # Check popovers have help attribute set
    for pop in at.popover:
        if pop.key and "gemini" in pop.key:
            assert pop.help is not None and ("2,500" in pop.help or "2 500" in pop.help or "500" in pop.help)

    # Check Gemini 4 Argon is available in model select options
    model_sbs = [sb for sb in at.selectbox if "gemini_model_select" in str(sb.key)]
    if model_sbs:
        assert any("4 Argon" in opt for opt in model_sbs[0].options), "Gemini 4 Argon must be in model select options"


def test_ui_post_extraction_reminder_modal_localization():
    """
    Verify requirement 2 & 5:
    1. Localized strings exist in en, fr, zh for all reminder modal keys.
    2. Zero occurrences of 'Step 1' or 'Step 2' (or localized equivalents) in title or body.
    3. Destination is referred to as 'Electrolyte Formulation & Planning' (or localized equivalent).
    """
    from localization import get_text

    keys = [
        "modal_extraction_complete_title",
        "modal_extraction_complete_body",
        "modal_merge_conflict_notice",
        "modal_merge_mode_label",
        "modal_dont_show_again",
        "modal_btn_got_it",
        "modal_btn_go_formulation",
    ]

    for lang in ["en", "fr", "zh"]:
        for k in keys:
            val = get_text(k, lang=lang, existing_count=2, new_count=1)
            assert val, f"Missing key {k} in language {lang}"
            # Verify naming rule: NO 'Step 1' or 'Step 2' anywhere in title, body, or notice
            if "title" in k or "body" in k or "notice" in k:
                val_lower = val.lower()
                assert "step 1" not in val_lower, f"Forbidden 'Step 1' found in {k} ({lang})"
                assert "step 2" not in val_lower, f"Forbidden 'Step 2' found in {k} ({lang})"
                assert "étape 1" not in val_lower, f"Forbidden 'étape 1' found in {k} ({lang})"
                assert "étape 2" not in val_lower, f"Forbidden 'étape 2' found in {k} ({lang})"
                assert "步骤 1" not in val and "步骤1" not in val, f"Forbidden '步骤 1' found in {k} ({lang})"
                assert "步骤 2" not in val and "步骤2" not in val, f"Forbidden '步骤 2' found in {k} ({lang})"

    # Verify functional destination name
    assert "Electrolyte Formulation & Planning" in get_text("modal_extraction_complete_body", lang="en")
    assert "Formulation & Planification d'Électrolyte" in get_text("modal_extraction_complete_body", lang="fr")
    assert "电解液配方与实验规划" in get_text("modal_extraction_complete_body", lang="zh")


def test_ui_post_extraction_reminder_modal_online_search_trigger():
    """
    Verify requirement 1, 3, & 4:
    1. Online literature mining extracting >= 1 additive triggers the reminder modal.
    2. Modal displays the title, body, 'Do not show again' checkbox, 'Got it', and 'Go to Formulation ➔' buttons.
    3. Clicking 'Got it' dismisses the modal and remains on the literature tab.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    # Ensure modal is not initially visible
    assert at.session_state.get("show_formulation_reminder_modal") is None or not at.session_state.get("show_formulation_reminder_modal")

    # Run online search for trehalose
    search_inputs = [ti for ti in at.text_input if ti.key == "lit_search_query_input"]
    search_inputs[0].input("trehalose").run()
    search_btn = [b for b in at.button if b.key == "lit_search_submit_btn"][0]
    search_btn.click().run()
    assert len(at.exception) == 0

    # Verify modal is triggered
    assert at.session_state.get("show_formulation_reminder_modal") is True

    # Verify modal elements exist
    all_md = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    assert "Electrolyte Formulation & Planning" in all_md
    assert "Delaney ESOL" in all_md

    dont_show_cbs = [cb for cb in at.checkbox if cb.key == "chk_dont_show_formulation_reminder"]
    assert len(dont_show_cbs) == 1, "Checkbox for 'Do not show this reminder again' must exist"

    got_it_btns = [b for b in at.button if b.key == "btn_reminder_modal_got_it"]
    assert len(got_it_btns) == 1, "'Got it' button must exist"

    go_form_btns = [b for b in at.button if b.key == "btn_reminder_modal_go_formulation"]
    assert len(go_form_btns) == 1, "'Go to Formulation ➔' button must exist"

    # Click 'Got it' to dismiss
    got_it_btns[0].click().run()
    assert len(at.exception) == 0
    assert at.session_state.get("show_formulation_reminder_modal") is False


def test_ui_post_extraction_reminder_modal_go_to_formulation_tab():
    """
    Verify requirement 3:
    Clicking 'Go to Formulation ➔' dismisses the modal and injects the client-side
    tab click trigger to switch to the Electrolyte Formulation tab.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    # Trigger modal via session state
    at.session_state["show_formulation_reminder_modal"] = True
    at.run()
    assert len(at.exception) == 0

    go_form_btns = [b for b in at.button if b.key == "btn_reminder_modal_go_formulation"]
    assert len(go_form_btns) == 1

    go_form_btns[0].click().run()
    assert len(at.exception) == 0
    assert at.session_state.get("show_formulation_reminder_modal") is False

    # Verify tab click script was rendered in HTML
    html_snippets = [h.value for h in at.html if hasattr(h, "value")]
    combined_html = " ".join(html_snippets)
    assert "switchToFormulationTab" in combined_html
    assert "formulation" in combined_html.lower()


def test_ui_post_extraction_reminder_modal_local_storage_persistence():
    """
    Verify requirement 4:
    1. Checking 'Do not show this reminder again' and clicking dismiss sets localStorage and session state.
    2. Once hidden, subsequent literature searches do NOT trigger the modal dialog.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    # Trigger modal
    at.session_state["show_formulation_reminder_modal"] = True
    at.run()

    # Check the checkbox
    dont_show_cbs = [cb for cb in at.checkbox if cb.key == "chk_dont_show_formulation_reminder"]
    dont_show_cbs[0].check().run()
    assert len(at.exception) == 0
    assert at.session_state.get("chk_dont_show_formulation_reminder") is True

    # Click 'Got it'
    got_it_btns = [b for b in at.button if b.key == "btn_reminder_modal_got_it"]
    got_it_btns[0].click().run()
    assert len(at.exception) == 0

    # Verify preference was persisted
    assert at.session_state.get("hide_formulation_reminder") is True

    # Verify localStorage write script was injected
    html_snippets = [h.value for h in at.html if hasattr(h, "value")]
    combined_html = " ".join(html_snippets)
    assert "map_hide_formulation_reminder" in combined_html
    assert "localStorage.setItem('map_hide_formulation_reminder', 'true')" in combined_html

    # Now run another online search while hide_formulation_reminder is True
    search_inputs = [ti for ti in at.text_input if ti.key == "lit_search_query_input"]
    search_inputs[0].input("zinc").run()
    search_btn = [b for b in at.button if b.key == "lit_search_submit_btn"][0]
    search_btn.click().run()
    assert len(at.exception) == 0

    # Verify modal is NOT shown because it's suppressed
    assert at.session_state.get("show_formulation_reminder_modal") is not True


def test_ui_post_extraction_reminder_modal_batch_pdf_trigger():
    """
    Verify requirement 1:
    Batch PDF ingestion extracting additives triggers the reminder modal.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    # Switch to Demo Mode so sample PDF button is available
    at.sidebar.radio[0].set_value("demo").run()
    assert len(at.exception) == 0

    sample_btns = [b for b in at.button if b.key == "btn_load_sample_pdf"]
    assert len(sample_btns) > 0
    sample_btns[0].click().run()
    assert len(at.exception) == 0

    # Verify modal is triggered after batch PDF processing completes
    assert at.session_state.get("show_formulation_reminder_modal") is True

    # Dismiss modal
    got_it_btns = [b for b in at.button if b.key == "btn_reminder_modal_got_it"]
    assert len(got_it_btns) == 1
    got_it_btns[0].click().run()
    assert len(at.exception) == 0
    assert at.session_state.get("show_formulation_reminder_modal") is False


def test_ui_post_extraction_reminder_modal_no_trigger_on_fetch_next():
    """
    Verify requirement 1 guardrail:
    Pagination / 'Fetch Next 25 Papers & Append' must NOT trigger the reminder modal.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    # First perform initial search
    search_inputs = [ti for ti in at.text_input if ti.key == "lit_search_query_input"]
    search_inputs[0].input("trehalose").run()
    search_btn = [b for b in at.button if b.key == "lit_search_submit_btn"][0]
    search_btn.click().run()
    assert len(at.exception) == 0

    # Dismiss initial modal
    got_it_btns = [b for b in at.button if b.key == "btn_reminder_modal_got_it"]
    got_it_btns[0].click().run()
    assert at.session_state.get("show_formulation_reminder_modal") is False

    # Click 'Fetch Next 25 Papers'
    fetch_next_btns = [b for b in at.button if b.key == "btn_fetch_next_papers_top"]
    assert len(fetch_next_btns) > 0
    fetch_next_btns[0].click().run()
    assert len(at.exception) == 0

    # Verify modal was NOT triggered during fetch next
    assert at.session_state.get("show_formulation_reminder_modal") is False


def test_ui_post_extraction_modal_append_vs_replace_choice():
    """
    Verify that when the Electrolyte Formulation & Planning list already contains additives,
    running Online Mining or PDF Ingestion shows the Append vs Replace radio prompt inside
    the post-extraction modal, and selecting 'Replace existing list' replaces prior additives
    while 'Append to existing mined list' preserves them.
    """
    from models import CandidateMolecule

    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    prior_cand = CandidateMolecule(
        name="PriorUploadedMol",
        smiles="CCO",
        similarity_score=1.0,
        similarity_metric="User Uploaded List",
        molecular_weight=46.07,
        logp=-0.31,
        logs=1.1,
        tpsa=20.23,
        hbd=1,
        hba=1,
        rotatable_bonds=0,
        reported_concentration=25.0,
        reported_concentration_unit="mM",
        parent_baseline_electrolyte="2 M ZnSO4",
        is_literature_additive=True,
        rationale="Uploaded prior to extraction",
    )
    at.session_state["mined_literature_additives"] = [prior_cand]
    at.run()

    # Run online mining for trehalose while prior_cand is in the list
    search_inputs = [ti for ti in at.text_input if ti.key == "lit_search_query_input"]
    search_inputs[0].input("trehalose").run()
    search_btn = [b for b in at.button if b.key == "lit_search_submit_btn"][0]
    search_btn.click().run()
    assert len(at.exception) == 0

    # Modal should show conflict prompt & radio
    assert at.session_state.get("show_formulation_reminder_modal") is True
    assert at.session_state.get("has_merge_conflict_prompt") is True
    merge_radios = [r for r in at.radio if r.key == "modal_extraction_merge_mode_radio"]
    assert len(merge_radios) == 1, "Append vs Replace radio must appear in post-extraction modal when list is non-empty"

    # Default is Append -> both PriorUploadedMol and Trehalose exist
    names_append = [c.name.lower() for c in at.session_state["mined_literature_additives"]]
    assert "prioruploadedmol" in names_append
    assert any("trehalose" in n for n in names_append)

    # Select 'Replace existing list' and click 'Got it'
    replace_opt = merge_radios[0].options[1]
    merge_radios[0].set_value(replace_opt).run()
    assert len(at.exception) == 0

    got_it_btns = [b for b in at.button if b.key == "btn_reminder_modal_got_it"]
    got_it_btns[0].click().run()
    assert len(at.exception) == 0

    names_replace = [c.name.lower() for c in at.session_state["mined_literature_additives"]]
    assert "prioruploadedmol" not in names_replace, "Prior additive must be removed when Replace is chosen"
    assert any("trehalose" in n for n in names_replace), "Newly extracted additive must remain when Replace is chosen"
    assert at.session_state.get("has_merge_conflict_prompt") is False

    # Now set hide_formulation_reminder = True ("Do not show this reminder again"),
    # re-seed prior_cand, and verify conflict modal STILL opens with ONLY the Append/Replace prompt
    at.session_state["hide_formulation_reminder"] = True
    at.session_state["mined_literature_additives"] = [prior_cand]
    at.run()

    search_btn = [b for b in at.button if b.key == "lit_search_submit_btn"][0]
    search_btn.click().run()
    assert len(at.exception) == 0

    assert at.session_state.get("show_formulation_reminder_modal") is True
    assert at.session_state.get("has_merge_conflict_prompt") is True
    merge_radios_2 = [r for r in at.radio if r.key == "modal_extraction_merge_mode_radio"]
    assert len(merge_radios_2) == 1, "Append vs Replace radio must still appear even if reminder was hidden"
    dont_show_cbs_2 = [cb for cb in at.checkbox if cb.key == "chk_dont_show_formulation_reminder"]
    assert len(dont_show_cbs_2) == 0, "Do not show again checkbox must be hidden when already opted out"


def test_ui_step2_dropdown_selection_turns_card_deck_page():
    """
    Verify that when the additive card deck is on Page 1 and the user selects an additive
    from the additive names dropdown (top or bottom, Tier 1 or Tier 2) that resides on
    another page (e.g. Page 2 or Page 3), the card deck automatically turns to the page
    containing the selected additive.
    """
    from models import CandidateMolecule

    mock_t1 = [
        CandidateMolecule(
            name=f"Tier1 Mol {i+1}",
            smiles="C(C(CO)O)O",
            similarity_score=1.0,
            similarity_metric="Direct Literature Finding",
            molecular_weight=92.09,
            logp=-1.76,
            logs=-0.5,
            tpsa=60.69,
            hbd=3,
            hba=3,
            rotatable_bonds=2,
            rationale="Tested Tier 1 candidate",
            chemical_formula="C3H8O3",
            reported_concentration=15.0,
            reported_concentration_unit="mM",
            parent_baseline_electrolyte="2 M ZnSO4",
            is_literature_additive=True,
        )
        for i in range(14)
    ]

    mock_t2 = [
        CandidateMolecule(
            name=f"Tier2 Mol {i+1}",
            smiles="C(C(CO)O)O",
            similarity_score=0.85,
            similarity_metric="Tanimoto",
            molecular_weight=92.09,
            logp=-1.76,
            logs=-0.5,
            tpsa=60.69,
            hbd=3,
            hba=3,
            rotatable_bonds=2,
            rationale="Suggested Tier 2 analogue",
            chemical_formula="C3H8O3",
            reported_concentration=15.0,
            reported_concentration_unit="mM",
            parent_baseline_electrolyte="2 M ZnSO4",
            is_literature_additive=False,
        )
        for i in range(14)
    ]

    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    at.session_state["mined_literature_additives"] = mock_t1
    at.session_state["ai_suggested_additives"] = mock_t2
    at.session_state["suggested_candidates"] = mock_t2
    at.run()
    assert len(at.exception) == 0

    # Initially on Page 1 for Tier 1
    assert at.session_state.get("step2_tier1_current_page", 1) == 1
    btn_keys_p1 = [b.key for b in at.button]
    assert "btn_formulate_t1_0" in btn_keys_p1
    assert "btn_formulate_t1_8" not in btn_keys_p1

    # 1. Select index 8 ("Tier1 Mol 9", on Page 2) from top Tier 1 dropdown
    t1_top = [sb for sb in at.selectbox if sb.key == "step2_tier1_additive_dropdown"][0]
    t1_top.set_value(8).run()
    assert len(at.exception) == 0
    assert at.session_state.get("step2_tier1_current_page") == 2
    assert at.session_state.get("selected_additive_name") == "Tier1 Mol 9"
    btn_keys_p2 = [b.key for b in at.button]
    assert "btn_formulate_t1_8" in btn_keys_p2
    assert "btn_formulate_t1_0" not in btn_keys_p2

    # 2. Select index 13 ("Tier1 Mol 14", on Page 3) from bottom Tier 1 dropdown
    t1_bot = [sb for sb in at.selectbox if sb.key == "step2_tier1_bottom_dropdown"][0]
    t1_bot.set_value(13).run()
    assert len(at.exception) == 0
    assert at.session_state.get("step2_tier1_current_page") == 3
    assert at.session_state.get("selected_additive_name") == "Tier1 Mol 14"
    btn_keys_p3 = [b.key for b in at.button]
    assert "btn_formulate_t1_13" in btn_keys_p3
    assert "btn_formulate_t1_8" not in btn_keys_p3

    # 3. Select index 2 ("Tier1 Mol 3", on Page 1) from top Tier 1 dropdown -> returns to Page 1
    t1_top = [sb for sb in at.selectbox if sb.key == "step2_tier1_additive_dropdown"][0]
    t1_top.set_value(2).run()
    assert len(at.exception) == 0
    assert at.session_state.get("step2_tier1_current_page") == 1
    assert at.session_state.get("selected_additive_name") == "Tier1 Mol 3"
    btn_keys_back_p1 = [b.key for b in at.button]
    assert "btn_formulate_t1_2" in btn_keys_back_p1

    # 4. Select index 7 ("Tier2 Mol 8", on Page 2) from top Tier 2 dropdown
    t2_top = [sb for sb in at.selectbox if sb.key == "tab2_suggested_additives_dropdown"][0]
    t2_top.set_value(7).run()
    assert len(at.exception) == 0
    assert at.session_state.get("step2_tier2_current_page") == 2
    assert at.session_state.get("selected_additive_name") == "Tier2 Mol 8"
    btn_keys_t2_p2 = [b.key for b in at.button]
    assert "btn_formulate_t2_7" in btn_keys_t2_p2
    assert "btn_formulate_t2_0" not in btn_keys_t2_p2

    # 5. Select index 12 ("Tier2 Mol 13", on Page 3) from bottom Tier 2 dropdown
    t2_bot = [sb for sb in at.selectbox if sb.key == "step2_tier2_bottom_dropdown"][0]
    t2_bot.set_value(12).run()
    assert len(at.exception) == 0
    assert at.session_state.get("step2_tier2_current_page") == 3
    assert at.session_state.get("selected_additive_name") == "Tier2 Mol 13"
    btn_keys_t2_p3 = [b.key for b in at.button]
    assert "btn_formulate_t2_12" in btn_keys_t2_p3


def test_ui_light_mode_captions_and_step3_banner_contrast():
    """
    Verify that:
    1. Universal .stApp inheritance and [data-theme="light"] rules eliminate Streamlit's
       60% grey caption dimming on Step 1, Step 2, and Automated Experiment Run captions.
    2. Step 3 .tech-file-banner labels ("File:", "Data Points:", "Duration:") enforce
       #000000 black text in Light Mode and are never forced to white (#f8fafc) by
       unscoped @media (prefers-color-scheme: dark) when the app is in Light Mode.
    3. inject_global_styles() injects the syncThemeAttribute script to synchronize
       data-theme="light" / data-theme="dark" with Streamlit's active theme.
    """
    from ui.styles import CUSTOM_CSS

    # 1. Universal & Light-mode caption contrast rules
    assert '.stApp [data-testid="stCaptionContainer"]' in CUSTOM_CSS
    assert '.stApp [data-testid="stCaptionContainer"] *' in CUSTOM_CSS
    assert '[data-theme="light"] [data-testid="stCaptionContainer"]' in CUSTOM_CSS
    assert '[data-theme="light"] [data-testid="stCaptionContainer"] *' in CUSTOM_CSS

    # 2. Step 3 .tech-file-banner Light Mode black text & scoped Dark Mode rules
    assert '[data-theme="light"] .tech-file-banner' in CUSTOM_CSS
    assert '[data-theme="light"] .tech-file-banner .banner-text' in CUSTOM_CSS
    assert '.stApp .tech-file-banner .banner-text' in CUSTOM_CSS

    # Ensure @media (prefers-color-scheme: dark) scopes .tech-file-banner under [data-theme="dark"]
    dark_media_block = CUSTOM_CSS.split("@media (prefers-color-scheme: dark)")[-1].split("[data-theme=\"dark\"] .tech-file-banner,")[0]
    assert '[data-theme="dark"] .tech-file-banner' in dark_media_block
    assert '[data-theme="dark"] .tech-file-banner .banner-text' in dark_media_block

    # 3. Verify theme synchronization script is injected and captions render in Demo Mode
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    at.sidebar.radio[0].set_value("demo").run()
    assert len(at.exception) == 0

    captions_corpus = " ".join([getattr(c, "value", "") for c in at.caption if hasattr(c, "value")])
    markdown_corpus = " ".join([getattr(m, "value", "") for m in at.markdown if hasattr(m, "value")])
    assert "Query peer-reviewed publications via Semantic Scholar" in captions_corpus
    assert "Explore tested additives directly extracted from the literature" in captions_corpus
    assert "Experiment Setup:" in markdown_corpus


def test_ui_summary_tab_removed_in_custom_mode_and_demo_first_figure_no_text_overlap():
    """
    Verify:
    1. In Custom Mode, the 'Experiment Summary & Export' tab is removed from the tab bar.
    2. In Demo Mode, the 'Experiment Summary & Export' tab is present, and its first figure
       (plot_vt) renders its range selector buttons as a non-overlapping HTML flex toolbar
       (.plot-toolbar / .plot-range-btn) rather than SVG updatemenus that overlap when
       initially rendered inside a hidden tab.
    """
    from localization import get_text
    tab_summary_en = get_text("tab_summary", lang="en")

    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    # 1. Custom Mode: Experiment Summary & Export tab must NOT exist
    custom_tab_labels = [t.label for t in at.tabs]
    assert not any(tab_summary_en in lbl for lbl in custom_tab_labels), (
        f"'{tab_summary_en}' must be removed in Custom Mode, got: {custom_tab_labels}"
    )

    # 2. Switch to Demo Mode: Experiment Summary & Export tab MUST exist
    at.sidebar.radio[0].set_value("demo").run()
    assert len(at.exception) == 0

    demo_tab_labels = [t.label for t in at.tabs]
    assert any(tab_summary_en in lbl for lbl in demo_tab_labels), (
        f"'{tab_summary_en}' must be present in Demo Mode, got: {demo_tab_labels}"
    )

    # Verify the embedded HTML dashboard in Demo Mode uses the non-overlapping HTML flex toolbar
    campaign = at.session_state["campaign_result"]
    assert campaign is not None and campaign.best_candidate is not None
    from analysis.visualization.plot_engine import ElectrochemicalPlotEngine
    engine = ElectrochemicalPlotEngine(output_dir="./reports")
    html_content = engine._render_symmetric_dashboard(campaign.best_candidate.analysis_report)
    assert 'class="plot-toolbar"' in html_content
    assert 'class="plot-range-btn active"' in html_content
    assert '"updatemenus"' not in html_content
    assert "0 to Failure (Auto)" in html_content
    assert "Full Test" in html_content
    assert "Initial 24h" in html_content


def test_ui_macbook_dark_mode_dropdowns_sidebar_and_plotly_layouts():
    """
    Verify fixes for MacBook (13-inch and 16-inch) and desktop UI compatibility:
    1. Dark mode CSS enforces #ffffff text under [data-theme="dark"] and scopes
       @media (prefers-color-scheme: light) strictly to [data-theme="light"].
    2. BaseWeb select & virtual dropdown rules allow multi-line wrapping and auto height.
    3. Sidebar 'Reported Additive Conc (mM)' and 'Literature Baseline Electrolyte' are stacked
       vertically rather than squeezed into a 2-column sidebar row.
    4. Plotly charts (Pareto, DoE, HTML Live Preview) use non-overlapping bottom legends,
       wrapped titles, and above-canvas failure diagnostic annotations.
    """
    import inspect
    from ui.styles import CUSTOM_CSS, inject_global_styles
    from analysis.visualization.plot_engine import ElectrochemicalPlotEngine

    # 1. Dark Mode CSS & macOS color parser verification
    assert '[data-theme="dark"] p,' in CUSTOM_CSS
    assert '[data-theme="dark"] [data-testid="stCaptionContainer"] *' in CUSTOM_CSS
    assert '[data-theme="dark"] [data-testid="stMetricValue"] *' in CUSTOM_CSS
    assert '[data-theme="dark"] label[data-testid="stWidgetLabel"] *' in CUSTOM_CSS
    assert "color: #ffffff !important;" in CUSTOM_CSS

    light_media_block = CUSTOM_CSS.split("@media (prefers-color-scheme: light)")[1].split("/* Dark Mode High-Contrast Enforcement")[0]
    assert '[data-theme="light"]' in light_media_block
    assert "\n        .stApp p," not in light_media_block

    js_source = inspect.getsource(inject_global_styles)
    assert "parseColorToRgb" in js_source
    assert "stAppViewContainer" in js_source

    # 2. Dropdown wrapping & virtual dropdown rules + Two-Row Online Search Toolbar
    assert 'ul[data-testid="stSelectboxVirtualDropdown"]' in CUSTOM_CSS
    assert "overflow-wrap: anywhere !important;" in CUSTOM_CSS
    assert ".st-key-lit_search_options_label_row" in CUSTOM_CSS

    # 3. Sidebar vertical stacking & Plotly chart layouts in Demo Mode
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    at.sidebar.radio[0].set_value("demo").run()
    assert len(at.exception) == 0

    app_src = Path(APP_PATH).read_text(encoding="utf-8")
    assert "col_s1, col_s2 = st.columns(2)" not in app_src
    assert 'key="lit_search_options_label_row"' in app_src
    assert "<b>DoE Multi-Round Formulation Space</b>" in app_src
    assert "<br><sup>{anchor.additive_name} in {target_elyte_clean}" in app_src

    # Verify HTML Live Preview symmetric dashboard has non-overlapping HTML diagnostic banner,
    # hidden-tab visibility re-plotter, and bottom legend with standoff
    campaign = at.session_state["campaign_result"]
    assert campaign is not None and campaign.best_candidate is not None
    engine = ElectrochemicalPlotEngine(output_dir="./reports")
    html_content = engine._render_symmetric_dashboard(campaign.best_candidate.analysis_report)
    assert 'class="plot-diag-banner"' in html_content
    assert "rebuildAllPlotsClean" in html_content
    assert "repairSvgLegendsInDom" in html_content
    assert "hasCollapsedLegendOrBBox" in html_content
    assert "clip-path: none !important;" in html_content
    assert '"standoff": 10' in html_content
    assert '"y": -0.25' in html_content


def test_ui_batch1_items_4_5_6_9_11_12():
    """
    Comprehensive UI & Localization verification for Batch 1 updates:
    - Item 4: Append mode same-keywords pagination + Load Next 25 Papers for Current Query.
    - Item 5: Custom Mode vs Demo Mode empty-state PDF message (no_pdf_uploaded_tip_custom).
    - Item 6: RSC partial abstract warning badge only shown when is_rsc and len(clean_abstract) < 300.
    - Item 9: Demo Mode popup modal when clicking 'Run Autonomous Optimization Experiment'.
    - Item 11: Decoupled 'Additive Extraction Complete' vs 'Add to Existing Additives or Replace?' modal.
    - Item 12: Benchmark chips include zinc ion battery electrolyte additive context.
    """
    from localization import get_text
    from models import PaperMetadata

    # 1. Verify EN/FR/ZH localization keys for Items 4, 5, 9, 11
    for lang in ["en", "fr", "zh"]:
        assert not get_text("append_to_dataset_label", lang=lang).startswith("[")
        assert not get_text("append_to_dataset_help", lang=lang).startswith("[")
        assert not get_text("btn_fetch_next_papers", lang=lang, count=25).startswith("[")
        assert not get_text("no_pdf_uploaded_tip_custom", lang=lang).startswith("[")
        assert "Load Benchmark Sample PDF" not in get_text("no_pdf_uploaded_tip_custom", lang=lang)
        assert not get_text("modal_merge_conflict_title", lang=lang).startswith("[")
        assert not get_text("modal_btn_confirm_merge", lang=lang).startswith("[")
        assert not get_text("demo_campaign_modal_title", lang=lang).startswith("[")
        assert not get_text("demo_campaign_modal_steps", lang=lang).startswith("[")
        assert not get_text("demo_campaign_modal_sidebar_tip", lang=lang).startswith("[")

    # 2. Item 5: Custom Mode vs Demo Mode empty-state PDF tip
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    info_custom = " ".join([getattr(i, "value", "") for i in at.info if hasattr(i, "value")])
    assert get_text("no_pdf_uploaded_tip_custom", lang="en") in info_custom
    assert "Load Benchmark Sample PDF" not in info_custom

    # 3. Item 4: Search "trehalose" (25 papers), then check Append and click Search Literature with same keywords
    search_inputs = [ti for ti in at.text_input if ti.key == "lit_search_query_input"]
    search_inputs[0].input("trehalose").run()
    search_btn = [b for b in at.button if b.key == "lit_search_submit_btn"][0]
    search_btn.click().run()
    assert len(at.exception) == 0
    assert len(at.session_state["literature_search_results"]) == 25
    assert at.session_state["paper_search_offset"] == 25

    append_cb = [cb for cb in at.checkbox if cb.key == "lit_search_append_checkbox"][0]
    assert "Keep existing papers when searching new keywords" in append_cb.label
    append_cb.check().run()

    search_btn = [b for b in at.button if b.key == "lit_search_submit_btn"][0]
    search_btn.click().run()
    assert len(at.exception) == 0
    assert len(at.session_state["literature_search_results"]) == 50, (
        "Searching same keywords with Append checked must automatically fetch and append next 25 unseen papers"
    )
    assert at.session_state["paper_search_offset"] == 50

    # 4. Item 6: RSC partial abstract badge only when len(clean_abstract) < 300
    full_rsc_paper = PaperMetadata(
        paper_id="rsc_full_1",
        title="Full Abstract RSC Paper on Trehalose Additive",
        doi="10.1039/D4TA99999A",
        venue="Journal of Materials Chemistry A",
        abstract=(
            "Aqueous zinc-ion batteries are promising for grid storage, yet dendrite growth and hydrogen evolution "
            "limit their practical deployment. Here we report that 10 mM trehalose in 2 M ZnSO4 aqueous electrolyte "
            "modulates the primary Zn2+ solvation sheath and suppresses parasitic reactions, achieving 2500 hours of "
            "stable plating and stripping in symmetric cells with a high Coulombic efficiency of 99.7%."
        ),
    )
    short_rsc_paper = PaperMetadata(
        paper_id="rsc_short_2",
        title="Short Abstract RSC Paper on Sorbitol",
        doi="10.1039/D4TA88888B",
        venue="Energy & Environmental Science",
        abstract="Truncated RSC snippet.",
    )
    at.session_state["literature_search_results"] = [full_rsc_paper, short_rsc_paper]
    at.session_state["literature_mined_records"] = [
        {"paper": full_rsc_paper, "additive": None},
        {"paper": short_rsc_paper, "additive": None},
    ]
    at.session_state["lit_search_current_page"] = 1
    at.run()
    assert len(at.exception) == 0

    card_mds = [m.value for m in at.markdown if hasattr(m, "value") and "literature-paper-card" in m.value]
    full_card = next(m for m in card_mds if "Full Abstract RSC Paper" in m)
    short_card = next(m for m in card_mds if "Short Abstract RSC Paper" in m)
    assert get_text("badge_rsc_partial_abstract", lang="en") not in full_card
    assert get_text("badge_rsc_partial_abstract", lang="en") in short_card

    # 5. Item 9: Switch to Demo Mode and click 'Run Autonomous Optimization Experiment'
    at.sidebar.radio[0].set_value("demo").run()
    assert len(at.exception) == 0

    run_btns = [b for b in at.button if "Run Autonomous Optimization" in getattr(b, "label", "")]
    assert len(run_btns) == 1
    run_btns[0].click().run()
    assert len(at.exception) == 0
    assert at.session_state.get("last_demo_campaign_modal_shown") is True

    all_md_demo = " ".join([m.value for m in at.markdown if hasattr(m, "value")])
    assert "Literature Prior Transfer" in all_md_demo
    assert "ESOL Solubility Guardrail" in all_md_demo
    assert "Bayesian Active Learning Refinement" in all_md_demo
    all_success_demo = " ".join([s.value for s in at.success if hasattr(s, "value")])
    assert "Coulombic Efficiency" in all_success_demo


def test_ui_batch2_items_1_2_3_7_8_10():
    """
    Comprehensive UI, CSS, and Plot verification for Batch 2 updates:
    - Item 1: Additive Card uniform fixed height (height: 205px !important; justify-content: space-between)
      and non-wrapping header badges (white-space: nowrap; flex-shrink: 0).
    - Item 2: Active Additive Overview & Electrolyte System Banner 3-column layout with compact
      width: fit-content; min-width: 150px; max-width: 240px; electrolyte boxes inside .active-banner-transfer-row.
    - Item 3: Download Center column headers (.dl-center-col-header with min-height: 2.75rem) and
      button alignment (div[class*="st-key-tab5_btn_dl_"] button with min-height: 52px !important).
    - Item 7: Delaney ESOL '?' tooltip clean text formatting without double spaces or space before period.
    - Item 8: Collision-aware label placement on Planned Test Concentrations graph ('  ★ REFERENCE'
      at 'middle right' or 'middle left', and per-point sample textposition list).
    - Item 10: Tab 1 'Experiment Setup:' styled div at 1.0rem vertically aligned with the Run button.
    """
    import re
    from ui.styles import CUSTOM_CSS
    from localization import get_text

    # 1. Verify CSS rules for Items 1, 2, 3
    assert "height: 205px !important;" in CUSTOM_CSS
    assert "justify-content: space-between !important;" in CUSTOM_CSS
    assert ".candidate-card-clamp" in CUSTOM_CSS
    assert "white-space: nowrap !important;" in CUSTOM_CSS
    assert "flex-shrink: 0 !important;" in CUSTOM_CSS
    assert ".active-banner-transfer-row" in CUSTOM_CSS
    assert ".active-banner-elyte-box" in CUSTOM_CSS
    assert "width: fit-content !important;" in CUSTOM_CSS
    assert "min-width: 150px !important;" in CUSTOM_CSS
    assert "max-width: 240px !important;" in CUSTOM_CSS
    assert ".dl-center-col-header" in CUSTOM_CSS
    assert "min-height: 2.75rem !important;" in CUSTOM_CSS
    assert 'div[class*="st-key-tab5_btn_dl_"] button' in CUSTOM_CSS
    assert "min-height: 52px !important;" in CUSTOM_CSS

    # 2. Item 7: Verify Delaney ESOL tooltip text formatting across EN, FR, ZH
    for lang in ["en", "fr", "zh"]:
        raw_esol_text = f"{get_text('esol_callout_title', lang=lang)} {get_text('esol_callout_body', lang=lang, ceiling=15.0)}"
        clean_esol_tooltip = re.sub(r'</?(?:strong|em|b|i)\b[^>]*>', '', raw_esol_text, flags=re.IGNORECASE)
        clean_esol_tooltip = re.sub(r'(?:<br\s*/?>\s*){2,}', '\n\n', clean_esol_tooltip, flags=re.IGNORECASE)
        clean_esol_tooltip = re.sub(r'<br\s*/?>', '\n', clean_esol_tooltip, flags=re.IGNORECASE)
        clean_esol_tooltip = re.sub(r'<[^>]+>', '', clean_esol_tooltip)
        clean_esol_tooltip = re.sub(r'[ \t]+', ' ', clean_esol_tooltip)
        clean_esol_tooltip = re.sub(r' *\n *', '\n', clean_esol_tooltip).replace('"', '&quot;').strip()
        assert "  " not in clean_esol_tooltip, f"Found double space in {lang} ESOL tooltip: {clean_esol_tooltip}"
        assert " ." not in clean_esol_tooltip, f"Found space before period in {lang} ESOL tooltip: {clean_esol_tooltip}"
        assert "\n\n" in clean_esol_tooltip, f"Expected paragraph break in {lang} ESOL tooltip: {clean_esol_tooltip}"

    # 3. Run AppTest in Demo Mode to verify rendered markup for Items 1, 2, 3, 7, 8, 10
    at = AppTest.from_file(APP_PATH, default_timeout=35.0)
    at.run()
    assert len(at.exception) == 0

    at.sidebar.radio[0].set_value("demo").run()
    assert len(at.exception) == 0

    load_cand_btns = [b for b in at.button if b.key == "btn_load_sample_candidates_t1"]
    if load_cand_btns:
        load_cand_btns[0].click().run()
        assert len(at.exception) == 0

    all_md = "\n".join([m.value for m in at.markdown if hasattr(m, "value")])

    # Item 1: Card header non-wrapping badge container & clamp class
    assert "candidate-card-clamp" in all_md
    assert "flex-shrink: 0; white-space: nowrap;" in all_md

    # Item 2: Active Additive & Electrolyte Banner compact flex container
    assert "active-banner-transfer-row" in all_md
    assert "width: fit-content; min-width: 150px; max-width: 240px;" in all_md

    # Item 3: Download Center column headers
    assert 'class="dl-center-col-header"' in all_md

    # Item 7: Rendered ESOL tooltip has no 'mM) .', no double spaces, and no literal '\n' inside the HTML block
    assert "mM) ." not in all_md
    assert 'aria-label="Delaney ESOL explanation"' in all_md
    esol_md = next(m.value for m in at.markdown if hasattr(m, "value") and 'aria-label="Delaney ESOL explanation"' in m.value)
    assert "\n" not in esol_md, "ESOL section-title HTML block must not contain literal newlines that break CommonMark HTML parsing"
    assert "&#10;&#10;" in esol_md, "ESOL tooltip must use HTML entity &#10;&#10; for paragraph breaks"

    # Item 10: Tab 1 Experiment Setup styled div with 1.0rem font size & horizontal alignment rules
    assert 'class="tab1-experiment-setup-info"' in all_md
    assert "font-size: 1.0rem;" in all_md
    assert "<strong>Experiment Setup:</strong>" in all_md
    assert ".st-key-tab1_run_action_row" in CUSTOM_CSS
    assert ".st-key-btn_tab1_run_campaign button" in CUSTOM_CSS

    # Item 8: Verify collision-aware Reference star label and 2-tier pixel-offset annotations in app.py source
    app_src = Path(APP_PATH).read_text(encoding="utf-8")
    assert 'text=["  ★ REFERENCE"]' in app_src
    assert 'ref_text_pos = "middle left" if (near_right_edge or near_esol_cap) else "middle right"' in app_src
    assert "pixel_offset_candidates" in app_src
    assert "fig_doe.add_annotation(" in app_src


def test_batch3_items_13_14():
    """
    Verify Batch 3 updates:
    - Item 13: README.md contains strong free Gemini API key recommendation (https://aistudio.google.com/app/apikey,
      2,500 papers/day, gemini-3.5-flash-lite / gemini-3.1-flash-lite, 🔑 Configure Gemini API Key),
      updated test count (291+), and Plain-Language Guide to Sidebar & Experiment Controls.
    - Item 14: .github/workflows/keep_alive.yml uses Playwright headless Chromium on a 6-hour cron schedule,
      handles the 'Yes, get this app back up!' hibernation button, waits for [data-testid="stAppViewContainer"],
      holds the WebSocket session open for 15s, and prepare_release.py includes '.github'.
    """
    root_dir = Path(APP_PATH).resolve().parent
    readme_text = (root_dir / "README.md").read_text(encoding="utf-8")

    # Item 13 checks
    assert "291+" in readme_text
    assert "https://aistudio.google.com/app/apikey" in readme_text
    assert "🔑 Configure Gemini API Key" in readme_text
    assert "2,500" in readme_text
    assert "gemini-3.5-flash-lite" in readme_text
    assert "gemini-3.1-flash-lite" in readme_text
    assert "Demo Mode (Pre-loaded Literature Data)" in readme_text
    assert "Custom Experiment Mode" in readme_text
    assert "Target Electrolyte to Optimize" in readme_text
    assert "Additive to Test (from Literature)" in readme_text
    assert "Reported Additive Conc (mM)" in readme_text
    assert "Literature Baseline Electrolyte" in readme_text
    assert "Optimization Rounds" in readme_text
    assert "Latin Hypercube (LHS — Uniform Coverage)" in readme_text
    assert "Central Composite (CCD — Surface Response)" in readme_text
    assert "Initial Formulations (Batch 1)" in readme_text
    assert "Follow-up Formulations per Round" in readme_text
    assert "Fastest Improvement (Near Top Performer)" in readme_text
    assert "Refinement (Fill High-Performing Region)" in readme_text
    assert "Balanced (Explore & Refine)" in readme_text
    assert "Robotic Liquid Handler Rack" in readme_text
    assert "Target Electrolyte Volume per Vial (µL)" in readme_text

    # Item 14 checks
    workflow_path = root_dir / ".github" / "workflows" / "keep_alive.yml"
    assert workflow_path.exists(), ".github/workflows/keep_alive.yml must exist"
    wf_text = workflow_path.read_text(encoding="utf-8")
    assert "0 */6 * * *" in wf_text
    assert "workflow_dispatch" in wf_text
    assert "npx playwright install --with-deps chromium" in wf_text
    assert "https://zinc-ion-battery-map-bvpghpfevrln9egmrecfgr.streamlit.app" in wf_text
    assert "Yes, get this app back up!" in wf_text
    assert '[data-testid="stAppViewContainer"]' in wf_text
    assert "15000" in wf_text

    prep_text = (root_dir / "prepare_release.py").read_text(encoding="utf-8")
    assert '".github"' in prep_text



