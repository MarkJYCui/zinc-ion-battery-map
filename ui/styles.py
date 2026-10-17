"""
Global CSS styling and theme configuration for the ZIB MAP Streamlit Dashboard.
"""

import streamlit as st

CUSTOM_CSS = """
<style>
    /* -------------------------------------------------------------------------
       Theme Tokens & Test Verifications
       ------------------------------------------------------------------------- */
    [data-theme="light"] {
        --text-color: #000000;
        --app-text: #000000;
        color: #000000 !important;
    }
    [data-theme="light"] .sidebar-section-title {
        color: #000000 !important;
    }

    [data-theme="dark"] {
        --text-color: #ffffff;
        --app-text: #ffffff;
        color: #ffffff !important;
    }

    @media (prefers-color-scheme: dark) {
        [data-theme="dark"] {
            --text-color: #ffffff;
            --app-text: #ffffff;
            color: #ffffff !important;
        }
    }

    /* Universal High-Contrast Inheritance: eliminate Streamlit's default 60% grey caption & label dimming */
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] *,
    .stApp small,
    .stApp [data-testid="stMetricLabel"],
    .stApp [data-testid="stMetricLabel"] *,
    .stApp [data-testid="stMetricValue"],
    .stApp [data-testid="stMetricValue"] *,
    .stApp label[data-testid="stWidgetLabel"],
    .stApp label[data-testid="stWidgetLabel"] *,
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] li,
    .stApp [data-testid="stExpander"] summary,
    .stApp [data-testid="stExpander"] summary *,
    .stApp [data-testid="stFileUploaderDropzoneInstructions"] *,
    .stApp [data-testid="stFileUploaderDropzone"] small,
    .stApp [data-testid="stFileUploaderDropzone"] span,
    .stApp [data-baseweb="tab"]:not([aria-selected="true"]) * {
        color: inherit !important;
        opacity: 1 !important;
    }

    /* Light Mode High-Contrast Enforcement: eliminate grey text */
    [data-theme="light"] p,
    [data-theme="light"] span:not(.badge-tag):not([style*="color"]),
    [data-theme="light"] label,
    [data-theme="light"] li,
    [data-theme="light"] small,
    [data-theme="light"] [data-testid="stCaptionContainer"],
    [data-theme="light"] [data-testid="stCaptionContainer"] *,
    [data-theme="light"] [data-testid="stMetricLabel"],
    [data-theme="light"] [data-testid="stMetricLabel"] *,
    [data-theme="light"] [data-testid="stMetricValue"],
    [data-theme="light"] [data-testid="stMetricValue"] *,
    [data-theme="light"] label[data-testid="stWidgetLabel"],
    [data-theme="light"] label[data-testid="stWidgetLabel"] *,
    [data-theme="light"] [data-testid="stMarkdownContainer"] p,
    [data-theme="light"] [data-testid="stMarkdownContainer"] span:not(.badge-tag):not([style*="color"]),
    [data-theme="light"] [data-testid="stExpander"] summary,
    [data-theme="light"] [data-testid="stExpander"] summary *,
    [data-theme="light"] [data-testid="stFileUploaderDropzoneInstructions"] *,
    [data-theme="light"] [data-testid="stFileUploaderDropzone"] small,
    [data-theme="light"] [data-testid="stFileUploaderDropzone"] span,
    [data-theme="light"] [data-baseweb="tab"]:not([aria-selected="true"]) *,
    [data-theme="light"] .tech-file-banner,
    [data-theme="light"] .tech-file-banner .banner-text,
    [data-theme="light"] .tech-file-banner .banner-sep,
    [data-theme="light"] .tech-file-banner .banner-code,
    [data-theme="light"] .hero-card-subtitle,
    [data-theme="light"] .hero-stat-label,
    [data-theme="light"] .hero-stat-sub,
    [data-theme="light"] .hero-sub,
    [data-theme="light"] .paper-card-meta,
    [data-theme="light"] .paper-abstract-box,
    [data-theme="light"] .callout-box *,
    [data-theme="light"] .stage-desc,
    [data-theme="light"] .section-title,
    [data-theme="light"] table,
    [data-theme="light"] th,
    [data-theme="light"] td {
        color: #000000 !important;
        opacity: 1 !important;
    }

    @media (prefers-color-scheme: light) {
        .stApp p,
        .stApp span:not(.badge-tag):not([style*="color"]),
        .stApp label,
        .stApp li,
        .stApp small,
        .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stCaptionContainer"] *,
        .stApp [data-testid="stMetricLabel"],
        .stApp [data-testid="stMetricLabel"] *,
        .stApp [data-testid="stMetricValue"],
        .stApp [data-testid="stMetricValue"] *,
        .stApp label[data-testid="stWidgetLabel"],
        .stApp label[data-testid="stWidgetLabel"] *,
        .stApp [data-testid="stMarkdownContainer"] p,
        .stApp [data-testid="stMarkdownContainer"] span:not(.badge-tag):not([style*="color"]),
        .stApp [data-testid="stExpander"] summary,
        .stApp [data-testid="stExpander"] summary *,
        .stApp [data-testid="stFileUploaderDropzoneInstructions"] *,
        .stApp [data-testid="stFileUploaderDropzone"] small,
        .stApp [data-testid="stFileUploaderDropzone"] span,
        .stApp [data-baseweb="tab"]:not([aria-selected="true"]) *,
        .stApp .tech-file-banner,
        .stApp .tech-file-banner .banner-text,
        .stApp .tech-file-banner .banner-sep,
        .stApp .tech-file-banner .banner-code,
        .stApp .hero-card-subtitle,
        .stApp .hero-stat-label,
        .stApp .hero-stat-sub,
        .stApp .hero-sub,
        .stApp .paper-card-meta,
        .stApp .paper-abstract-box,
        .stApp .callout-box *,
        .stApp .stage-desc,
        .stApp .section-title,
        .stApp table,
        .stApp th,
        .stApp td {
            color: #000000 !important;
            opacity: 1 !important;
        }
    }

    /* -------------------------------------------------------------------------
       Step 3 Tech File Info Banner (Dark & Light Mode Adaptation)
       ------------------------------------------------------------------------- */
    .tech-file-banner {
        background: rgba(2, 132, 199, 0.08);
        border: 1px solid rgba(2, 132, 199, 0.25);
        border-left: 5px solid #0284c7;
        padding: 11px 16px;
        border-radius: 6px;
        margin-bottom: 16px;
        display: flex;
        flex-wrap: wrap;
        gap: 14px;
        align-items: center;
        font-size: 0.92rem;
        color: inherit !important;
    }
    .tech-file-banner .banner-text {
        color: inherit !important;
        opacity: 1 !important;
    }
    .tech-file-banner .banner-sep {
        color: inherit !important;
        opacity: 1 !important;
    }
    .tech-file-banner .banner-code {
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.86rem;
        background: rgba(2, 132, 199, 0.12);
        color: inherit !important;
        opacity: 1 !important;
    }
    .tech-file-banner .banner-metric {
        color: #0284c7;
        font-weight: 700;
    }

    @media (prefers-color-scheme: dark) {
        [data-theme="dark"] .tech-file-banner {
            background: rgba(2, 132, 199, 0.16) !important;
            border-color: rgba(56, 189, 248, 0.35) !important;
            color: #f8fafc !important;
        }
        [data-theme="dark"] .tech-file-banner .banner-text {
            color: #f8fafc !important;
        }
        [data-theme="dark"] .tech-file-banner .banner-code {
            color: #7dd3fc !important;
            background: rgba(56, 189, 248, 0.2) !important;
        }
        [data-theme="dark"] .tech-file-banner .banner-metric {
            color: #38bdf8 !important;
        }
    }

    [data-theme="dark"] .tech-file-banner,
    .stApp[data-theme="dark"] .tech-file-banner,
    .stApp.dark .tech-file-banner {
        background: rgba(2, 132, 199, 0.16) !important;
        border-color: rgba(56, 189, 248, 0.35) !important;
        color: #f8fafc !important;
    }
    [data-theme="dark"] .tech-file-banner .banner-text,
    .stApp[data-theme="dark"] .tech-file-banner .banner-text,
    .stApp.dark .tech-file-banner .banner-text {
        color: #f8fafc !important;
    }
    [data-theme="dark"] .tech-file-banner .banner-code,
    .stApp[data-theme="dark"] .tech-file-banner .banner-code,
    .stApp.dark .tech-file-banner .banner-code {
        color: #7dd3fc !important;
        background: rgba(56, 189, 248, 0.2) !important;
    }
    [data-theme="dark"] .tech-file-banner .banner-metric,
    .stApp[data-theme="dark"] .tech-file-banner .banner-metric,
    .stApp.dark .tech-file-banner .banner-metric {
        color: #38bdf8 !important;
    }

    /* Step 3 Cycler File Uploader Compact 42px Layout */
    .st-key-uploader_step3_cycler_file [data-testid="stFileUploaderDropzone"] {
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
    .st-key-uploader_step3_cycler_file [data-testid="stFileUploaderDropzone"] button {
        min-height: 32px !important;
        height: 32px !important;
        padding: 2px 10px !important;
        font-size: 0.85rem !important;
        line-height: 1.1 !important;
    }
    .st-key-uploader_step3_cycler_file [data-testid="stFileUploaderDropzoneInstructions"] {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        gap: 6px !important;
        font-size: 0.82rem !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    /* BaseWeb Select & Popover: Full Text Visibility without Ellipsis */
    div[data-baseweb="select"] {
        width: 100% !important;
    }
    div[data-baseweb="select"] > div {
        min-height: 38px !important;
        height: auto !important;
    }
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div {
        white-space: normal !important;
        text-overflow: clip !important;
        overflow: visible !important;
        word-break: break-word !important;
        line-height: 1.35 !important;
    }
    div[data-baseweb="popover"] {
        max-width: 95vw !important;
        width: auto !important;
        min-width: 100% !important;
    }
    div[data-baseweb="popover"] ul[role="listbox"] {
        max-height: 360px !important;
    }
    div[data-baseweb="popover"] li[role="option"] {
        white-space: normal !important;
        word-break: break-word !important;
        height: auto !important;
        min-height: 38px !important;
        padding-top: 8px !important;
        padding-bottom: 8px !important;
        line-height: 1.35 !important;
    }
    div[data-baseweb="popover"] li[role="option"] > div {
        white-space: normal !important;
        word-break: break-word !important;
    }

    /* Universal Adaptive Metric Card styling */
    .metric-card {
        background: rgba(128, 128, 128, 0.08) !important;
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
        color: inherit !important;
    }
    .metric-card-hero {
        background: rgba(128, 128, 128, 0.08) !important;
        border: 1.5px solid rgba(128, 128, 128, 0.25);
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 20px;
        color: inherit !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    }

    /* Hero Card Stat Typography */
    .hero-card-title {
        margin: 0;
        color: inherit !important;
        font-weight: 700;
        font-size: 1.35rem;
    }
    .hero-card-subtitle {
        font-size: 0.88rem;
        color: inherit !important;
        margin-top: 4px;
    }
    .hero-stat-label {
        font-size: 0.85rem;
        color: inherit !important;
        font-weight: 700;
        letter-spacing: 0.4px;
    }
    .hero-stat-sub {
        font-size: 0.80rem;
        color: inherit !important;
        margin-top: 2px;
    }
    .hero-stat-val-teal {
        font-size: 1.5rem;
        font-weight: 700;
        color: #00A896 !important;
    }
    .hero-stat-val-blue {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1E90FF !important;
    }
    .hero-stat-val-green {
        font-size: 1.5rem;
        font-weight: 700;
        color: #10B981 !important;
    }
    .hero-stat-val-amber {
        font-size: 1.5rem;
        font-weight: 700;
        color: #F59E0B !important;
    }

    /* Badges */
    .badge-tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 14px;
        font-size: 0.80rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 4px;
        color: inherit !important;
    }
    .badge-teal { background-color: rgba(0, 168, 150, 0.18) !important; color: #00A896 !important; border: 1px solid rgba(0, 168, 150, 0.45) !important; }
    .badge-blue { background-color: rgba(30, 144, 255, 0.18) !important; color: #1E90FF !important; border: 1px solid rgba(30, 144, 255, 0.45) !important; }
    .badge-amber { background-color: rgba(245, 158, 11, 0.18) !important; color: #D97706 !important; border: 1px solid rgba(245, 158, 11, 0.45) !important; }
    .badge-green { background-color: rgba(16, 185, 129, 0.18) !important; color: #059669 !important; border: 1px solid rgba(16, 185, 129, 0.45) !important; }
    .badge-red { background-color: rgba(239, 68, 68, 0.18) !important; color: #DC2626 !important; border: 1px solid rgba(239, 68, 68, 0.45) !important; }
    .badge-gold { background-color: rgba(245, 158, 11, 0.18) !important; color: #D97706 !important; border: 1px solid rgba(245, 158, 11, 0.5) !important; font-weight: 700; }
    .badge-purple { background-color: rgba(139, 92, 246, 0.18) !important; color: #7C3AED !important; border: 1px solid rgba(139, 92, 246, 0.45) !important; }

    /* Custom Header Subtitle & Section Titles */
    .hero-sub {
        font-size: 1.05rem;
        color: inherit !important;
        margin-top: -10px;
        margin-bottom: 18px;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: inherit !important;
        border-bottom: 2px solid rgba(128, 128, 128, 0.25);
        padding-bottom: 6px;
        margin-top: 18px;
        margin-bottom: 14px;
    }

    /* Google Scholar-Style Literature Paper Cards */
    .literature-paper-card {
        background: rgba(128, 128, 128, 0.07) !important;
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        color: inherit !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .literature-paper-card:hover {
        border-color: rgba(0, 168, 150, 0.45);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    .paper-card-title {
        font-size: 1.12rem;
        font-weight: 700;
        line-height: 1.4;
        margin-bottom: 6px;
        color: inherit !important;
    }
    .paper-card-meta {
        font-size: 0.88rem;
        color: inherit !important;
        margin-bottom: 8px;
        line-height: 1.5;
    }
    .paper-abstract-box {
        font-size: 0.90rem;
        line-height: 1.55;
        padding: 12px 14px;
        border-radius: 8px;
        background: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.16);
        color: inherit !important;
        margin-top: 6px;
        margin-bottom: 12px;
    }
    .paper-analysis-box {
        font-size: 0.88rem;
        line-height: 1.5;
        padding: 14px 18px;
        border-radius: 8px;
        background: rgba(0, 168, 150, 0.06);
        border: 1px solid rgba(0, 168, 150, 0.3);
        color: inherit !important;
        margin-top: 10px;
        margin-bottom: 14px;
    }

    /* Callout Box */
    .callout-box {
        background-color: rgba(128, 128, 128, 0.08) !important;
        border-left: 4px solid currentColor;
        color: inherit !important;
        padding: 12px 18px;
        border-radius: 6px;
        margin: 10px 0 16px 0;
        font-size: 0.95rem;
        line-height: 1.5;
        border-top: 1px solid rgba(128, 128, 128, 0.15);
        border-right: 1px solid rgba(128, 128, 128, 0.15);
        border-bottom: 1px solid rgba(128, 128, 128, 0.15);
    }
    .callout-box strong, .callout-box b {
        color: inherit !important;
        font-weight: 700;
    }
    .callout-box em {
        color: inherit !important;
    }

    /* Stage 1 & 2 Cards (Electrochemical Analysis) */
    .stage1-card, .stage2-card {
        border-left: 5px solid currentColor;
        background: rgba(128, 128, 128, 0.08) !important;
        border-radius: 8px;
        padding: 14px 18px;
        color: inherit !important;
    }
    .stage1-title, .stage2-title {
        margin-top: 0;
        color: inherit !important;
        font-weight: 700;
    }
    .stage-desc {
        font-size: 0.85rem;
        color: inherit !important;
    }

    /* Sidebar Logo Container & SVGs (Uses currentColor to adapt natively to both Light and Dark mode) */
    .sidebar-logo-container {
        width: 100%;
        margin-bottom: 12px;
        color: inherit !important;
    }
    .sidebar-logo-container svg {
        width: 100%;
        height: auto;
        display: block;
        color: inherit !important;
    }
    .sidebar-logo-container text,
    .sidebar-logo-container tspan,
    .sidebar-logo-container .logo-title-zib,
    .sidebar-logo-container .logo-title-map,
    .sidebar-logo-container .logo-badge-txt,
    .sidebar-logo-container .logo-sub-txt {
        fill: currentColor !important;
        color: inherit !important;
    }

    /* Sidebar Navigation, Repositories & Tech Stack Cards */
    .sidebar-section-title {
        font-size: 0.96rem;
        font-weight: 700;
        margin-top: 18px;
        margin-bottom: 8px;
        color: inherit !important;
        letter-spacing: 0.2px;
    }
    .sidebar-link-card {
        display: flex;
        align-items: center;
        padding: 8px 12px;
        margin-bottom: 8px;
        border-radius: 8px;
        background: rgba(128, 128, 128, 0.08) !important;
        border: 1px solid rgba(128, 128, 128, 0.2);
        text-decoration: none;
        color: inherit !important;
        font-size: 0.88rem;
        font-weight: 500;
        transition: all 0.2s ease-in-out;
    }
    .sidebar-link-card:hover {
        background: rgba(128, 128, 128, 0.15) !important;
        border-color: currentColor;
        transform: translateY(-1px);
    }
    .sidebar-credits-box {
        background: rgba(128, 128, 128, 0.06) !important;
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 12px 14px;
        font-size: 0.86rem;
        margin-bottom: 12px;
        color: inherit !important;
    }
    .sidebar-credits-box strong,
    .sidebar-credits-box div {
        color: inherit !important;
    }
    .tech-icon-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 6px;
        margin-top: 8px;
    }
    .tech-pill {
        display: flex;
        align-items: center;
        background: rgba(128, 128, 128, 0.08) !important;
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 5px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        color: inherit !important;
    }

    /* GitHub Octocat Icon (Uses currentColor to adapt natively to both Light and Dark mode) */
    .github-icon-svg {
        fill: currentColor !important;
        min-width: 20px;
        margin-right: 10px;
        transition: fill 0.2s ease;
    }

    /* Step 1: Batch PDF & Online Search Action Row Equal Height & Centering */
    .st-key-btn_process_batch_pdfs button,
    .st-key-btn_clear_uploaded_pdfs button,
    .st-key-batch_pdf_gemini_popover button,
    .st-key-batch_pdf_gemini_popover [data-testid="stPopoverButton"],
    .st-key-online_search_gemini_popover button,
    .st-key-online_search_gemini_popover [data-testid="stPopoverButton"],
    div[data-testid="stColumn"]:has(.st-key-online_search_gemini_popover) button,
    div[data-testid="stColumn"]:has(.st-key-online_search_gemini_popover) [data-testid="stPopoverButton"],
    div[data-testid="stColumn"]:has(.st-key-lit_search_submit_btn) button,
    .st-key-lit_search_submit_btn button {
        height: 48px !important;
        min-height: 48px !important;
        max-height: 48px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        line-height: 1.1 !important;
        box-sizing: border-box !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    .st-key-online_search_gemini_popover [data-testid="stPopover"],
    .st-key-batch_pdf_gemini_popover [data-testid="stPopover"] {
        height: 48px !important;
        min-height: 48px !important;
        max-height: 48px !important;
    }
    .st-key-btn_process_batch_pdfs button div[data-testid="stMarkdownContainer"],
    .st-key-btn_clear_uploaded_pdfs button div[data-testid="stMarkdownContainer"],
    .st-key-batch_pdf_gemini_popover button div[data-testid="stMarkdownContainer"],
    .st-key-online_search_gemini_popover button div[data-testid="stMarkdownContainer"],
    .st-key-online_search_gemini_popover [data-testid="stPopoverButton"] div,
    .st-key-lit_search_submit_btn button div[data-testid="stMarkdownContainer"],
    .st-key-btn_process_batch_pdfs button p,
    .st-key-btn_clear_uploaded_pdfs button p,
    .st-key-batch_pdf_gemini_popover button p,
    .st-key-online_search_gemini_popover button p,
    .st-key-lit_search_submit_btn button p {
        margin: 0 !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    div.st-key-batch_pdf_gemini_popover,
    div.st-key-online_search_gemini_popover,
    div.st-key-lit_search_submit_btn,
    div.st-key-btn_process_batch_pdfs,
    div.st-key-btn_clear_uploaded_pdfs {
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
    }

    /* Step 1 Online Search: Equal 48px Height and matching width for all action bar controls */
    .st-key-lit_search_query_input div[data-baseweb="input"],
    .st-key-lit_search_query_input div[data-baseweb="base-input"] {
        height: 48px !important;
        min-height: 48px !important;
        max-height: 48px !important;
        box-sizing: border-box !important;
    }
    .st-key-lit_search_query_input input {
        height: 48px !important;
        min-height: 48px !important;
        max-height: 48px !important;
        line-height: 48px !important;
        box-sizing: border-box !important;
    }
    .st-key-lit_search_sort_select div[data-baseweb="select"],
    .st-key-lit_search_sort_select div[data-baseweb="select"] > div,
    .st-key-lit_search_limit_select div[data-baseweb="select"],
    .st-key-lit_search_limit_select div[data-baseweb="select"] > div {
        height: 48px !important;
        min-height: 48px !important;
        max-height: 48px !important;
        display: flex !important;
        align-items: center !important;
        box-sizing: border-box !important;
    }
    .st-key-lit_search_sort_select div[data-baseweb="select"] span,
    .st-key-lit_search_limit_select div[data-baseweb="select"] span {
        font-size: 0.88rem !important;
        line-height: 1.25 !important;
    }

    /* Step 1 columns: allow custom width ratios while preventing content overflow */
    div[data-testid="stColumn"]:has(.st-key-lit_search_query_input),
    div[data-testid="stColumn"]:has(.st-key-lit_search_sort_select),
    div[data-testid="stColumn"]:has(.st-key-lit_search_limit_select),
    div[data-testid="stColumn"]:has(.st-key-online_search_gemini_popover),
    div[data-testid="stColumn"]:has(.st-key-lit_search_submit_btn) {
        min-width: 0 !important;
    }
    /* Step 1 Online Search: Label row perfect horizontal alignment */
    .st-key-lit_search_label_row {
        margin-bottom: -10px !important;
    }
    .st-key-lit_search_label_row div[data-testid="stColumn"] {
        min-width: 0 !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        height: 24px !important;
        min-height: 24px !important;
        max-height: 24px !important;
    }
    .st-key-lit_search_label_row .lit-search-label {
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        line-height: 24px !important;
        height: 24px !important;
        display: flex !important;
        align-items: center !important;
        margin: 0 !important;
        padding: 0 !important;
        white-space: nowrap !important;
    }
    .st-key-lit_search_label_row .st-key-lit_search_append_checkbox,
    .st-key-lit_search_label_row .st-key-lit_search_append_checkbox div[data-testid="stCheckbox"],
    .st-key-lit_search_label_row div[data-testid="stCheckbox"] {
        min-height: 24px !important;
        height: 24px !important;
        max-height: 24px !important;
        margin: 0 !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
    }
    .st-key-lit_search_label_row .st-key-lit_search_append_checkbox label,
    .st-key-lit_search_label_row .st-key-lit_search_append_checkbox label[data-testid="stWidgetLabel"] {
        min-height: 24px !important;
        height: 24px !important;
        max-height: 24px !important;
        margin: 0 !important;
        padding: 0 !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        white-space: nowrap !important;
    }
    .st-key-lit_search_label_row .st-key-lit_search_append_checkbox label div[data-testid="stMarkdownContainer"],
    .st-key-lit_search_label_row .st-key-lit_search_append_checkbox label p {
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        line-height: 24px !important;
        height: 24px !important;
        margin: 0 !important;
        padding: 0 !important;
        display: inline-flex !important;
        align-items: center !important;
    }
    .st-key-lit_search_label_row .st-key-lit_search_append_checkbox label > div:first-child,
    .st-key-lit_search_label_row .st-key-lit_search_append_checkbox label svg {
        margin-top: 0 !important;
    }

    /* Step 3: Benchmark & template dataset action alignment */
    .st-key-btn_load_benchmark_step3 button,
    .st-key-btn_load_sample_step3_demo button,
    .st-key-step3_btn_dl_mpt_template button,
    .st-key-step3_btn_dl_landt_csv_template button,
    .st-key-step3_btn_dl_arbin_xlsx_template button {
        height: 42px !important;
        min-height: 42px !important;
    }
    .st-key-benchmark_scenario_step3 div[role="radiogroup"] {
        display: flex;
        align-items: center;
        gap: 16px;
    }

    /* Hidden utility container */
    iframe[height="0"] {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        border: none !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Step 2 Candidate Card Deck: Equal 165px Height and Button Alignment */
    .candidate-card-deck-box {
        height: 165px !important;
        min-height: 165px !important;
        max-height: 165px !important;
        overflow-y: auto !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
        box-sizing: border-box !important;
        padding: 10px 12px !important;
        margin-bottom: 8px !important;
    }
    div[class*="st-key-btn_formulate_t1_"] button,
    div[class*="st-key-btn_formulate_t2_"] button {
        height: 42px !important;
        min-height: 42px !important;
        max-height: 42px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
</style>
"""


def inject_global_styles() -> None:
    """Inject global CSS styles into the active Streamlit page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.components.v1.html(
        """
        <script>
        (function() {
            try {
                const parentDoc = window.parent.document;
                function syncThemeAttribute() {
                    const appEl = parentDoc.querySelector('.stApp');
                    if (!appEl) return;
                    const bg = window.parent.getComputedStyle(appEl).backgroundColor || '';
                    const m = bg.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
                    let isDark = false;
                    if (m) {
                        const r = parseInt(m[1], 10);
                        const g = parseInt(m[2], 10);
                        const b = parseInt(m[3], 10);
                        const lum = 0.299 * r + 0.587 * g + 0.114 * b;
                        isDark = lum < 128;
                    }
                    const themeVal = isDark ? 'dark' : 'light';
                    if (appEl.getAttribute('data-theme') !== themeVal) {
                        appEl.setAttribute('data-theme', themeVal);
                    }
                    if (parentDoc.documentElement.getAttribute('data-theme') !== themeVal) {
                        parentDoc.documentElement.setAttribute('data-theme', themeVal);
                    }
                    if (parentDoc.body && parentDoc.body.getAttribute('data-theme') !== themeVal) {
                        parentDoc.body.setAttribute('data-theme', themeVal);
                    }
                }
                syncThemeAttribute();
                const appEl = parentDoc.querySelector('.stApp');
                if (appEl && !window.parent.__mapThemeObserverAttached) {
                    window.parent.__mapThemeObserverAttached = true;
                    const obs = new MutationObserver(syncThemeAttribute);
                    obs.observe(appEl, { attributes: true, attributeFilter: ['class', 'style'] });
                    obs.observe(parentDoc.documentElement, { attributes: true, attributeFilter: ['class', 'style'] });
                    window.parent.setInterval(syncThemeAttribute, 500);
                }
            } catch(e) {}
        })();
        </script>
        """,
        height=0,
        width=0,
    )


