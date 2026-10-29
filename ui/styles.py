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
    [data-theme="dark"] .sidebar-section-title {
        color: #ffffff !important;
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
    .stApp [data-baseweb="tab"]:not([aria-selected="true"]) *,
    .stApp .tech-file-banner .banner-text {
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
    [data-theme="light"] .hero-card-title,
    [data-theme="light"] .hero-card-subtitle,
    [data-theme="light"] .hero-stat-label,
    [data-theme="light"] .hero-stat-sub,
    [data-theme="light"] .hero-sub,
    [data-theme="light"] .paper-card-title,
    [data-theme="light"] .paper-card-meta,
    [data-theme="light"] .paper-abstract-box,
    [data-theme="light"] .paper-analysis-box,
    [data-theme="light"] .callout-box *,
    [data-theme="light"] .stage1-title,
    [data-theme="light"] .stage2-title,
    [data-theme="light"] .stage-desc,
    [data-theme="light"] .section-title,
    [data-theme="light"] .sidebar-link-card,
    [data-theme="light"] .sidebar-credits-box,
    [data-theme="light"] .sidebar-credits-box *,
    [data-theme="light"] .tech-pill,
    [data-theme="light"] div[data-baseweb="select"] span,
    [data-theme="light"] div[data-baseweb="select"] div,
    [data-theme="light"] div[data-baseweb="input"] input,
    [data-theme="light"] div[data-baseweb="base-input"] input,
    [data-theme="light"] div[data-baseweb="popover"] li[role="option"],
    [data-theme="light"] div[data-baseweb="popover"] li[role="option"] *,
    [data-theme="light"] table,
    [data-theme="light"] th,
    [data-theme="light"] td {
        color: #000000 !important;
        opacity: 1 !important;
    }

    @media (prefers-color-scheme: light) {
        [data-theme="light"].stApp p,
        [data-theme="light"] .stApp p,
        [data-theme="light"].stApp span:not(.badge-tag):not([style*="color"]),
        [data-theme="light"] .stApp span:not(.badge-tag):not([style*="color"]),
        [data-theme="light"].stApp label,
        [data-theme="light"] .stApp label,
        [data-theme="light"].stApp li,
        [data-theme="light"] .stApp li,
        [data-theme="light"].stApp small,
        [data-theme="light"] .stApp small,
        [data-theme="light"] .stApp [data-testid="stCaptionContainer"],
        [data-theme="light"] .stApp [data-testid="stCaptionContainer"] *,
        [data-theme="light"] .stApp [data-testid="stMetricLabel"],
        [data-theme="light"] .stApp [data-testid="stMetricLabel"] *,
        [data-theme="light"] .stApp [data-testid="stMetricValue"],
        [data-theme="light"] .stApp [data-testid="stMetricValue"] *,
        [data-theme="light"] .stApp label[data-testid="stWidgetLabel"],
        [data-theme="light"] .stApp label[data-testid="stWidgetLabel"] *,
        [data-theme="light"] .stApp [data-testid="stMarkdownContainer"] p,
        [data-theme="light"] .stApp [data-testid="stMarkdownContainer"] span:not(.badge-tag):not([style*="color"]),
        [data-theme="light"] .stApp [data-testid="stExpander"] summary,
        [data-theme="light"] .stApp [data-testid="stExpander"] summary *,
        [data-theme="light"] .stApp [data-testid="stFileUploaderDropzoneInstructions"] *,
        [data-theme="light"] .stApp [data-testid="stFileUploaderDropzone"] small,
        [data-theme="light"] .stApp [data-testid="stFileUploaderDropzone"] span,
        [data-theme="light"] .stApp [data-baseweb="tab"]:not([aria-selected="true"]) *,
        [data-theme="light"] .stApp .tech-file-banner,
        [data-theme="light"] .stApp .tech-file-banner .banner-text,
        [data-theme="light"] .stApp .tech-file-banner .banner-sep,
        [data-theme="light"] .stApp .tech-file-banner .banner-code,
        [data-theme="light"] .stApp .hero-card-subtitle,
        [data-theme="light"] .stApp .hero-stat-label,
        [data-theme="light"] .stApp .hero-stat-sub,
        [data-theme="light"] .stApp .hero-sub,
        [data-theme="light"] .stApp .paper-card-meta,
        [data-theme="light"] .stApp .paper-abstract-box,
        [data-theme="light"] .stApp .callout-box *,
        [data-theme="light"] .stApp .stage-desc,
        [data-theme="light"] .stApp .section-title,
        [data-theme="light"] .stApp table,
        [data-theme="light"] .stApp th,
        [data-theme="light"] .stApp td {
            color: #000000 !important;
            opacity: 1 !important;
        }
    }

    /* Dark Mode High-Contrast Enforcement: ensure crisp white text on macOS & desktop */
    [data-theme="dark"] p,
    [data-theme="dark"] span:not(.badge-tag):not([style*="color"]),
    [data-theme="dark"] label,
    [data-theme="dark"] li,
    [data-theme="dark"] small,
    [data-theme="dark"] [data-testid="stCaptionContainer"],
    [data-theme="dark"] [data-testid="stCaptionContainer"] *,
    [data-theme="dark"] [data-testid="stMetricLabel"],
    [data-theme="dark"] [data-testid="stMetricLabel"] *,
    [data-theme="dark"] [data-testid="stMetricValue"],
    [data-theme="dark"] [data-testid="stMetricValue"] *,
    [data-theme="dark"] label[data-testid="stWidgetLabel"],
    [data-theme="dark"] label[data-testid="stWidgetLabel"] *,
    [data-theme="dark"] [data-testid="stMarkdownContainer"] p,
    [data-theme="dark"] [data-testid="stMarkdownContainer"] li,
    [data-theme="dark"] [data-testid="stMarkdownContainer"] span:not(.badge-tag):not([style*="color"]),
    [data-theme="dark"] [data-testid="stExpander"] summary,
    [data-theme="dark"] [data-testid="stExpander"] summary *,
    [data-theme="dark"] [data-testid="stFileUploaderDropzoneInstructions"] *,
    [data-theme="dark"] [data-testid="stFileUploaderDropzone"] small,
    [data-theme="dark"] [data-testid="stFileUploaderDropzone"] span,
    [data-theme="dark"] [data-baseweb="tab"]:not([aria-selected="true"]) *,
    [data-theme="dark"] .hero-card-title,
    [data-theme="dark"] .hero-card-subtitle,
    [data-theme="dark"] .hero-stat-label,
    [data-theme="dark"] .hero-stat-sub,
    [data-theme="dark"] .hero-sub,
    [data-theme="dark"] .paper-card-title,
    [data-theme="dark"] .paper-card-meta,
    [data-theme="dark"] .paper-abstract-box,
    [data-theme="dark"] .paper-analysis-box,
    [data-theme="dark"] .callout-box,
    [data-theme="dark"] .callout-box *,
    [data-theme="dark"] .stage1-card,
    [data-theme="dark"] .stage2-card,
    [data-theme="dark"] .stage1-title,
    [data-theme="dark"] .stage2-title,
    [data-theme="dark"] .stage-desc,
    [data-theme="dark"] .section-title,
    [data-theme="dark"] .sidebar-section-title,
    [data-theme="dark"] .sidebar-link-card,
    [data-theme="dark"] .sidebar-credits-box,
    [data-theme="dark"] .sidebar-credits-box *,
    [data-theme="dark"] .tech-pill,
    [data-theme="dark"] .lit-search-label,
    [data-theme="dark"] div[data-baseweb="select"] span,
    [data-theme="dark"] div[data-baseweb="select"] div,
    [data-theme="dark"] div[data-baseweb="input"] input,
    [data-theme="dark"] div[data-baseweb="base-input"] input,
    [data-theme="dark"] div[data-baseweb="popover"] li[role="option"],
    [data-theme="dark"] div[data-baseweb="popover"] li[role="option"] *,
    [data-theme="dark"] table,
    [data-theme="dark"] th,
    [data-theme="dark"] td {
        color: #ffffff !important;
        opacity: 1 !important;
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
        [data-theme="dark"] {
            --text-color: #ffffff;
            --app-text: #ffffff;
            color: #ffffff !important;
        }
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
        height: auto !important;
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

    /* BaseWeb Select & Popover: Full Text Visibility without Ellipsis on Small & Large Screens */
    div[data-baseweb="select"] {
        width: 100% !important;
    }
    div[data-baseweb="select"] > div {
        min-height: 38px !important;
        height: auto !important;
        max-height: none !important;
        padding-top: 3px !important;
        padding-bottom: 3px !important;
    }
    div[data-baseweb="select"] > div > div {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        flex-wrap: wrap !important;
    }
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div {
        white-space: normal !important;
        text-overflow: clip !important;
        overflow: visible !important;
        word-break: break-word !important;
        overflow-wrap: anywhere !important;
        line-height: 1.35 !important;
    }
    div[data-baseweb="popover"] {
        max-width: 95vw !important;
        width: max-content !important;
        min-width: 240px !important;
    }
    div[data-baseweb="popover"] > div,
    div[data-baseweb="popover"] [data-baseweb="menu"],
    div[data-baseweb="popover"] ul[role="listbox"],
    ul[data-testid="stSelectboxVirtualDropdown"] {
        max-height: 360px !important;
        max-width: min(92vw, 540px) !important;
        width: 100% !important;
    }
    div[data-baseweb="popover"] li[role="option"],
    ul[data-testid="stSelectboxVirtualDropdown"] li[role="option"] {
        white-space: normal !important;
        word-break: break-word !important;
        overflow-wrap: anywhere !important;
        height: auto !important;
        min-height: 38px !important;
        padding-top: 8px !important;
        padding-bottom: 8px !important;
        line-height: 1.35 !important;
        align-items: flex-start !important;
    }
    div[data-baseweb="popover"] li[role="option"] > div,
    div[data-baseweb="popover"] li[role="option"] *,
    ul[data-testid="stSelectboxVirtualDropdown"] li[role="option"] * {
        white-space: normal !important;
        word-break: break-word !important;
        overflow-wrap: anywhere !important;
        text-overflow: clip !important;
        overflow: visible !important;
        line-height: 1.35 !important;
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
        white-space: nowrap !important;
        flex-shrink: 0 !important;
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

    /* Hidden reactive tab-sync trigger buttons for Step 2 Two-Tier Card Deck */
    .st-key-btn_sync_tier1_active_tab,
    .st-key-btn_sync_tier2_active_tab,
    div[data-testid="stElementContainer"]:has(.st-key-btn_sync_tier1_active_tab),
    div[data-testid="stElementContainer"]:has(.st-key-btn_sync_tier2_active_tab) {
        position: absolute !important;
        width: 0 !important;
        height: 0 !important;
        min-height: 0 !important;
        overflow: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
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

    /* Step 1: Batch PDF & Online Search Action Row Equal Height & Centering (with wrap support on 13" MacBook) */
    .st-key-btn_process_batch_pdfs button,
    .st-key-btn_clear_uploaded_pdfs button,
    .st-key-batch_pdf_gemini_popover button,
    .st-key-batch_pdf_gemini_popover [data-testid="stPopoverButton"],
    div[class*="st-key-batch_pdf_gemini_popover"] button,
    div[class*="st-key-batch_pdf_gemini_popover"] [data-testid="stPopoverButton"],
    .st-key-online_search_gemini_popover button,
    .st-key-online_search_gemini_popover [data-testid="stPopoverButton"],
    div[class*="st-key-online_search_gemini_popover"] button,
    div[class*="st-key-online_search_gemini_popover"] [data-testid="stPopoverButton"],
    div[data-testid="stColumn"]:has(.st-key-online_search_gemini_popover) button,
    div[data-testid="stColumn"]:has(.st-key-online_search_gemini_popover) [data-testid="stPopoverButton"],
    div[data-testid="stColumn"]:has(div[class*="st-key-online_search_gemini_popover"]) button,
    div[data-testid="stColumn"]:has(div[class*="st-key-online_search_gemini_popover"]) [data-testid="stPopoverButton"],
    div[data-testid="stColumn"]:has(.st-key-lit_search_submit_btn) button,
    .st-key-lit_search_submit_btn button {
        height: auto !important;
        min-height: 48px !important;
        max-height: none !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        line-height: 1.2 !important;
        padding: 4px 10px !important;
        box-sizing: border-box !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }
    .st-key-online_search_gemini_popover [data-testid="stPopover"],
    .st-key-batch_pdf_gemini_popover [data-testid="stPopover"],
    div[class*="st-key-online_search_gemini_popover"] [data-testid="stPopover"],
    div[class*="st-key-batch_pdf_gemini_popover"] [data-testid="stPopover"] {
        height: auto !important;
        min-height: 48px !important;
        max-height: none !important;
    }
    .st-key-btn_process_batch_pdfs button div[data-testid="stMarkdownContainer"],
    .st-key-btn_clear_uploaded_pdfs button div[data-testid="stMarkdownContainer"],
    .st-key-batch_pdf_gemini_popover button div[data-testid="stMarkdownContainer"],
    div[class*="st-key-batch_pdf_gemini_popover"] button div[data-testid="stMarkdownContainer"],
    .st-key-online_search_gemini_popover button div[data-testid="stMarkdownContainer"],
    div[class*="st-key-online_search_gemini_popover"] button div[data-testid="stMarkdownContainer"],
    .st-key-online_search_gemini_popover [data-testid="stPopoverButton"] div,
    div[class*="st-key-online_search_gemini_popover"] [data-testid="stPopoverButton"] div,
    .st-key-lit_search_submit_btn button div[data-testid="stMarkdownContainer"],
    .st-key-btn_process_batch_pdfs button p,
    .st-key-btn_clear_uploaded_pdfs button p,
    .st-key-batch_pdf_gemini_popover button p,
    div[class*="st-key-batch_pdf_gemini_popover"] button p,
    .st-key-online_search_gemini_popover button p,
    div[class*="st-key-online_search_gemini_popover"] button p,
    .st-key-lit_search_submit_btn button p {
        margin: 0 !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        line-height: 1.2 !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }
    div.st-key-batch_pdf_gemini_popover,
    div[class*="st-key-batch_pdf_gemini_popover"],
    div.st-key-online_search_gemini_popover,
    div[class*="st-key-online_search_gemini_popover"],
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
        min-height: 48px !important;
        height: auto !important;
        max-height: none !important;
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
    /* Step 1 Online Search: Label row horizontal alignment with small-screen wrapping support (Row 1 & Row 2) */
    .st-key-lit_search_label_row {
        margin-bottom: -10px !important;
    }
    .st-key-lit_search_options_label_row {
        margin-top: 6px !important;
        margin-bottom: -10px !important;
    }
    .st-key-lit_search_label_row div[data-testid="stColumn"],
    .st-key-lit_search_options_label_row div[data-testid="stColumn"] {
        min-width: 0 !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        min-height: 24px !important;
        height: auto !important;
    }
    .st-key-lit_search_label_row .lit-search-label,
    .st-key-lit_search_options_label_row .lit-search-label {
        font-size: 0.86rem !important;
        font-weight: 600 !important;
        line-height: 1.25 !important;
        min-height: 24px !important;
        height: auto !important;
        display: flex !important;
        align-items: center !important;
        margin: 0 !important;
        padding: 0 !important;
        white-space: normal !important;
        overflow-wrap: break-word !important;
    }
    .st-key-lit_search_label_row .st-key-lit_search_append_checkbox,
    .st-key-lit_search_label_row .st-key-lit_search_append_checkbox div[data-testid="stCheckbox"],
    .st-key-lit_search_label_row div[data-testid="stCheckbox"] {
        min-height: 24px !important;
        height: auto !important;
        margin: 0 !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
    }
    .st-key-lit_search_label_row .st-key-lit_search_append_checkbox label,
    .st-key-lit_search_label_row .st-key-lit_search_append_checkbox label[data-testid="stWidgetLabel"] {
        min-height: 24px !important;
        height: auto !important;
        margin: 0 !important;
        padding: 0 !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        white-space: normal !important;
    }
    .st-key-lit_search_label_row .st-key-lit_search_append_checkbox label div[data-testid="stMarkdownContainer"],
    .st-key-lit_search_label_row .st-key-lit_search_append_checkbox label p {
        font-size: 0.86rem !important;
        font-weight: 600 !important;
        line-height: 1.25 !important;
        min-height: 24px !important;
        height: auto !important;
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
        position: absolute !important;
        width: 0 !important;
        height: 0 !important;
        min-height: 0 !important;
        border: none !important;
        margin: 0 !important;
        padding: 0 !important;
        opacity: 0 !important;
        pointer-events: none !important;
        overflow: hidden !important;
    }

    /* Step 2 Candidate Card Deck: Responsive Height and Button Alignment */
    .candidate-card-deck-box {
        height: 205px !important;
        min-height: 205px !important;
        max-height: 205px !important;
        overflow-y: auto !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        box-sizing: border-box !important;
        padding: 10px 12px !important;
        margin-bottom: 8px !important;
    }
    .candidate-card-deck-box .badge-tag {
        margin-right: 0 !important;
        margin-bottom: 0 !important;
        white-space: nowrap !important;
        flex-shrink: 0 !important;
    }
    .candidate-card-clamp {
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        text-overflow: ellipsis;
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

    /* Step 2 Active Additive & Electrolyte System Banner (13" and 31" Responsive Layout) */
    .active-banner-transfer-row {
        display: flex !important;
        align-items: center !important;
        justify-content: flex-end !important;
        gap: 8px !important;
        flex-wrap: nowrap !important;
        width: 100% !important;
    }
    .active-banner-elyte-box {
        width: fit-content !important;
        min-width: 150px !important;
        max-width: 240px !important;
        min-height: 52px !important;
        padding: 8px 12px !important;
        border-radius: 6px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        box-sizing: border-box !important;
    }
    @media (max-width: 1440px) {
        .active-banner-elyte-box {
            min-width: 140px !important;
            max-width: 220px !important;
            padding: 6px 10px !important;
        }
    }

    /* Tab 5 Download Center: Column Header & Button Row Alignment across 13" and 31" screens */
    .dl-center-col-header {
        min-height: 2.75rem !important;
        display: flex !important;
        align-items: flex-end !important;
        font-weight: 700 !important;
        line-height: 1.25 !important;
        margin-bottom: 6px !important;
    }
    div[class*="st-key-tab5_btn_dl_"] button,
    div[class*="st-key-tab5_btn_dl_"] [data-testid="stDownloadButton"] > button {
        min-height: 52px !important;
        height: auto !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        line-height: 1.25 !important;
        padding: 6px 10px !important;
        white-space: normal !important;
        box-sizing: border-box !important;
    }
    div[class*="st-key-tab5_btn_dl_"] button p,
    div[class*="st-key-tab5_btn_dl_"] button div[data-testid="stMarkdownContainer"] {
        margin: 0 !important;
        line-height: 1.25 !important;
        text-align: center !important;
        white-space: normal !important;
    }

    /* Tab 1 Run Autonomous Optimization Experiment Button & Setup Text Horizontal Alignment */
    .st-key-tab1_run_action_row div[data-testid="stHorizontalBlock"] {
        align-items: center !important;
    }
    .st-key-tab1_run_action_row div[data-testid="stColumn"] {
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
    }
    .st-key-tab1_run_action_row div[data-testid="stElementContainer"],
    .st-key-tab1_run_action_row div[data-testid="stMarkdown"],
    .st-key-tab1_run_action_row div[data-testid="stMarkdownContainer"],
    .st-key-tab1_run_action_row .stButton {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    .st-key-btn_tab1_run_campaign button {
        min-height: 44px !important;
        height: auto !important;
        padding: 8px 16px !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    .st-key-btn_tab1_run_campaign button p,
    .st-key-btn_tab1_run_campaign button div[data-testid="stMarkdownContainer"] {
        font-size: 1.0rem !important;
        line-height: 1.35 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .tab1-experiment-setup-info {
        font-size: 1.0rem !important;
        line-height: 1.35 !important;
        min-height: 44px !important;
        display: flex !important;
        align-items: center !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Prevent SVG legend clipPath from truncating legend labels in Plotly charts */
    .js-plotly-plot .legend .scrollbox,
    .plotly .legend .scrollbox,
    .js-plotly-plot g.legend {
        clip-path: none !important;
        overflow: visible !important;
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
                const parentWin = window.parent;
                let probeCanvasCtx = null;

                function parseColorToRgb(cssColor) {
                    if (!cssColor) return null;
                    const s = String(cssColor).trim().toLowerCase();
                    if (!s || s === 'transparent' || s === 'rgba(0, 0, 0, 0)' || s === 'rgba(0,0,0,0)') {
                        return null;
                    }
                    // Standard rgb(...) / rgba(...)
                    const rgbaMatch = s.match(/rgba?\\(\\s*([\\d.]+)[,\\s]+([\\d.]+)[,\\s]+([\\d.]+)(?:[,/\\s]+([\\d.]+))?\\s*\\)/);
                    if (rgbaMatch) {
                        const a = rgbaMatch[4] !== undefined ? parseFloat(rgbaMatch[4]) : 1.0;
                        if (a === 0) return null;
                        return [
                            Math.round(parseFloat(rgbaMatch[1])),
                            Math.round(parseFloat(rgbaMatch[2])),
                            Math.round(parseFloat(rgbaMatch[3]))
                        ];
                    }
                    // macOS Safari / Chrome Wide-Gamut CSS Color 4: color(srgb r g b [/ a]) or color(display-p3 r g b [/ a])
                    const colorFnMatch = s.match(/color\\(\\s*[a-z0-9-]+\\s+([\\d.]+)\\s+([\\d.]+)\\s+([\\d.]+)(?:\\s*\\/\\s*([\\d.]+))?\\s*\\)/);
                    if (colorFnMatch) {
                        const a = colorFnMatch[4] !== undefined ? parseFloat(colorFnMatch[4]) : 1.0;
                        if (a === 0) return null;
                        return [
                            Math.round(parseFloat(colorFnMatch[1]) * 255),
                            Math.round(parseFloat(colorFnMatch[2]) * 255),
                            Math.round(parseFloat(colorFnMatch[3]) * 255)
                        ];
                    }
                    // Fallback: normalize any CSS color format via 1x1 canvas
                    try {
                        if (!probeCanvasCtx) {
                            const c = parentDoc.createElement('canvas');
                            c.width = 1;
                            c.height = 1;
                            probeCanvasCtx = c.getContext('2d', { willReadFrequently: true });
                        }
                        if (probeCanvasCtx) {
                            probeCanvasCtx.clearRect(0, 0, 1, 1);
                            probeCanvasCtx.fillStyle = s;
                            probeCanvasCtx.fillRect(0, 0, 1, 1);
                            const d = probeCanvasCtx.getImageData(0, 0, 1, 1).data;
                            if (d[3] === 0) return null;
                            return [d[0], d[1], d[2]];
                        }
                    } catch (err) {}
                    return null;
                }

                function computeLuminance(rgb) {
                    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
                }

                function syncThemeAttribute() {
                    const appEl = parentDoc.querySelector('.stApp');
                    if (!appEl) return;
                    const candidates = [
                        appEl,
                        parentDoc.querySelector('[data-testid="stAppViewContainer"]'),
                        parentDoc.querySelector('[data-testid="stHeader"]'),
                        parentDoc.querySelector('[data-testid="stSidebar"]'),
                        parentDoc.body,
                        parentDoc.documentElement
                    ];
                    let isDark = null;
                    for (let i = 0; i < candidates.length; i++) {
                        const el = candidates[i];
                        if (!el) continue;
                        const bg = parentWin.getComputedStyle(el).backgroundColor || '';
                        const rgb = parseColorToRgb(bg);
                        if (rgb) {
                            isDark = computeLuminance(rgb) < 128;
                            break;
                        }
                    }
                    if (isDark === null) {
                        const txtRgb = parseColorToRgb(parentWin.getComputedStyle(appEl).color || '');
                        if (txtRgb) {
                            isDark = computeLuminance(txtRgb) > 160;
                        } else if (parentWin.matchMedia) {
                            isDark = parentWin.matchMedia('(prefers-color-scheme: dark)').matches;
                        } else {
                            isDark = false;
                        }
                    }
                    const themeVal = isDark ? 'dark' : 'light';
                    const targets = [
                        appEl,
                        parentDoc.documentElement,
                        parentDoc.body,
                        parentDoc.querySelector('[data-testid="stAppViewContainer"]')
                    ];
                    for (let j = 0; j < targets.length; j++) {
                        const t = targets[j];
                        if (t && t.getAttribute('data-theme') !== themeVal) {
                            t.setAttribute('data-theme', themeVal);
                        }
                    }
                }

                function refreshVisiblePlotlyCharts() {
                    try {
                        const plots = parentDoc.querySelectorAll('.js-plotly-plot');
                        for (let i = 0; i < plots.length; i++) {
                            const gd = plots[i];
                            if (gd && gd.offsetWidth > 50 && parentWin.Plotly) {
                                try {
                                    parentWin.Plotly.Plots.resize(gd);
                                    parentWin.Plotly.relayout(gd, { autosize: true });
                                } catch (err) {}
                            }
                        }
                    } catch (e) {}
                }

                syncThemeAttribute();
                const appEl = parentDoc.querySelector('.stApp');
                if (appEl && !parentWin.__mapThemeObserverAttached) {
                    parentWin.__mapThemeObserverAttached = true;
                    const obs = new MutationObserver(syncThemeAttribute);
                    obs.observe(appEl, { attributes: true, attributeFilter: ['class', 'style'] });
                    obs.observe(parentDoc.documentElement, { attributes: true, attributeFilter: ['class', 'style'] });
                    if (parentDoc.body) {
                        obs.observe(parentDoc.body, { attributes: true, attributeFilter: ['class', 'style'] });
                    }
                    parentWin.setInterval(syncThemeAttribute, 400);
                    parentDoc.addEventListener('click', function(ev) {
                        const tabBtn = ev.target && ev.target.closest ? ev.target.closest('button[role="tab"], button[data-baseweb="tab"]') : null;
                        if (tabBtn) {
                            try {
                                const openPopoverBtns = parentDoc.querySelectorAll('[data-testid="stPopoverButton"][aria-expanded="true"], [data-testid="stPopover"] button[aria-expanded="true"]');
                                openPopoverBtns.forEach(function(btn) {
                                    if (btn !== tabBtn) {
                                        btn.click();
                                    }
                                });
                            } catch (popErr) {}
                            setTimeout(refreshVisiblePlotlyCharts, 60);
                            setTimeout(refreshVisiblePlotlyCharts, 220);
                        }
                    }, true);
                }
            } catch(e) {}
        })();
        </script>
        """,
        height=0,
        width=0,
    )


