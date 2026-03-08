# helpers/style_utils.py
from __future__ import annotations

import streamlit as st


def apply_blue_clean_base_theme(force: bool = False) -> None:
    """Inject a shared blue/glow baseline theme for all tabs."""
    if st.session_state.get("_blue_clean_theme_applied", False) and not force:
        return

    st.markdown(
        """
        <style>
        :root{
          --tw-blue-1: rgba(74, 170, 255, 0.92);
          --tw-blue-2: rgba(36, 104, 176, 0.84);
          --tw-blue-border: rgba(128, 206, 255, 0.34);
          --tw-blue-border-strong: rgba(170, 232, 255, 0.78);
          --tw-panel: rgba(10, 18, 30, 0.40);
          --tw-panel-2: rgba(12, 24, 42, 0.56);
          --tw-text: rgba(236, 248, 255, 0.98);
          --tw-text-soft: rgba(196, 226, 246, 0.86);
        }

        .block-container{
          padding-top: 1.8rem;
        }

        div[data-testid="stButton"] > button{
          border-radius: 12px !important;
          border: 1px solid rgba(138, 214, 255, 0.58) !important;
          background: linear-gradient(180deg, rgba(28,74,120,0.72), rgba(12,36,68,0.66)) !important;
          color: var(--tw-text) !important;
          box-shadow: 0 8px 18px rgba(8,30,58,0.32), 0 0 12px rgba(74,170,255,0.18) !important;
          transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease !important;
        }
        div[data-testid="stButton"] > button:hover{
          transform: translateY(-1px);
          border-color: rgba(188, 238, 255, 0.86) !important;
          box-shadow: 0 12px 24px rgba(8,30,58,0.36), 0 0 16px rgba(96, 194, 255, 0.30) !important;
        }
        div[data-testid="stButton"] > button[kind="primary"]{
          border-color: var(--tw-blue-border-strong) !important;
          background: linear-gradient(180deg, var(--tw-blue-1), var(--tw-blue-2)) !important;
          box-shadow: 0 14px 24px rgba(12, 68, 124, 0.40), 0 0 18px rgba(96,194,255,0.34) !important;
        }
        div[data-testid="stButton"] > button:disabled{
          opacity: 0.78 !important;
          color: rgba(212,238,255,0.92) !important;
          border-color: rgba(128,206,255,0.32) !important;
          background: linear-gradient(180deg, rgba(24,62,102,0.52), rgba(12,34,64,0.48)) !important;
          box-shadow: 0 4px 10px rgba(8,30,58,0.20) !important;
        }

        div[data-testid="stMultiSelect"] div[data-baseweb="tag"],
        div[data-testid="stMultiSelect"] span[data-baseweb="tag"]{
          background: linear-gradient(180deg, rgba(70,160,238,0.92), rgba(32,96,168,0.90)) !important;
          background-color: rgba(44,124,206,0.92) !important;
          border: 1px solid rgba(170,232,255,0.78) !important;
          color: rgba(244,252,255,0.99) !important;
          box-shadow: 0 0 0 1px rgba(108,198,255,0.24), 0 4px 10px rgba(10,46,84,0.30) !important;
        }
        div[data-testid="stMultiSelect"] div[data-baseweb="tag"] *,
        div[data-testid="stMultiSelect"] span[data-baseweb="tag"] *{
          color: rgba(244,252,255,0.99) !important;
        }
        div[data-testid="stMultiSelect"] div[data-baseweb="tag"] svg,
        div[data-testid="stMultiSelect"] span[data-baseweb="tag"] svg{
          fill: rgba(238,250,255,0.98) !important;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div,
        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stDateInput"] input{
          background: rgba(10,18,30,0.60) !important;
          border: 1px solid rgba(128, 206, 255, 0.28) !important;
          color: var(--tw-text) !important;
          border-radius: 10px !important;
        }
        div[data-baseweb="input"] > div:focus-within,
        div[data-baseweb="select"] > div:focus-within,
        div[data-testid="stTextInput"] input:focus,
        div[data-testid="stTextArea"] textarea:focus,
        div[data-testid="stNumberInput"] input:focus,
        div[data-testid="stDateInput"] input:focus{
          border-color: rgba(164, 230, 255, 0.70) !important;
          box-shadow: 0 0 0 1px rgba(100, 196, 255, 0.36), 0 0 14px rgba(100,196,255,0.18) !important;
        }

        div[data-testid="stExpander"] details{
          border: 1px solid rgba(132, 214, 255, 0.22) !important;
          border-radius: 12px !important;
          background: linear-gradient(165deg, var(--tw-panel-2), var(--tw-panel)) !important;
        }
        div[data-testid="stExpander"] details[open]{
          border-color: rgba(170, 232, 255, 0.40) !important;
          box-shadow: 0 12px 22px rgba(0,0,0,0.26), 0 0 14px rgba(82, 182, 255, 0.20) !important;
        }
        div[data-testid="stExpander"] summary{
          border-bottom: 1px solid rgba(132, 214, 255, 0.16) !important;
        }

        div[data-testid="stDataFrame"]{
          border: 1px solid rgba(132, 214, 255, 0.22) !important;
          border-radius: 12px !important;
          overflow: hidden !important;
          box-shadow: 0 8px 18px rgba(0,0,0,0.18) !important;
        }

        div[data-testid="stSegmentedControl"]{
          border-radius: 12px !important;
          border: 1px solid rgba(128,206,255,0.32) !important;
          background: rgba(10,20,36,0.44) !important;
          padding: 2px !important;
        }
        div[data-testid="stSegmentedControl"] button{
          border-radius: 10px !important;
          border: 1px solid transparent !important;
          color: rgba(223,241,254,0.94) !important;
        }
        div[data-testid="stSegmentedControl"] button[aria-selected="true"]{
          border-color: rgba(172,234,255,0.86) !important;
          background: linear-gradient(180deg, rgba(76,168,255,0.90), rgba(32,98,172,0.88)) !important;
          color: rgba(242,252,255,0.99) !important;
          box-shadow: 0 8px 16px rgba(18,76,138,0.34), 0 0 14px rgba(84,174,255,0.24) !important;
        }

        input[type="radio"],
        input[type="checkbox"]{
          accent-color: #66c5ff !important;
        }
        [data-baseweb="radio"] input + div{
          border-color: rgba(132,214,255,0.52) !important;
          background: rgba(10,20,36,0.30) !important;
        }
        [data-baseweb="radio"] input:checked + div{
          border-color: rgba(178,236,255,0.92) !important;
          box-shadow: 0 0 0 1px rgba(120,208,255,0.48), 0 0 12px rgba(100,196,255,0.34) !important;
          background: linear-gradient(180deg, rgba(66,160,242,0.94), rgba(36,108,184,0.90)) !important;
        }

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stCaptionContainer"]{
          color: var(--tw-text-soft);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["_blue_clean_theme_applied"] = True


def color_status(val):
    s = str(val).strip()
    colors = {"Pending": "orange", "In Progress": "dodgerblue", "Scheduled": "teal", "Failed": "red", "Done": "green"}
    return f"color: {colors.get(s, 'black')}; font-weight: bold"


def color_priority(val):
    p = str(val).strip()
    colors = {"Low": "gray", "Normal": "black", "High": "crimson"}
    return f"color: {colors.get(p, 'black')}; font-weight: bold"
