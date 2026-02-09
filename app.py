"""
AI-Powered Customer Churn Prediction Dashboard
Streamlit app: upload data, view predictions, filters, charts, single-customer form.
"""
import os
import sys
import io
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    PREDICTIONS_PATH,
    PREDICTIONS_ENRICHED_PATH,
    MODEL_PATH,
    COLUMN_META_PATH,
    EMAIL_VERIFICATION_ENABLED,
)
from src.data_loader import prepare_data
from src.predict import predict_proba, ensure_columns
from src.roi import compute_revenue_at_risk, compute_roi
from src import auth_db
from src.email_sender import send_verification_email

# Page config — must be first Streamlit command for sidebar/layout to work
st.set_page_config(
    page_title="Telco Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session state for auth
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "terms_read" not in st.session_state:
    st.session_state.terms_read = False
if "auth_tab" not in st.session_state:
    st.session_state.auth_tab = "Login"
if "show_about" not in st.session_state:
    st.session_state.show_about = False
if "show_delete_account" not in st.session_state:
    st.session_state.show_delete_account = False
if "verified_message" not in st.session_state:
    st.session_state.verified_message = None

# Design system — rich theme with depth and interactivity
PALETTE = {
    "primary": "#0EA5E9",
    "primary_dark": "#0284C7",
    "accent": "#F97316",
    "sidebar": "linear-gradient(180deg, #1E293B 0%, #0F172A 100%)",
    "sidebar_solid": "#1E293B",
    "bg_main": "linear-gradient(160deg, #F0F9FF 0%, #E0F2FE 40%, #F8FAFC 100%)",
    "bg_main_solid": "#F0F9FF",
    "cards": "#FFFFFF",
    "cards_glass": "rgba(255,255,255,0.92)",
    "text": "#0F172A",
    "text_muted": "#64748B",
    "border": "#E2E8F0",
    "low_risk": "#22C55E",
    "medium_risk": "#F59E0B",
    "high_risk": "#EF4444",
    "primary_light": "#BAE6FD",
    "sidebar_text": "#F1F5F9",
    "sidebar_text_muted": "#94A3B8",
    "shadow_soft": "0 4px 14px rgba(14, 165, 233, 0.12)",
    "shadow_medium": "0 10px 40px -10px rgba(14, 165, 233, 0.2)",
    "shadow_lift": "0 20px 50px -15px rgba(0,0,0,0.15)",
}

# Login branding: navy text, teal accent (themeable via CSS)
LOGIN_NAVY = "#0F172A"
LOGIN_TEAL = "#0D9488"

# Inline SVG: line chart inside shield (no CDN; fill/stroke via currentColor)
LOGIN_LOGO_SVG = """
<svg class="login-logo-svg" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <path d="M24 4L8 10v14c0 10 6 18 16 20 10-2 16-10 16-20V10L24 4z" stroke="currentColor" stroke-width="2" stroke-linejoin="round" fill="none"/>
  <path d="M16 28l4-6 4 4 8-10" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
</svg>
"""

# Custom CSS — production-grade, maintainable
st.markdown(f"""
<style>
    /* ----- Hide Streamlit top toolbar (white bar with Deploy / menu) ----- */
    header[data-testid="stHeader"],
    [data-testid="stHeader"],
    .stApp header {{
        display: none !important;
    }}

    /* ----- Base — gradient background ----- */
    .stApp {{
        background: {PALETTE["bg_main"]} !important;
        background-attachment: fixed !important;
    }}
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }}

    /* ----- Dashboard header — title left, menu top right ----- */
    .dashboard-top-row {{
        margin: 0;
        padding: 0;
        height: 0;
        overflow: hidden;
        border: none;
    }}
    .dashboard-top-row + div {{
        display: flex !important;
        justify-content: space-between !important;
        align-items: flex-start !important;
        width: 100%;
    }}
    .dashboard-header-wrap .dashboard-header,
    .dashboard-header-wrap p.dashboard-header {{
        font-size: 6rem !important;
        font-weight: 700 !important;
        color: {PALETTE["text"]} !important;
        margin-bottom: 0.25rem !important;
        letter-spacing: -0.02em !important;
        line-height: 1.3 !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }}
    .dashboard-header-wrap {{
        padding: 0 0 1rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid transparent;
        border-image: linear-gradient(90deg, {PALETTE["primary"]}, {PALETTE["accent"]}) 1;
        box-shadow: 0 4px 20px rgba(14, 165, 233, 0.08);
    }}
    /* Hide popover chevron/arrow (keep ⋮ label only) */
    .main [data-testid="stPopover"] button svg {{
        display: none !important;
    }}
    .dashboard-subtitle {{
        color: {PALETTE["text_muted"]} !important;
        font-size: 0.9375rem !important;
        margin-bottom: 0 !important;
        line-height: 1.5 !important;
    }}

    /* ----- KPI cards — 3D lift on hover, soft primary shadow ----- */
    div[data-testid="stMetric"] {{
        background: {PALETTE["cards"]} !important;
        border: 1px solid {PALETTE["border"]};
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.06), 0 2px 6px rgba(14, 165, 233, 0.06);
        transition: transform 0.25s ease, box-shadow 0.25s ease !important;
    }}
    div[data-testid="stMetric"]:hover {{
        transform: translateY(-4px);
        box-shadow: {PALETTE["shadow_lift"]};
    }}
    div[data-testid="stMetric"] label {{
        color: {PALETTE["text_muted"]} !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
        font-size: 1.625rem !important;
        font-weight: 700 !important;
        color: {PALETTE["primary_dark"]} !important;
    }}

    /* ----- Section titles — accent left border ----- */
    .main h2, .main h3 {{
        color: {PALETTE["text"]} !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        margin-top: 1.75rem !important;
        padding-bottom: 0.5rem;
        padding-left: 0.75rem;
        border-left: 4px solid {PALETTE["primary"]};
        border-bottom: 1px solid {PALETTE["border"]};
    }}

    /* ----- Sidebar — force visible, gradient, glass effect ----- */
    section[data-testid="stSidebar"],
    [data-testid="stSidebar"] {{
        display: block !important;
        visibility: visible !important;
        transform: none !important;
        min-width: 21rem !important;
        background: {PALETTE["sidebar"]} !important;
        border-right: 1px solid rgba(255,255,255,0.08);
        box-shadow: 4px 0 24px rgba(0,0,0,0.12);
    }}
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
        overflow-y: auto !important;
        visibility: visible !important;
    }}
    [data-testid="stSidebar"] > div:first-child {{
        background: transparent !important;
    }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 600 !important;
    }}
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {{
        color: #FFFFFF !important;
    }}
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stMultiSelect [data-testid="stMarkdown"] {{
        color: #FFFFFF !important;
    }}
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stRadio div,
    [data-testid="stSidebar"] .stRadio span,
    [data-testid="stSidebar"] .stRadio p {{
        color: #FFFFFF !important;
    }}
    [data-testid="stSidebar"] .stSelectbox label {{
        color: #FFFFFF !important;
    }}
    /* Multiselect: dark input area to match sidebar (no big white box) */
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] > div {{
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 8px !important;
    }}
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"],
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] span {{
        background: rgba(14, 165, 233, 0.4) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(14, 165, 233, 0.8) !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.2);
    }}
    [data-testid="stSidebar"] .stCaptionContainer {{
        color: rgba(255,255,255,0.7) !important;
    }}
    [data-testid="stSidebar"] button {{
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }}
    [data-testid="stSidebar"] button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3) !important;
    }}

    /* ----- Dividers & spacing ----- */
    hr {{
        border-color: {PALETTE["border"]} !important;
    }}
    .dashboard-footer {{
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid {PALETTE["border"]};
        color: {PALETTE["text_muted"]};
        font-size: 0.75rem;
    }}

    /* ----- Auth (login/signup) — centered, clean on page background ----- */
    .main .block-container:has(.auth-container) {{
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        max-width: 100%;
    }}
    .main .block-container:has(.auth-container) > * {{
        width: 100% !important;
        max-width: 720px !important;
    }}
    .auth-container {{
        max-width: 720px;
        margin: 4rem auto 0;
        padding: 2rem 0;
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
    }}
    .auth-title {{
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, {PALETTE["primary_dark"]}, {PALETTE["primary"]}) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: -0.02em !important;
    }}
    .auth-subtitle {{
        color: {PALETTE["text_muted"]} !important;
        font-size: 1.1rem !important;
        margin-bottom: 1.75rem !important;
        line-height: 1.5 !important;
    }}
    /* Login branded header: SVG + title + tagline + badge */
    .login-brand-header {{
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        margin-bottom: 0.5rem;
    }}
    .login-logo-svg {{
        width: 120px;
        height: 120px;
        flex-shrink: 0;
        color: {LOGIN_TEAL};
    }}
    .login-brand-text {{
        flex: 1;
        min-width: 0;
    }}
    .login-brand-title {{
        font-size: 6rem !important;
        font-weight: 700 !important;
        color: {LOGIN_NAVY} !important;
        margin: 0 0 0.25rem 0 !important;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }}
    .login-brand-tagline {{
        font-size: 0.9rem !important;
        color: {PALETTE["text_muted"]} !important;
        margin: 0 !important;
        line-height: 1.4;
    }}
    .login-brand-badge {{
        display: inline-block;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        color: {LOGIN_TEAL} !important;
        background: rgba(13, 148, 136, 0.12);
        padding: 0.2rem 0.5rem;
        border-radius: 999px;
        margin-top: 0.35rem;
        letter-spacing: 0.02em;
    }}
    .login-header-label {{
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        color: {PALETTE["text_muted"]} !important;
        text-align: right;
        margin-top: 0.25rem;
    }}
    .auth-container.login-view {{
        max-width: 720px;
    }}
    .auth-container.signup-view {{
        max-width: 720px;
    }}
    /* Sign up screen: same SaaS-style card as login */
    .signup-screen-marker ~ div:has(form[data-testid="stForm"]),
    .signup-screen-marker ~ form[data-testid="stForm"] {{
        background: {PALETTE["cards"]} !important;
        border: 1px solid {PALETTE["border"]};
        border-radius: 20px;
        padding: 2rem 2rem 1.5rem !important;
        box-shadow: {PALETTE["shadow_soft"]}, 0 20px 50px -20px rgba(14, 165, 233, 0.15);
        margin: 0 auto 1rem !important;
        max-width: 420px;
        animation: loginFadeIn 0.45s ease-out;
    }}
    .auth-container.signup-view form[data-testid="stForm"] [data-testid="stTextInput"] input {{
        border-radius: 12px;
        border: 2px solid {PALETTE["border"]};
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }}
    .auth-container.signup-view form[data-testid="stForm"] [data-testid="stTextInput"] input:focus {{
        border-color: {PALETTE["primary"]} !important;
        box-shadow: 0 0 0 4px rgba(14, 165, 233, 0.15) !important;
        outline: none !important;
    }}
    .auth-container.signup-view form[data-testid="stForm"] [data-testid="stFormSubmitButton"] button {{
        border-radius: 12px;
        font-weight: 600;
        padding: 0.7rem 1.5rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }}
    .auth-container.signup-view form[data-testid="stForm"] [data-testid="stFormSubmitButton"] button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(14, 165, 233, 0.35) !important;
    }}
    /* Auth header row: title left, three-dots menu right */
    .auth-header-row {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1rem;
        margin-bottom: 0.5rem;
    }}
    .auth-header-row .auth-title {{
        margin-bottom: 0 !important;
    }}
    /* Tabs — bigger, hover with accent */
    [data-testid="stTabs"] [role="tab"] {{
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.25rem !important;
        transition: color 0.2s ease, transform 0.2s ease !important;
    }}
    [data-testid="stTabs"] [role="tab"]:hover {{
        color: {PALETTE["primary"]} !important;
        transform: translateY(-1px);
    }}
    /* Inputs — 3D inset, focus glow */
    [data-testid="stTextInput"] label {{
        font-size: 1.05rem !important;
        font-weight: 500 !important;
    }}
    [data-testid="stTextInput"] input {{
        font-size: 1.05rem !important;
        padding: 0.7rem 0.9rem !important;
        border-radius: 12px !important;
        border: 2px solid {PALETTE["border"]} !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.04);
        transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease !important;
    }}
    [data-testid="stTextInput"] input:hover {{
        border-color: {PALETTE["text_muted"]} !important;
    }}
    [data-testid="stTextInput"] input:focus {{
        border-color: {PALETTE["border"]} !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.04) !important;
        outline: none !important;
    }}
    /* Form submit button — 3D raised, gradient hover */
    [data-testid="stFormSubmitButton"] button,
    form [data-testid="baseButton-primary"] button,
    .stForm button {{
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding: 0.65rem 1.5rem !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 14px rgba(14, 165, 233, 0.35);
        transition: all 0.25s ease !important;
    }}
    [data-testid="stFormSubmitButton"] button:hover,
    form [data-testid="baseButton-primary"] button:hover,
    .stForm button:hover {{
        background: linear-gradient(135deg, {PALETTE["primary"]}, {PALETTE["primary_dark"]}) !important;
        color: white !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(14, 165, 233, 0.4) !important;
    }}
    [data-testid="stFormSubmitButton"] button:active,
    .stForm button:active {{
        transform: translateY(0) scale(0.98);
    }}

    /* ===== LOGIN SCREEN ONLY — SaaS-style card, icons, micro-effects ===== */
    .login-screen-marker ~ div:has(form[data-testid="stForm"]),
    .login-screen-marker ~ form[data-testid="stForm"] {{
        background: {PALETTE["cards"]} !important;
        border: 1px solid {PALETTE["border"]};
        border-radius: 20px;
        padding: 2rem 2rem 1.5rem !important;
        box-shadow: {PALETTE["shadow_soft"]}, 0 20px 50px -20px rgba(14, 165, 233, 0.15);
        margin: 0 auto 1rem !important;
        max-width: 420px;
        animation: loginFadeIn 0.45s ease-out;
    }}
    @keyframes loginFadeIn {{
        from {{ opacity: 0; transform: translateY(12px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .login-tagline {{
        color: {PALETTE["text_muted"]} !important;
        font-size: 0.95rem !important;
        margin: -0.5rem 0 1.25rem !important;
        line-height: 1.5;
        letter-spacing: 0.01em;
    }}
    /* Login form: stronger focus glow (scope by marker) */
    .auth-container form[data-testid="stForm"] [data-testid="stTextInput"] input {{
        border-radius: 12px;
        border: 2px solid {PALETTE["border"]};
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }}
    .auth-container form[data-testid="stForm"] [data-testid="stTextInput"] input:focus {{
        border-color: {PALETTE["primary"]} !important;
        box-shadow: 0 0 0 4px rgba(14, 165, 233, 0.15) !important;
        outline: none !important;
    }}
    .auth-container form[data-testid="stForm"] [data-testid="stFormSubmitButton"] button {{
        border-radius: 12px;
        font-weight: 600;
        padding: 0.7rem 1.5rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }}
    .auth-container form[data-testid="stForm"] [data-testid="stFormSubmitButton"] button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(14, 165, 233, 0.35) !important;
    }}
    .login-forgot-link {{
        color: {PALETTE["primary_dark"]} !important;
        font-size: 0.9rem;
        text-decoration: none;
        margin-top: 0.5rem;
        display: inline-block;
    }}
    .login-forgot-link:hover {{
        text-decoration: underline;
    }}
    .login-trust {{
        color: {PALETTE["text_muted"]} !important;
        font-size: 0.8rem !important;
        margin-top: 1rem;
        letter-spacing: 0.02em;
    }}
</style>
""", unsafe_allow_html=True)


# Placeholder Terms and Conditions; replace with your legal text.
TERMS_AND_CONDITIONS = """
**1. Acceptance of Terms**  
By creating an account and using the Churn Prediction Dashboard, you agree to these Terms and Conditions. If you do not agree, do not use the service.

**2. Use of Service**  
The Dashboard is provided for lawful business use. You must not misuse the service, attempt to gain unauthorized access, or use it in any way that could harm the system or other users.

**3. Data and Privacy**  
You are responsible for the data you upload and how it is used. Do not upload personally identifiable information or confidential data unless you have the right to do so and our Privacy Policy allows it.

**4. Account Security**  
You must keep your username and password confidential and notify us of any unauthorized use of your account.

**5. Disclaimer**  
The Dashboard and predictions are provided “as is.” We do not guarantee accuracy of predictions or uninterrupted availability of the service.

**6. Changes**  
We may update these terms from time to time. Continued use of the service after changes constitutes acceptance of the updated terms.
"""


@st.dialog("About")
def show_about_dialog():
    """About modal: app name, version, short description."""
    st.markdown("**Churn Prediction Dashboard**")
    st.markdown("AI-powered dashboard to identify at-risk customers and prioritize retention actions. Predictions reflect the latest model outputs.")
    st.caption("Version 1.0")
    if st.button("Close", type="primary", use_container_width=True):
        st.session_state.show_about = False
        st.rerun()


@st.dialog("Delete Account")
def show_delete_account_dialog():
    """Confirmation dialog: username, password, confirm checkbox, then delete."""
    st.warning("This will **permanently** remove your account. Your username, email and password will be deleted and cannot be recovered.")
    del_user = st.text_input("Username", key="delete_username", placeholder="Enter your username")
    del_pass = st.text_input("Password", type="password", key="delete_password", placeholder="Enter your password")
    confirm_tick = st.checkbox(
        "I confirm I want to delete my account and that my username, email and password will be permanently removed.",
        key="delete_confirm_tick",
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancel", use_container_width=True):
            st.session_state.show_delete_account = False
            if "delete_username" in st.session_state:
                del st.session_state["delete_username"]
            if "delete_password" in st.session_state:
                del st.session_state["delete_password"]
            if "delete_confirm_tick" in st.session_state:
                del st.session_state["delete_confirm_tick"]
            st.rerun()
    with col2:
        delete_btn = st.button("Delete my account", type="primary", use_container_width=True, disabled=not confirm_tick)
    if delete_btn and confirm_tick:
        ok, msg = auth_db.delete_user(del_user, del_pass)
        if ok:
            st.success(msg)
            st.session_state.show_delete_account = False
            for k in ["delete_username", "delete_password", "delete_confirm_tick"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()
        else:
            st.error(msg)


def show_login_signup():
    """Render login/signup form. On success sets session_state and reruns."""
    auth_db.init_db()
    # Handle verification link: ?token=... from email (user clicked link in Gmail etc.)
    token = None
    try:
        if hasattr(st, "query_params") and "token" in st.query_params:
            token = st.query_params.get("token", st.query_params["token"])
        elif hasattr(st, "experimental_get_query_params"):
            q = st.experimental_get_query_params()
            token = (q.get("token") or [None])[0] if q.get("token") else None
    except Exception:
        pass
    if token:
        ok, msg = auth_db.verify_email_token(token)
        st.session_state.verified_message = (ok, msg)
        st.session_state.auth_tab = "Login"
        try:
            if hasattr(st, "query_params"):
                del st.query_params["token"]
            elif hasattr(st, "experimental_set_query_params"):
                q = st.experimental_get_query_params()
                q.pop("token", None)
                st.experimental_set_query_params(**{k: v[0] if isinstance(v, list) and v else v for k, v in q.items()})
        except Exception:
            pass
        st.rerun()
    if st.session_state.get("verified_message"):
        ok, msg = st.session_state.verified_message
        st.session_state.verified_message = None
        if ok:
            st.success(msg)
        else:
            st.error(msg)
    if st.session_state.get("show_about"):
        show_about_dialog()
    if st.session_state.get("show_delete_account"):
        show_delete_account_dialog()
    auth_tab = st.session_state.get("auth_tab", "Login")
    container_class = "auth-container login-view" if auth_tab == "Login" else "auth-container signup-view"
    st.markdown(f"<div class='{container_class}'>", unsafe_allow_html=True)
    col_title, col_menu = st.columns([5, 1])
    with col_title:
        if auth_tab == "Login":
            st.markdown(
                "<div class=\"login-brand-header\">"
                + LOGIN_LOGO_SVG
                + "<div class=\"login-brand-text\">"
                "<p class=\"login-brand-title\">Churn Intelligence Platform</p>"
                "<p class=\"login-brand-tagline\">Predict risk • Explain drivers • Automate retention</p>"
                "<span class=\"login-brand-badge\">PredictEdge</span>"
                "</div></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class=\"login-brand-header\">"
                + LOGIN_LOGO_SVG
                + "<div class=\"login-brand-text\">"
                "<p class=\"login-brand-title\">Churn Intelligence Platform</p>"
                "<p class=\"login-brand-tagline\">Predict risk • Explain drivers • Automate retention</p>"
                "<span class=\"login-brand-badge\">PredictEdge</span>"
                "</div></div>",
                unsafe_allow_html=True,
            )
    with col_menu:
        if auth_tab == "Login":
            st.markdown("<p class='login-header-label'>Login</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='login-header-label'>Sign up</p>", unsafe_allow_html=True)
        with st.popover("⋮", help="Menu"):
            if st.button("Account", use_container_width=True, key="menu_account"):
                st.session_state.auth_tab = "Login"
                st.rerun()
            if st.button("About", use_container_width=True, key="menu_about"):
                st.session_state.show_about = True
                st.rerun()
    st.markdown(
        "<p class='auth-subtitle'>Predict risk • Explain drivers • Automate retention</p>",
        unsafe_allow_html=True,
    )
    auth_tab = st.radio("Login or Sign up", ["Login", "Sign up"], key="auth_tab", horizontal=True, label_visibility="collapsed")
    if auth_tab == "Login":
        st.markdown("<div class='login-screen-marker' aria-hidden='true'></div>", unsafe_allow_html=True)
        login_show_pass = st.session_state.get("login_show_password", False)
        with st.form("login_form"):
            login_user = st.text_input(
                "👤 Username",
                key="login_username",
                placeholder="Enter your username",
                label_visibility="visible",
            )
            login_pass = st.text_input(
                "🔒 Password",
                type="password" if not login_show_pass else "default",
                key="login_password",
                placeholder="Enter your password",
                label_visibility="visible",
            )
            show_pass_col, remember_col = st.columns(2)
            with show_pass_col:
                st.checkbox("Show password", key="login_show_password", value=login_show_pass)
            with remember_col:
                st.checkbox("Remember me", key="login_remember", value=st.session_state.get("login_remember", False))
            login_btn = st.form_submit_button("Log in →")
        if login_btn:
            with st.spinner("Signing in..."):
                ok, msg = auth_db.verify_user(login_user, login_pass)
            if ok:
                st.session_state.authenticated = True
                st.session_state.username = msg
                st.rerun()
            else:
                st.error(msg)
        st.markdown(
            "<p class='login-forgot-link'>Forgot password?</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p class='login-trust'>Secured login • Data privacy • Salesforce-ready pipeline</p>",
            unsafe_allow_html=True,
        )
        st.caption("Want to remove your account?")
        if st.button("Delete Account", type="secondary", key="delete_account_btn"):
            st.session_state.show_delete_account = True
            st.rerun()
    else:
        st.markdown("<div class='signup-screen-marker' aria-hidden='true'></div>", unsafe_allow_html=True)
        # User must open and read Terms before they can accept
        with st.expander("**Terms and Conditions** — please read before signing up", expanded=False):
            st.markdown(TERMS_AND_CONDITIONS)
            if st.button("I have read the Terms and Conditions", key="terms_read_btn"):
                st.session_state.terms_read = True
                st.rerun()
        if st.session_state.terms_read:
            st.caption("✓ You have opened and read the Terms and Conditions. You may accept below.")
        with st.form("signup_form"):
            signup_email = st.text_input(
                "Email",
                key="signup_email",
                placeholder="Enter your email (required for verification)" if EMAIL_VERIFICATION_ENABLED else "Enter your email",
                help="We will send a verification link to this address. You must verify before you can log in." if EMAIL_VERIFICATION_ENABLED else None,
            )
            signup_user = st.text_input("Username", key="signup_username", placeholder="Choose a username (min 2 chars)")
            signup_pass = st.text_input(
                "Password",
                type="password",
                key="signup_password",
                placeholder="Min 8 chars; include A–Z, a–z, 0–9, symbol (e.g. !@#$)",
                help="Minimum 8 characters (12+ recommended). Must include uppercase, lowercase, number, and symbol.",
            )
            signup_confirm = st.text_input("Confirm password", type="password", key="signup_confirm", placeholder="Re-enter your password")
            accept_tc = st.checkbox(
                "I accept the Terms and Conditions",
                key="signup_accept_tc",
                disabled=not st.session_state.get("terms_read", False),
                help="Open and read the Terms and Conditions above, then click \"I have read...\" to enable this.",
            )
            signup_btn = st.form_submit_button("Create account")
        if signup_btn:
            if not st.session_state.get("terms_read", False):
                st.error("Please open and read the Terms and Conditions above, then click \"I have read the Terms and Conditions\" before creating an account.")
            elif not st.session_state.get("signup_accept_tc", False):
                st.error("You must accept the Terms and Conditions to create an account.")
            else:
                ok, msg, verification_token = auth_db.add_user(
                    signup_user, signup_pass, email=signup_email, confirm_password=signup_confirm
                )
                if ok:
                    if verification_token and signup_email:
                        sent, err = send_verification_email(signup_email, signup_user, verification_token)
                        if not sent:
                            st.warning(f"Account created, but we could not send the verification email: {err}. Please contact support.")
                        else:
                            st.success(msg)
                    else:
                        st.success(msg)
                    # Reset so next signup must read terms again
                    st.session_state.terms_read = False
                    if "signup_accept_tc" in st.session_state:
                        del st.session_state["signup_accept_tc"]
                else:
                    st.error(msg)
    st.markdown("</div>", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_meta():
    """Load model and column metadata once."""
    if not os.path.exists(MODEL_PATH):
        return None, None, None, None
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(os.path.join(os.path.dirname(MODEL_PATH), "feature_encoder.joblib"))
    scaler = joblib.load(os.path.join(os.path.dirname(MODEL_PATH), "feature_scaler.joblib"))
    column_meta = joblib.load(COLUMN_META_PATH)
    return model, encoder, scaler, column_meta


@st.cache_data(ttl=60)
def load_predictions_data():
    """Load predictions CSV (prefer enriched with Top_Drivers) or run prediction on cleaned data."""
    if os.path.exists(PREDICTIONS_ENRICHED_PATH):
        return pd.read_csv(PREDICTIONS_ENRICHED_PATH)
    if os.path.exists(PREDICTIONS_PATH):
        return pd.read_csv(PREDICTIONS_PATH)
    model, encoder, scaler, column_meta = load_model_and_meta()
    if model is None:
        return None
    df = prepare_data(save_cleaned=True)
    proba = predict_proba(df, model=model, encoder=encoder, scaler=scaler, column_meta=column_meta)
    df = df.copy()
    df["Churn Probability"] = proba
    df["Risk Level"] = pd.cut(proba, bins=[-0.01, 0.33, 0.66, 1.01], labels=["Low", "Medium", "High"])
    return df


def feature_importance_chart(model, column_meta):
    """Feature importance bar chart (Random Forest / XGBoost)."""
    if not hasattr(model, "feature_importances_"):
        return None
    names = column_meta["feature_names"]
    imp = model.feature_importances_
    df_imp = pd.DataFrame({"Feature": names, "Importance": imp}).sort_values("Importance", ascending=True)
    fig = px.bar(df_imp.tail(20), x="Importance", y="Feature", orientation="h", title="Top 20 Feature Importance")
    fig.update_layout(height=400, margin=dict(l=120))
    return fig


def _format_feature_name(name):
    """Clean one-hot encoded feature names for display."""
    if "_" in str(name) and not str(name).startswith("Contract") and not str(name).startswith("Payment"):
        # e.g. "Paperless Billing_Yes" -> "Paperless Billing: Yes"
        parts = str(name).split("_", 1)
        return f"{parts[0]}: {parts[1]}" if len(parts) == 2 else name
    return str(name).replace("_", " ").title()


def _parse_top_drivers(drivers_str):
    """
    Parse Top_Drivers string into list of (label, impact).
    Example: "Contract=Month-to-month (+0.21), Tenure Months=2 (+0.14)" -> [("Contract=Month-to-month", 0.21), ...]
    """
    import re
    if not drivers_str or pd.isna(drivers_str) or not str(drivers_str).strip():
        return []
    out = []
    for part in str(drivers_str).split(","):
        part = part.strip()
        m = re.search(r"\(([+-]?\d+\.?\d*)\)\s*$", part)
        if m:
            try:
                impact = float(m.group(1))
                label = part[: m.start()].strip()
                if label:
                    out.append((label, impact))
            except ValueError:
                pass
    return out


@st.cache_resource
def _load_shap_explainer():
    """
    Load SHAP explainer for on-demand per-customer explanations.
    Returns (explainer, is_tree, feature_names, column_meta, model, encoder, scaler) or None if SHAP fails.
    """
    try:
        from src.shap_utils import load_artifacts as shap_load, build_explainer, compute_shap_values, get_top_drivers_per_row
        from src.train import build_features
        model, encoder, scaler, feature_names, column_meta = shap_load()
        df = prepare_data(save_cleaned=True)
        df = ensure_columns(df, column_meta["cat_cols"], column_meta["num_cols"])
        X, _, _, _, _, _ = build_features(
            df, encoder=encoder, scaler=scaler, fit=False,
            cat_cols_order=column_meta["cat_cols"], num_cols_order=column_meta["num_cols"],
        )
        n = min(200, len(X))
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), size=n, replace=False)
        X_bg = X[idx]
        explainer, is_tree = build_explainer(model, X_bg)
        return (explainer, is_tree, feature_names, column_meta, model, encoder, scaler)
    except Exception:
        return None


def main():
    if not st.session_state.get("authenticated"):
        show_login_signup()
        return

    if st.session_state.get("show_about"):
        show_about_dialog()
    st.markdown("<div class='dashboard-top-row'>", unsafe_allow_html=True)
    col_head, col_menu = st.columns([5, 1])
    with col_head:
        st.markdown("""
            <div class='dashboard-header-wrap'>
                <p class='dashboard-header'>Churn Intelligence Platform</p>
                <p class='dashboard-subtitle'>Identify at-risk customers and prioritize retention actions. Data reflects latest model predictions.</p>
            </div>
        """, unsafe_allow_html=True)
    with col_menu:
        st.markdown("<div class='dashboard-menu-wrap'>", unsafe_allow_html=True)
        with st.popover("⋮", help="Menu"):
            st.markdown("**Account**")
            st.caption(f"Logged in as **{st.session_state.get('username', '')}**")
            st.divider()
            if st.button("About", use_container_width=True, key="main_menu_about"):
                st.session_state.show_about = True
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption("Use the **sidebar** on the left to filter by risk, switch views (Predictions, ROI, Upload, Single Customer), and change chart options. If you don't see it, click the **▶** arrow at the left edge to open it.")
    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

    model, encoder, scaler, column_meta = load_model_and_meta()
    has_model = model is not None

    # Sidebar
    with st.sidebar:
        st.caption(f"Logged in as **{st.session_state.get('username', '')}**")
        if st.button("Logout", type="primary", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
        st.divider()
        st.header("Filters & Views")
        risk_filter = st.multiselect(
            "Filter by risk level",
            options=["Low", "Medium", "High"],
            default=["Low", "Medium", "High"],
            help="Show only selected risk tiers in charts and table.",
        )
        tab_choice = st.radio(
            "View",
            ["Predictions & Charts", "ROI Simulator", "Upload Dataset", "Single Customer Prediction"],
            index=0,
        )
        st.divider()
        st.subheader("Visualization options")
        viz_dist_type = st.selectbox(
            "Churn distribution",
            ["Histogram", "Density (KDE)", "Both"],
            index=0,
            help="Histogram: counts per bin. Density: smoothed curve. Both: overlay.",
        )
        viz_hist_bins = st.slider("Histogram bins", min_value=10, max_value=60, value=30, help="Number of bins for histogram.")
        viz_risk_chart = st.selectbox(
            "Risk mix chart",
            ["Donut", "Pie", "Bar"],
            index=0,
            help="Donut: ring with hole. Pie: full circle. Bar: horizontal bars.",
        )
        viz_top_n = st.selectbox("Top features (importance)", [10, 15, 20, 25, 30], index=2, help="How many top features to show.")
        viz_extra = st.multiselect(
            "Extra charts",
            ["Risk by contract", "Tenure vs charges (scatter)", "Churn by contract (box)"],
            default=[],
            help="Add optional visualizations.",
        )
        st.divider()
        pred_file = PREDICTIONS_ENRICHED_PATH if os.path.exists(PREDICTIONS_ENRICHED_PATH) else PREDICTIONS_PATH
        if os.path.exists(pred_file):
            mtime = os.path.getmtime(pred_file)
            from datetime import datetime
            st.caption(f"Last run: {datetime.fromtimestamp(mtime).strftime('%b %d, %Y %H:%M')}")

    if tab_choice == "Predictions & Charts":
        if not has_model:
            st.warning("No trained model found. Run `python -m src.train` from the project root, then ensure `data/predictions.csv` exists or re-run the app.")
            return
        df = load_predictions_data()
        if df is None or df.empty:
            st.warning("No predictions data. Run training and prediction first.")
            return

        total = len(df)
        high = (df["Risk Level"] == "High").sum()
        medium = (df["Risk Level"] == "Medium").sum()
        low = (df["Risk Level"] == "Low").sum()

        # Metrics — counts and percentages
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total customers", f"{total:,}", help="Full customer base in scope")
        with c2:
            pct_high = (high / total * 100) if total else 0
            st.metric("High risk", f"{high:,}", delta=f"{pct_high:.1f}% of base", delta_color="inverse", help="Priority for retention outreach")
        with c3:
            pct_med = (medium / total * 100) if total else 0
            st.metric("Medium risk", f"{medium:,}", delta=f"{pct_med:.1f}% of base", delta_color="off", help="Monitor and nurture")
        with c4:
            pct_low = (low / total * 100) if total else 0
            st.metric("Low risk", f"{low:,}", delta=f"{pct_low:.1f}% of base", delta_color="normal", help="Stable segment")

        # Risk filter
        df_filtered = df[df["Risk Level"].isin(risk_filter)] if risk_filter else df
        if len(df_filtered) < len(df):
            st.caption(f"Showing {len(df_filtered):,} customers matching selected risk levels.")

        # Chart theme — from PALETTE for consistency
        chart_bg = PALETTE["cards"]
        chart_plot_bg = PALETTE.get("bg_main_solid", "#F0F9FF")
        chart_grid = PALETTE["border"]
        primary = PALETTE["primary"]
        primary_light = PALETTE["primary_light"]
        high_c = PALETTE["high_risk"]
        medium_c = PALETTE["medium_risk"]
        low_c = PALETTE["low_risk"]
        chart_text = PALETTE["text"]

        # Two columns: distribution + risk mix (options applied)
        col_dist, col_pie = st.columns([3, 1])
        with col_dist:
            st.subheader("Churn probability distribution")
            proba_vals = df_filtered["Churn Probability"].dropna()
            fig_dist = go.Figure()
            if viz_dist_type in ("Histogram", "Both"):
                fig_dist.add_trace(
                    go.Histogram(
                        x=proba_vals,
                        nbinsx=viz_hist_bins,
                        name="Count",
                        marker_color=primary,
                        opacity=0.85 if viz_dist_type == "Both" else 1,
                    )
                )
            if viz_dist_type in ("Density (KDE)", "Both"):
                try:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(proba_vals, bw_method=0.15)
                    x_line = np.linspace(max(0, proba_vals.min() - 0.02), min(1, proba_vals.max() + 0.02), 150)
                    y_density = kde(x_line)
                    if viz_dist_type == "Density (KDE)":
                        fig_dist.add_trace(
                            go.Scatter(x=x_line, y=y_density, name="Density", line=dict(color=primary, width=2.5), fill="tozeroy")
                        )
                        fig_dist.update_layout(yaxis_title="Density")
                    else:
                        scale = np.histogram(proba_vals, bins=viz_hist_bins)[0].max() / (y_density.max() or 1)
                        fig_dist.add_trace(
                            go.Scatter(x=x_line, y=y_density * scale, name="Density", line=dict(color=chart_text, width=2), dash="dash")
                        )
                except Exception:
                    if viz_dist_type == "Density (KDE)":
                        fig_dist.add_annotation(text="KDE unavailable (install scipy)", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig_dist.update_layout(
                height=320,
                showlegend=(viz_dist_type == "Both"),
                template="plotly_white",
                paper_bgcolor=chart_bg,
                plot_bgcolor=chart_plot_bg,
                xaxis_title="Churn probability",
                yaxis_title="Number of customers",
                margin=dict(t=24, b=48),
                xaxis=dict(gridcolor=chart_grid, zeroline=False),
                yaxis=dict(gridcolor=chart_grid, zeroline=False),
                font=dict(size=12, color=chart_text),
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        with col_pie:
            st.subheader("Risk mix")
            risk_counts = df_filtered["Risk Level"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).reset_index()
            risk_counts.columns = ["Risk Level", "Count"]
            if viz_risk_chart == "Bar":
                fig_pie = px.bar(
                    risk_counts,
                    x="Risk Level",
                    y="Count",
                    color="Risk Level",
                    color_discrete_map={"High": high_c, "Medium": medium_c, "Low": low_c},
                )
                fig_pie.update_layout(showlegend=False, xaxis_title="", yaxis_title="Customers")
            else:
                fig_pie = px.pie(
                    risk_counts,
                    values="Count",
                    names="Risk Level",
                    color="Risk Level",
                    color_discrete_map={"High": high_c, "Medium": medium_c, "Low": low_c},
                    hole=0.45 if viz_risk_chart == "Donut" else 0,
                )
                fig_pie.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="top", y=-0.05))
            fig_pie.update_layout(
                height=320,
                margin=dict(t=24, b=24),
                template="plotly_white",
                paper_bgcolor=chart_bg,
                plot_bgcolor=chart_plot_bg,
                font=dict(size=11, color=chart_text),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Feature importance — primary teal gradient — teal gradient, top N from sidebar
        if column_meta and hasattr(model, "feature_importances_"):
            st.subheader("What drives churn risk?")
            names = column_meta["feature_names"]
            imp = model.feature_importances_
            df_imp = pd.DataFrame({"Feature": [_format_feature_name(n) for n in names], "Importance": imp}).sort_values("Importance", ascending=True)
            n_show = min(viz_top_n, len(df_imp))
            fig_imp = px.bar(
                df_imp.tail(n_show),
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale=[primary_light, primary],
                range_color=[0, df_imp["Importance"].max()],
            )
            fig_imp.update_layout(
                height=max(300, n_show * 18),
                margin=dict(l=140),
                showlegend=False,
                template="plotly_white",
                paper_bgcolor=chart_bg,
                plot_bgcolor=chart_plot_bg,
                xaxis_title="Relative importance",
                yaxis_title="",
                margin_autoexpand=True,
                xaxis=dict(gridcolor=chart_grid, zeroline=False),
                yaxis=dict(categoryorder="total ascending"),
                font=dict(size=11, color=chart_text),
            )
            fig_imp.update_coloraxes(showscale=False)
            st.plotly_chart(fig_imp, use_container_width=True)

        # Extra charts (from sidebar multiselect)
        if viz_extra:
            st.subheader("Extra visualizations")
            extra_charts = []
            if "Risk by contract" in viz_extra and "Contract" in df_filtered.columns:
                rc = df_filtered.groupby(["Contract", "Risk Level"]).size().reset_index(name="Count")
                fig_rc = px.bar(
                    rc,
                    x="Contract",
                    y="Count",
                    color="Risk Level",
                    color_discrete_map={"High": high_c, "Medium": medium_c, "Low": low_c},
                    barmode="stack",
                    title="Risk by contract",
                )
                fig_rc.update_layout(template="plotly_white", paper_bgcolor=chart_bg, plot_bgcolor=chart_plot_bg, height=280, margin=dict(t=40), font=dict(size=10, color=chart_text))
                extra_charts.append(("Risk by contract", fig_rc))
            if "Tenure vs charges (scatter)" in viz_extra and "Tenure Months" in df_filtered.columns and "Monthly Charges" in df_filtered.columns:
                scatter_df = df_filtered.sample(n=min(1500, len(df_filtered)), random_state=42) if len(df_filtered) > 1500 else df_filtered
                fig_sc = px.scatter(
                    scatter_df,
                    x="Tenure Months",
                    y="Monthly Charges",
                    color="Risk Level",
                    color_discrete_map={"High": high_c, "Medium": medium_c, "Low": low_c},
                    opacity=0.6,
                    title="Tenure vs monthly charges",
                )
                fig_sc.update_layout(template="plotly_white", paper_bgcolor=chart_bg, plot_bgcolor=chart_plot_bg, height=280, margin=dict(t=40), font=dict(size=10, color=chart_text))
                extra_charts.append(("Tenure vs charges", fig_sc))
            if "Churn by contract (box)" in viz_extra and "Contract" in df_filtered.columns:
                fig_box = px.box(
                    df_filtered,
                    x="Contract",
                    y="Churn Probability",
                    color="Contract",
                    color_discrete_sequence=[primary, primary_light, PALETTE["primary"]],
                    title="Churn probability by contract",
                )
                fig_box.update_layout(template="plotly_white", paper_bgcolor=chart_bg, plot_bgcolor=chart_plot_bg, height=280, margin=dict(t=40), showlegend=False, font=dict(size=10, color=chart_text))
                extra_charts.append(("Churn by contract", fig_box))
            if extra_charts:
                n_extra = len(extra_charts)
                cols = st.columns(n_extra)
                for i, (_, fig) in enumerate(extra_charts):
                    with cols[i]:
                        st.plotly_chart(fig, use_container_width=True)

        # Why is this customer high risk? (Top_Drivers + bar chart)
        st.subheader("Why is this customer high risk?")
        has_top_drivers = "Top_Drivers" in df_filtered.columns and (df_filtered["Top_Drivers"].fillna("").astype(str).str.len() > 0).any()
        # All high-risk customers (SHAP is computed for every Risk Level == "High"); sort by probability descending
        high_risk_df = df_filtered[df_filtered["Risk Level"] == "High"].sort_values("Churn Probability", ascending=False)
        if not high_risk_df.empty:
            opts = []
            seen_labels = {}
            for pos, (_, row) in enumerate(high_risk_df.iterrows()):
                drivers = row.get("Top_Drivers")
                if pd.isna(drivers) or drivers is None or (isinstance(drivers, str) and not drivers.strip()) or str(drivers).strip().lower() == "nan":
                    continue
                drivers_str = str(drivers).strip()
                base_label = "Contract: {}, Tenure: {} mo, Charges: ${:.0f}".format(
                    row.get("Contract", ""),
                    row.get("Tenure Months", ""),
                    float(row.get("Monthly Charges", 0)),
                )
                seen_labels[base_label] = seen_labels.get(base_label, 0) + 1
                label = "{} (#{})".format(base_label, seen_labels[base_label]) if seen_labels[base_label] > 1 else base_label
                opts.append((label, drivers_str, row))
            if opts:
                choice = st.selectbox(
                    "Select a high-risk customer to see top drivers",
                    range(len(opts)),
                    format_func=lambda i: opts[i][0],
                    key="top_drivers_select",
                )
                drivers_text = opts[choice][1]
                sel_row = opts[choice][2]
                if drivers_text and drivers_text.lower() != "nan":
                    st.markdown("**Top drivers:** " + drivers_text)
                    parsed = _parse_top_drivers(drivers_text)
                    if parsed:
                        labels = [p[0][:30] + ("…" if len(p[0]) > 30 else "") for p in parsed]
                        impacts = [p[1] for p in parsed]
                        colors = [PALETTE["high_risk"] if v >= 0 else PALETTE["low_risk"] for v in impacts]
                        fig_d = go.Figure(go.Bar(x=impacts, y=labels, orientation="h", marker_color=colors))
                        fig_d.update_layout(
                            title="Driver impact on churn probability",
                            xaxis_title="SHAP impact",
                            height=max(200, len(labels) * 28),
                            margin=dict(l=120),
                            template="plotly_white",
                            paper_bgcolor=PALETTE["cards"],
                            plot_bgcolor=PALETTE["bg_main_solid"],
                        )
                        st.plotly_chart(fig_d, use_container_width=True)
                else:
                    st.info("No SHAP drivers for this customer. Run the full prediction pipeline (python -m src.predict) to enrich data, or use Single Customer Prediction below for an on-demand explanation.")
            else:
                st.info("No SHAP drivers available for high-risk customers in this view. Run the full pipeline (python -m src.predict) to compute Top_Drivers for all high-risk customers, or use Single Customer Prediction for an on-demand explanation.")
        else:
            st.caption("No high-risk customers in the current filter.")
        st.divider()

        # Table in expander
        st.subheader("Customer list")
        display_cols = [c for c in ["Tenure Months", "Contract", "Monthly Charges", "Churn Probability", "Risk Level", "Top_Drivers"] if c in df_filtered.columns]
        tbl = df_filtered[display_cols].head(500).copy()
        if "Churn Probability" in tbl.columns:
            tbl["Churn Probability"] = tbl["Churn Probability"].apply(lambda x: f"{x:.1%}")
        if "Monthly Charges" in tbl.columns:
            tbl["Monthly Charges"] = tbl["Monthly Charges"].apply(lambda x: f"${x:.2f}")
        if "Top_Drivers" in tbl.columns:
            tbl["Top_Drivers"] = tbl["Top_Drivers"].fillna("").astype(str).str.slice(0, 80).apply(lambda s: s + "…" if len(s) == 80 else s)
        with st.expander("View detailed records (first 500 rows)", expanded=False):
            st.dataframe(tbl, use_container_width=True, height=360)
        st.markdown("<p class='dashboard-footer'>Churn predictions are model outputs. Use for prioritization only; verify with business rules and CRM data.</p>", unsafe_allow_html=True)

    elif tab_choice == "ROI Simulator":
        st.subheader("Retention ROI Simulator")
        df_roi = load_predictions_data()
        if df_roi is None or df_roi.empty:
            st.warning("No predictions data. Run training and prediction first.")
        else:
            has_cltv = "CLTV" in df_roi.columns and df_roi["CLTV"].notna().any()
            use_dataset_cltv = False
            avg_cltv_input = 1200.0
            if has_cltv:
                use_dataset_cltv = st.checkbox("Use dataset CLTV (sum per segment)", value=False, key="roi_use_cltv")
                if use_dataset_cltv:
                    avg_cltv_input = float(pd.to_numeric(df_roi["CLTV"], errors="coerce").fillna(0).mean() or 1200)
                else:
                    avg_cltv_input = st.number_input("Avg CLTV ($)", min_value=0.0, value=1200.0, step=50.0, key="roi_cltv")
            else:
                avg_cltv_input = st.number_input("Avg CLTV ($)", min_value=0.0, value=1200.0, step=50.0, key="roi_cltv")
            save_rate_pct = st.slider("Expected save rate (%)", 0, 50, 25, key="roi_save_rate") / 100.0
            cost_per_offer = st.number_input("Cost per offer ($)", min_value=0.0, value=25.0, step=5.0, key="roi_cost")
            segment_roi = st.selectbox("Target segment", ["High", "Medium", "High+Medium"], index=0, key="roi_segment")
            cltv_col = "CLTV" if (use_dataset_cltv and has_cltv) else None
            rev_dict = compute_revenue_at_risk(df_roi, avg_cltv_input, segment=segment_roi, cltv_col=cltv_col)
            roi_dict = compute_roi(
                rev_dict["revenue_at_risk"],
                save_rate_pct,
                cost_per_offer,
                rev_dict["targeted_customers"],
            )
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Revenue at risk", "${:,.0f}".format(roi_dict["revenue_at_risk"]))
            with c2:
                st.metric("Revenue saved", "${:,.0f}".format(roi_dict["revenue_saved"]))
            with c3:
                st.metric("Offer cost", "${:,.0f}".format(roi_dict["offer_cost"]))
            with c4:
                st.metric("Net benefit", "${:,.0f}".format(roi_dict["net_benefit"]), delta_color="normal" if roi_dict["net_benefit"] >= 0 else "inverse")
            st.caption("Targeted {} customers in segment '{}'.".format(rev_dict["targeted_customers"], segment_roi))
            save_rates_pct = list(range(0, 51, 2))
            net_benefits = []
            for sr in save_rates_pct:
                r = compute_roi(rev_dict["revenue_at_risk"], sr / 100.0, cost_per_offer, rev_dict["targeted_customers"])
                net_benefits.append(r["net_benefit"])
            fig_roi = go.Figure()
            fig_roi.add_trace(go.Scatter(x=save_rates_pct, y=net_benefits, mode="lines+markers", name="Net benefit", line=dict(color=PALETTE["primary"], width=2)))
            fig_roi.update_layout(
                title="Net benefit vs save rate (0–50%)",
                xaxis_title="Save rate (%)",
                yaxis_title="Net benefit ($)",
                template="plotly_white",
                paper_bgcolor=PALETTE["cards"],
                plot_bgcolor=PALETTE["bg_main_solid"],
                height=360,
            )
            st.plotly_chart(fig_roi, use_container_width=True)

    elif tab_choice == "Upload Dataset":
        st.subheader("Upload Dataset")
        st.caption("Upload a CSV or Excel file with customer features. It should match the Telco churn schema (same column names).")
        uploaded = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
        if uploaded and has_model:
            try:
                if uploaded.name.endswith(".csv"):
                    up_df = pd.read_csv(uploaded)
                else:
                    up_df = pd.read_excel(uploaded)
                # Minimal prep: ensure Total Charges numeric, add tenure bucket if needed
                if "Total Charges" in up_df.columns:
                    up_df["Total Charges"] = pd.to_numeric(up_df["Total Charges"], errors="coerce")
                    up_df["Total Charges"] = up_df["Total Charges"].fillna(up_df.get("Tenure Months", 0) * up_df.get("Monthly Charges", 0)).fillna(0)
                if "Tenure Months" in up_df.columns and "Tenure Bucket" not in up_df.columns:
                    up_df["Tenure Bucket"] = pd.cut(
                        up_df["Tenure Months"],
                        bins=[-1, 12, 24, 48, 72, 1000],
                        labels=["0-12", "13-24", "25-48", "49-72", "73+"],
                    )
                cat_cols = column_meta["cat_cols"]
                num_cols = column_meta["num_cols"]
                up_df = ensure_columns(up_df, cat_cols, num_cols)
                proba = predict_proba(up_df, model=model, encoder=encoder, scaler=scaler, column_meta=column_meta)
                up_df = up_df.copy()
                up_df["Churn Probability"] = proba
                up_df["Risk Level"] = pd.cut(proba, bins=[-0.01, 0.33, 0.66, 1.01], labels=["Low", "Medium", "High"])
                st.success(f"Predictions for {len(up_df)} rows.")
                display_cols = [c for c in ["Tenure Months", "Contract", "Monthly Charges", "Churn Probability", "Risk Level"] if c in up_df.columns]
                st.dataframe(up_df[display_cols], use_container_width=True, height=400)
            except Exception as e:
                st.error(f"Error processing file: {e}")
        elif uploaded and not has_model:
            st.warning("Train a model first (run `python -m src.train`).")

    else:
        # Single customer prediction form
        st.subheader("Single Customer Prediction")
        if not has_model:
            st.warning("Train a model first (run `python -m src.train`).")
        else:
            cat_cols = column_meta["cat_cols"]
            num_cols = column_meta["num_cols"]
            # Build form from column_meta
            col1, col2 = st.columns(2)
            inputs = {}
            with col1:
                for c in num_cols:
                    if c in ["Tenure Months", "Monthly Charges", "Total Charges", "Avg Monthly Spend"]:
                        inputs[c] = st.number_input(c, min_value=0.0, value=12.0 if c == "Tenure Months" else 50.0, step=1.0 if c == "Tenure Months" else 0.5)
            with col2:
                for c in cat_cols:
                    if c == "Tenure Bucket":
                        inputs[c] = st.selectbox(c, ["0-12", "13-24", "25-48", "49-72", "73+"])
                    elif c == "Gender":
                        inputs[c] = st.selectbox(c, ["Male", "Female"])
                    elif c == "Contract":
                        inputs[c] = st.selectbox(c, ["Month-to-month", "One year", "Two year"])
                    elif c == "Internet Service":
                        inputs[c] = st.selectbox(c, ["DSL", "Fiber optic", "No"])
                    elif c in ["Partner", "Dependents", "Phone Service", "Multiple Lines", "Online Security", "Online Backup", "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies", "Paperless Billing"]:
                        inputs[c] = st.selectbox(c, ["Yes", "No"])
                    elif c == "Senior Citizen":
                        inputs[c] = st.selectbox(c, ["Yes", "No"])
                    elif c == "Payment Method":
                        inputs[c] = st.selectbox(c, ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
                    else:
                        inputs[c] = st.text_input(c, value="No")
            for c in num_cols:
                if c not in inputs:
                    inputs[c] = 0.0
            if st.button("Predict Churn Probability"):
                row = pd.DataFrame([inputs])
                proba = predict_proba(row, model=model, encoder=encoder, scaler=scaler, column_meta=column_meta)[0]
                risk = "High" if proba >= 0.66 else ("Medium" if proba >= 0.33 else "Low")
                st.metric("Churn Probability", f"{proba:.1%}")
                st.metric("Risk Level", risk)
                expl = _load_shap_explainer()
                if expl is not None:
                    try:
                        from src.shap_utils import compute_shap_values, get_top_drivers_per_row
                        from src.train import build_features
                        explainer, is_tree, feature_names, col_meta, mod, enc, scl = expl
                        row_enc = ensure_columns(row, col_meta["cat_cols"], col_meta["num_cols"])
                        X_one, _, _, _, _, _ = build_features(
                            row_enc, encoder=enc, scaler=scl, fit=False,
                            cat_cols_order=col_meta["cat_cols"], num_cols_order=col_meta["num_cols"],
                        )
                        shap_one = compute_shap_values(explainer, X_one, is_tree=is_tree)
                        drivers_list = get_top_drivers_per_row(shap_one, feature_names, row_enc, col_meta, top_k=5)
                        drivers_text = drivers_list[0] if drivers_list else ""
                        if drivers_text:
                            st.markdown("**Why this risk?** " + drivers_text)
                            parsed = _parse_top_drivers(drivers_text)
                            if parsed:
                                labels = [p[0][:30] + ("…" if len(p[0]) > 30 else "") for p in parsed]
                                impacts = [p[1] for p in parsed]
                                colors = [PALETTE["high_risk"] if v >= 0 else PALETTE["low_risk"] for v in impacts]
                                fig_one = go.Figure(go.Bar(x=impacts, y=labels, orientation="h", marker_color=colors))
                                fig_one.update_layout(
                                    title="Driver impact (on-demand SHAP)",
                                    xaxis_title="SHAP impact",
                                    height=max(180, len(labels) * 28),
                                    margin=dict(l=120),
                                    template="plotly_white",
                                    paper_bgcolor=PALETTE["cards"],
                                    plot_bgcolor=PALETTE["bg_main_solid"],
                                )
                                st.plotly_chart(fig_one, use_container_width=True)
                    except Exception as e:
                        st.caption("On-demand SHAP could not be computed: {}.".format(e))
                else:
                    st.caption("SHAP explainer not available. Run the full pipeline to enable per-customer explanations.")


if __name__ == "__main__":
    main()
