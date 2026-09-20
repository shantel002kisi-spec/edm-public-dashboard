from pathlib import Path

APP = Path("app.py")
MARKER = "SIDEBAR_TOGGLE_VISIBILITY_FIX_V1"

text = APP.read_text(encoding="utf-8")
if MARKER in text:
    print("Sidebar toggle fix already present.")
    raise SystemExit(0)

anchor = "\nPAGES = ["
if anchor not in text:
    raise SystemExit("Could not find PAGES anchor in app.py")

block = r'''
# SIDEBAR_TOGGLE_VISIBILITY_FIX_V1
# Keep the sidebar open/close control available at full-screen desktop widths
# while continuing to hide Streamlit's unrelated top-right controls.
st.markdown(
    """
    <style>
    /* Restore only the header layer needed for Streamlit's sidebar toggle. */
    header[data-testid="stHeader"] {
        display: block !important;
        visibility: visible !important;
        background: transparent !important;
        box-shadow: none !important;
        pointer-events: none !important;
    }

    /* Always expose the hamburger/open-sidebar control when the sidebar is collapsed. */
    [data-testid="stSidebarCollapsedControl"],
    header[data-testid="stHeader"] [data-testid="stSidebarCollapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        position: fixed !important;
        top: 0.72rem !important;
        left: 0.72rem !important;
        z-index: 1000000 !important;
    }

    [data-testid="stSidebarCollapsedControl"] button,
    header[data-testid="stHeader"] [data-testid="stSidebarCollapsedControl"] button {
        display: inline-flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        background: #ffffff !important;
        border: 1px solid rgba(49, 51, 63, 0.18) !important;
        border-radius: 14px !important;
        min-width: 46px !important;
        min-height: 46px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.10) !important;
    }

    /* Keep the user's requested Streamlit top controls hidden. */
    header[data-testid="stHeader"] [data-testid="stToolbar"],
    header[data-testid="stHeader"] [data-testid="stStatusWidget"],
    header[data-testid="stHeader"] [data-testid="stMainMenu"],
    header[data-testid="stHeader"] [data-testid="stDecoration"] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }

    /* Extra desktop safeguard: do not let wide layouts suppress the toggle. */
    @media (min-width: 769px) {
        [data-testid="stSidebarCollapsedControl"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
'''
text = text.replace(anchor, "\n" + block + anchor, 1)
APP.write_text(text, encoding="utf-8")
print("Inserted full-screen sidebar toggle visibility fix.")
