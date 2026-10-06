"""ProcessLens dashboard entry point: `make app`."""

import streamlit as st

st.set_page_config(
    page_title="ProcessLens", page_icon=":material/precision_manufacturing:", layout="wide"
)

page = st.navigation(
    [
        st.Page("app_pages/overview.py", title="Overview", icon=":material/dashboard:"),
        st.Page("app_pages/model.py", title="Model & limits", icon=":material/insights:"),
        st.Page(
            "app_pages/rootcause.py", title="Root-cause explorer", icon=":material/troubleshoot:"
        ),
        st.Page("app_pages/benchmark.py", title="Benchmark", icon=":material/verified:"),
        st.Page("app_pages/copilot.py", title="Copilot", icon=":material/smart_toy:"),
    ],
    position="top",
)
st.title(page.title, anchor=False)
st.caption(
    "Suspects are sensor clusters *associated with* failure in observational data. "
    "They are not proven causes."
)
page.run()
