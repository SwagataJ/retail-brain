"""
Retail Brain AI Copilot — Streamlit web interface.

Reuses all backend logic (data loading, aggregations, system prompt,
tool definitions, tool execution, Bedrock API calls, chart rendering)
from ai_copilot.py. This module only handles the Streamlit UI layer.

Usage:
    streamlit run scripts/ai_copilot_app.py
"""

import json
import os
from datetime import datetime

import streamlit as st

from ai_copilot import (
    AWS_REGION,
    CHART_DIR,
    CSV_MANIFEST,
    EXPORT_DIR,
    MODEL_ID,
    build_system_prompt,
    build_tool_definitions,
    call_claude,
    compute_aggregations,
    create_bedrock_client,
    extract_chart_directives,
    load_all_data,
    render_all_charts,
    save_history,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Retail Brain AI Copilot",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

def init_session():
    """Load data, build prompt/tools, and create Bedrock client once."""
    if "initialized" in st.session_state:
        return

    with st.spinner("Loading datasets..."):
        data = load_all_data()
        if not data:
            st.error("No data files found. Run the pipelines first.")
            st.stop()
        st.session_state.data = data

    with st.spinner("Computing aggregations..."):
        aggregations = compute_aggregations(data)
        st.session_state.aggregations = aggregations

    st.session_state.system_prompt = build_system_prompt(data, aggregations)
    st.session_state.tools = build_tool_definitions()

    with st.spinner("Connecting to AWS Bedrock..."):
        try:
            st.session_state.client = create_bedrock_client()
        except Exception as exc:
            st.error(f"Could not create Bedrock client: {exc}")
            st.stop()

    st.session_state.conversation = []  # Bedrock message list
    st.session_state.chat_display = []  # UI display list
    st.session_state.initialized = True


init_session()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Retail Brain AI Copilot")
    st.markdown("---")

    # Dataset summary table
    st.subheader("Datasets")
    rows = []
    for name, df in sorted(st.session_state.data.items()):
        subdir, fname = CSV_MANIFEST[name]
        rows.append({"Dataset": name, "Rows": f"{len(df):,}", "Source": f"{subdir}/{fname}"})
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Clear history button
    if st.button("Clear History", use_container_width=True):
        st.session_state.conversation = []
        st.session_state.chat_display = []
        st.rerun()

    # Export chat button
    if st.button("Export Chat", use_container_width=True):
        if st.session_state.conversation:
            os.makedirs(EXPORT_DIR, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(EXPORT_DIR, f"conversation_{ts}.txt")
            with open(path, "w") as f:
                for msg in st.session_state.conversation:
                    role = msg["role"]
                    for block in msg["content"]:
                        if "text" in block:
                            f.write(f"[{role}] {block['text']}\n\n")
                        elif "toolUse" in block:
                            tu = block["toolUse"]
                            f.write(f"[tool call] {tu['name']}({json.dumps(tu['input'])})\n\n")
                        elif "toolResult" in block:
                            f.write("[tool result] ...\n\n")
            st.success(f"Exported to {path}")
        else:
            st.warning("No conversation to export.")

    st.markdown("---")
    st.caption(f"**Model:** {MODEL_ID.split('.')[-1].split('-v')[0]}")
    st.caption(f"**Region:** {AWS_REGION}")

# ---------------------------------------------------------------------------
# Chat display
# ---------------------------------------------------------------------------

# Welcome message
if not st.session_state.chat_display:
    with st.chat_message("assistant"):
        st.markdown(
            "Welcome! I'm the **Retail Brain AI Copilot**. Ask me about "
            "forecasting, customer intelligence, or pricing optimization. "
            "I can also generate charts to visualize the data."
        )

# Render chat history
for entry in st.session_state.chat_display:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        for chart_path in entry.get("charts", []):
            if os.path.exists(chart_path):
                st.image(chart_path)

# ---------------------------------------------------------------------------
# Chat input handling
# ---------------------------------------------------------------------------

if user_input := st.chat_input("Ask about your data..."):
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_display.append(
        {"role": "user", "content": user_input, "charts": []}
    )

    # Call Bedrock
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = call_claude(
                    st.session_state.client,
                    st.session_state.system_prompt,
                    st.session_state.tools,
                    st.session_state.conversation,
                    user_input,
                    st.session_state.data,
                )
            except Exception as exc:
                st.error(f"Error calling Bedrock: {exc}")
                response = None

        if response:
            # Extract charts and clean text
            clean_text, chart_specs = extract_chart_directives(response)
            st.markdown(clean_text)

            # Render and display charts inline
            chart_paths = []
            if chart_specs:
                chart_paths = render_all_charts(chart_specs)
                for p in chart_paths:
                    st.image(p)

            # Persist to display history
            st.session_state.chat_display.append(
                {"role": "assistant", "content": clean_text, "charts": chart_paths}
            )

            # Save conversation to disk
            save_history(st.session_state.conversation)
