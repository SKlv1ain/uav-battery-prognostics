"""Streamlit dashboard — UAV Battery Prognostics demo UI."""

import json

import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(
    page_title="UAV Battery Prognostics",
    page_icon="🔋",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("🔋 Battery Prognostics")
api_url = st.sidebar.text_input("API Base URL", value="http://localhost:8000")

battery_label = st.sidebar.selectbox(
    "Select Battery",
    ["B0005", "B0006", "B0007", "B0018"],
)
battery_id = int(battery_label.replace("B00", ""))

pipeline_choice = st.sidebar.radio(
    "Pipeline",
    ["Auto (Recommended)", "Single-Cycle", "Multi-Cycle"],
)
window_size = 15
if pipeline_choice == "Auto (Recommended)":
    window_size = st.sidebar.slider("Auto-route threshold (cycles)", 5, 30, 15)

run_btn = st.sidebar.button("▶ Run Prediction", use_container_width=True)

# ── Main ─────────────────────────────────────────────────────────────────────
st.title("UAV Battery State-of-Health & RUL Prediction")

if not run_btn:
    st.info("Configure a battery and pipeline in the sidebar, then press **Run Prediction**.")
    st.stop()

# Build request
if pipeline_choice == "Auto (Recommended)":
    endpoint = f"{api_url}/api/predict/auto"
    payload = {"battery_id": battery_id, "window_size": window_size}
elif pipeline_choice == "Single-Cycle":
    endpoint = f"{api_url}/api/predict/single-cycle"
    payload = {"battery_id": battery_id}
else:
    endpoint = f"{api_url}/api/predict/multi-cycle"
    payload = {"battery_id": battery_id}

# ── API Call ──────────────────────────────────────────────────────────────────
with st.spinner("Calling API and running inference…"):
    try:
        resp = requests.post(endpoint, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at `{api_url}`. Is the backend running?")
        st.stop()
    except requests.exceptions.HTTPError as e:
        st.error(f"API error {resp.status_code}: {resp.text}")
        st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_api, tab_cap, tab_soh = st.tabs(["📡 API Request / Response", "📈 Capacity Forecast", "💚 SOH & RUL"])

# ── Tab 1: Raw JSON ───────────────────────────────────────────────────────────
with tab_api:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Request")
        st.markdown(f"`POST {endpoint}`")
        st.code(json.dumps(payload, indent=2), language="json")

    with col2:
        st.subheader("Response")
        # Truncate long arrays for readability
        display_data = {
            k: (v[:5] if isinstance(v, list) and len(v) > 5 else v)
            for k, v in data.items()
            if k not in ("predictions", "soh", "rul")
        }
        display_data["predictions"] = {
            m: vals[:5] for m, vals in data["predictions"].items()
        }
        display_data["soh"] = {
            m: vals[:5] for m, vals in data["soh"].items()
        }
        display_data["rul"] = data["rul"]
        display_data["_note"] = "Arrays truncated to first 5 values for display"
        st.code(json.dumps(display_data, indent=2), language="json")

    st.caption(
        f"Pipeline used: **{data['pipeline']}** | "
        f"Battery: **{data['battery_id']}** | "
        f"Cycles: **{data['n_cycles']}**"
    )

# ── Tab 2: Capacity Chart ─────────────────────────────────────────────────────
with tab_cap:
    st.subheader(f"Capacity Forecast — {data['battery_id']} ({data['pipeline']})")
    cycles = data["cycle_nums"]
    actual = data["actual_capacity"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cycles, y=actual,
        mode="lines", name="Actual",
        line=dict(color="black", width=2, dash="dash"),
    ))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for (model, preds), color in zip(data["predictions"].items(), colors):
        fig.add_trace(go.Scatter(
            x=cycles, y=preds,
            mode="lines", name=model,
            line=dict(color=color, width=1.5),
        ))

    fig.add_hline(y=1.4, line_dash="dot", line_color="red",
                  annotation_text="EOL threshold (1.4 Ahr)", annotation_position="bottom right")

    fig.update_layout(
        xaxis_title="Cycle Number",
        yaxis_title="Capacity (Ahr)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 3: SOH & RUL ─────────────────────────────────────────────────────────
with tab_soh:
    st.subheader(f"State of Health — {data['battery_id']}")

    fig2 = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for (model, soh_vals), color in zip(data["soh"].items(), colors):
        fig2.add_trace(go.Scatter(
            x=cycles, y=soh_vals,
            mode="lines", name=model,
            line=dict(color=color, width=1.5),
        ))

    fig2.add_hline(y=70, line_dash="dot", line_color="red",
                   annotation_text="EOL threshold (70% SOH)", annotation_position="bottom right")

    fig2.update_layout(
        xaxis_title="Cycle Number",
        yaxis_title="SOH (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Remaining Useful Life Summary")
    rul_rows = []
    for model, rul in data["rul"].items():
        rul_rows.append({
            "Model": model,
            "SOH Now (%)": f"{rul['soh_now']:.1f}%",
            "RUL (cycles)": rul["rul_cycles"] if rul["rul_cycles"] is not None else "Not reached",
            "EOL Index": rul["eol_index"] if rul["eol_index"] is not None else "—",
            "Note": rul["note"],
        })
    st.table(rul_rows)
