import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px

# Configure page
st.set_page_config(
    page_title="OptiBatch Industrial AI Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Base URL
API_URL = "http://localhost:8000"

# Main styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e2f;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        color: white;
    }
    .status-normal { color: #2ecc71; font-weight: bold; }
    .status-warning { color: #f39c12; font-weight: bold; }
    .status-critical { color: #e74c3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🏭 OptiBatch Industrial AI Dashboard")
st.markdown("Predictive and Prescriptive Analytics for Batch Optimization")
st.divider()

# Sidebar Inputs
st.sidebar.header("Batch Parameters")
st.sidebar.markdown("Enter current running parameters:")

with st.sidebar.form("batch_params_form"):
    temperature = st.number_input("Temperature (°C)", min_value=150.0, max_value=240.0, value=182.0, step=1.0)
    pressure = st.number_input("Pressure (bar)", min_value=1.0, max_value=10.0, value=2.15, step=0.1)
    hold_time = st.number_input("Hold Time (mins)", min_value=10.0, max_value=120.0, value=47.0, step=1.0)
    catalyst_ratio = st.number_input("Catalyst Ratio", min_value=0.5, max_value=5.0, value=1.2, step=0.1)
    reaction_time = st.number_input("Reaction Time (mins)", min_value=30.0, max_value=300.0, value=120.0, step=5.0)
    mixing_speed = st.number_input("Mixing Speed (RPM)", min_value=100.0, max_value=1000.0, value=300.0, step=10.0)
    
    submit_btn = st.form_submit_button("Update Global Parameters")

batch_parameters = {
    "temperature": temperature,
    "pressure": pressure,
    "hold_time": hold_time,
    "catalyst_ratio": catalyst_ratio,
    "reaction_time": reaction_time,
    "mixing_speed": mixing_speed
}

# --- Layout ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Prediction", 
    "📈 Optimization", 
    "🚨 Monitoring", 
    "🧪 Simulation", 
    "🔍 Parameter Sweep"
])

def call_api(endpoint, payload):
    try:
        response = requests.post(f"{API_URL}{endpoint}", json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Connection Error: {e}")
        return None

# --- 1. Batch Prediction Panel ---
with tab1:
    st.header("Batch Prediction")
    if st.button("Predict Batch Performance", use_container_width=True):
        with st.spinner("Predicting..."):
            res = call_api("/predict", batch_parameters)
            if res:
                metrics = res.get("predicted_metrics", {})
                perf_class = res.get("performance_class", "unknown").title()
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Predicted Yield", f"{metrics.get('yield', 0):.4f}")
                col2.metric("Predicted Quality", f"{metrics.get('quality', 0):.2f}")
                col3.metric("Predicted Energy", f"{metrics.get('energy', 0):.2f}")
                col4.metric("Predicted Performance", f"{metrics.get('performance', 0):.3f}")
                
                st.subheader(f"Overall Performance Class: **{perf_class}**")

# --- 2. Optimization Panel ---
with tab2:
    st.header("Prescriptive Optimization")
    if st.button("Generate Optimization Strategy", use_container_width=True):
        with st.spinner("Analyzing parameters..."):
            # Optimization expects predicted metrics too, so we chain a predict call
            pred_res = call_api("/predict", batch_parameters)
            if pred_res:
                pred_metrics = {f"predicted_{k}": v for k, v in pred_res.get("predicted_metrics", {}).items()}
                opt_payload = {
                    "batch_parameters": batch_parameters,
                    "predicted_metrics": pred_metrics
                }
                opt_res = call_api("/optimize", opt_payload)
                if opt_res:
                    st.subheader(f"Target Signature: {opt_res.get('target_signature', '').replace('_', ' ').title()}")
                    
                    st.progress(opt_res.get("optimization_confidence", 0.0))
                    st.caption(f"AI Confidence Score: {opt_res.get('optimization_confidence', 0.0)}")
                    
                    colA, colB = st.columns(2)
                    with colA:
                        st.markdown("### 🔧 Parameter Recommendations")
                        for param, adj in opt_res.get("parameter_recommendations", {}).items():
                            st.info(f"**{param.replace('_', ' ').title()}**: {adj}")
                            
                    with colB:
                        st.markdown("### 📈 Expected Improvements")
                        for metric, imp in opt_res.get("expected_metric_improvement", {}).items():
                            st.success(f"**{metric.replace('_', ' ').title()}**: {imp}")

# --- 3. Monitoring Panel ---
with tab3:
    st.header("Real-Time Drift Monitoring")
    if st.button("Run Live Monitor", use_container_width=True):
        with st.spinner("Running diagnostics..."):
            mon_res = call_api("/monitor", {"batch_parameters": batch_parameters})
            if mon_res:
                status = mon_res.get("batch_status", "unknown").lower()
                
                status_color = "status-normal"
                icon = "✅"
                if status == "warning":
                    status_color = "status-warning"
                    icon = "⚠️"
                elif status == "critical":
                    status_color = "status-critical"
                    icon = "🚨"
                    
                st.markdown(f"### Status: <span class='{status_color}'>{status.upper()} {icon}</span>", unsafe_allow_html=True)
                st.warning(f"**Message:** {mon_res.get('alert_message', '')}")
                st.info(f"**Action Required:** {mon_res.get('recommended_action', '')}")
                
                col1, col2 = st.columns(2)
                col1.metric("Drift Score", f"{mon_res.get('drift_score', 0):.2f}")
                col2.metric("Closest Signature Tracked", mon_res.get('closest_signature', '').replace('_', ' ').title())

# --- 4. Simulation Panel ---
with tab4:
    st.header("Hypothetical Batch Simulator")
    st.markdown("Use this to safely test what-if scenarios isolated from the main live tracker.")
    if st.button("Run Isolated Simulation", use_container_width=True):
        with st.spinner("Simulating environment..."):
            sim_res = call_api("/simulate", {"batch_parameters": batch_parameters})
            if sim_res:
                st.json(sim_res)

# --- 5. Parameter Sweep Explorer ---
with tab5:
    st.header("Monte Carlo Parameter Sweep")
    st.markdown("Discover the absolute optimal frontier mathematically.")
    
    with st.form("sweep_form"):
        col1, col2 = st.columns(2)
        with col1:
            sweep_temp = st.slider("Temperature Range", 150.0, 240.0, (180.0, 190.0))
            sweep_press = st.slider("Pressure Range", 1.0, 10.0, (2.0, 3.0))
        with col2:
            sweep_hold = st.slider("Hold Time Range", 10.0, 120.0, (40.0, 60.0))
            sims = st.number_input("Number of Simulations", min_value=10, max_value=1000, value=50, step=10)
            
        run_sweep = st.form_submit_button("Run Parameter Sweep Engine")
        
    if run_sweep:
        payload = {
            "parameter_ranges": {
                "temperature": list(sweep_temp),
                "pressure": list(sweep_press),
                "hold_time": list(sweep_hold)
            },
            "num_simulations": sims
        }
        
        with st.spinner(f"Running {sims} Monte Carlo simulations..."):
            sweep_res = call_api("/sweep", payload)
            if sweep_res:
                df = pd.DataFrame(sweep_res.get("best_simulated_batches", []))
                
                if not df.empty:
                    st.subheader("Top Performing Configurations")
                    st.dataframe(df.style.highlight_max(axis=0, subset=['yield', 'quality']))
                    
                    st.subheader("Visualizations")
                    c1, c2 = st.columns(2)
                    
                    if 'yield' in df.columns and 'energy' in df.columns:
                        fig1 = px.scatter(df, x='energy', y='yield', color='yield', 
                                          title="Yield Vs Energy Envelope",
                                          color_continuous_scale="Viridis")
                        c1.plotly_chart(fig1, use_container_width=True)
                        
                    if 'quality' in df.columns and 'performance' in df.columns:
                        fig2 = px.scatter(df, x='performance', y='quality', size='yield',
                                          title="Quality Vs Performance (Bubble=Yield)",
                                          color_continuous_scale="Plasma")
                        c2.plotly_chart(fig2, use_container_width=True)
