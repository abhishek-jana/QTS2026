import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from research_lab.alpha_universe import AlphaUniverse
from research_lab.plugins.core_plugins import SequentialPlugin, SpatialPlugin

st.set_page_config(page_title="UQTS-2026 Alpha Dashboard", layout="wide")

st.title("🚀 UQTS-2026: Alpha Discovery Dashboard")
st.markdown("### Interactive Multi-Resolution Signal Analysis (Signal vs. Fluid)")

# 1. Sidebar Configuration
st.sidebar.header("Universe Settings")
tickers = st.sidebar.multiselect("Select Tickers", ["AAPL", "MSFT", "GOOG", "SPY"], default=["AAPL", "SPY"])
lookback = st.sidebar.slider("Lookback Window (T)", 21, 252, 63)
as_of_date = st.sidebar.date_input("Knowledge Date (PIT)", datetime(2020, 9, 7))

# 2. Lab Initialization
@st.cache_resource
def get_lab():
    plugins = [SequentialPlugin(d_param=0.4), SpatialPlugin()]
    return AlphaUniverse(plugins=plugins)

lab = get_lab()
lab.engine.generate_synthetic_pit_data(tickers, days=500)

# 3. Fetch Snapshot
as_of_dt = datetime.combine(as_of_date, datetime.min.time())
batch = lab.snapshot(as_of=as_of_dt, tickers=tickers, lookback=lookback)

if batch is None:
    st.error("No bi-temporally aligned data found for this configuration.")
else:
    # 4. Global Stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Aligned Samples", len(batch.labels))
    col2.metric("Modalities", len(batch.data))
    col3.metric("Latest Event Time", batch.times[-1].strftime("%Y-%m-%d"))

    # 5. Ticker Investigation
    st.divider()
    selected_ticker = st.selectbox("Select Ticker to Investigate", list(set(batch.tickers)))
    
    # Filter batch for selected ticker
    idx = [i for i, t in enumerate(batch.tickers) if t == selected_ticker]
    
    if idx:
        t_idx = idx[-1] # Show latest snapshot for this ticker
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Sequential Stream (LSTM Input)")
            seq_data = batch.data['x_seq'][t_idx].squeeze().numpy()
            fig_seq = px.line(seq_data, title=f"{selected_ticker} Fractionally Differenced Returns")
            fig_seq.update_layout(xaxis_title="Lookback Steps", yaxis_title="Stationary Signal")
            st.plotly_chart(fig_seq, use_container_width=True)

        with col_right:
            st.subheader("Spatial Stream (ViT Input)")
            spatial_data = batch.data['x_spatial'][t_idx].squeeze().numpy()
            fig_spec = px.imshow(spatial_data, color_continuous_scale='Jet', title=f"{selected_ticker} Market Spectrogram")
            fig_spec.update_layout(xaxis_title="Lookback Steps", yaxis_title="Log Scales (2^1 - 2^8)")
            st.plotly_chart(fig_spec, use_container_width=True)

    # 6. Cross-Sectional Alpha
    st.divider()
    st.subheader("Cross-Sectional Alpha Rankings")
    
    # Display table of latest scores (mock inference or labels)
    results = pd.DataFrame({
        'Ticker': batch.tickers,
        'Time': batch.times,
        'Z-Scored Alpha (Label)': batch.labels.numpy()
    }).sort_values('Time', ascending=False)
    
    st.dataframe(results.head(20), use_container_width=True)
    
    # Distribution Plot
    fig_dist = px.histogram(results, x="Z-Scored Alpha (Label)", nbins=50, title="Alpha Label Distribution (Z-Scores)")
    st.plotly_chart(fig_dist, use_container_width=True)

st.sidebar.info("UQTS-2026: Signal vs. Fluid Logic Engaged.")
