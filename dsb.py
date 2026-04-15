import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# ============================================================
# 🎨 UI & THEMING
# ============================================================
st.set_page_config(page_title="FuelIntel Pro", layout="wide", page_icon="⛽")

# Apply a "Terminal/Stock Market" aesthetic
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    [data-testid="stMetricValue"] { font-size: 32px; color: #00ffcc !important; }
    .stSlider [data-baseweb="slider"] { color: #00ffcc; }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# 📂 ROBUST DATA LOADING (Fixes the Empty Array Error)
# ============================================================
@st.cache_data
def load_clean_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
    
    # Use sep="\s+" to handle variable whitespace and na_values='?' for missing data
    df = pd.read_csv(url, names=columns, na_values='?', comment='\t', sep="\s+", skipinitialspace=True)
    
    # Clean the data: Drop rows with missing horsepower
    df = df.dropna().reset_index(drop=True)
    return df

try:
    df = load_clean_data()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# ============================================================
# 🧠 MACHINE LEARNING ENGINE
# ============================================================
@st.cache_resource
def train_prediction_model(data):
    features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']
    X = data[features]
    y = data['mpg']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_prediction_model(df)

# ============================================================
# 🕹️ SIDEBAR CONTROL PANEL
# ============================================================
st.sidebar.header("🛠️ Vehicle Configuration")
st.sidebar.write("Modify specs to simulate efficiency.")

# Dynamic inputs based on dataset ranges
in_weight = st.sidebar.slider("Weight (lbs)", int(df.weight.min()), int(df.weight.max()), 2800)
in_hp = st.sidebar.slider("Horsepower", int(df.horsepower.min()), int(df.horsepower.max()), 100)
in_disp = st.sidebar.slider("Displacement", int(df.displacement.min()), int(df.displacement.max()), 150)
in_cyl = st.sidebar.selectbox("Cylinders", sorted(df.cylinders.unique()), index=1)
in_acc = st.sidebar.slider("Acceleration (0-60mph)", 8.0, 25.0, 15.0)
in_year = st.sidebar.slider("Model Year (19xx)", 70, 82, 78)

# ============================================================
# 📊 DASHBOARD MAIN VIEW
# ============================================================
st.title("⛽ Fuel Efficiency Intelligence")
st.markdown(f"**Dataset status:** {len(df)} vehicles indexed | **Model:** Random Forest Regressor")

# --- ROW 1: KEY PERFORMANCE INDICATORS ---
col1, col2, col3, col4 = st.columns(4)

# Run Prediction
input_features = np.array([[in_cyl, in_disp, in_hp, in_weight, in_acc, in_year]])
prediction = model.predict(input_features)[0]
avg_mpg = df['mpg'].mean()

col1.metric("PREDICTED MPG", f"{prediction:.1f}", f"{prediction - avg_mpg:+.1f} vs Avg")
col2.metric("AVG FLEET MPG", f"{avg_mpg:.1f}")
col3.metric("EFFICIENCY PEAK", f"{df.mpg.max():.1f}")
col4.metric("ENGINE CORRELATION", "-0.78", "Negative")

st.divider()

# --- ROW 2: INTERACTIVE ANALYTICS ---
chart_col, corr_col = st.columns([2, 1])

with chart_col:
    st.subheader("🏎️ Horsepower vs. Fuel Consumption")
    fig = px.scatter(df, x="horsepower", y="mpg", color="weight",
                     size="displacement", hover_name="car_name",
                     template="plotly_dark", color_continuous_scale="Viridis")
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

with corr_col:
    st.subheader("🧬 Feature Importance")
    # Calculating simple correlation for the heatmap
    corr = df[['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale='RdBu_r', template="plotly_dark")
    fig_corr.update_layout(height=400)
    st.plotly_chart(fig_corr, use_container_width=True)

# --- ROW 3: TIME SERIES TREND ---
st.subheader("📅 Technological Progress (MPG over Years)")
yearly_data = df.groupby('model_year')['mpg'].mean().reset_index()
fig_line = px.line(yearly_data, x="model_year", y="mpg", template="plotly_dark")
fig_line.update_traces(line_color="#00ffcc", line_width=4)
fig_line.add_hrect(y0=avg_mpg, y1=avg_mpg+0.1, line_width=0, fillcolor="white", opacity=0.2)
st.plotly_chart(fig_line, use_container_width=True)

# --- ROW 4: DATA EXPLORER ---
with st.expander("📂 View Source Fleet Data"):
    st.dataframe(df.sort_values('mpg', ascending=False), use_container_width=True)