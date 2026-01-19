import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, EasterMonday, GoodFriday
from scipy.interpolate import CubicSpline
from typing import Dict, Any

# --- 1. GLOBAL SETTINGS ---
st.set_page_config(
    page_title="PPA Quant Engine | v12 Bankable",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {background-color: #f8fafc;}
    h1, h2, h3 {font-family: 'Inter', sans-serif; color: #0f172a;}
    .stMetric {background-color: white; border: 1px solid #cbd5e1; border-radius: 8px; padding: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);}
    .stDataFrame {border: 1px solid #cbd5e1; border-radius: 8px;}
    </style>
""", unsafe_allow_html=True)

# --- 2. PHYSICS & MATH LIBRARY ---

class GermanHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('New Years Day', month=1, day=1),  # Manuell definiert statt importiert
        GoodFriday, 
        EasterMonday,
        Holiday('Labor Day', month=5, day=1),
        Holiday('German Unity Day', month=10, day=3),
        Holiday('Christmas', month=12, day=25)     # Manuell definiert statt importiert
    ]
class MarketPhysics:
    def __init__(self, start_date, n_days, latitude):
        self.n_hours = int(n_days * 24)
        self.start_date = pd.to_datetime(start_date)
        self.latitude = np.radians(latitude)
        
    def get_time_index(self):
        return pd.date_range(start=self.start_date, periods=self.n_hours, freq='h')

    def compute_solar_profile(self, time_index):
        # Precise Astronomy
        doy = time_index.dayofyear.to_numpy()
        hod = time_index.hour.to_numpy() + (time_index.minute.to_numpy()/60)
        
        declination = 0.409 * np.sin((2*np.pi/365)*doy - 1.39)
        tan_lat_dec = np.clip(-np.tan(self.latitude) * np.tan(declination), -1, 1)
        sunset_angle = np.arccos(tan_lat_dec)
        sunrise, sunset = 12 - (sunset_angle*12/np.pi), 12 + (sunset_angle*12/np.pi)
        
        is_day = (hod > sunrise) & (hod < sunset)
        day_len = np.maximum(sunset - sunrise, 0.1)
        
        solar_shape = np.zeros_like(hod)
        solar_shape[is_day] = np.sin(np.pi * (hod[is_day] - sunrise[is_day]) / day_len[is_day])
        
        # Zenith Intensity
        zenith = np.arccos(np.sin(self.latitude)*np.sin(declination) + np.cos(self.latitude)*np.cos(declination)*np.cos(np.pi*(hod-12)/12))
        solar_shape *= np.clip(np.cos(zenith), 0, 1)
        
        # Demand & Holidays
        cal = GermanHolidayCalendar()
        is_off = (time_index.dayofweek >= 5) | time_index.normalize().isin(cal.holidays(start=time_index.min(), end=time_index.max()))
        demand = (1.0 + 0.15*np.sin((hod-8)*np.pi/12)) * np.where(is_off, 0.85, 1.0) * (1.0 + 0.15*np.cos(2*np.pi*(doy+10)/365))
        
        return solar_shape, demand

class StochasticEngine:
    def __init__(self, params):
        self.p = params
        np.random.seed(params['seed'])

    def _spline_interpolate(self, monthly_values, time_index):
        # High-precision interpolation using Day of Year
        x_months = np.array([15, 45, 75, 105, 135, 166, 196, 227, 258, 288, 319, 350]) # Mid-points
        y_values = np.array(list(monthly_values.values()))
        
        # Padding for cyclic continuity
        x_pad = np.concatenate(([x_months[-1]-365], x_months, [x_months[0]+365]))
        y_pad = np.concatenate(([y_values[-1]], y_values, [y_values[0]]))
        
        cs = CubicSpline(x_pad, y_pad)
        return cs(time_index.dayofyear)

    def run(self, physics):
        time_index = physics.get_time_index()
        n_hours, n_sims = len(time_index), self.p['n_sims']
        
        # 1. Deterministic Curves
        fwd_curve = self._spline_interpolate(self.p['price_curve'], time_index)
        vol_curve = self._spline_interpolate(self.p['vol_curve'], time_index)
        solar_shape, demand_shape = physics.compute_solar_profile(time_index)
        
        # 2. OU Process (Spot Price)
        prices = np.zeros((n_hours, n_sims))
        prices[0] = fwd_curve[0]
        shocks = np.random.normal(0, 1, (n_hours, n_sims))
        dt = 1/8760
        theta = self.p['theta']
        
        for t in range(1, n_hours):
            # Local Volatility Scaling
            sigma_t = vol_curve[t] * fwd_curve[t] 
            prices[t] = prices[t-1] + theta*(fwd_curve[t] - prices[t-1])*dt + sigma_t*np.sqrt(dt)*shocks[t]
            
        # 3. Log-Normal Jumps (Improved in v12)
        # Jumps are now proportional to current curve level (2x, 3x spikes)
        jump_prob = 0.003 # 0.3% probability per hour
        is_jump = np.random.rand(n_hours, n_sims) < jump_prob
        # Jump size: Log-normal distribution centered at 2x price
        jump_mult = np.random.lognormal(mean=0.8, sigma=0.5, size=(n_hours, n_sims)) 
        prices = np.where(is_jump, prices * (1 + jump_mult), prices)
        
        # 4. Correlation & Physics
        rho = self.p['correlation']
        z_mkt = np.random.normal(0, 1, (n_hours, n_sims))
        z_ast = rho*z_mkt + np.sqrt(1-rho**2)*np.random.normal(0, 1, (n_hours, n_sims))
        
        cloud_mkt = 1 / (1 + np.exp(-z_mkt * 1.5))
        cloud_ast = 1 / (1 + np.exp(-z_ast * 1.5))
        
        mkt_gen = solar_shape[:, None] * cloud_mkt
        asset_vol = solar_shape[:, None] * cloud_ast * self.p['capacity']
        
        # 5. Cannibalization & Duck Curve
        canni_decay = np.exp(-self.p['beta'] * mkt_gen)
        # Duck Curve Noise (Additive negative jumps)
        neg_noise = np.random.normal(0, 1, prices.shape) * (5.0 + 35.0*mkt_gen**2)
        
        final_spot = (prices * demand_shape[:, None] * canni_decay) + neg_noise
        
        # 6. Discounting
        years = np.arange(n_hours)/8760.0
        disc = np.exp(-self.p['rf_rate']*years)[:, None]
        
        return {
            "time": time_index, "spot": final_spot, "vol": asset_vol, 
            "disc": disc, "fwd": fwd_curve
        }

# --- 3. ANALYTICS ---

def compute_metrics(data, strike):
    # Financials
    rev_ppa = strike * data['vol']
    rev_spot = data['spot'] * data['vol']
    npvs = np.sum((rev_ppa - rev_spot) * data['disc'], axis=0)
    
    # Fair Value (Break Even)
    fair_strikes = np.sum(rev_spot*data['disc'], axis=0) / np.sum(data['vol']*data['disc'], axis=0)
    
    # Volume Risk (P-Values)
    total_vols = np.sum(data['vol'], axis=0)
    p50_vol = np.percentile(total_vols, 50)
    p90_vol = np.percentile(total_vols, 10) # Bankable Volume
    
    # Capture Rate
    avg_base = np.mean(data['spot'])
    avg_cap = np.mean(np.sum(rev_spot, axis=0)/total_vols)
    
    return {
        "npvs": npvs, "fair": fair_strikes, "p50_vol": p50_vol, "p90_vol": p90_vol,
        "capture": avg_cap/avg_base, "mean_npv": np.mean(npvs), "var_95": np.percentile(npvs, 5)
    }

# --- 4. UI ---

st.title("⚡ Quant PPA Engine | v12 Bankable")
st.markdown("**Status:** Production Grade | **New:** Stress Testing & P90 Volumes")

# SIDEBAR
with st.sidebar:
    st.header("1. Asset & Deal")
    cap = st.number_input("Capacity (MW)", 1.0, 1000.0, 50.0)
    strike = st.number_input("Strike (€)", -50.0, 500.0, 52.0)
    start = st.date_input("Start", pd.Timestamp("2026-01-01"))
    
    st.header("2. Market Curves")
    with st.expander("Edit Curves", expanded=True):
        m = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        df_in = pd.DataFrame({"Price": [95, 90, 75, 55, 45, 50, 60, 70, 80, 85, 95, 100], 
                              "Vol": [0.8, 0.7, 0.6, 0.4, 0.4, 0.3, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}, index=m)
        edt = st.data_editor(df_in, height=300)
        p_map, v_map = dict(zip(m, edt["Price"])), dict(zip(m, edt["Vol"]))

    st.header("3. Risk Settings")
    beta = st.slider("Merit Order Beta", 0.0, 5.0, 3.2)
    n_sims = st.selectbox("Simulations", [1000, 2500, 5000], index=1)
    
    run = st.button("🚀 Run Valuation", type="primary")

# EXECUTION
if run:
    params = {"price_curve": p_map, "vol_curve": v_map, "capacity": cap, "start_date": start, 
              "n_days": 365, "beta": beta, "theta": 4.0, "rf_rate": 0.04, 
              "correlation": 0.93, "n_sims": n_sims, "seed": 42, "latitude": 51.16}
    
    with st.spinner("Monte Carlo Simulation (Vectorized)..."):
        phy = MarketPhysics(start, 365, 51.16)
        eng = StochasticEngine(params)
        raw = eng.run(phy)
        res = compute_metrics(raw, strike)
        
    # METRICS ROW
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PPA NPV", f"{res['mean_npv']:,.0f} €", delta="vs Spot")
    c2.metric("Fair Value", f"{np.mean(res['fair']):.2f} €", help="Break-Even Strike")
    c3.metric("VaR 95%", f"{res['var_95']:,.0f} €", delta_color="inverse")
    c4.metric("Bankable Vol (P90)", f"{res['p90_vol']:,.0f} MWh", help="Conservative Volume Estimate")
    
    st.markdown("---")
    
    t1, t2, t3, t4 = st.tabs(["📉 Calibration & Jumps", "📊 Valuation", "🔥 Stress Test", "🔬 Microstructure"])
    
    with t1:
        st.subheader("Model Validation")
        df_c = pd.DataFrame({"Time": raw['time'], "Input": raw['fwd'], "Sim_Mean": np.mean(raw['spot'], axis=1), "Sim_Jump": raw['spot'][:, 0]})
        df_c = df_c.resample('D', on='Time').mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_c.index, y=df_c.Input, name="Input Curve", line=dict(color='black', width=3)))
        fig.add_trace(go.Scatter(x=df_c.index, y=df_c.Sim_Mean, name="Sim Average", line=dict(color='blue', dash='dot')))
        fig.add_trace(go.Scatter(x=df_c.index, y=df_c.Sim_Jump, name="Path with Jumps", line=dict(color='gray', width=1), opacity=0.5))
        st.plotly_chart(fig, use_container_width=True)
        
    with t2:
        c_a, c_b = st.columns(2)
        with c_a:
            fig_hist = px.histogram(res['npvs'], nbins=50, title="NPV Distribution", color_discrete_sequence=['#10b981'])
            fig_hist.add_vline(x=0, line_color="black")
            st.plotly_chart(fig_hist, use_container_width=True)
        with c_b:
            # Volume Risk
            vol_df = pd.DataFrame({"Volume": np.sum(raw['vol'], axis=0)})
            fig_vol = px.box(vol_df, y="Volume", title="Volume Uncertainty (P90/P50)", points="all")
            st.plotly_chart(fig_vol, use_container_width=True)

    with t3:
        st.subheader("Stress Testing Matrix")
        st.markdown("Impact of **Forward Price Shift** vs. **Capture Rate Decay** on NPV.")
        
        # Create Heatmap Data manually from current result
        shifts_price = [-0.2, -0.1, 0.0, 0.1, 0.2]
        shifts_capture = [-0.1, -0.05, 0.0, 0.05, 0.1]
        
        z_matrix = []
        base_rev_spot = np.mean(np.sum(raw['spot']*raw['vol'], axis=0))
        base_rev_ppa = np.mean(np.sum(strike*raw['vol'], axis=0))
        
        for sp in shifts_price:
            row = []
            for sc in shifts_capture:
                # Proxy Stress calculation
                stressed_spot_rev = base_rev_spot * (1 + sp) * (1 - sc) # Price up, Capture down
                npv_stress = base_rev_ppa - stressed_spot_rev
                row.append(npv_stress)
            z_matrix.append(row)
            
        fig_heat = go.Figure(data=go.Heatmap(
            z=z_matrix, x=[f"{x:+.0%}" for x in shifts_capture], y=[f"{y:+.0%}" for y in shifts_price],
            colorscale='RdBu', texttemplate="%{z:,.0f}", textfont={"size":10}
        ))
        fig_heat.update_layout(title="NPV Sensitivity (Y: Base Price, X: Cannibalization)", xaxis_title="Cannibalization Increase", yaxis_title="Base Price Shift")
        st.plotly_chart(fig_heat, use_container_width=True)

    with t4:
        st.subheader("Hourly Detail (Zoomable)")
        sl = slice(24*180, 24*187)
        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(x=raw['time'][sl], y=raw['vol'][sl, 0], name="MW", fill='tozeroy', line_color='#f59e0b'))
        fig_m.add_trace(go.Scatter(x=raw['time'][sl], y=raw['spot'][sl, 0], name="EUR", yaxis='y2', line_color='#1d4ed8'))
        fig_m.update_layout(yaxis2=dict(overlaying='y', side='right'))
        st.plotly_chart(fig_m, use_container_width=True)

else:
    st.info("Awaiting Input...")
