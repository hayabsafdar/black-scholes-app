import streamlit as st
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.set_page_config(
    page_title="Black-Scholes Pricer",
    page_icon="📈",
    layout="wide"
)

# ── Global styling
st.markdown("""
<style>
    /* Main background and font */
    .stApp {
        background-color: #0f1117;
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2e3347;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    /* Main title */
    h1 {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem !important;
    }

    /* Section headers */
    h2 {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #00c6ff !important;
        border-bottom: 1px solid #2e3347;
        padding-bottom: 6px;
        margin-top: 2rem !important;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #1a1d27;
        border: 1px solid #2e3347;
        border-radius: 12px;
        padding: 16px !important;
        transition: border-color 0.2s;
    }

    [data-testid="metric-container"]:hover {
        border-color: #00c6ff;
    }

    /* Metric label */
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        color: #8892a4 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Metric value */
    [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }

    /* Number inputs */
    [data-testid="stNumberInput"] input {
        background-color: #252836 !important;
        border: 1px solid #2e3347 !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        font-size: 0.95rem !important;
    }

    /* Text input */
    [data-testid="stTextInput"] input {
        background-color: #252836 !important;
        border: 1px solid #2e3347 !important;
        border-radius: 8px !important;
        color: #ffffff !important;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(90deg, #00c6ff, #0072ff) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.4rem 1.2rem !important;
        transition: opacity 0.2s !important;
    }

    .stButton button:hover {
        opacity: 0.85 !important;
    }

    /* Caption text */
    [data-testid="stCaptionContainer"] {
        color: #8892a4 !important;
        font-size: 0.85rem !important;
    }

    /* Divider */
    hr {
        border-color: #2e3347 !important;
    }

    /* Success / warning / error boxes */
    .stAlert {
        border-radius: 10px !important;
        font-size: 0.9rem !important;
    }

    /* Footer text */
    footer {
        color: #8892a4 !important;
    }

    /* Plotly chart background */
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ── Title
st.markdown("""
<div style="padding: 1.5rem 0 1rem 0;">
    <h1>Black-Scholes Option Pricing Dashboard</h1>
    <p style="color: #8892a4; font-size: 0.95rem; margin-top: 0.2rem;">
        Real-time European option pricing · Greeks · Sensitivity Analysis · Live Swiss Market Data
    </p>
</div>
""", unsafe_allow_html=True)

# ── Real Swiss Stock Data
st.sidebar.markdown("---")
st.sidebar.header("Live Swiss Stock Price")

ticker_input = st.sidebar.text_input(
    "Enter Swiss ticker (e.g. ROG.SW)",
    value="ROG.SW"
)

if ticker_input:
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker_input).history(period="5d")
        if not hist.empty:
            live_price = round(float(hist["Close"].iloc[-1]), 2)
            st.sidebar.success(f"Live price: {live_price}")
            st.session_state["live_price"] = live_price
            if st.sidebar.button(f"⬆ Use {live_price} as S"):
                st.session_state["S"] = float(live_price)
        else:
            st.sidebar.warning("No data found — try: ROG.SW / NOVN.SW / NESN.SW")
    except Exception as e:
        st.sidebar.warning("No data found — try: ROG.SW / NOVN.SW / NESN.SW")
# ── Sidebar inputs
# ── Sidebar header with LinkedIn
st.sidebar.markdown(
    """
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="28"/>
        <a href="https://www.linkedin.com/in/hayab-safdar-b08212157/" target="_blank" 
           style="font-size: 15px; font-weight: 600; text-decoration: none; color: #0A66C2;">
            Hayab Safdar
        </a>
    </div>
    <div style="font-size: 12px; color: grey; margin-bottom: 20px;">
        MSc Finance · University of Basel
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("Model Parameters")

if "S" not in st.session_state:
    st.session_state["S"] = 100.0
S = st.sidebar.number_input("Current Stock Price (S)",
                              min_value=1.0, max_value=10000.0,
                              value=st.session_state["S"],
                              step=1.0,
                              key="S")

K = st.sidebar.number_input("Strike Price (K)",
                              min_value=1.0, max_value=10000.0,
                              value=100.0, step=1.0)

T = st.sidebar.number_input("Time to Expiry (years)",
                              min_value=0.01, max_value=5.0,
                              value=0.5, step=0.01)

r = st.sidebar.number_input("Risk-Free Rate (%)",
                              min_value=0.0, max_value=20.0,
                              value=2.0, step=0.1) / 100

sigma = st.sidebar.number_input("Volatility σ (%)",
                                  min_value=1.0, max_value=200.0,
                                  value=20.0, step=0.5) / 100

# ── Heatmap parameters
st.sidebar.markdown("---")
st.sidebar.header("Heatmap Parameters")

spot_min_pct = st.sidebar.slider(
    "Spot Price Range (% around S)",
    min_value=10, max_value=50, value=30, step=5
)

vol_min = st.sidebar.slider(
    "Min Volatility for Heatmap (%)",
    min_value=1, max_value=50, value=5, step=1
)

vol_max = st.sidebar.slider(
    "Max Volatility for Heatmap (%)",
    min_value=10, max_value=200, value=60, step=5
)

# ── Black-Scholes core function
def black_scholes(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price  = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    delta_call =  norm.cdf(d1)
    delta_put  = -norm.cdf(-d1)
    gamma      =  norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega       =  S * norm.pdf(d1) * np.sqrt(T) / 100
    theta_call = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                  - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    theta_put  = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                  + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    rho_call   =  K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    rho_put    = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        "call_price":  round(call_price, 4),
        "put_price":   round(put_price, 4),
        "d1":          round(d1, 4),
        "d2":          round(d2, 4),
        "delta_call":  round(delta_call, 4),
        "delta_put":   round(delta_put, 4),
        "gamma":       round(gamma, 6),
        "vega":        round(vega, 4),
        "theta_call":  round(theta_call, 4),
        "theta_put":   round(theta_put, 4),
        "rho_call":    round(rho_call, 4),
        "rho_put":     round(rho_put, 4),
    }

# ── Run the model
results = black_scholes(S, K, T, r, sigma)

# ── Section 1: Prices
st.header("Option Prices")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Call Price",  f"${results['call_price']:.4f}")
col2.metric("Put Price",   f"${results['put_price']:.4f}")
col3.metric("d₁",         f"{results['d1']:.4f}")
col4.metric("d₂",         f"{results['d2']:.4f}")

# ── Section 2: Greeks
st.header("The Greeks")
st.caption("How sensitive the option price is to each input")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Delta (Call)", results["delta_call"],
            help="Change in call price per $1 move in stock")
col1.metric("Delta (Put)",  results["delta_put"])
col2.metric("Gamma",        results["gamma"],
            help="Rate of change of Delta per $1 move in stock")
col3.metric("Vega",         results["vega"],
            help="Change in price per 1% change in volatility")
col4.metric("Theta (Call)", results["theta_call"],
            help="Daily time decay of the call option")
col4.metric("Theta (Put)",  results["theta_put"])
col5.metric("Rho (Call)",   results["rho_call"],
            help="Change in price per 1% change in interest rate")
col5.metric("Rho (Put)",    results["rho_put"])

# ── Section 3: Payoff diagram
st.header("Option Payoff at Expiry")

stock_range = np.linspace(S * 0.5, S * 1.5, 200)
call_payoff = np.maximum(stock_range - K, 0) - results["call_price"]
put_payoff  = np.maximum(K - stock_range, 0) - results["put_price"]

fig_payoff = go.Figure()
fig_payoff.add_trace(go.Scatter(
    x=stock_range, y=call_payoff,
    name="Call P&L", line=dict(color="#2196F3", width=2)
))
fig_payoff.add_trace(go.Scatter(
    x=stock_range, y=put_payoff,
    name="Put P&L", line=dict(color="#E53935", width=2)
))
fig_payoff.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
fig_payoff.add_vline(x=K, line_dash="dot", line_color="gray", opacity=0.5,
                     annotation_text="Strike", annotation_position="top")
fig_payoff.update_layout(
    title="P&L at Expiry (Long Call vs Long Put)",
    xaxis_title="Stock Price at Expiry ($)",
    yaxis_title="Profit / Loss ($)",
    hovermode="x unified",
    height=400
)
st.plotly_chart(fig_payoff, width='stretch')

# ── Section 4: Heatmaps
st.header("Sensitivity Heatmaps")
st.caption("How call and put prices change across different stock prices and volatilities")

spot_range = np.linspace(S * (1 - spot_min_pct/100), S * (1 + spot_min_pct/100), 20)
vol_range  = np.linspace(vol_min/100, vol_max/100, 20)

call_matrix = np.zeros((len(vol_range), len(spot_range)))
put_matrix  = np.zeros((len(vol_range), len(spot_range)))

for i, v in enumerate(vol_range):
    for j, s in enumerate(spot_range):
        res = black_scholes(s, K, T, r, v)
        call_matrix[i, j] = res["call_price"]
        put_matrix[i, j]  = res["put_price"]

col1, col2 = st.columns(2)

with col1:
    fig_call = px.imshow(
        call_matrix,
        x=np.round(spot_range, 1),
        y=np.round(vol_range * 100, 1),
        labels=dict(x="Stock Price ($)", y="Volatility (%)", color="Call Price"),
        title="Call Price Heatmap",
        color_continuous_scale="Blues",
        aspect="auto"
    )
    st.plotly_chart(fig_call, width='stretch')

with col2:
    fig_put = px.imshow(
        put_matrix,
        x=np.round(spot_range, 1),
        y=np.round(vol_range * 100, 1),
        labels=dict(x="Stock Price ($)", y="Volatility (%)", color="Put Price"),
        title="Put Price Heatmap",
        color_continuous_scale="Reds",
        aspect="auto"
    )
    st.plotly_chart(fig_put, width='stretch')

# ── Section 5: Greeks vs Stock Price
st.header("Greeks vs Stock Price")

greek_range = np.linspace(S * 0.5, S * 1.5, 200)
deltas = [black_scholes(s, K, T, r, sigma)["delta_call"] for s in greek_range]
gammas = [black_scholes(s, K, T, r, sigma)["gamma"] for s in greek_range]

col1, col2 = st.columns(2)

with col1:
    fig_delta = go.Figure()
    fig_delta.add_trace(go.Scatter(
        x=greek_range, y=deltas,
        line=dict(color="#2196F3", width=2), name="Delta"
    ))
    fig_delta.add_vline(x=K, line_dash="dot", line_color="gray",
                        annotation_text="Strike")
    fig_delta.update_layout(
        title="Delta vs Stock Price",
        xaxis_title="Stock Price ($)",
        yaxis_title="Delta",
        height=350
    )
    st.plotly_chart(fig_delta, width='stretch')

with col2:
    fig_gamma = go.Figure()
    fig_gamma.add_trace(go.Scatter(
        x=greek_range, y=gammas,
        line=dict(color="#9C27B0", width=2), name="Gamma"
    ))
    fig_gamma.add_vline(x=K, line_dash="dot", line_color="gray",
                        annotation_text="Strike")
    fig_gamma.update_layout(
        title="Gamma vs Stock Price",
        xaxis_title="Stock Price ($)",
        yaxis_title="Gamma",
        height=350
    )
    st.plotly_chart(fig_gamma, width='stretch')

# ── Footer
# ── Footer
st.markdown("---")
st.markdown(
    "Built with Python · Black-Scholes (1973) · ")

# ── Sidebar LinkedIn
st.sidebar.markdown("---")
st.sidebar.markdown("**Built by**")
st.sidebar.markdown("[Hayab Safdar](https://www.linkedin.com/in/hayab-safdar-b08212157/)")