import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
import altair as alt
from datetime import datetime, time as dt_time
import pytz
from scipy.stats import norm
from arch import arch_model
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 页面配置 & 专业样式 ---
st.set_page_config(
    page_title="BTDR Pilot v10.8 Professional", 
    layout="centered",
    page_icon="🚀"
)

CUSTOM_CSS = """
<style>
    :root {
        --primary-color: #228be6;
        --secondary-color: #0ca678;
        --accent-color: #e03131;
        --neutral-color: #e9ecef;
        --text-dark: #212529;
        --text-light: #f8f9fa;
    }
    
    html { overflow-y: scroll; }
    .stApp > header { display: none; }
    .stApp { margin-top: -30px; background-color: #ffffff; }
    div[data-testid="stStatusWidget"] { visibility: hidden; }
    
    h1, h2, h3, div, p, span { 
        color: var(--text-dark) !important; 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important; 
    }
    
    div[data-testid="stAltairChart"] {
        height: 350px !important; min-height: 350px !important;
        overflow: hidden !important; border: 1px solid #f8f9fa;
    }
    
    /* Metric Card */
    .metric-card {
        background-color: #f8f9fa; 
        border: 1px solid #e9ecef;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        height: 105px; 
        padding: 0 16px;
        display: flex; 
        flex-direction: column; 
        justify-content: center;
        position: relative; 
        transition: all 0.2s;
        border-left: 4px solid var(--primary-color);
    }
    .metric-card.has-tooltip { cursor: help; }
    .metric-card.has-tooltip:hover { 
        border-color: #ced4da; 
        border-left-color: var(--secondary-color);
    }
    
    .metric-label { 
        font-size: 0.75rem; 
        color: #888; 
        margin-bottom: 2px; 
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value { 
        font-size: 2.0rem; 
        font-weight: 700; 
        color: var(--text-dark); 
        line-height: 1.2;
    }
    .metric-delta { 
        font-size: 0.9rem; 
        font-weight: 600; 
        margin-top: 2px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Miner Card */
    .miner-card {
        background-color: #fff; 
        border: 1px solid #e9ecef;
        border-radius: 10px; 
        padding: 8px 10px;
        text-align: center; 
        height: 100px;
        display: flex; 
        flex-direction: column; 
        justify-content: space-between;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .miner-sym { 
        font-size: 0.75rem; 
        color: #888; 
        font-weight: 600; 
        margin-bottom: 2px; 
        text-transform: uppercase;
    }
    .miner-price { 
        font-size: 1.2rem; 
        font-weight: 700; 
        color: var(--text-dark); 
    }
    .miner-sub { 
        font-size: 0.7rem; 
        display: flex; 
        justify-content: space-between; 
        margin-top: 4px; 
    }
    .miner-pct { 
        font-weight: 600; 
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .miner-turn { 
        color: #868e96; 
        font-size: 0.65rem;
    }
    
    /* Factor Box */
    .factor-box {
        background: #fff;
        border: 1px solid #eee; 
        border-radius: 8px; 
        padding: 6px; 
        text-align: center;
        height: 75px; 
        display: flex; 
        flex-direction: column; 
        justify-content: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02); 
        position: relative; 
        cursor: help; 
        transition: transform 0.1s;
        border-left: 3px solid #e9ecef;
    }
    .factor-box:hover { 
        border-color: #ced4da; 
        transform: translateY(-1px); 
        border-left-color: var(--primary-color);
    }
    .factor-title { 
        font-size: 0.65rem; 
        color: #999; 
        text-transform: uppercase; 
        letter-spacing: 0.5px; 
        margin-bottom: 3px;
    }
    .factor-val { 
        font-size: 1.2rem; 
        font-weight: bold; 
        color: #495057; 
        margin: 2px 0; 
    }
    .factor-sub { 
        font-size: 0.7rem; 
        font-weight: 600; 
        display: flex;
        justify-content: space-between;
    }
    
    /* Tooltip Core */
    .tooltip-text {
        visibility: hidden;
        width: 220px; 
        background-color: rgba(33, 37, 41, 0.95);
        color: #fff !important; 
        text-align: center; 
        border-radius: 6px; 
        padding: 8px;
        position: absolute; 
        z-index: 999;
        bottom: 110%; 
        left: 50%; 
        margin-left: -110px;
        opacity: 0; 
        transition: opacity 0.3s; 
        font-size: 0.7rem !important;
        font-weight: normal; 
        line-height: 1.4; 
        pointer-events: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .tooltip-text::after {
        content: "";
        position: absolute; 
        top: 100%; 
        left: 50%; 
        margin-left: -5px;
        border-width: 5px; 
        border-style: solid;
        border-color: rgba(33, 37, 41, 0.95) transparent transparent transparent;
    }
    
    .factor-box:hover .tooltip-text { 
        visibility: visible; 
        opacity: 1; 
    }
    .signal-box:hover .tooltip-text { 
        visibility: visible; 
        opacity: 1; 
    }
    .metric-card:hover .tooltip-text { 
        visibility: visible; 
        opacity: 1; 
    }
    
    .color-up { color: var(--secondary-color); } 
    .color-down { color: var(--accent-color); } 
    .color-neutral { color: #adb5bd; }
    
    .status-dot { 
        height: 6px; 
        width: 6px; 
        border-radius: 50%; 
        display: inline-block; 
        margin-left: 6px; 
        margin-bottom: 2px; 
    }
    .dot-pre { background-color: #f59f00; box-shadow: 0 0 4px #f59f00; }
    .dot-reg { background-color: var(--secondary-color); box-shadow: 0 0 4px var(--secondary-color); }
    .dot-post { background-color: #1c7ed6; box-shadow: 0 0 4px #1c7ed6; }
    .dot-night { background-color: #7048e8; box-shadow: 0 0 4px #7048e8; }
    .dot-closed { background-color: #adb5bd; }
    
    .pred-container-wrapper { 
        height: 110px; 
        width: 100%; 
        display: block; 
        margin-top: 5px; 
    }
    .pred-box { 
        padding: 0 10px; 
        border-radius: 12px; 
        text-align: center; 
        height: 100%; 
        display: flex; 
        flex-direction: column; 
        justify-content: center; 
    }
    
    .time-bar { 
        font-size: 0.75rem; 
        color: #999; 
        text-align: center; 
        margin-bottom: 20px; 
        padding: 6px; 
        background: #fafafa; 
        border-radius: 6px; 
        border: 1px solid #eee;
        display: inline-block;
        margin: 0 auto;
    }
    .badge-trend { 
        background: var(--secondary-color); 
        color: white; 
        padding: 1px 4px; 
        border-radius: 3px; 
        font-size: 0.6rem; 
        margin-left: 5px;
    }
    .badge-chop { 
        background: #868e96; 
        color: white; 
        padding: 1px 4px; 
        border-radius: 3px; 
        font-size: 0.6rem; 
        margin-left: 5px;
    }
    
    .ensemble-bar { 
        height: 4px; 
        width: 100%; 
        display: flex; 
        margin-top: 4px; 
        border-radius: 2px; 
        overflow: hidden;
        background: #e9ecef;
        margin-bottom: 10px;
    }
    .bar-kalman { background-color: #228be6; width: 30%; }
    .bar-hist { background-color: #fab005; width: 10%; }
    .bar-mom { background-color: #fa5252; width: 10%; }
    .bar-ai { background-color: #be4bdb; width: 50%; }
    
    /* Sniper Signals */
    .signal-box { 
        border-radius: 8px; 
        padding: 15px; 
        margin-bottom: 15px; 
        text-align: center; 
        font-weight: bold; 
        color: white; 
        display: flex; 
        flex-direction: column; 
        justify-content: center; 
        height: 100%; 
        position: relative; 
        cursor: help;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .sig-buy { 
        background-color: var(--secondary-color); 
        box-shadow: 0 4px 12px rgba(12, 166, 120, 0.3); 
        border: 1px solid #099268; 
    }
    .sig-sell { 
        background-color: var(--accent-color); 
        box-shadow: 0 4px 12px rgba(224, 49, 49, 0.3); 
        border: 1px solid #c92a2a; 
    }
    .sig-wait { 
        background-color: #ced4da; 
        color: #495057; 
        border: 1px solid #adb5bd; 
    }
    .signal-label { 
        font-size: 0.7rem; 
        opacity: 0.9; 
        margin-bottom: 2px; 
        text-transform: uppercase; 
        letter-spacing: 1px;
        font-weight: 600;
    }
    .signal-main { 
        font-size: 1.4rem; 
        line-height: 1.2; 
        font-weight: 700;
    }
    .signal-sub { 
        font-size: 0.75rem; 
        font-weight: normal; 
        margin-top: 4px; 
        opacity: 0.9; 
        color: rgba(255, 255, 255, 0.8);
    }

    /* Strategy Cards */
    .strategy-card {
        border-radius: 8px; 
        padding: 12px; 
        margin-bottom: 10px;
        text-align: left; 
        position: relative; 
        height: 100%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .strat-long { 
        background-color: #e6fcf5; 
        border: 1px solid #63e6be; 
        color: #087f5b; 
    }
    .strat-short { 
        background-color: #fff5f5; 
        border: 1px solid #ff8787; 
        color: #c92a2a; 
    }
    
    .strat-header { 
        font-size: 0.8rem; 
        font-weight: 700; 
        letter-spacing: 0.5px; 
        text-transform: uppercase; 
        margin-bottom: 8px; 
        display: flex; 
        justify-content: space-between;
    }
    .strat-row { 
        display: flex; 
        justify-content: space-between; 
        font-size: 0.85rem; 
        margin-bottom: 4px; 
        align-items: center; 
    }
    .strat-label { 
        opacity: 0.8; 
    }
    .strat-val { 
        font-weight: 700; 
        font-size: 1rem; 
    }
    .strat-rr { 
        margin-top: 8px; 
        padding-top: 8px; 
        border-top: 1px dashed rgba(0,0,0,0.1);
        display: flex; 
        justify-content: space-between; 
        font-size: 0.8rem; 
        font-weight: 600;
    }
    .rr-good { color: #2f9e44; } 
    .rr-bad { color: #e03131; }
    
    .ticket-card {
        border-radius: 10px; 
        padding: 15px; 
        margin-bottom: 10px;
        text-align: left; 
        position: relative; 
        border-left: 5px solid #ccc;
        background: #fff; 
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .ticket-buy { 
        border-left-color: var(--secondary-color); 
        background: #f0fff4; 
    }
    .ticket-sell { 
        border-left-color: var(--accent-color); 
        background: #fff5f5; 
    }
    
    .ticket-header { 
        font-size: 0.9rem; 
        font-weight: 800; 
        letter-spacing: 0.5px; 
        text-transform: uppercase; 
        margin-bottom: 10px; 
        display: flex; 
        justify-content: space-between; 
        align-items: center;
    }
    .ticket-price-row { 
        display: flex; 
        align-items: baseline; 
        margin-bottom: 8px; 
    }
    .ticket-price-label { 
        font-size: 0.8rem; 
        color: #555; 
        width: 80px; 
    }
    .ticket-price-val { 
        font-size: 1.7rem; 
        font-weight: 900; 
        color: var(--text-dark); 
        letter-spacing: -0.5px; 
    }
    
    .ticket-meta { 
        display: flex; 
        justify-content: space-between; 
        font-size: 0.75rem; 
        margin-top: 8px; 
        color: #666; 
        border-top: 1px solid rgba(0,0,0,0.05); 
        padding-top: 8px; 
    }
    .prob-container { 
        width: 100%; 
        height: 4px; 
        background: #eee; 
        margin-top: 5px; 
        border-radius: 2px; 
    }
    .prob-fill { 
        height: 100%; 
        border-radius: 2px; 
    }
    .prob-high { background: #2f9e44; } 
    .prob-med { background: #fab005; } 
    .prob-low { background: #ced4da; }
    .tag-smart { 
        background: var(--primary-color); 
        color: white; 
        padding: 1px 5px; 
        border-radius: 4px; 
        font-size: 0.6rem; 
        vertical-align: middle; 
        margin-left: 5px; 
    }
    
    /* Professional Chart Styles */
    .chart-title { 
        font-size: 1.1rem; 
        font-weight: 600; 
        color: var(--text-dark); 
        margin-bottom: 10px;
    }
    .chart-subtitle { 
        font-size: 0.8rem; 
        color: #6c757d; 
        margin-bottom: 15px;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .metric-card { height: 95px; }
        .miner-card { height: 90px; }
        .factor-box { height: 65px; }
        .signal-box { padding: 12px; }
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- 2. 基础配置 ---
MINER_SHARES = {"MARA": 300, "RIOT": 330, "CLSK": 220, "CORZ": 190, "IREN": 180, "WULF": 410, "CIFR": 300, "HUT": 100}
MINER_POOL = list(MINER_SHARES.keys())

# --- 3. 辅助函数 ---
def card_html(label, value_str, delta_str=None, delta_val=0, extra_tag="", tooltip_text=None):
    delta_html = ""
    if delta_str:
        color_class = "color-up" if delta_val >= 0 else "color-down"
        delta_html = f"<div class='metric-delta {color_class}'>{delta_str}</div>"
    
    tooltip_html = f"<div class='tooltip-text'>{tooltip_text}</div>" if tooltip_text else ""
    card_class = "metric-card has-tooltip" if tooltip_text else "metric-card"
    
    return f"""<div class="{card_class}">{tooltip_html}<div class="metric-label">{label} {extra_tag}</div><div class="metric-value">{value_str}</div>{delta_html}</div>"""

def factor_html(title, val, delta_str, delta_val, tooltip_text, reverse_color=False):
    is_positive = delta_val >= 0
    if reverse_color: is_positive = not is_positive
    color_class = "color-up" if is_positive else "color-down"
    return f"""<div class="factor-box"><div class="tooltip-text">{tooltip_text}</div><div class="factor-title">{title}</div><div class="factor-val">{val}</div><div class="factor-sub {color_class}">{delta_str}</div></div>"""

def miner_card_html(sym, price, pct, turnover):
    color_class = "color-up" if pct >= 0 else "color-down"
    return f"""<div class="miner-card"><div class="miner-sym">{sym}</div><div class="miner-price ${color_class}">${price:.2f}</div><div class="miner-sub"><span class="miner-pct {color_class}">{pct:+.1f}%</span><span class="miner-turn">换 {turnover:.1f}%</span></div></div>"""

# --- 4. 核心计算模块 ---
def run_kalman_filter(y, x, delta=1e-4):
    """改进的多变量Kalman滤波器，提高预测精度"""
    n = len(y)
    # 初始化状态向量和协方差矩阵
    beta = np.zeros(n)
    P = np.zeros(n)
    beta[0] = 1.0
    P[0] = 1.0
    
    # 状态转移矩阵和观测矩阵
    F = np.array()
    H = np.array([[x[0]]])
    
    # 过程噪声和观测噪声
    Q = np.array([[delta]])
    R = np.array([[0.002]])
    
    for t in range(1, n):
        # 预测步骤
        beta_pred = F @ np.array([beta[t-1]])
        P_pred = F @ P[t-1] @ F.T + Q
        
        # 更新步骤
        K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
        residual = y[t] - H @ np.array([beta[t-1]])
        beta[t] = beta_pred + K @ residual
        P[t] = (np.eye(1) - K @ H) @ P_pred
    
    return beta[-1]

def calculate_additional_features(df, window=14):
    """计算更多技术指标作为特征，增强模型预测能力"""
    df = df.copy()
    
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['std20'] = df['Close'].rolling(window=20).std()
    df['Upper_BB'] = df['MA20'] + (df['std20'] * 2)
    df['Lower_BB'] = df['MA20'] - (df['std20'] * 2)
    
    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    df['ATR'] = tr.rolling(window=14).mean()
    
    return df

def estimate_volatility(series, window=30):
    """使用GARCH(1,1)模型估计波动率，捕捉波动率聚类效应"""
    try:
        model = arch_model(series, vol='Garch', p=1, q=1, dist='Normal')
        results = model.fit(disp='off')
        return results.conditional_volatility[-1]
    except:
        return series.std()

def get_fng_index():
    """获取恐惧贪婪指数，用于情绪分析"""
    try:
        response = requests.get("https://api.alternative.me/fng/", timeout=0.8)
        data = response.json()['data'][0]
        return int(data['value']), data['value_classification']
    except Exception as e:
        print(f"Error getting FNG index: {e}")
        return 50, "Neutral"

@st.cache_data(ttl=600)
def run_grandmaster_analytics():
    """专业版分析引擎，结合多模型预测"""
    default_model = {
        "high": {"intercept": 0, "beta_gap": 0.5, "beta_btc": 0.5, "beta_vol": 0},
        "low": {"intercept": 0, "beta_gap": 0.5, "beta_btc": 0.5, "beta_vol": 0},
        "ensemble_hist_h": 0.05, "ensemble_hist_l": -0.05,
        "ensemble_mom_h": 0.08, "ensemble_mom_l": -0.08,
        "top_peers": ["MARA", "RIOT", "CLSK", "CORZ", "IREN"]
    }
    default_factors = {"vwap": 0, "adx": 20, "regime": "Neutral", "beta_btc": 1.5, "beta_qqq": 1.2, "rsi": 50, "vol_base": 0.05, "atr_ratio": 0.05}

    try:
        tickers_str = "BTDR BTC-USD QQQ " + " ".join(MINER_POOL)
        data = yf.download(tickers_str, period="6mo", interval="1d", group_by='ticker', threads=True, progress=False)
        if data.empty: 
            return default_model, default_factors, "No Data"
        
        # 数据清洗和异常值处理
        for ticker in data.columns.levels[0]:
            if ticker in ['BTDR', 'BTC-USD', 'QQQ']:
                data[ticker] = data[ticker].replace([np.inf, -np.inf], np.nan).dropna()
                data[ticker] = data[ticker][data[ticker]['Close'] > 0]
        
        btdr = data['BTDR'].dropna(); btc = data['BTC-USD'].dropna(); qqq = data['QQQ'].dropna()
        idx = btdr.index.intersection(btc.index).intersection(qqq.index)
        btdr, btc, qqq = btdr.loc[idx], btc.loc[idx], qqq.loc[idx]
        
        if len(btdr) < 30: 
            return default_model, default_factors, "Insufficient Data"

        # 计算更多技术指标
        btdr = calculate_additional_features(btdr)
        btc = calculate_additional_features(btc)
        qqq = calculate_additional_features(qqq)
        
        # 相关性分析
        correlations = {}
        for m in MINER_POOL:
            if m in data:
                miner_df = data[m]['Close'].pct_change().tail(30)
                btdr_df = btdr['Close'].pct_change().tail(30)
                common_idx = miner_df.index.intersection(btdr_df.index)
                if len(common_idx) > 10: 
                    correlations[m] = miner_df.loc[common_idx].corr(btdr_df.loc[common_idx])
                else: 
                    correlations[m] = 0
        top_peers = sorted(correlations, key=correlations.get, reverse=True)[:5]
        default_model["top_peers"] = top_peers

        # 计算波动率
        ret_btdr = btdr['Close'].pct_change().fillna(0).values
        vol_base = estimate_volatility(ret_btdr)
        if len(ret_btdr) > 20:
            vol_base = pd.Series(ret_btdr).ewm(span=20).std().iloc[-1]
        
        # 计算Beta
        beta_btc = run_kalman_filter(ret_btdr, btc['Close'].pct_change().fillna(0).values, delta=1e-4)
        beta_qqq = run_kalman_filter(ret_btdr, qqq['Close'].pct_change().fillna(0).values, delta=1e-4)
        beta_btc = np.clip(beta_btc, -1, 5)
        beta_qqq = np.clip(beta_qqq, -1, 4)

        # VWAP计算
        pv = (btdr['Close'] * btdr['Volume'])
        vwap_30d = pv.tail(30).sum() / btdr['Volume'].tail(30).sum()
        
        # ADX计算
        high, low, close = btdr['High'], btdr['Low'], btdr['Close']
        tr = np.maximum(high - low, np.abs(high - close.shift(1)))
        atr = tr.rolling(14).mean()
        
        up, down = high.diff(), -low.diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        atr_s = pd.Series(atr.values, index=btdr.index)
        plus_di = 100 * (pd.Series(plus_dm, index=btdr.index).rolling(14).mean() / atr_s)
        minus_di = 100 * (pd.Series(minus_dm, index=btdr.index).rolling(14).mean() / atr_s)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean().iloc[-1]
        adx = 20 if np.isnan(adx) else adx
        
        # RSI计算
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # ATR比率
        atr_ratio = (btdr['ATR'] / close).iloc[-1]

        factors = {
            "beta_btc": beta_btc, 
            "beta_qqq": beta_qqq, 
            "vwap": vwap_30d, 
            "adx": adx, 
            "regime": "Trend" if adx > 25 else "Chop", 
            "rsi": rsi, 
            "vol_base": vol_base, 
            "atr_ratio": atr_ratio,
            "macd": btdr['MACD'].iloc[-1],
            "macd_signal": btdr['MACD_signal'].iloc[-1],
            "bb_upper": btdr['Upper_BB'].iloc[-1],
            "bb_lower": btdr['Lower_BB'].iloc[-1]
        }

        # 回归模型构建
        df_reg = pd.DataFrame()
        df_reg['PrevClose'] = btdr['Close'].shift(1)
        df_reg['Open'] = btdr['Open']
        df_reg['High'] = btdr['High']
        df_reg['Low'] = btdr['Low']
        df_reg['Gap'] = (df_reg['Open'] - df_reg['PrevClose']) / df_reg['PrevClose']
        df_reg['BTC_Ret'] = btc['Close'].pct_change()
        df_reg['Vol_State'] = ((btdr['High'] - btdr['Low']) / btdr['Open']).shift(1)
        df_reg['Target_High'] = (df_reg['High'] - df_reg['PrevClose']) / df_reg['PrevClose']
        df_reg['Target_Low'] = (df_reg['Low'] - df_reg['PrevClose']) / df_reg['PrevClose']
        df_reg = df_reg.dropna().tail(90)
        
        # 加权回归
        weights = np.exp(np.linspace(-0.05 * len(df_reg), 0, len(df_reg)))
        W = np.diag(weights)
        X = np.column_stack([
            np.ones(len(df_reg)), 
            df_reg['Gap'].values, 
            df_reg['BTC_Ret'].values, 
            df_reg['Vol_State'].values,
            df_reg['RSI'].values
        ])
        Y_h = df_reg['Target_High'].values
        Y_l = df_reg['Target_Low'].values
        XtWX = X.T @ W @ X
        theta_h = np.linalg.lstsq(XtWX, X.T @ W @ Y_h, rcond=None)[0]
        theta_l = np.linalg.lstsq(XtWX, X.T @ W @ Y_l, rcond=None)[0]

        final_model = {
            "high": {
                "intercept": theta_h[0], 
                "beta_gap": theta_h[1], 
                "beta_btc": theta_h[2], 
                "beta_vol": theta_h[3],
                "beta_rsi": theta_h[4]
            },
            "low": {
                "intercept": theta_l[0], 
                "beta_gap": theta_l[1], 
                "beta_btc": theta_l[2], 
                "beta_vol": theta_l[3],
                "beta_rsi": theta_l[4]
            },
            "ensemble_hist_h": df_reg['Target_High'].tail(10).mean(), 
            "ensemble_hist_l": df_reg['Target_Low'].tail(10).mean(),
            "ensemble_mom_h": df_reg['Target_High'].tail(3).max(), 
            "ensemble_mom_l": df_reg['Target_Low'].tail(3).min(),
            "top_peers": top_peers
        }
        return final_model, factors, "v10.8 Professional"
    except Exception as e:
        print(f"Error: {e}")
        return default_model, default_factors, "Offline"

# --- 5. 实时数据 ---
def determine_market_state(now_ny):
    """确定市场状态（预市、开盘、收市、休市）"""
    weekday = now_ny.weekday()
    curr_min = now_ny.hour * 60 + now_ny.minute
    if weekday == 5: return "Weekend", "dot-closed"
    if weekday == 6 and now_ny.hour < 20: return "Weekend", "dot-closed"
    if 240 <= curr_min < 570: return "Pre-Mkt", "dot-pre"
    if 570 <= curr_min < 960: return "Mkt Open", "dot-reg"
    if 960 <= curr_min < 1200: return "Post-Mkt", "dot-post"
    return "Overnight", "dot-night"

def get_realtime_data():
    """获取实时市场数据"""
    tickers_list = "BTC-USD BTDR QQQ ^VIX " + " ".join(MINER_POOL)
