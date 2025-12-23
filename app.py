import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
import altair as alt
from datetime import datetime, time as dt_time
import pytz

# --- 1. 页面配置 & 样式 ---
st.set_page_config(page_title="BTDR Pilot v9.8 Kalman", layout="centered")

CUSTOM_CSS = """
<style>
    html { overflow-y: scroll; }
    .stApp > header { display: none; }
    .stApp { margin-top: -30px; background-color: #ffffff; }
    div[data-testid="stStatusWidget"] { visibility: hidden; }
    
    h1, h2, h3, div, p, span { 
        color: #212529 !important; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important; 
    }
    
    div[data-testid="stAltairChart"] {
        height: 320px !important; min-height: 320px !important;
        overflow: hidden !important; border: 1px solid #f8f9fa;
    }
    
    .metric-card {
        background-color: #f8f9fa; border: 1px solid #e9ecef;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02); height: 95px; padding: 0 16px;
        display: flex; flex-direction: column; justify-content: center;
    }
    .metric-label { font-size: 0.75rem; color: #888; margin-bottom: 2px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #212529; line-height: 1.2; }
    .metric-delta { font-size: 0.9rem; font-weight: 600; margin-top: 2px; }
    
    .factor-box {
        background: #fff;
        border: 1px solid #eee; border-radius: 8px; padding: 6px; text-align: center;
        height: 75px; display: flex; flex-direction: column; justify-content: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02); position: relative; cursor: help; transition: transform 0.1s;
    }
    .factor-box:hover { border-color: #ced4da; transform: translateY(-1px); }
    .factor-title { font-size: 0.65rem; color: #999; text-transform: uppercase; letter-spacing: 0.5px; }
    .factor-val { font-size: 1.1rem; font-weight: bold; color: #495057; margin: 2px 0; }
    .factor-sub { font-size: 0.7rem; font-weight: 600; }
    
    .tooltip-text {
        visibility: hidden;
        width: 180px; background-color: rgba(33, 37, 41, 0.95);
        color: #fff !important; text-align: center; border-radius: 6px; padding: 8px;
        position: absolute; z-index: 999;
        bottom: 110%; left: 50%; margin-left: -90px;
        opacity: 0; transition: opacity 0.3s; font-size: 0.7rem !important;
        font-weight: normal; line-height: 1.4; pointer-events: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .tooltip-text::after {
        content: "";
        position: absolute; top: 100%; left: 50%; margin-left: -5px;
        border-width: 5px; border-style: solid;
        border-color: rgba(33, 37, 41, 0.95) transparent transparent transparent;
    }
    .factor-box:hover .tooltip-text { visibility: visible; opacity: 1; }
    
    .color-up { color: #0ca678; } .color-down { color: #d6336c; } .color-neutral { color: #adb5bd; }
    
    .status-dot { height: 6px; width: 6px; border-radius: 50%; display: inline-block; margin-left: 6px; margin-bottom: 2px; }
    .dot-pre { background-color: #f59f00; box-shadow: 0 0 4px #f59f00; }
    .dot-reg { background-color: #0ca678; box-shadow: 0 0 4px #0ca678; }
    .dot-post { background-color: #1c7ed6; box-shadow: 0 0 4px #1c7ed6; }
    .dot-night { background-color: #7048e8; box-shadow: 0 0 4px #7048e8; }
    .dot-closed { background-color: #adb5bd; }
    
    .pred-container-wrapper { height: 110px; width: 100%; display: block; margin-top: 5px; }
    .pred-box { padding: 0 10px; border-radius: 12px; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center; }
    
    .time-bar { font-size: 0.75rem; color: #999; text-align: center; margin-bottom: 20px; padding: 6px; background: #fafafa; border-radius: 6px; }
    .badge-trend { background:#fd7e14; color:white; padding:1px 4px; border-radius:3px; font-size:0.6rem; }
    .badge-chop { background:#868e96; color:white; padding:1px 4px; border-radius:3px; font-size:0.6rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- 2. 辅助函数 ---
def card_html(label, value_str, delta_str=None, delta_val=0, extra_tag=""):
    delta_html = ""
    if delta_str:
        color_class = "color-up" if delta_val >= 0 else "color-down"
        delta_html = f"<div class='metric-delta {color_class}'>{delta_str}</div>"
    return f"""<div class="metric-card"><div class="metric-label">{label} {extra_tag}</div><div class="metric-value">{value_str}</div>{delta_html}</div>"""

def factor_html(title, val, delta_str, delta_val, tooltip_text, reverse_color=False):
    is_positive = delta_val >= 0
    if reverse_color: is_positive = not is_positive
    color_class = "color-up" if is_positive else "color-down"
    return f"""
    <div class="factor-box">
        <div class="tooltip-text">{tooltip_text}</div>
        <div class="factor-title">{title}</div>
        <div class="factor-val">{val}</div>
        <div class="factor-sub {color_class}">{delta_str}</div>
    </div>
    """

# --- 3. 核心计算逻辑 (v9.8: Kalman Filter + WLS) ---
def run_kalman_filter(y, x, delta=1e-4):
    """
    卡尔曼滤波：动态估计 Beta
    y: 标的资产收益率 (BTDR)
    x: 基准资产收益率 (BTC/QQQ)
    delta: 过程噪声 (Beta 变化的灵敏度)
    """
    n = len(y)
    beta = np.zeros(n)
    P = np.zeros(n)     # 误差协方差
    beta[0] = 1.0       # 初始猜测
    P[0] = 1.0
    
    # 噪声参数
    R = 0.002 # 观测噪声
    Q = delta / (1 - delta)
    
    for t in range(1, n):
        # 1. 预测
        beta_pred = beta[t-1]
        P_pred = P[t-1] + Q
        
        # 2. 更新
        if x[t] == 0: x[t] = 1e-6 # 防止除零
        
        residual = y[t] - beta_pred * x[t]
        S = P_pred * x[t]**2 + R
        K = P_pred * x[t] / S # 卡尔曼增益
        
        beta[t] = beta_pred + K * residual
        P[t] = (1 - K * x[t]) * P_pred
        
    return beta[-1]

@st.cache_data(ttl=600)
def run_grandmaster_analytics():
    # 默认值
    default_model = {
        "high": {"intercept": 0, "beta_gap": 0.5, "beta_btc": 0.5, "beta_vol": 0},
        "low": {"intercept": 0, "beta_gap": 0.5, "beta_btc": 0.5, "beta_vol": 0}
    }
    default_factors = {"vwap": 0, "adx": 20, "regime": "Neutral", "beta_btc": 1.5, "beta_qqq": 1.2, "rsi": 50, "vol_base": 0.05, "atr_ratio": 0.05}

    try:
        data = yf.download("BTDR BTC-USD QQQ", period="6mo", interval="1d", group_by='ticker', threads=True, progress=False)
        
        if data.empty: return default_model, default_factors, "No Data"

        # 数据对齐
        btdr = data['BTDR'].dropna()
        btc = data['BTC-USD'].dropna()
        qqq = data['QQQ'].dropna()
        
        idx = btdr.index.intersection(btc.index).intersection(qqq.index)
        btdr, btc, qqq = btdr.loc[idx], btc.loc[idx], qqq.loc[idx]
        
        if len(btdr) < 30: return default_model, default_factors, "Insufficient Data"

        # --- 1. 卡尔曼动态 Beta 计算 (v9.8 升级) ---
        ret_btdr = btdr['Close'].pct_change().fillna(0).values
        ret_btc = btc['Close'].pct_change().fillna(0).values
        ret_qqq = qqq['Close'].pct_change().fillna(0).values
        
        # 运行滤波器
        beta_btc = run_kalman_filter(ret_btdr, ret_btc, delta=1e-4)
        beta_qqq = run_kalman_filter(ret_btdr, ret_qqq, delta=1e-4)
        
        # 安全截断，防止数据异常导致 Beta 飞出天际
        beta_btc = np.clip(beta_btc, -1, 5)
        beta_qqq = np.clip(beta_qqq, -1, 4)

        # --- 2. 其他因子计算 ---
        # VWAP
        pv = (btdr['Close'] * btdr['Volume'])
        vwap_30d = pv.tail(30).sum() / btdr['Volume'].tail(30).sum()

        # ADX
        high, low, close = btdr['High'], btdr['Low'], btdr['Close']
        tr = np.maximum(high - low, np.abs(high - close.shift(1)))
        atr = tr.rolling(14).mean()
        
        up, down = high.diff(), -low.diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        plus_di = 100 * (pd.Series(plus_dm, index=btdr.index).rolling(14).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm, index=btdr.index).rolling(14).mean() / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean().iloc[-1]
        adx = 20 if np.isnan(adx) else adx

        # RSI & Volatility
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        vol_base = ret_btdr.ewm(span=20).std().iloc[-1] if hasattr(ret_btdr, 'ewm') else np.std(ret_btdr[-20:])
        # 重新获取 pd.Series 用于计算 atr_ratio (之前 ret_btdr 变为了 numpy)
        atr_ratio = (atr / close).iloc[-1]

        factors = {
            "beta_btc": beta_btc, "beta_qqq": beta_qqq, "vwap": vwap_30d, 
            "adx": adx, "regime": "Trend" if adx > 25 else "Chop", 
            "rsi": rsi, "vol_base": vol_base, "atr_ratio": atr_ratio
        }

        # --- 3. WLS 回归 (保持 v9.7 逻辑) ---
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
        
        decay_alpha = 0.05
        weights = np.exp(np.linspace(-decay_alpha * len(df_reg), 0, len(df_reg)))
        W = np.diag(weights)

        X = np.column_stack([
            np.ones(len(df_reg)),       
            df_reg['Gap'].values,       
            df_reg['BTC_Ret'].values,   
            df_reg['Vol_State'].values  
        ])
        
        Y_h = df_reg['Target_High'].values
        Y_l = df_reg['Target_Low'].values

        XtWX = X.T @ W @ X
        theta_h = np.linalg.lstsq(XtWX, X.T @ W @ Y_h, rcond=None)[0]
        theta_l = np.linalg.lstsq(XtWX, X.T @ W @ Y_l, rcond=None)[0]

        final_model = {
            "high": {"intercept": theta_h[0], "beta_gap": theta_h[1], "beta_btc": theta_h[2], "beta_vol": theta_h[3]},
            "low": {"intercept": theta_l[0], "beta_gap": theta_l[1], "beta_btc": theta_l[2], "beta_vol": theta_l[3]}
        }
        
        return final_model, factors, "v9.8 Kalman/WLS"

    except Exception as e:
        print(f"Algorithm Error: {e}")
        return default_model, default_factors, "Offline"

# --- 4. 实时数据 ---
def determine_market_state(now_ny):
    weekday = now_ny.weekday()
    curr_min = now_ny.hour * 60 + now_ny.minute
    if weekday == 5: return "Weekend", "dot-closed"
    if weekday == 6 and now_ny.hour < 20: return "Weekend", "dot-closed"
    if 240 <= curr_min < 570: return "Pre-Mkt", "dot-pre"
    if 570 <= curr_min < 960: return "Mkt Open", "dot-reg"
    if 960 <= curr_min < 1200: return "Post-Mkt", "dot-post"
    return "Overnight", "dot-night"

def get_realtime_data():
    tickers_list = "BTC-USD BTDR MARA RIOT CORZ CLSK IREN QQQ ^VIX"
    symbols = tickers_list.split()
    
    try:
        daily = yf.download(tickers_list, period="5d", interval="1d", group_by='ticker', threads=True, progress=False)
        live = yf.download(tickers_list, period="2d", interval="1m", prepost=True, group_by='ticker', threads=True, progress=False)
        
        quotes = {}
        tz_ny = pytz.timezone('America/New_York')
        now_ny = datetime.now(tz_ny)
        today_date = now_ny.date()
        state_tag, state_css = determine_market_state(now_ny)

        for sym in symbols:
            try:
                df_day = daily[sym].dropna(subset=['Close']) if sym in daily else pd.DataFrame()
                df_min = live[sym].dropna(subset=['Close']) if sym in live else pd.DataFrame()
                
                if not df_min.empty: current_price = df_min['Close'].iloc[-1]
                elif not df_day.empty: current_price = df_day['Close'].iloc[-1]
                else: current_price = 0.0

                prev_close = 1.0
                open_price = 0.0
                is_open_today = False
                
                if not df_day.empty:
                    last_day_date = df_day.index[-1].date()
                    if last_day_date == today_date:
                        is_open_today = True
                        open_price = df_day['Open'].iloc[-1]
                        if len(df_day) >= 2: prev_close = df_day['Close'].iloc[-2]
                        else: prev_close = df_day['Open'].iloc[-1]
                    else:
                        prev_close = df_day['Close'].iloc[-1]
                        open_price = prev_close
                
                pct = ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0
                quotes[sym] = {
                    "price": current_price, "pct": pct, "prev": prev_close,
                    "open": open_price, "tag": state_tag, "css": state_css,
                    "is_open_today": is_open_today
                }
            except:
                quotes[sym] = {"price": 0, "pct": 0, "prev": 1, "open": 0, "tag": "ERR", "css": "dot-closed", "is_open_today": False}
        
        try:
            fng = int(requests.get("https://api.alternative.me/fng/", timeout=0.8).json()['data'][0]['value'])
        except:
            fng = 50
            
        return quotes, fng
    except:
        return None, 50

# --- 5. 仪表盘展示 ---
@st.fragment(run_every=10)
def show_live_dashboard():
    quotes, fng_val = get_realtime_data()
    ai_model, factors, ai_status = run_grandmaster_analytics()
    
    if not quotes:
        st.warning("📡 连接中 (Initializing)...")
        time.sleep(1)
        st.rerun()
        return

    btc = quotes.get('BTC-USD', {'pct': 0, 'price': 0})
    qqq = quotes.get('QQQ', {'pct': 0})
    vix = quotes.get('^VIX', {'price': 20, 'pct': 0})
    btdr = quotes.get('BTDR', {'price': 0})
    
    # 状态栏
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny).strftime('%H:%M:%S')
    regime_tag = factors['regime']
    badge_class = "badge-trend" if regime_tag == "Trend" else "badge-chop"
    # 动态引擎标识
    st.markdown(f"<div class='time-bar'>美东 {now_ny} &nbsp;|&nbsp; 状态: <span class='{badge_class}'>{regime_tag}</span> &nbsp;|&nbsp; 引擎: <b>{ai_status}</b></div>", unsafe_allow_html=True)
    
    # Top Cards
    c1, c2 = st.columns(2)
    with c1: st.markdown(card_html("BTC (USD)", f"${btc['price']:,.0f}", f"{btc['pct']:+.2f}%", btc['pct']), unsafe_allow_html=True)
    with c2: st.markdown(card_html("恐慌指数", f"{fng_val}", None, 0), unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    
    # Peers
    st.caption("⚒️ 矿股板块 Beta")
    cols = st.columns(5)
    peers = ["MARA", "RIOT", "CORZ", "CLSK", "IREN"]
    for i, p in enumerate(peers):
        val = quotes.get(p, {'pct': 0})['pct']
        cols[i].markdown(card_html(p, f"{val:+.1f}%", f"{val:+.1f}%", val), unsafe_allow_html=True)
            
    st.markdown("---")
    
    # BTDR Main
    c3, c4, c5 = st.columns(3)
    status_tag = f"<span class='status-dot {btdr['css']}'></span> <span style='font-size:0.6rem; color:#999'>{btdr['tag']}</span>"
    with c3: st.markdown(card_html("BTDR 现价", f"${btdr['price']:.2f}", f"{btdr['pct']:+.2f}%", btdr['pct'], status_tag), unsafe_allow_html=True)
    
    open_label = "今日开盘" if btdr['is_open_today'] else "预计开盘/昨收"
    open_extra = "" if btdr['is_open_today'] else "(Pending)"
    with c4: st.markdown(card_html(open_label, f"${btdr['open']:.2f}", None, 0, open_extra), unsafe_allow_html=True)

    dist_vwap = ((btdr['price'] - factors['vwap']) / factors['vwap']) * 100 if factors['vwap'] > 0 else 0
    with c5: st.markdown(card_html("机构成本 (VWAP)", f"${factors['vwap']:.2f}", f"{dist_vwap:+.1f}%", dist_vwap), unsafe_allow_html=True)

    # --- 预测模型 ---
    btdr_open_pct = ((btdr['open'] - btdr['prev']) / btdr['prev']) # Gap
    btc_pct_factor = btc['pct'] / 100 
    vol_state_factor = factors['atr_ratio'] 

    mh, ml = ai_model['high'], ai_model['low']
    
    # Regression Eq
    pred_h_ret = mh['intercept'] + (mh['beta_gap'] * btdr_open_pct) + (mh['beta_btc'] * btc_pct_factor) + (mh['beta_vol'] * vol_state_factor)
    pred_l_ret = ml['intercept'] + (ml['beta_gap'] * btdr_open_pct) + (ml['beta_btc'] * btc_pct_factor) + (ml['beta_vol'] * vol_state_factor)
    
    sentiment_adj = (fng_val - 50) * 0.0005
    pred_h_ret += sentiment_adj
    pred_l_ret += sentiment_adj
    
    p_high = btdr['prev'] * (1 + pred_h_ret)
    p_low = btdr['prev'] * (1 + pred_l_ret)
    
    st.markdown("### 🎯 日内阻力/支撑 (WLS)")
    col_h, col_l = st.columns(2)
    h_bg = "#e6fcf5" if btdr['price'] < p_high else "#0ca678"; h_txt = "#087f5b" if btdr['price'] < p_high else "#ffffff"
    l_bg = "#fff5f5" if btdr['price'] > p_low else "#e03131"; l_txt = "#c92a2a" if btdr['price'] > p_low else "#ffffff"
    
    with col_h: st.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background-color: {h_bg}; color: {h_txt}; border: 1px solid #c3fae8;"><div style="font-size: 0.8rem; opacity: 0.8;">阻力位 (High)</div><div style="font-size: 1.5rem; font-weight: bold;">${p_high:.2f}</div></div></div>""", unsafe_allow_html=True)
    with col_l: st.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background-color: {l_bg}; color: {l_txt}; border: 1px solid #ffc9c9;"><div style="font-size: 0.8rem; opacity: 0.8;">支撑位 (Low)</div><div style="font-size: 1.5rem; font-weight: bold;">${p_low:.2f}</div></div></div>""", unsafe_allow_html=True)

    # --- 因子面板 (使用卡尔曼 Filter 后的 Beta) ---
    st.markdown("---")
    st.markdown("### 🌍 宏观环境 (Macro)")
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.markdown(factor_html("QQQ (纳指)", f"{qqq['pct']:+.2f}%", "Market", qqq['pct'], "科技股大盘风向标。"), unsafe_allow_html=True)
    with m2: st.markdown(factor_html("VIX (恐慌)", f"{vix['price']:.1f}", "Risk", 0, "市场恐慌指数，>25需警惕。", reverse_color=True), unsafe_allow_html=True)
    with m3: st.markdown(factor_html("Beta (BTC)", f"{factors['beta_btc']:.2f}", "Kalman", 0, "动态 Beta (卡尔曼滤波优化)，无滞后。"), unsafe_allow_html=True)
    with m4: st.markdown(factor_html("Beta (QQQ)", f"{factors['beta_qqq']:.2f}", "Kalman", 0, "动态 Beta (卡尔曼滤波优化)，无滞后。"), unsafe_allow_html=True)
    
    st.markdown("### 🔬 微观结构 (Micro)")
    mi1, mi2, mi3, mi4 = st.columns(4)
    
    # Drift 计算使用卡尔曼 Beta，更精准
    drift_est = (btc['pct']/100 * factors['beta_btc'] * 0.4) + (qqq['pct']/100 * factors['beta_qqq'] * 0.4)
    if abs(dist_vwap) > 10: drift_est -= (dist_vwap/100) * 0.05
    rsi_val = factors['rsi']
    rsi_status = "O/B" if rsi_val > 70 else ("O/S" if rsi_val < 30 else "Neu")

    with mi1: st.markdown(factor_html("ADX (强度)", f"{factors['adx']:.1f}", factors['regime'], 1 if factors['adx']>25 else -1, "趋势强度指标，>25为趋势。"), unsafe_allow_html=True)
    with mi2: st.markdown(factor_html("RSI (14d)", f"{rsi_val:.0f}", rsi_status, 0, "强弱指标，>70超买，<30超卖。"), unsafe_allow_html=True)
    with mi3: st.markdown(factor_html("Implied Vol", f"{factors['vol_base']*100:.1f}%", "Risk", 0, "预测波动率 (基于 EWM Std)。"), unsafe_allow_html=True)
    with mi4: st.markdown(factor_html("Exp. Drift", f"{drift_est*100:+.2f}%", "Day", drift_est, "基于动态 Beta 推算的今日动能。"), unsafe_allow_html=True)

    # --- 概率推演 (Student-t + Mean Reversion) ---
    st.markdown("### ☁️ 概率推演 (Fat-Tailed)")
    
    current_vol = factors['vol_base']
    long_term_vol = 0.05 
    drift = drift_est
    
    sims = 1500
    days = 5
    dt = 1
    
    price_paths = np.zeros((sims, days + 1))
    price_paths[:, 0] = btdr['price']
    
    kappa = 0.1 
    sim_vol = np.full(sims, current_vol)
    
    for t in range(1, days + 1):
        sim_vol = sim_vol + kappa * (long_term_vol - sim_vol)
        sim_vol = np.maximum(sim_vol, 0.01)
        shocks = np.random.standard_t(df=5, size=sims) # 肥尾分布
        daily_ret = np.exp((drift - 0.5 * sim_vol**2) * dt + sim_vol * np.sqrt(dt) * shocks)
        price_paths[:, t] = price_paths[:, t-1] * daily_ret
        
    percentiles = np.percentile(price_paths, [10, 50, 90], axis=0)
    
    chart_data = pd.DataFrame({
        "Day": np.arange(days + 1),
        "P90": np.round(percentiles[2], 2),
        "P50": np.round(percentiles[1], 2),
        "P10": np.round(percentiles[0], 2)
    })
    
    base = alt.Chart(chart_data).encode(x=alt.X('Day:O', title='未来交易日 (T+)'))
    area = base.mark_area(opacity=0.2, color='#4dabf7').encode(y=alt.Y('P10', title='价格预演 (USD)', scale=alt.Scale(zero=False)), y2='P90')
    l90 = base.mark_line(color='#0ca678', strokeDash=[5,5]).encode(y='P90')
    l50 = base.mark_line(color='#228be6', size=3).encode(y='P50')
    l10 = base.mark_line(color='#d6336c', strokeDash=[5,5]).encode(y='P10')
    
    nearest = alt.selection_point(nearest=True, on='mouseover', fields=['Day'], empty=False)
    points = base.mark_circle(size=60, color="black").encode(
        y='P50', opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip('Day'), alt.Tooltip('P90'), alt.Tooltip('P50'), alt.Tooltip('P10')]
    )
    
    st.altair_chart((area + l90 + l50 + l10 + base.mark_rule(opacity=0).encode(x='Day').add_params(nearest) + points).properties(height=300).interactive(), use_container_width=True)
    st.caption(f"Engine: v9.8 Kalman | Beta Update: Dynamic (Zero-Lag) | Dist: Student-t (df=5)")

# --- 6. 主程序 ---
st.markdown("### ⚡ BTDR 领航员 v9.8 Kalman")
show_live_dashboard()
