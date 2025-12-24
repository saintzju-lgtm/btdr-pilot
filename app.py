import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
import altair as alt
from datetime import datetime, time as dt_time, timedelta
import pytz
from scipy.stats import norm

# --- 1. 页面配置 & 样式 ---
st.set_page_config(page_title="BTDR Pilot v11.0 AI-Core", layout="centered")

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
        overflow: hidden !important; border: 1px solid #f8f9fa; border-radius: 8px;
    }
    
    /* Metric Card */
    .metric-card {
        background-color: #f8f9fa; border: 1px solid #e9ecef;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02); height: 95px; padding: 0 16px;
        display: flex; flex-direction: column; justify-content: center;
        position: relative; transition: all 0.2s;
    }
    .metric-card.has-tooltip { cursor: help; }
    .metric-card.has-tooltip:hover { border-color: #ced4da; }
    
    .metric-label { font-size: 0.75rem; color: #888; margin-bottom: 2px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #212529; line-height: 1.2; }
    .metric-delta { font-size: 0.9rem; font-weight: 600; margin-top: 2px; }
    
    /* Miner Card */
    .miner-card {
        background-color: #fff; border: 1px solid #e9ecef;
        border-radius: 10px; padding: 8px 10px;
        text-align: center; height: 100px;
        display: flex; flex-direction: column; justify-content: space-between;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .miner-sym { font-size: 0.75rem; color: #888; font-weight: 600; margin-bottom: 2px; }
    .miner-price { font-size: 1.1rem; font-weight: 700; color: #212529; }
    .miner-sub { font-size: 0.7rem; display: flex; justify-content: space-between; margin-top: 4px; }
    .miner-pct { font-weight: 600; }
    .miner-turn { color: #868e96; }
    
    /* Factor Box */
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
    
    /* Tooltip Core */
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
    .signal-box:hover .tooltip-text { visibility: visible; opacity: 1; }
    .metric-card:hover .tooltip-text { visibility: visible; opacity: 1; }
    
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
    .badge-ai { background: linear-gradient(90deg, #6366f1, #a855f7); color:white; padding:1px 6px; border-radius:3px; font-size:0.6rem; font-weight:bold;}
    
    .ensemble-bar { height: 4px; width: 100%; display: flex; margin-top: 4px; border-radius: 2px; overflow: hidden; }
    .bar-kalman { background-color: #228be6; transition: width 0.5s; }
    .bar-hist { background-color: #fab005; transition: width 0.5s; }
    .bar-mom { background-color: #fa5252; transition: width 0.5s; }
    .bar-ai { background-color: #be4bdb; transition: width 0.5s; }
    
    .ticket-card {
        border-radius: 10px; padding: 15px; margin-bottom: 10px;
        text-align: left; position: relative; border-left: 5px solid #ccc;
        background: #fff; box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .ticket-buy { border-left-color: #0ca678; background: #f0fff4; }
    .ticket-sell { border-left-color: #e03131; background: #fff5f5; }
    
    .ticket-header { 
        font-size: 0.9rem; font-weight: 800; letter-spacing: 0.5px; 
        text-transform: uppercase; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;
    }
    .ticket-price-row { display: flex; align-items: baseline; margin-bottom: 8px; }
    .ticket-price-label { font-size: 0.8rem; color: #555; width: 80px; }
    .ticket-price-val { font-size: 1.6rem; font-weight: 900; color: #212529; letter-spacing: -0.5px; }
    
    .ticket-meta { display: flex; justify-content: space-between; font-size: 0.75rem; margin-top: 8px; color: #666; border-top: 1px solid rgba(0,0,0,0.05); padding-top: 8px; }
    .prob-container { width: 100%; height: 4px; background: #eee; margin-top: 5px; border-radius: 2px; }
    .prob-fill { height: 100%; border-radius: 2px; }
    .prob-high { background: #2f9e44; } .prob-med { background: #fab005; } .prob-low { background: #ced4da; }
    .tag-smart { background: #228be6; color: white; padding: 1px 5px; border-radius: 4px; font-size: 0.6rem; vertical-align: middle; margin-left: 5px; }
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

# --- 4. 核心计算 (AI & Math Core) ---
def run_kalman_filter(y, x, delta=1e-4):
    n = len(y); beta = np.zeros(n); P = np.zeros(n); beta[0]=1.0; P[0]=1.0; R=0.002; Q=delta/(1-delta)
    for t in range(1, n):
        beta_pred = beta[t-1]; P_pred = P[t-1] + Q
        if x[t] == 0: x[t] = 1e-6
        residual = y[t] - beta_pred * x[t]; S = P_pred * x[t]**2 + R; K = P_pred * x[t] / S
        beta[t] = beta_pred + K * residual; P[t] = (1 - K * x[t]) * P_pred
    return beta[-1]

def calculate_hurst(series):
    """Simple Hurst Exponent proxy to determine trending vs mean reverting."""
    if len(series) < 20: return 0.5
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

@st.cache_data(ttl=300)
def run_grandmaster_analytics():
    default_model = {
        "high": {"intercept": 0, "beta_gap": 0.5, "beta_btc": 0.5, "beta_vol": 0},
        "low": {"intercept": 0, "beta_gap": 0.5, "beta_btc": 0.5, "beta_vol": 0},
        "ensemble_hist_h": 0.05, "ensemble_hist_l": -0.05,
        "ensemble_mom_h": 0.08, "ensemble_mom_l": -0.08,
        "top_peers": ["MARA", "RIOT", "CLSK", "CORZ", "IREN"]
    }
    default_factors = {"vwap": 0, "adx": 20, "regime": "Neutral", "beta_btc": 1.5, "beta_qqq": 1.2, "rsi": 50, "vol_base": 0.05, "atr_ratio": 0.05, "hurst": 0.5}

    try:
        tickers_str = "BTDR BTC-USD QQQ " + " ".join(MINER_POOL)
        data = yf.download(tickers_str, period="6mo", interval="1d", group_by='ticker', threads=True, progress=False)
        if data.empty: return default_model, default_factors, "No Data"

        btdr = data['BTDR'].dropna(); btc = data['BTC-USD'].dropna(); qqq = data['QQQ'].dropna()
        idx = btdr.index.intersection(btc.index).intersection(qqq.index)
        btdr, btc, qqq = btdr.loc[idx], btc.loc[idx], qqq.loc[idx]
        
        if len(btdr) < 30: return default_model, default_factors, "Insufficient Data"

        # Correlation Analysis
        correlations = {}
        for m in MINER_POOL:
            if m in data:
                miner_df = data[m]['Close'].pct_change().tail(30)
                btdr_df = btdr['Close'].pct_change().tail(30)
                common_idx = miner_df.index.intersection(btdr_df.index)
                if len(common_idx) > 10: correlations[m] = miner_df.loc[common_idx].corr(btdr_df.loc[common_idx])
                else: correlations[m] = 0
        top_peers = sorted(correlations, key=correlations.get, reverse=True)[:5]
        default_model["top_peers"] = top_peers

        ret_btdr = btdr['Close'].pct_change().fillna(0).values
        ret_btc = btc['Close'].pct_change().fillna(0).values
        ret_qqq = qqq['Close'].pct_change().fillna(0).values
        
        beta_btc = run_kalman_filter(ret_btdr, ret_btc, delta=1e-4)
        beta_qqq = run_kalman_filter(ret_btdr, ret_qqq, delta=1e-4)
        beta_btc = np.clip(beta_btc, -1, 5); beta_qqq = np.clip(beta_qqq, -1, 4)

        pv = (btdr['Close'] * btdr['Volume'])
        vwap_30d = pv.tail(30).sum() / btdr['Volume'].tail(30).sum()
        
        high, low, close = btdr['High'], btdr['Low'], btdr['Close']
        tr = np.maximum(high - low, np.abs(high - close.shift(1)))
        atr = tr.rolling(14).mean()
        
        # ADX Logic
        up, down = high.diff(), -low.diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0); minus_dm = np.where((down > up) & (down > 0), down, 0)
        atr_s = pd.Series(atr.values, index=btdr.index)
        plus_di = 100 * (pd.Series(plus_dm, index=btdr.index).rolling(14).mean() / atr_s)
        minus_di = 100 * (pd.Series(minus_dm, index=btdr.index).rolling(14).mean() / atr_s)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean().iloc[-1]; adx = 20 if np.isnan(adx) else adx
        
        # RSI Logic
        delta_p = close.diff()
        gain = delta_p.where(delta_p > 0, 0).rolling(14).mean(); loss = -delta_p.where(delta_p < 0, 0).rolling(14).mean()
        rs = gain / loss; rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        vol_base = ret_btdr.std()
        if len(ret_btdr) > 20: vol_base = pd.Series(ret_btdr).ewm(span=20).std().iloc[-1]
        atr_ratio = (atr / close).iloc[-1]

        hurst = calculate_hurst(btdr['Close'].values[-50:])
        
        regime = "Trend" if adx > 25 else ("MeanRev" if hurst < 0.4 else "Chop")

        factors = {"beta_btc": beta_btc, "beta_qqq": beta_qqq, "vwap": vwap_30d, "adx": adx, "regime": regime, "rsi": rsi, "vol_base": vol_base, "atr_ratio": atr_ratio, "hurst": hurst}

        # Weighted Least Squares Regression
        df_reg = pd.DataFrame()
        df_reg['PrevClose'] = btdr['Close'].shift(1); df_reg['Open'] = btdr['Open']
        df_reg['High'] = btdr['High']; df_reg['Low'] = btdr['Low']
        df_reg['Gap'] = (df_reg['Open'] - df_reg['PrevClose']) / df_reg['PrevClose']
        df_reg['BTC_Ret'] = btc['Close'].pct_change()
        df_reg['Vol_State'] = ((btdr['High'] - btdr['Low']) / btdr['Open']).shift(1)
        df_reg['Target_High'] = (df_reg['High'] - df_reg['PrevClose']) / df_reg['PrevClose']
        df_reg['Target_Low'] = (df_reg['Low'] - df_reg['PrevClose']) / df_reg['PrevClose']
        df_reg = df_reg.dropna().tail(90)
        
        weights = np.exp(np.linspace(-0.05 * len(df_reg), 0, len(df_reg))); W = np.diag(weights)
        X = np.column_stack([np.ones(len(df_reg)), df_reg['Gap'].values, df_reg['BTC_Ret'].values, df_reg['Vol_State'].values])
        Y_h = df_reg['Target_High'].values; Y_l = df_reg['Target_Low'].values
        XtWX = X.T @ W @ X
        theta_h = np.linalg.lstsq(XtWX, X.T @ W @ Y_h, rcond=None)[0]; theta_l = np.linalg.lstsq(XtWX, X.T @ W @ Y_l, rcond=None)[0]

        final_model = {
            "high": {"intercept": theta_h[0], "beta_gap": theta_h[1], "beta_btc": theta_h[2], "beta_vol": theta_h[3]},
            "low": {"intercept": theta_l[0], "beta_gap": theta_l[1], "beta_btc": theta_l[2], "beta_vol": theta_l[3]},
            "ensemble_hist_h": df_reg['Target_High'].tail(10).mean(), "ensemble_hist_l": df_reg['Target_Low'].tail(10).mean(),
            "ensemble_mom_h": df_reg['Target_High'].tail(3).max(), "ensemble_mom_l": df_reg['Target_Low'].tail(3).min(),
            "top_peers": top_peers
        }
        return final_model, factors, "v11.0 AI-Core"
    except Exception as e:
        print(f"Error: {e}")
        return default_model, default_factors, "Offline"

# --- 5. 实时数据与 AI 引擎 ---
def determine_market_state(now_ny):
    weekday = now_ny.weekday(); curr_min = now_ny.hour * 60 + now_ny.minute
    if weekday == 5: return "Weekend", "dot-closed"
    if weekday == 6 and now_ny.hour < 20: return "Weekend", "dot-closed"
    if 240 <= curr_min < 570: return "Pre-Mkt", "dot-pre"
    if 570 <= curr_min < 960: return "Mkt Open", "dot-reg"
    if 960 <= curr_min < 1200: return "Post-Mkt", "dot-post"
    return "Overnight", "dot-night"

def get_ai_weights(regime, rsi, volatility_state):
    """
    AI 引擎核心：根据市场状态动态分配预测模型的权重
    Returns: (w_kalman, w_hist, w_mom, w_ai)
    """
    if regime == "Trend":
        # 趋势行情：加重动能(Mom)和AI波动率预测，减少历史回归
        return 0.20, 0.05, 0.35, 0.40
    elif regime == "MeanRev":
        # 均值回归：加重历史统计(Hist)和Kalman，减少动能
        return 0.35, 0.35, 0.10, 0.20
    else:
        # 震荡/混沌 (Chop)：加重AI波动率范围控制
        return 0.25, 0.15, 0.10, 0.50

def get_realtime_data():
    tickers_list = "BTC-USD BTDR QQQ ^VIX " + " ".join(MINER_POOL)
    symbols = tickers_list.split()
    try:
        daily = yf.download(tickers_list, period="5d", interval="1d", group_by='ticker', threads=True, progress=False)
        live = yf.download(tickers_list, period="2d", interval="1m", prepost=True, group_by='ticker', threads=True, progress=False)
        
        quotes = {}
        tz_ny = pytz.timezone('America/New_York'); now_ny = datetime.now(tz_ny); state_tag, state_css = determine_market_state(now_ny)
        live_volatility = 0.01 
        
        for sym in symbols:
            try:
                df_day = daily[sym].dropna(subset=['Close']) if sym in daily else pd.DataFrame()
                df_min = live[sym].dropna(subset=['Close']) if sym in live else pd.DataFrame()
                
                current_volume = 0
                current_price = 0.0
                prev_close = 1.0; open_price = 0.0; is_open_today = False
                
                if not df_min.empty: 
                    current_price = df_min['Close'].iloc[-1]
                    if 'Volume' in df_min.columns: current_volume = df_min['Volume'].sum()
                    
                    if sym == 'BTDR':
                        # GARCH-Lite: 计算近期分钟级波动率
                        recent_min_std = df_min['Close'].tail(60).pct_change().std()
                        if np.isnan(recent_min_std) or recent_min_std == 0: 
                            live_volatility = 0.005 # Fallback
                        else:
                            live_volatility = recent_min_std * np.sqrt(60) # Annualize to hourly-ish scale
                        
                elif not df_day.empty: 
                    current_price = df_day['Close'].iloc[-1]; current_volume = df_day['Volume'].iloc[-1]
                
                if not df_day.empty:
                    last_day_date = df_day.index[-1].date()
                    if last_day_date == now_ny.date():
                        is_open_today = True; open_price = df_day['Open'].iloc[-1]
                        if len(df_day) >= 2: prev_close = df_day['Close'].iloc[-2]
                        else: prev_close = df_day['Open'].iloc[-1]
                    else: 
                        prev_close = df_day['Close'].iloc[-1]; open_price = prev_close
                
                pct = ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0
                quotes[sym] = {"price": current_price, "pct": pct, "prev": prev_close, "open": open_price, "volume": current_volume, "tag": state_tag, "css": state_css, "is_open_today": is_open_today}
            except Exception as e: 
                quotes[sym] = {"price": 0, "pct": 0, "prev": 1, "open": 0, "volume": 0, "tag": "ERR", "css": "dot-closed", "is_open_today": False}
        
        try: fng = int(requests.get("https://api.alternative.me/fng/", timeout=1.0).json()['data'][0]['value'])
        except: fng = 50
        return quotes, fng, live_volatility
    except: return None, 50, 0.01

# --- 6. 仪表盘展示 ---
@st.fragment(run_every=15)
def show_live_dashboard():
    quotes, fng_val, live_vol_btdr = get_realtime_data()
    ai_model, factors, ai_status = run_grandmaster_analytics()
    
    if not quotes: 
        st.warning("📡 建立安全连接中 (Initializing Secure Link)...")
        time.sleep(1)
        st.rerun()
        return

    btc = quotes.get('BTC-USD', {'pct': 0, 'price': 0}); qqq = quotes.get('QQQ', {'pct': 0})
    vix = quotes.get('^VIX', {'price': 20, 'pct': 0}); btdr = quotes.get('BTDR', {'price': 0})

    dist_vwap = ((btdr['price'] - factors['vwap']) / factors['vwap']) * 100 if factors['vwap'] > 0 else 0
    drift_est = (btc['pct']/100 * factors['beta_btc'] * 0.4) + (qqq['pct']/100 * factors['beta_qqq'] * 0.4)
    if abs(dist_vwap) > 10: drift_est -= (dist_vwap/100) * 0.05
    
    tz_ny = pytz.timezone('America/New_York'); now_ny = datetime.now(tz_ny).strftime('%H:%M:%S')
    regime_tag = factors['regime']; badge_class = "badge-ai"
    
    # 动态获取 AI 权重
    w_kalman, w_hist, w_mom, w_ai = get_ai_weights(regime_tag, factors['rsi'], vix['price'])
    
    st.markdown(f"<div class='time-bar'>美东 {now_ny} &nbsp;|&nbsp; AI 模式: <span class='{badge_class}'>{regime_tag}</span> &nbsp;|&nbsp; 引擎: <b>{ai_status}</b></div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1: st.markdown(card_html("BTC (USD)", f"${btc['price']:,.0f}", f"{btc['pct']:+.2f}%", btc['pct']), unsafe_allow_html=True)
    
    fng_tooltip = "0-24: 极度恐慌 (潜在买点)\n25-49: 恐慌\n50-74: 贪婪\n75-100: 极度贪婪 (风险较高)"
    with c2: st.markdown(card_html("恐慌指数 (F&G)", f"{fng_val}", None, 0, tooltip_text=fng_tooltip), unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    
    st.caption("⚒️ 矿股板块 Beta (Correlation Top 5)")
    cols = st.columns(5)
    top_peers = ai_model.get("top_peers", ["MARA", "RIOT", "CLSK", "CORZ", "IREN"])
    for i, p in enumerate(top_peers):
        data = quotes.get(p, {'pct': 0, 'price': 0, 'volume': 0})
        shares_m = MINER_SHARES.get(p, 200)
        turnover_rate = (data['volume'] / (shares_m * 1000000)) * 100
        cols[i].markdown(miner_card_html(p, data['price'], data['pct'], turnover_rate), unsafe_allow_html=True)
            
    st.markdown("---")
    
    c3, c4, c5 = st.columns(3)
    status_tag = f"<span class='status-dot {btdr['css']}'></span> <span style='font-size:0.6rem; color:#999'>{btdr['tag']}</span>"
    with c3: st.markdown(card_html("BTDR 现价", f"${btdr['price']:.2f}", f"{btdr['pct']:+.2f}%", btdr['pct'], status_tag), unsafe_allow_html=True)
    open_label = "今日开盘" if btdr['is_open_today'] else "预计开盘/昨收"
    open_extra = "" if btdr['is_open_today'] else "(Pending)"
    with c4: st.markdown(card_html(open_label, f"${btdr['open']:.2f}", None, 0, open_extra), unsafe_allow_html=True)
    with c5: st.markdown(card_html("机构成本 (VWAP)", f"${factors['vwap']:.2f}", f"{dist_vwap:+.1f}%", dist_vwap), unsafe_allow_html=True)

    # --- 预测模型计算 ---
    current_gap_pct = ((btdr['price'] - btdr['prev']) / btdr['prev']) if btdr['price'] > 0 else ((btdr['open'] - btdr['prev']) / btdr['prev'])
    btc_pct_factor = btc['pct'] / 100; vol_state_factor = factors['atr_ratio'] 
    mh, ml = ai_model['high'], ai_model['low']
    
    pred_h_kalman = mh['intercept'] + (mh['beta_gap'] * current_gap_pct) + (mh['beta_btc'] * btc_pct_factor) + (mh['beta_vol'] * vol_state_factor)
    pred_l_kalman = ml['intercept'] + (ml['beta_gap'] * current_gap_pct) + (ml['beta_btc'] * btc_pct_factor) + (ml['beta_vol'] * vol_state_factor)
    
    # AI 波动率修正 (考虑 VIX Skew)
    vix_factor = max(1.0, vix['price'] / 20.0)
    live_vol_pct = (live_vol_btdr / btdr['price']) * vix_factor if btdr['price'] > 0 else 0.02
    
    ai_upper_bound_pct = (btdr['price'] * (1 + 2.5 * live_vol_pct) - btdr['prev']) / btdr['prev']
    ai_lower_bound_pct = (btdr['price'] * (1 - 2.5 * live_vol_pct) - btdr['prev']) / btdr['prev']
    
    final_h_ret = (w_kalman * pred_h_kalman) + (w_hist * ai_model['ensemble_hist_h']) + (w_mom * ai_model['ensemble_mom_h']) + (w_ai * ai_upper_bound_pct)
    final_l_ret = (w_kalman * pred_l_kalman) + (w_hist * ai_model['ensemble_hist_l']) + (w_mom * ai_model['ensemble_mom_l']) + (w_ai * ai_lower_bound_pct)
    
    sentiment_adj = (fng_val - 50) * 0.0005; final_h_ret += sentiment_adj; final_l_ret += sentiment_adj
    p_high = btdr['prev'] * (1 + final_h_ret); p_low = btdr['prev'] * (1 + final_l_ret)

    curr_p = btdr['price']
    atr_buffer = live_vol_btdr * 0.6  # 稍微收紧 Buffer 以提高成交率
    
    buy_entry = p_low + atr_buffer
    buy_stop = buy_entry - (live_vol_btdr * 2.5)
    buy_target = p_high - atr_buffer
    buy_rr = (buy_target - buy_entry) / (buy_entry - buy_stop) if (buy_entry - buy_stop) > 0 else 0
    
    # 使用正态分布计算成交概率
    z_buy = (curr_p - buy_entry) / (live_vol_btdr * 8)
    buy_prob = max(min((1 - norm.cdf(z_buy)) * 100 * 2, 95), 5)
    buy_prob_class = "prob-high" if buy_prob > 60 else ("prob-med" if buy_prob > 30 else "prob-low")

    sell_entry = p_high - atr_buffer
    sell_stop = sell_entry + (live_vol_btdr * 2.5)
    sell_target = p_low + atr_buffer
    sell_rr = (sell_entry - sell_target) / (sell_stop - sell_entry) if (sell_stop - sell_entry) > 0 else 0
    
    z_sell = (sell_entry - curr_p) / (live_vol_btdr * 8)
    sell_prob = max(min((1 - norm.cdf(z_sell)) * 100 * 2, 95), 5)
    sell_prob_class = "prob-high" if sell_prob > 60 else ("prob-med" if sell_prob > 30 else "prob-low")

    tick1, tick2 = st.columns(2)
    
    with tick1:
        st.markdown(f"""
        <div class="ticket-card ticket-buy">
            <div class="ticket-header" style="color:#0ca678;">🟢 BUY LIMIT <span class="tag-smart">SMART</span></div>
            <div class="ticket-price-row"><span class="ticket-price-label">挂单价</span><span class="ticket-price-val">${buy_entry:.2f}</span></div>
            <div class="ticket-price-row"><span class="ticket-price-label">止损价</span><span class="ticket-price-val" style="color:#e03131; font-size:1.1rem;">${buy_stop:.2f}</span></div>
            <div class="ticket-price-row"><span class="ticket-price-label">目标价</span><span class="ticket-price-val" style="color:#1c7ed6; font-size:1.1rem;">${buy_target:.2f}</span></div>
            <div class="ticket-meta">
                <span>R/R: <b>1:{buy_rr:.1f}</b></span>
                <span>概率: <b>{buy_prob:.0f}%</b></span>
            </div>
            <div class="prob-container"><div class="prob-fill {buy_prob_class}" style="width:{buy_prob}%"></div></div>
        </div>
        """, unsafe_allow_html=True)
        
    with tick2:
        st.markdown(f"""
        <div class="ticket-card ticket-sell">
            <div class="ticket-header" style="color:#e03131;">🔴 SELL LIMIT <span class="tag-smart">SMART</span></div>
            <div class="ticket-price-row"><span class="ticket-price-label">挂单价</span><span class="ticket-price-val">${sell_entry:.2f}</span></div>
            <div class="ticket-price-row"><span class="ticket-price-label">止损价</span><span class="ticket-price-val" style="color:#e03131; font-size:1.1rem;">${sell_stop:.2f}</span></div>
            <div class="ticket-price-row"><span class="ticket-price-label">目标价</span><span class="ticket-price-val" style="color:#1c7ed6; font-size:1.1rem;">${sell_target:.2f}</span></div>
            <div class="ticket-meta">
                <span>R/R: <b>1:{sell_rr:.1f}</b></span>
                <span>概率: <b>{sell_prob:.0f}%</b></span>
            </div>
            <div class="prob-container"><div class="prob-fill {sell_prob_class}" style="width:{sell_prob}%"></div></div>
        </div>
        """, unsafe_allow_html=True)

    # 动态权重条
    st.markdown(f"""
    <div style="font-size:0.7rem; color:#888; margin-bottom:2px; display:flex; justify-content:space-between;">
        <span>🟦 Kalman ({w_kalman:.0%})</span>
        <span>🟨 History ({w_hist:.0%})</span>
        <span>🟥 Momentum ({w_mom:.0%})</span>
        <span>🟪 AI Volatility ({w_ai:.0%})</span>
    </div>
    <div class="ensemble-bar">
        <div class="bar-kalman" style="width: {w_kalman*100}%"></div>
        <div class="bar-hist" style="width: {w_hist*100}%"></div>
        <div class="bar-mom" style="width: {w_mom*100}%"></div>
        <div class="bar-ai" style="width: {w_ai*100}%"></div>
    </div>
    <div style="margin-bottom:10px;"></div>
    """, unsafe_allow_html=True)
    
    col_h, col_l = st.columns(2)
    h_bg = "#e6fcf5" if btdr['price'] < p_high else "#0ca678"; h_txt = "#087f5b" if btdr['price'] < p_high else "#ffffff"
    l_bg = "#fff5f5" if btdr['price'] > p_low else "#e03131"; l_txt = "#c92a2a" if btdr['price'] > p_low else "#ffffff"
    with col_h: st.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background-color: {h_bg}; color: {h_txt}; border: 1px solid #c3fae8;"><div style="font-size: 0.8rem; opacity: 0.8;">理论阻力 (High)</div><div style="font-size: 1.5rem; font-weight: bold;">${p_high:.2f}</div></div></div>""", unsafe_allow_html=True)
    with col_l: st.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background-color: {l_bg}; color: {l_txt}; border: 1px solid #ffc9c9;"><div style="font-size: 0.8rem; opacity: 0.8;">理论支撑 (Low)</div><div style="font-size: 1.5rem; font-weight: bold;">${p_low:.2f}</div></div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🌍 宏观环境 (Macro Data)")
    ma1, ma2, ma3, ma4 = st.columns(4)
    with ma1: st.markdown(factor_html("QQQ (纳指)", f"{qqq['pct']:+.2f}%", "Market", qqq['pct'], "科技股大盘风向标。"), unsafe_allow_html=True)
    with ma2: st.markdown(factor_html("VIX (恐慌)", f"{vix['price']:.1f}", "Risk", 0, "市场恐慌指数，>25需警惕。", reverse_color=True), unsafe_allow_html=True)
    with ma3: st.markdown(factor_html("Beta (BTC)", f"{factors['beta_btc']:.2f}", "Kalman", 0, "动态 Beta"), unsafe_allow_html=True)
    with ma4: st.markdown(factor_html("Beta (QQQ)", f"{factors['beta_qqq']:.2f}", "Kalman", 0, "动态 Beta"), unsafe_allow_html=True)

    st.markdown("### 🔬 微观结构 (Micro Data)")
    mi1, mi2, mi3, mi4 = st.columns(4)
    rsi_val = factors['rsi']; rsi_status = "O/B" if rsi_val > 70 else ("O/S" if rsi_val < 30 else "Neu")
    with mi1: st.markdown(factor_html("ADX (强度)", f"{factors['adx']:.1f}", factors['regime'], 1 if factors['adx']>25 else -1, "趋势强度指标，>25为趋势。"), unsafe_allow_html=True)
    with mi2: st.markdown(factor_html("RSI (14d)", f"{rsi_val:.0f}", rsi_status, 0, "强弱指标，>70超买，<30超卖。"), unsafe_allow_html=True)
    with mi3: st.markdown(factor_html("Hurst Exp", f"{factors['hurst']:.2f}", "Fractal", 0, "分形维数：<0.5均值回归，>0.5趋势。"), unsafe_allow_html=True)
    with mi4: st.markdown(factor_html("Exp. Drift", f"{drift_est*100:+.2f}%", "Day", drift_est, "当日预期动能"), unsafe_allow_html=True)
    
    st.markdown("### ☁️ 概率推演 (AI Probability)")
    
    # 概率密度图 (Bell Curve for Target)
    x_axis = np.linspace(p_low * 0.95, p_high * 1.05, 100)
    pdf_high = norm.pdf(x_axis, p_high, live_vol_btdr * btdr['price'])
    pdf_low = norm.pdf(x_axis, p_low, live_vol_btdr * btdr['price'])
    
    pdf_data = pd.DataFrame({'Price': x_axis, 'Resistance Probability': pdf_high, 'Support Probability': pdf_low})
    pdf_data = pdf_data.melt('Price', var_name='Type', value_name='Density')
    
    chart_pdf = alt.Chart(pdf_data).mark_area(opacity=0.3).encode(
        x=alt.X('Price', title='Price Levels', scale=alt.Scale(zero=False)),
        y='Density',
        color=alt.Color('Type', scale=alt.Scale(domain=['Resistance Probability', 'Support Probability'], range=['#d6336c', '#0ca678']))
    ).properties(height=180)
    
    # Monte Carlo (Keep existing)
    current_vol = factors['vol_base']; long_term_vol = 0.05; drift = drift_est
    sims, days, dt = 1500, 5, 1
    price_paths = np.zeros((sims, days + 1)); price_paths[:, 0] = btdr['price']
    kappa = 0.1; sim_vol = np.full(sims, current_vol)
    for t in range(1, days + 1):
        sim_vol = sim_vol + kappa * (long_term_vol - sim_vol); sim_vol = np.maximum(sim_vol, 0.01)
        shocks = np.random.standard_t(df=5, size=sims)
        daily_ret = np.exp((drift - 0.5 * sim_vol**2) * dt + sim_vol * np.sqrt(dt) * shocks)
        price_paths[:, t] = price_paths[:, t-1] * daily_ret
        
    percentiles = np.percentile(price_paths, [10, 50, 90], axis=0)
    chart_data = pd.DataFrame({"Day": np.arange(days+1), "P90": np.round(percentiles[2], 2), "P50": np.round(percentiles[1], 2), "P10": np.round(percentiles[0], 2)})
    base = alt.Chart(chart_data).encode(x=alt.X('Day:O', title='未来交易日 (T+)'))
    area = base.mark_area(opacity=0.1, color='#4dabf7').encode(y=alt.Y('P10', title='价格预演 (USD)', scale=alt.Scale(zero=False)), y2='P90')
    l90 = base.mark_line(color='#0ca678', strokeDash=[5,5]).encode(y='P90')
    l50 = base.mark_line(color='#228be6', size=3).encode(y='P50')
    l10 = base.mark_line(color='#d6336c', strokeDash=[5,5]).encode(y='P10')
    
    st.altair_chart((chart_pdf & (area + l90 + l50 + l10).properties(height=200)).resolve_scale(x='independent'), use_container_width=True)
    st.caption(f"AI Engine: v11.0 | Real-time Volatility: {live_vol_btdr:.4f}")

st.markdown("### ⚡ BTDR 领航员 v11.0 AI-Core")
show_live_dashboard()
