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

# --- 1. 页面配置 & 样式 ---
st.set_page_config(page_title="BTDR Pilot v10.9 Stable", layout="centered")

CUSTOM_CSS = """
<style>
    html { overflow-y: scroll; }
    .stApp > header { display: none; }
    .stApp { margin-top: -30px; background-color: #ffffff; }
    div[data-testid="stStatusWidget"] { visibility: hidden; }
    h1, h2, h3, div, p, span { color: #212529 !important; font-family: sans-serif !important; }
    div[data-testid="stAltairChart"] { height: 320px !important; min-height: 320px !important; border: 1px solid #f8f9fa; }
    
    /* Card Styles */
    .metric-card { background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 12px; height: 95px; padding: 0 16px; display: flex; flex-direction: column; justify-content: center; position: relative; }
    .metric-card:hover { border-color: #ced4da; }
    .metric-label { font-size: 0.75rem; color: #888; margin-bottom: 2px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #212529; line-height: 1.2; }
    .metric-delta { font-size: 0.9rem; font-weight: 600; margin-top: 2px; }
    
    .miner-card { background-color: #fff; border: 1px solid #e9ecef; border-radius: 10px; padding: 8px 10px; text-align: center; height: 100px; display: flex; flex-direction: column; justify-content: space-between; }
    .miner-sym { font-size: 0.75rem; color: #888; font-weight: 600; }
    .miner-price { font-size: 1.1rem; font-weight: 700; color: #212529; }
    .miner-sub { font-size: 0.7rem; display: flex; justify-content: space-between; margin-top: 4px; }
    
    .factor-box { background: #fff; border: 1px solid #eee; border-radius: 8px; padding: 6px; text-align: center; height: 75px; display: flex; flex-direction: column; justify-content: center; position: relative; cursor: help; }
    .factor-box:hover { border-color: #ced4da; transform: translateY(-1px); }
    .factor-title { font-size: 0.65rem; color: #999; text-transform: uppercase; }
    .factor-val { font-size: 1.1rem; font-weight: bold; color: #495057; margin: 2px 0; }
    
    .tooltip-text { visibility: hidden; width: 180px; background-color: rgba(33,37,41,0.95); color: #fff !important; text-align: center; border-radius: 6px; padding: 8px; position: absolute; z-index: 999; bottom: 110%; left: 50%; margin-left: -90px; opacity: 0; transition: opacity 0.3s; font-size: 0.7rem !important; pointer-events: none; }
    .tooltip-text::after { content: ""; position: absolute; top: 100%; left: 50%; margin-left: -5px; border-width: 5px; border-style: solid; border-color: rgba(33,37,41,0.95) transparent transparent transparent; }
    .factor-box:hover .tooltip-text, .metric-card:hover .tooltip-text { visibility: visible; opacity: 1; }
    
    .color-up { color: #0ca678; } .color-down { color: #d6336c; }
    .status-dot { height: 6px; width: 6px; border-radius: 50%; display: inline-block; margin-left: 6px; margin-bottom: 2px; }
    .dot-pre { background-color: #f59f00; box-shadow: 0 0 4px #f59f00; }
    .dot-reg { background-color: #0ca678; box-shadow: 0 0 4px #0ca678; }
    .dot-post { background-color: #1c7ed6; box-shadow: 0 0 4px #1c7ed6; }
    .dot-night { background-color: #7048e8; } .dot-closed { background-color: #adb5bd; }
    
    .pred-box { padding: 0 10px; border-radius: 12px; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center; }
    .time-bar { font-size: 0.75rem; color: #999; text-align: center; margin-bottom: 20px; padding: 6px; background: #fafafa; border-radius: 6px; }
    .badge-trend { background:#fd7e14; color:white; padding:1px 4px; border-radius:3px; font-size:0.6rem; }
    .badge-chop { background:#868e96; color:white; padding:1px 4px; border-radius:3px; font-size:0.6rem; }
    
    .ensemble-bar { height: 4px; width: 100%; display: flex; margin-top: 4px; border-radius: 2px; overflow: hidden; }
    .bar-kalman { background: #228be6; width: 40%; } .bar-hist { background: #fab005; width: 25%; } .bar-mom { background: #fa5252; width: 15%; } .bar-ai { background: #be4bdb; width: 20%; }
    
    .ticket-card { border-radius: 10px; padding: 15px; margin-bottom: 10px; text-align: left; position: relative; border-left: 5px solid #ccc; background: #fff; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .ticket-buy { border-left-color: #0ca678; background: #f0fff4; } .ticket-sell { border-left-color: #e03131; background: #fff5f5; }
    .ticket-header { font-size: 0.9rem; font-weight: 800; text-transform: uppercase; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }
    .ticket-price-row { display: flex; align-items: baseline; margin-bottom: 8px; }
    .ticket-price-label { font-size: 0.8rem; color: #555; width: 80px; }
    .ticket-price-val { font-size: 1.6rem; font-weight: 900; color: #212529; letter-spacing: -0.5px; }
    .prob-container { width: 100%; height: 4px; background: #eee; margin-top: 5px; border-radius: 2px; }
    .prob-fill { height: 100%; border-radius: 2px; }
    .prob-high { background: #2f9e44; } .prob-med { background: #fab005; } .prob-low { background: #ced4da; }
    .tag-smart { background: #228be6; color: white; padding: 1px 5px; border-radius: 4px; font-size: 0.6rem; margin-left: 5px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- 2. 基础配置 ---
MINER_SHARES = {"MARA": 300, "RIOT": 330, "CLSK": 220, "CORZ": 190, "IREN": 180, "WULF": 410, "CIFR": 300, "HUT": 100}
MINER_POOL = list(MINER_SHARES.keys())

# --- 3. 辅助函数 ---
def card_html(label, value_str, delta_str=None, delta_val=0, extra_tag="", tooltip_text=None):
    delta_html = f"<div class='metric-delta {'color-up' if delta_val>=0 else 'color-down'}'>{delta_str}</div>" if delta_str else ""
    tooltip_html = f"<div class='tooltip-text'>{tooltip_text}</div>" if tooltip_text else ""
    return f"""<div class="metric-card"><div class="metric-label">{label} {extra_tag}</div><div class="metric-value">{value_str}</div>{delta_html}{tooltip_html}</div>"""

def factor_html(title, val, delta_str, delta_val, tooltip_text, reverse_color=False):
    is_pos = delta_val >= 0
    if reverse_color: is_pos = not is_pos
    return f"""<div class="factor-box"><div class="tooltip-text">{tooltip_text}</div><div class="factor-title">{title}</div><div class="factor-val">{val}</div><div class="factor-sub {'color-up' if is_pos else 'color-down'}">{delta_str}</div></div>"""

def miner_card_html(sym, price, pct, turnover):
    c = "color-up" if pct >= 0 else "color-down"
    return f"""<div class="miner-card"><div class="miner-sym">{sym}</div><div class="miner-price {c}">${price:.2f}</div><div class="miner-sub"><span class="miner-pct {c}">{pct:+.1f}%</span><span class="miner-turn">换 {turnover:.1f}%</span></div></div>"""

# --- 4. 核心计算 (Kalman & AI) ---
def run_kalman_filter(y, x, delta=1e-4):
    n = len(y); beta = np.zeros(n); P = np.zeros(n); beta[0]=1.0; P[0]=1.0; R=0.002; Q=delta/(1-delta)
    for t in range(1, n):
        beta_pred = beta[t-1]; P_pred = P[t-1] + Q
        if x[t] == 0: x[t] = 1e-6
        residual = y[t] - beta_pred * x[t]
        S = P_pred * x[t]**2 + R; K = P_pred * x[t] / S
        beta[t] = beta_pred + K * residual
        P[t] = (1 - K * x[t]) * P_pred
    return beta[-1]

@st.cache_data(ttl=600)
def run_grandmaster_analytics():
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
        if data.empty: return default_model, default_factors, "No Data"

        btdr = data['BTDR'].dropna(); btc = data['BTC-USD'].dropna(); qqq = data['QQQ'].dropna()
        idx = btdr.index.intersection(btc.index).intersection(qqq.index)
        btdr, btc, qqq = btdr.loc[idx], btc.loc[idx], qqq.loc[idx]
        if len(btdr) < 30: return default_model, default_factors, "Insufficient Data"

        correlations = {}
        for m in MINER_POOL:
            if m in data:
                m_df = data[m]['Close'].pct_change().tail(30)
                b_df = btdr['Close'].pct_change().tail(30)
                c_idx = m_df.index.intersection(b_df.index)
                correlations[m] = m_df.loc[c_idx].corr(b_df.loc[c_idx]) if len(c_idx) > 10 else 0
        top_peers = sorted(correlations, key=correlations.get, reverse=True)[:5]
        default_model["top_peers"] = top_peers

        ret_btdr = btdr['Close'].pct_change().fillna(0).values
        ret_btc = btc['Close'].pct_change().fillna(0).values
        ret_qqq = qqq['Close'].pct_change().fillna(0).values
        
        beta_btc = np.clip(run_kalman_filter(ret_btdr, ret_btc), -1, 5)
        beta_qqq = np.clip(run_kalman_filter(ret_btdr, ret_qqq), -1, 4)

        pv = (btdr['Close'] * btdr['Volume'])
        vwap_30d = pv.tail(30).sum() / btdr['Volume'].tail(30).sum()
        
        tr = np.maximum(btdr['High'] - btdr['Low'], np.abs(btdr['High'] - btdr['Close'].shift(1)))
        atr = tr.rolling(14).mean()
        
        up, down = btdr['High'].diff(), -btdr['Low'].diff()
        p_dm = np.where((up > down) & (up > 0), up, 0); m_dm = np.where((down > up) & (down > 0), down, 0)
        atr_s = pd.Series(atr.values, index=btdr.index)
        p_di = 100 * (pd.Series(p_dm, index=btdr.index).rolling(14).mean() / atr_s)
        m_di = 100 * (pd.Series(m_dm, index=btdr.index).rolling(14).mean() / atr_s)
        adx = (100 * np.abs(p_di - m_di) / (p_di + m_di)).rolling(14).mean().iloc[-1]
        adx = 20 if np.isnan(adx) else adx
        
        delta_p = btdr['Close'].diff()
        gain = delta_p.where(delta_p > 0, 0).rolling(14).mean()
        loss = -delta_p.where(delta_p < 0, 0).rolling(14).mean()
        rsi = 100 - (100 / (1 + (gain / loss))).iloc[-1]
        
        vol_base = pd.Series(ret_btdr).ewm(span=20).std().iloc[-1] if len(ret_btdr)>20 else 0.05
        
        factors = {"beta_btc": beta_btc, "beta_qqq": beta_qqq, "vwap": vwap_30d, "adx": adx, "regime": "Trend" if adx > 25 else "Chop", "rsi": rsi, "vol_base": vol_base, "atr_ratio": (atr/btdr['Close']).iloc[-1]}

        df_reg = pd.DataFrame()
        df_reg['Gap'] = (btdr['Open'] - btdr['Close'].shift(1)) / btdr['Close'].shift(1)
        df_reg['BTC_Ret'] = btc['Close'].pct_change()
        df_reg['Vol_State'] = ((btdr['High'] - btdr['Low']) / btdr['Open']).shift(1)
        df_reg['Target_High'] = (btdr['High'] - btdr['Close'].shift(1)) / btdr['Close'].shift(1)
        df_reg['Target_Low'] = (btdr['Low'] - btdr['Close'].shift(1)) / btdr['Close'].shift(1)
        df_reg = df_reg.dropna().tail(90)
        
        weights = np.exp(np.linspace(-0.05 * len(df_reg), 0, len(df_reg))); W = np.diag(weights)
        X = np.column_stack([np.ones(len(df_reg)), df_reg['Gap'].values, df_reg['BTC_Ret'].values, df_reg['Vol_State'].values])
        XtWX = X.T @ W @ X
        theta_h = np.linalg.lstsq(XtWX, X.T @ W @ df_reg['Target_High'].values, rcond=None)[0]
        theta_l = np.linalg.lstsq(XtWX, X.T @ W @ df_reg['Target_Low'].values, rcond=None)[0]

        final_model = {
            "high": {"intercept": theta_h[0], "beta_gap": theta_h[1], "beta_btc": theta_h[2], "beta_vol": theta_h[3]},
            "low": {"intercept": theta_l[0], "beta_gap": theta_l[1], "beta_btc": theta_l[2], "beta_vol": theta_l[3]},
            "ensemble_hist_h": df_reg['Target_High'].tail(10).mean(), "ensemble_hist_l": df_reg['Target_Low'].tail(10).mean(),
            "ensemble_mom_h": df_reg['Target_High'].tail(3).max(), "ensemble_mom_l": df_reg['Target_Low'].tail(3).min(),
            "top_peers": top_peers
        }
        return final_model, factors, "v10.9 Stable"
    except Exception as e:
        return default_model, default_factors, "Offline"

# --- 5. 实时数据 ---
def determine_market_state(now_ny):
    curr_min = now_ny.hour * 60 + now_ny.minute
    if now_ny.weekday() >= 5: return "Weekend", "dot-closed" # Simple weekend check
    if 240 <= curr_min < 570: return "Pre-Mkt", "dot-pre"
    if 570 <= curr_min < 960: return "Mkt Open", "dot-reg"
    if 960 <= curr_min < 1200: return "Post-Mkt", "dot-post"
    return "Overnight", "dot-night"

def get_realtime_data():
    tickers = "BTC-USD BTDR QQQ ^VIX " + " ".join(MINER_POOL)
    try:
        daily = yf.download(tickers, period="5d", interval="1d", group_by='ticker', threads=True, progress=False)
        live = yf.download(tickers, period="1d", interval="1m", prepost=True, group_by='ticker', threads=True, progress=False)
        
        quotes = {}
        now_ny = datetime.now(pytz.timezone('America/New_York'))
        state_tag, state_css = determine_market_state(now_ny)
        live_vol_btdr = 0.01
        
        for sym in tickers.split():
            try:
                df_d = daily[sym].dropna(subset=['Close']) if sym in daily else pd.DataFrame()
                df_m = live[sym].dropna(subset=['Close']) if sym in live else pd.DataFrame()
                
                curr_p = 0.0; vol = 0
                if not df_m.empty:
                    curr_p = df_m['Close'].iloc[-1]; vol = df_m['Volume'].sum() if 'Volume' in df_m else 0
                    if sym == 'BTDR': live_vol_btdr = df_m['Close'].tail(60).std() if len(df_m) > 10 else curr_p * 0.005
                elif not df_d.empty:
                    curr_p = df_d['Close'].iloc[-1]; vol = df_d['Volume'].iloc[-1]
                
                prev = 1.0; open_p = 0.0; is_today = False
                if not df_d.empty:
                    if df_d.index[-1].date() == now_ny.date():
                        is_today = True; open_p = df_d['Open'].iloc[-1]
                        prev = df_d['Close'].iloc[-2] if len(df_d) >= 2 else open_p
                    else:
                        prev = df_d['Close'].iloc[-1]; open_p = prev
                
                quotes[sym] = {"price": curr_p, "pct": ((curr_p-prev)/prev)*100 if prev>0 else 0, "prev": prev, "open": open_p, "volume": vol, "tag": state_tag, "css": state_css, "is_today": is_today}
            except: quotes[sym] = {"price": 0, "pct": 0, "prev": 1, "open": 0, "volume": 0, "tag": "ERR", "css": "dot-closed", "is_today": False}
            
        fng = 50
        try: fng = int(requests.get("https://api.alternative.me/fng/", timeout=1).json()['data'][0]['value'])
        except: pass
        return quotes, fng, max(live_vol_btdr, 0.01)
    except: return None, 50, 0.01

# --- 6. 仪表盘展示 ---
@st.fragment(run_every=10)
def show_live_dashboard():
    quotes, fng_val, live_vol_btdr = get_realtime_data()
    ai_model, factors, ai_status = run_grandmaster_analytics()
    if not quotes: st.warning("📡 连接中..."); time.sleep(1); st.rerun(); return

    btc = quotes.get('BTC-USD'); qqq = quotes.get('QQQ'); vix = quotes.get('^VIX'); btdr = quotes.get('BTDR')
    
    # Header
    now_str = datetime.now(pytz.timezone('America/New_York')).strftime('%H:%M:%S')
    st.markdown(f"<div class='time-bar'>美东 {now_str} &nbsp;|&nbsp; 状态: <span class='{'badge-trend' if factors['regime']=='Trend' else 'badge-chop'}'>{factors['regime']}</span> &nbsp;|&nbsp; 引擎: <b>{ai_status}</b></div>", unsafe_allow_html=True)
    
    # Top Cards
    c1, c2 = st.columns(2)
    with c1: st.markdown(card_html("BTC (USD)", f"${btc['price']:,.0f}", f"{btc['pct']:+.2f}%", btc['pct']), unsafe_allow_html=True)
    with c2: st.markdown(card_html("恐慌指数", f"{fng_val}", None, 0, tooltip_text="0-25: 极度恐慌 (买点)\n75-100: 极度贪婪 (风险)"), unsafe_allow_html=True)
    
    st.caption("⚒️ 矿股板块 Beta (Correlation Top 5)")
    cols = st.columns(5)
    for i, p in enumerate(ai_model.get("top_peers", [])):
        d = quotes.get(p, {'pct':0, 'price':0, 'volume':0})
        cols[i].markdown(miner_card_html(p, d['price'], d['pct'], (d['volume']/(MINER_SHARES.get(p,200)*1e6))*100), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # BTDR Main
    c3, c4, c5 = st.columns(3)
    with c3: st.markdown(card_html("BTDR 现价", f"${btdr['price']:.2f}", f"{btdr['pct']:+.2f}%", btdr['pct'], f"<span class='status-dot {btdr['css']}'></span>"), unsafe_allow_html=True)
    with c4: st.markdown(card_html("今日开盘", f"${btdr['open']:.2f}", None, 0), unsafe_allow_html=True)
    vwap_dist = ((btdr['price']-factors['vwap'])/factors['vwap'])*100
    with c5: st.markdown(card_html("机构成本 (VWAP)", f"${factors['vwap']:.2f}", f"{vwap_dist:+.1f}%", vwap_dist), unsafe_allow_html=True)

    # --- 关键修改区：稳定性计算逻辑 ---
    gap_pct = ((btdr['open'] - btdr['prev']) / btdr['prev'])
    btc_factor = btc['pct'] / 100
    
    # 1. Kalman 预测 (基础)
    mh = ai_model['high']; ml = ai_model['low']
    pred_h = mh['intercept'] + mh['beta_gap']*gap_pct + mh['beta_btc']*btc_factor + mh['beta_vol']*factors['atr_ratio']
    pred_l = ml['intercept'] + ml['beta_gap']*gap_pct + ml['beta_btc']*btc_factor + ml['beta_vol']*factors['atr_ratio']
    
    # 2. 波动边界 (锚点锁定为 Open/Prev，不再随 Price 跳动)
    anchor = btdr['open'] if btdr['open'] > 0 else btdr['prev']
    vol_pct = live_vol_btdr / max(btdr['price'], 0.1)
    # 放大倍数 3.0，形成强阻力
    ai_h = (anchor * (1 + 3.0 * vol_pct) - btdr['prev']) / btdr['prev']
    ai_l = (anchor * (1 - 3.0 * vol_pct) - btdr['prev']) / btdr['prev']
    
    # 3. 权重重分配 (降低实时 AI 权重)
    w_k = 0.4; w_h = 0.25; w_m = 0.15; w_ai = 0.2
    final_h = w_k*pred_h + w_h*ai_model['ensemble_hist_h'] + w_m*ai_model['ensemble_mom_h'] + w_ai*ai_h
    final_l = w_k*pred_l + w_h*ai_model['ensemble_hist_l'] + w_m*ai_model['ensemble_mom_l'] + w_ai*ai_l
    
    # 4. 情绪微调
    adj = (fng_val - 50) * 0.0005
    raw_h = btdr['prev'] * (1 + final_h + adj)
    raw_l = btdr['prev'] * (1 + final_l + adj)
    
    # 5. 平滑滤波 (Session State)
    if 'smooth_h' not in st.session_state: st.session_state.smooth_h = raw_h
    if 'smooth_l' not in st.session_state: st.session_state.smooth_l = raw_l
    sf = 0.95 # 95% 保留旧值，极度平滑
    st.session_state.smooth_h = st.session_state.smooth_h * sf + raw_h * (1-sf)
    st.session_state.smooth_l = st.session_state.smooth_l * sf + raw_l * (1-sf)
    
    p_high = st.session_state.smooth_h
    p_low = st.session_state.smooth_l
    # --- 修改结束 ---

    # Ticket Logic
    atr_buf = live_vol_btdr * 0.5
    b_entry = p_low + atr_buf; b_stop = b_entry - live_vol_btdr*2; b_tgt = p_high - atr_buf
    s_entry = p_high - atr_buf; s_stop = s_entry + live_vol_btdr*2; s_tgt = p_low + atr_buf
    
    b_rr = (b_tgt - b_entry) / max(b_entry - b_stop, 0.01)
    s_rr = (s_entry - s_tgt) / max(s_stop - s_entry, 0.01)
    
    b_prob = min(max((1 - norm.cdf((btdr['price'] - b_entry)/(live_vol_btdr*10)))*200, 5), 95)
    s_prob = min(max((1 - norm.cdf((s_entry - btdr['price'])/(live_vol_btdr*10)))*200, 5), 95)

    t1, t2 = st.columns(2)
    with t1: st.markdown(f"""<div class="ticket-card ticket-buy"><div class="ticket-header" style="color:#0ca678;">🟢 BUY LIMIT <span class="tag-smart">SMART</span></div><div class="ticket-price-row"><span class="ticket-price-label">挂单价</span><span class="ticket-price-val">${b_entry:.2f}</span></div><div class="ticket-price-row"><span class="ticket-price-label">止损价</span><span class="ticket-price-val" style="color:#e03131;">${b_stop:.2f}</span></div><div class="ticket-meta"><span>R/R: <b>1:{b_rr:.1f}</b></span><span>成交概率: <b>{b_prob:.0f}%</b></span></div><div class="prob-container"><div class="prob-fill {'prob-high' if b_prob>60 else 'prob-med'}" style="width:{b_prob}%"></div></div></div>""", unsafe_allow_html=True)
    with t2: st.markdown(f"""<div class="ticket-card ticket-sell"><div class="ticket-header" style="color:#e03131;">🔴 SELL LIMIT <span class="tag-smart">SMART</span></div><div class="ticket-price-row"><span class="ticket-price-label">挂单价</span><span class="ticket-price-val">${s_entry:.2f}</span></div><div class="ticket-price-row"><span class="ticket-price-label">止损价</span><span class="ticket-price-val" style="color:#e03131;">${s_stop:.2f}</span></div><div class="ticket-meta"><span>R/R: <b>1:{s_rr:.1f}</b></span><span>成交概率: <b>{s_prob:.0f}%</b></span></div><div class="prob-container"><div class="prob-fill {'prob-high' if s_prob>60 else 'prob-med'}" style="width:{s_prob}%"></div></div></div>""", unsafe_allow_html=True)

    st.markdown("""<div style="font-size:0.7rem; color:#888; display:flex; justify-content:space-between;"><span>🟦 Kalman (40%)</span><span>🟨 History (25%)</span><span>🟥 Momentum (15%)</span><span>🟪 AI Volatility (20%)</span></div><div class="ensemble-bar"><div class="bar-kalman"></div><div class="bar-hist"></div><div class="bar-mom"></div><div class="bar-ai"></div></div>""", unsafe_allow_html=True)
    
    ch, cl = st.columns(2)
    h_bg = "#e6fcf5" if btdr['price'] < p_high else "#0ca678"; h_txt = "#087f5b" if btdr['price'] < p_high else "#fff"
    l_bg = "#fff5f5" if btdr['price'] > p_low else "#e03131"; l_txt = "#c92a2a" if btdr['price'] > p_low else "#fff"
    
    with ch: st.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background-color: {h_bg}; color: {h_txt}; border: 1px solid #c3fae8;"><div style="font-size: 0.8rem; opacity: 0.8;">理论阻力 (High)</div><div style="font-size: 1.5rem; font-weight: bold;">${p_high:.2f}</div></div></div>""", unsafe_allow_html=True)
    with cl: st.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background-color: {l_bg}; color: {l_txt}; border: 1px solid #ffc9c9;"><div style="font-size: 0.8rem; opacity: 0.8;">理论支撑 (Low)</div><div style="font-size: 1.5rem; font-weight: bold;">${p_low:.2f}</div></div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.markdown(factor_html("QQQ Trend", f"{qqq['pct']:+.2f}%", "Mkt", qqq['pct'], "纳指大盘"), unsafe_allow_html=True)
    with m2: st.markdown(factor_html("VIX", f"{vix['price']:.1f}", "Risk", 0, "恐慌指数", True), unsafe_allow_html=True)
    with m3: st.markdown(factor_html("Beta (BTC)", f"{factors['beta_btc']:.2f}", "K-Filter", 0, "动态相关性"), unsafe_allow_html=True)
    with m4: st.markdown(factor_html("ADX", f"{factors['adx']:.1f}", factors['regime'], 1 if factors['adx']>25 else -1, "趋势强度"), unsafe_allow_html=True)

    # Simulation Chart
    drift = (btc['pct']/100 * 0.5)
    sims, days = 1000, 5
    paths = np.zeros((sims, days+1)); paths[:,0] = btdr['price']
    vol = factors['vol_base']
    for t in range(1, days+1):
        paths[:,t] = paths[:,t-1] * np.exp((drift - 0.5*vol**2) + vol * np.random.standard_t(5, sims))
    
    res = np.percentile(paths, [10, 50, 90], axis=0)
    chart_df = pd.DataFrame({"Day": np.arange(days+1), "P90": res[2], "P50": res[1], "P10": res[0]})
    
    base = alt.Chart(chart_df).encode(x='Day:O')
    area = base.mark_area(opacity=0.2).encode(y=alt.Y('P10', scale=alt.Scale(zero=False)), y2='P90')
    line = base.mark_line(color='#228be6').encode(y='P50')
    st.altair_chart((area + line).properties(height=250), use_container_width=True)

st.markdown("### ⚡ BTDR Pilot v10.9 Final")
show_live_dashboard()
