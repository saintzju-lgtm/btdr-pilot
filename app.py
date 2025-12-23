import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
import altair as alt
from datetime import datetime
import pytz

# --- 1. 页面配置 (保留 v9.5 UI) ---
st.set_page_config(page_title="BTDR Pilot v9.6 Classic", layout="centered")

# CSS: 100% 还原 v9.5 的样式
st.markdown("""
    <style>
    html { overflow-y: scroll; }
    .stApp > header { display: none; }
    .stApp { margin-top: -30px; background-color: #ffffff; }
    div[data-testid="stStatusWidget"] { visibility: hidden; }
    
    h1, h2, h3, div, p, span { 
        color: #212529 !important; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important; 
    }
    
    /* 锁定图表高度 */
    div[data-testid="stAltairChart"] {
        height: 320px !important;
        min-height: 320px !important;
        overflow: hidden !important;
        border: 1px solid #f8f9fa;
    }
    
    /* 统一卡片样式 */
    .metric-card {
        background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02); height: 95px; padding: 0 16px;
        display: flex; flex-direction: column; justify-content: center;
    }
    .metric-label { font-size: 0.75rem; color: #888; margin-bottom: 2px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #212529; line-height: 1.2; }
    .metric-delta { font-size: 0.9rem; font-weight: 600; margin-top: 2px; }
    
    /* 因子卡片 & Tooltips */
    .factor-box {
        background: #fff; border: 1px solid #eee; border-radius: 8px; padding: 6px; text-align: center;
        height: 75px; display: flex; flex-direction: column; justify-content: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        position: relative; cursor: help; transition: transform 0.1s;
    }
    
    .factor-box:hover { border-color: #ced4da; transform: translateY(-1px); }
    .factor-title { font-size: 0.65rem; color: #999; text-transform: uppercase; letter-spacing: 0.5px; }
    .factor-val { font-size: 1.1rem; font-weight: bold; color: #495057; margin: 2px 0; }
    .factor-sub { font-size: 0.7rem; font-weight: 600; }
    
    .tooltip-text {
        visibility: hidden; width: 180px; background-color: rgba(33, 37, 41, 0.95);
        color: #fff !important; text-align: center; border-radius: 6px; padding: 8px;
        position: absolute; z-index: 999; bottom: 110%; left: 50%; margin-left: -90px;
        opacity: 0; transition: opacity 0.3s; font-size: 0.7rem !important;
        font-weight: normal; line-height: 1.4; pointer-events: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .tooltip-text::after {
        content: ""; position: absolute; top: 100%; left: 50%; margin-left: -5px;
        border-width: 5px; border-style: solid;
        border-color: rgba(33, 37, 41, 0.95) transparent transparent transparent;
    }

    .factor-box:hover .tooltip-text { visibility: visible; opacity: 1; }
    
    /* 颜色定义 */
    .color-up { color: #0ca678; } .color-down { color: #d6336c; } .color-neutral { color: #adb5bd; }
    
    /* 状态小圆点 (Pre/Post/Mkt/Night) */
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
    """, unsafe_allow_html=True)

# --- 2. 辅助函数 (保持 v9.5) ---
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

# --- 3. 核心计算逻辑 (V9.6 内核 + 修复版) ---
@st.cache_data(ttl=600) 
def run_quant_analytics():
    # 使用 v9.6 的逻辑，但返回结构适配 v9.5 的展示需求
    default_model = {
        "up_scenario": {"h_add": 0.05, "l_add": -0.02},
        "down_scenario": {"h_add": 0.02, "l_add": -0.05},
        "beta_btc": 0.6
    }
    default_factors = {
        "vwap": 0, "adx": 20, "regime": "Neutral", 
        "beta_btc": 1.5, "beta_qqq": 1.2, "rsi": 50, 
        "vol_base": 0.05, "vol_parkinson": 0.05, "jump_prob": 0.1 # 补全键值防止KeyError
    }
    
    try:
        # 使用 v9.6 的稳健下载策略 (threads=False)
        data = yf.download("BTDR BTC-USD QQQ ^VIX", period="6mo", interval="1d", group_by='ticker', threads=False, progress=False)
        
        if data.empty: return default_model, default_factors, "No Data"

        try:
            btdr = data['BTDR'].dropna(); btc = data['BTC-USD'].dropna(); qqq = data['QQQ'].dropna()
        except KeyError:
            return default_model, default_factors, "Ticker Err"

        idx = btdr.index.intersection(btc.index).intersection(qqq.index)
        btdr = btdr.loc[idx]; btc = btc.loc[idx]; qqq = qqq.loc[idx]
        
        # --- 计算因子 (V9.6 逻辑) ---
        ret_btdr = btdr['Close'].pct_change()
        ret_btc = btc['Close'].pct_change()
        ret_qqq = qqq['Close'].pct_change()
        
        cov_btc = ret_btdr.rolling(60).cov(ret_btc).iloc[-1]
        var_btc = ret_btc.rolling(60).var().iloc[-1]
        beta_btc = cov_btc / var_btc if var_btc > 1e-6 else 1.5
        
        cov_qqq = ret_btdr.rolling(60).cov(ret_qqq).iloc[-1]
        var_qqq = ret_qqq.rolling(60).var().iloc[-1]
        beta_qqq = cov_qqq / var_qqq if var_qqq > 1e-6 else 1.2
        
        # VWAP
        vwap_30d = (btdr['Close'] * btdr['Volume']).tail(30).sum() / btdr['Volume'].tail(30).sum()
        
        # ADX
        high, low, close = btdr['High'], btdr['Low'], btdr['Close']
        tr = np.maximum(high - low, np.abs(high - close.shift(1)))
        # Parkinson Volatility (V9.6 特性)
        safe_high = high.replace(0, np.nan); safe_low = low.replace(0, np.nan)
        vol_parkinson = np.sqrt(1 / (4 * np.log(2)) * ((np.log(safe_high / safe_low)) ** 2)).rolling(20).mean().iloc[-1]
        if np.isnan(vol_parkinson): vol_parkinson = 0.05
        
        adx_val = 20
        if len(tr) > 14:
            adx_raw = tr.rolling(14).mean().iloc[-1]
            if not np.isnan(adx_raw): adx_val = min(max(adx_raw, 0), 100)
        regime = "Trend" if adx_val > 25 else "Chop"
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rsi = 50
        if loss.iloc[-1] != 0: rsi = 100 - (100 / (1 + gain.iloc[-1]/loss.iloc[-1]))
        
        # Jump Probability (V9.6 特性)
        daily_ret = btdr['Close'].pct_change().dropna()
        std = daily_ret.std()
        jump_prob = 0.1
        if std > 0:
            jumps = daily_ret[np.abs(daily_ret) > 2.5 * std]
            jump_prob = max(0.05, len(jumps) / len(daily_ret))

        # --- 非对称回归模型 (V9.6 特性) ---
        df_reg = btdr.tail(60).copy()
        df_reg['PrevClose'] = df_reg['Close'].shift(1)
        df_reg = df_reg.dropna()
        df_reg['Gap'] = (df_reg['Open'] - df_reg['PrevClose']) / df_reg['PrevClose']
        df_reg['High_Ret'] = (df_reg['High'] - df_reg['PrevClose']) / df_reg['PrevClose']
        df_reg['Low_Ret'] = (df_reg['Low'] - df_reg['PrevClose']) / df_reg['PrevClose']
        
        gap_up = df_reg[df_reg['Gap'] > 0]
        gap_down = df_reg[df_reg['Gap'] <= 0]
        
        h_up = (gap_up['High_Ret'] - gap_up['Gap']).mean() if len(gap_up) > 5 else 0.05
        l_up = (gap_up['Low_Ret'] - gap_up['Gap']).mean() if len(gap_up) > 5 else -0.02
        h_down = (gap_down['High_Ret'] - gap_down['Gap']).mean() if len(gap_down) > 5 else 0.02
        l_down = (gap_down['Low_Ret'] - gap_down['Gap']).mean() if len(gap_down) > 5 else -0.05

        final_model = {
            "up_scenario": {"h_add": max(h_up, 0.01), "l_add": min(l_up, -0.005)},
            "down_scenario": {"h_add": max(h_down, 0.005), "l_add": min(l_down, -0.01)},
            "beta_btc": 0.6, "beta_sector": 0.25
        }
        
        factors = {
            "beta_btc": beta_btc, "beta_qqq": beta_qqq, "vwap": vwap_30d, 
            "adx": adx_val, "regime": regime, "rsi": rsi, 
            "vol_base": vol_parkinson, # v9.5 UI 用的是 vol_base 标签，这里传入更准的 Parkinson
            "vol_parkinson": vol_parkinson, "jump_prob": jump_prob
        }
        
        return final_model, factors, "Quant v9.6"
    except Exception as e:
        print(f"Model Error: {e}")
        return default_model, default_factors, "Offline"

# --- 4. 实时数据 (V9.6 修复版) ---
def get_realtime_data():
    tickers_list = "BTC-USD BTDR MARA RIOT CORZ CLSK IREN QQQ ^VIX"
    
    # 状态判断逻辑
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny)
    current_minutes = now_ny.hour * 60 + now_ny.minute
    state, css = "Overnight", "dot-night"
    
    if now_ny.weekday() == 5 or (now_ny.weekday() == 6 and now_ny.hour < 20):
        state, css = "Weekend", "dot-closed"
    elif 240 <= current_minutes < 570: state, css = "Pre-Mkt", "dot-pre"
    elif 570 <= current_minutes < 960: state, css = "Mkt Open", "dot-reg"
    elif 960 <= current_minutes < 1200: state, css = "Post-Mkt", "dot-post"

    try:
        # V9.6 优化：只取2天分钟线，减少卡顿
        daily = yf.download(tickers_list, period="5d", interval="1d", group_by='ticker', threads=False, progress=False)
        live = yf.download(tickers_list, period="2d", interval="1m", prepost=True, group_by='ticker', threads=False, progress=False)
        
        quotes = {}
        for sym in tickers_list.split():
            try:
                # 安全获取并处理 NaN
                df_day = daily[sym] if sym in daily else pd.DataFrame()
                df_min = live[sym] if sym in live else pd.DataFrame()
                
                price = 0.0
                if not df_min.empty:
                    val = df_min['Close'].iloc[-1]
                    if not pd.isna(val): price = float(val)
                if price == 0 and not df_day.empty:
                    val = df_day['Close'].iloc[-1]
                    if not pd.isna(val): price = float(val)
                
                prev = 1.0; open_price = 0.0; is_open_today = False
                if not df_day.empty:
                    clean = df_day.dropna(subset=['Close'])
                    if not clean.empty:
                        if clean.index[-1].date() == now_ny.date():
                            is_open_today = True
                            open_price = float(clean['Open'].iloc[-1])
                            prev = float(clean['Close'].iloc[-2]) if len(clean) > 1 else open_price
                        else:
                            prev = float(clean['Close'].iloc[-1])
                            open_price = prev

                pct = ((price - prev) / prev) * 100 if prev > 0 else 0
                
                quotes[sym] = {
                    "price": price, "pct": pct, "prev": prev, "open": open_price, 
                    "tag": state, "css": css, "is_open_today": is_open_today
                }
            except: 
                quotes[sym] = {"price": 0, "pct": 0, "prev": 1, "open": 0, "tag": "ERR", "css": "dot-closed", "is_open_today": False}
        
        return quotes, 50 # Mock FNG
    except:
        return None, 50

# --- 5. Fragment 局部刷新 (V9.5 UI 布局) ---
@st.fragment(run_every=5) 
def show_live_dashboard():
    quotes, fng_val = get_realtime_data()
    ai_model, factors, ai_status = run_quant_analytics()
    
    if not quotes:
        st.warning("📡 连接中 (Retrying)...")
        time.sleep(1)
        return

    btc_chg = quotes['BTC-USD']['pct']
    qqq_chg = quotes.get('QQQ', {'pct': 0})['pct']
    vix_val = quotes.get('^VIX', {'price': 20})['price']
    vix_chg = quotes.get('^VIX', {'pct': 0})['pct']
    btdr = quotes['BTDR']
    
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny).strftime('%H:%M:%S')
    
    regime_tag = factors['regime']
    badge_class = "badge-trend" if regime_tag == "Trend" else "badge-chop"
    
    # [UI] 顶部状态栏 (保持 V9.5 样式)
    st.markdown(f"<div class='time-bar'>美东 {now_ny} &nbsp;|&nbsp; 状态: <span class='{badge_class}'>{regime_tag}</span> &nbsp;|&nbsp; 引擎: {ai_status}</div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1: st.markdown(card_html("BTC (全时段)", f"{btc_chg:+.2f}%", f"{btc_chg:+.2f}%", btc_chg), unsafe_allow_html=True)
    with c2: st.markdown(card_html("恐慌指数", f"{fng_val}", None, 0), unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    st.caption("⚒️ 矿股板块 Beta")
    cols = st.columns(5)
    peers = ["MARA", "RIOT", "CORZ", "CLSK", "IREN"]
    for i, p in enumerate(peers):
        if p in quotes:
            val = quotes[p]['pct']
            cols[i].markdown(card_html(p, f"{val:+.1f}%", f"{val:+.1f}%", val), unsafe_allow_html=True)
            
    st.markdown("---")
    
    c3, c4, c5 = st.columns(3)
    status_tag = f"<span class='status-dot {btdr['css']}'></span> <span style='font-size:0.6rem; color:#999'>{btdr['tag']}</span>"
    
    with c3: st.markdown(card_html("BTDR 现价", f"${btdr['price']:.2f}", f"{btdr['pct']:+.2f}%", btdr['pct'], status_tag), unsafe_allow_html=True)
    
    open_label = "今日开盘" if btdr['is_open_today'] else "预计开盘/昨收"
    open_extra = "" if btdr['is_open_today'] else "(Pending)"
    with c4: st.markdown(card_html(open_label, f"${btdr['open']:.2f}", None, 0, open_extra), unsafe_allow_html=True)

    dist_vwap = ((btdr['price'] - factors['vwap']) / factors['vwap']) * 100 if factors['vwap'] > 0 else 0
    with c5: st.markdown(card_html("机构成本 (VWAP)", f"${factors['vwap']:.2f}", f"{dist_vwap:+.1f}%", dist_vwap), unsafe_allow_html=True)

    # --- 预测 (使用 V9.6 非对称逻辑，但渲染进 V9.5 的框框) ---
    btdr_gap_pct = ((btdr['open'] - btdr['prev']) / btdr['prev']) if btdr['prev'] else 0
    
    # 动态选择参数
    if btdr_gap_pct >= 0:
        params = ai_model.get('up_scenario', {'h_add': 0.05, 'l_add': -0.02})
        scenario_txt = "Gap Up"
    else:
        params = ai_model.get('down_scenario', {'h_add': 0.02, 'l_add': -0.05})
        scenario_txt = "Gap Down"
        
    btc_impact = (btc_chg/100) * 0.3
    pred_high_price = btdr['open'] * (1 + params['h_add'] + btc_impact)
    pred_low_price = btdr['open'] * (1 + params['l_add'] + btc_impact)
    
    st.markdown(f"### 🎯 日内阻力/支撑 ({scenario_txt})")
    col_h, col_l = st.columns(2)
    h_bg = "#e6fcf5" if btdr['price'] < pred_high_price else "#0ca678"; h_txt = "#087f5b" if btdr['price'] < pred_high_price else "#ffffff"
    l_bg = "#fff5f5" if btdr['price'] > pred_low_price else "#e03131"; l_txt = "#c92a2a" if btdr['price'] > pred_low_price else "#ffffff"
    
    with col_h: st.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background-color: {h_bg}; color: {h_txt}; border: 1px solid #c3fae8;"><div style="font-size: 0.8rem; opacity: 0.8;">日内阻力 (High)</div><div style="font-size: 1.5rem; font-weight: bold;">${pred_high_price:.2f}</div></div></div>""", unsafe_allow_html=True)
    with col_l: st.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background-color: {l_bg}; color: {l_txt}; border: 1px solid #ffc9c9;"><div style="font-size: 0.8rem; opacity: 0.8;">日内支撑 (Low)</div><div style="font-size: 1.5rem; font-weight: bold;">${pred_low_price:.2f}</div></div></div>""", unsafe_allow_html=True)

    # --- 因子面板 (V9.5 UI) ---
    st.markdown("---")
    st.markdown("### 🌍 宏观环境 (Macro)")
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.markdown(factor_html("QQQ (纳指)", f"{qqq_chg:+.2f}%", "Market", qqq_chg, "科技股大盘风向标"), unsafe_allow_html=True)
    with m2: st.markdown(factor_html("VIX (恐慌)", f"{vix_val:.1f}", f"{vix_chg:+.1f}%", -vix_chg, "市场恐慌指数", reverse_color=True), unsafe_allow_html=True)
    with m3: st.markdown(factor_html("Beta (BTC)", f"{factors['beta_btc']:.2f}", "Corr", 0, "相对BTC波动系数"), unsafe_allow_html=True)
    with m4: st.markdown(factor_html("Beta (QQQ)", f"{factors['beta_qqq']:.2f}", "Corr", 0, "相对QQQ波动系数"), unsafe_allow_html=True)
    
    st.markdown("### 🔬 微观结构 (Micro)")
    mi1, mi2, mi3, mi4 = st.columns(4)
    
    # 估算漂移 (用于展示)
    drift_est = (btc_chg/100 * factors['beta_btc'] * 0.4) + (qqq_chg/100 * factors['beta_qqq'] * 0.4)
    if abs(dist_vwap) > 10: drift_est -= (dist_vwap/100) * 0.05
    
    with mi1: st.markdown(factor_html("ADX (强度)", f"{factors['adx']:.1f}", factors['regime'], 1 if factors['adx']>25 else -1, "趋势强度"), unsafe_allow_html=True)
    with mi2: st.markdown(factor_html("RSI (14d)", f"{factors['rsi']:.0f}", "O/B" if factors['rsi']>70 else ("O/S" if factors['rsi']<30 else "Neu"), 0, "相对强弱"), unsafe_allow_html=True)
    with mi3: st.markdown(factor_html("Vol (Parkinson)", f"{factors['vol_parkinson']*100:.1f}%", "Risk", 0, "日内极值波动率"), unsafe_allow_html=True)
    with mi4: st.markdown(factor_html("Jump Prob", f"{factors['jump_prob']*100:.0f}%", "Tail", -1, "黑天鹅跳跃概率"), unsafe_allow_html=True)

    # --- 宗师级推演 (V9.6 MJD 算法 + V9.5 图表配色) ---
    st.markdown("### ☁️ 宗师级推演 (MJD Engine)")
    
    # MJD 算法准备
    S0 = btdr['price']
    T = 5; dt = 1
    mu = (btc_chg/100 * factors['beta_btc']) * 0.5
    sigma = factors['vol_parkinson']
    lam = factors['jump_prob']
    jump_mu = -0.02 if vix_val > 25 else 0.0
    jump_sigma = 0.05
    
    simulations = 1000
    # Vectorized MJD Simulation
    Z1 = np.random.normal(0, 1, (simulations, T))
    drift_part = (mu - 0.5 * sigma**2) * dt
    diffusion_part = sigma * np.sqrt(dt) * Z1
    N = np.random.poisson(lam * dt, (simulations, T))
    Jump_part = N * np.random.normal(jump_mu, jump_sigma, (simulations, T))
    
    daily_log_returns = drift_part + diffusion_part + Jump_part
    price_paths = np.zeros((simulations, T + 1))
    price_paths[:, 0] = S0
    price_paths[:, 1:] = S0 * np.exp(np.cumsum(daily_log_returns, axis=1))
    
    p90 = np.percentile(price_paths, 90, axis=0)
    p50 = np.percentile(price_paths, 50, axis=0)
    p10 = np.percentile(price_paths, 10, axis=0)
    
    df_chart = pd.DataFrame({"Day": range(T+1), "P90": np.round(p90, 2), "P50": np.round(p50, 2), "P10": np.round(p10, 2)})
    
    # [UI] 图表样式 100% 还原 V9.5 (蓝色区域，绿/红线)
    base = alt.Chart(df_chart).encode(x=alt.X('Day:O', title='未来交易日'))
    # V9.5: Blue Area
    area = base.mark_area(opacity=0.2, color='#4dabf7').encode(y=alt.Y('P10', title='价格预演 (USD)', scale=alt.Scale(zero=False)), y2='P90')
    # V9.5: Green Line 90, Red Line 10, Blue Line 50
    l90 = base.mark_line(color='#0ca678', strokeDash=[5,5]).encode(y='P90')
    l50 = base.mark_line(color='#228be6', size=3).encode(y='P50')
    l10 = base.mark_line(color='#d6336c', strokeDash=[5,5]).encode(y='P10')
    
    nearest = alt.selection_point(nearest=True, on='mouseover', fields=['Day'], empty=False)
    selectors = base.mark_rule(opacity=0).encode(x='Day:O').add_params(nearest)
    points = base.mark_circle(size=60, color="black").encode(
        y='P50', opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip('Day'), alt.Tooltip('P90'), alt.Tooltip('P50'), alt.Tooltip('P10')]
    )
    
    st.altair_chart((area + l90 + l50 + l10 + selectors + points).properties(height=300).interactive(), use_container_width=True)
    st.caption(f"Engine: v9.6 (MJD) | Vol: {sigma*100:.1f}% | JumpProb: {lam*100:.1f}%")

# --- 7. 主程序入口 ---
st.markdown("### ⚡ BTDR Pilot v9.6 Classic")
show_live_dashboard()
