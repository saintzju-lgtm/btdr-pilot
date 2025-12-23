import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
import altair as alt
from datetime import datetime, timedelta
import pytz
from streamlit_autorefresh import st_autorefresh

# --- 1. 页面配置 ---
st.set_page_config(page_title="BTDR Pilot v8.6", layout="centered")

# 5秒刷新
st_autorefresh(interval=5000, limit=None, key="realtime_counter")

# CSS: 磐石级防抖 (Anti-Jitter Rock Solid)
st.markdown("""
    <style>
    /* 1. 全局滚动条，防止因内容长短变化导致的左右抖动 */
    html { overflow-y: scroll; }
    
    .stApp > header { display: none; }
    .stApp { margin-top: -30px; background-color: #ffffff; }
    div[data-testid="stStatusWidget"] { visibility: hidden; }
    
    h1, h2, h3, div, p, span { 
        color: #212529 !important; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important; 
    }
    
    /* 2. 【核心修复】强力锁定图表外层容器 */
    /* 只要是包含 chart 的容器，强制最小高度，防止渲染瞬间塌陷 */
    div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stAltairChart"]) {
        min-height: 320px !important;
        height: 320px !important;
        overflow: hidden !important;
        display: block !important;
    }
    
    div[data-testid="stAltairChart"] {
        height: 320px !important;
        width: 100% !important;
    }
    
    /* 3. 隐藏加载时的闪烁动画 */
    canvas { animation: none !important; transition: none !important; }
    
    /* 指标卡片 */
    .metric-card {
        background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02); height: 95px; padding: 0 16px;
        display: flex; flex-direction: column; justify-content: center;
    }
    .metric-label { font-size: 0.75rem; color: #888; margin-bottom: 2px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #212529; line-height: 1.2; }
    .metric-delta { font-size: 0.9rem; font-weight: 600; margin-top: 2px; }
    
    /* 因子小卡片 */
    .factor-box {
        background: #fff; border: 1px solid #eee; border-radius: 8px; padding: 6px; text-align: center;
        height: 75px; display: flex; flex-direction: column; justify-content: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .factor-title { font-size: 0.65rem; color: #999; text-transform: uppercase; letter-spacing: 0.5px; }
    .factor-val { font-size: 1.1rem; font-weight: bold; color: #495057; margin: 2px 0; }
    .factor-sub { font-size: 0.7rem; font-weight: 600; }
    
    .color-up { color: #0ca678; } .color-down { color: #d6336c; } .color-neutral { color: #adb5bd; }
    .status-dot { height: 6px; width: 6px; border-radius: 50%; display: inline-block; margin-left: 6px; }
    .dot-reg { background-color: #0ca678; } .dot-closed { background-color: #adb5bd; }
    
    .pred-container-wrapper { height: 110px; width: 100%; display: block; margin-top: 5px; }
    .pred-box { padding: 0 10px; border-radius: 12px; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center; }
    
    .time-bar { font-size: 0.75rem; color: #999; text-align: center; margin-bottom: 20px; padding: 6px; background: #fafafa; border-radius: 6px; }
    
    .badge-trend { background:#fd7e14; color:white; padding:1px 4px; border-radius:3px; font-size:0.6rem; }
    .badge-chop { background:#868e96; color:white; padding:1px 4px; border-radius:3px; font-size:0.6rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 状态管理 ---
if 'data_cache' not in st.session_state: st.session_state['data_cache'] = None
if st.session_state['data_cache'] and 'grandmaster' not in st.session_state['data_cache']:
    st.session_state['data_cache'] = None
    st.rerun()

# --- 3. 辅助函数 ---
def card_html(label, value_str, delta_str=None, delta_val=0, extra_tag=""):
    delta_html = ""
    if delta_str:
        color_class = "color-up" if delta_val >= 0 else "color-down"
        delta_html = f"<div class='metric-delta {color_class}'>{delta_str}</div>"
    return f"""<div class="metric-card"><div class="metric-label">{label} {extra_tag}</div><div class="metric-value">{value_str}</div>{delta_html}</div>"""

def factor_html(title, val, delta_str, delta_val, reverse_color=False):
    color_class = "color-up" if delta_val >= 0 else "color-down"
    if reverse_color: color_class = "color-down" if delta_val >= 0 else "color-up"
    return f"""<div class="factor-box"><div class="factor-title">{title}</div><div class="factor-val">{val}</div><div class="factor-sub {color_class}">{delta_str}</div></div>"""

st.markdown("### ⚡ BTDR 领航员 v8.6")

# --- 4. UI 骨架 (Layout) ---
ph_time = st.empty()

c1, c2 = st.columns(2)
with c1: ph_btc = st.empty()
with c2: ph_fng = st.empty()

st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
st.caption("⚒️ 矿股板块 Beta")
cols = st.columns(5)
ph_peers = [col.empty() for col in cols]

st.markdown("---")
c3, c4 = st.columns(2)
with c3: ph_btdr_price = st.empty()
with c4: ph_btdr_open = st.empty()

# 日内预测
st.markdown("### 🎯 日内阻力/支撑 (Intraday)")
col_h, col_l = st.columns(2)
with col_h: ph_pred_high = st.empty()
with col_l: ph_pred_low = st.empty()

st.markdown("---")

st.markdown("### 🌍 宏观环境 (Macro)")
macro_cols = st.columns(4)
ph_macros = [col.empty() for col in macro_cols]

st.markdown("### 🔬 微观结构 (Micro)")
micro_cols = st.columns(4)
ph_micros = [col.empty() for col in micro_cols]

st.markdown("### ☁️ 宗师级推演 (P90-P50-P10)")
# 图表占位符 (通过CSS强制锁定高度)
ph_chart = st.empty()

ph_footer = st.empty()

# --- 5. 核心：宗师级量化引擎 ---
@st.cache_data(ttl=300) 
def run_grandmaster_analytics():
    default_model = {"high": {"intercept": 4.29, "beta_open": 0.67, "beta_btc": 0.52}, "low": {"intercept": -3.22, "beta_open": 0.88, "beta_btc": 0.42}, "beta_sector": 0.25}
    default_factors = {"vwap": 0, "adx": 0, "regime": "Neutral", "beta_btc": 1.5, "beta_qqq": 1.2, "rsi": 50, "vol_base": 0.05}
    
    try:
        data = yf.download("BTDR BTC-USD QQQ ^VIX", period="6mo", interval="1d", group_by='ticker', threads=True, progress=False)
        btdr = data['BTDR'].dropna(); btc = data['BTC-USD'].dropna(); qqq = data['QQQ'].dropna()
        idx = btdr.index.intersection(btc.index).intersection(qqq.index)
        btdr = btdr.loc[idx]; btc = btc.loc[idx]; qqq = qqq.loc[idx]
        
        ret_btdr = btdr['Close'].pct_change()
        ret_btc = btc['Close'].pct_change()
        ret_qqq = qqq['Close'].pct_change()
        
        cov_btc = ret_btdr.rolling(60).cov(ret_btc).iloc[-1]
        var_btc = ret_btc.rolling(60).var().iloc[-1]
        beta_btc = cov_btc / var_btc if var_btc != 0 else 1.5
        
        cov_qqq = ret_btdr.rolling(60).cov(ret_qqq).iloc[-1]
        var_qqq = ret_qqq.rolling(60).var().iloc[-1]
        beta_qqq = cov_qqq / var_qqq if var_qqq != 0 else 1.2
        
        btdr['TP'] = (btdr['High'] + btdr['Low'] + btdr['Close']) / 3
        btdr['PV'] = btdr['TP'] * btdr['Volume']
        vwap_30d = btdr['PV'].tail(30).sum() / btdr['Volume'].tail(30).sum()
        
        high = btdr['High']; low = btdr['Low']; close = btdr['Close']
        tr = np.maximum(high - low, np.abs(high - close.shift(1)))
        atr = tr.rolling(14).mean()
        up_move = high - high.shift(1); down_move = low.shift(1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / atr.replace(0, np.nan))
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean().iloc[-1]
        if np.isnan(adx): adx = 20
        regime = "Trend" if adx > 25 else "Chop"
        
        delta = btdr['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain/loss)).iloc[-1]
        vol_base = ret_btdr.ewm(span=20).std().iloc[-1]

        factors = {"beta_btc": beta_btc, "beta_qqq": beta_qqq, "vwap": vwap_30d, "adx": adx, "regime": regime, "rsi": rsi, "vol_base": vol_base}
        
        df_reg = btdr.tail(30).copy(); df_reg['PrevClose'] = df_reg['Close'].shift(1); df_reg = df_reg.dropna()
        x = ((df_reg['Open'] - df_reg['PrevClose']) / df_reg['PrevClose'] * 100).values
        y_high = ((df_reg['High'] - df_reg['PrevClose']) / df_reg['PrevClose'] * 100).values
        y_low = ((df_reg['Low'] - df_reg['PrevClose']) / df_reg['PrevClose'] * 100).values
        cov_h = np.cov(x, y_high); beta_h = cov_h[0, 1] / cov_h[0, 0] if cov_h[0, 0] != 0 else 0.67
        cov_l = np.cov(x, y_low); beta_l = cov_l[0, 1] / cov_l[0, 0] if cov_l[0, 0] != 0 else 0.88
        
        final_model = {
            "high": {"intercept": 0.7*4.29 + 0.3*(np.mean(y_high)-beta_h*np.mean(x)), "beta_open": 0.7*0.67 + 0.3*np.clip(beta_h,0.3,1.2), "beta_btc": 0.52},
            "low": {"intercept": 0.7*-3.22 + 0.3*(np.mean(y_low)-beta_l*np.mean(x)), "beta_open": 0.7*0.88 + 0.3*np.clip(beta_l,0.4,1.5), "beta_btc": 0.42},
            "beta_sector": 0.25
        }
        return final_model, factors, "Grandmaster"
    except: return default_model, default_factors, "Offline"

# --- 6. 渲染函数 ---
def render_ui(data):
    if not data or 'factors' not in data: return

    quotes = data['quotes']
    fng_val = data['fng']
    model_params = data['model']
    factors = data['factors']
    model_status = data['status']
    
    btc_chg = quotes['BTC-USD']['pct']
    qqq_chg = quotes.get('QQQ', {'pct': 0})['pct']
    vix_val = quotes.get('^VIX', {'price': 20})['price']
    vix_chg = quotes.get('^VIX', {'pct': 0})['pct']
    btdr = quotes['BTDR']
    
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny).strftime('%H:%M:%S')
    
    regime_tag = "Trend" if factors['regime'] == "Trend" else "Chop"
    badge_class = "badge-trend" if regime_tag == "Trend" else "badge-chop"
    ph_time.markdown(f"<div class='time-bar'>美东 {now_ny} &nbsp;|&nbsp; 状态: <span class='{badge_class}'>{regime_tag}</span></div>", unsafe_allow_html=True)
    
    ph_btc.markdown(card_html("BTC (全时段)", f"{btc_chg:+.2f}%", f"{btc_chg:+.2f}%", btc_chg), unsafe_allow_html=True)
    ph_fng.markdown(card_html("恐慌指数", f"{fng_val}", None, 0), unsafe_allow_html=True)
    
    peers = ["MARA", "RIOT", "CORZ", "CLSK", "IREN"]
    for i, p in enumerate(peers):
        if p in quotes:
            val = quotes[p]['pct']
            ph_peers[i].markdown(card_html(p, f"{val:+.1f}%", f"{val:+.1f}%", val), unsafe_allow_html=True)
            
    state_map = {"PRE": "dot-reg", "REG": "dot-reg", "POST": "dot-reg", "CLOSED": "dot-closed"}
    dot_class = state_map.get(btdr.get('tag', 'CLOSED'), 'dot-closed')
    status_tag = f"<span class='status-dot {dot_class}'></span>"
    ph_btdr_price.markdown(card_html("BTDR 实时", f"${btdr['price']:.2f}", f"{btdr['pct']:+.2f}%", btdr['pct'], status_tag), unsafe_allow_html=True)
    
    dist_vwap = ((btdr['price'] - factors['vwap']) / factors['vwap']) * 100
    ph_btdr_open.markdown(card_html("机构成本 (VWAP)", f"${factors['vwap']:.2f}", f"{dist_vwap:+.1f}% Prem.", dist_vwap), unsafe_allow_html=True)

    # --- 日内预测 (Intraday) ---
    if btdr['price'] > 0:
        btdr_open_pct = ((btdr['open'] - btdr['prev']) / btdr['prev']) * 100
        peers_avg = sum(quotes[p]['pct'] for p in peers if p in quotes) / 5
        sector_alpha = peers_avg - btc_chg
        sentiment_adj = (fng_val - 50) * 0.02
        
        MODEL = model_params
        pred_high_pct = (MODEL['high']['intercept'] + (MODEL['high']['beta_open'] * btdr_open_pct) + (MODEL['high']['beta_btc'] * btc_chg) + (MODEL['beta_sector'] * sector_alpha) + sentiment_adj)
        pred_low_pct = (MODEL['low']['intercept'] + (MODEL['low']['beta_open'] * btdr_open_pct) + (MODEL['low']['beta_btc'] * btc_chg) + (MODEL['beta_sector'] * sector_alpha) + sentiment_adj)
        pred_high_price = btdr['prev'] * (1 + pred_high_pct / 100)
        pred_low_price = btdr['prev'] * (1 + pred_low_pct / 100)
        
        h_bg = "#e6fcf5" if btdr['price'] < pred_high_price else "#0ca678"; h_txt = "#087f5b" if btdr['price'] < pred_high_price else "#ffffff"
        l_bg = "#fff5f5" if btdr['price'] > pred_low_price else "#e03131"; l_txt = "#c92a2a" if btdr['price'] > pred_low_price else "#ffffff"
        
        ph_pred_high.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background-color: {h_bg}; color: {h_txt}; border: 1px solid #c3fae8;"><div style="font-size: 0.8rem; opacity: 0.8;">日内阻力 (High)</div><div style="font-size: 1.5rem; font-weight: bold;">${pred_high_price:.2f}</div></div></div>""", unsafe_allow_html=True)
        ph_pred_low.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background-color: {l_bg}; color: {l_txt}; border: 1px solid #ffc9c9;"><div style="font-size: 0.8rem; opacity: 0.8;">日内支撑 (Low)</div><div style="font-size: 1.5rem; font-weight: bold;">${pred_low_price:.2f}</div></div></div>""", unsafe_allow_html=True)

    # 因子面板
    ph_macros[0].markdown(factor_html("QQQ (纳指)", f"{qqq_chg:+.2f}%", "Market", qqq_chg), unsafe_allow_html=True)
    ph_macros[1].markdown(factor_html("VIX (恐慌)", f"{vix_val:.1f}", f"{vix_chg:+.1f}%", -vix_chg, reverse_color=True), unsafe_allow_html=True)
    ph_macros[2].markdown(factor_html("Beta (BTC)", f"{factors['beta_btc']:.2f}", "Corr", 0), unsafe_allow_html=True)
    ph_macros[3].markdown(factor_html("Beta (QQQ)", f"{factors['beta_qqq']:.2f}", "Corr", 0), unsafe_allow_html=True)
    
    ph_micros[0].markdown(factor_html("ADX (强度)", f"{factors['adx']:.1f}", factors['regime'], 1 if factors['adx']>25 else -1), unsafe_allow_html=True)
    ph_micros[1].markdown(factor_html("RSI (14d)", f"{factors['rsi']:.0f}", "O/B" if factors['rsi']>70 else ("O/S" if factors['rsi']<30 else "Neu"), 0), unsafe_allow_html=True)
    ph_micros[2].markdown(factor_html("Implied Vol", f"{factors['vol_base']*100:.1f}%", "Risk", 0), unsafe_allow_html=True)
    
    drift_est = (btc_chg/100 * factors['beta_btc'] * 0.4) + (qqq_chg/100 * factors['beta_qqq'] * 0.4)
    if abs(dist_vwap) > 10: drift_est -= (dist_vwap/100) * 0.05
    ph_micros[3].markdown(factor_html("Exp. Drift", f"{drift_est*100:+.2f}%", "Day", drift_est), unsafe_allow_html=True)

    # --- 宗师级蒙特卡洛 (Grandmaster MC) ---
    if btdr['price'] > 0:
        vol = factors['vol_base']
        drift = drift_est
        if vix_val > 25: drift -= 0.005; vol *= 1.3
        if factors['rsi'] > 75: drift -= 0.003
        if factors['rsi'] < 25: drift += 0.003
        if factors['regime'] == "Chop": drift *= 0.5; vol *= 0.8
        
        simulations = 500; days_ahead = 5
        paths = []
        current = btdr['price']
        
        for i in range(simulations):
            path = [current]; p = current
            for d in range(days_ahead):
                shock = np.random.normal(0, 1)
                change = (drift - 0.5 * vol**2) + vol * shock
                p = p * np.exp(change)
                path.append(p)
            paths.append(path)
            
        paths = np.array(paths)
        p90 = np.percentile(paths, 90, axis=0)
        p50 = np.percentile(paths, 50, axis=0)
        p10 = np.percentile(paths, 10, axis=0)
        
        # 宽表数据 (确保 Tooltip 正确显示)
        chart_data = []
        for d in range(days_ahead + 1):
            chart_data.append({
                "Day": d,
                "P90": p90[d],
                "P50": p50[d],
                "P10": p10[d],
            })
        df_chart = pd.DataFrame(chart_data)
        
        # --- Altair 绘图 (Tooltip 修复) ---
        base = alt.Chart(df_chart).encode(x=alt.X('Day:O', title='未来交易日'))
        
        # 1. 区域 (Range P10-P90)
        area = base.mark_area(opacity=0.2, color='#4dabf7').encode(
            y=alt.Y('P10', title='价格预演 (USD)', scale=alt.Scale(zero=False)),
            y2='P90'
        )
        
        # 2. 三条线 (P90/P50/P10)
        l90 = base.mark_line(color='#0ca678', strokeDash=[5,5]).encode(y='P90')
        l50 = base.mark_line(color='#228be6', size=3).encode(y='P50')
        l10 = base.mark_line(color='#d6336c', strokeDash=[5,5]).encode(y='P10')
        
        # 3. 隐形触发器 (Nearest Selector)
        # 使用 selection_point(nearest=True) 确保鼠标靠近 X 轴任意位置都能触发 Tooltip
        nearest = alt.selection_point(nearest=True, on='mouseover', fields=['Day'], empty=False)
        
        # 隐形点，用于捕捉鼠标事件
        selectors = base.mark_point().encode(
            x='Day:O',
            opacity=alt.value(0),
        ).add_params(nearest)
        
        # 4. 显示点 (只有选中时才显示的小黑点)
        points = base.mark_circle(size=60, color="black").encode(
            y='P50',
            opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
            tooltip=[
                alt.Tooltip('Day', title='T+'),
                alt.Tooltip('P90', title='P90 (High)', format='.2f'),
                alt.Tooltip('P50', title='P50 (Median)', format='.2f'),
                alt.Tooltip('P10', title='P10 (Low)', format='.2f')
            ]
        )
        
        # 组合图表
        chart = (area + l90 + l50 + l10 + selectors + points).properties(height=280).interactive()
        
        ph_chart.altair_chart(chart, use_container_width=True)

    ph_footer.caption(f"Engine: v8.6 Rock-Solid | Drift: {drift*100:.2f}% | Vol: {vol*100:.1f}%")

# --- 7. 数据获取 ---
@st.cache_data(ttl=5)
def get_data_v82():
    tickers_list = "BTC-USD BTDR MARA RIOT CORZ CLSK IREN QQQ ^VIX"
    try:
        daily = yf.download(tickers_list, period="5d", interval="1d", group_by='ticker', threads=True, progress=False)
        live = yf.download(tickers_list, period="1d", interval="1m", prepost=True, group_by='ticker', threads=True, progress=False)
        quotes = {}
        symbols = tickers_list.split()
        today_ny = datetime.now(pytz.timezone('America/New_York')).date()
        for sym in symbols:
            try:
                df_day = daily[sym] if sym in daily else pd.DataFrame()
                if not df_day.empty: df_day = df_day.dropna(subset=['Close'])
                df_min = live[sym] if sym in live else pd.DataFrame()
                if not df_min.empty: df_min = df_min.dropna(subset=['Close'])
                
                state = "REG" if not df_min.empty else "CLOSED"
                current_price = df_min['Close'].iloc[-1] if not df_min.empty else (df_day['Close'].iloc[-1] if not df_day.empty else 0)
                
                prev_close = 1.0
                if not df_day.empty:
                    last_date = df_day.index[-1].date()
                    if last_date == today_ny:
                        if len(df_day) >= 2: prev_close = df_day['Close'].iloc[-2]
                        elif not df_day.empty: prev_close = df_day['Open'].iloc[-1]
                    else: prev_close = df_day['Close'].iloc[-1]
                
                pct = ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0
                open_price = df_day['Open'].iloc[-1] if not df_day.empty and df_day.index[-1].date() == today_ny else current_price
                quotes[sym] = {"price": current_price, "pct": pct, "prev": prev_close, "open": open_price, "tag": state}
            except: quotes[sym] = {"price": 0, "pct": 0, "prev": 0, "open": 0, "tag": "ERR"}
        return quotes
    except: return None

# --- 8. 执行流 ---
if st.session_state['data_cache'] and 'grandmaster' in st.session_state['data_cache']: render_ui(st.session_state['data_cache'])
else: ph_time.info("📡 正在融合全维数据...")

new_quotes = get_data_v82()
ai_model, ai_factors, ai_status = run_grandmaster_analytics()

if new_quotes:
    try: fng = int(requests.get("https://api.alternative.me/fng/", timeout=1).json()['data'][0]['value'])
    except: fng = 50
    st.session_state['data_cache'] = {'quotes': new_quotes, 'fng': fng, 'model': ai_model, 'factors': ai_factors, 'status': ai_status, 'grandmaster': True}
    render_ui(st.session_state['data_cache'])
