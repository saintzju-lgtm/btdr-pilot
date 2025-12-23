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
st.set_page_config(page_title="BTDR Pilot v8.0", layout="centered")

# 5秒刷新
st_autorefresh(interval=5000, limit=None, key="realtime_counter")

# CSS: 保持 UI 统一
st.markdown("""
    <style>
    html { overflow-y: scroll; }
    .stApp > header { display: none; }
    .stApp { margin-top: -30px; background-color: #ffffff; }
    div[data-testid="stStatusWidget"] { visibility: hidden; }
    
    h1, h2, h3, div, p, span { 
        color: #212529 !important; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important; 
    }
    
    .metric-card {
        background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02); height: 95px; padding: 0 16px;
        display: flex; flex-direction: column; justify-content: center;
    }
    .metric-label { font-size: 0.75rem; color: #888; margin-bottom: 2px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #212529; line-height: 1.2; }
    .metric-delta { font-size: 0.9rem; font-weight: 600; margin-top: 2px; }
    
    .macro-box {
        background: #fff; border: 1px solid #eee; border-radius: 8px; padding: 8px; text-align: center;
        height: 80px; display: flex; flex-direction: column; justify-content: center;
    }
    .macro-title { font-size: 0.7rem; color: #888; }
    .macro-val { font-size: 1.1rem; font-weight: bold; color: #495057; }
    .macro-sub { font-size: 0.7rem; font-weight: 600; }
    
    /* 状态徽章 */
    .regime-badge { padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; font-weight: bold; color: white; margin-left: 5px; }
    .regime-trend { background-color: #fd7e14; } /* 橙色趋势 */
    .regime-chop { background-color: #868e96; } /* 灰色震荡 */
    
    .color-up { color: #0ca678; } .color-down { color: #d6336c; } .color-neutral { color: #adb5bd; }
    .status-dot { height: 6px; width: 6px; border-radius: 50%; display: inline-block; margin-left: 6px; }
    .dot-reg { background-color: #0ca678; } .dot-closed { background-color: #adb5bd; }
    
    .pred-container-wrapper { height: 110px; width: 100%; display: block; }
    .pred-box { padding: 0 10px; border-radius: 12px; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center; }
    .time-bar { font-size: 0.75rem; color: #999; text-align: center; margin-bottom: 20px; padding: 6px; background: #fafafa; border-radius: 6px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 状态管理 ---
if 'data_cache' not in st.session_state: st.session_state['data_cache'] = None
# 缓存自检：如果缺少 v8.0 的新字段 (regime, vwap)，清空重跑
if st.session_state['data_cache'] and 'vwap' not in st.session_state['data_cache'].get('factors', {}):
    st.session_state['data_cache'] = None
    st.rerun()

# --- 3. 辅助函数 ---
def card_html(label, value_str, delta_str=None, delta_val=0, extra_tag=""):
    delta_html = ""
    if delta_str:
        color_class = "color-up" if delta_val >= 0 else "color-down"
        delta_html = f"<div class='metric-delta {color_class}'>{delta_str}</div>"
    return f"""<div class="metric-card"><div class="metric-label">{label} {extra_tag}</div><div class="metric-value">{value_str}</div>{delta_html}</div>"""

def macro_html(title, val, delta_str, delta_val):
    color_class = "color-up" if delta_val >= 0 else "color-down"
    if title == "VIX": color_class = "color-down" if delta_val >= 0 else "color-up"
    return f"""<div class="macro-box"><div class="macro-title">{title}</div><div class="macro-val">{val}</div><div class="macro-sub {color_class}">{delta_str}</div></div>"""

st.markdown("### ⚡ BTDR 领航员 v8.0")

# --- 4. UI 骨架 ---
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

st.markdown("### 🌍 机构因子 (Institutional Factors)")
m1, m2, m3, m4 = st.columns(4)
ph_macros = [m1.empty(), m2.empty(), m3.empty(), m4.empty()]

st.markdown("### 🎯 智能推演 (Smart MC)")
ph_chart = st.empty()
col_mc1, col_mc2 = st.columns(2)
with col_mc1: ph_mc_bull = st.empty()
with col_mc2: ph_mc_bear = st.empty()

st.markdown("---")
ph_footer = st.empty()

# --- 5. 核心：机构级量化引擎 ---
@st.cache_data(ttl=300) 
def run_institutional_analytics():
    default_model = {"high": {"intercept": 4.29, "beta_open": 0.67, "beta_btc": 0.52}, "low": {"intercept": -3.22, "beta_open": 0.88, "beta_btc": 0.42}, "beta_sector": 0.25}
    default_factors = {"vwap": 0, "adx": 0, "regime": "Neutral", "beta_btc": 1.5, "vol_base": 0.05}
    
    try:
        # 下载数据 (包含 Volume)
        data = yf.download("BTDR BTC-USD QQQ ^VIX", period="3mo", interval="1d", group_by='ticker', threads=True, progress=False)
        
        btdr = data['BTDR'].dropna()
        btc = data['BTC-USD'].dropna()
        qqq = data['QQQ'].dropna()
        
        # 对齐
        idx = btdr.index.intersection(btc.index).intersection(qqq.index)
        btdr = btdr.loc[idx]; btc = btc.loc[idx]; qqq = qqq.loc[idx]
        
        # --- A. 计算高级因子 ---
        
        # 1. VWAP (近似计算：基于收盘价和成交量)
        # 真实VWAP需要分钟线，这里用日线 (Close * Volume).cumsum / Volume.cumsum 近似长期趋势
        # 日内VWAP我们会在 render_ui 里用分钟线算，这里算长期支撑位
        btdr['TP'] = (btdr['High'] + btdr['Low'] + btdr['Close']) / 3
        btdr['PV'] = btdr['TP'] * btdr['Volume']
        vwap_30d = btdr['PV'].tail(30).sum() / btdr['Volume'].tail(30).sum()
        
        # 2. ADX (趋势强度)
        # 简化版 ADX 计算
        high = btdr['High']; low = btdr['Low']; close = btdr['Close']
        tr = np.maximum(high - low, np.abs(high - close.shift(1)))
        atr = tr.rolling(14).mean()
        up_move = high - high.shift(1); down_move = low.shift(1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean().iloc[-1]
        
        # 判断市场状态 (Regime)
        regime = "Trend" if adx > 25 else "Chop"
        
        # 3. 量价背离 (OBV Slope)
        obv = (np.sign(btdr['Close'].diff()) * btdr['Volume']).fillna(0).cumsum()
        obv_slope = (obv.iloc[-1] - obv.iloc[-5]) / 5 # 5日OBV斜率
        
        # 4. Beta & Vol
        ret_btdr = btdr['Close'].pct_change()
        ret_btc = btc['Close'].pct_change()
        cov = ret_btdr.rolling(60).cov(ret_btc).iloc[-1]
        var = ret_btc.rolling(60).var().iloc[-1]
        beta_btc = cov / var if var != 0 else 1.5
        vol_base = ret_btdr.ewm(span=20).std().iloc[-1]

        factors = {
            "vwap": vwap_30d,
            "adx": adx,
            "regime": regime,
            "obv_trend": "Bull" if obv_slope > 0 else "Bear",
            "beta_btc": beta_btc,
            "vol_base": vol_base
        }
        
        # --- B. 更新模型 ---
        # 如果是 Trend 状态，增加 Momentum 权重；如果是 Chop，增加 Mean Reversion
        # 这里直接复用之前的 Regression 逻辑，但参数会传递给前端
        df_reg = btdr.tail(30).copy()
        df_reg['PrevClose'] = df_reg['Close'].shift(1)
        df_reg = df_reg.dropna()
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
        
        return final_model, factors, "Inst. Engine"
        
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
    vix_val = quotes.get('^VIX', {'price': 20})['price']
    btdr = quotes['BTDR']
    
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny).strftime('%H:%M:%S')
    
    # 状态徽章颜色
    regime_color = "#fd7e14" if factors['regime'] == "Trend" else "#868e96"
    regime_html = f"<span style='background-color:{regime_color}; padding:2px 6px; border-radius:4px; font-size:0.7rem; font-weight:bold; color:white; margin-left:5px;'>{factors['regime']}</span>"
    
    ph_time.markdown(f"<div class='time-bar'>美东 {now_ny} &nbsp;|&nbsp; 市场状态: {regime_html}</div>", unsafe_allow_html=True)
    
    ph_btc.markdown(card_html("BTC (全时段)", f"{btc_chg:+.2f}%", f"{btc_chg:+.2f}%", btc_chg), unsafe_allow_html=True)
    ph_fng.markdown(card_html("恐慌指数", f"{fng_val}", None, 0), unsafe_allow_html=True)
    
    peers = ["MARA", "RIOT", "CORZ", "CLSK", "IREN"]
    for i, p in enumerate(peers):
        if p in quotes:
            val = quotes[p]['pct']
            ph_peers[i].markdown(card_html(p, f"{val:+.1f}%", f"{val:+.1f}%", val), unsafe_allow_html=True)
            
    # BTDR 本体
    state_map = {"PRE": "dot-reg", "REG": "dot-reg", "POST": "dot-reg", "CLOSED": "dot-closed"}
    dot_class = state_map.get(btdr.get('tag', 'CLOSED'), 'dot-closed')
    status_tag = f"<span class='status-dot {dot_class}'></span>"
    ph_btdr_price.markdown(card_html("BTDR 实时", f"${btdr['price']:.2f}", f"{btdr['pct']:+.2f}%", btdr['pct'], status_tag), unsafe_allow_html=True)
    
    # VWAP 显示 (日内)
    # 如果日内有数据，粗略估算一个日内 VWAP (P * 1 / 1)
    # 更好的方式是直接用 30日 VWAP 作为"机构成本线"
    ph_btdr_open.markdown(card_html("机构成本 (30d VWAP)", f"${factors['vwap']:.2f}", "Support Level", 0), unsafe_allow_html=True)

    # --- 渲染机构因子面板 ---
    # 显示最核心的4个因子：ADX (状态), VWAP距离, Beta, VIX
    adx_str = f"{factors['adx']:.1f}"
    dist_vwap = ((btdr['price'] - factors['vwap']) / factors['vwap']) * 100
    
    ph_macros[0].markdown(macro_html("ADX (趋势强度)", adx_str, factors['regime'], 1 if factors['adx']>25 else -1), unsafe_allow_html=True)
    ph_macros[1].markdown(macro_html("Vs 机构成本", f"{dist_vwap:+.1f}%", "Premium", dist_vwap), unsafe_allow_html=True)
    ph_macros[2].markdown(macro_html("Beta (BTC)", f"{factors['beta_btc']:.2f}", "Leverage", 0), unsafe_allow_html=True)
    ph_macros[3].markdown(macro_html("VIX (恐慌)", f"{vix_val:.1f}", "Risk", -1 if vix_val>20 else 1), unsafe_allow_html=True)

    # --- 智能推演 (Smart MC) ---
    if btdr['price'] > 0:
        vol = factors['vol_base']
        beta_b = factors['beta_btc']
        
        # 智能漂移 (Smart Drift)
        # 1. 基础漂移：由 BTC 驱动
        drift = (btc_chg/100 * beta_b * 0.5)
        
        # 2. 状态修正 (Regime Adjustment)
        # 如果是 "Chop" (震荡)，漂移趋向于 0 (均值回归)
        # 如果是 "Trend" (趋势)，漂移保持原样，甚至放大
        if factors['regime'] == "Chop":
            drift *= 0.5 # 震荡市，上涨动能打折
            vol *= 0.8   # 震荡市，波动率也通常收敛
        
        # 3. 恐慌修正
        if vix_val > 25: 
            drift -= 0.005 # 强行扣减
            vol *= 1.5     # 波动率激增
            
        # 4. VWAP 修正 (引力)
        # 价格偏离 VWAP 太远会有回归引力
        if dist_vwap > 15: drift -= 0.002 # 涨太多了，引力向下
        if dist_vwap < -15: drift += 0.002 # 跌太多了，引力向上

        simulations = 500
        days_ahead = 5
        paths = []
        current = btdr['price']
        
        for i in range(simulations):
            path = [current]
            p = current
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
        
        chart_df = []
        for d in range(days_ahead + 1):
            chart_df.append({"Day": d, "Price": p50[d], "Type": "P50 (中枢)"})
            chart_df.append({"Day": d, "Price": p90[d], "Type": "P90 (压力)"})
            chart_df.append({"Day": d, "Price": p10[d], "Type": "P10 (支撑)"})
        df_chart = pd.DataFrame(chart_df)
        
        base = alt.Chart(df_chart).encode(x=alt.X('Day:O', title='未来交易日'))
        area = base.mark_area(opacity=0.3, color='#4dabf7').encode(y=alt.Y('Price', scale=alt.Scale(zero=False)), y2='Price_Low').transform_filter(alt.FieldOneOfPredicate(field='Type', oneOf=['P90 (压力)'])).transform_lookup(lookup='Day', from_=alt.LookupData(df_chart[df_chart['Type'] == 'P10 (支撑)'], 'Day', ['Price']), as_=['Price_Low'])
        lines = base.mark_line().encode(y='Price', color=alt.Color('Type', scale=alt.Scale(domain=['P90 (压力)', 'P50 (中枢)', 'P10 (支撑)'], range=['#0ca678', '#228be6', '#fa5252'])))
        ph_chart.altair_chart((area + lines).properties(height=240).interactive(), use_container_width=True)
        
        p90_end = p90[-1]; p90_pct = (p90_end - current)/current * 100
        p10_end = p10[-1]; p10_pct = (p10_end - current)/current * 100
        ph_mc_bull.markdown(card_html("P90 强压位", f"${p90_end:.2f}", f"{p90_pct:+.1f}%", p90_pct), unsafe_allow_html=True)
        ph_mc_bear.markdown(card_html("P10 强撑位", f"${p10_end:.2f}", f"{p10_pct:+.1f}%", p10_pct), unsafe_allow_html=True)

    ph_footer.caption(f"Engine: v8.0 Institutional | Regime: {factors['regime']} | VWAP Adj: {'Active' if abs(dist_vwap)>10 else 'None'}")

# --- 7. 数据获取 ---
@st.cache_data(ttl=5)
def get_data_v80():
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
if st.session_state['data_cache'] and 'factors' in st.session_state['data_cache']: render_ui(st.session_state['data_cache'])
else: ph_time.info("📡 正在计算 VWAP 和 ADX...")

new_quotes = get_data_v80()
ai_model, ai_factors, ai_status = run_institutional_analytics()

if new_quotes:
    try: fng = int(requests.get("https://api.alternative.me/fng/", timeout=1).json()['data'][0]['value'])
    except: fng = 50
    st.session_state['data_cache'] = {'quotes': new_quotes, 'fng': fng, 'model': ai_model, 'factors': ai_factors, 'status': ai_status}
    render_ui(st.session_state['data_cache'])
