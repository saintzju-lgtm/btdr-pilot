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
st.set_page_config(page_title="BTDR Pilot v7.8", layout="centered")

# 5秒刷新
st_autorefresh(interval=5000, limit=None, key="realtime_counter")

# CSS: 保持 UI 统一 (增加因子面板样式)
st.markdown("""
    <style>
    html { overflow-y: scroll; }
    .stApp > header { display: none; }
    .stApp { margin-top: -30px; background-color: #ffffff; }
    div[data-testid="stStatusWidget"] { visibility: hidden; }
    
    h1, h2, h3, div, p, span { 
        color: #212529 !important; 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important; 
    }
    
    .metric-card {
        background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02); height: 95px; padding: 0 16px;
        display: flex; flex-direction: column; justify-content: center;
    }
    .metric-label { font-size: 0.75rem; color: #888; margin-bottom: 2px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #212529; line-height: 1.2; }
    .metric-delta { font-size: 0.9rem; font-weight: 600; margin-top: 2px; }
    
    .factor-box {
        background: #fff; border: 1px solid #eee; border-radius: 8px; padding: 8px; text-align: center;
    }
    .factor-title { font-size: 0.7rem; color: #888; }
    .factor-val { font-size: 1.1rem; font-weight: bold; color: #495057; }
    
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
# 缓存自检
if st.session_state['data_cache'] and 'factors' not in st.session_state['data_cache']:
    st.session_state['data_cache'] = None
    st.rerun()

# --- 3. 辅助函数 ---
def card_html(label, value_str, delta_str=None, delta_val=0, extra_tag=""):
    delta_html = ""
    if delta_str:
        color_class = "color-up" if delta_val >= 0 else "color-down"
        delta_html = f"<div class='metric-delta {color_class}'>{delta_str}</div>"
    return f"""
    <div class="metric-card"><div class="metric-label">{label} {extra_tag}</div><div class="metric-value">{value_str}</div>{delta_html}</div>
    """

def factor_html(title, val):
    return f"""<div class="factor-box"><div class="factor-title">{title}</div><div class="factor-val">{val}</div></div>"""

st.markdown("### ⚡ BTDR 领航员 v7.8")

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
# BTDR 本体
c3, c4 = st.columns(2)
with c3: ph_btdr_price = st.empty()
with c4: ph_btdr_open = st.empty()

# 日内预测
st.markdown("### 🎯 日内阻力/支撑 (Intraday)")
col_h, col_l = st.columns(2)
with col_h: ph_pred_high = st.empty()
with col_l: ph_pred_low = st.empty()

# 多因子量化推演
st.markdown("### 🧠 多因子量化推演 (Quant MC)")
# 因子展示栏
f1, f2, f3, f4 = st.columns(4)
ph_factors = [f1.empty(), f2.empty(), f3.empty(), f4.empty()]

ph_chart = st.empty()
col_mc1, col_mc2 = st.columns(2)
with col_mc1: ph_mc_bull = st.empty()
with col_mc2: ph_mc_bear = st.empty()

st.markdown("---")
ph_footer = st.empty()

# --- 5. 核心量化引擎 (Quant Engine) ---
@st.cache_data(ttl=3600) # 1小时重新计算一次因子
def run_quant_analytics():
    default_model = {"high": {"intercept": 4.29, "beta_open": 0.67, "beta_btc": 0.52}, "low": {"intercept": -3.22, "beta_open": 0.88, "beta_btc": 0.42}, "beta_sector": 0.25}
    default_factors = {"rsi": 50, "beta": 1.5, "vol": 0.05, "drift": 0}
    
    try:
        # 1. 获取数据 (BTDR + BTC) 过去90天
        # 使用 threads=True 加速
        data = yf.download("BTDR BTC-USD", period="3mo", interval="1d", group_by='ticker', threads=True, progress=False)
        
        # 提取数据
        btdr = data['BTDR'].dropna()
        btc = data['BTC-USD'].dropna()
        
        # 对齐索引
        common_index = btdr.index.intersection(btc.index)
        btdr = btdr.loc[common_index]
        btc = btc.loc[common_index]
        
        # --- A. 计算因子 ---
        
        # 1. 动态 Beta (Rolling 60 days)
        # 计算 BTDR 和 BTC 的收益率
        ret_btdr = btdr['Close'].pct_change()
        ret_btc = btc['Close'].pct_change()
        
        covariance = ret_btdr.rolling(window=60).cov(ret_btc)
        variance = ret_btc.rolling(window=60).var()
        rolling_beta = covariance / variance
        current_beta = rolling_beta.iloc[-1]
        if np.isnan(current_beta): current_beta = 1.8 # 默认值
        
        # 2. RSI (14 days)
        delta = btdr['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # 3. 波动率 (EWMA - 指数加权，近期权重更大)
        # 使用 span=10，相当于更看重最近10天的波动
        volatility = ret_btdr.ewm(span=10).std().iloc[-1]
        
        # 4. 趋势漂移 (Drift) - 结合 BTC 趋势
        # 如果 BTC 处于 20日均线之上，drift 为正，否则为负
        btc_ma20 = btc['Close'].rolling(window=20).mean().iloc[-1]
        btc_current = btc['Close'].iloc[-1]
        
        # 基础漂移 (BTDR 自身动量)
        base_drift = ret_btdr.mean() 
        
        # BTC 修正漂移
        btc_trend_adjust = 0.002 if btc_current > btc_ma20 else -0.002
        
        # RSI 均值回归修正 (Mean Reversion)
        # RSI > 70 视为超买，向下修正; RSI < 30 超卖，向上修正
        rsi_adjust = 0
        if rsi > 70: rsi_adjust = -0.005 
        elif rsi < 30: rsi_adjust = 0.005
        
        final_drift = base_drift + (btc_trend_adjust * current_beta) + rsi_adjust
        
        factors = {
            "rsi": rsi,
            "beta": current_beta,
            "vol": volatility,
            "drift": final_drift,
            "btc_trend": "Bull" if btc_current > btc_ma20 else "Bear"
        }
        
        # --- B. 更新日内模型 (Regression) ---
        # 保持之前的回归逻辑，不再赘述
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
        
        return final_model, factors, "Quant Active"
        
    except Exception as e:
        return default_model, default_factors, "Offline Mode"

# --- 6. 渲染函数 ---
def render_ui(data):
    if not data or 'factors' not in data: return

    quotes = data['quotes']
    fng_val = data['fng']
    model_params = data['model']
    factors = data['factors']
    model_status = data['status']
    
    btc_chg = quotes['BTC-USD']['pct']
    btdr = quotes['BTDR']
    
    # 时间
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny).strftime('%H:%M:%S')
    ph_time.markdown(f"<div class='time-bar'>美东时间 {now_ny} &nbsp;|&nbsp; 核心: {model_status}</div>", unsafe_allow_html=True)
    
    # 顶部指标
    ph_btc.markdown(card_html("BTC (全时段)", f"{btc_chg:+.2f}%", f"{btc_chg:+.2f}%", btc_chg), unsafe_allow_html=True)
    ph_fng.markdown(card_html("恐慌指数", f"{fng_val}", None, 0), unsafe_allow_html=True)
    
    # 板块
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
    
    # 日内预测计算 (略，保持不变)
    valid_peers = [p for p in peers if quotes[p]['price'] > 0]
    peers_avg = sum(quotes[p]['pct'] for p in valid_peers) / len(valid_peers) if valid_peers else 0
    sector_alpha = peers_avg - btc_chg
    sentiment_adj = (fng_val - 50) * 0.02
    if btdr['price'] > 0:
        btdr_open_pct = ((btdr['open'] - btdr['prev']) / btdr['prev']) * 100
        MODEL = model_params
        pred_high_pct = (MODEL['high']['intercept'] + (MODEL['high']['beta_open'] * btdr_open_pct) + (MODEL['high']['beta_btc'] * btc_chg) + (MODEL['beta_sector'] * sector_alpha) + sentiment_adj)
        pred_low_pct = (MODEL['low']['intercept'] + (MODEL['low']['beta_open'] * btdr_open_pct) + (MODEL['low']['beta_btc'] * btc_chg) + (MODEL['beta_sector'] * sector_alpha) + sentiment_adj)
        pred_high_price = btdr['prev'] * (1 + pred_high_pct / 100)
        pred_low_price = btdr['prev'] * (1 + pred_low_pct / 100)
        
        ph_btdr_open.markdown(card_html("计算用开盘", f"${btdr['open']:.2f}", f"{btdr_open_pct:+.2f}%", btdr_open_pct), unsafe_allow_html=True)
        h_bg = "#e6fcf5" if btdr['price'] < pred_high_price else "#0ca678"; h_txt = "#087f5b" if btdr['price'] < pred_high_price else "#ffffff"
        l_bg = "#fff5f5" if btdr['price'] > pred_low_price else "#e03131"; l_txt = "#c92a2a" if btdr['price'] > pred_low_price else "#ffffff"
        ph_pred_high.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background-color: {h_bg}; color: {h_txt}; border: 1px solid #c3fae8;"><div style="font-size: 0.8rem; opacity: 0.8;">日内阻力 (High)</div><div style="font-size: 1.5rem; font-weight: bold;">${pred_high_price:.2f}</div></div></div>""", unsafe_allow_html=True)
        ph_pred_low.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background-color: {l_bg}; color: {l_txt}; border: 1px solid #ffc9c9;"><div style="font-size: 0.8rem; opacity: 0.8;">日内支撑 (Low)</div><div style="font-size: 1.5rem; font-weight: bold;">${pred_low_price:.2f}</div></div></div>""", unsafe_allow_html=True)

    # --- 渲染因子面板 ---
    ph_factors[0].markdown(factor_html("RSI (14d)", f"{factors['rsi']:.1f}"), unsafe_allow_html=True)
    ph_factors[1].markdown(factor_html("Beta (vs BTC)", f"{factors['beta']:.2f}"), unsafe_allow_html=True)
    ph_factors[2].markdown(factor_html("Implied Vol", f"{factors['vol']*100:.1f}%"), unsafe_allow_html=True)
    drift_pct = factors['drift'] * 100
    ph_factors[3].markdown(factor_html("Drift/Day", f"{drift_pct:+.2f}%"), unsafe_allow_html=True)

    # --- 渲染多因子蒙特卡洛 (Advanced MC) ---
    if btdr['price'] > 0:
        vol = factors['vol']
        drift = factors['drift'] # 这是一个经过RSI和BTC趋势修正过的漂移值
        current = btdr['price']
        
        simulations = 500 # 增加模拟次数
        days_ahead = 5
        paths = []
        
        for i in range(simulations):
            path = [current]
            p = current
            for d in range(days_ahead):
                # GBM: S_t = S_{t-1} * exp((mu - 0.5*sigma^2) + sigma*Z)
                shock = np.random.normal(0, 1)
                change = (drift - 0.5 * vol**2) + vol * shock
                p = p * np.exp(change)
                path.append(p)
            paths.append(path)
            
        paths = np.array(paths)
        p90 = np.percentile(paths, 90, axis=0) # 乐观
        p50 = np.percentile(paths, 50, axis=0) # 中性
        p10 = np.percentile(paths, 10, axis=0) # 悲观
        
        chart_df = []
        for d in range(days_ahead + 1):
            chart_df.append({"Day": d, "Price": p50[d], "Type": "P50 (中位数)"})
            chart_df.append({"Day": d, "Price": p90[d], "Type": "P90 (乐观)"})
            chart_df.append({"Day": d, "Price": p10[d], "Type": "P10 (悲观)"})
            
        df_chart = pd.DataFrame(chart_df)
        
        # 扇形图 (Fan Chart)
        base = alt.Chart(df_chart).encode(x=alt.X('Day:O', title='未来交易日'))
        
        # 80% 置信区间 (P10-P90)
        area = base.mark_area(opacity=0.3, color='#4dabf7').encode(
            y=alt.Y('Price', title='价格分布 (USD)', scale=alt.Scale(zero=False)),
            y2='Price_Low'
        ).transform_filter(
            alt.FieldOneOfPredicate(field='Type', oneOf=['P90 (乐观)'])
        ).transform_lookup(
            lookup='Day',
            from_=alt.LookupData(df_chart[df_chart['Type'] == 'P10 (悲观)'], 'Day', ['Price']),
            as_=['Price_Low']
        )
        
        lines = base.mark_line().encode(
            y='Price',
            color=alt.Color('Type', scale=alt.Scale(domain=['P90 (乐观)', 'P50 (中位数)', 'P10 (悲观)'], range=['#0ca678', '#228be6', '#fa5252'])),
            tooltip=['Day', 'Price', 'Type']
        )
        
        chart = (area + lines).properties(height=240).interactive()
        ph_chart.altair_chart(chart, use_container_width=True)
        
        # 结果卡片
        p90_end = p90[-1]; p90_pct = (p90_end - current)/current * 100
        p10_end = p10[-1]; p10_pct = (p10_end - current)/current * 100
        
        ph_mc_bull.markdown(card_html("P90 乐观边界", f"${p90_end:.2f}", f"{p90_pct:+.1f}%", p90_pct), unsafe_allow_html=True)
        ph_mc_bear.markdown(card_html("P10 悲观边界", f"${p10_end:.2f}", f"{p10_pct:+.1f}%", p10_pct), unsafe_allow_html=True)

    ph_footer.caption(f"Risk Model: Multivariate MC | Simulations: 500 | Drift Adjusted")

# --- 7. 数据获取 ---
@st.cache_data(ttl=5)
def get_data_v78():
    tickers_list = "BTC-USD BTDR MARA RIOT CORZ CLSK IREN"
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
else: ph_time.info("📡 正在加载量化因子...")

new_quotes = get_data_v78()
ai_model, ai_factors, ai_status = run_quant_analytics()

if new_quotes:
    try: fng = int(requests.get("https://api.alternative.me/fng/", timeout=1).json()['data'][0]['value'])
    except: fng = 50
    st.session_state['data_cache'] = {'quotes': new_quotes, 'fng': fng, 'model': ai_model, 'factors': ai_factors, 'status': ai_status}
    render_ui(st.session_state['data_cache'])
