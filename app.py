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
st.set_page_config(page_title="BTDR Pilot v7.7", layout="centered")

# 5秒刷新
st_autorefresh(interval=5000, limit=None, key="realtime_counter")

# CSS: 保持 UI 统一
st.markdown("""
    <style>
    /* 基础重置 */
    html { overflow-y: scroll; }
    .stApp > header { display: none; }
    .stApp { margin-top: -30px; background-color: #ffffff; }
    div[data-testid="stStatusWidget"] { visibility: hidden; }
    
    h1, h2, h3, div, p, span { 
        color: #212529 !important; 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important; 
    }
    
    /* 统一指标卡片 */
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        height: 95px;
        padding: 0 16px;
        display: flex; flex-direction: column; justify-content: center;
        overflow: hidden;
    }
    
    .metric-label { font-size: 0.75rem; color: #888; display: flex; align-items: center; margin-bottom: 2px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #212529; line-height: 1.2; letter-spacing: -0.5px; }
    .metric-delta { font-size: 0.9rem; font-weight: 600; margin-top: 2px; }
    
    .color-up { color: #0ca678; }
    .color-down { color: #d6336c; }
    .color-neutral { color: #adb5bd; }
    
    /* 预测容器 */
    .pred-container-wrapper { height: 110px; width: 100%; display: block; }
    .pred-box {
        padding: 0 10px; border-radius: 12px; text-align: center;
        height: 100%; display: flex; flex-direction: column; justify-content: center;
    }
    
    /* 状态小圆点 */
    .status-dot { height: 6px; width: 6px; border-radius: 50%; display: inline-block; margin-left: 6px; margin-bottom: 2px;}
    .dot-pre { background-color: #f59f00; box-shadow: 0 0 4px #f59f00; }
    .dot-reg { background-color: #0ca678; box-shadow: 0 0 4px #0ca678; }
    .dot-post { background-color: #1c7ed6; box-shadow: 0 0 4px #1c7ed6; }
    .dot-closed { background-color: #adb5bd; }
    
    .time-bar {
        font-size: 0.75rem; color: #999; text-align: center;
        margin-bottom: 20px; padding: 6px; background: #fafafa; border-radius: 6px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 状态管理与自愈 ---
if 'data_cache' not in st.session_state: st.session_state['data_cache'] = None

# 自愈：如果缓存里没有 monte_carlo 数据，清空重跑
if st.session_state['data_cache'] is not None:
    if 'mc_data' not in st.session_state['data_cache']:
        st.session_state['data_cache'] = None
        st.rerun()

# --- 3. 辅助函数 ---
def card_html(label, value_str, delta_str=None, delta_val=0, extra_tag=""):
    delta_html = ""
    if delta_str:
        color_class = "color-up" if delta_val >= 0 else "color-down"
        delta_html = f"<div class='metric-delta {color_class}'>{delta_str}</div>"
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label} {extra_tag}</div>
        <div class="metric-value">{value_str}</div>
        {delta_html}
    </div>
    """

st.markdown("### ⚡ BTDR 领航员 v7.7")

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

# 蒙特卡洛预测
st.markdown("### ☁️ 概率云推演 (Monte Carlo)")
ph_chart = st.empty() # 图表占位符
col_mc1, col_mc2 = st.columns(2)
with col_mc1: ph_mc_bull = st.empty()
with col_mc2: ph_mc_bear = st.empty()

st.markdown("---")
ph_footer = st.empty()

# --- 5. 核心逻辑：蒙特卡洛模拟 ---
@st.cache_data(ttl=3600)
def run_analytics():
    default_model = {
        "high": {"intercept": 4.29, "beta_open": 0.67, "beta_btc": 0.52},
        "low":  {"intercept": -3.22, "beta_open": 0.88, "beta_btc": 0.42},
        "beta_sector": 0.25
    }
    # 默认模拟参数
    mc_params = {"volatility": 0.05, "drift": 0.0}

    try:
        # 1. 获取历史数据
        df = yf.download("BTDR", period="3mo", interval="1d", progress=False)
        if len(df) < 30: return default_model, mc_params, "数据不足"
        
        if isinstance(df.columns, pd.MultiIndex): df = df.xs('BTDR', axis=1, level=1)
        df = df.dropna()
        
        # 2. 计算日内模型 (Regression)
        df_reg = df.tail(30).copy() # 只用最近30天算回归
        df_reg['PrevClose'] = df_reg['Close'].shift(1)
        df_reg = df_reg.dropna()
        
        x = ((df_reg['Open'] - df_reg['PrevClose']) / df_reg['PrevClose'] * 100).values
        y_high = ((df_reg['High'] - df_reg['PrevClose']) / df_reg['PrevClose'] * 100).values
        y_low = ((df_reg['Low'] - df_reg['PrevClose']) / df_reg['PrevClose'] * 100).values
        
        cov_h = np.cov(x, y_high); beta_h = cov_h[0, 1] / cov_h[0, 0] if cov_h[0, 0] != 0 else 0.67
        cov_l = np.cov(x, y_low); beta_l = cov_l[0, 1] / cov_l[0, 0] if cov_l[0, 0] != 0 else 0.88
        
        # 3. 计算蒙特卡洛参数 (Volatility & Drift)
        # 使用对数收益率
        log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
        volatility = log_returns.std() # 日波动率
        drift = log_returns.mean() # 日均漂移 (趋势)
        
        mc_params = {"volatility": volatility, "drift": drift}

        final_model = {
            "high": {"intercept": 0.7*4.29 + 0.3*(np.mean(y_high)-beta_h*np.mean(x)), "beta_open": 0.7*0.67 + 0.3*np.clip(beta_h,0.3,1.2), "beta_btc": 0.52},
            "low": {"intercept": 0.7*-3.22 + 0.3*(np.mean(y_low)-beta_l*np.mean(x)), "beta_open": 0.7*0.88 + 0.3*np.clip(beta_l,0.4,1.5), "beta_btc": 0.42},
            "beta_sector": 0.25
        }
        return final_model, mc_params, "Monte Carlo Ready"
    except: 
        return default_model, mc_params, "离线模式"

# --- 6. 渲染函数 ---
def render_ui(data):
    if not data: return
    if 'mc_params' not in data: return

    quotes = data['quotes']
    fng_val = data['fng']
    model_params = data['model']
    mc_params = data['mc_params']
    model_status = data['status']
    
    btc_chg = quotes['BTC-USD']['pct']
    btdr = quotes['BTDR']
    
    # 时间
    tz_bj = pytz.timezone('Asia/Shanghai')
    tz_ny = pytz.timezone('America/New_York')
    now_bj = datetime.now(tz_bj).strftime('%H:%M:%S')
    now_ny = datetime.now(tz_ny).strftime('%H:%M:%S')
    ph_time.markdown(f"<div class='time-bar'>北京 {now_bj} &nbsp;|&nbsp; 美东 {now_ny} &nbsp;|&nbsp; 系统 {model_status}</div>", unsafe_allow_html=True)
    
    # 指标 & 板块
    ph_btc.markdown(card_html("BTC (全时段)", f"{btc_chg:+.2f}%", f"{btc_chg:+.2f}%", btc_chg), unsafe_allow_html=True)
    ph_fng.markdown(card_html("恐慌指数", f"{fng_val}", None, 0), unsafe_allow_html=True)
    
    peers = ["MARA", "RIOT", "CORZ", "CLSK", "IREN"]
    for i, p in enumerate(peers):
        if p in quotes:
            val = quotes[p]['pct']
            ph_peers[i].markdown(card_html(p, f"{val:+.1f}%", f"{val:+.1f}%", val), unsafe_allow_html=True)
            
    # 日内预测计算
    valid_peers = [p for p in peers if quotes[p]['price'] > 0]
    peers_avg = sum(quotes[p]['pct'] for p in valid_peers) / len(valid_peers) if valid_peers else 0
    sector_alpha = peers_avg - btc_chg
    sentiment_adj = (fng_val - 50) * 0.02
    
    pred_high_price, pred_low_price, pred_high_pct, pred_low_pct, btdr_open_pct = 0,0,0,0,0
    
    if btdr['price'] > 0:
        btdr_open_pct = ((btdr['open'] - btdr['prev']) / btdr['prev']) * 100
        MODEL = model_params
        pred_high_pct = (MODEL['high']['intercept'] + (MODEL['high']['beta_open'] * btdr_open_pct) + (MODEL['high']['beta_btc'] * btc_chg) + (MODEL['beta_sector'] * sector_alpha) + sentiment_adj)
        pred_low_pct = (MODEL['low']['intercept'] + (MODEL['low']['beta_open'] * btdr_open_pct) + (MODEL['low']['beta_btc'] * btc_chg) + (MODEL['beta_sector'] * sector_alpha) + sentiment_adj)
        pred_high_price = btdr['prev'] * (1 + pred_high_pct / 100)
        pred_low_price = btdr['prev'] * (1 + pred_low_pct / 100)

    # 渲染 BTDR
    state_map = {"PRE": "dot-pre", "REG": "dot-reg", "POST": "dot-post", "CLOSED": "dot-closed"}
    dot_class = state_map.get(btdr.get('tag', 'CLOSED'), 'dot-closed')
    status_tag = f"<span class='status-dot {dot_class}'></span> <span style='margin-left:2px; font-size:0.7rem;'>{btdr.get('tag', 'CLOSED')}</span>"
    
    ph_btdr_price.markdown(card_html("BTDR 实时", f"${btdr['price']:.2f}", f"{btdr['pct']:+.2f}%", btdr['pct'], status_tag), unsafe_allow_html=True)
    ph_btdr_open.markdown(card_html("计算用开盘", f"${btdr['open']:.2f}", f"{btdr_open_pct:+.2f}%", btdr_open_pct), unsafe_allow_html=True)
    
    # 渲染日内预测框
    h_bg = "#e6fcf5" if btdr['price'] < pred_high_price else "#0ca678"; h_txt = "#087f5b" if btdr['price'] < pred_high_price else "#ffffff"
    l_bg = "#fff5f5" if btdr['price'] > pred_low_price else "#e03131"; l_txt = "#c92a2a" if btdr['price'] > pred_low_price else "#ffffff"

    ph_pred_high.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background-color: {h_bg}; color: {h_txt}; border: 1px solid #c3fae8;">
        <div style="font-size: 0.8rem; opacity: 0.8;">日内阻力 (High)</div><div style="font-size: 1.5rem; font-weight: bold;">${pred_high_price:.2f}</div><div style="font-size: 0.75rem; opacity: 0.9;">预期: {pred_high_pct:+.2f}%</div>
    </div></div>""", unsafe_allow_html=True)
    ph_pred_low.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background-color: {l_bg}; color: {l_txt}; border: 1px solid #ffc9c9;">
        <div style="font-size: 0.8rem; opacity: 0.8;">日内支撑 (Low)</div><div style="font-size: 1.5rem; font-weight: bold;">${pred_low_price:.2f}</div><div style="font-size: 0.75rem; opacity: 0.9;">预期: {pred_low_pct:+.2f}%</div>
    </div></div>""", unsafe_allow_html=True)

    # --- 渲染蒙特卡洛图表 (Real-time MC) ---
    if btdr['price'] > 0:
        vol = mc_params.get('volatility', 0.05)
        drift = mc_params.get('drift', 0.0)
        current = btdr['price']
        
        # 1. 运行模拟 (200条路径)
        days_ahead = 5
        simulations = 200
        # 如果不是交易时间，波动率稍微调低一点点，避免过度夸张
        if btdr.get('tag') == 'CLOSED': vol *= 0.8 
        
        sim_data = []
        
        # 为了 Altair 绘图，我们需要生成置信区间
        # P95, P75, P50, P25, P5
        
        paths = []
        for i in range(simulations):
            path = [current]
            p = current
            for d in range(days_ahead):
                # 几何布朗运动: S_t = S_{t-1} * exp((mu - 0.5*sigma^2) + sigma*Z)
                shock = np.random.normal(0, 1)
                change = (drift - 0.5 * vol**2) + vol * shock
                p = p * np.exp(change)
                path.append(p)
            paths.append(path)
            
        paths = np.array(paths) # shape: (200, 6)
        
        # 计算百分位数
        percentiles = np.percentile(paths, [10, 50, 90], axis=0)
        
        chart_df = []
        for d in range(days_ahead + 1):
            chart_df.append({"Day": d, "Price": percentiles[1][d], "Type": "P50 (中位数)"}) # P50
            chart_df.append({"Day": d, "Price": percentiles[2][d], "Type": "P90 (乐观边界)"}) # P90
            chart_df.append({"Day": d, "Price": percentiles[0][d], "Type": "P10 (悲观边界)"}) # P10
            
        df_chart = pd.DataFrame(chart_df)
        
        # Altair 图表：用 Area 表示范围，用 Line 表示中位数
        
        # 1. 区域图 (P10 - P90) - 也就是80%的置信区间
        base = alt.Chart(df_chart).encode(x=alt.X('Day:O', title='未来交易日'))
        
        area = base.mark_area(opacity=0.3, color='#4dabf7').encode(
            y=alt.Y('Price', title='概率分布'),
            y2='Price_Low',
        ).transform_filter(
            alt.FieldOneOfPredicate(field='Type', oneOf=['P90 (乐观边界)'])
        ).transform_lookup(
            lookup='Day',
            from_=alt.LookupData(df_chart[df_chart['Type'] == 'P10 (悲观边界)'], 'Day', ['Price']),
            as_=['Price_Low']
        )
        
        # 线图
        lines = base.mark_line(point=True).encode(
            y='Price',
            color=alt.Color('Type', scale=alt.Scale(
                domain=['P90 (乐观边界)', 'P50 (中位数)', 'P10 (悲观边界)'],
                range=['#0ca678', '#339af0', '#d6336c']
            ))
        )
        
        chart = (lines).properties(height=220).interactive()
        ph_chart.altair_chart(chart, use_container_width=True)
        
        # 计算 P90 和 P10 的最终目标价
        p90_end = percentiles[2][-1]
        p10_end = percentiles[0][-1]
        p90_pct = ((p90_end - current) / current) * 100
        p10_pct = ((p10_end - current) / current) * 100
        
        ph_mc_bull.markdown(card_html("P90 乐观边界", f"${p90_end:.2f}", f"{p90_pct:+.1f}%", p90_pct), unsafe_allow_html=True)
        ph_mc_bear.markdown(card_html("P10 悲观边界", f"${p10_end:.2f}", f"{p10_pct:+.1f}%", p10_pct), unsafe_allow_html=True)

    ph_footer.caption(f"Update: {now_ny} ET | Volatility: {mc_params.get('volatility',0)*100:.1f}% | Sims: 200")

# --- 7. 数据获取 ---
@st.cache_data(ttl=5)
def get_data_v76():
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
if st.session_state['data_cache']: render_ui(st.session_state['data_cache'])
else: ph_time.info("📡 启动蒙特卡洛引擎...")

new_quotes = get_data_v76()
ai_model, ai_mc, ai_status = run_analytics()

if new_quotes:
    try: fng = int(requests.get("https://api.alternative.me/fng/", timeout=1).json()['data'][0]['value'])
    except: fng = 50
    st.session_state['data_cache'] = {
        'quotes': new_quotes, 
        'fng': fng, 
        'model': ai_model, 
        'mc_params': ai_mc,
        'mc_data': "ready", # 标记位
        'status': ai_status
    }
    render_ui(st.session_state['data_cache'])
