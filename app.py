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
st.set_page_config(page_title="BTDR Pilot v7.5", layout="centered")

# 5秒刷新
st_autorefresh(interval=5000, limit=None, key="realtime_counter")

# CSS: 保持 v7.4 的完美 UI
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

# --- 2. 辅助函数 ---
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

# --- 3. 状态管理 ---
if 'data_cache' not in st.session_state: st.session_state['data_cache'] = None

st.markdown("### ⚡ BTDR 领航员 v7.5")

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

# 【新增】趋势推演
st.markdown("### 📈 未来5日趋势推演 (Trend)")
ph_chart = st.empty() # 图表占位符
col_t1, col_t2 = st.columns(2)
with col_t1: ph_trend_bull = st.empty()
with col_t2: ph_trend_bear = st.empty()

st.markdown("---")
ph_footer = st.empty()

# --- 5. 核心逻辑：AI 调参 + 趋势计算 ---
@st.cache_data(ttl=3600)
def auto_tune_and_trend():
    # 默认参数
    default_model = {
        "high": {"intercept": 4.29, "beta_open": 0.67, "beta_btc": 0.52},
        "low":  {"intercept": -3.22, "beta_open": 0.88, "beta_btc": 0.42},
        "beta_sector": 0.25
    }
    trend_factors = {"volatility": 0.05, "drift": 0.0} # 默认波动率5%

    try:
        # 获取30天数据
        df = yf.download("BTDR", period="1mo", interval="1d", progress=False)
        if len(df) < 10: return default_model, trend_factors, "数据不足"
        
        if isinstance(df.columns, pd.MultiIndex): df = df.xs('BTDR', axis=1, level=1)
        df = df.dropna()
        
        # 1. 计算日内模型参数
        df_reg = df.copy()
        df_reg['PrevClose'] = df_reg['Close'].shift(1)
        df_reg = df_reg.dropna()
        x = ((df_reg['Open'] - df_reg['PrevClose']) / df_reg['PrevClose'] * 100).values
        y_high = ((df_reg['High'] - df_reg['PrevClose']) / df_reg['PrevClose'] * 100).values
        y_low = ((df_reg['Low'] - df_reg['PrevClose']) / df_reg['PrevClose'] * 100).values
        
        cov_h = np.cov(x, y_high); beta_h = cov_h[0, 1] / cov_h[0, 0] if cov_h[0, 0] != 0 else 0.67
        cov_l = np.cov(x, y_low); beta_l = cov_l[0, 1] / cov_l[0, 0] if cov_l[0, 0] != 0 else 0.88
        
        # 2. 计算趋势因子 (波动率 & 动量)
        # 日收益率
        returns = df['Close'].pct_change().dropna()
        # 历史波动率 (每日标准差)
        volatility = returns.std()
        # 简单动量 (过去10天平均收益)
        drift = returns.tail(10).mean()
        
        trend_factors = {
            "volatility": volatility, # 每日波动幅度
            "drift": drift            # 每日趋势倾向
        }

        final_model = {
            "high": {"intercept": 0.7*4.29 + 0.3*(np.mean(y_high)-beta_h*np.mean(x)), "beta_open": 0.7*0.67 + 0.3*np.clip(beta_h,0.3,1.2), "beta_btc": 0.52},
            "low": {"intercept": 0.7*-3.22 + 0.3*(np.mean(y_low)-beta_l*np.mean(x)), "beta_open": 0.7*0.88 + 0.3*np.clip(beta_l,0.4,1.5), "beta_btc": 0.42},
            "beta_sector": 0.25
        }
        return final_model, trend_factors, "AI在线"
    except: 
        return default_model, trend_factors, "离线模式"

# --- 6. 渲染函数 ---
def render_ui(data):
    if not data: return
    quotes = data['quotes']
    fng_val = data['fng']
    model_params = data['model']
    trend_factors = data['trend']
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
            
    # 计算日内预测
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

    ph_pred_high.markdown(f"""
    <div class="pred-container-wrapper"><div class="pred-box" style="background-color: {h_bg}; color: {h_txt}; border: 1px solid #c3fae8;">
        <div style="font-size: 0.8rem; opacity: 0.8;">阻力位 (High)</div>
        <div style="font-size: 1.5rem; font-weight: bold;">${pred_high_price:.2f}</div>
        <div style="font-size: 0.75rem; opacity: 0.9;">预期: {pred_high_pct:+.2f}%</div>
    </div></div>""", unsafe_allow_html=True)
    
    ph_pred_low.markdown(f"""
    <div class="pred-container-wrapper"><div class="pred-box" style="background-color: {l_bg}; color: {l_txt}; border: 1px solid #ffc9c9;">
        <div style="font-size: 0.8rem; opacity: 0.8;">支撑位 (Low)</div>
        <div style="font-size: 1.5rem; font-weight: bold;">${pred_low_price:.2f}</div>
        <div style="font-size: 0.75rem; opacity: 0.9;">预期: {pred_low_pct:+.2f}%</div>
    </div></div>""", unsafe_allow_html=True)

    # --- 渲染趋势图 (New) ---
    if btdr['price'] > 0:
        vol = trend_factors.get('volatility', 0.05)
        # 强制放大一点波动率，为了更明显的视觉提示 (Risk Premium)
        vol = max(vol, 0.04)
        
        current = btdr['price']
        days = 5
        
        # 生成未来5天的数据点
        future_data = []
        # Day 0 是现在
        future_data.append({"Day": 0, "Price": current, "Type": "Base"})
        future_data.append({"Day": 0, "Price": current, "Type": "Bull"})
        future_data.append({"Day": 0, "Price": current, "Type": "Bear"})
        
        for d in range(1, days + 1):
            # Bull: 每天涨 1倍标准差
            bull_p = current * (1 + vol * d)
            # Bear: 每天跌 1倍标准差
            bear_p = current * (1 - vol * d)
            # Base: 保持现状 (或微弱动量)
            base_p = current # 简化为横盘，避免误导
            
            future_data.append({"Day": d, "Price": base_p, "Type": "Base"})
            future_data.append({"Day": d, "Price": bull_p, "Type": "Bull"})
            future_data.append({"Day": d, "Price": bear_p, "Type": "Bear"})
            
        df_chart = pd.DataFrame(future_data)
        
        # 绘图
        chart = alt.Chart(df_chart).mark_line(point=True).encode(
            x=alt.X('Day:O', title='未来交易日'),
            y=alt.Y('Price', title='价格预演', scale=alt.Scale(zero=False)),
            color=alt.Color('Type', scale=alt.Scale(domain=['Bull', 'Base', 'Bear'], range=['#0ca678', '#adb5bd', '#d6336c'])),
            tooltip=['Day', 'Price', 'Type']
        ).properties(height=200).interactive()
        
        ph_chart.altair_chart(chart, use_container_width=True)
        
        # 渲染下方的价格目标卡片
        target_bull = current * (1 + vol * 5)
        target_bear = current * (1 - vol * 5)
        bull_pct = vol * 5 * 100
        bear_pct = -vol * 5 * 100
        
        ph_trend_bull.markdown(card_html("5日目标 (Bull)", f"${target_bull:.2f}", f"+{bull_pct:.1f}%", bull_pct), unsafe_allow_html=True)
        ph_trend_bear.markdown(card_html("5日目标 (Bear)", f"${target_bear:.2f}", f"{bear_pct:.1f}%", bear_pct), unsafe_allow_html=True)

    ph_footer.caption(f"Update: {now_ny} ET | Trend Volatility: {trend_factors.get('volatility',0)*100:.1f}%")

# --- 7. 数据获取 ---
@st.cache_data(ttl=5)
def get_data_v75():
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
else: ph_time.info("📡 正在计算趋势模型...")

new_quotes = get_data_v75()
ai_model, ai_trend, ai_status = auto_tune_and_trend()

if new_quotes:
    try: fng = int(requests.get("https://api.alternative.me/fng/", timeout=1).json()['data'][0]['value'])
    except: fng = 50
    st.session_state['data_cache'] = {
        'quotes': new_quotes, 
        'fng': fng, 
        'model': ai_model, 
        'trend': ai_trend,
        'status': ai_status
    }
    render_ui(st.session_state['data_cache'])
