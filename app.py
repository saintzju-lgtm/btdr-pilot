import streamlit as st
import yfinance as yf
import pandas as pd
import time
from datetime import datetime, timedelta
import pytz
from streamlit_autorefresh import st_autorefresh

# --- 1. 页面配置 ---
st.set_page_config(page_title="BTDR Pilot v7.1", layout="centered")

# CSS: 视觉冻结 + 移动端适配优化
st.markdown("""
    <style>
    html { overflow-y: scroll; }
    .stApp > header { display: none; }
    .stApp { margin-top: -30px; }
    div[data-testid="stStatusWidget"] { visibility: hidden; }
    
    .stApp { background-color: #ffffff; }
    h1, h2, h3, div, p, span { 
        color: #212529 !important; 
        font-family: -apple-system, BlinkMacSystemFont, sans-serif !important; 
    }
    
    /* 核心指标卡片 */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa !important;
        border: 1px solid #e9ecef;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        border-radius: 8px;
        height: 85px; /* 稍微调小一点，适配小屏 */
        display: flex; flex-direction: column; justify-content: center;
        overflow: hidden;
    }
    
    /* BTDR 价格大卡片 */
    .btdr-box {
        height: 90px;
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 0 15px;
        display: flex; flex-direction: column; justify-content: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }

    /* 预测框容器 */
    .pred-container-wrapper {
        height: 110px; width: 100%; display: block;
    }
    .pred-box {
        padding: 8px; border-radius: 10px; text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08); 
        height: 100%; display: flex; flex-direction: column; justify-content: center;
    }
    
    /* 时间栏 */
    .time-bar {
        font-size: 0.75rem; color: #888; text-align: center;
        margin-bottom: 10px; padding: 4px; background: #f8f9fa; border-radius: 4px;
        border: 1px solid #eee;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 侧边栏：控制中心 (新增功能) ---
with st.sidebar:
    st.header("🎛️ 模型参数微调")
    
    # 自动刷新开关
    refresh_rate = st.slider("刷新频率 (秒)", 5, 60, 10)
    st_autorefresh(interval=refresh_rate * 1000, limit=None, key="realtime_counter")
    
    st.markdown("---")
    st.caption("AI 权重参数 (Golden Params)")
    
    # 允许你在网页上调整 Beta 值
    beta_open_h = st.number_input("High Beta (Open)", value=0.67, step=0.01)
    beta_btc_h  = st.number_input("High Beta (BTC)", value=0.52, step=0.01)
    intercept_h = st.number_input("High Intercept", value=4.29, step=0.1)
    
    st.markdown("---")
    beta_open_l = st.number_input("Low Beta (Open)", value=0.88, step=0.01)
    beta_btc_l  = st.number_input("Low Beta (BTC)", value=0.42, step=0.01)
    intercept_l = st.number_input("Low Intercept", value=-3.22, step=0.1)
    
    # 动态构建模型
    MODEL = {
        "high": {"intercept": intercept_h, "beta_open": beta_open_h, "beta_btc": beta_btc_h},
        "low":  {"intercept": intercept_l, "beta_open": beta_open_l, "beta_btc": beta_btc_l},
        "beta_sector": 0.25
    }

# --- 3. 状态与骨架 ---
if 'data_cache' not in st.session_state:
    st.session_state['data_cache'] = None

st.markdown("### ⚡ BTDR 领航员 v7.1")

# UI 骨架
ph_time = st.empty()
c1, c2 = st.columns(2)
with c1: ph_btc = st.empty()
with c2: ph_fng = st.empty()

st.markdown("##### ⚒️ 矿股板块 Beta")
cols = st.columns(5)
ph_peers = [col.empty() for col in cols]

st.markdown("---")
c3, c4 = st.columns(2)
with c3: ph_btdr_price = st.empty()
with c4: ph_btdr_open = st.empty()

st.markdown("### 🎯 AI 全时段预测")
col_h, col_l = st.columns(2)
with col_h: ph_pred_high = st.empty()
with col_l: ph_pred_low = st.empty()

st.markdown("---")
ph_footer = st.empty()

# --- 4. 渲染函数 ---
def render_ui(data):
    if not data: return
    quotes = data['quotes']
    fng_val = data['fng']
    data_ts = data.get('ts', '') # 数据生成时间
    
    btc_chg = quotes['BTC-USD']['pct']
    btdr = quotes['BTDR']
    
    # 时间显示
    tz_bj = pytz.timezone('Asia/Shanghai')
    tz_ny = pytz.timezone('America/New_York')
    now_bj = datetime.now(tz_bj).strftime('%H:%M:%S')
    now_ny = datetime.now(tz_ny).strftime('%H:%M:%S')
    
    ph_time.markdown(f"<div class='time-bar'>北京: <b>{now_bj}</b> | 美东: <b>{now_ny}</b> | 数据: {data_ts}</div>", unsafe_allow_html=True)
    
    # 指标
    ph_btc.metric("BTC (全时段)", f"{btc_chg:+.2f}%")
    ph_fng.metric("恐慌指数", f"{fng_val}")
    
    # 板块
    peers = ["MARA", "RIOT", "CORZ", "CLSK", "IREN"]
    for i, p in enumerate(peers):
        if p in quotes: ph_peers[i].metric(p, f"{quotes[p]['pct']:+.1f}%")
            
    # 预测计算 (使用 Sidebar 里的动态 MODEL)
    valid_peers = [p for p in peers if quotes[p]['price'] > 0]
    peers_avg = sum(quotes[p]['pct'] for p in valid_peers) / len(valid_peers) if valid_peers else 0
    sector_alpha = peers_avg - btc_chg
    sentiment_adj = (fng_val - 50) * 0.02
    
    pred_high_price, pred_low_price, pred_high_pct, pred_low_pct = 0,0,0,0
    btdr_open_pct = 0
    
    if btdr['price'] > 0:
        btdr_open_pct = ((btdr['open'] - btdr['prev']) / btdr['prev']) * 100
        
        # 动态调用 MODEL
        pred_high_pct = (MODEL['high']['intercept'] + (MODEL['high']['beta_open'] * btdr_open_pct) + (MODEL['high']['beta_btc'] * btc_chg) + (MODEL['beta_sector'] * sector_alpha) + sentiment_adj)
        pred_low_pct = (MODEL['low']['intercept'] + (MODEL['low']['beta_open'] * btdr_open_pct) + (MODEL['low']['beta_btc'] * btc_chg) + (MODEL['beta_sector'] * sector_alpha) + sentiment_adj)
        
        pred_high_price = btdr['prev'] * (1 + pred_high_pct / 100)
        pred_low_price = btdr['prev'] * (1 + pred_low_pct / 100)

    # BTDR 渲染
    ph_btdr_price.markdown(f"""
    <div class="btdr-box">
        <div style='font-size:0.8rem; color:#666;'>BTDR 实时价</div>
        <div style='font-size:1.7rem; font-weight:bold; color:#212529;'>${btdr['price']:.2f}</div>
        <div style='font-size:0.9rem; color:{'#198754' if btdr['pct']>=0 else '#dc3545'}; font-weight:bold;'>{btdr['pct']:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    ph_btdr_open.metric("计算用开盘", f"${btdr['open']:.2f}", f"{btdr_open_pct:+.2f}%")
    
    # 预测框
    h_bg = "#d1e7dd" if btdr['price'] < pred_high_price else "#198754"; h_txt = "#0f5132" if btdr['price'] < pred_high_price else "#ffffff"
    l_bg = "#f8d7da" if btdr['price'] > pred_low_price else "#dc3545"; l_txt = "#842029" if btdr['price'] > pred_low_price else "#ffffff"

    ph_pred_high.markdown(f"""
    <div class="pred-container-wrapper">
        <div class="pred-box" style="background-color: {h_bg}; color: {h_txt}; border: 1px solid #badbcc;">
            <div style="font-size: 0.9rem;">阻力位 (High)</div>
            <div style="font-size: 1.5rem; font-weight: bold;">${pred_high_price:.2f}</div>
            <div style="font-size: 0.8rem;">预期: {pred_high_pct:+.2f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    ph_pred_low.markdown(f"""
    <div class="pred-container-wrapper">
        <div class="pred-box" style="background-color: {l_bg}; color: {l_txt}; border: 1px solid #f5c2c7;">
            <div style="font-size: 0.9rem;">支撑位 (Low)</div>
            <div style="font-size: 1.5rem; font-weight: bold;">${pred_low_price:.2f}</div>
            <div style="font-size: 0.8rem;">预期: {pred_low_pct:+.2f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    ph_footer.caption(f"Configurable AI Model | Source: yfinance | Update: {data_ts}")

# --- 5. 数据核心 ---
@st.cache_data(ttl=5)
def get_data_v71():
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
                
                # 价格逻辑
                if not df_min.empty: current_price = df_min['Close'].iloc[-1]
                elif not df_day.empty: current_price = df_day['Close'].iloc[-1]
                else: current_price = 0
                
                # 昨收逻辑
                prev_close = 1.0
                if not df_day.empty:
                    last_day_row = df_day.index[-1].date()
                    # 关键逻辑：如果最后一行是今天，则昨收是倒数第二行
                    if last_day_row == today_ny:
                        if len(df_day) >= 2: prev_close = df_day['Close'].iloc[-2]
                        elif not df_day.empty: prev_close = df_day['Open'].iloc[-1] # 新股或极端情况
                    else:
                        prev_close = df_day['Close'].iloc[-1]
                
                pct = ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0
                
                # 开盘逻辑
                if not df_day.empty and df_day.index[-1].date() == today_ny:
                     open_price = df_day['Open'].iloc[-1]
                else:
                     open_price = current_price

                quotes[sym] = {"price": current_price, "pct": pct, "prev": prev_close, "open": open_price}
            except:
                quotes[sym] = {"price": 0, "pct": 0, "prev": 0, "open": 0}
        return quotes
    except:
        return None

# --- 6. 执行 ---
if st.session_state['data_cache']: render_ui(st.session_state['data_cache'])

new_quotes = get_data_v71()
if new_quotes:
    # 记录数据生成的时间，方便确认数据是否滞后
    ts = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%H:%M:%S')
    
    # 简单的 fng 获取，不阻塞
    try: fng = int(requests.get("https://api.alternative.me/fng/", timeout=1).json()['data'][0]['value'])
    except: fng = 50
    
    new_data = {'quotes': new_quotes, 'fng': fng, 'ts': ts}
    st.session_state['data_cache'] = new_data
    render_ui(new_data)
