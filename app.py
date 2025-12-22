import streamlit as st
import yfinance as yf
import pandas as pd
import time
from datetime import datetime
import pytz
from streamlit_autorefresh import st_autorefresh

# --- 1. 页面配置 ---
st.set_page_config(page_title="BTDR Pilot v6.9", layout="centered")

# 5秒刷新
st_autorefresh(interval=5000, limit=None, key="realtime_counter")

# CSS: 强制锁定高度，解决抖动
st.markdown("""
    <style>
    .stApp {background-color: #ffffff;}
    h1, h2, h3, div, p, span {color: #212529 !important; font-family: -apple-system, BlinkMacSystemFont, sans-serif !important;}
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    
    /* 锁定 Metric 卡片高度 */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa !important;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
        border-radius: 10px;
        height: 100px; /* 强制固定高度 */
        overflow: hidden;
    }
    
    /* 锁定预测框高度 - 解决下半部分抖动的核心 */
    .pred-box-container {
        height: 120px; /* 强制占位高度 */
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
    }
    
    .pred-box {
        padding: 15px; border-radius: 12px; text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08); 
        width: 100%;
        height: 100%; /* 撑满容器 */
        display: flex; flex-direction: column; justify-content: center;
    }
    
    /* 时间栏 */
    .time-bar {
        font-size: 0.8rem; color: #666; text-align: center;
        margin-bottom: 15px; padding: 6px; background: #f1f3f5; border-radius: 6px;
        border: 1px solid #e9ecef;
        height: 36px; /* 固定高度 */
        display: flex; align-items: center; justify-content: center;
    }
    .pulse-dot {
        height: 8px; width: 8px; background-color: #00e676;
        border-radius: 50%; display: inline-block; margin-right: 6px;
        box-shadow: 0 0 0 0 rgba(0, 230, 118, 0.7);
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 230, 118, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 4px rgba(0, 230, 118, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 230, 118, 0); }
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 状态与占位符 ---
if 'data_cache' not in st.session_state:
    st.session_state['data_cache'] = None

st.markdown("### ⚡ BTDR 领航员 v6.9")

# 预先定义好所有 UI 的“坑位”，确保结构锁死
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

# --- 3. 渲染逻辑 (只更新内容，不动结构) ---
def render_ui(data):
    if not data: return
    quotes = data['quotes']
    fng_val = data['fng']
    btc_chg = quotes['BTC-USD']['pct']
    btdr = quotes['BTDR']
    
    # 时间
    tz_bj = pytz.timezone('Asia/Shanghai')
    tz_ny = pytz.timezone('America/New_York')
    now_bj = datetime.now(tz_bj).strftime('%H:%M:%S')
    now_ny = datetime.now(tz_ny).strftime('%H:%M:%S')
    
    ph_time.markdown(f"<div class='time-bar'><div class='pulse-dot'></div>北京: <b>{now_bj}</b> &nbsp;|&nbsp; 美东: <b>{now_ny}</b></div>", unsafe_allow_html=True)
    
    # 指标
    ph_btc.metric("BTC (全时段)", f"{btc_chg:+.2f}%")
    ph_fng.metric("恐慌指数", f"{fng_val}")
    
    # 板块
    peers = ["MARA", "RIOT", "CORZ", "CLSK", "IREN"]
    for i, p in enumerate(peers):
        if p in quotes: ph_peers[i].metric(p, f"{quotes[p]['pct']:+.1f}%")
            
    # 计算
    valid_peers = [p for p in peers if quotes[p]['price'] > 0]
    peers_avg = sum(quotes[p]['pct'] for p in valid_peers) / len(valid_peers) if valid_peers else 0
    sector_alpha = peers_avg - btc_chg
    sentiment_adj = (fng_val - 50) * 0.02
    
    # 模型
    MODEL = {"high": {"intercept": 4.29, "beta_open": 0.67, "beta_btc": 0.52}, "low":  {"intercept": -3.22, "beta_open": 0.88, "beta_btc": 0.42}, "beta_sector": 0.25}
    pred_high_price, pred_low_price, pred_high_pct, pred_low_pct, btdr_open_pct = 0,0,0,0,0
    
    if btdr['price'] > 0:
        btdr_open_pct = ((btdr['open'] - btdr['prev']) / btdr['prev']) * 100
        pred_high_pct = (MODEL['high']['intercept'] + (MODEL['high']['beta_open'] * btdr_open_pct) + (MODEL['high']['beta_btc'] * btc_chg) + (MODEL['beta_sector'] * sector_alpha) + sentiment_adj)
        pred_low_pct = (MODEL['low']['intercept'] + (MODEL['low']['beta_open'] * btdr_open_pct) + (MODEL['low']['beta_btc'] * btc_chg) + (MODEL['beta_sector'] * sector_alpha) + sentiment_adj)
        pred_high_price = btdr['prev'] * (1 + pred_high_pct / 100)
        pred_low_price = btdr['prev'] * (1 + pred_low_pct / 100)

    # BTDR 本体
    # 使用 container 包裹，高度固定
    ph_btdr_price.markdown(f"""
    <div style="height:100px; background-color:#f8f9fa; border:1px solid #e9ecef; border-radius:10px; padding:10px; display:flex; flex-direction:column; justify-content:center;">
        <div style='font-size:0.9rem; color:#666;'>BTDR 实时价</div>
        <div style='font-size:1.8rem; font-weight:bold; color:#212529;'>${btdr['price']:.2f}</div>
        <div style='color:{'#198754' if btdr['pct']>=0 else '#dc3545'}; font-weight:bold;'>{btdr['pct']:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    ph_btdr_open.metric("计算用开盘", f"${btdr['open']:.2f}", f"{btdr_open_pct:+.2f}%")
    
    # 预测 (样式优化)
    h_bg = "#d1e7dd" if btdr['price'] < pred_high_price else "#198754"; h_txt = "#0f5132" if btdr['price'] < pred_high_price else "#ffffff"
    l_bg = "#f8d7da" if btdr['price'] > pred_low_price else "#dc3545"; l_txt = "#842029" if btdr['price'] > pred_low_price else "#ffffff"

    # 关键：外层加 pred-box-container 锁定高度
    ph_pred_high.markdown(f"""
    <div class="pred-box-container">
        <div class="pred-box" style="background-color: {h_bg}; color: {h_txt}; border: 1px solid #badbcc;">
            <div style="font-size: 0.9rem;">阻力位 (High)</div>
            <div style="font-size: 1.6rem; font-weight: bold;">${pred_high_price:.2f}</div>
            <div style="font-size: 0.8rem;">预期: {pred_high_pct:+.2f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    ph_pred_low.markdown(f"""
    <div class="pred-box-container">
        <div class="pred-box" style="background-color: {l_bg}; color: {l_txt}; border: 1px solid #f5c2c7;">
            <div style="font-size: 0.9rem;">支撑位 (Low)</div>
            <div style="font-size: 1.6rem; font-weight: bold;">${pred_low_price:.2f}</div>
            <div style="font-size: 0.8rem;">预期: {pred_low_pct:+.2f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    ph_footer.caption(f"数据源: yfinance (Batch) | 模式: 磐石稳定 | 美东时间: {now_ny}")

# --- 4. 数据核心 (修复基准日问题) ---
@st.cache_data(ttl=5)
def get_data_v69():
    tickers_list = "BTC-USD BTDR MARA RIOT CORZ CLSK IREN"
    try:
        # 1. 获取日线 (不包含盘前盘后，只看正规收盘日)
        # 用 5d 是为了确保拿到最近的一个已完结交易日
        daily = yf.download(tickers_list, period="5d", interval="1d", group_by='ticker', threads=True, progress=False)
        
        # 2. 获取实时分钟线 (包含盘前盘后)
        live = yf.download(tickers_list, period="1d", interval="1m", prepost=True, group_by='ticker', threads=True, progress=False)
        
        quotes = {}
        symbols = tickers_list.split()
        
        # 获取美东当前日期，用于判断"日线最后一行"是不是"今天"
        today_ny = datetime.now(pytz.timezone('America/New_York')).date()
        
        for sym in symbols:
            try:
                df_day = daily[sym].dropna(subset=['Close'])
                df_min = live[sym].dropna(subset=['Close'])
                
                # --- A. 确定昨收 (Prev Close) - 核心修复 ---
                if not df_day.empty:
                    last_date = df_day.index[-1].date()
                    
                    if last_date == today_ny:
                        # 如果日线最后一行是"今天" (yfinance在开盘后会生成今天的临时日线)
                        # 那么"昨收"应该是倒数第二行
                        if len(df_day) >= 2:
                            prev_close = df_day['Close'].iloc[-2]
                        else:
                            prev_close = df_day['Close'].iloc[-1] # 数据不足
                    else:
                        # 如果日线最后一行是"昨天" (还没开盘，或者yfinance还没推今天的日线)
                        # 那么"昨收"就是最后一行
                        prev_close = df_day['Close'].iloc[-1]
                else:
                    prev_close = 1.0 # 兜底

                # --- B. 确定实时价 ---
                if not df_min.empty:
                    current_price = df_min['Close'].iloc[-1]
                else:
                    # 分钟线没数据，回退到日线
                    current_price = df_day['Close'].iloc[-1] if not df_day.empty else 0
                
                # --- C. 确定开盘价 ---
                # 只有今日已开盘，日线才有今日Open。否则用实时价模拟
                if not df_day.empty and df_day.index[-1].date() == today_ny:
                     open_price = df_day['Open'].iloc[-1]
                else:
                     open_price = current_price

                # 计算涨跌
                pct = ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0
                
                quotes[sym] = {"price": current_price, "pct": pct, "prev": prev_close, "open": open_price}
            except:
                quotes[sym] = {"price": 0, "pct": 0, "prev": 0, "open": 0}
        return quotes
    except:
        return None
        
def get_fng():
    try: return int(requests.get("https://api.alternative.me/fng/", timeout=1).json()['data'][0]['value'])
    except: return 50

# --- 5. 执行 ---

# 优先渲染缓存
if st.session_state['data_cache']: render_ui(st.session_state['data_cache'])
else: ph_time.info("📡 正在校准基准数据...")

# 更新数据
new_quotes = get_data_v69()
new_fng = get_fng()

if new_quotes:
    new_data = {'quotes': new_quotes, 'fng': new_fng}
    st.session_state['data_cache'] = new_data
    render_ui(new_data)
