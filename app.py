import streamlit as st
import yfinance as yf
import pandas as pd
import time
from datetime import datetime, timedelta
import pytz
from streamlit_autorefresh import st_autorefresh

# --- 1. 页面配置 ---
st.set_page_config(page_title="BTDR Pilot v7.0", layout="centered")

# 5秒刷新 (放在侧边栏隐藏执行，减少主页面干扰)
with st.sidebar:
    st_autorefresh(interval=5000, limit=None, key="realtime_counter")

# --- 2. CSS 视觉冻结技术 (关键) ---
st.markdown("""
    <style>
    /* 1. 强制显示垂直滚动条，防止因滚动条消失/出现导致的页面左右抖动 */
    html { overflow-y: scroll; }
    
    /* 2. 隐藏 Streamlit 顶部的彩色加载条和右上角的汉堡菜单/运行状态，减少视觉干扰 */
    .stApp > header { display: none; }
    .stApp { margin-top: -50px; } /* 把内容顶上去，填补 header 的空缺 */
    div[data-testid="stStatusWidget"] { visibility: hidden; }
    
    /* 3. 全局字体与颜色 */
    .stApp { background-color: #ffffff; }
    h1, h2, h3, div, p, span { 
        color: #212529 !important; 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important; 
    }
    
    /* 4. 锁定 Metric 卡片高度与布局 */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa !important;
        border: 1px solid #e9ecef;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border-radius: 8px;
        height: 90px; /* 绝对高度 */
        display: flex; 
        flex-direction: column; 
        justify-content: center;
        overflow: hidden;
    }
    
    /* 5. 锁定 BTDR 价格框高度 */
    .btdr-box {
        height: 90px;
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 0 15px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    /* 6. 锁定预测容器高度 (核心防抖) */
    .pred-container-wrapper {
        height: 120px; /* 占位高度 */
        width: 100%;
        display: block;
    }
    
    .pred-box {
        padding: 10px; 
        border-radius: 10px; 
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05); 
        height: 100%;
        display: flex; 
        flex-direction: column; 
        justify-content: center;
    }
    
    /* 时间栏 */
    .time-bar {
        font-size: 0.8rem; color: #666; text-align: center;
        margin-bottom: 15px; padding: 4px; background: #f1f3f5; border-radius: 4px;
        border: 1px solid #e9ecef;
        height: 30px;
        line-height: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. 状态与结构初始化 ---
if 'data_cache' not in st.session_state:
    st.session_state['data_cache'] = None

st.markdown("### ⚡ BTDR 领航员 v7.0")

# 预先渲染空容器 (占位符)，确立页面骨架
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

# --- 4. 渲染函数 (Render UI) ---
def render_ui(data):
    if not data: return
    quotes = data['quotes']
    fng_val = data['fng']
    
    # 提取数据
    btc_chg = quotes['BTC-USD']['pct']
    btdr = quotes['BTDR']
    
    # 时间
    tz_bj = pytz.timezone('Asia/Shanghai')
    tz_ny = pytz.timezone('America/New_York')
    now_bj = datetime.now(tz_bj).strftime('%H:%M:%S')
    now_ny = datetime.now(tz_ny).strftime('%H:%M:%S')
    
    # 渲染时间
    ph_time.markdown(f"<div class='time-bar'>北京: <b>{now_bj}</b> &nbsp;|&nbsp; 美东: <b>{now_ny}</b></div>", unsafe_allow_html=True)
    
    # 渲染指标
    ph_btc.metric("BTC (全时段)", f"{btc_chg:+.2f}%")
    ph_fng.metric("恐慌指数", f"{fng_val}")
    
    # 渲染板块
    peers = ["MARA", "RIOT", "CORZ", "CLSK", "IREN"]
    for i, p in enumerate(peers):
        if p in quotes: ph_peers[i].metric(p, f"{quotes[p]['pct']:+.1f}%")
            
    # 计算预测
    valid_peers = [p for p in peers if quotes[p]['price'] > 0]
    peers_avg = sum(quotes[p]['pct'] for p in valid_peers) / len(valid_peers) if valid_peers else 0
    sector_alpha = peers_avg - btc_chg
    sentiment_adj = (fng_val - 50) * 0.02
    
    MODEL = {"high": {"intercept": 4.29, "beta_open": 0.67, "beta_btc": 0.52}, "low":  {"intercept": -3.22, "beta_open": 0.88, "beta_btc": 0.42}, "beta_sector": 0.25}
    
    pred_high_price, pred_low_price, pred_high_pct, pred_low_pct, btdr_open_pct = 0,0,0,0,0
    
    if btdr['price'] > 0:
        btdr_open_pct = ((btdr['open'] - btdr['prev']) / btdr['prev']) * 100
        pred_high_pct = (MODEL['high']['intercept'] + (MODEL['high']['beta_open'] * btdr_open_pct) + (MODEL['high']['beta_btc'] * btc_chg) + (MODEL['beta_sector'] * sector_alpha) + sentiment_adj)
        pred_low_pct = (MODEL['low']['intercept'] + (MODEL['low']['beta_open'] * btdr_open_pct) + (MODEL['low']['beta_btc'] * btc_chg) + (MODEL['beta_sector'] * sector_alpha) + sentiment_adj)
        pred_high_price = btdr['prev'] * (1 + pred_high_pct / 100)
        pred_low_price = btdr['prev'] * (1 + pred_low_pct / 100)

    # 渲染 BTDR
    ph_btdr_price.markdown(f"""
    <div class="btdr-box">
        <div style='font-size:0.8rem; color:#666;'>BTDR 实时价</div>
        <div style='font-size:1.6rem; font-weight:bold; color:#212529; line-height:1.2;'>${btdr['price']:.2f}</div>
        <div style='font-size:0.9rem; color:{'#198754' if btdr['pct']>=0 else '#dc3545'}; font-weight:bold;'>{btdr['pct']:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    ph_btdr_open.metric("计算用开盘", f"${btdr['open']:.2f}", f"{btdr_open_pct:+.2f}%")
    
    # 渲染预测 (外层加 wrapper 固定高度)
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
    
    ph_footer.caption(f"数据源: yfinance | 模式: 视觉冻结 | 美东时间: {now_ny}")

# --- 5. 数据核心 (逻辑修复版) ---
@st.cache_data(ttl=5)
def get_data_v70():
    tickers_list = "BTC-USD BTDR MARA RIOT CORZ CLSK IREN"
    try:
        # 获取日线 (用于昨收) 和 分钟线 (用于实时)
        daily = yf.download(tickers_list, period="5d", interval="1d", group_by='ticker', threads=True, progress=False)
        live = yf.download(tickers_list, period="1d", interval="1m", prepost=True, group_by='ticker', threads=True, progress=False)
        
        quotes = {}
        symbols = tickers_list.split()
        
        # 获取美东当前日期
        today_ny = datetime.now(pytz.timezone('America/New_York')).date()
        
        for sym in symbols:
            try:
                # 安全提取 DataFrame
                df_day = daily[sym] if sym in daily else pd.DataFrame()
                if not df_day.empty: df_day = df_day.dropna(subset=['Close'])
                
                df_min = live[sym] if sym in live else pd.DataFrame()
                if not df_min.empty: df_min = df_min.dropna(subset=['Close'])
                
                # --- A. 实时价 ---
                if not df_min.empty:
                    current_price = df_min['Close'].iloc[-1]
                elif not df_day.empty:
                    current_price = df_day['Close'].iloc[-1]
                else:
                    current_price = 0
                
                # --- B. 昨收价 (核心修正) ---
                # 逻辑：昨收价必须是“上一个”完整K线的收盘价。
                # 如果日线最后一行是“今天”(日期=今天)，那么它是实时变动的，不是昨收，所以昨收是倒数第二行。
                # 如果日线最后一行是“昨天”(日期<今天)，那么它就是昨收。
                
                prev_close = 1.0 # 默认
                if not df_day.empty:
                    last_day_row = df_day.index[-1].date()
                    if last_day_row == today_ny:
                        # 列表里包含"今天"，所以昨收是倒数第二个
                        if len(df_day) >= 2:
                            prev_close = df_day['Close'].iloc[-2]
                        else:
                            # 极其罕见：新股上市第一天，没有昨收
                            prev_close = df_day['Open'].iloc[-1] 
                    else:
                        # 列表里只有"昨天"及以前，昨收就是最后一个
                        prev_close = df_day['Close'].iloc[-1]
                
                # --- C. 计算涨跌 ---
                pct = ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0
                
                # --- D. 开盘价 ---
                # 只有今日日线存在，且日期是今天，才用日线Open，否则用实时价模拟
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

def get_fng():
    try: return int(requests.get("https://api.alternative.me/fng/", timeout=1).json()['data'][0]['value'])
    except: return 50

# --- 6. 执行流 ---

# 先渲染缓存 (防白屏)
if st.session_state['data_cache']: 
    render_ui(st.session_state['data_cache'])
else:
    ph_time.info("📡 正在同步基准数据...")

# 抓取新数据
new_quotes = get_data_v70()
new_fng = get_fng()

if new_quotes:
    new_data = {'quotes': new_quotes, 'fng': new_fng}
    st.session_state['data_cache'] = new_data
    # 再次渲染 (更新数值)
    render_ui(new_data)
