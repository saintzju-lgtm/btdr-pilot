import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime
import pytz # 引入时区库
from streamlit_autorefresh import st_autorefresh

# --- 1. 页面配置 ---
st.set_page_config(page_title="BTDR Pilot Live", layout="centered")

# 自动刷新 (5秒)
count = st_autorefresh(interval=5000, limit=None, key="realtime_counter")

st.markdown("""
    <style>
    .stApp {background-color: #ffffff;}
    h1, h2, h3, div, p, span {color: #212529 !important; font-family: -apple-system, BlinkMacSystemFont, sans-serif !important;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    div[data-testid="stMetric"] {
        background-color: #f8f9fa !important;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
        border-radius: 10px;
    }
    [data-testid="stMetricValue"] {
        font-weight: 700; 
        font-size: 1.4rem !important;
        color: #212529 !important;
    }
    .pred-box {
        padding: 15px; border-radius: 12px; margin-top: 10px; text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    .live-indicator {
        height: 8px; width: 8px; background-color: #00e676;
        border-radius: 50%; display: inline-block; margin-right: 5px;
        box-shadow: 0 0 5px #00e676;
    }
    /* 时间显示样式 */
    .time-bar {
        font-size: 0.8rem; color: #666; text-align: center;
        margin-bottom: 15px; padding: 5px;
        background: #f1f3f5; border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("### ⚡ BTDR 实盘监控")

# --- 2. 黄金参数 ---
MODEL = {
    "high": {"intercept": 4.29, "beta_open": 0.67, "beta_btc": 0.52},
    "low":  {"intercept": -3.22, "beta_open": 0.88, "beta_btc": 0.42},
    "beta_sector": 0.25
}

# --- 3. 时区时间计算函数 ---
def get_current_times():
    # 定义时区
    tz_bj = pytz.timezone('Asia/Shanghai')
    tz_ny = pytz.timezone('America/New_York')
    
    # 获取当前时间
    now_bj = datetime.now(tz_bj).strftime('%H:%M:%S')
    now_ny = datetime.now(tz_ny).strftime('%H:%M:%S')
    
    return now_bj, now_ny

# --- 4. 数据获取 ---
def fetch_yahoo_raw(symbol):
    try:
        t = int(time.time() * 1000)
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&useYfid=true&_={t}"
        
        resp = requests.get(url, headers=headers, timeout=3)
        data = resp.json()
        
        meta = data['chart']['result'][0]['meta']
        current = meta['regularMarketPrice']
        prev_close = meta['chartPreviousClose']
        open_price = meta.get('regularMarketOpen')
        if open_price is None: open_price = current 

        pct = ((current - prev_close) / prev_close) * 100
        
        return {"price": current, "pct": pct, "prev": prev_close, "open": open_price}
    except:
        return {"price": 0, "pct": 0, "prev": 0, "open": 0}

def get_all_data():
    btc_data = fetch_yahoo_raw("BTC-USD")
    try:
        fng = requests.get("https://api.alternative.me/fng/", timeout=1).json()
        fng_val = int(fng['data'][0]['value'])
    except:
        fng_val = 50 

    tickers = ["BTDR", "MARA", "RIOT", "CORZ", "CLSK", "IREN"]
    quotes = {}
    for t in tickers:
        quotes[t] = fetch_yahoo_raw(t)
            
    return btc_data['pct'], fng_val, quotes

# --- 5. 核心计算 ---
btc_chg, fng_val, quotes = get_all_data()

peers = ["MARA", "RIOT", "CORZ", "CLSK", "IREN"]
valid_peers = [p for p in peers if quotes[p]['price'] > 0]
peers_avg = sum(quotes[p]['pct'] for p in valid_peers) / len(valid_peers) if valid_peers else 0

sector_alpha = peers_avg - btc_chg
sentiment_adj = (fng_val - 50) * 0.02

btdr = quotes['BTDR']
pred_high_price, pred_low_price, pred_high_pct, pred_low_pct = 0, 0, 0, 0

if btdr['price'] > 0 and btdr['prev'] > 0:
    btdr_open_pct = ((btdr['open'] - btdr['prev']) / btdr['prev']) * 100
    
    pred_high_pct = (MODEL['high']['intercept'] + (MODEL['high']['beta_open'] * btdr_open_pct) + (MODEL['high']['beta_btc'] * btc_chg) + (MODEL['beta_sector'] * sector_alpha) + sentiment_adj)
    pred_low_pct = (MODEL['low']['intercept'] + (MODEL['low']['beta_open'] * btdr_open_pct) + (MODEL['low']['beta_btc'] * btc_chg) + (MODEL['beta_sector'] * sector_alpha) + sentiment_adj)
    
    pred_high_price = btdr['prev'] * (1 + pred_high_pct / 100)
    pred_low_price = btdr['prev'] * (1 + pred_low_pct / 100)
else:
    btdr_open_pct = 0

# --- 6. 渲染界面 ---

# 获取双时区时间
time_bj, time_ny = get_current_times()

# 顶部时间栏 (新设计)
st.markdown(f"""
<div class='time-bar'>
    <div class='live-indicator'></div>
    北京时间: <b>{time_bj}</b> &nbsp;|&nbsp; 
    美东交易时间: <b>{time_ny}</b>
</div>
""", unsafe_allow_html=True)

# 核心指标
c1, c2 = st.columns(2)
c1.metric("BTC (Yahoo)", f"{btc_chg:+.2f}%")
c2.metric("恐慌指数", f"{fng_val}")

# 板块微缩图
st.markdown("##### ⚒️ 矿股板块 Beta")
cols = st.columns(5)
for i, p in enumerate(peers):
    cols[i].metric(p, f"{quotes[p]['pct']:+.1f}%")

st.markdown("---")

# BTDR 数据
c3, c4 = st.columns(2)
c3.metric("BTDR 现价", f"${btdr['price']:.2f}", f"{btdr['pct']:+.2f}%")
c4.metric("今日开盘", f"${btdr['open']:.2f}", f"{btdr_open_pct:+.2f}%")

# 预测结果
st.markdown("### 🎯 AI 实时预测")
col_h, col_l = st.columns(2)

high_style = "border: 2px solid #00e676;" if btdr['price'] >= pred_high_price else "border: 1px solid #badbcc;"
low_style = "border: 2px solid #ff1744;" if btdr['price'] <= pred_low_price else "border: 1px solid #f5c2c7;"

with col_h:
    st.markdown(f"""
    <div class="pred-box" style="background-color: #d1e7dd; color: #0f5132; {high_style}">
        <div style="font-size: 0.8rem;">阻力位 (High)</div>
        <div style="font-size: 1.5rem; font-weight: bold;">${pred_high_price:.2f}</div>
        <div style="font-size: 0.8rem;">预期: {pred_high_pct:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col_l:
    st.markdown(f"""
    <div class="pred-box" style="background-color: #f8d7da; color: #842029; {low_style}">
        <div style="font-size: 0.8rem;">支撑位 (Low)</div>
        <div style="font-size: 1.5rem; font-weight: bold;">${pred_low_price:.2f}</div>
        <div style="font-size: 0.8rem;">预期: {pred_low_pct:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
