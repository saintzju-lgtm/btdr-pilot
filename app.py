import streamlit as st
import requests
import pandas as pd
import time

# --- 1. 页面配置 ---
st.set_page_config(page_title="BTDR Pilot v5.3", layout="centered")

st.markdown("""
    <style>
    .stApp {background-color: #f8f9fa;}
    h1, h2, h3, div, p, span {color: #212529 !important;}
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e9ecef;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricValue"] {font-weight: 700; color: #212529 !important;}
    [data-testid="stMetricLabel"] {color: #6c757d !important;}
    .pred-box {
        padding: 15px; border-radius: 8px; margin-top: 10px; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("⚡ BTDR 领航员 v5.3 (原生接口版)")

# --- 2. 黄金参数 (保持不变) ---
MODEL = {
    "high": {"intercept": 4.29, "beta_open": 0.67, "beta_btc": 0.52},
    "low":  {"intercept": -3.22, "beta_open": 0.88, "beta_btc": 0.42},
    "beta_sector": 0.25
}

# --- 3. 核心数据获取 (完全模拟 JS 插件) ---
# 抛弃 yfinance 库，直接请求 Yahoo 原生 API，确保数据源 100% 一致

def fetch_yahoo_raw(symbol):
    """
    完全复刻插件中的 fetchQuote 函数逻辑
    """
    try:
        t = int(time.time() * 1000)
        # 必须加 User-Agent，否则 Yahoo API 会拒绝 Python 请求
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&useYfid=true&_={t}"
        
        resp = requests.get(url, headers=headers, timeout=5)
        data = resp.json()
        
        meta = data['chart']['result'][0]['meta']
        
        current = meta['regularMarketPrice']
        prev_close = meta['chartPreviousClose']
        # 优先用 regularMarketOpen，如果为0或空(盘前)，则回退到 current (和插件逻辑一致)
        open_price = meta.get('regularMarketOpen', current)
        if open_price is None: open_price = current

        pct = ((current - prev_close) / prev_close) * 100
        
        return {
            "price": current,
            "pct": pct,
            "prev": prev_close,
            "open": open_price
        }
    except Exception as e:
        # print(f"Error fetching {symbol}: {e}")
        return {"price": 0, "pct": 0, "prev": 0, "open": 0}

@st.cache_data(ttl=10) # 10秒刷新，保证即时性
def get_all_data():
    # 1. 获取 BTC (使用 Yahoo 原生接口替代 Binance，以绕过封锁并保持计算逻辑一致)
    # 注意：Yahoo 的涨跌是"日内涨跌"，Binance 是"24h滚动"。
    # 为了完全一致，建议插件端也改用 fetchQuote('BTC-USD')
    btc_data = fetch_yahoo_raw("BTC-USD")
    btc_chg = btc_data['pct']

    # 2. 获取情绪
    try:
        fng = requests.get("https://api.alternative.me/fng/", timeout=3).json()
        fng_val = int(fng['data'][0]['value'])
    except:
        fng_val = 50

    # 3. 获取所有股票 (串行或并发均可，Python requests 是同步的，这里直接循环)
    tickers = ["BTDR", "MARA", "RIOT", "CORZ", "CLSK", "IREN"]
    quotes = {}
    for t in tickers:
        quotes[t] = fetch_yahoo_raw(t)
            
    return btc_chg, fng_val, quotes

# --- 4. 主逻辑 (完全一致的数学公式) ---

with st.spinner('正在通过原生接口同步...'):
    btc_chg, fng_val, quotes = get_all_data()

# 板块 Alpha 计算
peers = ["MARA", "RIOT", "CORZ", "CLSK", "IREN"]
valid_peers = [p for p in peers if quotes[p]['price'] > 0]
if valid_peers:
    peers_avg = sum(quotes[p]['pct'] for p in valid_peers) / len(valid_peers)
else:
    peers_avg = 0

sector_alpha = peers_avg - btc_chg
sentiment_adj = (fng_val - 50) * 0.02

# BTDR 计算
btdr = quotes['BTDR']
if btdr['price'] > 0 and btdr['prev'] > 0:
    # 核心：Open Pct 计算
    btdr_open_pct = ((btdr['open'] - btdr['prev']) / btdr['prev']) * 100
    
    # 公式计算
    pred_high_pct = (MODEL['high']['intercept'] 
                     + (MODEL['high']['beta_open'] * btdr_open_pct) 
                     + (MODEL['high']['beta_btc'] * btc_chg) 
                     + (MODEL['beta_sector'] * sector_alpha) 
                     + sentiment_adj)
    
    pred_low_pct = (MODEL['low']['intercept'] 
                    + (MODEL['low']['beta_open'] * btdr_open_pct) 
                    + (MODEL['low']['beta_btc'] * btc_chg) 
                    + (MODEL['beta_sector'] * sector_alpha) 
                    + sentiment_adj)
    
    pred_high_price = btdr['prev'] * (1 + pred_high_pct / 100)
    pred_low_price = btdr['prev'] * (1 + pred_low_pct / 100)
else:
    btdr_open_pct = 0
    pred_high_price = 0
    pred_low_price = 0
    pred_high_pct = 0
    pred_low_pct = 0

# --- 5. 渲染 ---

c1, c2 = st.columns(2)
# 使用 BTC-USD 替代 Binance 数据
c1.metric("BTC (Yahoo源)", f"{btc_chg:+.2f}%") 
c2.metric("恐慌指数", f"{fng_val}")

st.markdown("##### ⚒️ 矿股板块")
cols = st.columns(5)
for i, p in enumerate(peers):
    cols[i].metric(p, f"{quotes[p]['pct']:+.1f}%")

st.markdown("---")

c3, c4 = st.columns(2)
c3.metric("BTDR 现价", f"${btdr['price']:.2f}", f"{btdr['pct']:+.2f}%")
c4.metric("今日开盘", f"${btdr['open']:.2f}", f"{btdr_open_pct:+.2f}%")

st.markdown("### 🎯 AI 预测")
col_h, col_l = st.columns(2)

# 颜色
bg_high = "#d1e7dd"
text_high = "#0f5132"
bg_low = "#f8d7da"
text_low = "#842029"

with col_h:
    st.markdown(f"""
    <div class="pred-box" style="background-color: {bg_high}; color: {text_high}; border: 1px solid #badbcc;">
        <div style="font-size: 0.9rem;">阻力位 (High)</div>
        <div style="font-size: 1.6rem; font-weight: bold;">${pred_high_price:.2f}</div>
        <div style="font-size: 0.85rem;">预期: {pred_high_pct:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col_l:
    st.markdown(f"""
    <div class="pred-box" style="background-color: {bg_low}; color: {text_low}; border: 1px solid #f5c2c7;">
        <div style="font-size: 0.9rem;">支撑位 (Low)</div>
        <div style="font-size: 1.6rem; font-weight: bold;">${pred_low_price:.2f}</div>
        <div style="font-size: 0.85rem;">预期: {pred_low_pct:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("ℹ️ 已启用 Yahoo 原生接口模式。")
