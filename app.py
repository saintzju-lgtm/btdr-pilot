import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime
import pytz
from streamlit_autorefresh import st_autorefresh

# --- 1. 页面配置 ---
st.set_page_config(page_title="BTDR Pilot v6.1", layout="centered")

# 自动刷新 (5秒)
count = st_autorefresh(interval=5000, limit=None, key="realtime_counter")

st.markdown("""
    <style>
    .stApp {background-color: #ffffff;}
    h1, h2, h3, div, p, span {color: #212529 !important; font-family: -apple-system, BlinkMacSystemFont, sans-serif !important;}
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    
    div[data-testid="stMetric"] {
        background-color: #f8f9fa !important;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
        border-radius: 10px;
    }
    [data-testid="stMetricValue"] {
        font-weight: 700; font-size: 1.4rem !important; color: #212529 !important;
    }
    .pred-box {
        padding: 15px; border-radius: 12px; margin-top: 10px; text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08); transition: all 0.3s ease;
    }
    .status-tag {
        font-size: 0.7rem; padding: 2px 6px; border-radius: 4px; font-weight: bold;
        vertical-align: middle; margin-left: 5px;
    }
    /* 状态颜色定义 */
    .tag-pre { background: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
    .tag-reg { background: #d1e7dd; color: #0f5132; border: 1px solid #badbcc; }
    .tag-post { background: #cff4fc; color: #055160; border: 1px solid #b6effb; }
    .tag-closed { background: #e2e3e5; color: #383d41; border: 1px solid #d6d8db; }
    
    .time-bar {
        font-size: 0.8rem; color: #666; text-align: center;
        margin-bottom: 15px; padding: 5px; background: #f1f3f5; border-radius: 5px;
    }
    .live-dot {
        height: 8px; width: 8px; background-color: #ff5252;
        border-radius: 50%; display: inline-block; margin-right: 5px;
        box-shadow: 0 0 5px #ff5252; animation: blink 1s infinite;
    }
    @keyframes blink { 50% { opacity: 0.5; } }
    </style>
    """, unsafe_allow_html=True)

st.markdown("### ⚡ BTDR 全时段监控")

# --- 2. 黄金参数 ---
MODEL = {
    "high": {"intercept": 4.29, "beta_open": 0.67, "beta_btc": 0.52},
    "low":  {"intercept": -3.22, "beta_open": 0.88, "beta_btc": 0.42},
    "beta_sector": 0.25
}

# --- 3. 核心数据获取 (改用 v7/finance/quote 接口) ---
def fetch_yahoo_quote(symbol):
    try:
        # 使用 quote 接口，这是获取实时报价最准的接口
        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=3)
        data = resp.json()
        
        if not data['quoteResponse']['result']:
            return {"price": 0, "pct": 0, "prev": 0, "open": 0, "state": "ERR"}
            
        q = data['quoteResponse']['result'][0]
        
        # 提取关键字段
        # Yahoo 的 marketState 通常是: PRE, REGULAR, POST, CLOSED
        state = q.get('marketState', 'REGULAR')
        prev_close = q.get('regularMarketPreviousClose', 0)
        
        display_price = 0
        display_pct = 0
        
        # 【智能价格判定逻辑】
        if state == 'PRE' and 'preMarketPrice' in q:
            display_price = q['preMarketPrice']
            # 盘前涨跌幅通常是相对于昨收
            display_pct = q.get('preMarketChangePercent', 0)
            tag = "PRE"
        elif state == 'POST' and 'postMarketPrice' in q:
            display_price = q['postMarketPrice']
            display_pct = q.get('postMarketChangePercent', 0)
            tag = "POST"
        elif state == 'REGULAR':
            display_price = q.get('regularMarketPrice', 0)
            display_pct = q.get('regularMarketChangePercent', 0)
            tag = "REG"
        elif state == 'CLOSED':
            # 如果休市，检查有没有盘后价格 (Post Market Price)
            # 有些时候 CLOSED 状态下 postMarketPrice 才是最新的
            if 'postMarketPrice' in q and q.get('postMarketPrice'):
                display_price = q['postMarketPrice']
                display_pct = q.get('postMarketChangePercent', 0)
                tag = "POST" # 显示为盘后价
            else:
                display_price = q.get('regularMarketPrice', 0)
                display_pct = q.get('regularMarketChangePercent', 0)
                tag = "CLOSED"
        else:
            # 兜底
            display_price = q.get('regularMarketPrice', 0)
            display_pct = q.get('regularMarketChangePercent', 0)
            tag = state

        # 获取开盘价 (用于预测)
        # 优先用 regularMarketOpen，如果是盘前/盘后且没开盘，用当前价模拟
        open_price = q.get('regularMarketOpen')
        if open_price is None: 
            open_price = display_price
            
        return {
            "price": display_price,
            "pct": display_pct,
            "prev": prev_close,
            "open": open_price,
            "state": tag
        }
    except Exception as e:
        return {"price": 0, "pct": 0, "prev": 0, "open": 0, "state": "ERR"}

def get_all_data():
    btc_data = fetch_yahoo_quote("BTC-USD")
    try:
        fng = requests.get("https://api.alternative.me/fng/", timeout=1).json()
        fng_val = int(fng['data'][0]['value'])
    except:
        fng_val = 50 

    tickers = ["BTDR", "MARA", "RIOT", "CORZ", "CLSK", "IREN"]
    quotes = {}
    for t in tickers:
        quotes[t] = fetch_yahoo_quote(t)
            
    return btc_data['pct'], fng_val, quotes

# --- 4. 核心计算 ---
btc_chg, fng_val, quotes = get_all_data()

peers = ["MARA", "RIOT", "CORZ", "CLSK", "IREN"]
# 过滤掉价格为0的无效数据
valid_peers = [p for p in peers if quotes[p]['price'] > 0]
peers_avg = sum(quotes[p]['pct'] for p in valid_peers) / len(valid_peers) if valid_peers else 0

sector_alpha = peers_avg - btc_chg
sentiment_adj = (fng_val - 50) * 0.02

btdr = quotes['BTDR']
btdr_state = btdr['state']

if btdr['price'] > 0:
    # 动态开盘涨跌幅
    if btdr['prev'] > 0:
        btdr_open_pct = ((btdr['open'] - btdr['prev']) / btdr['prev']) * 100
    else:
        btdr_open_pct = 0
        
    pred_high_pct = (MODEL['high']['intercept'] + (MODEL['high']['beta_open'] * btdr_open_pct) + (MODEL['high']['beta_btc'] * btc_chg) + (MODEL['beta_sector'] * sector_alpha) + sentiment_adj)
    pred_low_pct = (MODEL['low']['intercept'] + (MODEL['low']['beta_open'] * btdr_open_pct) + (MODEL['low']['beta_btc'] * btc_chg) + (MODEL['beta_sector'] * sector_alpha) + sentiment_adj)
    
    pred_high_price = btdr['prev'] * (1 + pred_high_pct / 100)
    pred_low_price = btdr['prev'] * (1 + pred_low_pct / 100)
else:
    btdr_open_pct = 0; pred_high_price = 0; pred_low_price = 0; pred_high_pct = 0; pred_low_pct = 0

# --- 5. 渲染界面 ---

# 时间栏
tz_bj = pytz.timezone('Asia/Shanghai')
tz_ny = pytz.timezone('America/New_York')
now_bj = datetime.now(tz_bj).strftime('%H:%M:%S')
now_ny = datetime.now(tz_ny).strftime('%H:%M:%S')

st.markdown(f"""
<div class='time-bar'>
    <div class='live-dot'></div>
    北京: <b>{now_bj}</b> &nbsp;|&nbsp; 美东: <b>{now_ny}</b>
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

# BTDR 数据 (带状态标签)
state_html = ""
if "PRE" in btdr_state:
    state_html = "<span class='status-tag tag-pre'>盘前 Pre</span>"
elif "POST" in btdr_state:
    state_html = "<span class='status-tag tag-post'>盘后 Post</span>"
elif "REG" in btdr_state:
    state_html = "<span class='status-tag tag-reg'>盘中 Reg</span>"
elif "CLOSED" in btdr_state:
    state_html = "<span class='status-tag tag-closed'>休市 Closed</span>"
else:
    state_html = f"<span class='status-tag tag-closed'>{btdr_state}</span>"

c3, c4 = st.columns(2)
c3.markdown(f"<div style='font-size:0.9rem; color:#666;'>BTDR 现价 {state_html}</div>", unsafe_allow_html=True)
c3.markdown(f"<div style='font-size:1.6rem; font-weight:bold; color:#212529;'>${btdr['price']:.2f}</div>", unsafe_allow_html=True)
c3.markdown(f"<div style='color:{'#198754' if btdr['pct']>=0 else '#dc3545'}; font-weight:bold;'>{btdr['pct']:+.2f}%</div>", unsafe_allow_html=True)

c4.metric("今日开盘 (计算用)", f"${btdr['open']:.2f}", f"{btdr_open_pct:+.2f}%")

# 预测结果
st.markdown("### 🎯 AI 全时段预测")
col_h, col_l = st.columns(2)

bg_high = "#d1e7dd"; text_high = "#0f5132"
bg_low = "#f8d7da"; text_low = "#842029"

# 动态高亮
high_border = "2px solid #00e676" if btdr['price'] >= pred_high_price else "1px solid #badbcc"
low_border = "2px solid #ff1744" if btdr['price'] <= pred_low_price else "1px solid #f5c2c7"

with col_h:
    st.markdown(f"""
    <div class="pred-box" style="background-color: {bg_high}; color: {text_high}; border: {high_border};">
        <div style="font-size: 0.9rem;">阻力位 (High)</div>
        <div style="font-size: 1.6rem; font-weight: bold;">${pred_high_price:.2f}</div>
        <div style="font-size: 0.8rem;">预期: {pred_high_pct:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col_l:
    st.markdown(f"""
    <div class="pred-box" style="background-color: {bg_low}; color: {text_low}; border: {low_border};">
        <div style="font-size: 0.9rem;">支撑位 (Low)</div>
        <div style="font-size: 1.6rem; font-weight: bold;">${pred_low_price:.2f}</div>
        <div style="font-size: 0.8rem;">预期: {pred_low_pct:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption(f"状态: 自动巡航 (Quote API v2) | 更新于: {time.strftime('%H:%M:%S')}")
