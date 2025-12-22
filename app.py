import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime
import pytz
from streamlit_autorefresh import st_autorefresh

# --- 1. 页面配置 ---
st.set_page_config(page_title="BTDR Pilot 24h", layout="centered")

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
    .tag-pre { background: #fff3cd; color: #856404; border: 1px solid #ffeeba; } /* 盘前-黄 */
    .tag-reg { background: #d1e7dd; color: #0f5132; border: 1px solid #badbcc; } /* 盘中-绿 */
    .tag-post { background: #cff4fc; color: #055160; border: 1px solid #b6effb; } /* 盘后-蓝 */
    .tag-closed { background: #e2e3e5; color: #383d41; border: 1px solid #d6d8db; } /* 休市-灰 */
    
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

# --- 3. 核心数据获取 (全时段逻辑) ---
def fetch_yahoo_extended(symbol):
    try:
        t = int(time.time() * 1000)
        headers = {'User-Agent': 'Mozilla/5.0'}
        # 【关键】加入 includePrePost=true 参数，强制获取盘前盘后数据
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&includePrePost=true&useYfid=true&_={t}"
        
        resp = requests.get(url, headers=headers, timeout=3)
        data = resp.json()
        meta = data['chart']['result'][0]['meta']
        
        # 1. 获取常规数据
        regular_price = meta['regularMarketPrice']
        prev_close = meta['chartPreviousClose']
        
        # 2. 智能价格选择器
        # Yahoo 会返回 regularMarketTime, preMarketTime, postMarketTime
        # 我们比较哪个时间最新，就显示哪个价格
        
        # 初始化为常规价格
        display_price = regular_price
        market_state = "REG" # REG, PRE, POST, CLOSED
        
        now_ts = int(time.time())
        reg_time = meta.get('regularMarketTime', 0)
        pre_time = meta.get('preMarketTime', 0)
        post_time = meta.get('postMarketTime', 0)
        
        # 逻辑：谁的时间戳最大（最新），就用谁
        # 注意：盘后时间 > 常规时间 > 盘前时间 (时间轴上)
        
        latest_time = reg_time
        
        # 检查盘后 (Post-Market)
        # 只有当 post_time 比 reg_time 晚，且有价格时才用
        if post_time > latest_time and meta.get('postMarketPrice'):
            display_price = meta['postMarketPrice']
            latest_time = post_time
            market_state = "POST"
            
        # 检查盘前 (Pre-Market)
        # 盘前比较特殊，通常是在 Regular 开始之前。如果 pre_time 比 reg_time 晚（跨日了），或者比 last_close 晚
        # 简单判定：如果 pre_time 比 reg_time 还新（说明是新的一天的盘前），或者当前时间处于美东4:00-9:30
        # 实际上 Yahoo 只要返回了 preMarketPrice 且时间很新，就是盘前
        if pre_time > latest_time and meta.get('preMarketPrice'):
            display_price = meta['preMarketPrice']
            latest_time = pre_time
            market_state = "PRE"

        # 计算实时涨跌幅 (相对于昨收)
        pct = ((display_price - prev_close) / prev_close) * 100
        
        # 获取开盘价 (用于预测)
        # 如果是盘前/盘后，且没有正式开盘价，就用当前价作为"模拟开盘价"来跑预测
        open_price = meta.get('regularMarketOpen')
        if open_price is None: 
            open_price = display_price

        return {
            "price": display_price,
            "pct": pct,
            "prev": prev_close,
            "open": open_price,
            "state": market_state
        }
    except Exception as e:
        return {"price": 0, "pct": 0, "prev": 0, "open": 0, "state": "ERR"}

def get_all_data():
    # BTC 永远是全时段的
    btc_data = fetch_yahoo_extended("BTC-USD")
    try:
        fng = requests.get("https://api.alternative.me/fng/", timeout=1).json()
        fng_val = int(fng['data'][0]['value'])
    except:
        fng_val = 50 

    tickers = ["BTDR", "MARA", "RIOT", "CORZ", "CLSK", "IREN"]
    quotes = {}
    for t in tickers:
        quotes[t] = fetch_yahoo_extended(t)
            
    return btc_data['pct'], fng_val, quotes

# --- 4. 核心计算 ---
btc_chg, fng_val, quotes = get_all_data()

peers = ["MARA", "RIOT", "CORZ", "CLSK", "IREN"]
valid_peers = [p for p in peers if quotes[p]['price'] > 0]
peers_avg = sum(quotes[p]['pct'] for p in valid_peers) / len(valid_peers) if valid_peers else 0

sector_alpha = peers_avg - btc_chg
sentiment_adj = (fng_val - 50) * 0.02

btdr = quotes['BTDR']
btdr_state = btdr['state'] # 获取当前状态

if btdr['price'] > 0:
    # 动态开盘涨跌幅：
    # 如果是盘前(PRE)，btdr['open'] 近似等于当前盘前价，预测结果会随盘前波动
    btdr_open_pct = ((btdr['open'] - btdr['prev']) / btdr['prev']) * 100
    
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
c1.metric("BTC (24h)", f"{btc_chg:+.2f}%")
c2.metric("恐慌指数", f"{fng_val}")

# 板块微缩图 (只显示涨跌幅，紧凑)
st.markdown("##### ⚒️ 矿股板块 Beta")
cols = st.columns(5)
for i, p in enumerate(peers):
    state_label = ""
    # 如果同行也是盘后/盘前，可以加个小点，这里为了界面简洁暂不加
    cols[i].metric(p, f"{quotes[p]['pct']:+.1f}%")

st.markdown("---")

# BTDR 数据 (带状态标签)
state_html = ""
if btdr_state == "PRE":
    state_html = "<span class='status-tag tag-pre'>盘前 Pre</span>"
elif btdr_state == "POST":
    state_html = "<span class='status-tag tag-post'>盘后 Post</span>"
elif btdr_state == "REG":
    state_html = "<span class='status-tag tag-reg'>盘中 Reg</span>"
else:
    state_html = "<span class='status-tag tag-closed'>休市 Closed</span>"

c3, c4 = st.columns(2)
c3.markdown(f"<div style='font-size:0.9rem; color:#666;'>BTDR 现价 {state_html}</div>", unsafe_allow_html=True)
c3.markdown(f"<div style='font-size:1.6rem; font-weight:bold; color:#212529;'>${btdr['price']:.2f}</div>", unsafe_allow_html=True)
c3.markdown(f"<div style='color:{'#198754' if btdr['pct']>=0 else '#dc3545'}; font-weight:bold;'>{btdr['pct']:+.2f}%</div>", unsafe_allow_html=True)

c4.metric("计算用开盘", f"${btdr['open']:.2f}", f"{btdr_open_pct:+.2f}%")

# 预测结果
st.markdown("### 🎯 AI 全时段预测")
col_h, col_l = st.columns(2)

bg_high = "#d1e7dd"; text_high = "#0f5132"
bg_low = "#f8d7da"; text_low = "#842029"

# 动态高亮逻辑：如果现价在盘后突破了，照样高亮
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
st.caption(f"已启用全时段数据流 (Pre/Post/Reg) | 更新于: {time.strftime('%H:%M:%S')}")
