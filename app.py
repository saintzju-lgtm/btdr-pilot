import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime
import pytz
from streamlit_autorefresh import st_autorefresh

# --- 1. 页面配置 ---
st.set_page_config(page_title="BTDR Pilot v6.4", layout="centered")

# 自动刷新 (5秒)
# 这一版因为请求极快，刷新会非常平滑
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
        box-shadow: 0 4px 10px rgba(0,0,0,0.08); transition: all 0.2s ease;
    }
    .time-bar {
        font-size: 0.8rem; color: #666; text-align: center;
        margin-bottom: 10px; padding: 5px; background: #f1f3f5; border-radius: 5px;
    }
    .status-badge {
        font-size: 0.7rem; padding: 2px 6px; border-radius: 4px; font-weight: bold; margin-left: 5px; vertical-align: middle;
    }
    .tag-pre { background: #fff3cd; color: #856404; }
    .tag-reg { background: #d1e7dd; color: #0f5132; }
    .tag-post { background: #cff4fc; color: #055160; }
    .tag-closed { background: #e2e3e5; color: #383d41; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("### ⚡ BTDR 全时段极速版")

# --- 2. 黄金参数 ---
MODEL = {
    "high": {"intercept": 4.29, "beta_open": 0.67, "beta_btc": 0.52},
    "low":  {"intercept": -3.22, "beta_open": 0.88, "beta_btc": 0.42},
    "beta_sector": 0.25
}

# --- 3. 极速批量数据获取 (Batch Fetch) ---
def get_batch_data():
    # 一次性请求所有代码，速度提升 5 倍，消除闪烁
    symbols = "BTC-USD,BTDR,MARA,RIOT,CORZ,CLSK,IREN"
    url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbols}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    quotes = {}
    try:
        # 极短超时，为了不卡UI
        resp = requests.get(url, headers=headers, timeout=2)
        data = resp.json()
        results = data['quoteResponse']['result']
        
        for item in results:
            sym = item['symbol']
            
            # --- 智能价格选择逻辑 (核心) ---
            # 优先级：Post > Pre > Regular
            # 但要根据 marketState 判断
            
            state = item.get('marketState', 'REGULAR')
            regular_price = item.get('regularMarketPrice', 0)
            prev_close = item.get('regularMarketPreviousClose', regular_price)
            
            display_price = regular_price
            tag = "REG"
            
            # 判断逻辑：如果现在是盘前/盘后，且有对应价格，就强制显示那个价格
            if state == 'PRE' and 'preMarketPrice' in item and item['preMarketPrice']:
                display_price = item['preMarketPrice']
                tag = "PRE"
            elif state == 'POST' and 'postMarketPrice' in item and item['postMarketPrice']:
                display_price = item['postMarketPrice']
                tag = "POST"
            elif state == 'CLOSED':
                # 休市时，如果有盘后价，优先显示盘后价
                if 'postMarketPrice' in item and item['postMarketPrice']:
                    display_price = item['postMarketPrice']
                    tag = "POST"
                else:
                    tag = "CLOSED"

            # 计算相对于昨收的涨跌
            if prev_close and prev_close > 0:
                pct = ((display_price - prev_close) / prev_close) * 100
            else:
                pct = 0
            
            # 获取开盘价 (用于 BTDR 预测)
            # 优先用 regularMarketOpen，如果没有(盘前)，用当前 display_price 模拟
            open_price = item.get('regularMarketOpen')
            if open_price is None:
                open_price = display_price

            quotes[sym] = {
                "price": display_price,
                "pct": pct,
                "prev": prev_close,
                "open": open_price,
                "tag": tag
            }
            
    except Exception as e:
        # 出错兜底
        pass
        
    return quotes

def get_sentiment():
    try:
        fng = requests.get("https://api.alternative.me/fng/", timeout=1).json()
        return int(fng['data'][0]['value'])
    except:
        return 50

# --- 4. 主逻辑 (先计算，后渲染) ---

# 1. 批量抓取 (Batch)
raw_quotes = get_batch_data()
fng_val = get_sentiment()

# 数据完整性检查 (防止刚启动报错)
if 'BTDR' not in raw_quotes:
    st.warning("正在建立高速连接...")
    st.stop()

# 2. 提取数据
btc_chg = raw_quotes.get('BTC-USD', {'pct': 0})['pct']
btdr = raw_quotes['BTDR']

peers = ["MARA", "RIOT", "CORZ", "CLSK", "IREN"]
# 过滤有效数据
valid_peers = [p for p in peers if p in raw_quotes and raw_quotes[p]['price'] > 0]
if valid_peers:
    peers_avg = sum(raw_quotes[p]['pct'] for p in valid_peers) / len(valid_peers)
else:
    peers_avg = 0

# 3. 预测计算
sector_alpha = peers_avg - btc_chg
sentiment_adj = (fng_val - 50) * 0.02

if btdr['price'] > 0 and btdr['prev'] > 0:
    # 动态开盘涨跌幅
    btdr_open_pct = ((btdr['open'] - btdr['prev']) / btdr['prev']) * 100
    
    pred_high_pct = (MODEL['high']['intercept'] + (MODEL['high']['beta_open'] * btdr_open_pct) + (MODEL['high']['beta_btc'] * btc_chg) + (MODEL['beta_sector'] * sector_alpha) + sentiment_adj)
    pred_low_pct = (MODEL['low']['intercept'] + (MODEL['low']['beta_open'] * btdr_open_pct) + (MODEL['low']['beta_btc'] * btc_chg) + (MODEL['beta_sector'] * sector_alpha) + sentiment_adj)
    
    pred_high_price = btdr['prev'] * (1 + pred_high_pct / 100)
    pred_low_price = btdr['prev'] * (1 + pred_low_pct / 100)
else:
    btdr_open_pct = 0; pred_high_price = 0; pred_low_price = 0; pred_high_pct = 0; pred_low_pct = 0

# --- 5. 界面渲染 ---

# 时间栏
tz_bj = pytz.timezone('Asia/Shanghai')
tz_ny = pytz.timezone('America/New_York')
now_bj = datetime.now(tz_bj).strftime('%H:%M:%S')
now_ny = datetime.now(tz_ny).strftime('%H:%M:%S')

st.markdown(f"""
<div class='time-bar'>
    北京: <b>{now_bj}</b> &nbsp;|&nbsp; 美东: <b>{now_ny}</b>
</div>
""", unsafe_allow_html=True)

# 核心指标
c1, c2 = st.columns(2)
c1.metric("BTC (实时)", f"{btc_chg:+.2f}%")
c2.metric("恐慌指数", f"{fng_val}")

# 板块微缩图
st.markdown("##### ⚒️ 矿股板块 Beta")
cols = st.columns(5)
for i, p in enumerate(peers):
    if p in raw_quotes:
        cols[i].metric(p, f"{raw_quotes[p]['pct']:+.1f}%")

st.markdown("---")

# BTDR 数据 (高亮显示状态)
state_tag = btdr['tag']
tag_class = f"tag-{state_tag.lower()}"
tag_text = {"PRE":"盘前 Pre", "REG":"盘中 Reg", "POST":"盘后 Post", "CLOSED":"休市"}.get(state_tag, state_tag)

c3, c4 = st.columns(2)
c3.markdown(f"<div style='font-size:0.9rem; color:#666;'>BTDR <span class='status-badge {tag_class}'>{tag_text}</span></div>", unsafe_allow_html=True)
c3.markdown(f"<div style='font-size:1.8rem; font-weight:bold; color:#212529;'>${btdr['price']:.2f}</div>", unsafe_allow_html=True)
c3.markdown(f"<div style='color:{'#198754' if btdr['pct']>=0 else '#dc3545'}; font-weight:bold;'>{btdr['pct']:+.2f}%</div>", unsafe_allow_html=True)

# 这里的开盘价，会随着盘前/盘后自动切换逻辑，保证预测始终动态
c4.metric("计算用开盘", f"${btdr['open']:.2f}", f"{btdr_open_pct:+.2f}%")

# 预测结果
st.markdown("### 🎯 AI 全时段预测")
col_h, col_l = st.columns(2)

bg_high = "#d1e7dd"; text_high = "#0f5132"
bg_low = "#f8d7da"; text_low = "#842029"

# 动态高亮 (如果现价突破预测位)
h_border = "2px solid #00e676" if btdr['price'] >= pred_high_price else "1px solid #badbcc"
l_border = "2px solid #ff1744" if btdr['price'] <= pred_low_price else "1px solid #f5c2c7"

with col_h:
    st.markdown(f"""
    <div class="pred-box" style="background-color: {bg_high}; color: {text_high}; border: {h_border};">
        <div style="font-size: 0.9rem;">阻力位 (High)</div>
        <div style="font-size: 1.6rem; font-weight: bold;">${pred_high_price:.2f}</div>
        <div style="font-size: 0.8rem;">预期: {pred_high_pct:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col_l:
    st.markdown(f"""
    <div class="pred-box" style="background-color: {bg_low}; color: {text_low}; border: {l_border};">
        <div style="font-size: 0.9rem;">支撑位 (Low)</div>
        <div style="font-size: 1.6rem; font-weight: bold;">${pred_low_price:.2f}</div>
        <div style="font-size: 0.8rem;">预期: {pred_low_pct:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
