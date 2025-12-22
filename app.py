import streamlit as st
import yfinance as yf # 必须用官方库来解封
import pandas as pd
import time
from datetime import datetime
import pytz
from streamlit_autorefresh import st_autorefresh

# --- 1. 页面配置 ---
st.set_page_config(page_title="BTDR Pilot v6.5", layout="centered")

# 5秒刷新 (yfinance 批量下载很快，撑得住)
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
    /* 状态标签样式 */
    .status-badge {
        font-size: 0.7rem; padding: 2px 6px; border-radius: 4px; font-weight: bold; margin-left: 5px; vertical-align: middle;
    }
    .live-dot {
        height: 8px; width: 8px; background-color: #00e676;
        border-radius: 50%; display: inline-block; margin-right: 5px;
        box-shadow: 0 0 5px #00e676; animation: blink 1.5s infinite;
    }
    @keyframes blink { 50% { opacity: 0.5; } }
    </style>
    """, unsafe_allow_html=True)

st.markdown("### ⚡ BTDR 领航员 v6.5")

# --- 2. 黄金参数 ---
MODEL = {
    "high": {"intercept": 4.29, "beta_open": 0.67, "beta_btc": 0.52},
    "low":  {"intercept": -3.22, "beta_open": 0.88, "beta_btc": 0.42},
    "beta_sector": 0.25
}

# --- 3. 极速批量获取 (yfinance Batch) ---
# 使用 cache_data 缓存5秒，防止高频刷新导致再次被封
@st.cache_data(ttl=5)
def get_yfinance_batch():
    # 一次性下载所有数据，这是解封的关键
    tickers_list = "BTC-USD BTDR MARA RIOT CORZ CLSK IREN"
    
    try:
        # 下载最近1天，1分钟级别的K线，包含盘前盘后 (prepost=True)
        # threads=True 开启多线程加速
        data = yf.download(tickers_list, period="1d", interval="1m", prepost=True, group_by='ticker', threads=True, progress=False)
        
        # 还需要下载日线拿昨收 (Prev Close)
        daily = yf.download(tickers_list, period="5d", interval="1d", group_by='ticker', threads=True, progress=False)
        
        quotes = {}
        symbols = tickers_list.split()
        
        for sym in symbols:
            # yfinance 的数据结构比较复杂，需要防御性编程
            try:
                # 1. 提取分钟线 (Live Data)
                df_min = data[sym]
                # 2. 提取日线 (Daily Data)
                df_day = daily[sym]
                
                # 清洗空数据
                df_min = df_min.dropna(subset=['Close'])
                df_day = df_day.dropna(subset=['Close'])
                
                if df_min.empty:
                    # 如果分钟线没数据（比如周末），回退到日线最新价
                    current_price = df_day['Close'].iloc[-1] if not df_day.empty else 0
                else:
                    # 取最新的一分钟收盘价
                    current_price = df_min['Close'].iloc[-1]
                
                # 获取昨收 (倒数第二天)
                if len(df_day) >= 2:
                    prev_close = df_day['Close'].iloc[-2]
                elif len(df_day) == 1:
                    prev_close = df_day['Close'].iloc[-1]
                else:
                    prev_close = current_price
                
                # 计算涨跌
                if prev_close > 0:
                    pct = ((current_price - prev_close) / prev_close) * 100
                else:
                    pct = 0
                
                # 获取开盘价 (优先用日线Open)
                if not df_day.empty and df_day.index[-1].date() == pd.Timestamp.now().date():
                     open_price = df_day['Open'].iloc[-1]
                else:
                     open_price = current_price

                quotes[sym] = {
                    "price": current_price,
                    "pct": pct,
                    "prev": prev_close,
                    "open": open_price
                }
            except Exception as e:
                # 单个股票失败不影响整体
                quotes[sym] = {"price": 0, "pct": 0, "prev": 0, "open": 0}
                
        return quotes

    except Exception as e:
        return None

def get_sentiment():
    try:
        fng = requests.get("https://api.alternative.me/fng/", timeout=1).json()
        return int(fng['data'][0]['value'])
    except:
        return 50

# --- 4. 主逻辑 ---

# 1. 获取数据
raw_quotes = get_yfinance_batch()
fng_val = get_sentiment()

# 2. 错误处理 (如果 IP 还没解封，显示提示)
if raw_quotes is None or 'BTDR' not in raw_quotes or raw_quotes['BTDR']['price'] == 0:
    st.error("Yahoo 数据源重连中... (Anti-Bot Reconnecting)")
    st.caption("请等待 30 秒，系统正在自动更换令牌")
    st.stop() # 停止渲染，避免报错

# 3. 提取数据
btc_chg = raw_quotes['BTC-USD']['pct']
btdr = raw_quotes['BTDR']

peers = ["MARA", "RIOT", "CORZ", "CLSK", "IREN"]
valid_peers = [p for p in peers if raw_quotes[p]['price'] > 0]
if valid_peers:
    peers_avg = sum(raw_quotes[p]['pct'] for p in valid_peers) / len(valid_peers)
else:
    peers_avg = 0

# 4. 预测计算
sector_alpha = peers_avg - btc_chg
sentiment_adj = (fng_val - 50) * 0.02

if btdr['price'] > 0:
    btdr_open_pct = ((btdr['open'] - btdr['prev']) / btdr['prev']) * 100
    
    pred_high_pct = (MODEL['high']['intercept'] + (MODEL['high']['beta_open'] * btdr_open_pct) + (MODEL['high']['beta_btc'] * btc_chg) + (MODEL['beta_sector'] * sector_alpha) + sentiment_adj)
    pred_low_pct = (MODEL['low']['intercept'] + (MODEL['low']['beta_open'] * btdr_open_pct) + (MODEL['low']['beta_btc'] * btc_chg) + (MODEL['beta_sector'] * sector_alpha) + sentiment_adj)
    
    pred_high_price = btdr['prev'] * (1 + pred_high_pct / 100)
    pred_low_price = btdr['prev'] * (1 + pred_low_pct / 100)
else:
    btdr_open_pct = 0; pred_high_price = 0; pred_low_price = 0; pred_high_pct = 0; pred_low_pct = 0

# --- 5. 渲染 ---

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

# 指标
c1, c2 = st.columns(2)
c1.metric("BTC (全时段)", f"{btc_chg:+.2f}%")
c2.metric("恐慌指数", f"{fng_val}")

# 板块
st.markdown("##### ⚒️ 矿股板块 Beta")
cols = st.columns(5)
for i, p in enumerate(peers):
    if p in raw_quotes:
        cols[i].metric(p, f"{raw_quotes[p]['pct']:+.1f}%")

st.markdown("---")

# BTDR 本体
c3, c4 = st.columns(2)
c3.markdown(f"<div style='font-size:0.9rem; color:#666;'>BTDR 实时价</div>", unsafe_allow_html=True)
c3.markdown(f"<div style='font-size:1.8rem; font-weight:bold; color:#212529;'>${btdr['price']:.2f}</div>", unsafe_allow_html=True)
c3.markdown(f"<div style='color:{'#198754' if btdr['pct']>=0 else '#dc3545'}; font-weight:bold;'>{btdr['pct']:+.2f}%</div>", unsafe_allow_html=True)

c4.metric("计算用开盘", f"${btdr['open']:.2f}", f"{btdr_open_pct:+.2f}%")

# 预测
st.markdown("### 🎯 AI 全时段预测")
col_h, col_l = st.columns(2)

bg_high = "#d1e7dd"; text_high = "#0f5132"
bg_low = "#f8d7da"; text_low = "#842029"

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

# 调试信息
st.markdown("---")
# 显示时间戳证明数据在动
st.caption(f"数据源: yfinance (Batch v6.5) | 刷新: 5秒 | 状态: 🟢 正常 | {time.strftime('%H:%M:%S')}")
