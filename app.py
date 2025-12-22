import streamlit as st
import yfinance as yf
import pandas as pd
import time
from datetime import datetime
import pytz
from streamlit_autorefresh import st_autorefresh

# --- 1. 页面配置 ---
st.set_page_config(page_title="BTDR Pilot v6.3", layout="centered")

# 【核心修改】 interval=5000 (5秒刷新一次)
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
    .time-bar {
        font-size: 0.8rem; color: #666; text-align: center;
        margin-bottom: 15px; padding: 5px; background: #f1f3f5; border-radius: 5px;
    }
    /* 增加一个呼吸灯动效，提示正在刷新 */
    .pulse-dot {
        height: 8px; width: 8px; background-color: #00e676;
        border-radius: 50%; display: inline-block; margin-right: 5px;
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

st.markdown("### ⚡ BTDR 极速盯盘")

# --- 2. 黄金参数 ---
MODEL = {
    "high": {"intercept": 4.29, "beta_open": 0.67, "beta_btc": 0.52},
    "low":  {"intercept": -3.22, "beta_open": 0.88, "beta_btc": 0.42},
    "beta_sector": 0.25
}

# --- 3. 极速数据获取 ---
# 【核心修改】缓存 ttl 改为 5 秒，配合自动刷新，保证不读旧数据
@st.cache_data(ttl=5)
def get_fast_data():
    tickers = ["BTC-USD", "BTDR", "MARA", "RIOT", "CORZ", "CLSK", "IREN"]
    
    # 1. 下载最近 5 天日线 (拿昨收)
    daily = yf.download(tickers, period="5d", interval="1d", progress=False)
    
    # 2. 下载最近 1 天分钟线 (拿实时价，含盘前盘后)
    # interval="1m" 比 "2m" 更快，适合5秒刷新
    live = yf.download(tickers, period="1d", interval="1m", prepost=True, progress=False)
    
    quotes = {}
    
    for t in tickers:
        try:
            # --- 提取昨收 (Prev Close) ---
            if isinstance(daily.columns, pd.MultiIndex):
                closes_d = daily.xs('Close', axis=1, level=0)[t].dropna()
                opens_d = daily.xs('Open', axis=1, level=0)[t].dropna()
            else:
                closes_d = daily['Close'][t].dropna()
                opens_d = daily['Open'][t].dropna()
            
            # 取倒数第二个作为昨收（因为倒数第一个是今天的日线，还没收盘）
            if len(closes_d) >= 2:
                prev_close = closes_d.iloc[-2]
            elif len(closes_d) == 1:
                prev_close = closes_d.iloc[-1]
            else:
                prev_close = 1.0

            # --- 提取实时价 (Live Price) ---
            if isinstance(live.columns, pd.MultiIndex):
                closes_l = live.xs('Close', axis=1, level=0)[t].dropna()
            else:
                closes_l = live['Close'][t].dropna()
            
            if len(closes_l) > 0:
                current_price = closes_l.iloc[-1]
            else:
                # 如果分钟线没数据(极端情况)，用日线的最新价兜底
                if len(closes_d) > 0:
                    current_price = closes_d.iloc[-1]
                else:
                    current_price = prev_close

            # --- 计算涨跌幅 ---
            pct = ((current_price - prev_close) / prev_close) * 100
            
            # --- 提取今日开盘 (Open) ---
            # 逻辑：如果日线数据里有今天的 Open，就用；否则用当前价暂代
            if len(opens_d) > 0 and opens_d.index[-1].date() == pd.Timestamp.now().date():
                 open_price = opens_d.iloc[-1]
            else:
                 open_price = current_price

            quotes[t] = {
                "price": current_price,
                "pct": pct,
                "prev": prev_close,
                "open": open_price
            }
        except:
            quotes[t] = {"price": 0, "pct": 0, "prev": 0, "open": 0}
            
    return quotes

def get_sentiment():
    try:
        # 情绪接口超时设短一点，别卡住主线程
        fng = requests.get("https://api.alternative.me/fng/", timeout=1).json()
        return int(fng['data'][0]['value'])
    except:
        return 50

# --- 4. 主计算逻辑 ---

# 不显示 spinner 圈圈了，免得5秒闪一次眼晕
quotes = get_fast_data()
fng_val = get_sentiment()

# 提取 BTC
btc_chg = quotes['BTC-USD']['pct']

# 板块 Alpha
peers = ["MARA", "RIOT", "CORZ", "CLSK", "IREN"]
valid_peers = [p for p in peers if quotes[p]['price'] > 0]
peers_avg = sum(quotes[p]['pct'] for p in valid_peers) / len(valid_peers) if valid_peers else 0
sector_alpha = peers_avg - btc_chg
sentiment_adj = (fng_val - 50) * 0.02

# BTDR 预测
btdr = quotes['BTDR']

if btdr['price'] > 0:
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
    <div class='pulse-dot'></div>
    北京: <b>{now_bj}</b> &nbsp;|&nbsp; 美东: <b>{now_ny}</b>
</div>
""", unsafe_allow_html=True)

# 核心指标
c1, c2 = st.columns(2)
c1.metric("BTC (24h)", f"{btc_chg:+.2f}%")
c2.metric("恐慌指数", f"{fng_val}")

# 板块
st.markdown("##### ⚒️ 矿股板块 Beta")
cols = st.columns(5)
for i, p in enumerate(peers):
    val = quotes[p]['pct']
    cols[i].metric(p, f"{val:+.1f}%")

st.markdown("---")

# BTDR 本体
c3, c4 = st.columns(2)
c3.markdown(f"<div style='font-size:0.9rem; color:#666;'>BTDR 实时价</div>", unsafe_allow_html=True)
c3.markdown(f"<div style='font-size:1.8rem; font-weight:bold; color:#212529;'>${btdr['price']:.2f}</div>", unsafe_allow_html=True)
c3.markdown(f"<div style='color:{'#198754' if btdr['pct']>=0 else '#dc3545'}; font-weight:bold;'>{btdr['pct']:+.2f}%</div>", unsafe_allow_html=True)

c4.metric("计算用开盘", f"${btdr['open']:.2f}", f"{btdr_open_pct:+.2f}%")

# 预测结果
st.markdown("### 🎯 AI 全时段预测")

col_h, col_l = st.columns(2)
bg_high = "#d1e7dd"; text_high = "#0f5132"
bg_low = "#f8d7da"; text_low = "#842029"

# 动态高亮
high_border = "2px solid #00e676" if btdr['price'] >= pred_high_price else "1px solid #badbcc"
low_border = "2px solid #ff1744" if btdr['price'] <= pred_low_price else "1px solid #f5c2c7"

if btdr['price'] > 0:
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
else:
    st.info("正在建立全时段数据连接...")

st.caption(f"刷新频率: 5秒/次 | 模式: 极速抢单 (1m Data) | 状态: 🟢 在线")
