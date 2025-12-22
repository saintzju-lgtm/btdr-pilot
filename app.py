import streamlit as st
import yfinance as yf
import pandas as pd
import requests

# --- 1. 页面配置与美化 (保持白底风格) ---
st.set_page_config(page_title="BTDR Pilot v5.2", layout="centered")

st.markdown("""
    <style>
    .stApp {background-color: #f8f9fa;}
    h1, h2, h3, div, p {color: #212529 !important;}
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricValue"] {font-weight: 700; color: #212529 !important;}
    [data-testid="stMetricLabel"] {color: #6c757d !important;}
    .pred-box {
        padding: 15px; border-radius: 8px; margin-top: 10px; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("⚡ BTDR 领航员 v5.2 (一致性修正)")

# --- 2. 核心模型：直接复用插件的“黄金参数” ---
# 这是您觉得最准的那一套参数，不再让服务器乱算
MODEL = {
    "high": {"intercept": 4.29, "beta_open": 0.67, "beta_btc": 0.52},
    "low":  {"intercept": -3.22, "beta_open": 0.88, "beta_btc": 0.42},
    "beta_sector": 0.25  # 板块权重
}

# --- 3. 数据获取 ---
@st.cache_data(ttl=30) # 30秒刷新一次
def get_data():
    # A. 获取 BTC (yfinance)
    try:
        btc = yf.Ticker("BTC-USD").history(period="2d")
        if len(btc) >= 2:
            btc_chg = ((btc['Close'].iloc[-1] - btc['Close'].iloc[-2]) / btc['Close'].iloc[-2]) * 100
        else:
            btc_chg = 0.0
    except:
        btc_chg = 0.0

    # B. 获取情绪 (API)
    try:
        fng = requests.get("https://api.alternative.me/fng/", timeout=3).json()
        fng_val = int(fng['data'][0]['value'])
    except:
        fng_val = 50

    # C. 获取股票 (BTDR + 5 Peers)
    tickers = ["BTDR", "MARA", "RIOT", "CORZ", "CLSK", "IREN"]
    data = yf.download(tickers, period="5d", interval="1d", progress=False)
    
    quotes = {}
    for t in tickers:
        try:
            # 兼容 yfinance 不同版本的数据结构
            if isinstance(data.columns, pd.MultiIndex):
                closes = data.xs('Close', axis=1, level=0)[t].dropna()
                opens = data.xs('Open', axis=1, level=0)[t].dropna()
            else:
                closes = data['Close'][t].dropna()
                opens = data['Open'][t].dropna()

            if len(closes) >= 2:
                curr = closes.iloc[-1]
                prev = closes.iloc[-2]
                pct = ((curr - prev) / prev) * 100
                
                # BTDR 开盘价逻辑
                open_val = 0
                if t == "BTDR":
                    if len(opens) > 0:
                        # 优先取今日开盘，如果没有(盘前)则用当前价
                        # 注意：yf 在盘中 open 数据有时会有延迟，这里做容错
                        open_val = opens.iloc[-1]
                        # 如果 API 返回的 Open 日期比 Close 日期老，说明今日还没 Open 数据
                        if opens.index[-1] < closes.index[-1]:
                           open_val = curr 
                    else:
                        open_val = curr

                quotes[t] = {"price": curr, "pct": pct, "prev": prev, "open": open_val}
            else:
                quotes[t] = {"price":0, "pct":0, "prev":0, "open":0}
        except:
            quotes[t] = {"price":0, "pct":0, "prev":0, "open":0}
            
    return btc_chg, fng_val, quotes

# --- 4. 主计算逻辑 (完全复刻 JS 插件逻辑) ---

with st.spinner('正在同步数据...'):
    btc_chg, fng_val, quotes = get_data()

# 计算板块 Beta (5只股票平均)
peers = ["MARA", "RIOT", "CORZ", "CLSK", "IREN"]
peers_sum = sum([quotes[p]['pct'] for p in peers if quotes[p]['price'] > 0])
peers_count = sum([1 for p in peers if quotes[p]['price'] > 0])
peers_avg = peers_sum / peers_count if peers_count > 0 else 0

# 关键：板块 Alpha 计算
sector_alpha = peers_avg - btc_chg

# 情绪修正
sentiment_adj = (fng_val - 50) * 0.02

# BTDR 数据准备
btdr = quotes['BTDR']
if btdr['price'] > 0 and btdr['prev'] > 0:
    # 计算开盘涨跌幅 (核心输入)
    btdr_open_pct = ((btdr['open'] - btdr['prev']) / btdr['prev']) * 100
    
    # --- 核心公式 (直接使用 MODEL 常量，不再训练) ---
    
    # High 预测
    pred_high_pct = (MODEL['high']['intercept'] 
                     + (MODEL['high']['beta_open'] * btdr_open_pct) 
                     + (MODEL['high']['beta_btc'] * btc_chg) 
                     + (MODEL['beta_sector'] * sector_alpha) 
                     + sentiment_adj)
    
    # Low 预测
    pred_low_pct = (MODEL['low']['intercept'] 
                    + (MODEL['low']['beta_open'] * btdr_open_pct) 
                    + (MODEL['low']['beta_btc'] * btc_chg) 
                    + (MODEL['beta_sector'] * sector_alpha) 
                    + sentiment_adj)
    
    # 价格换算
    pred_high_price = btdr['prev'] * (1 + pred_high_pct / 100)
    pred_low_price = btdr['prev'] * (1 + pred_low_pct / 100)
    
else:
    btdr_open_pct = 0
    pred_high_price = 0
    pred_low_price = 0
    pred_high_pct = 0
    pred_low_pct = 0

# --- 5. 渲染界面 ---

# 头部数据
c1, c2 = st.columns(2)
c1.metric("BTC 实时", f"{btc_chg:+.2f}%")
c2.metric("恐慌指数", f"{fng_val}")

# 矿股板块
st.markdown("##### ⚒️ 矿股板块 (Sector Beta)")
cols = st.columns(5)
for i, p in enumerate(peers):
    cols[i].metric(p, f"{quotes[p]['pct']:+.1f}%")

st.markdown("---")

# BTDR 数据
c3, c4 = st.columns(2)
c3.metric("BTDR 现价", f"${btdr['price']:.2f}", f"{btdr['pct']:+.2f}%")
c4.metric("今日开盘", f"${btdr['open']:.2f}", f"{btdr_open_pct:+.2f}%")

# 预测结果展示
st.markdown("### 🎯 AI 预测 (黄金参数版)")

# 颜色定义
bg_high = "#d1e7dd"
text_high = "#0f5132"
bg_low = "#f8d7da"
text_low = "#842029"

col_h, col_l = st.columns(2)

with col_h:
    st.markdown(f"""
    <div class="pred-box" style="background-color: {bg_high}; color: {text_high}; border: 1px solid #badbcc;">
        <div style="font-size: 0.9rem;">阻力位 (High)</div>
        <div style="font-size: 1.6rem; font-weight: bold;">${pred_high_price:.2f}</div>
        <div style="font-size: 0.85rem;">预期涨幅: {pred_high_pct:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col_l:
    st.markdown(f"""
    <div class="pred-box" style="background-color: {bg_low}; color: {text_low}; border: 1px solid #f5c2c7;">
        <div style="font-size: 0.9rem;">支撑位 (Low)</div>
        <div style="font-size: 1.6rem; font-weight: bold;">${pred_low_price:.2f}</div>
        <div style="font-size: 0.85rem;">预期涨幅: {pred_low_pct:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("ℹ️ 模型说明：已强制对齐 Chrome 插件 v4.3 核心算法。使用人工校准的黄金参数，剔除实时训练噪音。")
