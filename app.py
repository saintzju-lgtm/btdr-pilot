import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# --- 页面配置 ---
st.set_page_config(page_title="BTDR Pilot v5.0", layout="centered")

# 自定义 CSS 让手机端显示更紧凑
st.markdown("""
    <style>
    .stMetric {background-color: #1e1e1e; padding: 10px; border-radius: 5px; border: 1px solid #333;}
    [data-testid="stMetricValue"] {font-size: 1.2rem !important;}
    h1 {text-align: center; color: #bb86fc; font-size: 1.5rem !important;}
    .big-font {font-size:20px !important; font-weight: bold;}
    .green {color: #00e676;}
    .red {color: #cf6679;}
    </style>
    """, unsafe_allow_html=True)

st.title("⚡ BTDR 领航员 v5.0 Cloud")

# --- 1. 数据获取函数 ---
@st.cache_data(ttl=60) # 缓存60秒，防止刷新太快被封
def get_data():
    # A. 获取 BTC
    try:
        btc_url = "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"
        btc_res = requests.get(btc_url).json()
        btc_chg = float(btc_res['priceChangePercent'])
    except:
        btc_chg = 0.0

    # B. 获取情绪指数
    try:
        fng_url = "https://api.alternative.me/fng/"
        fng_res = requests.get(fng_url).json()
        fng_val = int(fng_res['data'][0]['value'])
    except:
        fng_val = 50

    # C. 获取股票数据 (BTDR + 5 Peers)
    tickers = ["BTDR", "MARA", "RIOT", "CORZ", "CLSK", "IREN"]
    data = yf.download(tickers, period="5d", interval="1d", progress=False)
    
    # 整理最新的涨跌幅
    quotes = {}
    for t in tickers:
        try:
            # 获取最新价和昨收
            # yfinance 的多层索引处理
            current = data['Close'][t].iloc[-1]
            prev = data['Close'][t].iloc[-2]
            
            # 如果是盘中，Close[-1] 是实时价；如果是盘前，这可能不准
            # 为了简单，我们这里用 Close 计算涨跌。
            # 实际上 Streamlit 部署在美区，yfinance 通常能拿延迟15分钟的数据
            pct = ((current - prev) / prev) * 100
            
            # 获取 BTDR 的开盘价用于预测
            open_price = 0
            if t == "BTDR":
                open_price = data['Open'][t].iloc[-1]
            
            quotes[t] = {
                "price": current,
                "pct": pct,
                "prev": prev,
                "open": open_price if t == "BTDR" else 0
            }
        except:
            quotes[t] = {"price":0, "pct":0, "prev":0, "open":0}
            
    return btc_chg, fng_val, quotes

# --- 2. 实时训练 AI 模型 ---
@st.cache_resource(ttl=3600) # 模型缓存在内存里1小时训练一次即可
def train_model():
    # 下载过去30天数据进行回归
    try:
        df = yf.download("BTDR", period="1mo", interval="1d", progress=False)
        df['PrevClose'] = df['Close'].shift(1)
        df['OpenPct'] = (df['Open'] - df['PrevClose']) / df['PrevClose'] * 100
        df['HighPct'] = (df['High'] - df['PrevClose']) / df['PrevClose'] * 100
        df['LowPct'] = (df['Low'] - df['PrevClose']) / df['PrevClose'] * 100
        
        df = df.dropna()
        
        # 简单的一元回归 (为了稳健性，云端版只用 OpenPct 回归，BTC/Sector 作为外部修正)
        X = df[['OpenPct']]
        y_high = df['HighPct']
        y_low = df['LowPct']
        
        model_high = LinearRegression().fit(X, y_high)
        model_low = LinearRegression().fit(X, y_low)
        
        return {
            "high_coef": model_high.coef_[0],
            "high_int": model_high.intercept_,
            "low_coef": model_low.coef_[0],
            "low_int": model_low.intercept_
        }
    except:
        # 默认参数 (如果下载失败)
        return {"high_coef": 0.67, "high_int": 4.29, "low_coef": 0.88, "low_int": -3.22}

# --- 3. 主逻辑 ---

# 显示加载状态
with st.spinner('正在连接华尔街数据中心...'):
    btc_chg, fng_val, quotes = get_data()
    model = train_model()

# 计算板块溢价
# 黄金组合: (MARA + RIOT + CORZ + CLSK + IREN) / 5
peers_avg = (quotes['MARA']['pct'] + quotes['RIOT']['pct'] + quotes['CORZ']['pct'] + quotes['CLSK']['pct'] + quotes['IREN']['pct']) / 5
sector_alpha = peers_avg - btc_chg

# 情绪修正
sentiment_adj = (fng_val - 50) * 0.02

# BTDR 数据
btdr = quotes['BTDR']
btdr_open_pct = ((btdr['open'] - btdr['prev']) / btdr['prev']) * 100

# 预测计算
# 公式: 基础回归 + BTC修正(0.52) + 板块Alpha(0.25) + 情绪修正
beta_btc = 0.52
beta_sector = 0.25

pred_high_pct = model['high_int'] + (model['high_coef'] * btdr_open_pct) + (beta_btc * btc_chg) + (beta_sector * sector_alpha) + sentiment_adj
pred_low_pct = model['low_int'] + (model['low_coef'] * btdr_open_pct) + (beta_btc * btc_chg) + (beta_sector * sector_alpha) + sentiment_adj

pred_high_price = btdr['prev'] * (1 + pred_high_pct / 100)
pred_low_price = btdr['prev'] * (1 + pred_low_pct / 100)

# --- 4. 界面渲染 ---

# 第一排：BTC & 情绪
c1, c2 = st.columns(2)
c1.metric("BTC 实时", f"{btc_chg:+.2f}%")
c2.metric("恐慌指数", f"{fng_val}")

# 第二排：板块五虎
st.caption("矿股板块对标 (Sector Beta)")
cols = st.columns(5)
peers = ["MARA", "RIOT", "CORZ", "CLSK", "IREN"]
for i, p in enumerate(peers):
    cols[i].metric(p, f"{quotes[p]['pct']:+.1f}%", label_visibility="visible")

# 第三排：BTDR 本体
st.markdown("---")
c3, c4 = st.columns(2)
c3.metric("BTDR 现价", f"${btdr['price']:.2f}", f"{btdr['pct']:+.2f}%")
c4.metric("今日开盘", f"${btdr['open']:.2f}")

# 第四排：预测结果 (重点高亮)
st.markdown("### 🤖 AI 今日预测")
col_h, col_l = st.columns(2)

with col_h:
    st.success(f"阻力位: ${pred_high_price:.2f}")
    st.caption(f"预期涨幅: {pred_high_pct:+.2f}%")

with col_l:
    st.error(f"支撑位: ${pred_low_price:.2f}")
    st.caption(f"预期涨幅: {pred_low_pct:+.2f}%")

st.markdown("---")
st.caption(f"模型参数: High_Beta={model['high_coef']:.2f}, Low_Beta={model['low_coef']:.2f} | 实时回归")
