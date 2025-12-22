import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression

# --- 1. 页面配置与强制白底 CSS ---
st.set_page_config(page_title="BTDR Pilot v5.1", layout="centered")

# 强制明亮主题 CSS
st.markdown("""
    <style>
    /* 全局背景设为极淡的灰色，护眼 */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* 标题颜色强制为深色 */
    h1, h2, h3, h4, h5, p, div {
        color: #212529 !important;
    }
    
    /* 卡片样式：白底、阴影、圆角 */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e9ecef;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #212529 !important;
    }
    
    /* 指标数值颜色 */
    [data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
        color: #212529 !important;
        font-weight: 700;
    }
    
    /* 指标标签颜色 */
    [data-testid="stMetricLabel"] {
        color: #6c757d !important;
        font-size: 0.9rem;
    }
    
    /* 涨跌幅颜色覆盖 (Streamlit默认会处理，这里增强一下) */
    .green-text { color: #198754 !important; font-weight: bold; }
    .red-text { color: #dc3545 !important; font-weight: bold; }
    
    /* 预测框特别样式 */
    .pred-box {
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("⚡ BTDR 领航员 v5.1 (Light)")

# --- 2. 数据获取 (修复 BTC 问题) ---
@st.cache_data(ttl=60) 
def get_data():
    # A. 获取 BTC (改用 yfinance，避免Binance封锁IP)
    try:
        # 获取 BTC-USD 过去2天数据来计算涨跌
        btc_ticker = yf.Ticker("BTC-USD")
        # 只要最新2行
        btc_hist = btc_ticker.history(period="2d")
        
        if len(btc_hist) >= 2:
            current_btc = btc_hist['Close'].iloc[-1]
            prev_btc = btc_hist['Close'].iloc[-2]
            btc_chg = ((current_btc - prev_btc) / prev_btc) * 100
        else:
            btc_chg = 0.0
    except Exception as e:
        btc_chg = 0.0

    # B. 获取情绪指数 (Alternative.me API 通常不封IP，保留)
    try:
        fng_url = "https://api.alternative.me/fng/"
        fng_res = requests.get(fng_url, timeout=5).json()
        fng_val = int(fng_res['data'][0]['value'])
    except:
        fng_val = 50 # 默认中性

    # C. 获取股票数据 (BTDR + 5 Peers)
    tickers = ["BTDR", "MARA", "RIOT", "CORZ", "CLSK", "IREN"]
    # 下载数据，progress=False 不显示进度条
    data = yf.download(tickers, period="5d", interval="1d", progress=False)
    
    quotes = {}
    
    # yfinance 返回的 DataFrame 可能是 MultiIndex，需要处理
    # 结构通常是: data['Close']['BTDR']
    
    for t in tickers:
        try:
            # 提取 Close 列
            if isinstance(data.columns, pd.MultiIndex):
                # 新版 yfinance
                close_series = data.xs('Close', axis=1, level=0)[t]
                open_series = data.xs('Open', axis=1, level=0)[t]
            else:
                # 旧版兼容
                close_series = data['Close'][t]
                open_series = data['Open'][t]

            # 取最后两个有效值 (dropna)
            valid_close = close_series.dropna()
            
            if len(valid_close) >= 2:
                current = valid_close.iloc[-1]
                prev = valid_close.iloc[-2]
                pct = ((current - prev) / prev) * 100
                
                # BTDR 还需要开盘价
                open_price = 0
                if t == "BTDR":
                    # 尝试取当天的 Open
                    valid_open = open_series.dropna()
                    if len(valid_open) > 0:
                        open_price = valid_open.iloc[-1]
                    else:
                        open_price = current # 降级处理
                
                quotes[t] = {
                    "price": current,
                    "pct": pct,
                    "prev": prev,
                    "open": open_price if t == "BTDR" else 0
                }
            else:
                quotes[t] = {"price":0, "pct":0, "prev":0, "open":0}
        except Exception as e:
            quotes[t] = {"price":0, "pct":0, "prev":0, "open":0}
            
    return btc_chg, fng_val, quotes

# --- 3. 实时训练模型 (保持不变) ---
@st.cache_resource(ttl=3600)
def train_model():
    try:
        df = yf.download("BTDR", period="1mo", interval="1d", progress=False)
        # 兼容 yfinance 新版 MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1) # 去掉 Ticker 层级
            
        df['PrevClose'] = df['Close'].shift(1)
        df['OpenPct'] = (df['Open'] - df['PrevClose']) / df['PrevClose'] * 100
        df['HighPct'] = (df['High'] - df['PrevClose']) / df['PrevClose'] * 100
        df['LowPct'] = (df['Low'] - df['PrevClose']) / df['PrevClose'] * 100
        df = df.dropna()
        
        X = df[['OpenPct']]
        model_high = LinearRegression().fit(X, df['HighPct'])
        model_low = LinearRegression().fit(X, df['LowPct'])
        
        return {
            "high_coef": model_high.coef_[0], "high_int": model_high.intercept_,
            "low_coef": model_low.coef_[0], "low_int": model_low.intercept_
        }
    except:
        return {"high_coef": 0.67, "high_int": 4.29, "low_coef": 0.88, "low_int": -3.22}

# --- 4. 主程序逻辑 ---

with st.spinner('正在同步全球市场数据...'):
    btc_chg, fng_val, quotes = get_data()
    model = train_model()

# 计算
peers_sum = 0
count = 0
for t in ["MARA", "RIOT", "CORZ", "CLSK", "IREN"]:
    if quotes[t]['price'] > 0:
        peers_sum += quotes[t]['pct']
        count += 1
peers_avg = peers_sum / count if count > 0 else 0
sector_alpha = peers_avg - btc_chg
sentiment_adj = (fng_val - 50) * 0.02

btdr = quotes['BTDR']
if btdr['price'] > 0:
    btdr_open_pct = ((btdr['open'] - btdr['prev']) / btdr['prev']) * 100
    
    pred_high_pct = model['high_int'] + (model['high_coef'] * btdr_open_pct) + (0.52 * btc_chg) + (0.25 * sector_alpha) + sentiment_adj
    pred_low_pct = model['low_int'] + (model['low_coef'] * btdr_open_pct) + (0.52 * btc_chg) + (0.25 * sector_alpha) + sentiment_adj
    
    pred_high_price = btdr['prev'] * (1 + pred_high_pct / 100)
    pred_low_price = btdr['prev'] * (1 + pred_low_pct / 100)
else:
    # 数据获取失败时的兜底
    btdr_open_pct = 0
    pred_high_price = 0
    pred_low_price = 0
    pred_high_pct = 0
    pred_low_pct = 0

# --- 5. 渲染界面 (使用原生 Metric 组件，自动适配白底) ---

# 第一排
c1, c2 = st.columns(2)
c1.metric("BTC 实时", f"{btc_chg:+.2f}%")
c2.metric("恐慌指数", f"{fng_val}")

# 第二排
st.markdown("##### ⚒️ 矿股板块 (Sector Beta)")
cols = st.columns(5)
peers = ["MARA", "RIOT", "CORZ", "CLSK", "IREN"]
for i, p in enumerate(peers):
    cols[i].metric(p, f"{quotes[p]['pct']:+.1f}%")

# 第三排
st.markdown("---")
c3, c4 = st.columns(2)
c3.metric("BTDR 现价", f"${btdr['price']:.2f}", f"{btdr['pct']:+.2f}%")
c4.metric("今日开盘", f"${btdr['open']:.2f}", f"{btdr_open_pct:+.2f}%")

# 第四排：预测 (使用自定义 HTML 渲染好看的色块)
st.markdown("### 🎯 AI 预测")

# 定义颜色样式
bg_high = "#d1e7dd" # 浅绿背景
text_high = "#0f5132" # 深绿字
bg_low = "#f8d7da" # 浅红背景
text_low = "#842029" # 深红字

col_h, col_l = st.columns(2)

with col_h:
    st.markdown(f"""
    <div class="pred-box" style="background-color: {bg_high}; color: {text_high}; border: 1px solid #badbcc;">
        <div style="font-size: 0.9rem;">阻力位 (High)</div>
        <div style="font-size: 1.5rem; font-weight: bold;">${pred_high_price:.2f}</div>
        <div style="font-size: 0.8rem;">预期: {pred_high_pct:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col_l:
    st.markdown(f"""
    <div class="pred-box" style="background-color: {bg_low}; color: {text_low}; border: 1px solid #f5c2c7;">
        <div style="font-size: 0.9rem;">支撑位 (Low)</div>
        <div style="font-size: 1.5rem; font-weight: bold;">${pred_low_price:.2f}</div>
        <div style="font-size: 0.8rem;">预期: {pred_low_pct:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption(f"数据源: Yahoo Finance (延迟约1-15分钟) | 模型: 实时线性回归 | 刷新时间: {pd.Timestamp.now().strftime('%H:%M:%S')}")
