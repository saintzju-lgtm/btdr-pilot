import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# --- 页面配置 ---
st.set_page_config(page_title="BTDR 量化终端 V2", layout="wide")

# --- 核心数据获取函数 (修复版) ---
@st.cache_data(ttl=600)
def get_clean_data():
    ticker = "BTDR"
    # 获取数据并强制转换格式
    raw_df = yf.download(ticker, period="120d", interval="1d")
    
    # 修复 yfinance 的 MultiIndex 问题
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)
    
    df = raw_df.copy()
    
    # 1. 基础指标计算 (使用 .copy() 避免 SettingWithCopyWarning)
    df['Prev_Close'] = df['Close'].shift(1)
    
    # 2. 功能 (1): 拟合比例计算 (基于截图公式)
    # 计算开盘比、最高比、最低比
    df['Open_Ratio'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    df['Max_Ratio'] = (df['High'] - df['Prev_Close']) / df['Prev_Close']
    df['Min_Ratio'] = (df['Low'] - df['Prev_Close']) / df['Prev_Close']
    
    # 3. 功能 (2): 形态与量化指标
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Vol_MA5'] = df['Volume'].rolling(5).mean()
    df['Body_Size'] = (df['Close'] - df['Open']) / df['Open']
    
    return df.dropna()

# --- 逻辑处理 ---
try:
    df = get_clean_data()
    last_row = df.iloc[-1]
    
    # 获取实时数据
    ticker_obj = yf.Ticker("BTDR")
    # 优先获取最新的实时价
    live_info = ticker_obj.fast_info
    current_price = live_info['last_price']
    # 如果处于盘中，使用今日开盘；如果还没开盘，使用预期价格
    today_open = live_info.get('open', current_price) 
    
except Exception as e:
    st.error(f"数据加载失败，请检查网络或代码: {e}")
    st.stop()

# --- 1. 波动范围拟合 (Linear Regression) ---
X = df[['Open_Ratio']].values
y_h = df['Max_Ratio'].values
y_l = df['Min_Ratio'].values

model_h = LinearRegression().fit(X, y_h)
model_l = LinearRegression().fit(X, y_l)

# 今日预测
today_open_ratio = (today_open - last_row['Close']) / last_row['Close']
pred_h_ratio = model_h.predict([[today_open_ratio]])[0]
pred_l_ratio = model_l.predict([[today_open_ratio]])[0]

pred_high_price = last_row['Close'] * (1 + pred_h_ratio)
pred_low_price = last_row['Close'] * (1 + pred_l_ratio)

# --- 2. 复杂形态识别 ---
def detect_complex_patterns(data):
    pats = []
    curr = data.iloc[-1]
    prev = data.iloc[-2]
    
    # A. 吞没形态
    if abs(curr['Close']-curr['Open']) > abs(prev['Close']-prev['Open']):
        if curr['Close'] > curr['Open'] and prev['Close'] < prev['Open']:
            pats.append("🌟 看涨吞没")
        elif curr['Close'] < curr['Open'] and prev['Close'] > prev['Open']:
            pats.append("🌑 看跌吞没")
            
    # B. 量价背离
    if curr['Close'] > prev['Close'] and curr['Volume'] < prev['Vol_MA5'] * 0.8:
        pats.append("⚠️ 缩量上涨 (动能不足)")
        
    # C. 支撑位判断
    if curr['Low'] <= pred_low_price * 1.01:
        pats.append("🛡️ 触及回归支撑区间")
        
    return pats

active_patterns = detect_complex_patterns(df)

# --- 3. UI 展示 ---
st.title("BTDR 实时预测与形态终端")

# 指标卡
m1, m2, m3, m4 = st.columns(4)
m1.metric("当前成交价", f"${current_price:.2f}")
m2.metric("今日开盘涨幅", f"{today_open_ratio:.2%}")
m3.metric("预测最高点", f"${pred_high_price:.2f}", f"{pred_h_ratio:.2%}")
m4.metric("预测最低点", f"${pred_low_price:.2f}", f"{pred_l_ratio:.2%}", delta_color="inverse")

st.divider()

left, right = st.columns([2, 1])

with left:
    st.subheader("📊 价格走势与预测边界")
    fig = go.Figure()
    # K线图
    plot_df = df.tail(30)
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], 
                                 low=plot_df['Low'], close=plot_df['Close'], name="K线"))
    # 预测线
    fig.add_hline(y=pred_high_price, line_dash="dash", line_color="red", annotation_text="今日压力")
    fig.add_hline(y=pred_low_price, line_dash="dash", line_color="green", annotation_text="今日支撑")
    fig.update_layout(xaxis_rangeslider_visible=False, height=450)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("🤖 智能操作建议")
    # 计算操作分数
    score = 0
    if "看涨吞没" in active_patterns: score += 2
    if current_price < pred_low_price * 1.02: score += 2
    if current_price > pred_high_price * 0.98: score -= 3
    
    # 渲染建议
    if score >= 2:
        st.success("### 建议：加仓/买入")
    elif score <= -2:
        st.error("### 建议：减仓/止盈")
    else:
        st.warning("### 建议：观望/持股")
        
    st.write("**当前识别形态/信号：**")
    for p in active_patterns:
        st.write(f"- {p}")
    
    st.write(f"**成交量比 (Vol Ratio):** {last_row['Volume']/last_row['Vol_MA5']:.2f}")

st.dataframe(df.tail(5))
