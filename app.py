import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from datetime import datetime

# --- 配置 ---
st.set_page_config(page_title="BTDR 高级量化终端", layout="wide")

# --- 1. 形态识别算法库 ---
def identify_patterns(df):
    """
    识别基础 K 线形态逻辑 (基于价格关系)
    """
    patterns = []
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    # 实体大小与上下影线
    body = last_row['Close'] - last_row['Open']
    abs_body = abs(body)
    upper_shadow = last_row['High'] - max(last_row['Close'], last_row['Open'])
    lower_shadow = min(last_row['Close'], last_row['Open']) - last_row['Low']
    
    # 1. 吞没形态 (Engulfing)
    if abs(prev_row['Close'] - prev_row['Open']) < abs_body:
        if body > 0 and prev_row['Close'] < prev_row['Open']:
            patterns.append("✨ 看涨吞没 (Bullish Engulfing)")
        elif body < 0 and prev_row['Close'] > prev_row['Open']:
            patterns.append("📉 看跌吞没 (Bearish Engulfing)")

    # 2. 锤子线/流星线 (Hammer/Shooting Star)
    if lower_shadow > abs_body * 2 and upper_shadow < abs_body * 0.5:
        patterns.append("🔨 锤子线 (底部信号?)")
    if upper_shadow > abs_body * 2 and lower_shadow < abs_body * 0.5:
        patterns.append("☄️ 流星线 (顶部压力?)")

    # 3. 跳空缺口
    if last_row['Low'] > prev_row['High']:
        patterns.append("🚀 向上跳空缺口")
    elif last_row['High'] < prev_row['Low']:
        patterns.append("🕳️ 向下跳空缺口")

    return patterns if patterns else ["趋势延续/盘整"]

# --- 2. 数据获取与处理 ---
@st.cache_data(ttl=300)
def get_advanced_data():
    ticker = "BTDR"
    # 获取数据
    df = yf.download(ticker, period="120d", interval="1d")
    
    # 基础比例计算
    df['Prev_Close'] = df['Close'].shift(1)
    df['Open_Ratio'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    df['Max_Ratio'] = (df['High'] - df['Prev_Close']) / df['Prev_Close']
    df['Min_Ratio'] = (df['Low'] - df['Prev_Close']) / df['Prev_Close']
    
    # 量能指标
    df['Vol_MA5'] = df['Volume'].rolling(5).mean()
    df['Vol_Ratio'] = df['Volume'] / df['Vol_MA5']
    
    # 均线
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    
    return df.dropna()

data = get_advanced_data()
last_data = data.iloc[-1]
ticker_info = yf.Ticker("BTDR")
live_price = ticker_info.fast_info['last_price']

# --- 3. 回归预测范围 ---
X_train = data[['Open_Ratio']].values
model_h = LinearRegression().fit(X_train, data['Max_Ratio'].values)
model_l = LinearRegression().fit(X_train, data['Min_Ratio'].values)

today_open_r = (last_data['Open'] - last_data['Prev_Close']) / last_data['Prev_Close']
pred_h = last_data['Prev_Close'] * (1 + model_h.predict([[today_open_r]])[0])
pred_l = last_data['Prev_Close'] * (1 + model_l.predict([[today_open_r]])[0])

# --- 4. 界面布局 ---
st.title("BTDR 实时量化形态终端")

col_info, col_chart = st.columns([1, 2])

with col_info:
    st.subheader("🛡️ 决策矩阵")
    
    # 形态识别展示
    current_patterns = identify_patterns(data)
    for p in current_patterns:
        st.warning(p)
    
    # 换手率与成交量
    vol_status = "放量" if last_data['Vol_Ratio'] > 1.5 else "缩量" if last_data['Vol_Ratio'] < 0.7 else "平量"
    st.metric("成交量状态", f"{vol_status}", f"量比: {last_data['Vol_Ratio']:.2f}")
    
    # 操作建议逻辑
    score = 0
    reasons = []
    
    # 逻辑判断
    if live_price < pred_l * 1.01: 
        score += 2; reasons.append("价格接近预测支撑位")
    if "看涨吞没" in str(current_patterns): 
        score += 2; reasons.append("出现看涨形态")
    if last_data['Close'] > last_data['MA20']: 
        score += 1; reasons.append("站稳20日线")
    if last_data['Vol_Ratio'] > 1.2 and last_data['Close'] > last_data['Open']:
        score += 1; reasons.append("量价配合上涨")

    # 输出建议
    st.divider()
    if score >= 4:
        st.success("🎯 综合建议：积极做多 / 加仓")
    elif score >= 2:
        st.info("⚖️ 综合建议：持仓观望")
    else:
        st.error("⚠️ 综合建议：减仓 / 避险")
    
    with st.expander("查看评分逻辑"):
        for r in reasons: st.write(f"- {r}")

with col_chart:
    st.subheader("🕯️ K线与预测区间")
    fig = go.Figure(data=[go.Candlestick(
        x=data.index[-20:],
        open=data['Open'][-20:],
        high=data['High'][-20:],
        low=data['Low'][-20:],
        close=data['Close'][-20:],
        name="K线"
    )])
    
    # 加入预测区间线
    fig.add_hline(y=pred_h, line_dash="dash", line_color="red", annotation_text="预测最高")
    fig.add_hline(y=pred_l, line_dash="dash", line_color="green", annotation_text="预测最低")
    
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

# --- 5. 换手率分析 ---
st.divider()
st.subheader("📊 市场热度 (Market Heat)")
c1, c2, c3 = st.columns(3)
# 换手率计算 (假设流通盘约为 30M, 实际可根据 yf.info['floatShares'] 获取)
float_shares = ticker_info.info.get('floatShares', 30000000)
turnover_rate = (last_data['Volume'] / float_shares) * 100

c1.write(f"**今日估计换手率:** {turnover_rate:.2f}%")
c2.write(f"**MA5/MA20 偏离度:** {((last_data['MA5']/last_data['MA20'])-1)*100:.2f}%")
c3.write(f"**昨日收盘价:** ${last_data['Prev_Close']:.2f}")

st.dataframe(data.tail(5).style.highlight_max(axis=0, subset=['Volume']))
