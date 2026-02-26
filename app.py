import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from datetime import datetime, time

# --- 页面配置 ---
st.set_page_config(page_title="BTDR 盘前决策终端", layout="wide")

@st.cache_data(ttl=60) # 盘前数据建议缓存时间设短（1分钟）
def get_premarket_data():
    ticker_symbol = "BTDR"
    ticker = yf.Ticker(ticker_symbol)
    
    # 1. 获取流通盘 (Float)
    float_shares = ticker.info.get('floatShares', 35000000)
    
    # 2. 获取包含盘前数据的 1分钟 K线 (只取最近1天)
    # yfinance 的 prepost=True 会包含 4:00 AM 以后的数据
    data_1m = yf.download(ticker_symbol, period="1d", interval="1m", prepost=True)
    
    if isinstance(data_1m.columns, pd.MultiIndex):
        data_1m.columns = data_1m.columns.get_level_values(0)
        
    # 3. 筛选盘前时段 (美东时间 04:00 - 09:30)
    # 转换索引为美东时间
    data_1m.index = data_1m.index.tz_convert('America/New_York')
    pre_market = data_1m.between_time('04:00', '09:29')
    
    return pre_market, float_shares, ticker

# --- 获取基础历史数据用于回归 ---
@st.cache_data(ttl=3600)
def get_hist_for_reg():
    df = yf.download("BTDR", period="60d")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df['Prev_Close'] = df['Close'].shift(1)
    df['Open_Ratio'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    df['Max_Ratio'] = (df['High'] - df['Prev_Close']) / df['Prev_Close']
    df['Min_Ratio'] = (df['Low'] - df['Prev_Close']) / df['Prev_Close']
    return df.dropna()

# --- 主逻辑执行 ---
pre_df, float_shares, ticker_obj = get_premarket_data()
hist_df = get_hist_for_reg()
last_close = hist_df['Close'].iloc[-1]

# --- 1. 盘前核心指标计算 ---
if not pre_df.empty:
    pre_vol = pre_df['Volume'].sum()
    pre_high = pre_df['High'].max()
    pre_low = pre_df['Low'].min()
    pre_last_price = pre_df['Close'].iloc[-1]
    
    pre_turnover = (pre_vol / float_shares) * 100
    pre_change = (pre_last_price - last_close) / last_close
else:
    # 盘前未开始或无数据
    pre_vol, pre_last_price, pre_turnover, pre_change = 0, last_close, 0, 0

# --- 2. 基于盘前价格进行回归预测 ---
X = hist_df[['Open_Ratio']].values
model_h = LinearRegression().fit(X, hist_df['Max_Ratio'].values)
model_l = LinearRegression().fit(X, hist_df['Min_Ratio'].values)

# 假设盘前最后价格即为今日大概率开盘价
pred_h_price = last_close * (1 + model_h.predict([[pre_change]])[0])
pred_l_price = last_close * (1 + model_l.predict([[pre_change]])[0])

# --- 3. UI 渲染 ---
st.title(f"🚀 BTDR 盘前异动监控系统")

# 顶部盘前状态栏
c1, c2, c3, c4 = st.columns(4)
c1.metric("盘前现价", f"${pre_last_price:.2f}", f"{pre_change:.2%)")
c2.metric("盘前换手率", f"{pre_turnover:.2f}%")
c3.metric("预测全天最高", f"${pred_h_price:.2f}")
c4.metric("预测全天最低", f"${pred_l_price:.2f}")

st.divider()

col_l, col_r = st.columns([2, 1])

with col_l:
    st.subheader("⏰ 盘前 1分钟 走势图")
    if not pre_df.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=pre_df.index, open=pre_df['Open'], high=pre_df['High'],
            low=pre_df['Low'], close=pre_df['Close'], name="盘前K线"
        )])
        fig.update_layout(xaxis_rangeslider_visible=False, height=400, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("当前非盘前时段或无成交数据")

with col_r:
    st.subheader("🚨 盘前异动分析")
    
    # 盘前逻辑判定
    analysis_points = []
    
    # 判定 A: 异常放量
    # 通常盘前换手率超过 1% 就算非常活跃
    if pre_turnover > 2.0:
        st.error("### 信号：盘前异常爆量")
        analysis_points.append("盘前换手率异常，主力在剧烈换手。")
    elif pre_turnover > 0.5:
        st.warning("### 信号：盘前交投活跃")
        analysis_points.append("活跃度高于平均水平。")
    
    # 判定 B: 盘前走势对全天影响
    if pre_change > 0.05 and pre_last_price >= pre_high * 0.98:
        analysis_points.append("盘前强势且收在最高点附近，开盘惯性冲高概率大。")
    elif pre_change < -0.05:
        analysis_points.append("盘前深幅跳空，需关注回归预测的底部支撑位。")

    for p in analysis_points:
        st.write(f"📌 {p}")

    st.divider()
    st.write("**今日关键点位参考：**")
    st.write(f"- 盘前高点：`${pre_high:.2f}`")
    st.write(f"- 盘前低点：`${pre_low:.2f}`")
    st.write(f"- 预测波动区间：`${pred_l_price:.2f} ~ ${pred_h_price:.2f}`")
