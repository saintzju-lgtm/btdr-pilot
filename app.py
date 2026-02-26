import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 数据抓取与预处理 ---
@st.cache_data(ttl=60)
def get_premarket_intel():
    ticker_symbol = "BTDR"
    # 获取 1d/1m 含盘前数据
    df = yf.download(ticker_symbol, period="1d", interval="1m", prepost=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 转换美东时间并截取盘前 (04:00 - 09:30)
    df.index = df.index.tz_convert('America/New_York')
    pre_df = df.between_time('04:00', '09:29').copy()
    
    # 计算基础指标
    info = yf.Ticker(ticker_symbol).info
    float_shares = info.get('floatShares', 35000000)
    
    return pre_df, float_shares

# --- 2. 背离识别逻辑 ---
def detect_divergence(df):
    if len(df) < 15:  # 数据不足时不分析
        return "等待更多盘前数据...", "gray"
    
    # 取最近 15 分钟的数据进行趋势对比
    recent = df.tail(15)
    price_trend = recent['Close'].iloc[-1] - recent['Close'].iloc[0]
    
    # 使用移动平均成交量判断量能趋势
    vol_sma_start = recent['Volume'].iloc[:5].mean()
    vol_sma_end = recent['Volume'].iloc[-5:].mean()
    vol_trend = vol_sma_end - vol_sma_start

    # 情况 A: 价升量缩 (看跌背离)
    if price_trend > 0 and vol_trend < 0:
        return "🚨 警惕：价涨量缩 (看跌背离)", "red"
    
    # 情况 B: 价跌量缩 (潜在买点)
    if price_trend < 0 and vol_trend < 0:
        return "📉 信号：价跌量缩 (抛压衰竭)", "orange"
    
    # 情况 C: 价升量增 (健康上涨)
    if price_trend > 0 and vol_trend > 0:
        return "🚀 强劲：价量齐升 (真实趋势)", "green"
    
    return "➡️ 状态：盘前波动较小", "gray"

# --- 3. UI 渲染 ---
st.title("🏹 BTDR 盘前智能量价终端")

pre_df, float_shares = get_premarket_intel()

if not pre_df.empty:
    # 状态判定
    status_msg, status_color = detect_divergence(pre_df)
    pre_vol = pre_df['Volume'].sum()
    pre_turnover = (pre_vol / float_shares) * 100
    
    # 顶部看板
    c1, c2, c3 = st.columns(3)
    c1.metric("盘前成交量", f"{pre_vol:,}")
    c2.metric("盘前换手率", f"{pre_turnover:.2f}%")
    c3.markdown(f"### 当前量价态势:\n:{status_color}[{status_msg}]")

    st.divider()

    # 画图：K线 + 成交量
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # K线
    fig.add_trace(go.Candlestick(x=pre_df.index, open=pre_df['Open'], high=pre_df['High'],
                                 low=pre_df['Low'], close=pre_df['Close'], name="1m K线"), row=1, col=1)
    # 成交量
    fig.add_trace(go.Bar(x=pre_df.index, y=pre_df['Volume'], name="成交量", 
                         marker_color='royalblue', opacity=0.5), row=2, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False, height=600, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # 操作建议
    st.subheader("💡 盘前操作建议")
    if "看跌背离" in status_msg:
        st.error("分析：盘前价格由虚火带动，缺乏实际买盘。建议：正式开盘后不要盲目追高，等待第一次回调支撑。")
    elif "价量齐升" in status_msg:
        st.success("分析：机构在盘前真实建仓。建议：关注 9:30 开盘后的放量突破机会。")
    elif pre_turnover > 10:
        st.warning(f"分析：换手率已达 {pre_turnover:.2f}%，日内波动极剧。建议：严格设置止损。")

else:
    st.info("目前暂无盘前成交数据。美股盘前通常在东部时间 04:00 开始。")
