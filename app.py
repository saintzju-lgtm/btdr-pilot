import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 数据引擎 ---
@st.cache_data(ttl=60)
def get_btdr_full_data():
    ticker = "BTDR"
    hist = yf.download(ticker, period="60d", interval="1d")
    if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
    
    live_1m = yf.download(ticker, period="1d", interval="1m", prepost=True)
    if isinstance(live_1m.columns, pd.MultiIndex): live_1m.columns = live_1m.columns.get_level_values(0)
    
    float_shares = 118000000 
    hist['昨收'] = hist['Close'].shift(1)
    hist['今开比例'] = (hist['Open'] - hist['昨收']) / hist['昨收']
    hist['最高比例'] = (hist['High'] - hist['昨收']) / hist['昨收']
    hist['最低比例'] = (hist['Low'] - hist['昨收']) / hist['昨收']
    hist['换手率'] = (hist['Volume'] / float_shares) * 100
    hist['5日均值'] = hist['Close'].rolling(5).mean()
    
    return hist.dropna(), live_1m, float_shares

# --- 2. 核心 UI 与逻辑 ---
st.title("🏹 BTDR 专业量化决策终端 (场景推算版)")

try:
    hist_df, live_df, float_shares = get_btdr_full_data()
    last_hist = hist_df.iloc[-1]
    
    # 获取实时价格
    curr_p = live_df['Close'].iloc[-1]
    live_df.index = live_df.index.tz_convert('America/New_York')
    regular_market = live_df.between_time('09:30', '16:00')
    today_open = regular_market['Open'].iloc[0] if not regular_market.empty else live_df['Open'].iloc[-1]
    
    # --- A. 核心场景预测逻辑 ---
    # 根据提供的公式拟合图
    # 最高比例 = 0.04052 + 1.033 * 今开比例
    # 最低比例 = -0.03777 + 1.009 * 今开比例
    today_open_ratio = (today_open - last_hist['Close']) / last_hist['Close']
    
    # 1. 中性场景 (核心回归点)
    pred_h_ratio_neutral = 0.04052 + 1.033 * today_open_ratio
    pred_l_ratio_neutral = -0.03777 + 1.009 * today_open_ratio
    
    # 2. 场景偏离度 (基于提供的 6% 场景波动率)
    # 乐观场景 = 中性 + 6% | 悲观场景 = 中性 - 6%
    
    scenarios = {
        "中性场景": {
            "high": last_hist['Close'] * (1 + pred_h_ratio_neutral),
            "low": last_hist['Close'] * (1 + pred_l_ratio_neutral),
            "color": "#1E90FF"
        },
        "乐观场景": {
            "high": (last_hist['Close'] * (1 + pred_h_ratio_neutral)) * 1.06,
            "low": (last_hist['Close'] * (1 + pred_l_ratio_neutral)) * 1.06,
            "color": "#00FF00"
        },
        "悲观场景": {
            "high": (last_hist['Close'] * (1 + pred_h_ratio_neutral)) * 0.94,
            "low": (last_hist['Close'] * (1 + pred_l_ratio_neutral)) * 0.94,
            "color": "#FF4B4B"
        }
    }

    # --- 1. 场景预测显示区 ---
    st.subheader("🎯 三维度空间场景预测 (基于拟合公式)")
    cols = st.columns(3)
    for i, (name, val) in enumerate(scenarios.items()):
        with cols[i]:
            st.markdown(f"#### :{val['color']}[{name}]")
            st.write(f"预测最高：**${val['high']:.2f}**")
            st.write(f"预测最低：**${val['low']:.2f}**")

    st.divider()

    # --- 2. 图表渲染 (加入场景带) ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    plot_df = hist_df.tail(20)
    
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
                                 low=plot_df['Low'], close=plot_df['Close'], name="日K"), row=1, col=1)
    
    # 在图表上画出三个场景的压力线
    fig.add_hline(y=scenarios["乐观场景"]["high"], line_dash="dot", line_color="#00FF00", annotation_text="乐观上限", row=1, col=1)
    fig.add_hline(y=scenarios["中性场景"]["high"], line_dash="dash", line_color="#1E90FF", annotation_text="回归中值", row=1, col=1)
    fig.add_hline(y=scenarios["悲观场景"]["low"], line_dash="dot", line_color="#FF4B4B", annotation_text="悲观下限", row=1, col=1)

    # 换手率 Bar
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['换手率'], name="换手率 (%)", 
                         marker_color=['red' if x >= 20 else 'orange' if x >= 10 else 'gray' for x in plot_df['换手率']]), row=2, col=1)
    
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # --- 3. 历史数据表 (带百分数格式) ---
    st.subheader("📋 历史数据明细 (百分比格式)")
    
    # 格式化 DataFrame
    show_df = hist_df.tail(15).copy()
    
    # 转换百分比列显示格式
    show_df['换手率'] = show_df['换手率'].map('{:.2f}%'.format)
    show_df['今开比例'] = show_df['今开比例'].map('{:.2%}'.format)
    show_df['最高比例'] = show_df['最高比例'].map('{:.2%}'.format)
    show_df['最低比例'] = show_df['最低比例'].map('{:.2%}'.format)
    
    st.dataframe(show_df[['Open', 'High', 'Low', 'Close', '昨收', 'Volume', '换手率', '今开比例', '最高比例', '最低比例']])

    # --- 4. 拟合逻辑回顾 ---
    st.divider()
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### 📈 最高比例回归拟合")
        st.latex(r"High\_Ratio = 0.04052 + 1.033 \times Open\_Ratio")
        st.caption("R²: 0.556 | F: 82.57")
    with col_r:
        st.markdown("### 📉 最低比例回归拟合")
        st.latex(r"Low\_Ratio = -0.03777 + 1.009 \times Open\_Ratio")
        st.caption("R²: 0.554 | F: 82.07")

except Exception as e:
    st.error(f"系统运行错误: {e}")
