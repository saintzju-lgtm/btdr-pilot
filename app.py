import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# --- 1. 页面配置 ---
st.set_page_config(page_title="BTDR 实时量化监控", layout="wide")

# --- 2. 核心数据获取 (yfinance API) ---
@st.cache_data(ttl=60)  # 缓存1分钟，保证实时性同时减少请求压力
def get_live_and_hist_data():
    ticker_symbol = "BTDR"
    # 获取历史 60 天日线数据用于回归拟合
    hist = yf.download(ticker_symbol, period="60d", interval="1d")
    if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
    
    # 获取今日实时数据 (含盘前)
    today_1m = yf.download(ticker_symbol, period="1d", interval="1m", prepost=True)
    if isinstance(today_1m.columns, pd.MultiIndex): today_1m.columns = today_1m.columns.get_level_values(0)
    
    # 基础信息 (流通股本按表中 1.18亿 计算)
    float_shares = 118000000 
    prev_close = hist['Close'].iloc[-2]  # 昨日收盘
    
    # 模拟表中计算字段
    hist['Prev_Close'] = hist['Close'].shift(1)
    hist['Open_Ratio'] = (hist['Open'] - hist['Prev_Close']) / hist['Prev_Close']
    hist['Max_Ratio'] = (hist['High'] - hist['Prev_Close']) / hist['Prev_Close']
    hist['Min_Ratio'] = (hist['Low'] - hist['Prev_Close']) / hist['Prev_Close']
    hist['Turnover'] = (hist['Volume'] / float_shares) * 100
    hist['MA5'] = hist['Close'].rolling(window=5).mean()
    
    return hist.dropna(), today_1m, float_shares, prev_close

# --- 3. 逻辑处理 ---
try:
    hist_df, live_df, float_shares, prev_close = get_live_and_hist_data()
    
    # 回归预测逻辑 (功能 1)
    X = hist_df[['Open_Ratio']].values
    y_h = hist_df['Max_Ratio'].values
    y_l = hist_df['Min_Ratio'].values
    model_h = LinearRegression().fit(X, y_h)
    model_l = LinearRegression().fit(X, y_l)
    
    # 确定当前状态
    curr_price = live_df['Close'].iloc[-1]
    # 区分盘前与盘中开盘价
    live_df.index = live_df.index.tz_convert('America/New_York')
    regular_market = live_df.between_time('09:30', '16:00')
    today_open = regular_market['Open'].iloc[0] if not regular_market.empty else live_df['Open'].iloc[-1]
    
    today_open_ratio = (today_open - prev_close) / prev_close
    pred_h = prev_close * (1 + model_h.predict([[today_open_ratio]])[0])
    pred_l = prev_close * (1 + model_l.predict([[today_open_ratio]])[0])
    
    # 今日换手率
    today_vol = live_df['Volume'].sum()
    today_turnover = (today_vol / float_shares) * 100

except Exception as e:
    st.error(f"API 获取失败: {e}")
    st.stop()

# --- 4. 界面布局 ---
st.title("🏹 BTDR 实时量化交易看板 (API 驱动)")

# 侧边栏：核心数据摘要
st.sidebar.header("实时行情摘要")
st.sidebar.metric("当前价", f"${curr_price:.2f}", f"{(curr_price/prev_close-1):.2%}")
st.sidebar.metric("今日预测高点", f"${pred_h:.2f}")
st.sidebar.metric("今日预测低点", f"${pred_l:.2f}")

# 主页标签
tab_main, tab_prepost = st.tabs(["📊 日线与操作决策", "🕒 盘前/盘后折叠"])

with tab_main:
    # 预警状态显示
    c1, c2 = st.columns([2, 1])
    
    with c1:
        # 换手率颜色逻辑 (功能 3)
        t_color = "red" if today_turnover >= 20 else "orange" if today_turnover >= 10 else "green"
        st.markdown(f"#### 今日实时换手率预警: :{t_color}[{today_turnover:.2f}%]")
        
        # 主图：K线 + 5日均线 + 预测带
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
        # 展示最近15天日线
        plot_df = hist_df.tail(15)
        fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
                                     low=plot_df['Low'], close=plot_df['Close'], name="日线"), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA5'], name="5日均值", line=dict(color='yellow')), row=1, col=1)
        
        # 预测虚线
        fig.add_hline(y=pred_h, line_dash="dash", line_color="red", annotation_text="预测压力", row=1, col=1)
        fig.add_hline(y=pred_l, line_dash="dash", line_color="green", annotation_text="预测支撑", row=1, col=1)
        
        # 换手率柱状图
        fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Turnover'], name="历史换手率", 
                             marker_color=['red' if x >= 20 else 'orange' if x >= 10 else 'gray' for x in plot_df['Turnover']]), row=2, col=1)
        
        fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # 功能 (2)：操作建议
        st.subheader("🤖 AI 操作建议")
        score = 0
        reasons = []
        
        # 位置建议
        if curr_price >= pred_h * 0.98:
            score -= 2; reasons.append("触及回归压力位")
        elif curr_price <= pred_l * 1.02:
            score += 2; reasons.append("接近回归支撑位")
            
        # 成交量建议
        if today_turnover > 15:
            reasons.append("今日成交显著放量")
            
        # 最终指令
        if score >= 2: st.success("### 建议操作：买入/加仓")
        elif score <= -2: st.error("### 建议操作：止盈/减仓")
        else: st.warning("### 建议操作：持仓观望")
        
        st.write("**分析逻辑：**")
        for r in reasons: st.write(f"- {r}")

    # 底部显示最近10日数据表格
    st.subheader("📋 最近10个交易日明细")
    st.dataframe(hist_df.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover', 'MA5']].style.format(precision=2))

with tab_prepost:
    # 功能：折叠查看盘前盘后
    with st.expander("🕒 盘前数据详情 (04:00 - 09:30)"):
        pre_market = live_df.between_time('04:00', '09:29')
        if not pre_market.empty:
            st.write(f"盘前最高: ${pre_market['High'].max():.2f}")
            st.write(f"盘前成交量: {pre_market['Volume'].sum():,}")
            # 背离识别逻辑
            p_change = pre_market['Close'].iloc[-1] - pre_market['Close'].iloc[0]
            v_trend = pre_market['Volume'].tail(5).mean() < pre_market['Volume'].head(5).mean()
            if p_change > 0 and v_trend:
                st.error("⚠️ 检测到盘前【价涨量缩】背离")
        else:
            st.write("暂无盘前交易数据")

    with st.expander("🌙 盘后数据详情 (16:00 - 20:00)"):
        after_market = live_df.between_time('16:00', '20:00')
        if not after_market.empty:
            st.dataframe(after_market.tail(10))
        else:
            st.write("尚未进入盘后交易时段")
