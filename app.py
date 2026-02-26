import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# --- 页面配置 ---
st.set_page_config(page_title="BTDR 换手率与波动终端", layout="wide")

@st.cache_data(ttl=600)
def get_comprehensive_data():
    ticker_symbol = "BTDR"
    # 获取行情数据
    raw_df = yf.download(ticker_symbol, period="120d", interval="1d")
    
    # 修复 yfinance MultiIndex 问题
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)
    
    df = raw_df.copy()
    
    # 获取流通盘数据用于换手率 (yf.Ticker 较慢，建议缓存)
    t_info = yf.Ticker(ticker_symbol).info
    float_shares = t_info.get('floatShares', 35000000) # 若获取不到，默认给一个近似值
    
    # 基础计算
    df['Prev_Close'] = df['Close'].shift(1)
    df['Open_Ratio'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    df['Max_Ratio'] = (df['High'] - df['Prev_Close']) / df['Prev_Close']
    df['Min_Ratio'] = (df['Low'] - df['Prev_Close']) / df['Prev_Close']
    
    # 换手率计算
    df['Turnover_Rate'] = (df['Volume'] / float_shares) * 100
    df['MA5_Turnover'] = df['Turnover_Rate'].rolling(5).mean()
    
    return df.dropna(), float_shares

# --- 执行数据抓取 ---
try:
    df, float_shares = get_comprehensive_data()
    last_row = df.iloc[-1]
    current_turnover = last_row['Turnover_Rate']
    
    # 实时价获取
    live_price = yf.Ticker("BTDR").fast_info['last_price']
    prev_close = last_row['Close']
    today_open = yf.Ticker("BTDR").fast_info['open']
    if np.isnan(today_open): today_open = live_price
except Exception as e:
    st.error(f"数据加载异常: {e}")
    st.stop()

# --- 逻辑 (1): 回归预测 ---
X = df[['Open_Ratio']].values
model_h = LinearRegression().fit(X, df['Max_Ratio'].values)
model_l = LinearRegression().fit(X, df['Min_Ratio'].values)

today_ratio = (today_open - prev_close) / prev_close
pred_h = prev_close * (1 + model_h.predict([[today_ratio]])[0])
pred_l = prev_close * (1 + model_l.predict([[today_ratio]])[0])

# --- 逻辑 (2): 换手率预警颜色判断 ---
def get_turnover_color(val):
    if val >= 20: return "red", "🔥 极度过热 (风险极大)"
    if val >= 10: return "orange", "⚠️ 活跃放量 (警惕波动)"
    return "green", "✅ 成交平稳"

turnover_color, turnover_msg = get_turnover_color(current_turnover)

# --- UI 展示 ---
st.title("🏹 BTDR 实时量化监控: 换手率与波动预测")

# 顶层指标
c1, c2, c3, c4 = st.columns(4)
c1.metric("当前股价", f"${live_price:.2f}")
c2.metric("今日预测上限", f"${pred_h:.2f}")
c3.metric("今日预测下限", f"${pred_l:.2f}")
# 换手率 Metric 带颜色显示
st.sidebar.subheader("实时换_手率预警")
st.sidebar.markdown(f"### 当前换手率: :{turnover_color}[{current_turnover:.2f}%]")
st.sidebar.info(turnover_msg)

# 图表区
col_chart, col_advice = st.columns([2, 1])

with col_chart:
    # 1. 价格 K 线与预测区间
    fig_price = go.Figure()
    plot_df = df.tail(30)
    fig_price.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], 
                                      low=plot_df['Low'], close=plot_df['Close'], name="K线"))
    fig_price.add_hline(y=pred_h, line_dash="dash", line_color="red", annotation_text="预测压力")
    fig_price.add_hline(y=pred_l, line_dash="dash", line_color="green", annotation_text="预测支撑")
    fig_price.update_layout(title="价格趋势与今日波动范围预测", xaxis_rangeslider_visible=False, height=400)
    st.plotly_chart(fig_price, use_container_width=True)
    
    # 2. 换手率曲线图
    fig_turnover = go.Figure()
    fig_turnover.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Turnover_Rate'], 
                                     fill='tozeroy', name="日换手率", line_color="royalblue"))
    fig_turnover.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA5_Turnover'], 
                                     name="5日均换手", line_color="orange"))
    # 预警线
    fig_turnover.add_hline(y=10, line_dash="dot", line_color="orange", annotation_text="10% 活跃线")
    fig_turnover.add_hline(y=20, line_dash="dot", line_color="red", annotation_text="20% 警戒线")
    fig_turnover.update_layout(title="历史换手率波动监控", height=300)
    st.plotly_chart(fig_turnover, use_container_width=True)

with col_advice:
    st.subheader("💡 综合操作策略")
    
    # 判定分值
    score = 0
    reasons = []
    
    # 换手率逻辑
    if current_turnover > 20:
        score -= 2; reasons.append("换手率超20%，注意主力出货或情绪极值")
    elif current_turnover > 10:
        reasons.append("换手率超10%，市场热度高，波动加剧")
        
    # 位置逻辑
    if live_price > pred_h * 0.98:
        score -= 2; reasons.append("接近预测压力位，建议止盈")
    elif live_price < pred_l * 1.02:
        score += 2; reasons.append("接近预测支撑位，具备博弈价值")
        
    # 展示结果
    if score >= 1:
        st.success("### 操作建议：偏多")
    elif score <= -1:
        st.error("### 操作建议：偏空/减仓")
    else:
        st.warning("### 操作建议：持仓观望")
        
    st.write("**核心信号清单：**")
    for r in reasons:
        st.write(f"- {r}")

st.dataframe(df.tail(10))
