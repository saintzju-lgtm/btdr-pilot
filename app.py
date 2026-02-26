import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 数据引擎 ---
@st.cache_data(ttl=60)
def get_btdr_data():
    ticker = "BTDR"
    hist = yf.download(ticker, period="60d", interval="1d")
    if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
    
    live_1m = yf.download(ticker, period="1d", interval="1m", prepost=True)
    if isinstance(live_1m.columns, pd.MultiIndex): live_1m.columns = live_1m.columns.get_level_values(0)
    
    # 模拟表中关键字段
    float_shares = 118000000 
    hist['Prev_Close'] = hist['Close'].shift(1)
    hist['MA5'] = hist['Close'].rolling(5).mean()
    hist['Turnover'] = (hist['Volume'] / float_shares) * 100
    
    # 回归特征计算
    hist['Open_R'] = (hist['Open'] - hist['Prev_Close']) / hist['Prev_Close']
    hist['Max_R'] = (hist['High'] - hist['Prev_Close']) / hist['Prev_Close']
    hist['Min_R'] = (hist['Low'] - hist['Prev_Close']) / hist['Prev_Close']
    
    return hist.dropna(), live_1m, float_shares

# --- 2. 深度分析函数 ---
def perform_deep_analysis(curr_p, p_h, p_l, turnover, ma5, live_df):
    logs = []
    score = 0
    
    # 维度1：空间定位
    if curr_p >= p_h * 0.98:
        logs.append("📍 **空间定位：极高位压力。** 股价已触及统计学回归天花板，上攻动能受限。")
        score -= 2
    elif curr_p <= p_l * 1.02:
        logs.append("📍 **空间定位：底部支撑区。** 股价回落至回归地板，具备波段反弹的统计学基础。")
        score += 2
    
    # 维度2：量能预警
    if turnover >= 20:
        logs.append(f"🔥 **量能警报：极度过热 ({turnover:.2f}%)。** 换手率突破20%警戒线。需观察：若在高位则是派发，在低位则是主力换手。")
    elif turnover >= 10:
        logs.append(f"🟠 **量能状态：高度活跃 ({turnover:.2f}%)。** 筹码交换频繁，日内震幅将显著放大。")

    # 维度3：量价背离 (取最近15分钟)
    if len(live_df) > 15:
        recent = live_df.tail(15)
        p_change = recent['Close'].iloc[-1] - recent['Close'].iloc[0]
        v_trend = recent['Volume'].tail(5).mean() - recent['Volume'].head(5).mean()
        if p_change > 0 and v_trend < 0:
            logs.append("⚠️ **量价特征：缩量拉升（诱多）。** 价格创新高但买盘动能衰减，随时可能反杀。")
            score -= 1
        elif p_change < 0 and v_trend < 0:
            logs.append("📉 **量价特征：缩量回调。** 抛压衰竭，属于良性洗盘，可关注支撑位博弈。")
            score += 1
            
    # 维度4：均线趋势
    if curr_p > ma5:
        logs.append(f"📈 **趋势协同：多头占优。** 站稳 5 日均线 (${ma5:.2f})，短期动能依然存在。")
    else:
        logs.append(f"📉 **趋势协同：空头反压。** 跌破 5 日线，短期重心下移，防守为主。")

    return logs, score

# --- 3. UI 界面 ---
st.title("🏹 BTDR 专业量化决策终端")

hist, live, f_shares = get_btdr_data()
curr_p = live['Close'].iloc[-1]
prev_c = hist['Close'].iloc[-1]

# 执行回归预测
X = hist[['Open_R']].values
m_h = LinearRegression().fit(X, hist['Max_R'].values)
m_l = LinearRegression().fit(X, hist['Min_R'].values)

# 今日预测 (基于开盘或当前价)
today_open = live.between_time('09:30', '16:00')['Open'].iloc[0] if not live.between_time('09:30', '16:00').empty else live['Open'].iloc[-1]
o_ratio = (today_open - prev_c) / prev_c
p_h = prev_c * (1 + m_h.predict([[o_ratio]])[0])
p_l = prev_c * (1 + m_l.predict([[o_ratio]])[0])
turnover = (live['Volume'].sum() / f_shares) * 100

# 侧边栏分析报告
st.sidebar.header("📋 深度形态报告")
analysis_logs, final_score = perform_deep_analysis(curr_p, p_h, p_l, turnover, hist['MA5'].iloc[-1], live)

if final_score >= 1: st.sidebar.success("🎯 **操作策略：建议逢低布局 / 持仓**")
elif final_score <= -1: st.sidebar.error("🎯 **操作策略：建议逢高止盈 / 避险**")
else: st.sidebar.warning("🎯 **操作策略：震荡行情，建议观望**")

for log in analysis_logs: st.sidebar.write(log)

# 主页标签页
tab1, tab2 = st.tabs(["📊 综合日线监控", "🕒 盘前/盘后异动 (折叠)"])

with tab1:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    # K线与均线
    plot_df = hist.tail(20)
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], 
                                 low=plot_df['Low'], close=plot_df['Close'], name="日线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA5'], name="5日线", line=dict(color='yellow')), row=1, col=1)
    # 预测线
    fig.add_hline(y=p_h, line_dash="dash", line_color="red", annotation_text="预测压力", row=1, col=1)
    fig.add_hline(y=p_l, line_dash="dash", line_color="green", annotation_text="预测支撑", row=1, col=1)
    # 换手率与预警线
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Turnover'], name="换手率",
                         marker_color=['red' if x >= 20 else 'orange' if x >= 10 else 'gray' for x in plot_df['Turnover']]), row=2, col=1)
    fig.add_hline(y=10, line_dash="dot", line_color="orange", row=2, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="red", row=2, col=1)
    
    fig.update_layout(height=650, xaxis_rangeslider_visible=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    with st.expander("🕒 盘前数据异动扫描 (04:00 - 09:30)"):
        pre_market = live.between_time('04:00', '09:29')
        if not pre_market.empty:
            st.metric("盘前成交量", f"{pre_market['Volume'].sum():,}")
            st.dataframe(pre_market.tail(10))
