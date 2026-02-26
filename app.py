import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 核心分析函数：深度形态拆解 ---
def get_advanced_analysis(curr_p, p_high, p_low, turnover, hist_df, live_df):
    """
    结合量、换手、回归高低位、K线形态进行多维度综合评分
    """
    analysis_log = []
    risk_level = "中性"
    
    # A. 空间定位 (Space): 股价在预测区间的位置
    dist_to_high = (p_high - curr_p) / p_high
    dist_to_low = (curr_p - p_low) / p_low
    
    if curr_p >= p_high * 0.98:
        analysis_log.append("📍 **空间定位：极高位。** 股价已触及统计学回归压力位，继续向上空间受历史惯性压制。")
        score = -3
    elif curr_p <= p_low * 1.02:
        analysis_log.append("📍 **空间定位：支撑位。** 股价回落至预测波动下沿，具备技术性反弹的统计学基础。")
        score = 2
    else:
        analysis_log.append(f"📍 **空间定位：震荡区。** 处于预测区间 [${p_low:.2f} - ${p_high:.2f}] 中部，趋势不明朗。")
        score = 0

    # B. 量能热度 (Energy): 换手率级联分析
    if turnover >= 20:
        analysis_log.append("🔥 **量能状态：极度过热。** 换手率突破 20%，说明多空筹码剧烈换手。若在高位则是主力派发，在低位则是机构吸筹。")
        risk_level = "极高"
    elif turnover >= 10:
        analysis_log.append("🟠 **量能状态：高度活跃。** 10%-20% 的换手率代表市场关注度极高，日内波动将显著放大。")
        risk_level = "较高"
    else:
        analysis_log.append("🟢 **量能状态：温和。** 换手率处于正常水平，价格波动受散户情绪驱动较小。")

    # C. 量价协同 (Divergence): 实时动态背离
    # 获取最近 15 分钟的 1m 线趋势
    recent_15 = live_df.tail(15)
    if len(recent_15) >= 15:
        price_change = recent_15['Close'].iloc[-1] - recent_15['Close'].iloc[0]
        vol_trend = recent_15['Volume'].tail(5).mean() - recent_15['Volume'].head(5).mean()
        
        if price_change > 0 and vol_trend < 0:
            analysis_log.append("⚠️ **量价背离：缩量拉升。** 股价上涨但买盘动能衰减，盘中诱多风险增大。")
            score -= 2
        elif price_change < 0 and vol_trend < 0:
            analysis_log.append("📉 **量价协同：缩量回调。** 下跌时抛压同步减小，属于健康的洗盘形态。")
            score += 1

    # D. 5日均线 (Trend)
    ma5_val = hist_df['MA5'].iloc[-1]
    if curr_p > ma5_val:
        analysis_log.append(f"📈 **趋势特征：多头占优。** 当前价位于 5 日均线 (${ma5_val:.2f}) 之上，短期趋势向上。")
    else:
        analysis_log.append(f"📉 **趋势特征：空头压制。** 股价受 5 日均线反压，需关注下方支撑。")

    return analysis_log, score, risk_level

# --- 2. 界面展示逻辑 (部分展示) ---
# (假设前面已接入 yfinance 数据获取部分)

with st.sidebar:
    st.header("📊 实时决策报告")
    logs, final_score, risk = get_advanced_analysis(curr_p, pred_h, pred_l, today_turnover, hist_df, live_df)
    
    # 根据得分给出最终结论
    if final_score >= 2:
        st.success("🎯 **综合策略：建议试探性买入**")
    elif final_score <= -2:
        st.error("🎯 **综合策略：建议分批逢高减仓**")
    else:
        st.warning("🎯 **综合策略：建议继续观望**")

    st.write(f"**风险等级：{risk}**")
    st.divider()
    for log in logs:
        st.markdown(log)

# --- 3. 增强版图表渲染 ---
# 加入换手率预警线和回归预测带
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

# K线主图
fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], ...), row=1, col=1)
# 加上 5 日均线
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA5'], name="5日线"), row=1, col=1)
# 加上回归线
fig.add_hline(y=pred_h, line_dash="dash", line_color="red", row=1, col=1)
fig.add_hline(y=pred_l, line_dash="dash", line_color="green", row=1, col=1)

# 换手率预警图
fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Turnover'], ...), row=2, col=1)
# 加上 10% 和 20% 警戒线
fig.add_hline(y=10, line_dash="dot", line_color="orange", row=2, col=1)
fig.add_hline(y=20, line_dash="dot", line_color="red", row=2, col=1)
