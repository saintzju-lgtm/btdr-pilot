import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 数据引擎 (包含历史与实时) ---
@st.cache_data(ttl=60)
def get_btdr_full_data():
    ticker = "BTDR"
    # 获取历史 60 天日线 (用于回归拟合)
    hist = yf.download(ticker, period="60d", interval="1d")
    if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
    
    # 获取今日 1m 实时 (含盘前)
    live_1m = yf.download(ticker, period="1d", interval="1m", prepost=True)
    if isinstance(live_1m.columns, pd.MultiIndex): live_1m.columns = live_1m.columns.get_level_values(0)
    
    # 流通股本参考图中数据 (约1.18亿)
    float_shares = 118000000 
    
    # 计算历史关键指标
    hist['昨收'] = hist['Close'].shift(1)
    hist['今开比例'] = (hist['Open'] - hist['昨收']) / hist['昨收']
    hist['最高比例'] = (hist['High'] - hist['昨收']) / hist['昨收']
    hist['最低比例'] = (hist['Low'] - hist['昨收']) / hist['昨收']
    hist['换手率'] = (hist['Volume'] / float_shares) * 100
    hist['5日均值'] = hist['Close'].rolling(5).mean()
    
    return hist.dropna(), live_1m, float_shares

# --- 2. 核心 UI 与逻辑 ---
st.title("🏹 BTDR 量化决策终端 (全功能集成版)")

try:
    hist_df, live_df, float_shares = get_btdr_full_data()
    last_hist = hist_df.iloc[-1]
    
    # A. 执行回归预测 (拟合最高/最低比例)
    X = hist_df[['今开比例']].values
    m_h = LinearRegression().fit(X, hist_df['最高比例'].values)
    m_l = LinearRegression().fit(X, hist_df['最低比例'].values)
    
    # B. 获取实时状态
    curr_p = live_df['Close'].iloc[-1]
    # 确定今日开盘价
    live_df.index = live_df.index.tz_convert('America/New_York')
    regular_market = live_df.between_time('09:30', '16:00')
    today_open = regular_market['Open'].iloc[0] if not regular_market.empty else live_df['Open'].iloc[-1]
    
    # C. 计算今日预测范围
    today_open_ratio = (today_open - last_hist['Close']) / last_hist['Close']
    pred_high = last_hist['Close'] * (1 + m_h.predict([[today_open_ratio]])[0])
    pred_low = last_hist['Close'] * (1 + m_l.predict([[today_open_ratio]])[0])
    
    # 今日实时累计换手
    today_turnover = (live_df['Volume'].sum() / float_shares) * 100

    # --- 1. 价格预测指标区 ---
    st.subheader("🎯 今日波动范围预测")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("当前实时价", f"${curr_p:.2f}", f"{(curr_p/last_hist['Close']-1):.2%}")
    p2.metric("今日预测最高", f"${pred_high:.2f}", "压力位", delta_color="inverse")
    p3.metric("今日预测最低", f"${pred_low:.2f}", "支撑位")
    
    t_color = "red" if today_turnover >= 20 else "orange" if today_turnover >= 10 else "green"
    p4.markdown(f"**实时换手率预警**\n### :{t_color}[{today_turnover:.2f}%]")

    st.divider()

    # --- 2. 综合形态分析结论 ---
    st.subheader("🤖 深度形态决策分析")
    
    analysis_col, advice_col = st.columns([2, 1])
    
    with analysis_col:
        # 维度拆解
        analysis_points = []
        
        # 空间定位
        if curr_p >= pred_high * 0.98:
            analysis_points.append(f"🔴 **高位风险**：当前价 `${curr_p:.2f}` 已触及回归压力位。结合历史数据，该位置抛压通常激增。")
        elif curr_p <= pred_low * 1.02:
            analysis_points.append(f"🟢 **低位支撑**：股价回落至回归支撑线 `${pred_low:.2f}`。若换手率未失控，具备博弈价值。")
        
        # 量能与换手
        if today_turnover >= 20:
            analysis_points.append(f"🔥 **极度放量**：换手率 ({today_turnover:.2f}%) 超过 20% 警戒线。需严防主力在高位“击鼓传花”或在低位“恐慌洗盘”。")
        elif today_turnover >= 10:
            analysis_points.append(f"🟠 **活跃放量**：市场博弈加剧，波动将显著偏离 5 日均值 (${last_hist['5日均值']:.2f})。")
            
        # 盘前量价背离 (从 live_df 提取)
        pre_market = live_df.between_time('04:00', '09:29')
        if not pre_market.empty:
            p_change = pre_market['Close'].iloc[-1] - pre_market['Close'].iloc[0]
            v_trend = pre_market['Volume'].tail(10).mean() < pre_market['Volume'].head(10).mean()
            if p_change > 0 and v_trend:
                analysis_points.append("⚠️ **量价背离**：检测到盘前“价涨量缩”。说明拉升缺乏资金真实承接，谨防开盘诱多。")

        for point in analysis_points:
            st.write(point)

    with advice_col:
        # 综合建议总结
        if curr_p >= pred_high * 0.98 and today_turnover > 15:
            st.error("### 综合建议：逢高减仓")
            st.write("理由：触及高位压力且换手过热，风险收益比极低。")
        elif curr_p <= pred_low * 1.02 and today_turnover < 10:
            st.success("### 综合建议：分批低吸")
            st.write("理由：缩量回踩预测支撑位，技术形态健康。")
        else:
            st.warning("### 综合建议：持仓观望")
            st.write("理由：处于震荡中轴，等待换手率或价格突破关键点位。")

    # --- 3. 可视化图表 ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    plot_df = hist_df.tail(20)
    
    # 主图 K线 + 5日均值
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
                                 low=plot_df['Low'], close=plot_df['Close'], name="日线K"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['5日均值'], name="5日线", line=dict(color='yellow')), row=1, col=1)
    # 预测压力/支撑线
    fig.add_hline(y=pred_high, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=pred_low, line_dash="dash", line_color="green", row=1, col=1)

    # 换手率柱状图 + 预警线
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['换手率'], name="换手率", 
                         marker_color=['red' if x >= 20 else 'orange' if x >= 10 else 'gray' for x in plot_df['换手率']]), row=2, col=1)
    fig.add_hline(y=10, line_dash="dot", line_color="orange", row=2, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="red", row=2, col=1)
    
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # --- 4. 底部数据表 (包含所有要求指标) ---
    st.subheader("📋 历史参考数据表 (最近 10 个交易日)")
    show_df = hist_df.tail(10)[['Open', 'High', 'Low', 'Close', '昨收', 'Volume', '换手率', '5日均值', '今开比例']]
    st.dataframe(show_df.style.format(precision=2).applymap(
        lambda x: "background-color: #800000" if isinstance(x, float) and x >= 20 else "", subset=['换手率']
    ))

    # --- 5. 盘前盘后折叠标签 ---
    st.divider()
    with st.expander("🕒 查看盘前/盘后实时明细数据"):
        col_pre, col_post = st.columns(2)
        with col_pre:
            st.write("**盘前 (Pre-Market)**")
            st.dataframe(pre_market.tail(5))
        with col_post:
            st.write("**盘后 (After-Hours)**")
            st.dataframe(live_df.between_time('16:00', '20:00').tail(5))

except Exception as e:
    st.error(f"数据加载异常: {e}")
