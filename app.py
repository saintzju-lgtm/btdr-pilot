import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 数据引擎 (实时拟合刷新) ---
@st.cache_data(ttl=60)
def get_btdr_final_data():
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
    
    # 动态执行线性回归
    fit_df = hist.dropna()
    X = fit_df[['今开比例']].values
    m_h = LinearRegression().fit(X, fit_df['最高比例'].values)
    m_l = LinearRegression().fit(X, fit_df['最低比例'].values)
    
    reg_params = {
        'slope_h': m_h.coef_[0], 'inter_h': m_h.intercept_,
        'slope_l': m_l.coef_[0], 'inter_l': m_l.intercept_
    }
    return fit_df, live_1m, float_shares, reg_params

# --- 2. 界面显示 ---
st.set_page_config(layout="wide")
st.title("🏹 BTDR 专业量化终端 (大形态+场景识别版)")

try:
    hist_df, live_df, float_shares, reg = get_btdr_final_data()
    last_hist = hist_df.iloc[-1]
    curr_p = live_df['Close'].iloc[-1]
    
    # 锁定今日开盘数据
    live_df.index = live_df.index.tz_convert('America/New_York')
    regular_market = live_df.between_time('09:30', '16:00')
    today_open = regular_market['Open'].iloc[0] if not regular_market.empty else live_df['Open'].iloc[-1]
    today_open_ratio = (today_open - last_hist['Close']) / last_hist['Close']

    # 计算中性预测线
    p_h_mid = last_hist['Close'] * (1 + (reg['inter_h'] + reg['slope_h'] * today_open_ratio))
    p_l_mid = last_hist['Close'] * (1 + (reg['inter_l'] + reg['slope_l'] * today_open_ratio))
    today_turnover = (live_df['Volume'].sum() / float_shares) * 100

    # --- 场景自动定位 (基于开盘预测位的偏离) ---
    def get_market_scene(p, h, l, vol):
        if p >= h * 1.01 and vol >= 10: return "乐观场景", "#00FF00", "价格有效突破中性压力，量能配合。"
        elif p <= l * 0.99 and vol >= 10: return "悲观场景", "#FF4B4B", "价格击穿中性支撑，空头占优。"
        else: return "中性场景", "#1E90FF", "价格在回归预测区间内运行，大形态稳健。"

    s_name, s_color, s_desc = get_market_scene(curr_p, p_h_mid, p_l_mid, today_turnover)

    # 1. 顶部场景与指标
    st.markdown(f"### 当前定位：:{s_color}[{s_name}] <small>({s_desc})</small>", unsafe_allow_html=True)
    
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("当前实时价", f"${curr_p:.2f}", f"{(curr_p/last_hist['Close']-1):.2%}")
    p2.metric("中性压力 (H-Mid)", f"${p_h_mid:.2f}")
    p3.metric("中性支撑 (L-Mid)", f"${p_l_mid:.2f}")
    t_status = "red" if today_turnover >= 20 else "orange" if today_turnover >= 10 else "green"
    p4.markdown(f"**当日累计换手**\n### :{t_status}[{today_turnover:.2f}%]")

    st.divider()

    # 2. 深度形态分析 (保持原有判断逻辑)
    col_analysis, col_table = st.columns([2, 1])
    
    with col_analysis:
        st.subheader("🤖 深度形态分析结论")
        analysis_points = []
        
        # 维度1：空间定位
        if curr_p >= p_h_mid * 0.98:
            analysis_points.append(f"🔴 **位置压力**：股价已触及/接近预测压力位 `${p_h_mid:.2f}`。需观察此处是否有大笔卖单抛压。")
        elif curr_p <= p_l_mid * 1.02:
            analysis_points.append(f"🟢 **位置支撑**：股价回落至预测支撑位 `${p_l_mid:.2f}` 附近，具备博弈反弹的价值。")
        
        # 维度2：均线趋势 (MA5)
        if curr_p > last_hist['5日均值']:
            analysis_points.append(f"📈 **趋势特征**：当前运行在5日均线 `${last_hist['5日均值']:.2f}` 之上，属于强势多头形态。")
        else:
            analysis_points.append(f"📉 **趋势特征**：受5日均线反压，重心有所下移，短期形态转入弱势调整。")

        # 维度3：量能异动
        if today_turnover >= 20:
            analysis_points.append(f"🔥 **极度放量**：换手率已达 {today_turnover:.2f}%。这种“死亡换手”在顶部通常是派发，在底部则是剧烈洗盘。")
        elif today_turnover >= 10:
            analysis_points.append(f"🟠 **活跃量能**：成交活跃，股价波动将大概率触及场景预测的边界。")

        # 维度4：盘前背离检测
        pre_m = live_df.between_time('04:00', '09:29')
        if not pre_m.empty and (pre_m['Close'].iloc[-1] > pre_m['Close'].iloc[0]) and (today_turnover < 2):
            analysis_points.append("⚠️ **潜在背离**：盘前虽有拉升但量能极其匮乏，警惕开盘后的诱多形态。")

        for point in analysis_points: st.write(point)

    with col_table:
        st.subheader("📊 场景预测明细")
        # 保持要求的 ±6% 推算
        scenario_data = {
            "场景": ["乐观(+6%)", "中性(回归)", "悲观(-6%)"],
            "最高价预测": [p_h_mid * 1.06, p_h_mid, p_h_mid * 0.94],
            "最低价预测": [p_l_mid * 1.06, p_l_mid, p_l_mid * 0.94]
        }
        st.table(pd.DataFrame(scenario_data).style.format(precision=2))

    # 3. 可视化图表 (时间轴垂直 MM/DD)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    plot_df = hist_df.tail(20).copy()
    plot_df['date_label'] = plot_df.index.strftime('%m/%d')

    fig.add_trace(go.Candlestick(x=plot_df['date_label'], open=plot_df['Open'], high=plot_df['High'], 
                                 low=plot_df['Low'], close=plot_df['Close'], name="K线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df['date_label'], y=plot_df['5日均值'], name="MA5", line=dict(color='yellow', width=1.5)), row=1, col=1)
    
    # 画出预测线
    fig.add_hline(y=p_h_mid, line_dash="dash", line_color="cyan", annotation_text="回归压力", row=1, col=1)
    fig.add_hline(y=p_l_mid, line_dash="dash", line_color="cyan", annotation_text="回归支撑", row=1, col=1)

    fig.add_trace(go.Bar(x=plot_df['date_label'], y=plot_df['换手率'], name="换手率",
                         marker_color=['red' if x >= 20 else 'orange' if x >= 10 else 'gray' for x in plot_df['换手率']]), row=2, col=1)

    fig.update_xaxes(tickangle=-90, dtick=1, row=1, col=1)
    fig.update_xaxes(tickangle=-90, dtick=1, row=2, col=1)
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # 4. 底部明细表
    st.subheader("📋 历史参考 (百分比格式)")
    show_df = hist_df.tail(12).copy()
    show_df.index = show_df.index.date
    for c in ['换手率', '今开比例', '最高比例', '最低比例']:
        show_df[c] = show_df[c].map('{:.2f}%'.format)
    st.dataframe(show_df[['Open', 'High', 'Low', 'Close', '换手率', '今开比例', '最高比例', '最低比例', '5日均值']].style.format(precision=2))

except Exception as e:
    st.error(f"分析模块自动刷新中: {e}")
