import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 数据引擎 ---
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
st.title("🏹 BTDR 专业量化终端 (场景识别版)")

try:
    hist_df, live_df, float_shares, reg = get_btdr_final_data()
    last_hist = hist_df.iloc[-1]
    curr_p = live_df['Close'].iloc[-1]
    
    # 锁定今日开盘数据
    live_df.index = live_df.index.tz_convert('America/New_York')
    regular_market = live_df.between_time('09:30', '16:00')
    today_open = regular_market['Open'].iloc[0] if not regular_market.empty else live_df['Open'].iloc[-1]
    today_open_ratio = (today_open - last_hist['Close']) / last_hist['Close']

    # 计算中性预测 (a + b * open_ratio)
    p_h_mid = last_hist['Close'] * (1 + (reg['inter_h'] + reg['slope_h'] * today_open_ratio))
    p_l_mid = last_hist['Close'] * (1 + (reg['inter_l'] + reg['slope_l'] * today_open_ratio))
    
    # 定义场景边界 (±6%)
    p_h_opt = p_h_mid * 1.06
    p_l_pes = p_l_mid * 0.94
    
    today_turnover = (live_df['Volume'].sum() / float_shares) * 100

    # --- 核心：场景自动识别器 ---
    def detect_scenario(price, h_mid, l_mid, turnover):
        if price >= h_mid and turnover >= 10:
            return "乐观场景", "#00FF00", "突破中性压力，量能配合，目标看向乐观位上限。"
        elif price <= l_mid and turnover >= 15:
            return "悲观场景", "#FF4B4B", "击穿中性支撑，恐慌盘触发，正在下探悲观位。"
        else:
            return "中性场景", "#1E90FF", "运行于统计区间内，波动受回归线锚定。"

    scene_name, scene_color, scene_desc = detect_scenario(curr_p, p_h_mid, p_l_mid, today_turnover)

    # 1. 顶部指标与场景状态
    st.subheader(f"当前市场状态：:{scene_color}[{scene_name}]")
    st.info(f"**识别逻辑**：{scene_desc}")
    
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("当前实时价", f"${curr_p:.2f}", f"{(curr_p/last_hist['Close']-1):.2%}")
    p2.metric("中性最高 (H-Mid)", f"${p_h_mid:.2f}")
    p3.metric("中性最低 (L-Mid)", f"${p_l_mid:.2f}")
    t_status = "red" if today_turnover >= 20 else "orange" if today_turnover >= 10 else "green"
    p4.markdown(f"**实时换手率**\n### :{t_status}[{today_turnover:.2f}%]")

    st.divider()

    # 2. 形态建议与场景表
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("🤖 场景化形态解析")
        if scene_name == "乐观场景":
            st.success(f"🔥 **强势形态**：当前价格 `${curr_p:.2f}` 已站稳中性压力位。MA5 (${last_hist['5日均值']:.2f}) 形成支撑，建议持筹待涨。")
        elif scene_name == "悲观场景":
            st.error(f"⚠️ **弱势形态**：跌破中性支撑位。若换手率继续放大且无法收复 `${p_l_mid:.2f}`，需警惕下探 `${p_l_pes:.2f}`。")
        else:
            st.warning(f"⚖️ **震荡形态**：价格在回归区间内横盘。建议在 `${p_l_mid:.2f}` 附近低吸，`${p_h_mid:.2f}` 附近减仓。")

    with col_b:
        st.subheader("📊 完整预测场景表")
        sc_df = pd.DataFrame({
            "场景": ["乐观(+6%)", "中性(回归)", "悲观(-6%)"],
            "预测最高": [p_h_opt, p_h_mid, p_h_mid * 0.94],
            "预测最低": [p_l_mid * 1.06, p_l_mid, p_l_pes]
        })
        st.table(sc_df.style.format(precision=2))

    # 3. 可视化图表 (MM/DD 垂直坐标)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    plot_df = hist_df.tail(20).copy()
    plot_df['date_label'] = plot_df.index.strftime('%m/%d')

    fig.add_trace(go.Candlestick(x=plot_df['date_label'], open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name="K线"), row=1, col=1)
    
    # 动态压力支撑线
    fig.add_hline(y=p_h_mid, line_dash="dash", line_color="#1E90FF", annotation_text="中性压力", row=1, col=1)
    fig.add_hline(y=p_l_mid, line_dash="dash", line_color="#1E90FF", annotation_text="中性支撑", row=1, col=1)
    if scene_name == "乐观场景":
        fig.add_hline(y=p_h_opt, line_dash="dot", line_color="#00FF00", annotation_text="乐观目标", row=1, col=1)

    fig.add_trace(go.Bar(x=plot_df['date_label'], y=plot_df['换手率'], name="换手率", marker_color=['red' if x >= 20 else 'orange' if x >= 10 else 'gray' for x in plot_df['换手率']]), row=2, col=1)
    
    fig.update_xaxes(tickangle=-90, dtick=1, row=1, col=1)
    fig.update_xaxes(tickangle=-90, dtick=1, row=2, col=1)
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # 4. 底部历史数据
    st.subheader("📋 历史明细 (百分比格式)")
    show_df = hist_df.tail(10).copy()
    show_df.index = show_df.index.date
    for c in ['换手率', '今开比例', '最高比例', '最低比例']:
        show_df[c] = show_df[c].map('{:.2f}%'.format)
    st.dataframe(show_df[['Open', 'High', 'Low', 'Close', '换手率', '今开比例', '最高比例', '最低比例']].style.format(precision=2))

except Exception as e:
    st.error(f"计算逻辑刷新中: {e}")
