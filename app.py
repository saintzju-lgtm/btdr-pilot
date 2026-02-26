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
    
    # 执行动态回归
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
st.title("🏹 BTDR 专业量化终端 (视觉优化版)")

try:
    hist_df, live_df, float_shares, reg = get_btdr_final_data()
    last_hist = hist_df.iloc[-1]
    curr_p = live_df['Close'].iloc[-1]
    
    # 计算今日预测
    live_df.index = live_df.index.tz_convert('America/New_York')
    regular_market = live_df.between_time('09:30', '16:00')
    today_open = regular_market['Open'].iloc[0] if not regular_market.empty else live_df['Open'].iloc[-1]
    today_open_ratio = (today_open - last_hist['Close']) / last_hist['Close']

    pred_h_r = reg['inter_h'] + reg['slope_h'] * today_open_ratio
    pred_l_r = reg['inter_l'] + reg['slope_l'] * today_open_ratio
    pred_h = last_hist['Close'] * (1 + pred_h_r)
    pred_l = last_hist['Close'] * (1 + pred_l_r)
    today_turnover = (live_df['Volume'].sum() / float_shares) * 100

    # 1. 顶部指标
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("当前价", f"${curr_p:.2f}", f"{(curr_p/last_hist['Close']-1):.2%}")
    p2.metric("中性预测最高", f"${pred_h:.2f}")
    p3.metric("中性预测最低", f"${pred_l:.2f}")
    t_color = "red" if today_turnover >= 20 else "orange" if today_turnover >= 10 else "green"
    p4.markdown(f"**实时累计换手**\n### :{t_color}[{today_turnover:.2f}%]")

    st.divider()

    # 2. 形态分析与建议
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("🤖 深度形态解析")
        pts = []
        if curr_p >= pred_h * 0.98:
            pts.append(f"🔴 **压力确认**：股价触及回归压力位 `${pred_h:.2f}`。结合历史，若换手率此时剧增，则高位派发风险极大。")
        elif curr_p <= pred_l * 1.02:
            pts.append(f"🟢 **支撑确认**：股价回落至预测底线 `${pred_l:.2f}`。若成交量显著萎缩，视为缩量回踩，支撑有效。")
        
        if curr_p > last_hist['5日均值']:
            pts.append(f"📈 **趋势特征**：当前运行在5日均线（${last_hist['5日均值']:.2f}）上方，多头动能强劲。")
        else:
            pts.append(f"📉 **趋势特征**：短期受5日线反压，需防守 `${pred_l:.2f}` 支撑。")

        for p in pts: st.write(p)

    with col_b:
        st.subheader("📊 场景模拟预测")
        sc_df = pd.DataFrame({
            "场景": ["乐观(+6%)", "中性", "悲观(-6%)"],
            "预测最高": [pred_h * 1.06, pred_h, pred_h * 0.94],
            "预测最低": [pred_l * 1.06, pred_l, pred_l * 0.94]
        })
        st.table(sc_df.style.format(precision=2))

    # 3. 视觉优化后的图表
    st.subheader("🕒 走势与量能监控 (MM/DD 垂直坐标)")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    plot_df = hist_df.tail(22).copy() # 取约一个月数据
    
    # 转换日期格式用于显示
    plot_df['date_label'] = plot_df.index.strftime('%m/%d')

    fig.add_trace(go.Candlestick(x=plot_df['date_label'], open=plot_df['Open'], high=plot_df['High'], 
                                 low=plot_df['Low'], close=plot_df['Close'], name="K线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df['date_label'], y=plot_df['5日均值'], name="MA5", line=dict(color='yellow', width=1)), row=1, col=1)
    
    # 换手率柱状图
    fig.add_trace(go.Bar(x=plot_df['date_label'], y=plot_df['换手率'], name="换手率",
                         marker_color=['red' if x >= 20 else 'orange' if x >= 10 else 'gray' for x in plot_df['换手率']]), row=2, col=1)

    # 坐标轴美化
    fig.update_xaxes(tickangle=-90, dtick=1, tickformat='%m/%d', row=1, col=1)
    fig.update_xaxes(tickangle=-90, dtick=1, tickformat='%m/%d', row=2, col=1)
    fig.update_layout(height=650, xaxis_rangeslider_visible=False, template="plotly_dark", 
                      margin=dict(l=10, r=10, t=30, b=30))
    st.plotly_chart(fig, use_container_width=True)

    # 4. 底部历史数据明细
    st.subheader("📋 历史明细 (日期及数值规范化)")
    show_df = hist_df.tail(15).copy()
    show_df.index = show_df.index.date # 只保留Day
    
    # 百分数显示转换
    for c in ['换手率', '今开比例', '最高比例', '最低比例']:
        show_df[c] = show_df[c].map('{:.2f}%'.format)

    st.dataframe(show_df[['Open', 'High', 'Low', 'Close', '换手率', '今开比例', '最高比例', '最低比例', '5日均值']].style.format(
        precision=2, subset=['Open', 'High', 'Low', 'Close', '5日均值']
    ))

except Exception as e:
    st.error(f"数据处理中: {e}")
