import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 数据引擎 (动态刷新与回归) ---
@st.cache_data(ttl=60)
def get_btdr_full_data():
    ticker = "BTDR"
    hist = yf.download(ticker, period="60d", interval="1d")
    if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
    
    live_1m = yf.download(ticker, period="1d", interval="1m", prepost=True)
    if isinstance(live_1m.columns, pd.MultiIndex): live_1m.columns = live_1m.columns.get_level_values(0)
    
    float_shares = 118000000 # 基于图中流通股本 1.18亿
    hist['昨收'] = hist['Close'].shift(1)
    hist['今开比例'] = (hist['Open'] - hist['昨收']) / hist['昨收']
    hist['最高比例'] = (hist['High'] - hist['昨收']) / hist['昨收']
    hist['最低比例'] = (hist['Low'] - hist['昨收']) / hist['昨收']
    hist['换手率'] = (hist['Volume'] / float_shares) * 100
    hist['5日均值'] = hist['Close'].rolling(5).mean()
    
    # 执行动态回归拟合
    fit_df = hist.dropna()
    X = fit_df[['今开比例']].values
    m_h = LinearRegression().fit(X, fit_df['最高比例'].values)
    m_l = LinearRegression().fit(X, fit_df['最低比例'].values)
    
    reg_params = {
        'slope_h': m_h.coef_[0], 'inter_h': m_h.intercept_, 'r2_h': m_h.score(X, fit_df['最高比例'].values),
        'slope_l': m_l.coef_[0], 'inter_l': m_l.intercept_, 'r2_l': m_l.score(X, fit_df['最低比例'].values)
    }
    return fit_df, live_1m, float_shares, reg_params

# --- 2. 界面显示 ---
st.set_page_config(page_title="BTDR 量化分析终端", layout="wide")
st.title("🏹 BTDR 量化决策终端 (形态增强版)")

try:
    hist_df, live_df, float_shares, reg = get_btdr_full_data()
    last_hist = hist_df.iloc[-1]
    curr_p = live_df['Close'].iloc[-1]
    
    # 开盘价计算逻辑
    live_df.index = live_df.index.tz_convert('America/New_York')
    regular_market = live_df.between_time('09:30', '16:00')
    today_open = regular_market['Open'].iloc[0] if not regular_market.empty else live_df['Open'].iloc[-1]
    today_open_ratio = (today_open - last_hist['Close']) / last_hist['Close']

    # 预测中性场景
    pred_h_neutral = last_hist['Close'] * (1 + (reg['inter_h'] + reg['slope_h'] * today_open_ratio))
    pred_l_neutral = last_hist['Close'] * (1 + (reg['inter_l'] + reg['slope_l'] * today_open_ratio))
    today_turnover = (live_df['Volume'].sum() / float_shares) * 100

    # --- 1. 价格预测指标区 ---
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("当前实时价", f"${curr_p:.2f}", f"{(curr_p/last_hist['Close']-1):.2%}")
    p2.metric("动态最高预测", f"${pred_h_neutral:.2f}", "压力位")
    p3.metric("动态最低预测", f"${pred_l_neutral:.2f}", "支撑位")
    t_color = "red" if today_turnover >= 20 else "orange" if today_turnover >= 10 else "green"
    p4.markdown(f"**今日实时换手率**\n### :{t_color}[{today_turnover:.2f}%]")

    st.divider()

    # --- 2. 深度分析与场景推算 ---
    analysis_col, scenario_col = st.columns([2, 1])
    
    with analysis_col:
        st.subheader("🤖 形态综合分析结论")
        pts = []
        # A. 空间与形态 (结合K线影线)
        body = abs(curr_p - today_open)
        upper_shadow = last_hist['High'] - max(last_hist['Open'], last_hist['Close'])
        
        if curr_p >= pred_h_neutral * 0.98:
            pts.append(f"🔴 **位置建议**：股价触及回归压力区。若伴随上影线 > {upper_shadow:.2f}，则是典型的遇阻回落形态。")
        elif curr_p <= pred_l_neutral * 1.02:
            pts.append(f"🟢 **位置建议**：股价处于预测支撑位。若换手率未突破10%且缩量，则洗盘接近尾声。")
            
        # B. 趋势协同
        if curr_p > last_hist['5日均值']:
            pts.append(f"📈 **趋势状态**：当前运行在5日均线之上，属于强势多头。支撑参考 `${last_hist['5日均值']:.2f}`。")
        else:
            pts.append(f"📉 **趋势状态**：跌破5日均线，短期重心下移，谨防阴跌形态。")

        # C. 量能热度
        if today_turnover >= 20:
            pts.append("🔥 **量能警报**：换手率 > 20% 预示筹码剧烈松动。在高位通常是“击鼓传花”，在低位则是“恐慌盘出局”。")

        for p in pts: st.write(p)

    with scenario_col:
        st.subheader("📈 场景模拟预测")
        # 按照您的表示例推算 (±6% 偏离)
        sc_data = {
            "场景": ["乐观(+6%)", "中性", "悲观(-6%)"],
            "最高股价": [pred_h_neutral * 1.06, pred_h_neutral, pred_h_neutral * 0.94],
            "最低股价": [pred_l_neutral * 1.06, pred_l_neutral, pred_l_neutral * 0.94]
        }
        st.table(pd.DataFrame(sc_data).style.format(precision=2))

    # --- 3. 可视化图表 (主图保留) ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    plot_df = hist_df.tail(20)
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name="日K"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['5日均值'], name="MA5", line=dict(color='yellow')), row=1, col=1)
    fig.add_hline(y=pred_h_neutral, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=pred_l_neutral, line_dash="dash", line_color="green", row=1, col=1)
    
    # 换手率图
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['换手率'], name="换手率", marker_color=['red' if x >= 20 else 'orange' if x >= 10 else 'gray' for x in plot_df['换手率']]), row=2, col=1)
    fig.add_hline(y=10, line_dash="dot", line_color="orange", row=2, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="red", row=2, col=1)
    fig.update_layout(height=550, xaxis_rangeslider_visible=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # --- 4. 底部数据表 (百分比显示增强) ---
    st.subheader("📋 历史参考数据明细 (百分比格式)")
    show_df = hist_df.tail(10).copy()
    # 转换百分比格式显示
    pct_cols = ['换手率', '今开比例', '最高比例', '最低比例']
    for c in pct_cols: show_df[c] = show_df[c].map('{:.2f}%'.format)
    
    st.dataframe(show_df[['Open', 'High', 'Low', 'Close', '换手率', '今开比例', '最高比例', '最低比例', '5日均值']].style.applymap(
        lambda x: "background-color: #800000" if isinstance(x, str) and float(x.strip('%')) >= 20 else "", subset=['换手率']
    ))

    # --- 5. 盘前盘后 (折叠) ---
    with st.expander("🕒 盘前/盘后实时明细"):
        st.dataframe(live_df.tail(10))

except Exception as e:
    st.error(f"分析引擎刷新中... {e}")
