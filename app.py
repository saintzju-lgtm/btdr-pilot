import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 数据引擎 (动态刷新回归系数) ---
@st.cache_data(ttl=60)
def get_btdr_full_data():
    ticker = "BTDR"
    # 获取历史日线数据
    hist = yf.download(ticker, period="60d", interval="1d")
    if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
    
    # 获取实时分钟线
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
st.set_page_config(page_title="BTDR 量化分析终端", layout="wide")
st.title("🏹 BTDR 量化决策终端 (形态分析增强版)")

try:
    hist_df, live_df, float_shares, reg = get_btdr_full_data()
    last_hist = hist_df.iloc[-1]
    curr_p = live_df['Close'].iloc[-1]
    
    # 时间戳处理及开盘价锁定
    live_df.index = live_df.index.tz_convert('America/New_York')
    regular_market = live_df.between_time('09:30', '16:00')
    today_open = regular_market['Open'].iloc[0] if not regular_market.empty else live_df['Open'].iloc[-1]
    today_open_ratio = (today_open - last_hist['Close']) / last_hist['Close']

    # 基于最新拟合计算中性预测
    pred_h_neutral_r = reg['inter_h'] + reg['slope_h'] * today_open_ratio
    pred_l_neutral_r = reg['inter_l'] + reg['slope_l'] * today_open_ratio
    pred_h_neutral = last_hist['Close'] * (1 + pred_h_neutral_r)
    pred_l_neutral = last_hist['Close'] * (1 + pred_l_neutral_r)
    
    today_turnover = (live_df['Volume'].sum() / float_shares) * 100

    # --- 1. 核心指标卡 ---
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("当前价格", f"${curr_p:.2f}", f"{(curr_p/last_hist['Close']-1):.2%}")
    p2.metric("中性预测最高", f"${pred_h_neutral:.2f}", "压力位")
    p3.metric("中性预测最低", f"${pred_l_neutral:.2f}", "支撑位")
    t_color = "red" if today_turnover >= 20 else "orange" if today_turnover >= 10 else "green"
    p4.markdown(f"**当日累计换手**\n### :{t_color}[{today_turnover:.2f}%]")

    st.divider()

    # --- 2. 深度形态分析与场景表 ---
    analysis_col, scenario_col = st.columns([2, 1])
    
    with analysis_col:
        st.subheader("🤖 形态综合分析结论")
        analysis_points = []
        
        # 空间位置逻辑
        if curr_p >= pred_h_neutral * 0.98:
            analysis_points.append(f"🔴 **高位预警**：股价进入回归压力区 [${pred_h_neutral:.2f}]。若换手率超过15%，谨防主力高位对倒出货。")
        elif curr_p <= pred_l_neutral * 1.02:
            analysis_points.append(f"🟢 **低位支撑**：触及动态回归下沿 [${pred_l_neutral:.2f}]。若成交量萎缩，则是缩量洗盘的低吸信号。")
        
        # 均线与趋势协同
        if curr_p > last_hist['5日均值']:
            analysis_points.append(f"📈 **趋势特征**：站稳5日均线 (${last_hist['5日均值']:.2f})，短期动能向上，属于强势形态。")
        else:
            analysis_points.append(f"📉 **趋势特征**：受5日均线压制。股价若无法放量收复，恐将延续弱势震荡。")

        # 实时量价异动
        if today_turnover > 20:
            analysis_points.append("🔥 **量能状态**：换手率已触及“死亡预警线” (20%)。需高度警惕放量滞涨或放量大跌，筹码正在剧烈换手。")

        for pt in analysis_points: st.write(pt)

    with scenario_col:
        st.subheader("📈 预测场景推算表")
        # 按照表示例计算乐观/悲观偏离
        sc_data = {
            "场景描述": ["乐观场景 (+6%)", "中性场景", "悲观场景 (-6%)"],
            "预测最高": [pred_h_neutral * 1.06, pred_h_neutral, pred_h_neutral * 0.94],
            "预测最低": [pred_l_neutral * 1.06, pred_l_neutral, pred_l_neutral * 0.94]
        }
        st.table(pd.DataFrame(sc_data).style.format(precision=2))

    # --- 3. 实时主图 ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    plot_df = hist_df.tail(20)
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name="日K"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['5日均值'], name="MA5", line=dict(color='yellow')), row=1, col=1)
    
    # 绘制回归预测线
    fig.add_hline(y=pred_h_neutral, line_dash="dash", line_color="red", annotation_text="预测压力", row=1, col=1)
    fig.add_hline(y=pred_l_neutral, line_dash="dash", line_color="green", annotation_text="预测支撑", row=1, col=1)

    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['换手率'], name="换手率", 
                         marker_color=['red' if x >= 20 else 'orange' if x >= 10 else 'gray' for x in plot_df['换手率']]), row=2, col=1)
    
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # --- 4. 历史数据表 (日期精确到天，价格2位，比例百分号) ---
    st.subheader("📋 历史参考数据明细")
    show_df = hist_df.tail(15).copy()
    
    # 格式化日期和数值
    show_df.index = show_df.index.strftime('%Y-%m-%d')
    pct_cols = ['换手率', '今开比例', '最高比例', '最低比例']
    for col in pct_cols: show_df[col] = show_df[col].map('{:.2f}%'.format)
    
    st.dataframe(show_df[['Open', 'High', 'Low', 'Close', '昨收', 'Volume', '换手率', '今开比例', '最高比例', '最低比例']].style.format(
        precision=2, subset=['Open', 'High', 'Low', 'Close', '昨收']
    ))

except Exception as e:
    st.error(f"分析模块正在初始化或数据不足: {e}")
