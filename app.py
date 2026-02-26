import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 数据引擎 (核心模型与拟合) ---
@st.cache_data(ttl=60)
def get_btdr_final_data():
    ticker = "BTDR"
    # 获取历史日线数据用于拟合 (60天)
    hist = yf.download(ticker, period="60d", interval="1d")
    if isinstance(hist.columns, pd.MultiIndex): 
        hist.columns = hist.columns.get_level_values(0)
    
    # 获取实时分钟线 (含盘前盘后)
    live_1m = yf.download(ticker, period="1d", interval="1m", prepost=True)
    if isinstance(live_1m.columns, pd.MultiIndex): 
        live_1m.columns = live_1m.columns.get_level_values(0)
    
    float_shares = 118000000 # 流通股本
    hist['昨收'] = hist['Close'].shift(1)
    hist['今开比例'] = (hist['Open'] - hist['昨收']) / hist['昨收']
    hist['最高比例'] = (hist['High'] - hist['昨收']) / hist['昨收']
    hist['最低比例'] = (hist['Low'] - hist['昨收']) / hist['昨收']
    hist['换手率'] = (hist['Volume'] / float_shares) * 100
    hist['5日均值'] = hist['Close'].rolling(5).mean()
    
    # 动态线性回归计算 a 和 b
    fit_df = hist.dropna()
    X = fit_df[['今开比例']].values
    m_h = LinearRegression().fit(X, fit_df['最高比例'].values)
    m_l = LinearRegression().fit(X, fit_df['最低比例'].values)
    
    reg_params = {
        'slope_h': m_h.coef_[0], 'inter_h': m_h.intercept_,
        'slope_l': m_l.coef_[0], 'inter_l': m_l.intercept_
    }
    return fit_df, live_1m, float_shares, reg_params

# --- 2. 页面配置与样式 ---
st.set_page_config(layout="wide", page_title="BTDR 量化终端")
st.title("🏹 BTDR 专业量化终端 (形态+雷达全功能版)")

try:
    hist_df, live_df, float_shares, reg = get_btdr_final_data()
    last_hist = hist_df.iloc[-1]
    curr_p = live_df['Close'].iloc[-1]
    
    # 锁定今日开盘价 (美东时间)
    live_df.index = live_df.index.tz_convert('America/New_York')
    regular_market = live_df.between_time('09:30', '16:00')
    today_open = regular_market['Open'].iloc[0] if not regular_market.empty else live_df['Open'].iloc[-1]
    today_open_ratio = (today_open - last_hist['Close']) / last_hist['Close']

    # 基于开盘比例计算中性场景位 (a + b * ratio)
    p_h_mid = last_hist['Close'] * (1 + (reg['inter_h'] + reg['slope_h'] * today_open_ratio))
    p_l_mid = last_hist['Close'] * (1 + (reg['inter_l'] + reg['slope_l'] * today_open_ratio))
    today_turnover = (live_df['Volume'].sum() / float_shares) * 100

    # --- 场景识别逻辑 ---
    def get_market_scene(p, h, l, vol):
        if p >= h * 1.01 and vol >= 10: 
            return "乐观场景", "#00FF00", "价格有效突破中性压力，量能配合，多头极强。"
        elif p <= l * 0.99 and vol >= 10: 
            return "悲观场景", "#FF4B4B", "价格击穿中性支撑，恐慌抛售压力较大。"
        else: 
            return "中性场景", "#1E90FF", "价格在回归预测区间内运行，大形态保持稳健。"

    s_name, s_color, s_desc = get_market_scene(curr_p, p_h_mid, p_l_mid, today_turnover)

    # 1. 顶部场景渲染 (强制使用 HTML 颜色)
    st.markdown(f"""
        <div style="background-color: rgba(255,255,255,0.05); padding: 18px; border-radius: 10px; border-left: 10px solid {s_color}; margin-bottom: 25px;">
            <h2 style="margin:0; font-size: 26px;">当前定位：<span style="color:{s_color};">{s_name}</span></h2>
            <p style="margin:8px 0 0 0; color:#AAAAAA; font-size: 16px;">{s_desc}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # 指标卡
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("当前实时价", f"${curr_p:.2f}", f"{(curr_p/last_hist['Close']-1):.2%}")
    c2.metric("中性压力 (H-Mid)", f"${p_h_mid:.2f}")
    c3.metric("中性支撑 (L-Mid)", f"${p_l_mid:.2f}")
    t_color = "red" if today_turnover >= 20 else "orange" if today_turnover >= 10 else "green"
    c4.markdown(f"**实时换手率**\n### :{t_color}[{today_turnover:.2f}%]")

    st.divider()

    # 2. 深度形态分析 (文字) 与 形态雷达图 (可视化)
    col_left, col_right = st.columns([1.6, 1])

    with col_left:
        st.subheader("🤖 深度形态分析结论")
        analysis_txt = []
        # 空间位置
        if curr_p >= p_h_mid * 0.98:
            analysis_txt.append(f"🔴 **高位压力**：股价已逼近回归压力 `${p_h_mid:.2f}`。若换手持续放大但价格滞涨，需防诱多。")
        elif curr_p <= p_l_mid * 1.02:
            analysis_txt.append(f"🟢 **支撑确认**：股价处于支撑位 `${p_l_mid:.2f}` 附近。若量能萎缩，通常为缩量洗盘的低吸信号。")
        # 均线形态
        if curr_p > last_hist['5日均值']:
            analysis_txt.append(f"📈 **趋势特征**：站稳5日线 `${last_hist['5日均值']:.2f}` 上方，短期重心上移，大形态向好。")
        else:
            analysis_txt.append(f"📉 **趋势特征**：受5日均线压制。股价若不能放量反抽，恐将向支撑位 `${p_l_mid:.2f}` 靠拢。")
        # 量能异动
        if today_turnover >= 15:
            analysis_txt.append(f"🔥 **活跃警报**：换手率已达 {today_turnover:.2f}%。主力资金介入深，波动将剧烈脱离中位区间。")
        
        for t in analysis_txt: st.write(t)

    with col_right:
        st.subheader("🎯 形态雷达评分")
        # 计算 0-100 分值
        m_score = min(max(((curr_p / today_open - 1) + 0.05) / 0.1 * 100, 0), 100) # 动能
        t_score = min(max(((curr_p / last_hist['5日均值'] - 1) + 0.05) / 0.1 * 100, 0), 100) # 趋势
        v_score = min((today_turnover / 20) * 100, 100) # 换手
        s_score = min(max((1 - abs(curr_p - p_l_mid) / p_l_mid) * 100, 0), 100) # 支撑
        
        radar_fig = go.Figure()
        radar_fig.add_trace(go.Scatterpolar(
            r=[m_score, s_score, v_score, t_score],
            theta=['动能(MOM)', '支撑(SUP)', '换手(TRN)', '趋势(TRD)'],
            fill='toself',
            fillcolor=f'rgba{tuple(int(s_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.3,)}',
            line=dict(color=s_color, width=2)
        ))
        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor="#444")),
            showlegend=False, height=350, margin=dict(l=50, r=50, t=30, b=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(radar_fig, use_container_width=True)

    # 3. K线与量能图 (MM/DD 垂直)
    st.subheader("🕒 实时走势监控")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    plot_df = hist_df.tail(20).copy()
    plot_df['label'] = plot_df.index.strftime('%m/%d')

    fig.add_trace(go.Candlestick(x=plot_df['label'], open=plot_df['Open'], high=plot_df['High'], 
                                 low=plot_df['Low'], close=plot_df['Close'], name="日K"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df['label'], y=plot_df['5日均值'], name="MA5", line=dict(color='yellow')), row=1, col=1)
    
    # 绘制回归预测水平线
    fig.add_hline(y=p_h_mid, line_dash="dash", line_color="cyan", annotation_text="回归压力", row=1, col=1)
    fig.add_hline(y=p_l_mid, line_dash="dash", line_color="cyan", annotation_text="回归支撑", row=1, col=1)

    fig.add_trace(go.Bar(x=plot_df['label'], y=plot_df['换手率'], name="换手率",
                         marker_color=['red' if x >= 20 else 'orange' if x >= 10 else 'gray' for x in plot_df['换手率']]), row=2, col=1)
    
    fig.update_xaxes(tickangle=-90, dtick=1, row=1, col=1)
    fig.update_xaxes(tickangle=-90, dtick=1, row=2, col=1)
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # 4. 场景推算与历史表
    st.subheader("📋 详细预测场景与历史明细")
    col_hist, col_scene = st.columns([2, 1])
    
    with col_hist:
        show_df = hist_df.tail(10).copy()
        show_df.index = show_df.index.date
        for c in ['换手率', '今开比例', '最高比例', '最低比例']:
            show_df[c] = show_df[c].map('{:.2f}%'.format)
        st.dataframe(show_df[['Open', 'High', 'Low', 'Close', '换手率', '今开比例', '最高比例', '最低比例']].style.format(precision=2))
        
    with col_scene:
        scene_data = {
            "场景描述": ["乐观(+6%)", "中性(回归)", "悲观(-6%)"],
            "预测最高": [p_h_mid * 1.06, p_h_mid, p_h_mid * 0.94],
            "预测最低": [p_l_mid * 1.06, p_l_mid, p_l_mid * 0.94]
        }
        st.table(pd.DataFrame(scene_data).style.format(precision=2))

except Exception as e:
    st.error(f"数据加载中或发生错误: {e}")
