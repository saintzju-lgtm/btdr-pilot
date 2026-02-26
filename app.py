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
    
    reg_params = {'slope_h': m_h.coef_[0], 'inter_h': m_h.intercept_, 'slope_l': m_l.coef_[0], 'inter_l': m_l.intercept_}
    return fit_df, live_1m, float_shares, reg_params

# --- 2. 页面配置 ---
st.set_page_config(layout="wide", page_title="BTDR 深度量化终端")
st.title("🏹 BTDR 深度形态量化终端")

try:
    hist_df, live_df, float_shares, reg = get_btdr_final_data()
    last_hist = hist_df.iloc[-1]
    curr_p = live_df['Close'].iloc[-1]
    
    live_df.index = live_df.index.tz_convert('America/New_York')
    regular_market = live_df.between_time('09:30', '16:00')
    today_open = regular_market['Open'].iloc[0] if not regular_market.empty else live_df['Open'].iloc[-1]
    today_open_ratio = (today_open - last_hist['Close']) / last_hist['Close']

    # 核心计算：中性/乐观/悲观预测
    p_h_mid = last_hist['Close'] * (1 + (reg['inter_h'] + reg['slope_h'] * today_open_ratio))
    p_l_mid = last_hist['Close'] * (1 + (reg['inter_l'] + reg['slope_l'] * today_open_ratio))
    today_turnover = (live_df['Volume'].sum() / float_shares) * 100

    # 场景识别
    def get_market_scene(p, h, l, vol):
        if p >= h * 1.005 and vol >= 10: return "乐观场景", "#00FF00", "向上突破统计边界，量能支撑强劲。"
        elif p <= l * 0.995 and vol >= 10: return "悲观场景", "#FF4B4B", "向下击穿统计支撑，恐慌盘正在释放。"
        else: return "中性场景", "#1E90FF", "处于历史统计区间内，波动受回归线锚定。"

    s_name, s_color, s_desc = get_market_scene(curr_p, p_h_mid, p_l_mid, today_turnover)

    # --- 第一板块：场景预测位 (位置上移) ---
    st.markdown(f"""
        <div style="background-color: rgba(255,255,255,0.05); padding: 20px; border-radius: 10px; border-left: 10px solid {s_color}; margin-bottom: 20px;">
            <h2 style="margin:0;">当前定位：<span style="color:{s_color};">{s_name}</span></h2>
            <p style="margin:5px 0 0 0; color:#888;">{s_desc}</p>
        </div>
    """, unsafe_allow_html=True)

    col_target1, col_target2 = st.columns([1, 1])
    with col_target1:
        st.subheader("📍 场景股价预测目标")
        sc_df = pd.DataFrame({
            "场景描述": ["乐观上限 (+6%)", "中性压力 (H-Mid)", "中性支撑 (L-Mid)", "悲观下限 (-6%)"],
            "股价位置": [p_h_mid * 1.06, p_h_mid, p_l_mid, p_l_mid * 0.94]
        })
        st.table(sc_df.style.format({"股价位置": "{:.2f}"}))
    
    with col_target2:
        st.subheader("📊 实时核心指标")
        i1, i2 = st.columns(2)
        i1.metric("当前成交价", f"${curr_p:.2f}", f"{(curr_p/last_hist['Close']-1):.2%}")
        t_status = "red" if today_turnover >= 20 else "orange" if today_turnover >= 10 else "green"
        i2.markdown(f"**实时累计换手**\n### :{t_status}[{today_turnover:.2f}%]")

    st.divider()

    # --- 第二板块：形态分析与雷达解读 ---
    col_text, col_radar = st.columns([1.5, 1])

    with col_text:
        st.subheader("🤖 深度形态解析")
        
        # 深度逻辑判定
        analysis_points = []
        # 1. 量价配合分析
        if curr_p > last_hist['Close'] and today_turnover > 15:
            analysis_points.append(f"🟢 **放量拉升形态**：当前价格 `${curr_p:.2f}` 伴随 {today_turnover:.2f}% 的高换手。这通常意味着主力资金活跃度极高，若能站稳 `${p_h_mid:.2f}`，则今日有望冲击乐观位。")
        elif curr_p < last_hist['Close'] and today_turnover > 15:
            analysis_points.append(f"🔴 **放量下跌形态**：价格走弱且换手剧增，显示筹码正在剧烈派发。需严防跌破 `${p_l_mid:.2f}` 导致的恐慌盘溢出。")
        else:
            analysis_points.append(f"⚪ **缩量震荡形态**：换手率处于正常区间。股价大概率在 `${p_l_mid:.2f}` 到 `${p_h_mid:.2f}` 之间进行技术性休整。")

        # 2. 空间压力分析
        dist_to_h = abs(curr_p - p_h_mid) / p_h_mid
        if dist_to_h < 0.015:
            analysis_points.append(f"⚠️ **压力位警示**：股价极度接近预测压力位 `${p_h_mid:.2f}`。结合当前量能，若无持续大单买入，短线极易在此处遇阻回落。")
        
        # 3. 均线协同
        if curr_p > last_hist['5日均值'] * 1.05:
            analysis_points.append(f"🚀 **乖离过大**：当前价显著高于MA5（${last_hist['5日均值']:.2f}），短线存在超买修正需求，建议不宜在此时盲目追高。")
        elif curr_p < last_hist['5日均值']:
            analysis_points.append(f"📉 **趋势承压**：运行在MA5下方，趋势重心下移。每一轮反弹至 `${p_h_mid:.2f}` 附近都应视为减仓博弈点。")

        for p in analysis_points: st.markdown(f"> {p}")

    with col_radar:
        st.subheader("🎯 形态评分与解读")
        # 计算分值
        mom = min(max(((curr_p / today_open - 1) + 0.05) / 0.1 * 100, 0), 100)
        trd = min(max(((curr_p / last_hist['5日均值'] - 1) + 0.05) / 0.1 * 100, 0), 100)
        trn = min((today_turnover / 20) * 100, 100)
        sup = min(max((1 - abs(curr_p - p_l_mid) / p_l_mid) * 100, 0), 100)
        
        radar_fig = go.Figure(data=go.Scatterpolar(
            r=[mom, sup, trn, trd], theta=['动能(MOM)', '支撑(SUP)', '换手(TRN)', '趋势(TRD)'],
            fill='toself', fillcolor=f'rgba{tuple(int(s_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.3,)}',
            line=dict(color=s_color, width=2)
        ))
        radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=300, margin=dict(l=40, r=40, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(radar_fig, use_container_width=True)
        
        # 雷达维度解读
        radar_desc = []
        if trn > 75: radar_desc.append("🔸 **换手极高**：筹码交换剧烈，日内波动将远超平均水平。")
        if sup > 85: radar_desc.append("🔸 **支撑强劲**：价格极度贴合预测底线，具备统计学支撑。")
        if mom > 80: radar_desc.append("🔸 **动能爆发**：日内多头力量占据绝对统治地位。")
        for d in radar_desc: st.caption(d)

    # --- 第三板块：可视化图表 ---
    st.subheader("🕒 实时趋势监控 (垂直 MM/DD 坐标)")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    plot_df = hist_df.tail(20).copy()
    plot_df['label'] = plot_df.index.strftime('%m/%d')

    fig.add_trace(go.Candlestick(x=plot_df['label'], open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name="日K"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df['label'], y=plot_df['5日均值'], name="MA5", line=dict(color='yellow')), row=1, col=1)
    fig.add_hline(y=p_h_mid, line_dash="dash", line_color="cyan", annotation_text="预测压力", row=1, col=1)
    fig.add_hline(y=p_l_mid, line_dash="dash", line_color="cyan", annotation_text="预测支撑", row=1, col=1)
    fig.add_trace(go.Bar(x=plot_df['label'], y=plot_df['换手率'], name="换手率", marker_color=['red' if x >= 20 else 'orange' if x >= 10 else 'gray' for x in plot_df['换手率']]), row=2, col=1)
    fig.update_xaxes(tickangle=-90, dtick=1, row=1, col=1)
    fig.update_xaxes(tickangle=-90, dtick=1, row=2, col=1)
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # --- 第四板块：历史明细 ---
    st.subheader("📋 历史数据参考")
    show_df = hist_df.tail(12).copy()
    show_df.index = show_df.index.date
    for c in ['换手率', '今开比例', '最高比例', '最低比例']:
        show_df[c] = show_df[c].map('{:.2f}%'.format)
    st.dataframe(show_df[['Close', '换手率', '今开比例', '最高比例', '最低比例', '5日均值']].style.format(precision=2))

except Exception as e:
    st.error(f"引擎初始化中: {e}")
