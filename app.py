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

# --- 2. 界面显示 ---
st.set_page_config(layout="wide", page_title="BTDR 深度智能终端")
st.title("🏹 BTDR 深度形态量化终端 (AI 深度解析版)")

try:
    hist_df, live_df, float_shares, reg = get_btdr_final_data()
    last_hist = hist_df.iloc[-1]
    curr_p = live_df['Close'].iloc[-1]
    ma5_curr = last_hist['5日均值']
    
    live_df.index = live_df.index.tz_convert('America/New_York')
    regular_market = live_df.between_time('09:30', '16:00')
    today_open = regular_market['Open'].iloc[0] if not regular_market.empty else live_df['Open'].iloc[-1]
    today_open_ratio = (today_open - last_hist['Close']) / last_hist['Close']

    # 场景计算
    p_h_mid = last_hist['Close'] * (1 + (reg['inter_h'] + reg['slope_h'] * today_open_ratio))
    p_l_mid = last_hist['Close'] * (1 + (reg['inter_l'] + reg['slope_l'] * today_open_ratio))
    today_turnover = (live_df['Volume'].sum() / float_shares) * 100

    # 场景识别
    def get_market_scene(p, h, l, vol):
        if p >= h * 1.005 and vol >= 10: return "乐观场景", "#00FF00", "价格突破统计边界，向上空间打开。"
        elif p <= l * 0.995 and vol >= 10: return "悲观场景", "#FF4B4B", "击穿统计支撑位，下行惯性较强。"
        else: return "中性场景", "#1E90FF", "波动受回归线锚定，维持区间震荡。"

    s_name, s_color, s_desc = get_market_scene(curr_p, p_h_mid, p_l_mid, today_turnover)

    # --- 板块 1：实时状态与场景预测 ---
    st.markdown(f"""
        <div style="background-color: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border-left: 10px solid {s_color}; margin-bottom: 20px;">
            <h2 style="margin:0;">当前定位：<span style="color:{s_color};">{s_name}</span></h2>
            <p style="margin:5px 0 0 0; color:#888;">{s_desc}</p>
        </div>
    """, unsafe_allow_html=True)

    col_metric, col_target = st.columns([1, 1.5])
    with col_metric:
        st.subheader("📊 实时状态")
        m1, m2 = st.columns(2)
        m1.metric("当前价", f"${curr_p:.2f}", f"{(curr_p/last_hist['Close']-1):.2%}")
        t_status = "red" if today_turnover >= 20 else "orange" if today_turnover >= 10 else "green"
        m2.markdown(f"**实时换手率**\n### :{t_status}[{today_turnover:.2f}%]")

    with col_target:
        st.subheader("📍 场景股价预测目标")
        scenario_table = pd.DataFrame({
            "场景": ["中性场景 (回归拟合)", "乐观场景 (+6% 偏移)", "悲观场景 (-6% 偏移)"],
            "最高股价预测": [p_h_mid, p_h_mid * 1.06, p_h_mid * 0.94],
            "最低股价预测": [p_l_mid, p_l_mid * 1.06, p_l_mid * 0.94]
        })
        st.table(scenario_table.style.format(precision=2))

    st.divider()

    # --- 板块 2：AI 形态分析引擎 (重点优化) ---
    col_text, col_radar = st.columns([1.6, 0.9])

    with col_text:
        st.subheader("🤖 智能形态深度解析")
        
        # 1. 形态特征
        trend_desc = "震荡上行" if curr_p > ma5_curr else "震荡下跌"
        ma5_str = "站稳 MA5" if curr_p >= ma5_curr else "受 MA5 压制"
        loc_str = "测试‘预测最低’支撑" if abs(curr_p - p_l_mid) < abs(curr_p - p_h_mid) else "挑战‘预测最高’压力"
        
        st.markdown(f"**🔍 形态特征**")
        st.write(f"当前股价处于 **{trend_desc}** 阶段，K 线形态表现为 **{ma5_str}**（约 {ma5_curr:.2f}）。"
                 f"目前股价正在 **{loc_str}**（约 {min(p_h_mid, p_l_mid, key=lambda x:abs(x-curr_p)):.2f}）的关键转换位。")

        # 2. 量价配合
        prev_turnover = hist_df.iloc[-2]['换手率']
        vol_change = "缩量" if today_turnover < prev_turnover else "放量"
        vol_status = "由热转冷" if (prev_turnover > 15 and today_turnover < 10) else "持续活跃"
        st.markdown(f"**📊 量价配合**")
        st.write(f"换手率呈现 **{vol_change}** 态势，从前期高点 **{vol_status}**。当前 {today_turnover:.2f}% 的活跃度暗示"
                 f"{'底部筹码正在完成交换' if vol_change == '缩量' else '多空双方博弈仍趋于白热化'}。")

        # 3. 市场心理
        st.markdown(f"**🧠 市场心理**")
        if today_turnover < 8:
            psych = "恐慌盘已基本洗出，市场进入‘缩量观望’状态，抛压极轻，处于等待方向选择的真空期。"
        elif today_turnover > 18:
            psych = "市场情绪处于极度亢奋或恐慌的临界点，底部筹码与获利筹码正在剧烈换手，预期将有大级别波动。"
        else:
            psych = "多空情绪相对均衡，投资者正在锚定‘预测位’进行心理博弈，震荡筑底意图明显。"
        st.write(psych)

        # 4. 后市操作建议 (动态逻辑)
        st.markdown(f"**💡 后市操作建议**")
        if curr_p <= p_l_mid * 1.01 and today_turnover < 10:
            advice = "当前具备极高‘缩量踩支’特征。建议在支撑位附近分批吸纳，博弈向中轴回归。跌破悲观线即止损。"
        elif curr_p >= p_h_mid * 0.99 and today_turnover > 15:
            advice = "属于‘放量触压’形态。当前价位抛压巨大，建议先行兑现部分利润，待回踩 MA5 企稳后再行介入。"
        else:
            advice = "形态处于区间中段，建议持筹观望，不宜在此时点盲目加仓，重点关注股价对 MA5 的争夺。"
        st.success(advice)

    with col_radar:
        st.subheader("🎯 实时评分雷达")
        mom = min(max(((curr_p / today_open - 1) + 0.05) / 0.1 * 100, 0), 100)
        sup = min(max((1 - abs(curr_p - p_l_mid) / p_l_mid) * 100, 0), 100)
        trn = min((today_turnover / 20) * 100, 100)
        trd = min(max(((curr_p / ma5_curr - 1) + 0.05) / 0.1 * 100, 0), 100)
        
        radar_fig = go.Figure(data=go.Scatterpolar(
            r=[mom, sup, trn, trd], theta=['动能(MOM)', '支撑(SUP)', '换手(TRN)', '趋势(TRD)'],
            fill='toself', fillcolor=f'rgba{tuple(int(s_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.3,)}',
            line=dict(color=s_color, width=2)
        ))
        radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=320, margin=dict(l=40, r=40, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(radar_fig, use_container_width=True)
        st.caption(f"🔸 支撑强弱: {int(sup)} | 换手热度: {int(trn)} | 趋势乖离: {int(trd)}")

    # --- 板块 3：主走势图 (MM/DD 垂直坐标) ---
    st.subheader("🕒 趋势监控图表")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    plot_df = hist_df.tail(20).copy()
    plot_df['label'] = plot_df.index.strftime('%m/%d')

    fig.add_trace(go.Candlestick(x=plot_df['label'], open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name="日K"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df['label'], y=plot_df['5日均值'], name="MA5", line=dict(color='yellow', width=1)), row=1, col=1)
    fig.add_hline(y=p_h_mid, line_dash="dash", line_color="cyan", annotation_text="预测最高", row=1, col=1)
    fig.add_hline(y=p_l_mid, line_dash="dash", line_color="cyan", annotation_text="预测最低", row=1, col=1)
    fig.add_trace(go.Bar(x=plot_df['label'], y=plot_df['换手率'], name="换手率", marker_color=['red' if x >= 20 else 'orange' if x >= 10 else 'gray' for x in plot_df['换手率']]), row=2, col=1)
    fig.update_xaxes(tickangle=-90, dtick=1, row=1, col=1)
    fig.update_xaxes(tickangle=-90, dtick=1, row=2, col=1)
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # --- 板块 4：历史明细 (完整保留) ---
    st.subheader("📋 历史参考明细")
    show_df = hist_df.tail(15).copy()
    show_df.index = show_df.index.date
    for c in ['换手率', '今开比例', '最高比例', '最低比例']:
        show_df[c] = show_df[c].map('{:.2f}%'.format)
    st.dataframe(show_df[['Open', 'High', 'Low', 'Close', '换手率', '今开比例', '最高比例', '最低比例', '5日均值']].style.format(precision=2))

except Exception as e:
    st.error(f"引擎初始化中: {e}")
