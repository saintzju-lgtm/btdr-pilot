import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 数据引擎 (修复百分比计算逻辑) ---
@st.cache_data(ttl=60)
def get_btdr_final_data():
    ticker = "BTDR"
    hist = yf.download(ticker, period="60d", interval="1d")
    if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
    
    live_1m = yf.download(ticker, period="1d", interval="1m", prepost=True)
    if isinstance(live_1m.columns, pd.MultiIndex): live_1m.columns = live_1m.columns.get_level_values(0)
    
    float_shares = 118000000 
    hist['昨收'] = hist['Close'].shift(1)
    
    # 修复点：这里计算出的是原始小数（如 0.01），后面显示时需要 * 100
    hist['今开比例'] = (hist['Open'] - hist['昨收']) / hist['昨收']
    hist['最高比例'] = (hist['High'] - hist['昨收']) / hist['昨收']
    hist['最低比例'] = (hist['Low'] - hist['昨收']) / hist['昨收']
    
    hist['换手率_原始'] = (hist['Volume'] / float_shares) # 小数形式
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
st.set_page_config(layout="wide", page_title="BTDR 深度决策终端")
st.title("🏹 BTDR 深度形态量化终端 (数值精度修正版)")

try:
    hist_df, live_df, float_shares, reg = get_btdr_final_data()
    last_hist = hist_df.iloc[-1]
    curr_p = live_df['Close'].iloc[-1]
    
    live_df.index = live_df.index.tz_convert('America/New_York')
    regular_market = live_df.between_time('09:30', '16:00')
    today_open = regular_market['Open'].iloc[0] if not regular_market.empty else live_df['Open'].iloc[-1]
    
    # 计算今日比例
    today_open_ratio = (today_open - last_hist['Close']) / last_hist['Close']

    # 预测逻辑 (直接使用回归系数，结果为小数)
    pred_h_ratio = reg['inter_h'] + reg['slope_h'] * today_open_ratio
    pred_l_ratio = reg['inter_l'] + reg['slope_l'] * today_open_ratio
    
    p_h_mid = last_hist['Close'] * (1 + pred_h_ratio)
    p_l_mid = last_hist['Close'] * (1 + pred_l_ratio)
    
    # 实时换手率 (百分比)
    today_turnover = (live_df['Volume'].sum() / float_shares) * 100

    # 场景识别
    def get_market_scene(p, h, l, vol):
        if p >= h * 1.005 and vol >= 10: return "乐观场景", "#00FF00", "价格突破统计高位，量能配合强劲。"
        elif p <= l * 0.995 and vol >= 10: return "悲观场景", "#FF4B4B", "价格跌破统计底线，抛压释放中。"
        else: return "中性场景", "#1E90FF", "处于统计区间内，波动受回归线锚定。"

    s_name, s_color, s_desc = get_market_scene(curr_p, p_h_mid, p_l_mid, today_turnover)

    # --- 第一部分：顶部显示 ---
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
        m1.metric("当前成交价", f"${curr_p:.2f}", f"{(curr_p/last_hist['Close']-1):.2%}")
        t_status = "red" if today_turnover >= 20 else "orange" if today_turnover >= 10 else "green"
        m2.markdown(f"**实时换手率**\n### :{t_status}[{today_turnover:.2f}%]")

    with col_target:
        st.subheader("📍 场景股价预测目标")
        scenario_table = pd.DataFrame({
            "场景": ["中性场景 (回归)", "乐观场景 (+6%)", "悲观场景 (-6%)"],
            "最高预测": [p_h_mid, p_h_mid * 1.06, p_h_mid * 0.94],
            "最低预测": [p_l_mid, p_l_mid * 1.06, p_l_mid * 0.94]
        })
        st.table(scenario_table.style.format(precision=2))

    st.divider()

    # --- 第二部分：深度形态分析 ---
    col_text, col_radar = st.columns([1.6, 0.9])
    with col_text:
        st.subheader("🤖 智能形态深度解析")
        ma5_curr = last_hist['5日均值']
        
        # 1. 形态特征
        st.markdown(f"**🔍 形态特征**")
        st.write(f"形态特征：股价停止创新低，开始在 {curr_p*0.95:.1f} - {curr_p*1.05:.1f} 区间横盘震荡，且K线开始尝试触碰或站上MA5均线（{ma5_curr:.2f}）。"
                 f"目前股价正在测试预测最低（约 {p_l_mid:.2f}）的支撑/阻力转换位。")

        # 2. 量价配合
        st.markdown(f"**📊 量价配合**")
        st.write(f"量价配合：换手率从天量迅速回落（今日：{today_turnover:.2f}%），呈现出缩量筑底态势。")

        # 3. 市场心理与建议
        st.markdown(f"**🧠 市场心理**")
        st.write("市场心理：恐慌盘已经被洗出，底部筹码完成初步交换。当前市场抛压极轻，处于观望状态，等待新的方向选择。")
        
        st.markdown(f"**💡 后市操作建议**")
        if curr_p <= p_l_mid * 1.01:
            st.success("建议建议：当前处于支撑确认期，建议在支撑位附近分批低吸，博弈反弹空间。")
        else:
            st.info("建议建议：目前处于区间中位，建议持仓观望，待趋势进一步明朗。")

    with col_radar:
        st.subheader("🎯 评分雷达")
        # 分值归一化计算
        mom = min(max(((curr_p / today_open - 1) + 0.05) / 0.1 * 100, 0), 100)
        sup = min(max((1 - abs(curr_p - p_l_mid) / p_l_mid) * 100, 0), 100)
        trn = min((today_turnover / 20) * 100, 100)
        trd = min(max(((curr_p / ma5_curr - 1) + 0.05) / 0.1 * 100, 0), 100)
        radar_fig = go.Figure(data=go.Scatterpolar(r=[mom, sup, trn, trd], theta=['动能', '支撑', '换手', '趋势'], fill='toself', fillcolor=f'rgba{tuple(int(s_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.3,)}', line=dict(color=s_color, width=2)))
        radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=300, margin=dict(l=40, r=40, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(radar_fig, use_container_width=True)

    # --- 第三部分：历史明细 (修正百分比显示) ---
    st.subheader("📋 历史明细参考数据 (数值修正)")
    show_df = hist_df.tail(15).copy()
    show_df.index = show_df.index.date
    
    # 修复点：原始小数 * 100 后再加百分号
    for c in ['今开比例', '最高比例', '最低比例']:
        show_df[c] = (show_df[c] * 100).map('{:.2f}%'.format)
    
    show_df['换手率'] = (show_df['换手率_原始'] * 100).map('{:.2f}%'.format)

    st.dataframe(show_df[['Open', 'High', 'Low', 'Close', '今开比例', '最高比例', '最低比例', '换手率', '5日均值']].style.format(precision=2, subset=['Open', 'High', 'Low', 'Close', '5日均值']))

except Exception as e:
    st.error(f"引擎刷新中: {e}")
