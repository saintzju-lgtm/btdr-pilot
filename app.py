import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
import re

# --- 1. 数据引擎 (保持原样) ---
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
    hist['换手率_原始'] = (hist['Volume'] / float_shares)
    hist['5日均值'] = hist['Close'].rolling(5).mean()
    
    fit_df = hist.dropna()
    X = fit_df[['今开比例']].values
    m_h = LinearRegression().fit(X, fit_df['最高比例'].values)
    m_l = LinearRegression().fit(X, fit_df['最低比例'].values)
    
    reg_params = {'slope_h': m_h.coef_[0], 'inter_h': m_h.intercept_, 'slope_l': m_l.coef_[0], 'inter_l': m_l.intercept_}
    return fit_df, live_1m, float_shares, reg_params

# --- 2. DeepSeek AI 引擎 (解析 + 评分提取) ---
def get_ai_analysis_and_scores(data_summary):
    if "DEEPSEEK_API_KEY" not in st.secrets:
        return "⚠️ 请在 Secrets 中配置 DEEPSEEK_API_KEY", [50, 50, 50, 50]
    
    try:
        client = OpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
        prompt = f"""
        你是一名资深美股量化交易员。分析 BTDR 实时快照：
        - 当前价: ${data_summary['curr_p']:.2f} (涨跌幅: {data_summary['change']})
        - MA5: ${data_summary['ma5']:.2f}, 换手率: {data_summary['turnover']}
        - 预测支撑: ${data_summary['p_l_mid']:.2f}, 预测压力: ${data_summary['p_h_mid']:.2f}
        
        要求回复：
        1. 形态特征、量价配合、市场心理、操作建议（专业、精炼）。
        2. 最后一行必须严格按照此格式输出分值（0-100）：
        SCORES: 动能=X, 支撑=X, 换手=X, 趋势=X
        """
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": "你是一个严谨的投资专家。"}, {"role": "user", "content": prompt}]
        )
        full_text = response.choices[0].message.content
        
        # 提取分值
        scores = [50, 50, 50, 50]
        score_line = re.findall(r"SCORES:.*", full_text)
        if score_line:
            nums = re.findall(r"\d+", score_line[0])
            if len(nums) == 4: scores = [int(n) for n in nums]
        
        clean_text = re.sub(r"SCORES:.*", "", full_text).strip()
        return clean_text, scores
    except Exception as e:
        return f"❌ AI 调用失败: {str(e)}", [0, 0, 0, 0]

# --- 3. 页面布局 ---
st.set_page_config(layout="wide", page_title="BTDR 决策终端")
st.title("🏹 BTDR 深度形态量化终端 (AI 实时驱动)")

try:
    hist_df, live_df, float_shares, reg = get_btdr_final_data()
    last_hist = hist_df.iloc[-1]
    curr_p = live_df['Close'].iloc[-1]
    
    # 核心计算 (保持原样)
    live_df.index = live_df.index.tz_convert('America/New_York')
    regular_market = live_df.between_time('09:30', '16:00')
    today_open = regular_market['Open'].iloc[0] if not regular_market.empty else live_df['Open'].iloc[-1]
    today_open_ratio = (today_open - last_hist['Close']) / last_hist['Close']
    p_h_mid = last_hist['Close'] * (1 + (reg['inter_h'] + reg['slope_h'] * today_open_ratio))
    p_l_mid = last_hist['Close'] * (1 + (reg['inter_l'] + reg['slope_l'] * today_open_ratio))
    today_turnover = (live_df['Volume'].sum() / float_shares) * 100

    # --- UI 模块 1：实时指标与预测表 ---
    col_metric, col_target = st.columns([1, 1.5])
    with col_metric:
        st.subheader("📊 实时状态")
        m1, m2 = st.columns(2)
        m1.metric("当前价", f"${curr_p:.2f}", f"{(curr_p/last_hist['Close']-1):.2%}")
        t_status = "red" if today_turnover >= 20 else "orange" if today_turnover >= 10 else "green"
        m2.markdown(f"**当日实时换手**\n### :{t_status}[{today_turnover:.2f}%]")

    with col_target:
        st.subheader("📍 场景股价预测目标")
        st.table(pd.DataFrame({
            "场景描述": ["中性 (回归)", "乐观 (+6%)", "悲观 (-6%)"],
            "最高股价预测": [p_h_mid, p_h_mid * 1.06, p_h_mid * 0.94],
            "最低股价预测": [p_l_mid, p_l_mid * 1.06, p_l_mid * 0.94]
        }).style.format(precision=2))

    st.divider()

    # --- UI 模块 2：AI 分析按钮及结果 ---
    # 初始化 session state 
    if 'ai_scores' not in st.session_state: st.session_state.ai_scores = [50, 50, 50, 50]
    if 'ai_text' not in st.session_state: st.session_state.ai_text = "等待 AI 扫描..."

    # 按钮单独一行，位于解析区和雷达图的正上方
    btn_col1, btn_col2 = st.columns([1.6, 0.9])
    with btn_col1:
        if st.button("🚀 点击启动 AI 深度解析 & 自动评分雷达", use_container_width=True):
            data_summary = {
                'curr_p': curr_p, 'ma5': last_hist['5日均值'], 'turnover': f"{today_turnover:.2f}%",
                'p_l_mid': p_l_mid, 'p_h_mid': p_h_mid, 'change': f"{(curr_p/last_hist['Close']-1)*100:.2f}%"
            }
            with st.spinner('DeepSeek 正在解析盘面...'):
                text, scores = get_ai_analysis_and_scores(data_summary)
                st.session_state.ai_text = text
                st.session_state.ai_scores = scores

    # 文字解析与雷达图
    col_text, col_radar = st.columns([1.6, 0.9])
    with col_text:
        st.markdown(f"""
        <div style="background-color: rgba(30, 144, 255, 0.05); padding: 20px; border-radius: 12px; border-left: 6px solid #1E90FF; min-height: 250px;">
            <h4 style="margin-top:0;">🤖 AI 形态解析报告</h4>
            {st.session_state.ai_text}
        </div>
        """, unsafe_allow_html=True)

    with col_radar:
        s = st.session_state.ai_scores
        radar_fig = go.Figure(data=go.Scatterpolar(r=s + [s[0]], theta=['动能', '支撑', '换手', '趋势', '动能'], fill='toself', fillcolor='rgba(30, 144, 255, 0.3)', line=dict(color='#1E90FF', width=3)))
        radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=300, margin=dict(l=40, r=40, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(radar_fig, use_container_width=True)
        st.caption(f"AI 综合评分：动能 {s[0]} | 支撑 {s[1]} | 换手 {s[2]} | 趋势 {s[3]}")

    st.divider()

    # --- UI 模块 3：K线图 ---
    st.subheader("🕒 近期日K走势 (垂直 MM/DD)")
    fig_k = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    plot_df = hist_df.tail(20).copy()
    plot_df['label'] = plot_df.index.strftime('%m/%d')
    fig_k.add_trace(go.Candlestick(x=plot_df['label'], open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name="日K"), row=1, col=1)
    fig_k.add_trace(go.Scatter(x=plot_df['label'], y=plot_df['5日均值'], name="MA5", line=dict(color='yellow', width=1)), row=1, col=1)
    fig_k.add_trace(go.Bar(x=plot_df['label'], y=plot_df['换手率_原始']*100, name="换手率", marker_color='gray'), row=2, col=1)
    fig_k.update_xaxes(tickangle=-90, dtick=1)
    fig_k.update_layout(height=550, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_k, use_container_width=True)

    # --- UI 模块 4：数据明细 ---
    st.subheader("📋 历史明细数据参考")
    show_df = hist_df.tail(15).copy()
    show_df.index = show_df.index.date
    for c in ['今开比例', '最高比例', '最低比例']:
        show_df[c] = (show_df[c] * 100).map('{:.2f}%'.format)
    show_df['换手率'] = (show_df['换手率_原始'] * 100).map('{:.2f}%'.format)
    st.dataframe(show_df[['Open', 'High', 'Low', 'Close', '今开比例', '最高比例', '最低比例', '换手率', '5日均值']].style.format(precision=2))

except Exception as e:
    st.error(f"终端运行中: {e}")
