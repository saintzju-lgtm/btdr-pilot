import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
import re  # 用于提取 AI 文本中的数值

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
    hist['换手率_小数'] = (hist['Volume'] / float_shares)
    hist['5日均值'] = hist['Close'].rolling(5).mean()
    
    fit_df = hist.dropna()
    X = fit_df[['今开比例']].values
    m_h = LinearRegression().fit(X, fit_df['最高比例'].values)
    m_l = LinearRegression().fit(X, fit_df['最低比例'].values)
    
    reg_params = {'slope_h': m_h.coef_[0], 'inter_h': m_h.intercept_, 'slope_l': m_l.coef_[0], 'inter_l': m_l.intercept_}
    return fit_df, live_1m, float_shares, reg_params

# --- 2. AI 深度解析与自动评分引擎 ---
def get_ai_analysis_and_scores(data_summary):
    if "DEEPSEEK_API_KEY" not in st.secrets:
        return "⚠️ 未配置 API Key", [50, 50, 50, 50]
    
    try:
        client = OpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
        
        prompt = f"""
        你是一名资深量化交易员。分析 BTDR 实时快照：
        - 当前价: ${data_summary['curr_p']:.2f}, 涨跌: {data_summary['change']}
        - MA5: ${data_summary['ma5']:.2f}, 换手率: {data_summary['turnover']}
        - 预测支撑: ${data_summary['p_l_mid']:.2f}, 预测压力: ${data_summary['p_h_mid']:.2f}
        
        【任务要求】
        1. 提供精炼的形态解析（形态、量价、心理、建议）。
        2. 请在回答的最后一行，按照以下格式输出四个维度的评分（0-100）：
           SCORES: 动能=X, 支撑=X, 换手=X, 趋势=X
        """
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": "你是一个严谨的量化分析专家。"}, {"role": "user", "content": prompt}]
        )
        full_text = response.choices[0].message.content
        
        # 提取分值逻辑
        scores = [50, 50, 50, 50] # 默认中值
        score_line = re.findall(r"SCORES:.*", full_text)
        if score_line:
            nums = re.findall(r"\d+", score_line[0])
            if len(nums) == 4:
                scores = [int(n) for n in nums]
        
        # 返回过滤掉评分行的文本和分值
        display_text = re.sub(r"SCORES:.*", "", full_text)
        return display_text, scores
        
    except Exception as e:
        return f"❌ 调用失败: {str(e)}", [0, 0, 0, 0]

# --- 3. 页面布局 ---
st.set_page_config(layout="wide", page_title="BTDR AI 全驱动终端")
st.title("🏹 BTDR 量化终端 (DeepSeek AI 综合评价版)")

try:
    hist_df, live_df, float_shares, reg = get_btdr_final_data()
    last_hist = hist_df.iloc[-1]
    curr_p = live_df['Close'].iloc[-1]
    
    # 动态数据计算
    live_df.index = live_df.index.tz_convert('America/New_York')
    regular_market = live_df.between_time('09:30', '16:00')
    today_open = regular_market['Open'].iloc[0] if not regular_market.empty else live_df['Open'].iloc[-1]
    today_open_ratio = (today_open - last_hist['Close']) / last_hist['Close']
    p_h_mid = last_hist['Close'] * (1 + (reg['inter_h'] + reg['slope_h'] * today_open_ratio))
    p_l_mid = last_hist['Close'] * (1 + (reg['inter_l'] + reg['slope_l'] * today_open_ratio))
    today_turnover = (live_df['Volume'].sum() / float_shares) * 100

    # 数据摘要
    data_summary = {
        'curr_p': curr_p, 'ma5': last_hist['5日均值'], 'turnover': f"{today_turnover:.2f}%",
        'p_l_mid': p_l_mid, 'p_h_mid': p_h_mid, 'change': f"{(curr_p/last_hist['Close']-1)*100:.2f}%"
    }

    # 初始化 Session State 存储分值
    if 'ai_scores' not in st.session_state:
        st.session_state.ai_scores = [50, 50, 50, 50]
    if 'ai_text' not in st.session_state:
        st.session_state.ai_text = "请点击下方按钮启动 AI 深度扫描..."

    # --- UI 展示 ---
    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        st.subheader("🤖 AI 深度解析与策略")
        if st.button("🚀 启动 AI 逻辑扫描 & 自动评分"):
            with st.spinner('DeepSeek 正在多维度量化盘面...'):
                text, scores = get_ai_analysis_and_scores(data_summary)
                st.session_state.ai_text = text
                st.session_state.ai_scores = scores
        
        st.markdown(f"""<div style="background-color: rgba(0, 255, 255, 0.05); padding: 20px; border-radius: 12px; border-left: 6px solid #00CCCC;">{st.session_state.ai_text}</div>""", unsafe_allow_html=True)

    with col_right:
        st.subheader("🎯 AI 评测雷达图")
        s = st.session_state.ai_scores
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=s + [s[0]], # 闭合曲线
            theta=['动能', '支撑', '换手', '趋势', '动能'],
            fill='toself',
            fillcolor='rgba(0, 204, 204, 0.3)',
            line=dict(color='#00CCCC', width=3)
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            margin=dict(l=40, r=40, t=40, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption(f"📊 AI 实时分值：动能 {s[0]} | 支撑 {s[1]} | 换手 {s[2]} | 趋势 {s[3]}")

    st.divider()
    # 历史明细等部分保持不变... (省略部分重复 UI 代码)

except Exception as e:
    st.error(f"终端运行异常: {e}")
