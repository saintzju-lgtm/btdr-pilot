import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
import re

# --- 0. 访问权限控制 (保护 API 流量) ---
def check_password():
    """只有输入正确密码才显示主界面"""
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    
    if not st.session_state.password_correct:
        st.set_page_config(page_title="身份验证", page_icon="🔒")
        st.title("🔒 BTDR 决策终端授权")
        pwd = st.text_input("请输入访问授权码以开启 AI 审计系统", type="password")
        if st.button("进入系统"):
            if pwd == st.secrets.get("ACCESS_PASSWORD", "123456"): # 默认或从Secret读取
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("授权码错误，拒绝访问。")
        st.stop()

# 执行密码检查
check_password()

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
    hist['换手率_原始'] = (hist['Volume'] / float_shares)
    hist['5日均值'] = hist['Close'].rolling(5).mean()
    
    fit_df = hist.dropna()
    X = fit_df[['今开比例']].values
    m_h = LinearRegression().fit(X, fit_df['最高比例'].values)
    m_l = LinearRegression().fit(X, fit_df['最低比例'].values)
    
    reg_params = {'slope_h': m_h.coef_[0], 'inter_h': m_h.intercept_, 'slope_l': m_l.coef_[0], 'inter_l': m_l.intercept_}
    return fit_df, live_1m, float_shares, reg_params

# --- 2. 硬核 AI 决策引擎 ---
def get_ai_analysis_and_scores(data_summary):
    if "DEEPSEEK_API_KEY" not in st.secrets:
        return "⚠️ 未配置 API Key", [50, 50, 50, 50]
    
    # 支撑位偏离度计算
    dev_sup = ((data_summary['curr_p'] / data_summary['p_l_mid']) - 1) * 100
    
    try:
        client = OpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
        prompt = f"""
        你是一名严格的量化策略师。审计 BTDR 实时盘面并给出硬核指令：
        【量化快照】
        - 现价: ${data_summary['curr_p']:.2f}, MA5: ${data_summary['ma5']:.2f}
        - 支撑偏离度: {dev_sup:.2f}%, 换手: {data_summary['turnover']}
        - 支撑/压力位: ${data_summary['p_l_mid']:.2f} / ${data_summary['p_h_mid']:.2f}
        
        【任务】
        1. 空间定位：当前处于安全区、风险区还是破位区？
        2. 异常审计：量价是否背离？
        3. 战术指令：以[看多/看空/观望]开头，给出具体操作逻辑及建议止损价。
        4. 评分输出：SCORES: 动能=X, 支撑=X, 换手=X, 趋势=X
        """
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": "你只讲实战干货，不啰嗦，逻辑严密。"}, {"role": "user", "content": prompt}]
        )
        full_text = response.choices[0].message.content
        
        scores = [50, 50, 50, 50]
        score_line = re.findall(r"SCORES:.*", full_text)
        if score_line:
            nums = re.findall(r"\d+", score_line[0])
            if len(nums) == 4: scores = [int(n) for n in nums]
        
        return re.sub(r"SCORES:.*", "", full_text).strip(), scores
    except Exception as e:
        return f"❌ AI 审计失败: {str(e)}", [0, 0, 0, 0]

# --- 3. 主界面布局 ---
st.set_page_config(layout="wide", page_title="BTDR 决策终端")
st.title("🏹 BTDR 深度形态量化终端 (AI 保护版)")

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

    # 1. 实时状态与预测表
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.subheader("📊 实时状态")
        m1, m2 = st.columns(2)
        m1.metric("当前成交价", f"${curr_p:.2f}", f"{(curr_p/last_hist['Close']-1):.2%}")
        m2.metric("实时换手率", f"{today_turnover:.2f}%")
    with c2:
        st.subheader("📍 场景股价预测目标")
        st.table(pd.DataFrame({
            "场景": ["中性回归", "乐观(+6%)", "悲观(-6%)"],
            "最高点预测": [p_h_mid, p_h_mid * 1.06, p_h_mid * 0.94],
            "最低点预测": [p_l_mid, p_l_mid * 1.06, p_l_mid * 0.94]
        }).style.format(precision=2))

    st.divider()

    # 2. AI 审计区 (带 Session State 缓存)
    if 'ai_scores' not in st.session_state: st.session_state.ai_scores = [50, 50, 50, 50]
    if 'ai_text' not in st.session_state: st.session_state.ai_text = "等待启动 AI 审计..."

    col_text, col_radar = st.columns([1.6, 0.9])
    with col_text:
        st.subheader("🤖 AI 硬核形态研判")
        if st.button("🚀 启动 AI 实战审计 (消耗额度)", use_container_width=True):
            data_sum = {'curr_p': curr_p, 'ma5': last_hist['5日均值'], 'turnover': f"{today_turnover:.2f}%", 
                        'p_l_mid': p_l_mid, 'p_h_mid': p_h_mid, 'change': f"{(curr_p/last_hist['Close']-1)*100:.2f}%"}
            with st.spinner('DeepSeek 正在执行审计...'):
                t, s = get_ai_analysis_and_scores(data_sum)
                st.session_state.ai_text, st.session_state.ai_scores = t, s

        st.markdown(f"""<div style="background-color: rgba(0, 255, 255, 0.05); padding: 20px; border-radius: 12px; border-left: 6px solid #00CCCC;">{st.session_state.ai_text}</div>""", unsafe_allow_html=True)

    with col_radar:
        st.subheader("🎯 风险评估雷达")
        s = st.session_state.ai_scores
        fig_r = go.Figure(data=go.Scatterpolar(r=s + [s[0]], theta=['动能','支撑','换手','趋势','动能'], fill='toself', line=dict(color='#00CCCC', width=3)))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=320, margin=dict(l=40, r=40, t=30, b=30), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_r, use_container_width=True)

    # 3. K线主图 (垂直坐标)
    st.subheader("🕒 近期走势监控 (垂直 MM/DD)")
    fig_k = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    p_df = hist_df.tail(20).copy()
    p_df['label'] = p_df.index.strftime('%m/%d')
    fig_k.add_trace(go.Candlestick(x=p_df['label'], open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name="K线"), row=1, col=1)
    fig_k.add_trace(go.Scatter(x=p_df['label'], y=p_df['5日均值'], name="MA5", line=dict(color='yellow', width=1)), row=1, col=1)
    fig_k.add_trace(go.Bar(x=p_df['label'], y=p_df['换手率_原始']*100, name="换手", marker_color='gray'), row=2, col=1)
    fig_k.update_xaxes(tickangle=-90, dtick=1)
    fig_k.update_layout(height=550, xaxis_rangeslider_visible=False, template="plotly_dark")
    st.plotly_chart(fig_k, use_container_width=True)

    # 4. 数据明细
    st.subheader("📋 历史参考数据")
    show_df = hist_df.tail(15).copy()
    show_df.index = show_df.index.date
    for c in ['今开比例', '最高比例', '最低比例']:
        show_df[c] = (show_df[c] * 100).map('{:.2f}%'.format)
    show_df['换手率'] = (show_df['换手率_原始'] * 100).map('{:.2f}%'.format)
    st.dataframe(show_df[['Open', 'High', 'Low', 'Close', '今开比例', '最高比例', '最低比例', '换手率', '5日均值']].style.format(precision=2))

except Exception as e:
    st.error(f"终端异常: {e}")
