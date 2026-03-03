import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai  # 导入 AI 引擎库

# --- 0. AI 引擎初始化 ---
# 注意：实际使用需在 Streamlit Secret 或环境变量中配置 API KEY
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    # 演示用途：如果没有 Key，代码会降级到逻辑模拟模式
    pass

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

# --- 2. AI 深度解析模块 ---
def get_ai_analysis(data_summary):
    """
    接入真正的 AI 引擎进行动态形态分析
    """
    prompt = f"""
    你是一名专业的量化交易员。请根据以下 BTDR 股票的实时数据快照进行形态深度解析：
    
    【实时数据】
    - 当前价: {data_summary['curr_p']:.2f}
    - 5日均线(MA5): {data_summary['ma5']:.2f}
    - 今日换手率: {data_summary['turnover']:.2f}%
    - 预测中性支撑位: {data_summary['p_l_mid']:.2f}
    - 预测中性压力位: {data_summary['p_h_mid']:.2f}
    - 较昨收涨跌幅: {data_summary['change']:.2%}
    
    【分析要求】
    请分四个板块提供分析（字数控制在300字以内）：
    1. 形态特征：描述当前股价与MA5及支撑位的争夺情况。
    2. 量价配合：结合换手率判断是缩量筑底还是放量派发。
    3. 市场心理：分析当前多空情绪及筹码交换情况。
    4. 后市操作建议：给出具体的仓位与买卖逻辑建议。
    
    语言风格要专业且接地气。
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return "⚠️ AI 引擎解析暂时不可用，请检查 API 配置或网络。"

# --- 3. 界面显示 ---
st.set_page_config(layout="wide", page_title="BTDR AI量化终端")
st.title("🏹 BTDR 深度形态量化终端 (AI 实时驱动版)")

try:
    hist_df, live_df, float_shares, reg = get_btdr_final_data()
    last_hist = hist_df.iloc[-1]
    curr_p = live_df['Close'].iloc[-1]
    
    # 关键计算
    live_df.index = live_df.index.tz_convert('America/New_York')
    regular_market = live_df.between_time('09:30', '16:00')
    today_open = regular_market['Open'].iloc[0] if not regular_market.empty else live_df['Open'].iloc[-1]
    today_open_ratio = (today_open - last_hist['Close']) / last_hist['Close']

    p_h_mid = last_hist['Close'] * (1 + (reg['inter_h'] + reg['slope_h'] * today_open_ratio))
    p_l_mid = last_hist['Close'] * (1 + (reg['inter_l'] + reg['slope_l'] * today_open_ratio))
    today_turnover = (live_df['Volume'].sum() / float_shares) * 100

    # 场景识别颜色
    s_color = "#00FF00" if curr_p >= p_h_mid else "#FF4B4B" if curr_p <= p_l_mid else "#1E90FF"

    # --- 第一部分：顶部显示 ---
    col_metric, col_target = st.columns([1, 1.5])
    with col_metric:
        st.subheader("📊 实时状态")
        m1, m2 = st.columns(2)
        m1.metric("当前成交价", f"${curr_p:.2f}", f"{(curr_p/last_hist['Close']-1):.2%}")
        m2.metric("实时换手率", f"{today_turnover:.2f}%")

    with col_target:
        st.subheader("📍 场景股价预测目标")
        scenario_table = pd.DataFrame({
            "场景": ["中性场景", "乐观场景(+6%)", "悲观场景(-6%)"],
            "最高预测": [p_h_mid, p_h_mid * 1.06, p_h_mid * 0.94],
            "最低预测": [p_l_mid, p_l_mid * 1.06, p_l_mid * 0.94]
        })
        st.table(scenario_table.style.format(precision=2))

    st.divider()

    # --- 第二部分：AI 深度解析板块 ---
    st.subheader("🤖 AI 引擎深度形态解析")
    
    # 打包数据汇总给 AI
    data_summary = {
        'curr_p': curr_p,
        'ma5': last_hist['5日均值'],
        'turnover': today_turnover,
        'p_l_mid': p_l_mid,
        'p_h_mid': p_h_mid,
        'change': (curr_p / last_hist['Close'] - 1)
    }

    # 调用 AI 引擎（带缓存或手动刷新避免浪费 API 额度）
    if st.button("🔄 运行 AI 深度分析"):
        with st.spinner('AI 正在扫描盘面形态...'):
            analysis_text = get_ai_analysis(data_summary)
            st.markdown(f"""
            <div style="background-color: rgba(30, 144, 255, 0.1); padding: 20px; border-radius: 10px; border-left: 5px solid #1E90FF;">
                {analysis_text}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("点击上方按钮，让 AI 引擎对当前 BTDR 形态进行深度扫描。")

    # --- 第三部分：雷达图与走势图 ---
    st.divider()
    col_radar, col_chart = st.columns([1, 2])
    
    with col_radar:
        st.subheader("🎯 实时评分雷达")
        # 评分逻辑
        mom = min(max(((curr_p / today_open - 1) + 0.05) / 0.1 * 100, 0), 100)
        sup = min(max((1 - abs(curr_p - p_l_mid) / p_l_mid) * 100, 0), 100)
        trn = min((today_turnover / 20) * 100, 100)
        trd = min(max(((curr_p / last_hist['5日均值'] - 1) + 0.05) / 0.1 * 100, 0), 100)
        
        radar_fig = go.Figure(data=go.Scatterpolar(r=[mom, sup, trn, trd], theta=['动能', '支撑', '换手', '趋势'], fill='toself', line=dict(color=s_color, width=2)))
        radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=350, template="plotly_dark")
        st.plotly_chart(radar_fig, use_container_width=True)

    with col_chart:
        st.subheader("🕒 走势监控 (MM/DD)")
        plot_df = hist_df.tail(20).copy()
        plot_df['label'] = plot_df.index.strftime('%m/%d')
        fig = go.Figure(data=[go.Candlestick(x=plot_df['label'], open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name="日K")])
        fig.update_xaxes(tickangle=-90)
        fig.update_layout(height=400, template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # --- 第四部分：历史明细 (修正百分比显示) ---
    st.subheader("📋 历史数据明细 (修正比例)")
    show_df = hist_df.tail(15).copy()
    show_df.index = show_df.index.date
    # 修正显示：小数 -> 百分比文本
    for c in ['今开比例', '最高比例', '最低比例']:
        show_df[c] = (show_df[c] * 100).map('{:.2f}%'.format)
    show_df['换手率'] = (show_df['换手率_小数'] * 100).map('{:.2f}%'.format)

    st.dataframe(show_df[['Open', 'High', 'Low', 'Close', '今开比例', '最高比例', '最低比例', '换手率', '5日均值']].style.format(precision=2, subset=['Open', 'High', 'Low', 'Close', '5日均值']))

except Exception as e:
    st.error(f"引擎刷新中: {e}")
