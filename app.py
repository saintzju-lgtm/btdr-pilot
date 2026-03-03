import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI  # DeepSeek 使用 OpenAI 兼容库

# --- 1. 数据引擎 ---
@st.cache_data(ttl=60)
def get_btdr_final_data():
    ticker = "BTDR"
    # 获取历史数据
    hist = yf.download(ticker, period="60d", interval="1d")
    if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
    
    # 获取实时数据
    live_1m = yf.download(ticker, period="1d", interval="1m", prepost=True)
    if isinstance(live_1m.columns, pd.MultiIndex): live_1m.columns = live_1m.columns.get_level_values(0)
    
    float_shares = 118000000 
    hist['昨收'] = hist['Close'].shift(1)
    
    # 核心计算逻辑：回归模型输入（小数形式）
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

# --- 2. DeepSeek AI 深度解析引擎 ---
def get_deepseek_analysis(data_summary):
    if "DEEPSEEK_API_KEY" not in st.secrets:
        return "⚠️ 请在 Streamlit Secrets 中配置 DEEPSEEK_API_KEY 以激活 AI 解析。"
    
    try:
        client = OpenAI(
            api_key=st.secrets["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com" # DeepSeek 官方接口地址
        )
        
        prompt = f"""
        你是一名资深美股量化交易员。请根据 BTDR 的实时快照进行形态深度解析。
        
        【盘面实时数据】
        - 当前成交价: ${data_summary['curr_p']:.2f} (较昨日涨跌: {data_summary['change']})
        - 5日均线(MA5): ${data_summary['ma5']:.2f}
        - 实时换手率: {data_summary['turnover']}
        - 回归预测支撑位: ${data_summary['p_l_mid']:.2f}
        - 回归预测压力位: ${data_summary['p_h_mid']:.2f}
        
        【解析要求】
        1. 形态特征：分析股价与MA5及支撑位的关系。
        2. 量价配合：结合换手率判断是缩量筑底还是主力出货。
        3. 市场心理：分析多空情绪，筹码是否完成交换。
        4. 后市操作建议：给出明确的策略（分批低吸/逢高减仓/持仓观望）及逻辑。
        
        请使用专业且直白的中文，不要废话。
        """
        
        response = client.chat.completions.create(
            model="deepseek-chat", # 使用 DeepSeek 聊天模型
            messages=[
                {"role": "system", "content": "你是一个严谨的量化投资分析专家。"},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ DeepSeek 调用失败: {str(e)}"

# --- 3. 页面配置与显示 ---
st.set_page_config(layout="wide", page_title="BTDR DeepSeek量化终端")
st.title("🏹 BTDR 深度形态量化终端 (DeepSeek AI 驱动版)")

try:
    hist_df, live_df, float_shares, reg = get_btdr_final_data()
    last_hist = hist_df.iloc[-1]
    curr_p = live_df['Close'].iloc[-1]
    
    # 动态计算今日场景
    live_df.index = live_df.index.tz_convert('America/New_York')
    regular_market = live_df.between_time('09:30', '16:00')
    today_open = regular_market['Open'].iloc[0] if not regular_market.empty else live_df['Open'].iloc[-1]
    today_open_ratio = (today_open - last_hist['Close']) / last_hist['Close']

    p_h_mid = last_hist['Close'] * (1 + (reg['inter_h'] + reg['slope_h'] * today_open_ratio))
    p_l_mid = last_hist['Close'] * (1 + (reg['inter_l'] + reg['slope_l'] * today_open_ratio))
    today_turnover = (live_df['Volume'].sum() / float_shares) * 100

    # 布局 A：实时指标与预测
    col_met, col_table = st.columns([1, 1.5])
    with col_met:
        st.subheader("📊 实时状态")
        m1, m2 = st.columns(2)
        m1.metric("当前价", f"${curr_p:.2f}", f"{(curr_p/last_hist['Close']-1):.2%}")
        m2.metric("实时换手率", f"{today_turnover:.2f}%")

    with col_table:
        st.subheader("📍 场景股价预测目标")
        st.table(pd.DataFrame({
            "场景描述": ["回归中性", "乐观场景(+6%)", "悲观场景(-6%)"],
            "最高预测": [p_h_mid, p_h_mid * 1.06, p_h_mid * 0.94],
            "最低预测": [p_l_mid, p_l_mid * 1.06, p_l_mid * 0.94]
        }).style.format(precision=2))

    st.divider()

    # 布局 B：AI 解析区 (核心)
    st.subheader("🤖 DeepSeek AI 深度形态扫描")
    
    data_summary = {
        'curr_p': curr_p,
        'ma5': last_hist['5日均值'],
        'turnover': f"{today_turnover:.2f}%",
        'p_l_mid': p_l_mid,
        'p_h_mid': p_h_mid,
        'change': f"{(curr_p/last_hist['Close']-1)*100:.2f}%"
    }

    if st.button("🚀 启动 DeepSeek 深度形态分析"):
        with st.spinner('DeepSeek 正在解析盘面逻辑...'):
            analysis = get_deepseek_analysis(data_summary)
            st.markdown(f"""
            <div style="background-color: rgba(0, 255, 255, 0.05); padding: 25px; border-radius: 12px; border-left: 6px solid #00CCCC; line-height: 1.6;">
                {analysis}
            </div>
            """, unsafe_allow_html=True)

    # 布局 C：图表区
    st.divider()
    col_radar, col_chart = st.columns([1, 2])
    with col_radar:
        st.subheader("🎯 评分雷达")
        # 简化评分逻辑
        sup = min(max((1 - abs(curr_p - p_l_mid) / p_l_mid) * 100, 0), 100)
        trn = min((today_turnover / 20) * 100, 100)
        fig_r = go.Figure(data=go.Scatterpolar(r=[sup, trn, 50, 50], theta=['支撑', '换手', '动能', '趋势'], fill='toself', line=dict(color="#00CCCC")))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=False)), height=350, template="plotly_dark")
        st.plotly_chart(fig_r, use_container_width=True)

    with col_chart:
        st.subheader("🕒 近期日K监控 (MM/DD)")
        plot_df = hist_df.tail(15).copy()
        plot_df['label'] = plot_df.index.strftime('%m/%d')
        fig_k = go.Figure(data=[go.Candlestick(x=plot_df['label'], open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'])])
        fig_k.update_xaxes(tickangle=-90)
        fig_k.update_layout(height=400, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_k, use_container_width=True)

    # 布局 D：历史明细 (修正百分比)
    st.subheader("📋 历史明细数据")
    show_df = hist_df.tail(15).copy()
    show_df.index = show_df.index.date
    for c in ['今开比例', '最高比例', '最低比例']:
        show_df[c] = (show_df[c] * 100).map('{:.2f}%'.format)
    show_df['换手率'] = (show_df['换手率_小数'] * 100).map('{:.2f}%'.format)
    st.dataframe(show_df[['Open', 'High', 'Low', 'Close', '今开比例', '最高比例', '最低比例', '换手率', '5日均值']].style.format(precision=2))

except Exception as e:
    st.error(f"终端运行中: {e}")
