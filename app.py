import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
import re

# --- 0. 访问权限控制 ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    if not st.session_state.password_correct:
        st.set_page_config(page_title="身份验证", page_icon="🔒")
        st.title("🔒 BTDR 决策终端授权")
        pwd = st.text_input("请输入访问授权码以开启系统", type="password")
        if st.button("进入系统"):
            if pwd == st.secrets.get("ACCESS_PASSWORD", "123456"):
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("授权码错误")
        st.stop()

check_password()

# --- 1. 数据引擎 ---
@st.cache_data(ttl=60)
def get_btdr_data():
    ticker_symbol = "BTDR"
    tk = yf.Ticker(ticker_symbol)
    hist = tk.history(period="60d", interval="1d")
    if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
    live_1m = tk.history(period="1d", interval="1m", prepost=True)
    if isinstance(live_1m.columns, pd.MultiIndex): live_1m.columns = live_1m.columns.get_level_values(0)
    
    try:
        realtime_v = tk.fast_info['last_volume']
    except:
        realtime_v = live_1m['Volume'].sum()
    
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
    return fit_df, live_1m, float_shares, reg_params, realtime_v

# --- 2. AI 决策引擎 (带结果保留) ---
@st.cache_data(ttl=3600, show_spinner=False)
def get_ai_analysis_cached(p_curr, p_ma5, turnover_str, p_low, p_high):
    if "DEEPSEEK_API_KEY" not in st.secrets:
        return "⚠️ 未配置 API Key", [50, 50, 50, 50]
    
    dev_sup = ((p_curr / p_low) - 1) * 100
    try:
        client = OpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
        prompt = f"""
        你是一名量化专家。审计 BTDR：现价${p_curr:.2f}, 偏离支撑{dev_sup:.2f}%, 换手率{turnover_str}, MA5 ${p_ma5:.2f}。
        给出：1.空间定位；2.量价审计；3.建议[看多/看空/观望]及止损位。
        最后一行格式: SCORES: 动能=X, 支撑=X, 换手=X, 趋势=X
        """
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": "实战干货逻辑。"}, {"role": "user", "content": prompt}]
        )
        full_text = response.choices[0].message.content
        scores = [int(n) for n in re.findall(r"\d+", re.findall(r"SCORES:.*", full_text)[0])] if re.findall(r"SCORES:.*", full_text) else [50,50,50,50]
        return re.sub(r"SCORES:.*", "", full_text).strip(), scores
    except Exception as e:
        return f"❌ AI 审计失败: {str(e)}", [0, 0, 0, 0]

# --- 3. 页面展示 ---
st.set_page_config(layout="wide", page_title="BTDR 决策终端")
st.title("🏹 BTDR 深度量化决策终端")

try:
    hist_df, live_df, float_shares, reg, rt_volume = get_btdr_data()
    last_hist = hist_df.iloc[-1]
    curr_p = live_df['Close'].iloc[-1]
    
    # 场景计算
    today_open = live_df.between_time('09:30', '16:00')['Open'].iloc[0] if not live_df.between_time('09:30', '16:00').empty else live_df['Open'].iloc[-1]
    today_open_ratio = (today_open - last_hist['Close']) / last_hist['Close']
    p_h_mid = last_hist['Close'] * (1 + (reg['inter_h'] + reg['slope_h'] * today_open_ratio))
    p_l_mid = last_hist['Close'] * (1 + (reg['inter_l'] + reg['slope_l'] * today_open_ratio))
    today_turnover = (rt_volume / float_shares) * 100

    # UI 模块 1: 实时指标
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.subheader("📊 实时状态")
        m1, m2 = st.columns(2)
        m1.metric("当前成交价", f"${curr_p:.2f}", f"{(curr_p/last_hist['Close']-1):.2%}")
        t_color = "red" if today_turnover > 15 else "orange" if today_turnover > 8 else "green"
        m2.markdown(f"**实时换手率**\n### :{t_color}[{today_turnover:.2f}%]")
    with c2:
        st.subheader("📍 场景预测目标")
        st.table(pd.DataFrame({"场景": ["中性回归", "乐观(+6%)", "悲观(-6%)"], "最高预测": [p_h_mid, p_h_mid*1.06, p_h_mid*0.94], "最低预测": [p_l_mid, p_l_mid*1.06, p_l_mid*0.94]}).style.format(precision=2))

    st.divider()

    # UI 模块 2: AI 研判 (带结果保留)
    col_text, col_radar = st.columns([1.6, 0.9])
    with col_text:
        st.subheader("🤖 AI 硬核形态研判")
        # 自动加载缓存结果
        text, scores = get_ai_analysis_cached(round(curr_p, 2), round(last_hist['5日均值'], 2), f"{today_turnover:.1f}%", round(p_l_mid, 2), round(p_h_mid, 2))
        
        if st.button("🔄 刷新 AI 实时审计 (消耗额度)", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.markdown(f"""<div style="background-color: rgba(0, 255, 255, 0.05); padding: 20px; border-radius: 12px; border-left: 6px solid #00CCCC; min-height: 250px;">{text}</div>""", unsafe_allow_html=True)

    with col_radar:
        fig_r = go.Figure(data=go.Scatterpolar(r=scores + [scores[0]], theta=['动能','支撑','换手','趋势','动能'], fill='toself', line=dict(color='#00CCCC')))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=320, margin=dict(l=40, r=40, t=30, b=30), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_r, use_container_width=True)

    # UI 模块 3: K线主图 (MA5 标红 + 换手率柱状图)
    st.subheader("🕒 实时监控主图 (垂直 MM/DD)")
    fig_k = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    p_df = hist_df.tail(20).copy()
    p_df['label'] = p_df.index.strftime('%m/%d')
    
    # K线
    fig_k.add_trace(go.Candlestick(x=p_df['label'], open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name="K线"), row=1, col=1)
    # MA5 改为红色并加粗
    fig_k.add_trace(go.Scatter(x=p_df['label'], y=p_df['5日均值'], name="MA5 (Red)", line=dict(color='#FF0000', width=2)), row=1, col=1)
    
    # 换手率柱状图回归
    colors = ['green' if (p_df['Close'].iloc[i] >= p_df['Open'].iloc[i]) else 'red' for i in range(len(p_df))]
    fig_k.add_trace(go.Bar(x=p_df['label'], y=p_df['换手率_原始']*100, name="换手率%", marker_color=colors, opacity=0.7), row=2, col=1)
    
    fig_k.update_xaxes(tickangle=-90, dtick=1)
    fig_k.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10))
    fig_k.update_yaxes(title_text="Price", row=1, col=1)
    fig_k.update_yaxes(title_text="Turnover%", row=2, col=1)
    st.plotly_chart(fig_k, use_container_width=True)

    # UI 模块 4: 数据明细
    st.subheader("📋 历史参考数据明细")
    show_df = hist_df.tail(15).copy()
    show_df.index = show_df.index.date
    for c in ['今开比例', '最高比例', '最低比例']:
        show_df[c] = (show_df[c] * 100).map('{:.2f}%'.format)
    show_df['换手率'] = (show_df['换手率_原始'] * 100).map('{:.2f}%'.format)
    st.dataframe(show_df[['Open', 'High', 'Low', 'Close', '今开比例', '最高比例', '最低比例', '换手率', '5日均值']].style.format(precision=2))

except Exception as e:
    st.error(f"终端异常: {e}")
