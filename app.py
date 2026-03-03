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
        st.title("🔒 BTDR 终端安全授权")
        pwd = st.text_input("请输入访问授权码", type="password")
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
    tk = yf.Ticker("BTDR")
    hist = tk.history(period="60d", interval="1d")
    if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
    live_1m = tk.history(period="1d", interval="1m", prepost=True)
    if isinstance(live_1m.columns, pd.MultiIndex): live_1m.columns = live_1m.columns.get_level_values(0)
    
    try:
        rt_v = tk.fast_info['last_volume']
    except:
        rt_v = live_1m['Volume'].sum()
    
    float_sh = tk.info.get('floatShares', 118000000)
    hist['昨收'] = hist['Close'].shift(1)
    hist['今开比例'] = (hist['Open'] - hist['昨收']) / hist['昨收']
    hist['最高比例'] = (hist['High'] - hist['昨收']) / hist['昨收']
    hist['最低比例'] = (hist['Low'] - hist['昨收']) / hist['昨收']
    hist['换手率_原始'] = (hist['Volume'] / float_sh)
    hist['MA5'] = hist['Close'].rolling(5).mean()
    
    fit_df = hist.dropna()
    X = fit_df[['今开比例']].values
    m_h = LinearRegression().fit(X, fit_df['最高比例'].values)
    m_l = LinearRegression().fit(X, fit_df['最低比例'].values)
    reg_params = {'slope_h': m_h.coef_[0], 'inter_h': m_h.intercept_, 'slope_l': m_l.coef_[0], 'inter_l': m_l.intercept_}
    return fit_df, live_1m, float_sh, reg_params, rt_v

# --- 2. AI 决策引擎 (逻辑高亮 + 格式对齐) ---
@st.cache_data(ttl=3600, show_spinner=False)
def get_ai_analysis_cached(p_curr, p_ma5, turnover_str, p_low, p_high):
    if "DEEPSEEK_API_KEY" not in st.secrets:
        return "⚠️ 未配置 API Key", [50, 50, 50, 50]
    
    dev_sup = ((p_curr / p_low) - 1) * 100
    try:
        client = OpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
        prompt = f"""
        你是一名严谨的量化审计师。根据以下数据对 BTDR 进行形态审计：
        现价: ${p_curr:.2f}, MA5: ${p_ma5:.2f}, 偏离回归支撑: {dev_sup:.2f}%, 实时换手: {turnover_str}。
        
        【输出指令】：
        1. 使用 Markdown 格式。标题对齐，严禁格式混乱。
        2. **必须高亮** 所有的价格数值、关键支撑/阻力位、及“背离/放量”等描述词。
        3. 结构要求：
           ## 1. 空间定位
           (分析当前价位在回归模型中的安全等级，加粗高亮支撑距离)
           ## 2. 量价审计
           (分析换手率是否健康，量价是否匹配，加粗高亮异常信号)
           ## 3. 核心研判结论
           (给出最终操作建议：**分批低吸/持币观望/逢高减仓**，并给出**明确止损价位**)
        4. 评分输出：最后一行格式为 SCORES: 动能=X, 支撑=X, 换手=X, 趋势=X
        """
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": "你只讲量化逻辑，文字简练，重点必须加粗。"}, {"role": "user", "content": prompt}]
        )
        full_text = response.choices[0].message.content
        scores = [int(n) for n in re.findall(r"\d+", re.findall(r"SCORES:.*", full_text)[0])] if re.findall(r"SCORES:.*", full_text) else [50,50,50,50]
        return re.sub(r"SCORES:.*", "", full_text).strip(), scores
    except Exception as e:
        return f"❌ AI 调用失败: {str(e)}", [0, 0, 0, 0]

# --- 3. 页面展示 ---
st.set_page_config(layout="wide", page_title="BTDR 决策终端", page_icon="🎯")
st.title("🏹 BTDR 深度量化决策终端")

try:
    hist_df, live_df, float_sh, reg, rt_v = get_btdr_data()
    last_h = hist_df.iloc[-1]
    curr_p = live_df['Close'].iloc[-1]
    
    # 回归场景计算
    today_o = live_df.between_time('09:30', '16:00')['Open'].iloc[0] if not live_df.between_time('09:30', '16:00').empty else live_df['Open'].iloc[-1]
    ratio_o = (today_o - last_h['Close']) / last_h['Close']
    p_h_pred = last_h['Close'] * (1 + (reg['inter_h'] + reg['slope_h'] * ratio_o))
    p_l_pred = last_h['Close'] * (1 + (reg['inter_l'] + reg['slope_l'] * ratio_o))
    today_to = (rt_v / float_sh) * 100

    # UI 1: 实时状态
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.subheader("📊 实时状态")
        m1, m2 = st.columns(2)
        m1.metric("当前价", f"${curr_p:.2f}", f"{(curr_p/last_h['Close']-1):.2%}")
        m2.markdown(f"**当日实时换手**\n### :{'red' if today_to > 15 else 'orange' if today_to > 8 else 'green'}[{today_to:.2f}%]")
    with c2:
        st.subheader("📍 场景回归预测")
        st.table(pd.DataFrame({"场景描述": ["中性回归", "乐观 (+6%)", "悲观 (-6%)"], "最高点": [p_h_pred, p_h_pred*1.06, p_h_pred*0.94], "最低点": [p_l_pred, p_l_pred*1.06, p_l_pred*0.94]}).style.format(precision=2))

    st.divider()

    # UI 2: AI 解析区
    col_t, col_r = st.columns([1.6, 0.9])
    with col_t:
        st.subheader("🤖 AI 硬核形态研判")
        st.markdown("""<style> div.stButton > button { background: linear-gradient(45deg, #0052D4, #6FB1FC); color:white; border:none; height:35px; width:100%; border-radius:5px; font-weight:bold; } </style>""", unsafe_allow_html=True)
        
        text, scores = get_ai_analysis_cached(round(curr_p, 2), round(last_h['MA5'], 2), f"{today_to:.1f}%", round(p_l_pred, 2), round(p_h_pred, 2))
        
        if st.button("🔄 刷新实时 AI 审计报告"):
            st.cache_data.clear()
            st.rerun()
            
        st.markdown(f"""<div style="border: 1px solid #444; padding: 20px; border-radius: 10px; border-left: 5px solid #0052D4; color: #EEE;">{text}</div>""", unsafe_allow_html=True)

    with col_r:
        fig_radar = go.Figure(data=go.Scatterpolar(r=scores + [scores[0]], theta=['动能','支撑','换手','趋势','动能'], fill='toself', line=dict(color='#0052D4', width=3)))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=350, margin=dict(l=40, r=40, t=30, b=30), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_radar, use_container_width=True)

    # UI 3: K线主图 (红涨绿跌)
    st.subheader("🕒 实时监控主图 (阳红阴绿)")
    fig_k = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    p_df = hist_df.tail(20).copy()
    p_df['label'] = p_df.index.strftime('%m/%d')
    
    fig_k.add_trace(go.Candlestick(x=p_df['label'], open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name="K线",
                                  increasing_line_color='#FF3131', decreasing_line_color='#00C805'), row=1, col=1)
    fig_k.add_trace(go.Scatter(x=p_df['label'], y=p_df['MA5'], name="MA5 (RED)", line=dict(color='#FF3131', width=2)), row=1, col=1)
    
    # 柱状图红涨绿跌
    vol_colors = ['#FF3131' if (p_df['Close'].iloc[i] >= p_df['Open'].iloc[i]) else '#00C805' for i in range(len(p_df))]
    fig_k.add_trace(go.Bar(x=p_df['label'], y=p_df['换手率_原始']*100, name="Turnover%", marker_color=vol_colors, opacity=0.8), row=2, col=1)
    
    fig_k.update_xaxes(tickangle=-90, dtick=1)
    fig_k.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_k, use_container_width=True)

    # UI 4: 历史明细
    st.subheader("📋 历史数据参考")
    show_df = hist_df.tail(15).copy()
    show_df.index = show_df.index.date
    for c in ['今开比例', '最高比例', '最低比例']:
        show_df[c] = (show_df[c] * 100).map('{:.2f}%'.format)
    show_df['换手率'] = (show_df['换手率_原始'] * 100).map('{:.2f}%'.format)
    st.dataframe(show_df[['Open', 'High', 'Low', 'Close', '今开比例', '最高比例', '最低比例', '换手率', 'MA5']].style.format(precision=2))

except Exception as e:
    st.error(f"终端运行中: {e}")
