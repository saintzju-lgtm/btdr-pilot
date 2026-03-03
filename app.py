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
        st.set_page_config(page_title="授权登录", page_icon="🔒")
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

# --- 2. AI 决策引擎 (强化全内容输出与黑色字体指令) ---
@st.cache_data(ttl=3600, show_spinner=False)
def get_ai_analysis_cached(p_curr, p_ma5, turnover_str, p_low, p_high):
    if "DEEPSEEK_API_KEY" not in st.secrets:
        return "⚠️ 未配置 API Key", [50, 50, 50, 50]
    
    dev_sup = ((p_curr / p_low) - 1) * 100
    try:
        client = OpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
        prompt = f"""
        你是一名资深量化交易员。请针对 BTDR 进行深度形态审计。
        当前数据：现价 ${p_curr:.2f}，MA5 ${p_ma5:.2f}，偏离回归支撑位 {dev_sup:.2f}%，换手率 {turnover_str}。

        【输出规范】：
        1. 必须保留所有详细的逻辑分析，禁止删减。
        2. 结构必须包含：
           ## 1. 空间定位
           ## 2. 量价审计
           ## 3. 核心研判结论
        3. 对关键数值、核心位、操作结论（如 **持币观望**、**止损价**）进行 **加粗** 高亮。
        4. 语气要专业且直接。最后一行格式为 SCORES: 动能=X, 支撑=X, 换手=X, 趋势=X
        """
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": "你提供内容详实、逻辑严密的专业分析报告，重点突出。"}, {"role": "user", "content": prompt}]
        )
        full_text = response.choices[0].message.content
        scores = [int(n) for n in re.findall(r"\d+", re.findall(r"SCORES:.*", full_text)[0])] if re.findall(r"SCORES:.*", full_text) else [50,50,50,50]
        return re.sub(r"SCORES:.*", "", full_text).strip(), scores
    except Exception as e:
        return f"❌ AI 调用失败: {str(e)}", [0, 0, 0, 0]

# --- 3. 页面展示 ---
st.set_page_config(layout="wide", page_title="BTDR 决策终端", page_icon="📈")

# 核心 CSS 优化：强制黑字，移除所有背景和边框干扰
st.markdown("""
    <style>
    /* 页面背景设为纯白/浅灰白以适配黑字 */
    .main { background-color: #FFFFFF !important; }
    
    /* 按钮样式 */
    div.stButton > button {
        background-color: #000000; color: white; border-radius: 5px; height: 45px; width: 100%; font-weight: bold;
    }
    
    /* 强制 AI 研判文字为黑色，字号清晰 */
    .ai-text-container {
        color: #000000 !important;
        font-size: 18px !important;
        line-height: 1.8 !important;
        font-family: sans-serif;
    }
    
    /* 标题颜色与对齐 */
    h2 { color: #003366 !important; font-weight: bold !important; border-bottom: 2px solid #EEE; padding-bottom: 8px; margin-top: 30px !important; }
    
    /* 修正表格显示 */
    .stTable { color: black !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🏹 BTDR 深度量化决策终端")

try:
    hist_df, live_df, float_sh, reg, rt_v = get_btdr_data()
    last_h = hist_df.iloc[-1]
    curr_p = live_df['Close'].iloc[-1]
    
    # 回归计算
    today_o = live_df.between_time('09:30', '16:00')['Open'].iloc[0] if not live_df.between_time('09:30', '16:00').empty else live_df['Open'].iloc[-1]
    ratio_o = (today_o - last_h['Close']) / last_h['Close']
    p_h_pred = last_h['Close'] * (1 + (reg['inter_h'] + reg['slope_h'] * ratio_o))
    p_l_pred = last_h['Close'] * (1 + (reg['inter_l'] + reg['slope_l'] * ratio_o))
    today_to = (rt_v / float_sh) * 100

    # UI 1: 实时数据
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.subheader("📊 实时状态")
        m1, m2 = st.columns(2)
        m1.metric("当前成交价", f"${curr_p:.2f}", f"{(curr_p/last_h['Close']-1):.2%}")
        m2.markdown(f"**实时换手率**\n### :{'red' if today_to > 15 else 'orange' if today_to > 8 else 'green'}[{today_to:.2f}%]")
    with c2:
        st.subheader("📍 场景回归预测")
        st.table(pd.DataFrame({"场景描述": ["中性回归", "乐观 (+6%)", "悲观 (-6%)"], "最高点": [p_h_pred, p_h_pred*1.06, p_h_pred*0.94], "最低点": [p_l_pred, p_l_pred*1.06, p_l_pred*0.94]}).style.format(precision=2))

    st.divider()

    # UI 2: AI 解析区（纯净模式，强制黑字）
    col_t, col_r = st.columns([1.6, 0.9])
    with col_t:
        st.subheader("🤖 AI 硬核形态研判 (审计模式)")
        
        text, scores = get_ai_analysis_cached(round(curr_p, 2), round(last_h['MA5'], 2), f"{today_to:.1f}%", round(p_l_pred, 2), round(p_h_mid if 'p_h_mid' in locals() else p_h_pred, 2))
        
        if st.button("🔄 刷新实时研判报告 (点击更新)"):
            st.cache_data.clear()
            st.rerun()
            
        # 使用 div 包装但仅用于强制颜色，不带边框背景
        st.markdown(f'<div class="ai-text-container">{text}</div>', unsafe_allow_html=True)

    with col_r:
        fig_radar = go.Figure(data=go.Scatterpolar(r=scores + [scores[0]], theta=['动能','支撑','换手','趋势','动能'], fill='toself', line=dict(color='#003366', width=3)))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=350, margin=dict(l=40, r=40, t=30, b=30), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # UI 3: K线图 (红涨绿跌)
    st.subheader("🕒 实时监控主图 (红涨绿跌)")
    fig_k = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    p_df = hist_df.tail(20).copy()
    p_df['label'] = p_df.index.strftime('%m/%d')
    
    fig_k.add_trace(go.Candlestick(x=p_df['label'], open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name="K线",
                                  increasing_line_color='#FF3131', decreasing_line_color='#00C805'), row=1, col=1)
    fig_k.add_trace(go.Scatter(x=p_df['label'], y=p_df['MA5'], name="MA5 (RED)", line=dict(color='#FF3131', width=2)), row=1, col=1)
    
    vol_colors = ['#FF3131' if (p_df['Close'].iloc[i] >= p_df['Open'].iloc[i]) else '#00C805' for i in range(len(p_df))]
    fig_k.add_trace(go.Bar(x=p_df['label'], y=p_df['换手率_原始']*100, name="Turnover%", marker_color=vol_colors, opacity=0.8), row=2, col=1)
    
    fig_k.update_xaxes(tickangle=-90, dtick=1)
    fig_k.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_k, use_container_width=True)

    # UI 4: 明细表
    st.subheader("📋 历史参考数据明细")
    show_df = hist_df.tail(15).copy()
    show_df.index = show_df.index.date
    for c in ['今开比例', '最高比例', '最低比例']:
        show_df[c] = (show_df[c] * 100).map('{:.2f}%'.format)
    show_df['换手率'] = (show_df['换手率_原始'] * 100).map('{:.2f}%'.format)
    st.dataframe(show_df[['Open', 'High', 'Low', 'Close', '今开比例', '最高比例', '最低比例', '换手率', 'MA5']].style.format(precision=2))

except Exception as e:
    st.error(f"终端运行中: {e}")
