import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
import re

# --- 0. 授权验证 ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    if not st.session_state.password_correct:
        st.set_page_config(page_title="终端授权", page_icon="🔒")
        st.title("🔒 BTDR 终端安全授权")
        pwd = st.text_input("请输入访问码", type="password")
        if st.button("登录系统"):
            if pwd == st.secrets.get("ACCESS_PASSWORD", "123456"):
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("授权码错误")
        st.stop()

check_password()

# --- 1. 数据引擎 (实时校准版) ---
@st.cache_data(ttl=60)
def get_btdr_data_pro():
    tk = yf.Ticker("BTDR")
    # 抓取包含常规时段成交量的快照
    info = tk.info
    rt_vol = info.get('regularMarketVolume', 0)
    float_sh = info.get('floatShares', 118000000)
    
    hist = tk.history(period="60d", interval="1d")
    if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
    
    live_df = tk.history(period="1d", interval="1m", prepost=False)
    if rt_vol == 0: rt_vol = live_df['Volume'].sum()
    
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
    return fit_df, live_df, float_sh, reg_params, rt_vol

# --- 2. AI 审计引擎 (强化防幻觉指令) ---
@st.cache_data(ttl=3600, show_spinner=False)
def get_ai_audit_pro(p_curr, p_ma5, turnover_str, p_low, p_high):
    if "DEEPSEEK_API_KEY" not in st.secrets:
        return "⚠️ 未配置 API Key", [50, 50, 50, 50]
    
    dev_sup = ((p_curr / p_low) - 1) * 100
    try:
        client = OpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
        prompt = f"""
        你是一名严谨的量化审计师。根据以下参数提供审计报告：
        现价 ${p_curr:.2f}, MA5 ${p_ma5:.2f}, 支撑偏离 {dev_sup:.2f}%, 实时换手率 {turnover_str}。
        
        【重要审计规则】：
        1. 必须使用详尽的逻辑描述，禁止大幅删减分析内容。
        2. **禁止出现虚假数据**：如“换手率68%”等幻觉描述，必须基于提供的 {turnover_str} 进行研判。
        3. 格式规范：
           ## 1. 空间定位
           对比现价与 MA5 和回归支撑位，详细分析下跌动能。
           ## 2. 量价审计
           分析当前换手率是否处于缩量筑底还是恐慌抛售状态。
           ## 3. 核心研判结论
           给出结论（如 **持币观望**）和 **明确的参考止损位**。
        4. 关键点请使用 **加粗** 处理。最后一行格式: SCORES: 动能=X, 支撑=X, 换手=X, 趋势=X
        """
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": "你只讲实战逻辑，重点必须加粗。"}, {"role": "user", "content": prompt}]
        )
        full_text = response.choices[0].message.content
        scores = [int(n) for n in re.findall(r"\d+", re.findall(r"SCORES:.*", full_text)[0])] if re.findall(r"SCORES:.*", full_text) else [50,50,50,50]
        return re.sub(r"SCORES:.*", "", full_text).strip(), scores
    except:
        return "❌ 引擎繁忙，请稍后刷新。", [0,0,0,0]

# --- 3. UI 渲染 (全局白底黑字) ---
st.set_page_config(layout="wide", page_title="BTDR 决策终端")

# CSS 强制修正：白背景、黑文字、亮蓝标题、红加粗重点
st.markdown("""
    <style>
    .main { background-color: #FFFFFF !important; }
    /* 文字颜色修正 */
    .stMarkdown, p, li, h2, h3, div { color: #000000 !important; font-family: sans-serif; }
    h2 { color: #0044BB !important; border-bottom: 2px solid #EEE; padding-bottom: 5px; margin-top: 25px !important; }
    /* 重点高亮修正 */
    strong, b { color: #CC0000 !important; font-weight: bold !important; }
    /* 按钮重塑：取消黑背景，使用蓝边框 */
    div.stButton > button {
        background-color: #FFFFFF; color: #0044BB; border: 2px solid #0044BB; 
        width: 100%; height: 45px; font-weight: bold; border-radius: 8px; transition: 0.3s;
    }
    div.stButton > button:hover { background-color: #0044BB; color: #FFFFFF; }
    /* 表格颜色 */
    .stTable { color: black !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🏹 BTDR 深度量化决策终端")

try:
    hist_df, live_df, float_sh, reg, rt_vol = get_btdr_data_pro()
    curr_p = live_df['Close'].iloc[-1]
    last_h = hist_df.iloc[-1]
    
    # 精准数据计算
    today_to = (rt_vol / float_sh) * 100
    today_o = live_df['Open'].iloc[0]
    ratio_o = (today_o - last_h['Close']) / last_h['Close']
    p_h_pred = last_h['Close'] * (1 + (reg['inter_h'] + reg['slope_h'] * ratio_o))
    p_l_pred = last_h['Close'] * (1 + (reg['inter_l'] + reg['slope_l'] * ratio_o))

    # UI 1: 顶部状态
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.subheader("📊 实时状态")
        st.metric("当前价", f"${curr_p:.2f}", f"{(curr_p/last_h['Close']-1):.2%}")
        st.write(f"实时换手率 (Regular): **{today_to:.2f}%**")
    with c2:
        st.subheader("📍 场景回归预测")
        st.table(pd.DataFrame({"场景": ["中性回归", "乐观(+6%)", "悲观(-6%)"], "上限": [p_h_pred, p_h_pred*1.06, p_h_pred*0.94], "下限": [p_l_pred, p_l_pred*1.06, p_l_pred*0.94]}).style.format(precision=2))

    st.divider()

    # UI 2: AI 审计
    col_t, col_r = st.columns([1.6, 0.9])
    with col_t:
        st.subheader("🤖 AI 硬核形态审计")
        text, scores = get_ai_audit_pro(curr_p, last_h['MA5'], f"{today_to:.2f}%", p_l_pred, p_h_pred)
        if st.button("🔄 刷新审计报告 (保留结果)"):
            st.cache_data.clear()
            st.rerun()
        st.markdown(text)

    with col_r:
        fig_radar = go.Figure(data=go.Scatterpolar(r=scores + [scores[0]], theta=['动能','支撑','换手','趋势','动能'], fill='toself', line=dict(color='#0044BB', width=2)))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=350, margin=dict(l=40, r=40, t=30, b=30), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # UI 3: K线主图
    st.subheader("🕒 实时监控主图 (阳红阴绿)")
    fig_k = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    p_df = hist_df.tail(20).copy()
    p_df['label'] = p_df.index.strftime('%m/%d')
    fig_k.add_trace(go.Candlestick(x=p_df['label'], open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name="K线",
                                  increasing_line_color='#FF3131', decreasing_line_color='#00C805'), row=1, col=1)
    fig_k.add_trace(go.Scatter(x=p_df['label'], y=p_df['MA5'], name="MA5", line=dict(color='#FF3131', width=2)), row=1, col=1)
    
    vol_colors = ['#FF3131' if (p_df['Close'].iloc[i] >= p_df['Open'].iloc[i]) else '#00C805' for i in range(len(p_df))]
    fig_k.add_trace(go.Bar(x=p_df['label'], y=p_df['换手率_原始']*100, name="Vol%", marker_color=vol_colors), row=2, col=1)
    fig_k.update_xaxes(tickangle=-90, dtick=1)
    fig_k.update_layout(height=550, xaxis_rangeslider_visible=False, template="plotly_white")
    st.plotly_chart(fig_k, use_container_width=True)

    # UI 4: 明细表回归
    st.subheader("📋 历史数据明细 (保留不改)")
    show_df = hist_df.tail(15).copy()
    show_df.index = show_df.index.date
    for c in ['今开比例', '最高比例', '最低比例']:
        show_df[c] = (show_df[c] * 100).map('{:.2f}%'.format)
    show_df['换手率'] = (show_df['换手率_原始'] * 100).map('{:.2f}%'.format)
    st.dataframe(show_df[['Open', 'High', 'Low', 'Close', '今开比例', '最高比例', '最低比例', '换手率', 'MA5']].style.format(precision=2))

except Exception as e:
    st.error(f"终端运行中: {e}")
