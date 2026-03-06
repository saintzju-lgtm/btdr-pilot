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
        st.set_page_config(layout="wide", page_title="BTDR 激进终端")
        st.title("🏹 BTDR 激进交易员决策终端")
        pwd = st.text_input("输入授权码，开启杀戮模式", type="password")
        if st.button("进入战场"):
            if pwd == st.secrets.get("ACCESS_PASSWORD", "123456"):
                st.session_state.password_correct = True
                st.rerun()
            else: st.error("弱者禁入：授权码错误")
        st.stop()

check_password()

# --- 1. 动态数据引擎 (集成期权指标与历史对齐) ---
@st.cache_data(ttl=60)
def get_btdr_advanced_engine():
    tk = yf.Ticker("BTDR")
    info = tk.info
    
    # 动态获取流通股本
    current_float = info.get('floatShares') or info.get('shares') or 118500000
    rt_v = info.get('regularMarketVolume', 0)
    
    # 获取期权实时数据
    current_iv = info.get('impliedVolatility', 0)
    current_pcr = info.get('putCallRatio', 0)
    
    hist = tk.history(period="60d", interval="1d")
    if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
    
    # 💎 核心对齐：强制今日换手率数据闭环
    if not hist.empty and rt_v > 0:
        hist.iloc[-1, hist.columns.get_loc('Volume')] = rt_v
    
    hist['昨收'] = hist['Close'].shift(1)
    hist['今开比例'] = (hist['Open'] - hist['昨收']) / hist['昨收']
    hist['最高比例'] = (hist['High'] - hist['昨收']) / hist['昨收']
    hist['最低比例'] = (hist['Low'] - hist['昨收']) / hist['昨收']
    hist['换手率_计算'] = (hist['Volume'] / current_float)
    hist['MA5'] = hist['Close'].rolling(5).mean()
    
    # 模拟历史期权波动 (由于API限制，历史明细显示当前值作为参考)
    hist['IV'] = current_iv
    hist['PCR'] = current_pcr
    
    # 机构成本锚点
    inst_cost = hist['Close'].tail(20).mean()
    
    # 回归模型预测
    fit_df = hist.dropna()
    X = fit_df[['今开比例']].values
    m_h = LinearRegression().fit(X, fit_df['最高比例'].values)
    m_l = LinearRegression().fit(X, fit_df['最低比例'].values)
    reg_params = {'slope_h': m_h.coef_[0], 'inter_h': m_h.intercept_, 'slope_l': m_l.coef_[0], 'inter_l': m_l.intercept_}
    
    return fit_df, current_float, reg_params, rt_v, inst_cost, current_iv, current_pcr

# --- 2. 激进审计引擎 (DeepSeek-R1 深度推理 + 期权实时驱动) ---
@st.cache_data(ttl=3600, show_spinner=False)
def get_ai_aggressive_audit(p_curr, p_ma5, turnover, p_low, inst_cost, iv, pcr):
    if "DEEPSEEK_API_KEY" not in st.secrets:
        return "⚠️ Key未配置，猎枪没子弹"
    
    dev_sup = ((p_curr / p_low) - 1) * 100
    try:
        client = OpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
        prompt = f"""
        你是一名华尔街顶级激进操盘手，说话犀利，直戳要害。现在审计 BTDR 筹码意图。
        【实时核心数据】：
        现价: ${p_curr:.2f} | MA5: ${p_ma5:.2f} | 实时换手: {turnover:.2f}%
        机构成本锚点: ${inst_cost:.2f} | 支撑偏离: {dev_sup:.2f}%
        期权雷达: IV {iv:.2%} | PCR {pcr:.2f}
        
        【激进审计任务】：
        ## 1. 空间定位与庄家意图
        直接判定庄家是在割肉离场、自救还是诱多。对比现价与机构成本线(${inst_cost:.2f})，揭示这背后的筹码杀戮真相。
        ## 2. 期权雷达分析
        根据实时抓取的 IV({iv:.2%}) 和 PCR({pcr:.2f})，深度拆解空头是否已近强弩之末。是否有大户在通过期权市场悄悄建仓或对冲抛售？
        ## 3. 量价审计
        结合 {turnover:.2f}% 的换手率，判定这是虚假繁荣还是真正的筹码换手。
        ## 4. 核心研判指令：敢于亮剑
        禁止使用“建议、可能”。必须给出明确的方向：**全仓杀入 / 逢高清仓 / 激进埋伏 / 绝不抄底**。给出确切的撤退/止损点位。
        
        要求：文字黑字，重点**红色加粗**。逻辑要狠，不留情面。
        """
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "system", "content": "你是一名实战派操盘手，洞察庄家底牌，说话极度直接犀利。"}, {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except: return "❌ 引擎卡死，庄家在干扰信号。"

# --- 3. UI 渲染 ---
st.markdown("""<style> 
    .main { background-color: #FFFFFF !important; } 
    .stMarkdown, p, li, h2, h3, span { color: #000000 !important; font-family: sans-serif; font-size: 16px; } 
    h2 { color: #AA0000 !important; border-bottom: 3px solid #AA0000; padding-bottom: 5px; font-weight: 900 !important; } 
    strong, b { color: #FF0000 !important; font-weight: 900 !important; }
    div.stButton > button { background-color: #FFF; color: #AA0000; border: 2px solid #AA0000; width: 100%; height: 45px; font-weight: bold; font-size: 18px; border-radius: 8px; }
    div.stButton > button:hover { background-color: #AA0000; color: #FFF; }
</style>""", unsafe_allow_html=True)

st.title("🎯 BTDR 深度量化决策终端 (激进 Reasoner 版)")

try:
    hist_df, dynamic_float, reg, rt_v, inst_cost, iv, pcr = get_btdr_advanced_engine()
    last_h = hist_df.iloc[-1]
    curr_p = last_h['Close']
    
    today_to = (rt_v / dynamic_float) * 100
    ratio_o = (hist_df['Open'].iloc[-1] - last_h['昨收']) / last_h['昨收']
    
    p_h_mid = last_h['昨收'] * (1 + (reg['inter_h'] + reg['slope_h'] * ratio_o))
    p_l_mid = last_h['昨收'] * (1 + (reg['inter_l'] + reg['slope_l'] * ratio_o))

    # UI 1: 实时面板 (包含期权指标)
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.subheader("📊 实时激进数据")
        st.metric("现价", f"${curr_p:.2f}", f"{(curr_p/last_h['昨收']-1):.2%}")
        st.write(f"实时 IV: **{iv:.2%}** | PCR: **{pcr:.2f}**")
        st.write(f"动态换手率: **{today_to:.2f}%**")
        st.write(f"机构成本锚点: **${inst_cost:.2f}**")
    with c2:
        st.subheader("📍 激进回归场景预测")
        st.table(pd.DataFrame({
            "场景": ["乐观上攻 (+6%)", "中性平衡 (模型)", "悲观回撤 (-6%)"],
            "压力位(卖)": [p_h_mid*1.06, p_h_mid, p_h_mid*0.94],
            "支撑位(买)": [p_l_mid*1.06, p_l_mid, p_l_mid*0.94]
        }).style.format(precision=2))

    st.divider()

    # UI 2: AI 审计 (使用实时抓取的期权数据)
    st.subheader("🤖 DeepSeek-R1 庄家博弈审计报告 (实时期权驱动)")
    if st.button("🚀 强制刷新实战审计 (Reasoner 深度推演)"):
        st.cache_data.clear()
        st.rerun()
    
    report = get_ai_aggressive_audit(curr_p, last_h['MA5'], today_to, p_l_mid, inst_cost, iv, pcr)
    st.markdown(report)

    st.divider()

    # UI 3: K线与换手率
    st.subheader("🕒 走势主图 (MA5红线 + 阳红阴绿)")
    fig_k = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    p_df = hist_df.tail(20).copy()
    p_df['label'] = p_df.index.strftime('%m/%d')
    fig_k.add_trace(go.Candlestick(x=p_df['label'], open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name="K线",
                                  increasing_line_color='#FF3131', decreasing_line_color='#00C805'), row=1, col=1)
    fig_k.add_trace(go.Scatter(x=p_df['label'], y=p_df['MA5'], name="MA5 生命线", line=dict(color='#FF3131', width=3)), row=1, col=1)
    
    vol_colors = ['#FF3131' if (p_df['Close'].iloc[i] >= p_df['Open'].iloc[i]) else '#00C805' for i in range(len(p_df))]
    fig_k.add_trace(go.Bar(x=p_df['label'], y=p_df['换手率_计算']*100, name="换手率%", marker_color=vol_colors), row=2, col=1)
    fig_k.update_xaxes(tickangle=-90, dtick=1)
    fig_k.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_k, use_container_width=True)

    # UI 4: 明细表 (增加 IV, PCR 列)
    st.subheader("📋 历史参考数据明细 (集成期权指标)")
    show_df = hist_df.tail(15).copy()
    show_df.index = show_df.index.date
    for c in ['今开比例', '最高比例', '最低比例']:
        show_df[c] = (show_df[c] * 100).map('{:.2f}%'.format)
    show_df['换手率'] = (show_df['换手率_计算'] * 100).map('{:.2f}%'.format)
    show_df['IV%'] = (show_df['IV'] * 100).map('{:.1f}%'.format)
    st.dataframe(show_df[['Open', 'High', 'Low', 'Close', '今开比例', '换手率', 'IV%', 'PCR', 'MA5']].style.format(precision=2))

except Exception as e: st.error(f"终端异常: {e}")
