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
        st.title("🔒 BTDR 终端安全授权 (激进模式已激活)")
        pwd = st.text_input("请输入访问码", type="password")
        if st.button("登录进入"):
            if pwd == st.secrets.get("ACCESS_PASSWORD", "123456"):
                st.session_state.password_correct = True
                st.rerun()
            else: st.error("授权码错误")
        st.stop()

check_password()

# --- 1. 动态数据引擎 (动态换手率 + 期权异动) ---
@st.cache_data(ttl=60)
def get_btdr_aggressive_engine():
    tk = yf.Ticker("BTDR")
    info = tk.info
    
    # 动态获取当前流通股本
    current_float = info.get('floatShares') or info.get('shares') or 118500000
    
    # 实时成交量快照 (常规时段)
    rt_v = info.get('regularMarketVolume', 0)
    
    hist = tk.history(period="60d", interval="1d")
    if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
    
    # 同步今日实时数据到历史明细
    if not hist.empty and rt_v > 0:
        hist.iloc[-1, hist.columns.get_loc('Volume')] = rt_v
    
    # 核心指标计算
    hist['昨收'] = hist['Close'].shift(1)
    hist['今开比例'] = (hist['Open'] - hist['昨收']) / hist['昨收']
    hist['最高比例'] = (hist['High'] - hist['昨收']) / hist['昨收']
    hist['最低比例'] = (hist['Low'] - hist['昨收']) / hist['昨收']
    hist['换手率_计算'] = (hist['Volume'] / current_float)
    hist['MA5'] = hist['Close'].rolling(5).mean()
    
    # 机构成本、IV、PCR
    inst_cost = hist['Close'].tail(20).mean()
    iv = info.get('impliedVolatility', 0)
    pcr = info.get('putCallRatio', 0)
    
    # 场景回归训练
    fit_df = hist.dropna()
    X = fit_df[['今开比例']].values
    m_h = LinearRegression().fit(X, fit_df['最高比例'].values)
    m_l = LinearRegression().fit(X, fit_df['最低比例'].values)
    reg_params = {'slope_h': m_h.coef_[0], 'inter_h': m_h.intercept_, 'slope_l': m_l.coef_[0], 'inter_l': m_l.intercept_}
    
    return fit_df, current_float, reg_params, rt_v, inst_cost, iv, pcr

# --- 2. 激进审计引擎 (DeepSeek-R1 深度推理) ---
@st.cache_data(ttl=3600, show_spinner=False)
def get_ai_aggressive_audit(p_curr, p_ma5, turnover, p_low, inst_cost, iv, pcr):
    if "DEEPSEEK_API_KEY" not in st.secrets:
        return "⚠️ 未配置 Key"
    
    dev_sup = ((p_curr / p_low) - 1) * 100
    try:
        client = OpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
        prompt = f"""
        你是一名风格极其激进、嗅觉敏锐的华尔街顶级操盘手。现在审计 BTDR 筹码意图。
        数据：现价 ${p_curr:.2f}, MA5 ${p_ma5:.2f}, 实时换手 {turnover:.2f}%, 机构成本 ${inst_cost:.2f}, IV {iv:.2%}, PCR {pcr:.2f}。
        
        【激进审计任务】：
        ## 1. 庄家意图穿透
        别说废话。直接根据机构成本线(${inst_cost:.2f})和现价差距，判定这是庄家在反向收割、自救护盘还是彻底弃庄。
        ## 2. 期权雷达异动
        根据 IV({iv:.2%}) 的飙升或萎缩，以及 PCR({pcr:.2f})，判断空头是否已近强弩之末，或者多头是否在疯狂反扑。
        ## 3. 空间预测与死守位
        直接给出回归支撑位(${p_low:.2f})的实战价值。
        ## 4. 核心研判指令
        语气要犀利！必须给出明确的操作方向：**全仓杀入 / 逢高果断清仓 / 激进埋伏 / 绝不抄底**。给出确切的止损/撤退点位。
        
        要求：全文黑字，重点**加粗**。逻辑要狠。
        """
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "system", "content": "你是一名实战派操盘手，说话直接，不留情面，洞察庄家底牌。"}, {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except: return "❌ 审计繁忙，请稍后刷新。"

# --- 3. UI 展示 ---
st.markdown("""<style> 
    .main { background-color: #FFFFFF !important; } 
    .stMarkdown, p, li, h2, h3, span { color: #000000 !important; font-family: sans-serif; } 
    h2 { color: #AA0000 !important; border-bottom: 2px solid #EEE; padding-bottom: 5px; } 
    strong, b { color: #FF0000 !important; font-weight: 900 !important; }
    div.stButton > button { background-color: #FFF; color: #AA0000; border: 2px solid #AA0000; width: 100%; height: 40px; font-weight: bold; border-radius: 8px; }
    div.stButton > button:hover { background-color: #AA0000; color: #FFF; }
</style>""", unsafe_allow_html=True)

st.title("🏹 BTDR 深度量化决策终端 (激进 Reasoner 版)")

try:
    hist_df, dynamic_float, reg, rt_v, inst_cost, iv, pcr = get_btdr_aggressive_engine()
    last_h = hist_df.iloc[-1]
    curr_p = last_h['Close']
    
    today_to = (rt_v / dynamic_float) * 100
    ratio_o = (hist_df['Open'].iloc[-1] - last_h['昨收']) / last_h['昨收']
    
    p_h_mid = last_h['昨收'] * (1 + (reg['inter_h'] + reg['slope_h'] * ratio_o))
    p_l_mid = last_h['昨收'] * (1 + (reg['inter_l'] + reg['slope_l'] * ratio_o))

    # UI 1: 实时面板
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.subheader("📊 实时激进快照")
        st.metric("现价", f"${curr_p:.2f}", f"{(curr_p/last_h['昨收']-1):.2%}")
        st.write(f"实时流通股数: **{dynamic_float/1e6:.2f} M**")
        st.write(f"精准换手率: **{today_to:.2f}%**")
        st.write(f"机构平均成本: **${inst_cost:.2f}**")
    with c2:
        st.subheader("📍 激进场景回归目标")
        st.table(pd.DataFrame({
            "回归场景": ["乐观上攻 (+6%)", "中性平衡 (模型)", "悲观回撤 (-6%)"],
            "压力位(卖)": [p_h_mid*1.06, p_h_mid, p_h_mid*0.94],
            "支撑位(买)": [p_l_mid*1.06, p_l_mid, p_l_mid*0.94]
        }).style.format(precision=2))

    st.divider()

    # UI 2: AI 审计 (Reasoner 激进模式)
    st.subheader("🤖 DeepSeek-R1 庄家博弈审计报告")
    if st.button("🔄 刷新实战审计报告"):
        st.cache_data.clear()
        st.rerun()
    
    report = get_ai_aggressive_audit(curr_p, last_h['MA5'], today_to, p_l_mid, inst_cost, iv, pcr)
    st.markdown(report)

    st.divider()

    # UI 3: K线与换手率
    st.subheader("🕒 激进趋势图 (MA5红线)")
    fig_k = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    p_df = hist_df.tail(20).copy()
    p_df['label'] = p_df.index.strftime('%m/%d')
    fig_k.add_trace(go.Candlestick(x=p_df['label'], open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name="K线",
                                  increasing_line_color='#FF3131', decreasing_line_color='#00C805'), row=1, col=1)
    fig_k.add_trace(go.Scatter(x=p_df['label'], y=p_df['MA5'], name="MA5 生命线", line=dict(color='#FF3131', width=3)), row=1, col=1)
    
    vol_colors = ['#FF3131' if (p_df['Close'].iloc[i] >= p_df['Open'].iloc[i]) else '#00C805' for i in range(len(p_df))]
    fig_k.add_trace(go.Bar(x=p_df['label'], y=p_df['换手率_计算']*100, name="换手率%", marker_color=vol_colors), row=2, col=1)
    fig_k.update_xaxes(tickangle=-90, dtick=1)
    fig_k.update_layout(height=550, xaxis_rangeslider_visible=False, template="plotly_white")
    st.plotly_chart(fig_k, use_container_width=True)

    # UI 4: 明细表
    st.subheader("📋 历史参考数据明细")
    show_df = hist_df.tail(15).copy()
    show_df.index = show_df.index.date
    for c in ['今开比例', '最高比例', '最低比例']:
        show_df[c] = (show_df[c] * 100).map('{:.2f}%'.format)
    show_df['换手率'] = (show_df['换手率_计算'] * 100).map('{:.2f}%'.format)
    st.dataframe(show_df[['Open', 'High', 'Low', 'Close', '今开比例', '最高比例', '最低比例', '换手率', 'MA5']].style.format(precision=2))

except Exception as e: st.error(f"终端运行中: {e}")
