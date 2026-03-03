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
        st.set_page_config(layout="wide", page_title="BTDR 决策终端")
        st.title("🔒 BTDR 终端安全授权")
        pwd = st.text_input("请输入访问码", type="password")
        if st.button("登录进入"):
            if pwd == st.secrets.get("ACCESS_PASSWORD", "123456"):
                st.session_state.password_correct = True
                st.rerun()
            else: st.error("授权码错误")
        st.stop()

check_password()

# --- 1. 动态数据引擎 (统一口径版) ---
@st.cache_data(ttl=60)
def get_btdr_final_pro_engine():
    tk = yf.Ticker("BTDR")
    info = tk.info
    
    # 动态获取流通股本
    current_float = info.get('floatShares') or info.get('shares') or 118500000
    
    # 获取实时成交量 (快照)
    rt_v = info.get('regularMarketVolume', 0)
    
    # 获取历史日线
    hist = tk.history(period="60d", interval="1d")
    if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
    
    # 💎 核心对齐：强制历史表最后一行(今日)使用实时成交量数据
    if not hist.empty and rt_v > 0:
        hist.iloc[-1, hist.columns.get_loc('Volume')] = rt_v
    
    # 指标计算
    hist['昨收'] = hist['Close'].shift(1)
    hist['今开比例'] = (hist['Open'] - hist['昨收']) / hist['昨收']
    hist['最高比例'] = (hist['High'] - hist['昨收']) / hist['昨收']
    hist['最低比例'] = (hist['Low'] - hist['昨收']) / hist['昨收']
    
    # 统一换手率计算公式 (基于当日/历史Volume / 最新Float)
    hist['换手率_计算'] = (hist['Volume'] / current_float)
    hist['MA5'] = hist['Close'].rolling(5).mean()
    
    # 机构成本与期权指标
    inst_cost = hist['Close'].tail(20).mean()
    iv = info.get('impliedVolatility', 0)
    pcr = info.get('putCallRatio', 0)
    
    # 回归模型训练 (中性预测基准)
    fit_df = hist.dropna()
    X = fit_df[['今开比例']].values
    m_h = LinearRegression().fit(X, fit_df['最高比例'].values)
    m_l = LinearRegression().fit(X, fit_df['最低比例'].values)
    reg_params = {'slope_h': m_h.coef_[0], 'inter_h': m_h.intercept_, 'slope_l': m_l.coef_[0], 'inter_l': m_l.intercept_}
    
    return fit_df, current_float, reg_params, rt_v, inst_cost, iv, pcr

# --- 2. AI 审计引擎 (集成期权、庄家意图与结构化分析) ---
@st.cache_data(ttl=3600, show_spinner=False)
def get_ai_audit_ultimate(p_curr, p_ma5, turnover, p_low, inst_cost, iv, pcr):
    if "DEEPSEEK_API_KEY" not in st.secrets:
        return "⚠️ 未配置 Key"
    
    dev_sup = ((p_curr / p_low) - 1) * 100
    try:
        client = OpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
        prompt = f"""
        你是一名顶尖量化分析师，审计 BTDR 筹码意图。
        数据：现价 ${p_curr:.2f}, MA5 ${p_ma5:.2f}, 实时换手 {turnover:.2f}%, 机构成本 ${inst_cost:.2f}, IV {iv:.2%}, PCR {pcr:.2f}。
        
        【审计任务】：
        ## 1. 空间定位
        分析现价与 MA5、回归下限(${p_low:.2f})的关系。加粗高亮支撑偏离度。
        ## 2. 量价审计与庄家意图
        结合机构成本 ${inst_cost:.2f}，研判当前庄家是在洗盘吸筹还是破位出货。
        ## 3. 期权雷达分析
        解读 IV({iv:.2%}) 与 PCR({pcr:.2f}) 反映的市场对冲情绪。
        ## 4. 核心研判结论
        给出具体建议（**买入/观望/止损**）及参考位。
        
        要求：全文黑字，重点**加粗**高亮。禁止数据幻觉。
        """
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": "实战严密逻辑，揭秘筹码真相。"}, {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except: return "❌ 审计繁忙，请点击刷新按钮。"

# --- 3. UI 渲染 ---
st.markdown("""<style> 
    .main { background-color: #FFFFFF !important; } 
    .stMarkdown, p, li, h2, h3, span { color: #000000 !important; font-family: sans-serif; } 
    h2 { color: #0044BB !important; border-bottom: 2px solid #EEE; padding-bottom: 5px; } 
    strong, b { color: #CC0000 !important; font-weight: 900 !important; }
    div.stButton > button { background-color: #FFF; color: #0044BB; border: 1px solid #0044BB; width: 100%; height: 35px; border-radius: 8px; font-weight: bold; }
    div.stButton > button:hover { background-color: #0044BB; color: #FFF; }
</style>""", unsafe_allow_html=True)

st.title("🏹 BTDR 深度量化决策终端")

try:
    hist_df, dynamic_float, reg, rt_v, inst_cost, iv, pcr = get_btdr_final_pro_engine()
    last_h = hist_df.iloc[-1]
    curr_p = last_h['Close']
    
    # 计算实时换手率
    today_to = (rt_v / dynamic_float) * 100
    ratio_o = (hist_df['Open'].iloc[-1] - last_h['昨收']) / last_h['昨收']
    
    # 场景回归预测 (逻辑不变)
    p_h_mid = last_h['昨收'] * (1 + (reg['inter_h'] + reg['slope_h'] * ratio_o))
    p_l_mid = last_h['昨收'] * (1 + (reg['inter_l'] + reg['slope_l'] * ratio_o))

    # UI 1: 实时面板
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.subheader("📊 实时状态")
        st.metric("当前价", f"${curr_p:.2f}", f"{(curr_p/last_h['昨收']-1):.2%}")
        st.write(f"实时流通股: **{dynamic_float/1e6:.2f} M**")
        st.write(f"动态换手率: **{today_to:.2f}%**")
        st.write(f"机构估算成本: **${inst_cost:.2f}**")
    with c2:
        st.subheader("📍 场景回归预测目标")
        st.table(pd.DataFrame({
            "回归场景": ["乐观场景 (+6%)", "中性回归 (模型)", "悲观场景 (-6%)"],
            "上限压力位": [p_h_mid*1.06, p_h_mid, p_h_mid*0.94],
            "下限支撑位": [p_l_mid*1.06, p_l_mid, p_l_mid*0.94]
        }).style.format(precision=2))

    st.divider()

    # UI 2: AI 解析
    st.subheader("🤖 DeepSeek 筹码意图与期权审计报告")
    if st.button("🔄 刷新 AI 深度审计报告"):
        st.cache_data.clear()
        st.rerun()
    
    report = get_ai_audit_ultimate(curr_p, last_h['MA5'], today_to, p_l_mid, inst_cost, iv, pcr)
    st.markdown(report)

    st.divider()

    # UI 3: K线主图 (阳红阴绿 + MA5红线)
    st.subheader("🕒 走势主图 (红色 MA5 生命线)")
    fig_k = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    p_df = hist_df.tail(20).copy()
    p_df['label'] = p_df.index.strftime('%m/%d')
    fig_k.add_trace(go.Candlestick(x=p_df['label'], open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name="K线",
                                  increasing_line_color='#FF3131', decreasing_line_color='#00C805'), row=1, col=1)
    fig_k.add_trace(go.Scatter(x=p_df['label'], y=p_df['MA5'], name="MA5", line=dict(color='#FF3131', width=2.5)), row=1, col=1)
    
    vol_colors = ['#FF3131' if (p_df['Close'].iloc[i] >= p_df['Open'].iloc[i]) else '#00C805' for i in range(len(p_df))]
    fig_k.add_trace(go.Bar(x=p_df['label'], y=p_df['换手率_计算']*100, name="换手率%", marker_color=vol_colors), row=2, col=1)
    fig_k.update_xaxes(tickangle=-90, dtick=1)
    fig_k.update_layout(height=550, xaxis_rangeslider_visible=False, template="plotly_white")
    st.plotly_chart(fig_k, use_container_width=True)

    # UI 4: 明细表 (已实现今日换手率对齐)
    st.subheader("📋 历史数据明细 (换手率已同步)")
    show_df = hist_df.tail(15).copy()
    show_df.index = show_df.index.date
    for c in ['今开比例', '最高比例', '最低比例']:
        show_df[c] = (show_df[c] * 100).map('{:.2f}%'.format)
    show_df['换手率'] = (show_df['换手率_计算'] * 100).map('{:.2f}%'.format)
    st.dataframe(show_df[['Open', 'High', 'Low', 'Close', '今开比例', '最高比例', '最低比例', '换手率', 'MA5']].style.format(precision=2))

except Exception as e: st.error(f"终端异常: {e}")
