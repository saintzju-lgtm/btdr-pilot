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

# --- 1. 数据引擎 (基数锁定与场景回归) ---
@st.cache_data(ttl=60)
def get_btdr_calibrated_data():
    tk = yf.Ticker("BTDR")
    info = tk.info
    # 锁定分母确保 3/2 等历史日期换手率精准对标 4.73%
    float_sh = 118000000 
    rt_v = info.get('regularMarketVolume', 0)
    
    hist = tk.history(period="60d", interval="1d")
    if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
    
    hist['昨收'] = hist['Close'].shift(1)
    hist['今开比例'] = (hist['Open'] - hist['昨收']) / hist['昨收']
    hist['最高比例'] = (hist['High'] - hist['昨收']) / hist['昨收']
    hist['最低比例'] = (hist['Low'] - hist['昨收']) / hist['昨收']
    hist['换手率_计算'] = (hist['Volume'] / float_sh)
    hist['MA5'] = hist['Close'].rolling(5).mean()
    
    # 回归参数训练
    fit_df = hist.dropna()
    X = fit_df[['今开比例']].values
    m_h = LinearRegression().fit(X, fit_df['最高比例'].values)
    m_l = LinearRegression().fit(X, fit_df['最低比例'].values)
    
    # 机构估算成本 (AI 审计锚点)
    inst_cost = hist['Close'].tail(20).mean()
    
    reg_params = {'slope_h': m_h.coef_[0], 'inter_h': m_h.intercept_, 'slope_l': m_l.coef_[0], 'inter_l': m_l.intercept_}
    return fit_df, float_sh, reg_params, rt_v, inst_cost

# --- 2. AI 审计引擎 (强化防幻觉与结构化) ---
@st.cache_data(ttl=3600, show_spinner=False)
def get_ai_audit_v5(p_curr, p_ma5, turnover, p_low, inst_cost):
    if "DEEPSEEK_API_KEY" not in st.secrets:
        return "⚠️ 未配置 Key"
    
    dev_sup = ((p_curr / p_low) - 1) * 100
    try:
        client = OpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
        prompt = f"""
        你是一名顶尖量化分析师。针对 BTDR 进行深度形态审计。
        现价: ${p_curr:.2f}, MA5: ${p_ma5:.2f}, 换手率: {turnover:.2f}%, 支撑偏离: {dev_sup:.2f}%, 机构成本: ${inst_cost:.2f}。
        
        【重要审计任务】：
        1. 严禁简写，严禁编造任何如“换手率68.07”等虚假数据。
        2. 必须严格按照以下 Markdown 标题输出：
           ## 1. 空间定位
           (对比现价与 MA5 及回归支撑。详细描述**支撑距离**)
           ## 2. 量价审计
           (分析换手率真实意图，指出是否存在**筹码松动**或**缩量筑底**)
           ## 3. 核心研判结论
           (给出最终结论，加粗显示**具体操作动作**及**止损位**)
        3. 重点内容使用 **加粗** 高亮。
        """
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": "实战严密逻辑，禁止幻觉，字迹清晰。"}, {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except: return "❌ 审计接口繁忙，请点击上方按钮刷新。"

# --- 3. UI 展示 ---
st.markdown("""<style> 
    .main { background-color: #FFFFFF !important; } 
    .stMarkdown, p, li, h2, h3, span { color: #000000 !important; font-family: sans-serif; } 
    h2 { color: #0044BB !important; border-bottom: 2px solid #EEE; margin-top: 20px !important; } 
    strong, b { color: #D32F2F !important; font-weight: 900 !important; }
    div.stButton > button { background-color: #FFF; color: #0044BB; border: 1px solid #0044BB; width: 100%; height: 35px; border-radius: 5px; font-weight: bold; }
    div.stButton > button:hover { background-color: #0044BB; color: #FFF; }
</style>""", unsafe_allow_html=True)

try:
    hist_df, float_sh, reg, rt_v, inst_cost = get_btdr_calibrated_data()
    last_h = hist_df.iloc[-1]
    curr_p = last_h['Close']
    
    # 换手率与回归计算
    today_to = (rt_v / float_sh) * 100
    ratio_o = (hist_df['Open'].iloc[-1] - last_h['昨收']) / last_h['昨收']
    
    # 恢复中性/乐观/悲观三种情形
    p_h_mid = last_h['昨收'] * (1 + (reg['inter_h'] + reg['slope_h'] * ratio_o))
    p_l_mid = last_h['昨收'] * (1 + (reg['inter_l'] + reg['slope_l'] * ratio_o))

    # UI 1: 实时面板
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.subheader("📊 实时状态")
        st.metric("当前成交价", f"${curr_p:.2f}", f"{(curr_p/last_h['昨收']-1):.2%}")
        st.write(f"精准换手率: **{today_to:.2f}%**")
        st.write(f"估算机构成本: **${inst_cost:.2f}**")
    with c2:
        st.subheader("📍 场景回归预测目标")
        st.table(pd.DataFrame({
            "回归场景": ["乐观场景 (+6%)", "中性回归 (模型)", "悲观场景 (-6%)"],
            "上限压力位": [p_h_mid*1.06, p_h_mid, p_h_mid*0.94],
            "下限支撑位": [p_l_mid*1.06, p_l_mid, p_l_mid*0.94]
        }).style.format(precision=2))

    st.divider()

    # UI 2: AI 解析
    st.subheader("🤖 AI 硬核形态审计 (保留记忆模式)")
    if st.button("🔄 刷新实时审计报告"):
        st.cache_data.clear()
        st.rerun()
    
    report = get_ai_audit_v5(curr_p, last_h['MA5'], today_to, p_l_mid, inst_cost)
    st.markdown(report)

    st.divider()

    # UI 3: K线主图 (阳红阴绿 + MA5红线)
    st.subheader("🕒 实时监控主图 (红涨绿跌)")
    fig_k = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    p_df = hist_df.tail(20).copy()
    p_df['label'] = p_df.index.strftime('%m/%d')
    fig_k.add_trace(go.Candlestick(x=p_df['label'], open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name="K线",
                                  increasing_line_color='#FF3131', decreasing_line_color='#00C805'), row=1, col=1)
    fig_k.add_trace(go.Scatter(x=p_df['label'], y=p_df['MA5'], name="MA5 (RED)", line=dict(color='#FF3131', width=2.5)), row=1, col=1)
    
    vol_colors = ['#FF3131' if (p_df['Close'].iloc[i] >= p_df['Open'].iloc[i]) else '#00C805' for i in range(len(p_df))]
    fig_k.add_trace(go.Bar(x=p_df['label'], y=p_df['换手率_计算']*100, name="换手率%", marker_color=vol_colors), row=2, col=1)
    fig_k.update_xaxes(tickangle=-90, dtick=1)
    fig_k.update_layout(height=550, xaxis_rangeslider_visible=False, template="plotly_white")
    st.plotly_chart(fig_k, use_container_width=True)

    # UI 4: 明细表 (完整保留)
    st.subheader("📋 历史参考数据明细")
    show_df = hist_df.tail(15).copy()
    show_df.index = show_df.index.date
    for c in ['今开比例', '最高比例', '最低比例']:
        show_df[c] = (show_df[c] * 100).map('{:.2f}%'.format)
    show_df['换手率'] = (show_df['换手率_计算'] * 100).map('{:.2f}%'.format)
    st.dataframe(show_df[['Open', 'High', 'Low', 'Close', '今开比例', '最高比例', '最低比例', '换手率', 'MA5']].style.format(precision=2))

except Exception as e: st.error(f"终端异常: {e}")
