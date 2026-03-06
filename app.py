import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI

# --- 0. 授权验证 ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    if not st.session_state.password_correct:
        st.set_page_config(layout="wide", page_title="BTDR Quant")
        st.title("🏹 BTDR 专业量化决策终端")
        pwd = st.text_input("输入访问码", type="password")
        if st.button("进入系统"):
            if pwd == st.secrets.get("ACCESS_PASSWORD", "123456"):
                st.session_state.password_correct = True
                st.rerun()
            else: st.error("访问码错误")
        st.stop()

check_password()

# --- 1. 量化引擎 (MFI资金流 + 期权计算 + 换手色标) ---
@st.cache_data(ttl=60)
def get_btdr_professional_engine():
    tk = yf.Ticker("BTDR")
    info = tk.info
    float_sh = info.get('floatShares') or info.get('shares') or 118500000
    rt_v = info.get('regularMarketVolume', 0)
    
    # --- 期权数据深度抓取 ---
    try:
        current_iv = info.get('impliedVolatility')
        # 如果 putCallRatio 为空，尝试通过期权链计算
        current_pcr = info.get('putCallRatio')
        if not current_pcr and tk.options:
            chain = tk.option_chain(tk.options[0])
            p_vol = chain.puts['volume'].sum()
            c_vol = chain.calls['volume'].sum()
            current_pcr = p_vol / c_vol if c_vol > 0 else None
    except:
        current_iv, current_pcr = None, None

    hist = tk.history(period="60d", interval="1d")
    if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
    if not hist.empty and rt_v > 0: hist.iloc[-1, hist.columns.get_loc('Volume')] = rt_v
    
    # 指标计算：MA5 + MFI
    hist['昨收'] = hist['Close'].shift(1)
    hist['MA5'] = hist['Close'].rolling(5).mean()
    hist['换手率_计算'] = (hist['Volume'] / float_sh)
    
    # 资金流量指标 MFI 计算 (14日)
    tp = (hist['High'] + hist['Low'] + hist['Close']) / 3
    rmf = tp * hist['Volume']
    pos_flow = np.where(tp > tp.shift(1), rmf, 0)
    neg_flow = np.where(tp < tp.shift(1), rmf, 0)
    mfr = pd.Series(pos_flow).rolling(14).sum() / pd.Series(neg_flow).rolling(14).sum()
    hist['MFI'] = 100 - (100 / (1 + mfr.values))
    
    inst_cost = hist['Close'].tail(20).mean()
    
    # 回归场景
    hist['今开比例'] = (hist['Open'] - hist['昨收']) / hist['昨收']
    fit_df = hist.dropna()
    X = fit_df[['今开比例']].values
    m_h = LinearRegression().fit(X, fit_df['High'].values / fit_df['昨收'].values - 1)
    m_l = LinearRegression().fit(X, fit_df['Low'].values / fit_df['昨收'].values - 1)
    reg_params = {'slope_h': m_h.coef_[0], 'inter_h': m_h.intercept_, 'slope_l': m_l.coef_[0], 'inter_l': m_l.intercept_}
    
    return fit_df, float_sh, reg_params, rt_v, inst_cost, current_iv, current_pcr

# --- 2. 深度研判 (Reasoner: 纠错机制 + 冷酷语态) ---
@st.cache_data(ttl=3600, show_spinner=False)
def get_ai_reasoner_professional(p_curr, p_ma5, turnover, p_low, p_high, inst_cost, iv, pcr, mfi):
    if "DEEPSEEK_API_KEY" not in st.secrets: return "⚠️ API Missing"
    
    opt_context = f"IV: {iv:.2%}, PCR: {pcr:.2f}" if (iv and pcr) else "期权维度数据缺失 (N/A)"
    
    client = OpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    prompt = f"""
    作为量化对冲基金审计师，对 BTDR 进行冷酷、专业的风险评估。
    【核心指标】：现价 ${p_curr:.2f} | MA5 ${p_ma5:.2f} | 换手率 {turnover:.2f}% | 机构成本锚点 ${inst_cost:.2f} | MFI资金流 {mfi:.2f} | {opt_context}。
    
    【审计任务】：
    ## 1. 筹码分布与流动性测试
    分析现价相对于机构成本的偏离，判定是否存在“阴跌测试支撑”或“筹码派发”。
    ## 2. 策略逻辑纠错 (纠错机制)
    明确指出：若股价放量站稳回归上限 ${p_high:.2f}，则当前下行逻辑失效，空头需强制平仓。
    ## 3. 核心研判结论
    给出一个基于概率的操作区间。禁止情绪化词汇，使用冷酷的金融工程描述。
    """
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "system", "content": "你是一名冷酷、理性的量化对冲基金经理，只看数据。"}, {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# --- 3. UI 渲染 ---
st.markdown("""<style> 
    .main { background-color: #FFFFFF !important; } 
    .stMarkdown, p, li, h2, h3, span { color: #1E1E1E !important; font-family: 'Segoe UI', sans-serif; } 
    h2 { color: #1A237E !important; border-bottom: 2px solid #EEE; padding-bottom: 5px; font-weight: 600; } 
    strong, b { color: #B71C1C !important; font-weight: 600; }
    /* 极简蓝色按钮：无背景色，仅边框 */
    div.stButton > button { background-color: transparent; color: #1A237E; border: 1px solid #1A237E; width: 100%; border-radius: 4px; height: 35px; transition: 0.3s; }
    div.stButton > button:hover { background-color: #1A237E; color: #FFF; }
</style>""", unsafe_allow_html=True)

st.title("🎯 BTDR 量化决策终端 (专业版)")

try:
    hist_df, dynamic_float, reg, rt_v, inst_cost, iv, pcr = get_btdr_professional_engine()
    last_h = hist_df.iloc[-1]
    curr_p = last_h['Close']
    mfi_val = last_h['MFI']
    
    today_to = (rt_v / dynamic_float) * 100
    ratio_o = (hist_df['Open'].iloc[-1] - last_h['昨收']) / last_h['昨收']
    
    p_h_mid = last_h['昨收'] * (1 + (reg['inter_h'] + reg['slope_h'] * ratio_o))
    p_l_mid = last_h['昨收'] * (1 + (reg['inter_l'] + reg['slope_l'] * ratio_o))

    # UI 1: 实时面板
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.subheader("📋 实时量化指标")
        st.metric("现价", f"${curr_p:.2f}", f"{(curr_p/last_h['昨收']-1):.2%}")
        st.write(f"资金流向 (MFI): **{mfi_val:.2f}**")
        st.write(f"期权状态: IV **{iv*100:.1f}%** | PCR **{pcr:.2f}**" if iv else "期权状态: **N/A**")
    with c2:
        st.subheader("📍 场景回归预测")
        st.table(pd.DataFrame({"场景": ["看空失效 (乐观)", "中性回归", "支撑测试 (悲观)"], "上限": [p_h_mid*1.06, p_h_mid, p_h_mid*0.94], "下限": [p_l_mid*1.06, p_l_mid, p_l_mid*0.94]}).style.format(precision=2))

    st.divider()

    # UI 2: Reasoner 审计
    st.subheader("🔬 DeepSeek-R1 量化逻辑审计")
    if st.button("🔄 刷新审计研判"):
        st.cache_data.clear()
        st.rerun()
    
    report = get_ai_reasoner_professional(curr_p, last_h['MA5'], today_to, p_l_mid, p_h_mid, inst_cost, iv, pcr, mfi_val)
    st.markdown(report)

    st.divider()

    # UI 3: K线主图 (阳红阴绿 + 换手变色)
    st.subheader("🕒 实时监控主图 (换手红涨绿跌)")
    fig_k = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    p_df = hist_df.tail(20).copy()
    p_df['label'] = p_df.index.strftime('%m/%d')
    fig_k.add_trace(go.Candlestick(x=p_df['label'], open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name="K线",
                                  increasing_line_color='#E53935', decreasing_line_color='#43A047'), row=1, col=1)
    fig_k.add_trace(go.Scatter(x=p_df['label'], y=p_df['MA5'], name="MA5", line=dict(color='#1E88E5', width=2)), row=1, col=1)
    
    # 换手率颜色：基于 K线涨跌
    vol_colors = ['#E53935' if (p_df['Close'].iloc[i] >= p_df['Open'].iloc[i]) else '#43A047' for i in range(len(p_df))]
    fig_k.add_trace(go.Bar(x=p_df['label'], y=p_df['换手率_计算']*100, name="换手率%", marker_color=vol_colors), row=2, col=1)
    fig_k.update_layout(height=550, xaxis_rangeslider_visible=False, template="plotly_white")
    st.plotly_chart(fig_k, use_container_width=True)

    # UI 4: 明细表
    st.subheader("📋 历史数据明细 (集成 MFI)")
    show_df = hist_df.tail(15).copy()
    show_df.index = show_df.index.date
    show_df['换手率'] = (show_df['换手率_计算'] * 100).map('{:.2f}%'.format)
    st.dataframe(show_df[['Open', 'High', 'Low', 'Close', '换手率', 'MFI', 'MA5']].style.format(precision=2))

except Exception as e: st.error(f"Error: {e}")
