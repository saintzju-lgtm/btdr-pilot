import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
from scipy.stats import norm
from scipy.optimize import newton

# --- 0. 授权验证 (原封不动) ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    if not st.session_state.password_correct:
        st.set_page_config(layout="wide", page_title="BTDR Quant")
        st.title("🎯 BTDR 专业量化决策终端")
        pwd = st.text_input("输入访问码", type="password")
        if st.button("进入系统"):
            if pwd == st.secrets.get("ACCESS_PASSWORD", "123456"):
                st.session_state.password_correct = True
                st.rerun()
            else: st.error("访问受限")
        st.stop()

check_password()

# --- 新增辅助：期权 IV 反推函数 ---
def find_iv(market_price, S, K, T, r, option_type='call'):
    if market_price <= 0 or T <= 0: return None
    try:
        def bs_price(sigma):
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if option_type == 'call':
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            return price - market_price
        return newton(bs_price, 0.5, maxiter=50)
    except: return None

# --- 1. 量化引擎 (集成 MFI 与期权容错 + 新增布林带) ---
@st.cache_data(ttl=60)
def get_btdr_quant_engine():
    tk = yf.Ticker("BTDR")
    info = tk.info
    current_float = info.get('floatShares') or info.get('shares') or 118500000
    rt_v = info.get('regularMarketVolume', 0)
    
    # 期权指标容错
    raw_iv = info.get('impliedVolatility')
    raw_pcr = info.get('putCallRatio')
    pcr = raw_pcr if raw_pcr and raw_pcr > 0 else None
    
    hist = tk.history(period="100d", interval="1d") # 延长周期以计算布林带
    if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
    if not hist.empty and rt_v > 0:
        hist.iloc[-1, hist.columns.get_loc('Volume')] = rt_v
    
    # 基础指标计算 (原逻辑)
    hist['昨收'] = hist['Close'].shift(1)
    hist['MA5'] = hist['Close'].rolling(5).mean()
    hist['换手率_计算'] = (hist['Volume'] / current_float)
    
    # 资金流量指标 MFI 计算 (原逻辑)
    tp = (hist['High'] + hist['Low'] + hist['Close']) / 3
    rmf = tp * hist['Volume']
    pos_flow = np.where(tp > tp.shift(1), rmf, 0)
    neg_flow = np.where(tp < tp.shift(1), rmf, 0)
    mfr = pd.Series(pos_flow).rolling(14).sum() / pd.Series(neg_flow).rolling(14).sum()
    hist['MFI'] = 100 - (100 / (1 + mfr.values))
    
    # 新增：布林带计算
    hist['MA20'] = hist['Close'].rolling(20).mean()
    hist['Std20'] = hist['Close'].rolling(20).std()
    hist['Upper'] = hist['MA20'] + (hist['Std20'] * 2)
    hist['Lower'] = hist['MA20'] - (hist['Std20'] * 2)

    inst_cost = hist['Close'].tail(20).mean()
    
    # 场景回归 (原逻辑)
    hist['今开比例'] = (hist['Open'] - hist['昨收']) / hist['昨收']
    fit_df = hist.dropna()
    X = fit_df[['今开比例']].values
    m_h = LinearRegression().fit(X, fit_df['High'].values / fit_df['昨收'].values - 1)
    m_l = LinearRegression().fit(X, fit_df['Low'].values / fit_df['昨收'].values - 1)
    reg_params = {'slope_h': m_h.coef_[0], 'inter_h': m_h.intercept_, 'slope_l': m_l.coef_[0], 'inter_l': m_l.intercept_}
    
    # 实时 IV 增强逻辑
    iv = raw_iv if raw_iv and raw_iv > 0 else None
    if not iv:
        try:
            curr_p = hist['Close'].iloc[-1]
            exp = tk.options[0]
            opts = tk.option_chain(exp).calls
            atm = opts.iloc[(opts['strike'] - curr_p).abs().argsort()[:1]]
            iv = find_iv(atm['lastPrice'].values[0], curr_p, atm['strike'].values[0], 0.08, 0.04)
        except: pass

    return hist, current_float, reg_params, rt_v, inst_cost, iv, pcr

# --- 2. 深度审计 (Reasoner: 保留原 Prompt 并增强) ---
@st.cache_data(ttl=3600, show_spinner=False)
def get_ai_reasoner_audit(p_curr, p_ma5, turnover, p_low, p_high, inst_cost, iv, pcr, mfi):
    if "DEEPSEEK_API_KEY" not in st.secrets: return "⚠️ API Missing"
    opt_context = f"IV: {iv:.2%}, PCR: {pcr:.2f}" if iv and pcr else "期权维度 N/A"
    client = OpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    prompt = f"""
    作为量化审计师，请基于以下数据评估风险。要求：语气冷酷、客观，严禁使用感性词汇。
    【核心数据】：
    现价: ${p_curr:.2f} | MA5: ${p_ma5:.2f} | 换手率: {turnover:.2f}% | 机构成本: ${inst_cost:.2f} | MFI: {mfi:.2f} | {opt_context}
    【审计任务】：
    ## 1. 筹码博弈评估
    ## 2. 资金流向博弈 (结合 MFI)
    ## 3. 逻辑失效纠错点: 若股价放量站稳回归上限 ${p_high:.2f}，空头判定失效。
    ## 4. 核心研判结论: 给出操作区间建议。
    """
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "system", "content": "你是一名冷酷理性的对冲基金分析师。"}, {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# --- 3. UI 样式 (原封不动) ---
st.markdown("""<style> 
    .main { background-color: #FFFFFF !important; } 
    .stMarkdown, p, li, h2, h3, span { color: #1E1E1E !important; } 
    h2 { color: #1A237E !important; border-bottom: 2px solid #EEE; padding-bottom: 5px; } 
    strong, b { color: #B71C1C !important; }
    div.stButton > button { background-color: transparent !important; color: #1A237E !important; border: 1px solid #1A237E !important; font-weight: bold; }
    div.stButton > button:hover { background-color: #1A237E !important; color: #FFF !important; }
</style>""", unsafe_allow_html=True)

try:
    hist_df, dynamic_float, reg, rt_v, inst_cost, iv, pcr = get_btdr_quant_engine()
    last_h = hist_df.iloc[-1]
    curr_p = last_h['Close']
    mfi_val = last_h['MFI']
    today_to = (rt_v / dynamic_float) * 100
    ratio_o = (hist_df['Open'].iloc[-1] - last_h['昨收']) / last_h['昨收']
    
    p_h_mid = last_h['昨收'] * (1 + (reg['inter_h'] + reg['slope_h'] * ratio_o))
    p_l_mid = last_h['昨收'] * (1 + (reg['inter_l'] + reg['slope_l'] * ratio_o))

    # UI 1: 看板 (保留原版，并在空位增加回测胜率展示)
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.subheader("📊 实时量化看板")
        st.metric("现价", f"${curr_p:.2f}", f"{(curr_p/last_h['昨收']-1):.2%}")
        st.write(f"实时换手率: **{today_to:.2f}%**")
        st.write(f"机构成本锚点: **${inst_cost:.2f}**")
        st.write(f"资金流向 (MFI): **{mfi_val:.2f}**")
        st.write(f"期权状态 (IV): **{f'{iv:.2%}' if iv else 'N/A'}**")
        
        # 新增：简单回测展示
        test_df = hist_df.tail(60).copy()
        test_df['pred_l'] = test_df['昨收'] * (1 + (reg['inter_l'] + reg['slope_l'] * test_df['今开比例']))
        hits = test_df[test_df['Low'] <= test_df['pred_l']].dropna()
        win_rate = (hits['Close'].shift(-1) > hits['Close']).sum() / len(hits) if len(hits) > 0 else 0
        st.write(f"回归支撑位回测胜率: **{win_rate:.1%}**")

    with c2:
        st.subheader("📍 场景回归预测 (逻辑参考)")
        st.table(pd.DataFrame({
            "场景": ["看空失效点 (乐观)", "中性回归", "支撑测试位 (悲观)"],
            "压力位(上限)": [p_h_mid*1.06, p_h_mid, p_h_mid*0.94],
            "支撑位(下限)": [p_l_mid*1.06, p_l_mid, p_l_mid*0.94]
        }).style.format(precision=2))

    st.divider()

    # UI 2: Reasoner 审计 (原封不动)
    st.subheader("🔬 DeepSeek-R1 量化逻辑审计")
    if st.button("🔄 刷新研判"):
        st.cache_data.clear()
        st.rerun()
    report = get_ai_reasoner_audit(curr_p, last_h['MA5'], today_to, p_l_mid, p_h_mid, inst_cost, iv, pcr, mfi_val)
    st.markdown(report)

    st.divider()

    # UI 3: K线与换手柱 (增强：加入布林带共振可视化)
    st.subheader("🕒 走势主图 (MA5 + 布林带 + 换手色标)")
    fig_k = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    p_df = hist_df.tail(30).copy()
    p_df['label'] = p_df.index.strftime('%m/%d')
    
    # K线
    fig_k.add_trace(go.Candlestick(x=p_df['label'], open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name="K线",
                                 increasing_line_color='#E53935', decreasing_line_color='#43A047'), row=1, col=1)
    # MA5
    fig_k.add_trace(go.Scatter(x=p_df['label'], y=p_df['MA5'], name="MA5", line=dict(color='#1E88E5', width=2)), row=1, col=1)
    # 新增：布林带阴影层
    fig_k.add_trace(go.Scatter(x=p_df['label'], y=p_df['Upper'], line=dict(color='rgba(150,150,150,0.1)'), showlegend=False), row=1, col=1)
    fig_k.add_trace(go.Scatter(x=p_df['label'], y=p_df['Lower'], line=dict(color='rgba(150,150,150,0.1)'), fill='tonexty', fillcolor='rgba(150,150,150,0.05)', name="布林带区间"), row=1, col=1)
    
    vol_colors = ['#E53935' if (p_df['Close'].iloc[i] >= p_df['Open'].iloc[i]) else '#43A047' for i in range(len(p_df))]
    fig_k.add_trace(go.Bar(x=p_df['label'], y=p_df['换手率_计算']*100, name="换手率%", marker_color=vol_colors), row=2, col=1)
    fig_k.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_k, use_container_width=True)

    # UI 4: 明细 (原封不动，集成 MFI)
    st.subheader("📋 历史明细 (集成 MFI)")
    show_df = hist_df.tail(15).copy()
    show_df.index = show_df.index.date
    show_df['换手率'] = (show_df['换手率_计算'] * 100).map('{:.2f}%'.format)
    st.dataframe(show_df[['Open', 'High', 'Low', 'Close', '换手率', 'MFI', 'MA5']].style.format(precision=2))

except Exception as e: st.error(f"Error: {e}")
