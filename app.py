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
import time

# --- 0. 授权验证 ---
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
            else:
                st.error("访问受限")
        st.stop()
        return False
    return True

# --- 辅助：期权 IV 反推 ---
def find_iv(market_price, S, K, T, r, option_type='call'):
    if market_price <= 0 or T <= 0: return None
    try:
        def bs_price(sigma):
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) if option_type == 'call' else K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            return price - market_price
        return newton(bs_price, 0.5, maxiter=50)
    except: return None

# --- 1. 量化引擎 (增加抗封锁逻辑) ---
@st.cache_data(ttl=600) # 登录后才调用，TTL 设长一点 (10分钟)
def get_btdr_quant_engine():
    tk = yf.Ticker("BTDR")
    try:
        # 强制延迟 1.5秒，模拟人工
        time.sleep(1.5)
        info = tk.info
        hist = tk.history(period="100d", interval="1d")
        if hist.empty: return None
    except:
        return None

    # 数据处理
    current_float = info.get('floatShares') or info.get('shares') or 118500000
    rt_v = info.get('regularMarketVolume', 0)
    if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
    if rt_v > 0: hist.iloc[-1, hist.columns.get_loc('Volume')] = rt_v
    
    hist['昨收'] = hist['Close'].shift(1)
    hist['MA5'] = hist['Close'].rolling(5).mean()
    hist['换手率_计算'] = (hist['Volume'] / current_float)
    
    # MFI
    tp = (hist['High'] + hist['Low'] + hist['Close']) / 3
    rmf = tp * hist['Volume']
    pos_flow = np.where(tp > tp.shift(1), rmf, 0)
    neg_flow = np.where(tp < tp.shift(1), rmf, 0)
    mfr = pd.Series(pos_flow).rolling(14).sum() / pd.Series(neg_flow).rolling(14).sum()
    hist['MFI'] = 100 - (100 / (1 + mfr.values))
    
    # 布林带 (BOLL)
    hist['MA20'] = hist['Close'].rolling(20).mean()
    hist['Std20'] = hist['Close'].rolling(20).std()
    hist['Upper'] = hist['MA20'] + (hist['Std20'] * 2)
    hist['Lower'] = hist['MA20'] - (hist['Std20'] * 2)

    inst_cost = hist['Close'].tail(20).mean()
    hist['今开比例'] = (hist['Open'] - hist['昨收']) / hist['昨收']
    
    # 场景回归
    fit_df = hist.dropna()
    X = fit_df[['今开比例']].values
    m_h = LinearRegression().fit(X, fit_df['High'].values / fit_df['昨收'].values - 1)
    m_l = LinearRegression().fit(X, fit_df['Low'].values / fit_df['昨收'].values - 1)
    reg_params = {'slope_h': m_h.coef_[0], 'inter_h': m_h.intercept_, 'slope_l': m_l.coef_[0], 'inter_l': m_l.intercept_}
    
    iv = info.get('impliedVolatility')
    if not iv:
        try:
            curr_p = hist['Close'].iloc[-1]
            exp = tk.options[0]
            opts = tk.option_chain(exp).calls
            atm = opts.iloc[(opts['strike'] - curr_p).abs().argsort()[:1]]
            iv = find_iv(atm['lastPrice'].values[0], curr_p, atm['strike'].values[0], 0.08, 0.04)
        except: iv = None

    return hist, current_float, reg_params, rt_v, inst_cost, iv, info.get('putCallRatio')

# --- 2. 深度审计 ---
def get_ai_reasoner_audit(p_curr, p_ma5, turnover, p_low, p_high, inst_cost, iv, pcr, mfi):
    if "DEEPSEEK_API_KEY" not in st.secrets: return "⚠️ API Key 缺失"
    client = OpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    prompt = f"审计数据: 现价${p_curr:.2f}, MFI {mfi:.2f}, IV {iv if iv else 'N/A'}。"
    try:
        response = client.chat.completions.create(model="deepseek-reasoner", messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content
    except: return "审计暂时无法生成。"

# --- 3. UI 渲染 ---
if check_password(): # 关键：只有登录成功才会执行以下逻辑
    st.markdown("""<style> .main { background-color: #FFFFFF !important; } .stMarkdown, p, li, h2, h3, span { color: #1E1E1E !important; } h2 { color: #1A237E !important; border-bottom: 2px solid #EEE; padding-bottom: 5px; } div.stButton > button { background-color: transparent !important; color: #1A237E !important; border: 1px solid #1A237E !important; font-weight: bold; } </style>""", unsafe_allow_html=True)

    data = get_btdr_quant_engine()
    
    if data:
        hist_df, dynamic_float, reg, rt_v, inst_cost, iv, pcr = data
        last_h = hist_df.iloc[-1]
        curr_p = last_h['Close']
        today_to = (rt_v / dynamic_float) * 100
        ratio_o = (hist_df['Open'].iloc[-1] - last_h['昨收']) / last_h['昨收']
        p_h_mid = last_h['昨收'] * (1 + (reg['inter_h'] + reg['slope_h'] * ratio_o))
        p_l_mid = last_h['昨收'] * (1 + (reg['inter_l'] + reg['slope_l'] * ratio_o))

        # --- 实时看板 ---
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.subheader("📊 实时状态")
            st.metric("现价", f"${curr_p:.2f}", f"{(curr_p/last_h['昨收']-1):.2%}")
            st.write(f"BOLL 高 (Upper): **${last_h['Upper']:.2f}**")
            st.write(f"BOLL 中 (Median): **${last_h['MA20']:.2f}**")
            st.write(f"BOLL 低 (Lower): **${last_h['Lower']:.2f}**")
            st.write(f"资金 MFI: **{last_h['MFI']:.2f}**")
        with c2:
            st.subheader("📍 场景回归预测")
            st.table(pd.DataFrame({
                "场景": ["看空失效", "中性回归", "支撑测试"],
                "压力位": [p_h_mid*1.06, p_h_mid, p_h_mid*0.94],
                "支撑位": [p_l_mid*1.06, p_l_mid, p_l_mid*0.94]
            }).style.format(precision=2))

        # --- 主图 (强化图例数值) ---
        st.divider()
        st.subheader("🕒 走势主图 (图例集成实时 BOLL)")
        fig_k = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
        p_df = hist_df.tail(40).copy()
        p_df['label'] = p_df.index.strftime('%m/%d')
        
        # 布林带图例带数值
        fig_k.add_trace(go.Scatter(x=p_df['label'], y=p_df['Upper'], line=dict(color='rgba(0,102,204,0.6)', width=1.5), name=f"BOLL High: {last_h['Upper']:.2f}"), row=1, col=1)
        fig_k.add_trace(go.Scatter(x=p_df['label'], y=p_df['Lower'], line=dict(color='rgba(0,102,204,0.6)', width=1.5), fill='tonexty', fillcolor='rgba(0,102,204,0.15)', name=f"BOLL Low: {last_h['Lower']:.2f}"), row=1, col=1)
        fig_k.add_trace(go.Scatter(x=p_df['label'], y=p_df['MA20'], line=dict(color='rgba(0,102,204,0.9)', dash='dot'), name=f"BOLL Mid: {last_h['MA20']:.2f}"), row=1, col=1)
        
        fig_k.add_trace(go.Candlestick(x=p_df['label'], open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name="K线"), row=1, col=1)
        fig_k.add_trace(go.Scatter(x=p_df['label'], y=p_df['MA5'], name=f"MA5: {last_h['MA5']:.2f}", line=dict(color='#FF9800', width=2)), row=1, col=1)
        
        vol_colors = ['#E53935' if (p_df['Close'].iloc[i] >= p_df['Open'].iloc[i]) else '#43A047' for i in range(len(p_df))]
        fig_k.add_trace(go.Bar(x=p_df['label'], y=p_df['换手率_计算']*100, name="换手率%", marker_color=vol_colors), row=2, col=1)
        
        fig_k.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_k, use_container_width=True)

        # --- 历史明细 ---
        st.subheader("📋 历史明细 (集成 MFI)")
        st.dataframe(hist_df.tail(15)[['Open', 'High', 'Low', 'Close', '换手率_计算', 'MFI', 'MA5']].style.format(precision=2), use_container_width=True)

        # --- AI 审计 (置底 & 手动) ---
        st.divider()
        st.subheader("🔬 DeepSeek-R1 审计")
        if "audit_report" not in st.session_state: st.session_state.audit_report = "等待指令..."
        if st.button("🚀 刷新 AI 研判"):
            with st.spinner("AI 逻辑审计中..."):
                st.session_state.audit_report = get_ai_reasoner_audit(curr_p, last_h['MA5'], today_to, p_l_mid, p_h_mid, inst_cost, iv, pcr, last_h['MFI'])
        st.info(st.session_state.audit_report)
        
    else:
        st.error("⚠️ Yahoo 接口被封锁。请停止操作 5 分钟，或点击右侧按钮尝试更换 IP (如使用代理)。")
        if st.button("重试"):
            st.rerun()
