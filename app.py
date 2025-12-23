import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
import altair as alt
from datetime import datetime
import pytz

# --- 1. 页面配置 & 样式 ---
st.set_page_config(page_title="BTDR Pilot v9.6 Quant", layout="centered")

CUSTOM_CSS = """
<style>
    html { overflow-y: scroll; }
    .stApp > header { display: none; }
    .stApp { margin-top: -30px; background-color: #ffffff; }
    div[data-testid="stStatusWidget"] { visibility: hidden; }
    
    div[data-testid="stAltairChart"] {
        height: 320px !important; min-height: 320px !important;
        overflow: hidden !important; border: 1px solid #f8f9fa;
    }
    
    .metric-card {
        background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02); height: 95px; padding: 0 16px;
        display: flex; flex-direction: column; justify-content: center;
    }
    .metric-label { font-size: 0.75rem; color: #888; margin-bottom: 2px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #212529; line-height: 1.2; }
    .metric-delta { font-size: 0.9rem; font-weight: 600; margin-top: 2px; }
    
    .factor-box {
        background: #fff; border: 1px solid #eee; border-radius: 8px; padding: 6px; text-align: center;
        height: 75px; display: flex; flex-direction: column; justify-content: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02); position: relative; cursor: help;
    }
    .factor-title { font-size: 0.65rem; color: #999; text-transform: uppercase; }
    .factor-val { font-size: 1.1rem; font-weight: bold; color: #495057; margin: 2px 0; }
    .factor-sub { font-size: 0.7rem; font-weight: 600; }
    
    .color-up { color: #0ca678; } .color-down { color: #d6336c; }
    
    .status-dot { height: 6px; width: 6px; border-radius: 50%; display: inline-block; margin-left: 6px; margin-bottom: 2px; }
    .dot-pre { background-color: #f59f00; box-shadow: 0 0 4px #f59f00; }
    .dot-reg { background-color: #0ca678; box-shadow: 0 0 4px #0ca678; }
    .dot-post { background-color: #1c7ed6; box-shadow: 0 0 4px #1c7ed6; }
    .dot-night { background-color: #7048e8; box-shadow: 0 0 4px #7048e8; }
    .dot-closed { background-color: #adb5bd; }
    
    .pred-container-wrapper { height: 110px; width: 100%; display: block; margin-top: 5px; }
    .pred-box { padding: 0 10px; border-radius: 12px; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center; }
    
    .time-bar { font-size: 0.75rem; color: #999; text-align: center; margin-bottom: 20px; padding: 6px; background: #fafafa; border-radius: 6px; }
    .badge-trend { background:#fd7e14; color:white; padding:1px 4px; border-radius:3px; font-size:0.6rem; }
    .badge-chop { background:#868e96; color:white; padding:1px 4px; border-radius:3px; font-size:0.6rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- 2. 辅助函数 ---
def card_html(label, value_str, delta_str=None, delta_val=0, extra_tag=""):
    delta_html = ""
    if delta_str:
        color_class = "color-up" if delta_val >= 0 else "color-down"
        delta_html = f"<div class='metric-delta {color_class}'>{delta_str}</div>"
    return f"""<div class="metric-card"><div class="metric-label">{label} {extra_tag}</div><div class="metric-value">{value_str}</div>{delta_html}</div>"""

def factor_html(title, val, delta_str, delta_val, reverse_color=False):
    is_positive = delta_val >= 0
    if reverse_color: is_positive = not is_positive
    color_class = "color-up" if is_positive else "color-down"
    return f"""<div class="factor-box"><div class="factor-title">{title}</div><div class="factor-val">{val}</div><div class="factor-sub {color_class}">{delta_str}</div></div>"""

# --- 3. 核心计算逻辑 (Quant Upgrade) ---
@st.cache_data(ttl=600)
def run_quant_analytics():
    default_model = {"high": {"base": 1.04}, "low": {"base": 0.96}, "beta_sector": 0.25}
    default_factors = {"vwap": 0, "adx": 20, "regime": "Neutral", "beta_btc": 1.5, "beta_qqq": 1.2, "rsi": 50, "vol_base": 0.05, "jump_prob": 0.1}
    
    try:
        data = yf.download("BTDR BTC-USD QQQ", period="6mo", interval="1d", group_by='ticker', threads=True, progress=False)
        if data.empty: return default_model, default_factors, "No Data"

        btdr = data['BTDR'].dropna(); btc = data['BTC-USD'].dropna(); qqq = data['QQQ'].dropna()
        idx = btdr.index.intersection(btc.index).intersection(qqq.index)
        btdr, btc, qqq = btdr.loc[idx], btc.loc[idx], qqq.loc[idx]
        
        # 基础因子
        ret_btdr = btdr['Close'].pct_change()
        ret_btc = btc['Close'].pct_change()
        
        # Beta (Rolling)
        cov_btc = ret_btdr.rolling(60).cov(ret_btc).iloc[-1]
        var_btc = ret_btc.rolling(60).var().iloc[-1]
        beta_btc = cov_btc / var_btc if var_btc > 1e-6 else 1.5
        
        # Volatility & Regime
        high, low, close = btdr['High'], btdr['Low'], btdr['Close']
        # Parkinson Volatility (利用 High-Low range 计算，比收盘价标准差更精准)
        vol_parkinson = np.sqrt(1 / (4 * np.log(2)) * ((np.log(high / low)) ** 2)).rolling(20).mean().iloc[-1]
        
        # ADX
        tr = np.maximum(high - low, np.abs(high - close.shift(1)))
        atr = tr.rolling(14).mean()
        up, down = high.diff(), -low.diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        adx = 100 * (np.abs(plus_dm - minus_dm) / (plus_dm + minus_dm + 1e-6)).rolling(14).mean().iloc[-1] # Simplified DX
        regime = "Trend" if adx > 25 else "Chop"
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain/loss)).iloc[-1]
        
        # --- 算法 1: 非对称动态回归 (Asymmetric Regression) ---
        df_reg = btdr.tail(60).copy()
        df_reg['PrevClose'] = df_reg['Close'].shift(1)
        df_reg = df_reg.dropna()
        
        # 特征工程
        df_reg['Gap'] = (df_reg['Open'] - df_reg['PrevClose']) / df_reg['PrevClose']
        df_reg['High_Ret'] = (df_reg['High'] - df_reg['PrevClose']) / df_reg['PrevClose']
        df_reg['Low_Ret'] = (df_reg['Low'] - df_reg['PrevClose']) / df_reg['PrevClose']
        
        # 分离 Gap Up 和 Gap Down 的样本
        gap_up = df_reg[df_reg['Gap'] > 0]
        gap_down = df_reg[df_reg['Gap'] <= 0]
        
        # 计算不同情境下的条件期望 (Conditional Expectation)
        # 如果样本太少，回退到整体均值
        
        # 高开时的模型参数
        if len(gap_up) > 5:
            # 这里的逻辑：高开时，High通常能再冲多少？Low通常会回补多少？
            # 使用简单的均值而不是线性回归，在小样本下更稳健
            h_ratio_up = (gap_up['High_Ret'] - gap_up['Gap']).mean() # Gap之后还能涨多少
            l_ratio_up = (gap_up['Low_Ret'] - gap_up['Gap']).mean()  # Gap之后会跌多少
        else:
            h_ratio_up, l_ratio_up = 0.02, -0.01

        # 低开时的模型参数
        if len(gap_down) > 5:
            h_ratio_down = (gap_down['High_Ret'] - gap_down['Gap']).mean()
            l_ratio_down = (gap_down['Low_Ret'] - gap_down['Gap']).mean()
        else:
            h_ratio_down, l_ratio_down = 0.01, -0.03

        model_params = {
            "up_scenario": {"h_add": h_ratio_up, "l_add": l_ratio_up},
            "down_scenario": {"h_add": h_ratio_down, "l_add": l_ratio_down},
            "beta_btc": 0.6, # 盘中联动系数
            "beta_sector": 0.3
        }
        
        # --- 算法 2: 跳跃因子估算 ---
        # 简单估算：超过 2 倍标准差的波动视为 Jump
        daily_ret = btdr['Close'].pct_change().dropna()
        std = daily_ret.std()
        jumps = daily_ret[np.abs(daily_ret) > 2.5 * std]
        jump_intensity = len(jumps) / len(daily_ret) # 跳跃发生的概率 lambda
        
        factors = {
            "beta_btc": beta_btc, "vwap": (close*btdr['Volume']).sum()/btdr['Volume'].sum(), 
            "adx": adx, "regime": regime, "rsi": rsi, 
            "vol_parkinson": vol_parkinson, 
            "jump_prob": max(0.05, jump_intensity) # 至少给 5% 的跳跃概率
        }
        
        return model_params, factors, "MJD-Engine"
        
    except Exception as e:
        print(f"Error: {e}")
        return default_model, default_factors, "Offline"

# --- 4. 实时状态判断 ---
def determine_market_state(now_ny):
    weekday = now_ny.weekday()
    curr_min = now_ny.hour * 60 + now_ny.minute
    if weekday == 5: return "Weekend", "dot-closed"
    if weekday == 6 and now_ny.hour < 20: return "Weekend", "dot-closed"
    if 240 <= curr_min < 570: return "Pre-Mkt", "dot-pre"
    if 570 <= curr_min < 960: return "Mkt Open", "dot-reg"
    if 960 <= curr_min < 1200: return "Post-Mkt", "dot-post"
    return "Overnight", "dot-night"

def get_realtime_data():
    tickers = "BTC-USD BTDR MARA RIOT CLSK QQQ ^VIX"
    try:
        daily = yf.download(tickers, period="5d", interval="1d", group_by='ticker', threads=True, progress=False)
        live = yf.download(tickers, period="2d", interval="1m", prepost=True, group_by='ticker', threads=True, progress=False)
        
        if daily.empty: return None, 50
        
        quotes = {}
        now_ny = datetime.now(pytz.timezone('America/New_York'))
        tag, css = determine_market_state(now_ny)
        
        for sym in tickers.split():
            d_df = daily[sym] if sym in daily else pd.DataFrame()
            l_df = live[sym] if sym in live else pd.DataFrame()
            
            # 价格优先取 live
            if not l_df.empty: price = l_df['Close'].iloc[-1]
            elif not d_df.empty: price = d_df['Close'].iloc[-1]
            else: price = 0
            
            # 昨收/开盘
            prev = 1; open_p = 0; is_open = False
            if not d_df.empty:
                last_dt = d_df.index[-1].date()
                if last_dt == now_ny.date():
                    is_open = True
                    open_p = d_df['Open'].iloc[-1]
                    prev = d_df['Close'].iloc[-2] if len(d_df) > 1 else open_p
                else:
                    prev = d_df['Close'].iloc[-1]
                    open_p = prev # 还没开盘
            
            quotes[sym] = {
                "price": price, "pct": (price-prev)/prev*100, 
                "prev": prev, "open": open_p, 
                "tag": tag, "css": css, "is_open_today": is_open
            }
            
        return quotes, 50 # 假定 FNG 50，省去 API 请求加速
    except: return None, 50

# --- 5. 主界面 Fragment ---
@st.fragment(run_every=5)
def show_dashboard():
    quotes, fng = get_realtime_data()
    model, factors, eng_status = run_quant_analytics()
    
    if not quotes: st.warning("Connecting..."); return

    btdr = quotes['BTDR']
    btc = quotes['BTC-USD']
    vix = quotes['^VIX']
    
    # Header
    now_str = datetime.now(pytz.timezone('America/New_York')).strftime('%H:%M:%S')
    st.markdown(f"<div class='time-bar'>美东 {now_str} | 引擎: {eng_status} (v9.6 Quant)</div>", unsafe_allow_html=True)

    # Cards
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(card_html("BTDR", f"${btdr['price']:.2f}", f"{btdr['pct']:+.2f}%", btdr['pct'], f"<span class='status-dot {btdr['css']}'></span>"), unsafe_allow_html=True)
    with c2: st.markdown(card_html("BTC", f"${btc['price']:.0f}", f"{btc['pct']:+.2f}%", btc['pct']), unsafe_allow_html=True)
    with c3: st.markdown(card_html("VIX", f"{vix['price']:.1f}", None, 0), unsafe_allow_html=True)

    # --- 预测模型 (Asymmetric Logic) ---
    btdr_gap_pct = ((btdr['open'] - btdr['prev']) / btdr['prev'])
    
    # 动态选择参数：Gap Up 用 Up 参数，Gap Down 用 Down 参数
    if btdr_gap_pct >= 0:
        params = model.get('up_scenario', {'h_add': 0.05, 'l_add': -0.02})
        base_scenario = "Gap Up (多头惯性)"
    else:
        params = model.get('down_scenario', {'h_add': 0.02, 'l_add': -0.05})
        base_scenario = "Gap Down (回补/延续)"
        
    # 计算预测
    # High = Open * (1 + 历史平均冲高幅度 + BTC加成)
    pred_h_price = btdr['open'] * (1 + params['h_add'] + (btc['pct']/100 * 0.3))
    pred_l_price = btdr['open'] * (1 + params['l_add'] + (btc['pct']/100 * 0.3))
    
    st.markdown(f"### 🎯 动态阻力/支撑 ({base_scenario})")
    col_h, col_l = st.columns(2)
    
    # 颜色逻辑：突破阻力变红(卖点)，跌破支撑变绿(买点) - 反转策略
    h_bg = "#e6fcf5" if btdr['price'] < pred_h_price else "#ffc9c9" 
    l_bg = "#fff5f5" if btdr['price'] > pred_l_price else "#b2f2bb"
    
    with col_h: st.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background-color: {h_bg}; border: 1px solid #ddd;"><div style="font-size: 0.8rem;">阻力 (Resistance)</div><div style="font-size: 1.5rem; font-weight: bold;">${pred_h_price:.2f}</div></div></div>""", unsafe_allow_html=True)
    with col_l: st.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background-color: {l_bg}; border: 1px solid #ddd;"><div style="font-size: 0.8rem;">支撑 (Support)</div><div style="font-size: 1.5rem; font-weight: bold;">${pred_l_price:.2f}</div></div></div>""", unsafe_allow_html=True)

    # --- 因子 ---
    st.markdown("---")
    f1, f2, f3, f4 = st.columns(4)
    with f1: st.markdown(factor_html("Vol (Parkinson)", f"{factors['vol_parkinson']*100:.1f}%", "Risk", 0), unsafe_allow_html=True)
    with f2: st.markdown(factor_html("RSI", f"{factors['rsi']:.0f}", "Neu", 0), unsafe_allow_html=True)
    with f3: st.markdown(factor_html("ADX", f"{factors['adx']:.0f}", factors['regime'], 1 if factors['adx']>25 else -1), unsafe_allow_html=True)
    with f4: st.markdown(factor_html("Jump Prob", f"{factors['jump_prob']*100:.0f}%", "Tail", -1), unsafe_allow_html=True)

    # --- 蒙特卡洛：默顿跳跃扩散 (Merton Jump Diffusion) ---
    st.markdown("### ☁️ 概率推演 (Merton Jump Diffusion)")
    
    # 参数准备
    S0 = btdr['price']
    T = 5 # days
    dt = 1
    mu = (btc['pct']/100 * factors['beta_btc']) * 0.5 # 漂移项受 BTC 影响
    sigma = factors['vol_parkinson']
    lam = factors['jump_prob'] # 跳跃强度 (lambda)
    
    # 如果 VIX 很高，增加跳跃幅度的均值和方差
    jump_mu = -0.02 if vix['price'] > 25 else 0.0 # 恐慌时跳跃倾向于向下
    jump_sigma = 0.05 # 跳跃本身的波动
    
    simulations = 1000
    
    # 1. 几何布朗部分 (GBM)
    # Z1 ~ N(0, 1)
    Z1 = np.random.normal(0, 1, (simulations, T))
    drift_part = (mu - 0.5 * sigma**2) * dt
    diffusion_part = sigma * np.sqrt(dt) * Z1
    
    # 2. 泊松跳跃部分 (Poisson Jump)
    # N ~ Poisson(lam * dt) -> 每天发生几次跳跃
    N = np.random.poisson(lam * dt, (simulations, T))
    # Jump Size ~ N(jump_mu, jump_sigma)
    J_mean = jump_mu
    J_std = jump_sigma
    # 这里简化处理：假设每次跳跃的大小叠加 (Compound Jump)
    # 总跳跃幅度 = N * (随机跳跃大小) -> 这种写法是近似，为了向量化速度
    # 更严谨写法是 sum(Normal) for k in N，但在 python 中很难完全向量化而不慢
    # 这里用近似：JumpFactor = N * Normal(mu, sigma)
    Jump_part = N * np.random.normal(J_mean, J_std, (simulations, T))
    
    # 3. 组合路径
    daily_log_returns = drift_part + diffusion_part + Jump_part
    price_paths = np.zeros((simulations, T + 1))
    price_paths[:, 0] = S0
    price_paths[:, 1:] = S0 * np.exp(np.cumsum(daily_log_returns, axis=1))
    
    # 4. 绘图数据
    p90 = np.percentile(price_paths, 90, axis=0)
    p50 = np.percentile(price_paths, 50, axis=0)
    p10 = np.percentile(price_paths, 10, axis=0)
    
    chart_df = pd.DataFrame({"Day": range(T+1), "P90": p90, "P50": p50, "P10": p10})
    
    base = alt.Chart(chart_df).encode(x='Day:O')
    area = base.mark_area(opacity=0.15, color='#7048e8').encode(y=alt.Y('P10', scale=alt.Scale(zero=False)), y2='P90')
    line50 = base.mark_line(color='#7048e8').encode(y='P50')
    st.altair_chart((area + line50).interactive(), use_container_width=True)
    
    st.caption(f"Engine: MJD (Jump Diff) | $\lambda$: {lam:.2f} | $\sigma$: {sigma:.2f} | Jump Mean: {jump_mu:.2f}")

st.markdown("### ⚡ BTDR Pilot v9.6 Quant")
show_dashboard()
