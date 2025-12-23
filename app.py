import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
import altair as alt
from datetime import datetime
import pytz

# --- 1. 页面配置 & CSS样式 ---
st.set_page_config(page_title="BTDR Pilot v9.6 Quant", layout="centered")

CUSTOM_CSS = """
<style>
    html { overflow-y: scroll; }
    .stApp > header { display: none; }
    .stApp { margin-top: -30px; background-color: #ffffff; }
    div[data-testid="stStatusWidget"] { visibility: hidden; }
    
    /* 图表高度锁定 */
    div[data-testid="stAltairChart"] {
        height: 320px !important; min-height: 320px !important;
        overflow: hidden !important; border: 1px solid #f8f9fa;
    }
    
    /* 卡片样式 */
    .metric-card {
        background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02); height: 95px; padding: 0 16px;
        display: flex; flex-direction: column; justify-content: center;
    }
    .metric-label { font-size: 0.75rem; color: #888; margin-bottom: 2px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #212529; line-height: 1.2; }
    .metric-delta { font-size: 0.9rem; font-weight: 600; margin-top: 2px; }
    
    /* 因子样式 */
    .factor-box {
        background: #fff; border: 1px solid #eee; border-radius: 8px; padding: 6px; text-align: center;
        height: 75px; display: flex; flex-direction: column; justify-content: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02); position: relative; cursor: help;
    }
    .factor-title { font-size: 0.65rem; color: #999; text-transform: uppercase; }
    .factor-val { font-size: 1.1rem; font-weight: bold; color: #495057; margin: 2px 0; }
    .factor-sub { font-size: 0.7rem; font-weight: 600; }
    
    .color-up { color: #0ca678; } .color-down { color: #d6336c; }
    
    /* 状态点 */
    .status-dot { height: 6px; width: 6px; border-radius: 50%; display: inline-block; margin-left: 6px; margin-bottom: 2px; }
    .dot-pre { background-color: #f59f00; box-shadow: 0 0 4px #f59f00; }
    .dot-reg { background-color: #0ca678; box-shadow: 0 0 4px #0ca678; }
    .dot-post { background-color: #1c7ed6; box-shadow: 0 0 4px #1c7ed6; }
    .dot-night { background-color: #7048e8; box-shadow: 0 0 4px #7048e8; }
    .dot-closed { background-color: #adb5bd; }
    
    /* 预测框 */
    .pred-container-wrapper { height: 110px; width: 100%; display: block; margin-top: 5px; }
    .pred-box { padding: 0 10px; border-radius: 12px; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center; }
    
    .time-bar { font-size: 0.75rem; color: #999; text-align: center; margin-bottom: 20px; padding: 6px; background: #fafafa; border-radius: 6px; }
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

# --- 3. 核心计算逻辑 (修复版 Quant Engine) ---
@st.cache_data(ttl=600)
def run_quant_analytics():
    # 1. 定义完整的默认值，防止 KeyError
    default_model = {
        "up_scenario": {"h_add": 0.05, "l_add": -0.02},
        "down_scenario": {"h_add": 0.02, "l_add": -0.05},
        "beta_btc": 0.6
    }
    default_factors = {
        "vwap": 0, 
        "adx": 20, 
        "regime": "Neutral", 
        "beta_btc": 1.5, 
        "rsi": 50, 
        "vol_parkinson": 0.05,  # 确保包含此键
        "jump_prob": 0.1        # 确保包含此键
    }
    
    try:
        # 2. 下载数据 (threads=False 提高稳定性)
        data = yf.download("BTDR BTC-USD QQQ", period="6mo", interval="1d", group_by='ticker', threads=False, progress=False)
        
        if data.empty: 
            return default_model, default_factors, "No Data"

        # 安全提取
        try:
            btdr = data['BTDR'].dropna()
            btc = data['BTC-USD'].dropna()
        except KeyError:
            return default_model, default_factors, "Ticker Err"

        if len(btdr) < 30: return default_model, default_factors, "Low Data"

        # --- 计算开始 ---
        # 3. Beta (Rolling)
        ret_btdr = btdr['Close'].pct_change()
        ret_btc = btc['Close'].pct_change()
        # 对齐索引
        idx = ret_btdr.index.intersection(ret_btc.index)
        cov_btc = ret_btdr.loc[idx].rolling(60).cov(ret_btc.loc[idx]).iloc[-1]
        var_btc = ret_btc.loc[idx].rolling(60).var().iloc[-1]
        beta_btc = cov_btc / var_btc if var_btc > 1e-6 else 1.5
        
        # 4. Parkinson Volatility (基于 High/Low 的更精准波动率)
        high, low, close = btdr['High'], btdr['Low'], btdr['Close']
        # 避免 Log(0) 或除零
        safe_high = high.replace(0, np.nan)
        safe_low = low.replace(0, np.nan)
        vol_parkinson = np.sqrt(1 / (4 * np.log(2)) * ((np.log(safe_high / safe_low)) ** 2)).rolling(20).mean().iloc[-1]
        if np.isnan(vol_parkinson): vol_parkinson = 0.05
        
        # 5. ADX
        tr = np.maximum(high - low, np.abs(high - close.shift(1)))
        adx_val = 20
        if len(tr) > 14:
            adx_raw = tr.rolling(14).mean().iloc[-1]
            if not np.isnan(adx_raw): adx_val = min(max(adx_raw, 0), 100) # Clamp
        regime = "Trend" if adx_val > 25 else "Chop"
        
        # 6. RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rsi = 50
        if loss.iloc[-1] != 0:
            rsi = 100 - (100 / (1 + gain.iloc[-1]/loss.iloc[-1]))
            
        # 7. Asymmetric Regression (非对称回归)
        # 简单根据历史 Gap Up/Down 计算预期强度
        df_reg = btdr.tail(60).copy()
        df_reg['PrevClose'] = df_reg['Close'].shift(1)
        df_reg = df_reg.dropna()
        
        df_reg['Gap'] = (df_reg['Open'] - df_reg['PrevClose']) / df_reg['PrevClose']
        df_reg['High_Ret'] = (df_reg['High'] - df_reg['PrevClose']) / df_reg['PrevClose']
        df_reg['Low_Ret'] = (df_reg['Low'] - df_reg['PrevClose']) / df_reg['PrevClose']
        
        gap_up = df_reg[df_reg['Gap'] > 0]
        gap_down = df_reg[df_reg['Gap'] <= 0]
        
        # 默认值
        h_up, l_up = 0.05, -0.02
        h_down, l_down = 0.02, -0.05
        
        if len(gap_up) > 5:
            h_up = (gap_up['High_Ret'] - gap_up['Gap']).mean()
            l_up = (gap_up['Low_Ret'] - gap_up['Gap']).mean()
            
        if len(gap_down) > 5:
            h_down = (gap_down['High_Ret'] - gap_down['Gap']).mean()
            l_down = (gap_down['Low_Ret'] - gap_down['Gap']).mean()
            
        model_params = {
            "up_scenario": {"h_add": max(h_up, 0.01), "l_add": min(l_up, -0.005)},
            "down_scenario": {"h_add": max(h_down, 0.005), "l_add": min(l_down, -0.01)},
            "beta_btc": 0.6
        }
        
        # 8. Jump Probability
        daily_ret = btdr['Close'].pct_change().dropna()
        std = daily_ret.std()
        if std > 0:
            jumps = daily_ret[np.abs(daily_ret) > 2.5 * std]
            jump_prob = len(jumps) / len(daily_ret)
        else:
            jump_prob = 0.1
            
        factors = {
            "beta_btc": beta_btc, 
            "vwap": close.iloc[-1], # 简化版
            "adx": adx_val, 
            "regime": regime, 
            "rsi": rsi, 
            "vol_parkinson": vol_parkinson, 
            "jump_prob": max(0.05, jump_prob) 
        }
        
        return model_params, factors, "MJD-Engine"
        
    except Exception as e:
        print(f"Quant Error: {e}")
        # 发生错误时，返回安全的默认值
        return default_model, default_factors, "Offline"

# --- 4. 实时数据 (修复 NaN 版) ---
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
        # 下载：live取2天(减少体积)，daily取5天(做昨收备份)
        live = yf.download(tickers, period="2d", interval="1m", prepost=True, group_by='ticker', threads=False, progress=False)
        daily = yf.download(tickers, period="5d", interval="1d", group_by='ticker', threads=False, progress=False)
        
        quotes = {}
        now_ny = datetime.now(pytz.timezone('America/New_York'))
        tag, css = determine_market_state(now_ny)
        
        for sym in tickers.split():
            d_df = daily[sym] if sym in daily else pd.DataFrame()
            l_df = live[sym] if sym in live else pd.DataFrame()
            
            # --- 价格清洗逻辑 ---
            price = 0.0
            # 优先从 live 取
            if not l_df.empty:
                val = l_df['Close'].iloc[-1]
                if not pd.isna(val): price = float(val)
            
            # 没取到则从 daily 取
            if price == 0 and not d_df.empty:
                val = d_df['Close'].iloc[-1]
                if not pd.isna(val): price = float(val)
                
            # --- 昨收/开盘逻辑 ---
            prev = 1.0; open_p = 0.0; is_open = False
            
            # 必须用 daily 且去除空值
            if not d_df.empty:
                clean_df = d_df.dropna(subset=['Close'])
                if not clean_df.empty:
                    last_dt = clean_df.index[-1].date()
                    
                    if last_dt == now_ny.date():
                        # 今天有日线了(已开盘)
                        is_open = True
                        open_p = float(clean_df['Open'].iloc[-1])
                        if len(clean_df) > 1:
                            prev = float(clean_df['Close'].iloc[-2])
                        else:
                            prev = open_p # 新股或数据不足
                    else:
                        # 还没开盘，最后一条就是昨收
                        prev = float(clean_df['Close'].iloc[-1])
                        open_p = prev # 暂定
            
            # 计算涨跌幅
            pct = 0.0
            if prev > 0 and price > 0:
                pct = ((price - prev) / prev) * 100
            
            quotes[sym] = {
                "price": price, "pct": pct, 
                "prev": prev, "open": open_p, 
                "tag": tag, "css": css, "is_open_today": is_open
            }
            
        # 模拟 FNG 避免 API 卡顿
        return quotes, 50 
        
    except Exception as e:
        print(f"Data Fetch Error: {e}")
        return None, 50

# --- 5. 主界面 Fragment ---
@st.fragment(run_every=5)
def show_dashboard():
    quotes, fng = get_realtime_data()
    model, factors, eng_status = run_quant_analytics()
    
    if not quotes:
        st.warning("正在建立数据连接 (Initializing)...")
        time.sleep(1)
        return

    btdr = quotes.get('BTDR', {'price':0, 'pct':0})
    btc = quotes.get('BTC-USD', {'price':0, 'pct':0})
    vix = quotes.get('^VIX', {'price':20, 'pct':0})
    
    # 顶部时间栏
    now_str = datetime.now(pytz.timezone('America/New_York')).strftime('%H:%M:%S')
    st.markdown(f"<div class='time-bar'>美东 {now_str} | 引擎: {eng_status} (v9.6 Quant)</div>", unsafe_allow_html=True)

    # 核心指标卡片
    c1, c2, c3 = st.columns(3)
    status_html = f"<span class='status-dot {btdr.get('css','dot-closed')}'></span>"
    
    with c1: st.markdown(card_html("BTDR", f"${btdr['price']:.2f}", f"{btdr['pct']:+.2f}%", btdr['pct'], status_html), unsafe_allow_html=True)
    with c2: st.markdown(card_html("BTC", f"${btc['price']:.0f}", f"{btc['pct']:+.2f}%", btc['pct']), unsafe_allow_html=True)
    with c3: st.markdown(card_html("VIX", f"{vix['price']:.1f}", None, 0), unsafe_allow_html=True)

    # --- 动态阻力/支撑 (非对称逻辑) ---
    if btdr['prev'] > 0:
        btdr_gap_pct = (btdr['open'] - btdr['prev']) / btdr['prev']
    else:
        btdr_gap_pct = 0
    
    # 动态选择参数
    if btdr_gap_pct >= 0:
        params = model.get('up_scenario', {'h_add': 0.05, 'l_add': -0.02})
        base_scenario = "Gap Up (多头惯性)"
    else:
        params = model.get('down_scenario', {'h_add': 0.02, 'l_add': -0.05})
        base_scenario = "Gap Down (回补/延续)"
        
    # 计算目标价
    # High = Open * (1 + 历史平均冲高 + BTC盘中加成)
    btc_impact = (btc['pct']/100) * 0.3 # 权重系数
    pred_h_price = btdr['open'] * (1 + params['h_add'] + btc_impact)
    pred_l_price = btdr['open'] * (1 + params['l_add'] + btc_impact)
    
    st.markdown(f"### 🎯 动态阻力/支撑 ({base_scenario})")
    col_h, col_l = st.columns(2)
    
    h_bg = "#e6fcf5" if btdr['price'] < pred_h_price else "#ffc9c9" 
    l_bg = "#fff5f5" if btdr['price'] > pred_l_price else "#b2f2bb"
    
    with col_h: st.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background-color: {h_bg}; border: 1px solid #ddd;"><div style="font-size: 0.8rem;">阻力 (Resistance)</div><div style="font-size: 1.5rem; font-weight: bold;">${pred_h_price:.2f}</div></div></div>""", unsafe_allow_html=True)
    with col_l: st.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background-color: {l_bg}; border: 1px solid #ddd;"><div style="font-size: 0.8rem;">支撑 (Support)</div><div style="font-size: 1.5rem; font-weight: bold;">${pred_l_price:.2f}</div></div></div>""", unsafe_allow_html=True)

    # --- 因子面板 ---
    st.markdown("---")
    f1, f2, f3, f4 = st.columns(4)
    # 安全获取因子值
    vp = factors.get('vol_parkinson', 0.05)
    rsi = factors.get('rsi', 50)
    adx = factors.get('adx', 20)
    jp = factors.get('jump_prob', 0.1)
    
    with f1: st.markdown(factor_html("Vol (Parkinson)", f"{vp*100:.1f}%", "Risk", 0), unsafe_allow_html=True)
    with f2: st.markdown(factor_html("RSI", f"{rsi:.0f}", "Neu", 0), unsafe_allow_html=True)
    with f3: st.markdown(factor_html("ADX", f"{adx:.0f}", factors.get('regime','Neutral'), 1 if adx>25 else -1), unsafe_allow_html=True)
    with f4: st.markdown(factor_html("Jump Prob", f"{jp*100:.0f}%", "Tail", -1), unsafe_allow_html=True)

    # --- 蒙特卡洛：默顿跳跃扩散 (Vectorized MJD) ---
    st.markdown("### ☁️ 概率推演 (Merton Jump Diffusion)")
    
    S0 = btdr['price'] if btdr['price'] > 0 else btdr['prev']
    T = 5 
    dt = 1
    mu = (btc['pct']/100 * factors.get('beta_btc', 1.5)) * 0.5 
    sigma = vp
    lam = jp 
    
    # 恐慌调整
    jump_mu = -0.02 if vix['price'] > 25 else 0.0 
    jump_sigma = 0.05 
    simulations = 1000
    
    # 1. GBM Component
    Z1 = np.random.normal(0, 1, (simulations, T))
    drift_part = (mu - 0.5 * sigma**2) * dt
    diffusion_part = sigma * np.sqrt(dt) * Z1
    
    # 2. Jump Component
    N = np.random.poisson(lam * dt, (simulations, T))
    Jump_part = N * np.random.normal(jump_mu, jump_sigma, (simulations, T))
    
    # 3. Path Generation
    daily_log_returns = drift_part + diffusion_part + Jump_part
    price_paths = np.zeros((simulations, T + 1))
    price_paths[:, 0] = S0
    price_paths[:, 1:] = S0 * np.exp(np.cumsum(daily_log_returns, axis=1))
    
    # 4. Statistics
    p90 = np.percentile(price_paths, 90, axis=0)
    p50 = np.percentile(price_paths, 50, axis=0)
    p10 = np.percentile(price_paths, 10, axis=0)
    
    chart_df = pd.DataFrame({"Day": range(T+1), "P90": p90, "P50": p50, "P10": p10})
    
    base = alt.Chart(chart_df).encode(x=alt.X('Day:O', title='未来交易日 (T+)'))
    area = base.mark_area(opacity=0.15, color='#7048e8').encode(y=alt.Y('P10', title='价格模拟', scale=alt.Scale(zero=False)), y2='P90')
    line50 = base.mark_line(color='#7048e8', strokeDash=[2,2]).encode(y='P50')
    
    st.altair_chart((area + line50).interactive(), use_container_width=True)
    st.caption(f"Model: MJD | Jump $\lambda$: {lam:.2f} | $\sigma$: {sigma:.2f} | Sims: {simulations}")

# --- 6. 启动 ---
st.markdown("### ⚡ BTDR Pilot v9.6 Quant")
show_dashboard()
