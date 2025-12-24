import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
import plotly.graph_objects as go
from datetime import datetime, time as dt_time
import pytz
from scipy.stats import norm

# --- 1. 页面配置 & 样式 ---
st.set_page_config(page_title="BTDR Pilot v11.1 Ultimate", layout="wide")

CUSTOM_CSS = """
<style>
    .stApp > header { display: none; }
    div[data-testid="stStatusWidget"] { visibility: hidden; }
    h1, h2, h3, h4, div, p, span { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important; }

    /* 核心卡片 (修复版) */
    .metric-card {
        background-color: #f8f9fa; border: 1px solid #e9ecef;
        border-radius: 12px; padding: 15px; height: 100px;
        display: flex; flex-direction: column; justify-content: center;
        position: relative; transition: transform 0.1s;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .metric-card.has-tooltip { cursor: help; }
    .metric-label { font-size: 0.8rem; color: #6c757d; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { font-size: 1.8rem; font-weight: 800; color: #212529; line-height: 1.1; }
    .metric-delta { font-size: 0.9rem; font-weight: 600; margin-top: 4px; }

    .color-up { color: #0ca678; } .color-down { color: #d6336c; } .color-neutral { color: #adb5bd; }

    /* 信号盒子 */
    .signal-box { 
        border-radius: 8px; padding: 15px; text-align: center; color: white; 
        height: 100%; display: flex; flex-direction: column; justify-content: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); cursor: help; position: relative;
    }
    .sig-buy { background: linear-gradient(135deg, #0ca678 0%, #087f5b 100%); }
    .sig-sell { background: linear-gradient(135deg, #fa5252 0%, #c92a2a 100%); }
    .sig-wait { background: linear-gradient(135deg, #ced4da 0%, #adb5bd 100%); color: #495057; }
    .signal-title { font-size: 0.75rem; opacity: 0.9; letter-spacing: 1px; margin-bottom: 5px; }
    .signal-main { font-size: 1.6rem; font-weight: 900; }
    .signal-sub { font-size: 0.8rem; opacity: 0.9; margin-top: 5px; font-weight: normal; }

    /* 交易计划卡片 */
    .plan-card {
        background: #fff; border: 1px solid #eee; border-radius: 10px; padding: 15px;
        height: 100%; box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .plan-header { font-size: 0.9rem; font-weight: 700; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px dashed #eee; display: flex; justify-content: space-between; align-items: center; }
    .plan-row { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.9rem; }
    .plan-label { color: #888; }
    .plan-val { font-weight: 600; font-family: 'Roboto Mono', monospace; }
    
    /* Tooltip 悬浮框 (核心修复) */
    .tooltip-text {
        visibility: hidden; width: 200px; background-color: rgba(33, 37, 41, 0.95);
        color: #fff !important; text-align: center; border-radius: 6px; padding: 8px;
        position: absolute; z-index: 999; bottom: 110%; left: 50%; margin-left: -100px;
        opacity: 0; transition: opacity 0.3s; font-size: 0.75rem !important;
        font-weight: normal; line-height: 1.4; pointer-events: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .tooltip-text::after {
        content: ""; position: absolute; top: 100%; left: 50%; margin-left: -5px;
        border-width: 5px; border-style: solid; border-color: rgba(33, 37, 41, 0.95) transparent transparent transparent;
    }
    .metric-card:hover .tooltip-text, .signal-box:hover .tooltip-text, .factor-box:hover .tooltip-text { visibility: visible; opacity: 1; }

    /* 鲸鱼警报 */
    .whale-alert { background-color: #fff9db; border: 1px solid #fcc419; color: #e67700; padding: 8px 12px; border-radius: 6px; font-size: 0.85rem; font-weight: 600; margin-bottom: 15px; display: flex; align-items: center; }
    
    /* 矿股小卡片 */
    .miner-card { background-color: #fff; border: 1px solid #e9ecef; border-radius: 8px; padding: 8px; text-align: center; height: 80px; display: flex; flex-direction: column; justify-content: center; }
    .miner-sym { font-size: 0.7rem; color: #888; font-weight: 700; }
    .miner-price { font-size: 1rem; font-weight: 700; color: #212529; margin: 2px 0; }
    .miner-sub { font-size: 0.65rem; color: #868e96; }

    /* 顶部栏 */
    .top-bar { display: flex; justify-content: space-between; align-items: center; background: #f8f9fa; padding: 8px 15px; border-radius: 8px; margin-bottom: 20px; font-size: 0.8rem; color: #666; }
    .status-tag { padding: 2px 6px; border-radius: 4px; color: white; font-weight: 600; font-size: 0.7rem; margin-left: 5px;}
    .tag-open { background: #0ca678; } .tag-closed { background: #868e96; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- 2. 基础配置 ---
MINER_SHARES = {"MARA": 300, "RIOT": 330, "CLSK": 220, "CORZ": 190, "IREN": 180, "WULF": 410, "CIFR": 300, "HUT": 100}
MINER_POOL = list(MINER_SHARES.keys())

# --- 3. 辅助函数 (修复了 HTML 渲染问题) ---
def card_html(label, value_str, delta_str=None, delta_val=0, extra_tag="", tooltip_text=None):
    delta_html = ""
    if delta_str:
        color_class = "color-up" if delta_val >= 0 else "color-down"
        delta_html = f"<div class='metric-delta {color_class}'>{delta_str}</div>"
    
    tooltip_html = f"<div class='tooltip-text'>{tooltip_text}</div>" if tooltip_text else ""
    card_class = "metric-card has-tooltip" if tooltip_text else "metric-card"
    
    # 强制单行返回，防止 Streamlit 解析错误
    return f"""<div class="{card_class}">{tooltip_html}<div class="metric-label">{label} {extra_tag}</div><div class="metric-value">{value_str}</div>{delta_html}</div>"""

def miner_card_html(sym, price, pct, turnover):
    color_class = "color-up" if pct >= 0 else "color-down"
    return f"""<div class="miner-card"><div class="miner-sym">{sym}</div><div class="miner-price ${color_class}">${price:.2f}</div><div class="miner-sub"><span class="{color_class}">{pct:+.1f}%</span> | 换{turnover:.1f}%</div></div>"""

# --- 4. 核心计算 (Kalman + WLS) ---
def run_kalman_filter(y, x, delta=1e-4):
    n = len(y); beta = np.zeros(n); P = np.zeros(n); beta[0]=1.0; P[0]=1.0; R=0.002; Q=delta/(1-delta)
    for t in range(1, n):
        beta_pred = beta[t-1]; P_pred = P[t-1] + Q
        if x[t] == 0: x[t] = 1e-6
        residual = y[t] - beta_pred * x[t]; S = P_pred * x[t]**2 + R; K = P_pred * x[t] / S
        beta[t] = beta_pred + K * residual; P[t] = (1 - K * x[t]) * P_pred
    return beta[-1]

@st.cache_data(ttl=600)
def run_grandmaster_analytics():
    default_model = {"high": {"intercept": 0, "beta_gap": 0.5}, "low": {"intercept": 0, "beta_gap": 0.5}, "top_peers": MINER_POOL[:5]}
    default_factors = {"vwap": 0, "adx": 20, "regime": "Neutral", "beta_btc": 1.5, "beta_qqq": 1.2, "rsi": 50, "vol_base": 0.05, "atr_ratio": 0.05}

    try:
        tickers_str = "BTDR BTC-USD QQQ " + " ".join(MINER_POOL)
        data = yf.download(tickers_str, period="1mo", interval="1d", group_by='ticker', threads=True, progress=False)
        if data.empty: return default_model, default_factors, "No Data"

        btdr = data['BTDR'].dropna(); btc = data['BTC-USD'].dropna(); qqq = data['QQQ'].dropna()
        idx = btdr.index.intersection(btc.index).intersection(qqq.index)
        btdr, btc, qqq = btdr.loc[idx], btc.loc[idx], qqq.loc[idx]
        
        # Correlations
        correlations = {}
        for m in MINER_POOL:
            if m in data:
                miner_df = data[m]['Close'].pct_change().tail(30)
                btdr_df = btdr['Close'].pct_change().tail(30)
                if len(miner_df) > 10: correlations[m] = miner_df.corr(btdr_df)
                else: correlations[m] = 0
        top_peers = sorted(correlations, key=correlations.get, reverse=True)[:5]

        # Factors
        ret_btdr = btdr['Close'].pct_change().fillna(0).values
        ret_btc = btc['Close'].pct_change().fillna(0).values
        ret_qqq = qqq['Close'].pct_change().fillna(0).values
        
        beta_btc = run_kalman_filter(ret_btdr, ret_btc, delta=1e-4)
        beta_qqq = run_kalman_filter(ret_btdr, ret_qqq, delta=1e-4)
        avg_vol_5d = btdr['Volume'].tail(5).mean()

        close = btdr['Close']
        delta_p = close.diff(); gain = delta_p.where(delta_p>0,0).rolling(14).mean(); loss = -delta_p.where(delta_p<0,0).rolling(14).mean()
        rs = gain/loss; rsi = 100 - (100/(1+rs)).iloc[-1]
        
        high, low = btdr['High'], btdr['Low']
        tr = np.maximum(high - low, np.abs(high - close.shift(1)))
        atr = tr.rolling(14).mean(); atr_ratio = (atr/close).iloc[-1]
        
        vol_base = ret_btdr.std()
        if len(ret_btdr) > 20: vol_base = pd.Series(ret_btdr).ewm(span=20).std().iloc[-1]
        
        vwap_30d = (btdr['Close']*btdr['Volume']).tail(30).sum() / btdr['Volume'].tail(30).sum()

        factors = {"beta_btc": beta_btc, "beta_qqq": beta_qqq, "vwap": vwap_30d, "adx": 25, "regime": "Trend", "rsi": rsi, "vol_base": vol_base, "atr_ratio": atr_ratio, "avg_vol": avg_vol_5d}

        # WLS
        df_reg = pd.DataFrame()
        df_reg['PrevClose'] = btdr['Close'].shift(1); df_reg['Open'] = btdr['Open']
        df_reg['High'] = btdr['High']; df_reg['Low'] = btdr['Low']
        df_reg['Gap'] = (df_reg['Open'] - df_reg['PrevClose']) / df_reg['PrevClose']
        df_reg['BTC_Ret'] = btc['Close'].pct_change()
        df_reg['Vol_State'] = ((btdr['High'] - btdr['Low']) / btdr['Open']).shift(1)
        df_reg['Target_High'] = (df_reg['High'] - df_reg['PrevClose']) / df_reg['PrevClose']
        df_reg['Target_Low'] = (df_reg['Low'] - df_reg['PrevClose']) / df_reg['PrevClose']
        df_reg = df_reg.dropna().tail(90)
        
        weights = np.exp(np.linspace(-0.05 * len(df_reg), 0, len(df_reg))); W = np.diag(weights)
        X = np.column_stack([np.ones(len(df_reg)), df_reg['Gap'].values, df_reg['BTC_Ret'].values, df_reg['Vol_State'].values])
        Y_h = df_reg['Target_High'].values; Y_l = df_reg['Target_Low'].values
        XtWX = X.T @ W @ X
        theta_h = np.linalg.lstsq(XtWX, X.T @ W @ Y_h, rcond=None)[0]; theta_l = np.linalg.lstsq(XtWX, X.T @ W @ Y_l, rcond=None)[0]

        final_model = {
            "high": {"intercept": theta_h[0], "beta_gap": theta_h[1], "beta_btc": theta_h[2], "beta_vol": theta_h[3]},
            "low": {"intercept": theta_l[0], "beta_gap": theta_l[1], "beta_btc": theta_l[2], "beta_vol": theta_l[3]},
            "ensemble_hist_h": df_reg['Target_High'].tail(10).mean(), "ensemble_hist_l": df_reg['Target_Low'].tail(10).mean(),
            "ensemble_mom_h": df_reg['Target_High'].tail(3).max(), "ensemble_mom_l": df_reg['Target_Low'].tail(3).min(),
            "top_peers": top_peers
        }
        return final_model, factors, "v11.1 Ultimate"
    except Exception as e:
        return default_model, default_factors, "Offline"

# --- 5. 实时数据 ---
def determine_market_state(now_ny):
    weekday = now_ny.weekday(); curr_min = now_ny.hour * 60 + now_ny.minute
    if weekday == 5: return "Weekend", "tag-closed"
    if weekday == 6 and now_ny.hour < 20: return "Weekend", "tag-closed"
    if 240 <= curr_min < 570: return "Pre-Mkt", "tag-open"
    if 570 <= curr_min < 960: return "Mkt Open", "tag-open"
    if 960 <= curr_min < 1200: return "Post-Mkt", "tag-open"
    return "Closed", "tag-closed"

def get_realtime_data():
    tickers_list = "BTC-USD BTDR QQQ ^VIX " + " ".join(MINER_POOL)
    try:
        live = yf.download(tickers_list, period="5d", interval="15m", prepost=True, group_by='ticker', threads=True, progress=False)
        daily = yf.download(tickers_list, period="5d", interval="1d", group_by='ticker', threads=True, progress=False)
        
        quotes = {}
        tz_ny = pytz.timezone('America/New_York'); now_ny = datetime.now(tz_ny); state_tag, state_css = determine_market_state(now_ny)
        live_volatility = 0.01 
        
        for sym in tickers_list.split():
            try:
                df_min = live[sym].dropna(subset=['Close']) if sym in live else pd.DataFrame()
                df_day = daily[sym].dropna(subset=['Close']) if sym in daily else pd.DataFrame()
                
                curr_price = 0.0; curr_vol = 0
                if not df_min.empty:
                    curr_price = df_min['Close'].iloc[-1]
                    curr_vol = df_min['Volume'].iloc[-1]
                    if sym == 'BTDR':
                        recent_std = df_min['Close'].tail(20).std()
                        if recent_std > 0: live_volatility = recent_std
                elif not df_day.empty:
                    curr_price = df_day['Close'].iloc[-1]
                
                prev_close = 1.0
                if not df_day.empty:
                    prev_close = df_day['Close'].iloc[-1]
                    if df_day.index[-1].date() == now_ny.date():
                        if len(df_day) >= 2: prev_close = df_day['Close'].iloc[-2]
                
                pct = ((curr_price - prev_close)/prev_close)*100 if prev_close else 0
                quotes[sym] = {"price": curr_price, "pct": pct, "prev": prev_close, "volume": curr_vol}
            except: quotes[sym] = {"price": 0, "pct": 0, "prev": 1, "volume": 0}
            
        try: fng = int(requests.get("https://api.alternative.me/fng/", timeout=0.8).json()['data'][0]['value'])
        except: fng = 50
        
        return quotes, fng, live_volatility, live, state_tag, state_css
    except: return None, 50, 0.01, None, "ERR", "tag-closed"

# --- 6. 核心看板 ---
@st.fragment(run_every=15)
def show_live_dashboard():
    data_pack = get_realtime_data()
    if not data_pack or not data_pack[0]: 
        st.warning("📡 连接中 (Initializing)..."); time.sleep(1); st.rerun(); return
    
    quotes, fng_val, live_vol_btdr, df_chart_data, state_tag, state_css = data_pack
    ai_model, factors, ai_status = run_grandmaster_analytics()
    
    btc = quotes.get('BTC-USD'); btdr = quotes.get('BTDR'); vix = quotes.get('^VIX')
    tz_ny = pytz.timezone('America/New_York'); now_ny = datetime.now(tz_ny)
    
    # RVOL Logic
    avg_vol_min = factors['avg_vol'] / 390
    curr_vol_min = btdr['volume']
    rvol = curr_vol_min / (avg_vol_min * 15) if avg_vol_min > 0 else 1.0
    
    # Prediction
    dist_vwap = ((btdr['price'] - factors['vwap']) / factors['vwap']) * 100 if factors['vwap'] > 0 else 0
    curr_p = btdr['price']; prev = btdr['prev']
    current_gap_pct = ((curr_p - prev) / prev)
    
    mh, ml = ai_model['high'], ai_model['low']
    # Kalman Base
    base_h = mh['intercept'] + (mh['beta_gap']*current_gap_pct) + (mh['beta_btc']*(btc['pct']/100))
    base_l = ml['intercept'] + (ml['beta_gap']*current_gap_pct) + (ml['beta_btc']*(btc['pct']/100))
    # AI Volatility
    vol_pct = live_vol_btdr / curr_p if curr_p > 0 else 0.01
    ai_h = ((curr_p * (1 + 2.5 * vol_pct)) - prev) / prev
    ai_l = ((curr_p * (1 - 2.5 * vol_pct)) - prev) / prev
    
    # Weighted Ensemble (30% Kalman, 20% Hist/Mom, 50% AI)
    final_h_pct = (0.3 * base_h) + (0.1 * ai_model['ensemble_hist_h']) + (0.1 * ai_model['ensemble_mom_h']) + (0.5 * ai_h)
    final_l_pct = (0.3 * base_l) + (0.1 * ai_model['ensemble_hist_l']) + (0.1 * ai_model['ensemble_mom_l']) + (0.5 * ai_l)
    
    final_h_pct += (fng_val - 50) * 0.0005
    final_l_pct += (fng_val - 50) * 0.0005
    
    p_high = prev * (1 + final_h_pct); p_low = prev * (1 + final_l_pct)
    
    # --- UI ---
    st.markdown(f"""
    <div class="top-bar">
        <div><span style="font-weight:bold; font-size:1rem;">BTDR PILOT v11.1</span> <span class="status-tag {state_css}">{state_tag}</span></div>
        <div>{now_ny.strftime('%H:%M:%S')} NY | 状态: {factors['regime']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if rvol > 3.0: st.markdown(f"""<div class="whale-alert">🐋 巨量警报 (WHALE ALERT): 当前成交量是均值的 {rvol:.1f}倍！关注变盘！</div>""", unsafe_allow_html=True)

    # Chart & Signal
    main_c1, main_c2 = st.columns([2, 1])
    
    with main_c1:
        if df_chart_data is not None and 'BTDR' in df_chart_data:
            df_plot = df_chart_data['BTDR'].dropna().tail(50)
            fig = go.Figure(data=[go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name='BTDR')])
            fig.add_trace(go.Scatter(x=[df_plot.index[0], df_plot.index[-1]], y=[p_high, p_high], mode='lines', line=dict(color='red', width=1, dash='dash'), name='Resist'))
            fig.add_trace(go.Scatter(x=[df_plot.index[0], df_plot.index[-1]], y=[p_low, p_low], mode='lines', line=dict(color='green', width=1, dash='dash'), name='Support'))
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("图表加载中... (需安装 plotly)")

    with main_c2:
        dist_low = (curr_p - p_low)/curr_p
        if dist_low < 0.01: sig_title="STRONG BUY"; sig_css="sig-buy"; sig_sub="触及支撑 + 超卖"
        elif curr_p > p_high * 0.99: sig_title="STRONG SELL"; sig_css="sig-sell"; sig_sub="触及阻力 + 超买"
        else: sig_title="WAIT / WATCH"; sig_css="sig-wait"; sig_sub="区间震荡"
        
        buy_entry = p_low + (live_vol_btdr * 0.5)
        buy_stop = buy_entry - (live_vol_btdr * 2.0)
        buy_target = p_high - (live_vol_btdr * 0.5)
        
        st.markdown(f"""
        <div class="signal-box {sig_css}" style="margin-bottom:10px;">
            <div class="tooltip-text">信号基于AI预测区间与实时价格的偏离度生成。</div>
            <div class="signal-title">AI SIGNAL</div>
            <div class="signal-main">{sig_title}</div>
            <div class="signal-sub">{sig_sub}</div>
        </div>
        <div class="plan-card">
            <div class="plan-header" style="color:#0ca678">🟢 做多计划</div>
            <div class="plan-row"><span class="plan-label">挂单</span><span class="plan-val">${buy_entry:.2f}</span></div>
            <div class="plan-row"><span class="plan-label">目标</span><span class="plan-val">${buy_target:.2f}</span></div>
            <div class="plan-row"><span class="plan-label">止损</span><span class="plan-val" style="color:#fa5252">${buy_stop:.2f}</span></div>
        </div>
        """, unsafe_allow_html=True)

    # Metrics
    st.markdown("### 📊 核心数据")
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.markdown(card_html("BTDR 现价", f"${btdr['price']:.2f}", f"{btdr['pct']:+.2f}%", btdr['pct']), unsafe_allow_html=True)
    with m2: st.markdown(card_html("BTC (USD)", f"${btc['price']:,.0f}", f"{btc['pct']:+.2f}%", btc['pct']), unsafe_allow_html=True)
    with m3: st.markdown(card_html("恐慌指数", f"{fng_val}", None, 0, tooltip_text="0-25: 极度恐慌 (买点)\n75-100: 极度贪婪 (卖点)"), unsafe_allow_html=True)
    with m4: st.markdown(card_html("RSI (14d)", f"{factors['rsi']:.0f}", "Strength", 0), unsafe_allow_html=True)

    # Peers
    st.markdown("---")
    st.caption("⚒️ 矿股板块 Top 5 (Correlation)")
    cols = st.columns(5)
    for i, p in enumerate(ai_model['top_peers']):
        d = quotes.get(p, {'pct': 0, 'price': 0, 'volume': 0})
        shares = MINER_SHARES.get(p, 200)
        tr = (d['volume'] / (shares*1000000))*100
        cols[i].markdown(miner_card_html(p, d['price'], d['pct'], tr), unsafe_allow_html=True)

    # Risk Calc
    with st.expander("🧮 账户风控计算器"):
        rc1, rc2 = st.columns(2)
        with rc1: shares = st.number_input("计划股数", value=1000, step=100)
        with rc2: 
            risk = (buy_entry - buy_stop) * shares
            st.metric("潜在亏损风险", f"${risk:.0f}", f"-${buy_entry - buy_stop:.2f}/股", delta_color="inverse")

st.markdown("### ⚡ BTDR 领航员 v11.1 Ultimate")
show_live_dashboard()
