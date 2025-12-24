import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
import plotly.graph_objects as go
import pytz
from datetime import datetime
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor

# --- 1. 页面配置 ---
st.set_page_config(page_title="BTDR Pilot v11.0 Turbo", layout="wide")

# --- CSS 样式 ---
CUSTOM_CSS = """
<style>
    html { overflow-y: scroll; }
    .stApp > header { display: none; }
    .stApp { margin-top: -30px; background-color: #ffffff; }
    div[data-testid="stStatusWidget"] { visibility: hidden; }
    
    h1, h2, h3, div, p, span { 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important; 
    }
    
    /* Metric Card */
    .metric-card {
        background-color: #f8f9fa; border: 1px solid #e9ecef;
        border-radius: 12px; height: 95px; padding: 0 16px;
        display: flex; flex-direction: column; justify-content: center;
        position: relative; transition: all 0.2s;
    }
    .metric-card:hover { border-color: #ced4da; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .metric-label { font-size: 0.75rem; color: #888; margin-bottom: 2px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #212529; line-height: 1.2; }
    .metric-delta { font-size: 0.9rem; font-weight: 600; margin-top: 2px; }
    
    .color-up { color: #0ca678; } 
    .color-down { color: #d6336c; } 
    
    /* Ticket Card */
    .ticket-card {
        border-radius: 10px; padding: 15px; margin-bottom: 10px;
        text-align: left; position: relative; border-left: 5px solid #ccc;
        background: #fff; box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .ticket-buy { border-left-color: #0ca678; background: #f0fff4; }
    .ticket-sell { border-left-color: #e03131; background: #fff5f5; }
    
    /* Miner Box */
    .miner-box {
        text-align:center; border:1px solid #eee; border-radius:8px; 
        padding:8px 4px; background: #fff;
    }
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

# --- 3. 核心算法 (Grandmaster Analytics) ---
def run_kalman_filter(y, x, delta=1e-4):
    n = len(y); beta = np.zeros(n); P = np.zeros(n); beta[0]=1.0; P[0]=1.0; R=0.002;
    Q = delta/(1-delta)
    for t in range(1, n):
        beta_pred = beta[t-1]; P_pred = P[t-1] + Q
        if x[t] == 0: x[t] = 1e-6
        residual = y[t] - beta_pred * x[t]
        S = P_pred * x[t]**2 + R; K = P_pred * x[t] / S
        beta[t] = beta_pred + K * residual
        P[t] = (1 - K * x[t]) * P_pred
    return beta[-1]

@st.cache_data(ttl=300) # 缓存5分钟
def run_grandmaster_analytics(miner_pool):
    # 默认模型参数
    default_model = {
        "high": {"intercept": 0, "beta_gap": 0.5, "beta_btc": 0.5, "beta_vol": 0},
        "low": {"intercept": 0, "beta_gap": 0.5, "beta_btc": 0.5, "beta_vol": 0},
        "ensemble_hist_h": 0.05, "ensemble_hist_l": -0.05,
        "ensemble_mom_h": 0.08, "ensemble_mom_l": -0.08,
        "top_peers": miner_pool[:5]
    }
    default_factors = {"vwap": 0, "adx": 20, "regime": "Neutral", "beta_btc": 1.5, "beta_qqq": 1.2, "rsi": 50, "vol_base": 0.05, "atr_ratio": 0.05}

    try:
        tickers_str = "BTDR BTC-USD QQQ " + " ".join(miner_pool)
        data = yf.download(tickers_str, period="3mo", interval="1d", group_by='ticker', threads=True, progress=False)
        
        if data.empty or 'BTDR' not in data: return default_model, default_factors, "No Data"

        btdr = data['BTDR'].dropna(); btc = data['BTC-USD'].dropna(); qqq = data['QQQ'].dropna()
        idx = btdr.index.intersection(btc.index).intersection(qqq.index)
        
        if len(idx) < 30: return default_model, default_factors, "Low Data"
        
        btdr, btc, qqq = btdr.loc[idx], btc.loc[idx], qqq.loc[idx]

        # Peers Correlation
        correlations = {}
        for m in miner_pool:
            if m in data:
                miner_df = data[m]['Close'].pct_change().tail(30)
                btdr_df = btdr['Close'].pct_change().tail(30)
                common = miner_df.index.intersection(btdr_df.index)
                if len(common) > 10: correlations[m] = miner_df.loc[common].corr(btdr_df.loc[common])
                else: correlations[m] = 0
        top_peers = sorted(correlations, key=correlations.get, reverse=True)[:5]
        
        # Factors
        ret_btdr = btdr['Close'].pct_change().fillna(0).values
        ret_btc = btc['Close'].pct_change().fillna(0).values
        ret_qqq = qqq['Close'].pct_change().fillna(0).values
        
        beta_btc = run_kalman_filter(ret_btdr, ret_btc)
        beta_qqq = run_kalman_filter(ret_btdr, ret_qqq)
        
        pv = (btdr['Close'] * btdr['Volume'])
        vwap_30d = pv.tail(30).sum() / btdr['Volume'].tail(30).sum()
        
        # ADX / RSI
        high, low, close = btdr['High'], btdr['Low'], btdr['Close']
        tr = np.maximum(high - low, np.abs(high - close.shift(1)))
        atr = tr.rolling(14).mean()
        up, down = high.diff(), -low.diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        plus_di = 100 * (pd.Series(plus_dm, index=btdr.index).rolling(14).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm, index=btdr.index).rolling(14).mean() / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean().iloc[-1]
        
        delta_p = close.diff()
        gain = delta_p.where(delta_p > 0, 0).rolling(14).mean()
        loss = -delta_p.where(delta_p < 0, 0).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]

        vol_base = pd.Series(ret_btdr).ewm(span=20).std().iloc[-1]
        atr_ratio = (atr / close).iloc[-1]

        factors = {"beta_btc": beta_btc, "beta_qqq": beta_qqq, "vwap": vwap_30d, "adx": adx if not np.isnan(adx) else 20, 
                   "regime": "Trend" if adx > 25 else "Chop", "rsi": rsi, "vol_base": vol_base, "atr_ratio": atr_ratio}
        
        # Regression
        df_reg = pd.DataFrame()
        df_reg['PrevClose'] = btdr['Close'].shift(1); df_reg['Open'] = btdr['Open']
        df_reg['Gap'] = (df_reg['Open'] - df_reg['PrevClose']) / df_reg['PrevClose']
        df_reg['BTC_Ret'] = btc['Close'].pct_change()
        df_reg['Vol_State'] = ((btdr['High'] - btdr['Low']) / btdr['Open']).shift(1)
        df_reg['Target_High'] = (btdr['High'] - df_reg['PrevClose']) / df_reg['PrevClose']
        df_reg['Target_Low'] = (btdr['Low'] - df_reg['PrevClose']) / df_reg['PrevClose']
        df_reg = df_reg.dropna().tail(90)
        
        weights = np.exp(np.linspace(-0.05 * len(df_reg), 0, len(df_reg))); W = np.diag(weights)
        X = np.column_stack([np.ones(len(df_reg)), df_reg['Gap'].values, df_reg['BTC_Ret'].values, df_reg['Vol_State'].values])
        
        theta_h = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ df_reg['Target_High'].values, rcond=None)[0]
        theta_l = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ df_reg['Target_Low'].values, rcond=None)[0]

        final_model = {
            "high": {"intercept": theta_h[0], "beta_gap": theta_h[1], "beta_btc": theta_h[2], "beta_vol": theta_h[3]},
            "low": {"intercept": theta_l[0], "beta_gap": theta_l[1], "beta_btc": theta_l[2], "beta_vol": theta_l[3]},
            "ensemble_hist_h": df_reg['Target_High'].tail(10).mean(), "ensemble_hist_l": df_reg['Target_Low'].tail(10).mean(),
            "ensemble_mom_h": df_reg['Target_High'].tail(3).max(), "ensemble_mom_l": df_reg['Target_Low'].tail(3).min(),
            "top_peers": top_peers
        }
        return final_model, factors, "v11.0 Active"

    except Exception:
        return default_model, default_factors, "Offline"

# --- 4. 实时数据 (并行线程池) ---
def fetch_single_ticker(ticker):
    try:
        data = yf.download(ticker, period="5d", interval="1m", progress=False, threads=False)
        return ticker, data
    except:
        return ticker, None

def get_parallel_realtime_data(miner_pool):
    tickers_list = ["BTDR", "BTC-USD", "QQQ", "^VIX"] + miner_pool
    quotes = {}
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(fetch_single_ticker, tickers_list)
        
    raw_data = {t: d for t, d in results if d is not None and not d.empty}
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny)
    
    for sym in tickers_list:
        if sym not in raw_data:
            quotes[sym] = {"price": 0, "pct": 0, "prev": 1, "data": pd.DataFrame()}
            continue
            
        df = raw_data[sym]
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(sym, axis=1, level=1)
            except:
                pass
        
        if df.empty:
            quotes[sym] = {"price": 0, "pct": 0, "prev": 1, "data": pd.DataFrame()}
            continue

        current_price = df['Close'].iloc[-1]
        
        last_date = df.index[-1].date()
        if last_date < now_ny.date():
            prev = df['Close'].iloc[-1]
        else:
            mask = df.index.date < now_ny.date()
            if mask.any():
                prev = df[mask]['Close'].iloc[-1]
            else:
                prev = df['Open'].iloc[0]
        
        pct = ((current_price - prev) / prev * 100) if prev > 0 else 0
        quotes[sym] = {
            "price": current_price, "pct": pct, "prev": prev, 
            "volume": df['Volume'].sum() if 'Volume' in df else 0,
            "data": df
        }
        
    try:
        fng = int(requests.get("https://api.alternative.me/fng/", timeout=1).json()['data'][0]['value'])
    except:
        fng = 50
        
    return quotes, fng

# --- 5. 仪表盘 Fragment ---
@st.fragment(run_every=15)
def show_live_dashboard(shares, risk_tol, selected_miners):
    # 1. 获取数据
    quotes, fng_val = get_parallel_realtime_data(selected_miners)
    
    if quotes['BTDR']['price'] == 0:
        st.warning("⚠️ 数据源连接不稳定，3秒后重试...")
        time.sleep(3) 
        st.rerun()
        return

    # 2. 运行模型
    ai_model, factors, ai_status = run_grandmaster_analytics(selected_miners)
    
    btdr = quotes['BTDR']
    btc = quotes['BTC-USD']
    
    # 3. 计算预测
    current_gap_pct = ((btdr['price'] - btdr['prev']) / btdr['prev'])
    btc_pct_factor = btc['pct'] / 100
    vol_state_factor = factors['atr_ratio']
    
    mh, ml = ai_model['high'], ai_model['low']
    pred_h_kalman = mh['intercept'] + (mh['beta_gap'] * current_gap_pct) + (mh['beta_btc'] * btc_pct_factor) + (mh['beta_vol'] * vol_state_factor)
    pred_l_kalman = ml['intercept'] + (ml['beta_gap'] * current_gap_pct) + (ml['beta_btc'] * btc_pct_factor) + (ml['beta_vol'] * vol_state_factor)
    
    live_df = btdr['data']
    live_vol_btdr = live_df['Close'].std() if len(live_df) > 10 else (btdr['price'] * 0.01)
    if np.isnan(live_vol_btdr) or live_vol_btdr == 0: live_vol_btdr = btdr['price'] * 0.01
    live_vol_pct = live_vol_btdr / btdr['price']
    
    w_kalman, w_hist, w_mom, w_ai = 0.3, 0.1, 0.1, 0.5
    final_h_ret = (w_kalman * pred_h_kalman) + (w_hist * ai_model['ensemble_hist_h']) + (w_mom * ai_model['ensemble_mom_h']) + (w_ai * (2.5 * live_vol_pct))
    final_l_ret = (w_kalman * pred_l_kalman) + (w_hist * ai_model['ensemble_hist_l']) + (w_mom * ai_model['ensemble_mom_l']) + (w_ai * (-2.5 * live_vol_pct))
    
    sentiment_adj = (fng_val - 50) * 0.0005
    p_high = btdr['prev'] * (1 + final_h_ret + sentiment_adj)
    p_low = btdr['prev'] * (1 + final_l_ret + sentiment_adj)
    
    atr_buffer = live_vol_btdr * 0.5
    buy_entry = p_low + atr_buffer
    buy_stop = buy_entry - (live_vol_btdr * risk_tol)
    buy_target = p_high - atr_buffer
    
    sell_entry = p_high - atr_buffer
    sell_stop = sell_entry + (live_vol_btdr * risk_tol)
    sell_target = p_low + atr_buffer

    # --- UI 渲染 ---
    st.markdown(f"### ⚡ BTDR Pilot v11.0 <span style='font-size:0.8rem; color:#888'>| 引擎: {ai_status} | FNG: {fng_val}</span>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(card_html("BTDR 现价", f"${btdr['price']:.2f}", f"{btdr['pct']:+.2f}%", btdr['pct']), unsafe_allow_html=True)
    with c2: st.markdown(card_html("BTC 联动", f"${btc['price']:,.0f}", f"{btc['pct']:+.2f}%", btc['pct']), unsafe_allow_html=True)
    
    dist_vwap = ((btdr['price'] - factors['vwap']) / factors['vwap']) * 100 if factors['vwap'] > 0 else 0
    with c3: st.markdown(card_html("VWAP (乖离)", f"${factors['vwap']:.2f}", f"{dist_vwap:+.1f}%", dist_vwap), unsafe_allow_html=True)
    
    pnl = (btdr['pct'] / 100) * (btdr['prev'] * shares)
    with c4: st.markdown(card_html("当日盈亏 (Est.)", f"${pnl:+.0f}", "USD", pnl), unsafe_allow_html=True)

    st.markdown("---")
    
    col_chart, col_tickets = st.columns([2, 1])
    
    with col_chart:
        if not btdr['data'].empty:
            df_plot = btdr['data'].copy()
            if len(df_plot) > 300: df_plot = df_plot.iloc[::2]
            
            fig = go.Figure(data=[go.Candlestick(x=df_plot.index,
                            open=df_plot['Open'], high=df_plot['High'],
                            low=df_plot['Low'], close=df_plot['Close'], name="BTDR")])
            
            fig.add_hline(y=buy_entry, line_dash="dot", line_color="green", annotation_text="Buy Entry")
            fig.add_hline(y=sell_entry, line_dash="dot", line_color="red", annotation_text="Sell Entry")
            fig.add_hline(y=p_high, line_color="rgba(255, 0, 0, 0.3)", annotation_text="High Target")
            fig.add_hline(y=p_low, line_color="rgba(0, 255, 0, 0.3)", annotation_text="Low Support")
            
            fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("图表数据加载中...")

    with col_tickets:
        rr_buy = (buy_target - buy_entry) / (buy_entry - buy_stop) if (buy_entry - buy_stop) > 0 else 0
        z_buy = (btdr['price'] - buy_entry) / (live_vol_btdr * 10)
        buy_prob = max(min((1 - norm.cdf(z_buy)) * 100 * 2, 95), 5)
        
        st.markdown(f"""
        <div class="ticket-card ticket-buy">
            <div style="font-weight:bold; color:#0ca678; margin-bottom:5px;">🟢 智能买单 (BUY LIMIT)</div>
            <div style="display:flex; justify-content:space-between;"><span>挂单</span><span style="font-weight:bold">${buy_entry:.2f}</span></div>
            <div style="display:flex; justify-content:space-between; color:#e03131"><span>止损</span><span>${buy_stop:.2f}</span></div>
            <div style="display:flex; justify-content:space-between; color:#1c7ed6"><span>目标</span><span>${buy_target:.2f}</span></div>
            <div style="margin-top:5px; font-size:0.8rem; color:#666">R/R: {rr_buy:.1f} | 胜率: {buy_prob:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        rr_sell = (sell_entry - sell_target) / (sell_stop - sell_entry) if (sell_stop - sell_entry) > 0 else 0
        st.markdown(f"""
        <div class="ticket-card ticket-sell">
            <div style="font-weight:bold; color:#e03131; margin-bottom:5px;">🔴 智能卖单 (SELL LIMIT)</div>
            <div style="display:flex; justify-content:space-between;"><span>挂单</span><span style="font-weight:bold">${sell_entry:.2f}</span></div>
            <div style="display:flex; justify-content:space-between; color:#e03131"><span>止损</span><span>${sell_stop:.2f}</span></div>
            <div style="display:flex; justify-content:space-between; color:#1c7ed6"><span>目标</span><span>${sell_target:.2f}</span></div>
            <div style="margin-top:5px; font-size:0.8rem; color:#666">R/R: {rr_sell:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.caption("⚒️ 矿股板块 (Real-time Peers)")
    cols = st.columns(len(ai_model['top_peers']))
    for i, p in enumerate(ai_model['top_peers']):
        d = quotes.get(p, {'pct': 0, 'price': 0})
        color = "color-up" if d['pct'] >= 0 else "color-down"
        cols[i].markdown(f"<div class='miner-box'><div style='font-size:0.7rem; color:#888'>{p}</div><div style='font-weight:bold'>${d['price']:.2f}</div><div class='{color}' style='font-size:0.8rem'>{d['pct']:+.1f}%</div></div>", unsafe_allow_html=True)

# --- 主入口 ---
if __name__ == "__main__":
    with st.sidebar:
        st.header("⚙️ Pilot Config")
        shares = st.number_input("持有股数 (BTDR)", value=2000, step=100)
        risk_tol = st.slider("风险偏好 (Vol Mult)", 1.0, 3.0, 2.0)
        
        default_miners = ["MARA", "RIOT", "CLSK", "CORZ", "IREN", "WULF", "CIFR", "HUT"]
        selected_miners = st.multiselect("关注矿股 (Peers)", default_miners, default=default_miners)
    
    show_live_dashboard(shares, risk_tol, selected_miners)
