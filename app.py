import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
import altair as alt
from datetime import datetime, timedelta
import pytz

# --- 1. 页面配置 ---
st.set_page_config(page_title="BTDR Pilot v9.6", layout="centered")

# CSS: 样式定义 (Overnight 呼吸灯 + 布局优化)
st.markdown("""
    <style>
    html { overflow-y: scroll; }
    .stApp > header { display: none; }
    .stApp { margin-top: -30px; background-color: #ffffff; }
    div[data-testid="stStatusWidget"] { visibility: hidden; }
    
    div[data-testid="stAltairChart"] {
        height: 320px !important; min-height: 320px !important; overflow: hidden !important; border: 1px solid #f8f9fa;
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
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        position: relative; cursor: help; transition: transform 0.1s;
    }
    .factor-box:hover { border-color: #ced4da; transform: translateY(-1px); }
    .factor-title { font-size: 0.65rem; color: #999; text-transform: uppercase; letter-spacing: 0.5px; }
    .factor-val { font-size: 1.1rem; font-weight: bold; color: #495057; margin: 2px 0; }
    .factor-sub { font-size: 0.7rem; font-weight: 600; }
    
    .tooltip-text {
        visibility: hidden; width: 180px; background-color: rgba(33, 37, 41, 0.95);
        color: #fff !important; text-align: center; border-radius: 6px; padding: 8px;
        position: absolute; z-index: 999; bottom: 110%; left: 50%; margin-left: -90px;
        opacity: 0; transition: opacity 0.3s; font-size: 0.7rem !important;
        font-weight: normal; line-height: 1.4; pointer-events: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .tooltip-text::after {
        content: ""; position: absolute; top: 100%; left: 50%; margin-left: -5px;
        border-width: 5px; border-style: solid; border-color: rgba(33, 37, 41, 0.95) transparent transparent transparent;
    }
    .factor-box:hover .tooltip-text { visibility: visible; opacity: 1; }
    
    .color-up { color: #0ca678; } .color-down { color: #d6336c; } .color-neutral { color: #adb5bd; }
    
    @keyframes pulse-purple { 0% { box-shadow: 0 0 0 0 rgba(112, 72, 232, 0.4); } 70% { box-shadow: 0 0 0 6px rgba(112, 72, 232, 0); } 100% { box-shadow: 0 0 0 0 rgba(112, 72, 232, 0); } }
    
    .status-dot { height: 6px; width: 6px; border-radius: 50%; display: inline-block; margin-left: 6px; margin-bottom: 2px; }
    .dot-pre { background-color: #f59f00; }
    .dot-reg { background-color: #0ca678; }
    .dot-post { background-color: #1c7ed6; }
    .dot-night { background-color: #7048e8; animation: pulse-purple 2s infinite; }
    .dot-closed { background-color: #adb5bd; }
    
    .pred-container-wrapper { height: 110px; width: 100%; display: block; margin-top: 5px; }
    .pred-box { padding: 0 10px; border-radius: 12px; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center; }
    
    .time-bar { font-size: 0.75rem; color: #999; text-align: center; margin-bottom: 20px; padding: 6px; background: #fafafa; border-radius: 6px; }
    .badge-trend { background:#fd7e14; color:white; padding:1px 4px; border-radius:3px; font-size:0.6rem; }
    .badge-chop { background:#868e96; color:white; padding:1px 4px; border-radius:3px; font-size:0.6rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 辅助函数 ---
def card_html(label, value_str, delta_str=None, delta_val=0, extra_tag=""):
    delta_html = ""
    if delta_str:
        color_class = "color-up" if delta_val >= 0 else "color-down"
        delta_html = f"<div class='metric-delta {color_class}'>{delta_str}</div>"
    return f"""<div class="metric-card"><div class="metric-label">{label} {extra_tag}</div><div class="metric-value">{value_str}</div>{delta_html}</div>"""

def factor_html(title, val, delta_str, delta_val, tooltip_text, reverse_color=False):
    color_class = "color-up" if delta_val >= 0 else "color-down"
    if reverse_color: color_class = "color-down" if delta_val >= 0 else "color-up"
    return f"""
    <div class="factor-box">
        <div class="tooltip-text">{tooltip_text}</div>
        <div class="factor-title">{title}</div>
        <div class="factor-val">{val}</div>
        <div class="factor-sub {color_class}">{delta_str}</div>
    </div>
    """

# --- 3. 核心计算逻辑 ---
@st.cache_data(ttl=300) 
def run_grandmaster_analytics():
    default_model = {"high": {"intercept": 4.29, "beta_open": 0.67, "beta_btc": 0.52}, "low": {"intercept": -3.22, "beta_open": 0.88, "beta_btc": 0.42}, "beta_sector": 0.25}
    default_factors = {"vwap": 0, "adx": 0, "regime": "Neutral", "beta_btc": 1.5, "beta_qqq": 1.2, "rsi": 50, "vol_base": 0.05}
    
    try:
        data = yf.download("BTDR BTC-USD QQQ ^VIX", period="6mo", interval="1d", group_by='ticker', threads=False, progress=False)
        if data.empty: return default_model, default_factors, "No Data"

        btdr = data['BTDR'].dropna(); btc = data['BTC-USD'].dropna(); qqq = data['QQQ'].dropna()
        idx = btdr.index.intersection(btc.index).intersection(qqq.index)
        btdr = btdr.loc[idx]; btc = btc.loc[idx]; qqq = qqq.loc[idx]
        
        ret_btdr = btdr['Close'].pct_change()
        ret_btc = btc['Close'].pct_change()
        ret_qqq = qqq['Close'].pct_change()
        
        cov_btc = ret_btdr.rolling(60).cov(ret_btc).iloc[-1]
        var_btc = ret_btc.rolling(60).var().iloc[-1]
        beta_btc = cov_btc / var_btc if var_btc != 0 else 1.5
        
        cov_qqq = ret_btdr.rolling(60).cov(ret_qqq).iloc[-1]
        var_qqq = ret_qqq.rolling(60).var().iloc[-1]
        beta_qqq = cov_qqq / var_qqq if var_qqq != 0 else 1.2
        
        btdr['TP'] = (btdr['High'] + btdr['Low'] + btdr['Close']) / 3
        btdr['PV'] = btdr['TP'] * btdr['Volume']
        vwap_30d = btdr['PV'].tail(30).sum() / btdr['Volume'].tail(30).sum()
        
        high = btdr['High']; low = btdr['Low']; close = btdr['Close']
        tr = np.maximum(high - low, np.abs(high - close.shift(1)))
        atr = tr.rolling(14).mean()
        
        up_move = high - high.shift(1); down_move = low.shift(1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / atr.replace(0, np.nan))
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean().iloc[-1]
        if np.isnan(adx): adx = 20
        regime = "Trend" if adx > 25 else "Chop"
        
        delta = btdr['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain/loss)).iloc[-1]
        vol_base = ret_btdr.ewm(span=20).std().iloc[-1]

        factors = {"beta_btc": beta_btc, "beta_qqq": beta_qqq, "vwap": vwap_30d, "adx": adx, "regime": regime, "rsi": rsi, "vol_base": vol_base}
        
        df_reg = btdr.tail(30).copy(); df_reg['PrevClose'] = df_reg['Close'].shift(1); df_reg = df_reg.dropna()
        x = ((df_reg['Open'] - df_reg['PrevClose']) / df_reg['PrevClose'] * 100).values
        y_high = ((df_reg['High'] - df_reg['PrevClose']) / df_reg['PrevClose'] * 100).values
        y_low = ((df_reg['Low'] - df_reg['PrevClose']) / df_reg['PrevClose'] * 100).values
        cov_h = np.cov(x, y_high); beta_h = cov_h[0, 1] / cov_h[0, 0] if cov_h[0, 0] != 0 else 0.67
        cov_l = np.cov(x, y_low); beta_l = cov_l[0, 1] / cov_l[0, 0] if cov_l[0, 0] != 0 else 0.88
        
        final_model = {
            "high": {"intercept": 0.7*4.29 + 0.3*(np.mean(y_high)-beta_h*np.mean(x)), "beta_open": 0.7*0.67 + 0.3*np.clip(beta_h,0.3,1.2), "beta_btc": 0.52},
            "low": {"intercept": 0.7*-3.22 + 0.3*(np.mean(y_low)-beta_l*np.mean(x)), "beta_open": 0.7*0.88 + 0.3*np.clip(beta_l,0.4,1.5), "beta_btc": 0.42},
            "beta_sector": 0.25
        }
        return final_model, factors, "Grandmaster"
    except Exception as e:
        print(f"Model Error: {e}")
        return default_model, default_factors, "Offline"

# --- 4. 实时数据 (v9.6 Fix: 双轨抓取 + key修正) ---
def get_realtime_data():
    tickers_list = "BTC-USD BTDR MARA RIOT CORZ CLSK IREN QQQ ^VIX"
    symbols = tickers_list.split()
    
    for attempt in range(3):
        try:
            # A. 基础K线 (用于计算指标)
            daily = yf.download(tickers_list, period="5d", interval="1d", group_by='ticker', threads=False, progress=False)
            
            quotes = {}
            tz_ny = pytz.timezone('America/New_York')
            now_ny = datetime.now(tz_ny)
            today_date = now_ny.date()
            
            # B. 状态判定 (修正版)
            current_minutes = now_ny.hour * 60 + now_ny.minute
            
            state = "Overnight"
            if now_ny.weekday() == 5: state = "Weekend"
            elif now_ny.weekday() == 6 and now_ny.hour < 20: state = "Weekend"
            else:
                if 240 <= current_minutes < 570: state = "Pre-Mkt"
                elif 570 <= current_minutes < 960: state = "Mkt Open"
                elif 960 <= current_minutes < 1200: state = "Post-Mkt"
                else: state = "Overnight"

            for sym in symbols:
                try:
                    df_day = daily[sym] if sym in daily else pd.DataFrame()
                    if not df_day.empty: df_day = df_day.dropna(subset=['Close'])
                    
                    prev_close = 1.0
                    open_price = 0.0
                    is_open_today = False
                    
                    if not df_day.empty:
                        last_day_date = df_day.index[-1].date()
                        if last_day_date == today_date:
                            prev_close = df_day['Close'].iloc[-2] if len(df_day) >= 2 else df_day['Open'].iloc[-1]
                            open_price = df_day['Open'].iloc[-1]
                            is_open_today = True
                        else:
                            prev_close = df_day['Close'].iloc[-1]
                            open_price = prev_close 
                    
                    # --- C. 强力抓取现价 (Fix) ---
                    current_price = 0.0
                    
                    # 1. 尝试 fast_info['lastPrice'] (CamelCase 是关键)
                    try:
                        ticker_obj = yf.Ticker(sym)
                        # 注意：不同版本 yfinance key 可能不同，这里做多重尝试
                        fast_info = ticker_obj.fast_info
                        if hasattr(fast_info, 'last_price'): val = fast_info.last_price
                        elif 'lastPrice' in fast_info: val = fast_info['lastPrice']
                        elif 'regularMarketPrice' in fast_info: val = fast_info['regularMarketPrice']
                        else: val = None
                        
                        if val is not None and str(val) != 'nan':
                            current_price = float(val)
                    except: pass
                    
                    # 2. 如果失败，尝试 info['currentPrice'] (网络请求更重，但包含夜盘)
                    if current_price == 0:
                        try:
                            info = ticker_obj.info
                            if 'currentPrice' in info: current_price = info['currentPrice']
                            elif 'regularMarketPrice' in info: current_price = info['regularMarketPrice']
                        except: pass
                        
                    # 3. 实在不行，回退到 K 线
                    if current_price == 0 and not df_day.empty:
                        current_price = df_day['Close'].iloc[-1]

                    # 容错
                    if current_price == 0 and prev_close > 0: current_price = prev_close

                    pct = ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0
                    
                    quotes[sym] = {
                        "price": current_price, 
                        "pct": pct, 
                        "prev": prev_close, 
                        "open": open_price, 
                        "tag": state,
                        "is_open_today": is_open_today
                    }
                except:
                    quotes[sym] = {"price": 0, "pct": 0, "prev": 0, "open": 0, "tag": "ERR", "is_open_today": False}
            
            try: fng = int(requests.get("https://api.alternative.me/fng/", timeout=1).json()['data'][0]['value'])
            except: fng = 50
            
            return quotes, fng
            
        except Exception:
            time.sleep(0.5)
            continue
            
    return None, 50

# --- 5. Fragment 局部刷新 ---
@st.fragment(run_every=5) 
def show_live_dashboard():
    quotes, fng_val = get_realtime_data()
    ai_model, factors, ai_status = run_grandmaster_analytics()
    
    if not quotes:
        st.warning("📡 夜盘数据连接中 (Connecting to Blue Ocean)...")
        return

    btc_chg = quotes['BTC-USD']['pct']
    qqq_chg = quotes.get('QQQ', {'pct': 0})['pct']
    vix_val = quotes.get('^VIX', {'price': 20})['price']
    vix_chg = quotes.get('^VIX', {'pct': 0})['pct']
    btdr = quotes['BTDR']
    
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny).strftime('%H:%M:%S')
    
    regime_tag = "Trend" if factors['regime'] == "Trend" else "Chop"
    badge_class = "badge-trend" if regime_tag == "Trend" else "badge-chop"
    st.markdown(f"<div class='time-bar'>美东 {now_ny} &nbsp;|&nbsp; 状态: <span class='{badge_class}'>{regime_tag}</span> &nbsp;|&nbsp; 引擎: v9.6 Final Fix</div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1: st.markdown(card_html("BTC (全时段)", f"{btc_chg:+.2f}%", f"{btc_chg:+.2f}%", btc_chg), unsafe_allow_html=True)
    with c2: st.markdown(card_html("恐慌指数", f"{fng_val}", None, 0), unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    st.caption("⚒️ 矿股板块 Beta")
    cols = st.columns(5)
    peers = ["MARA", "RIOT", "CORZ", "CLSK", "IREN"]
    for i, p in enumerate(peers):
        if p in quotes:
            val = quotes[p]['pct']
            cols[i].markdown(card_html(p, f"{val:+.1f}%", f"{val:+.1f}%", val), unsafe_allow_html=True)
            
    st.markdown("---")
    
    c3, c4, c5 = st.columns(3)
    
    state_color_map = {
        "Overnight": "dot-night", 
        "Pre-Mkt": "dot-pre", 
        "Mkt Open": "dot-reg", 
        "Post-Mkt": "dot-post", 
        "Weekend": "dot-closed", 
        "ERR": "dot-closed"
    }
    dot_class = state_color_map.get(btdr.get('tag', 'Overnight'), 'dot-night')
    status_tag = f"<span class='status-dot {dot_class}'></span> <span style='font-size:0.6rem; color:#999'>{btdr['tag']}</span>"
    
    with c3: st.markdown(card_html("BTDR 现价", f"${btdr['price']:.2f}", f"{btdr['pct']:+.2f}%", btdr['pct'], status_tag), unsafe_allow_html=True)
    
    open_val_str = f"${btdr['open']:.2f}"
    if not btdr['is_open_today']:
        open_label = "预计开盘/昨收"
        open_extra = "(Pending)"
    else:
        open_label = "今日开盘"
        open_extra = ""
        
    with c4: st.markdown(card_html(open_label, open_val_str, None, 0, open_extra), unsafe_allow_html=True)

    dist_vwap = ((btdr['price'] - factors['vwap']) / factors['vwap']) * 100 if factors['vwap'] > 0 else 0
    with c5: st.markdown(card_html("机构成本 (VWAP)", f"${factors['vwap']:.2f}", f"{dist_vwap:+.1f}%", dist_vwap), unsafe_allow_html=True)

    # --- 预测 ---
    btdr_open_pct = ((btdr['open'] - btdr['prev']) / btdr['prev']) * 100
    peers_avg = sum(quotes[p]['pct'] for p in peers if p in quotes) / 5
    sector_alpha = peers_avg - btc_chg
    sentiment_adj = (fng_val - 50) * 0.02
    
    MODEL = ai_model
    pred_high_pct = (MODEL['high']['intercept'] + (MODEL['high']['beta_open'] * btdr_open_pct) + (MODEL['high']['beta_btc'] * btc_chg) + (MODEL['beta_sector'] * sector_alpha) + sentiment_adj)
    pred_low_pct = (MODEL['low']['intercept'] + (MODEL['low']['beta_open'] * btdr_open_pct) + (MODEL['low']['beta_btc'] * btc_chg) + (MODEL['beta_sector'] * sector_alpha) + sentiment_adj)
    pred_high_price = btdr['prev'] * (1 + pred_high_pct / 100)
    pred_low_price = btdr['prev'] * (1 + pred_low_pct / 100)
    
    st.markdown("### 🎯 日内阻力/支撑 (Intraday)")
    col_h, col_l = st.columns(2)
    h_bg = "#e6fcf5" if btdr['price'] < pred_high_price else "#0ca678"; h_txt = "#087f5b" if btdr['price'] < pred_high_price else "#ffffff"
    l_bg = "#fff5f5" if btdr['price'] > pred_low_price else "#e03131"; l_txt = "#c92a2a" if btdr['price'] > pred_low_price else "#ffffff"
    
    # [SyntaxError Fix]: 使用多行字符串，避免单行过长导致解析错误
    with col_h: 
        st.markdown(f"""
        <div class="pred-container-wrapper">
            <div class="pred-box" style="background-color: {h_bg}; color: {h_txt}; border: 1px solid #c3fae8;">
                <div style="font-size: 0.8rem; opacity: 0.8;">日内阻力 (High)</div>
                <div style="font-size: 1.5rem; font-weight: bold;">${pred_high_price:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_l: 
        st.markdown(f"""
        <div class="pred-container-wrapper">
            <div class="pred-box" style="background-color: {l_bg}; color: {l_txt}; border: 1px solid #ffc9c9;">
                <div style="font-size: 0.8rem; opacity: 0.8;">日内支撑 (Low)</div>
                <div style="font-size: 1.5rem; font-weight: bold;">${pred_low_price:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- 因子面板 ---
    st.markdown("---")
    st.markdown("### 🌍 宏观环境 (Macro)")
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.markdown(factor_html("QQQ (纳指)", f"{qqq_chg:+.2f}%", "Market", qqq_chg, "科技股大盘风向标。<br>QQQ 跌则 BTDR 承压。"), unsafe_allow_html=True)
    with m2: st.markdown(factor_html("VIX (恐慌)", f"{vix_val:.1f}", f"{vix_chg:+.1f}%", -vix_chg, "市场恐慌指数。<br>>20: 紧张<br>>30: 极度危险 (崩盘风险)", reverse_color=True), unsafe_allow_html=True)
    with m3: st.markdown(factor_html("Beta (BTC)", f"{factors['beta_btc']:.2f}", "Corr", 0, "联动系数。<br>1.5 代表 BTC 涨 1%<br>BTDR 往往能涨 1.5%"), unsafe_allow_html=True)
    with m4: st.markdown(factor_html("Beta (QQQ)", f"{factors['beta_qqq']:.2f}", "Corr", 0, "美股大盘联动系数。<br>数值越高，受美股影响越大。"), unsafe_allow_html=True)
    
    st.markdown("### 🔬 微观结构 (Micro)")
    mi1, mi2, mi3, mi4 = st.columns(4)
    drift_est = (btc_chg/100 * factors['beta_btc'] * 0.4) + (qqq_chg/100 * factors['beta_qqq'] * 0.4)
    if abs(dist_vwap) > 10: drift_est -= (dist_vwap/100) * 0.05
    
    with mi1: st.markdown(factor_html("ADX (强度)", f"{factors['adx']:.1f}", factors['regime'], 1 if factors['adx']>25 else -1, "趋势强度指标。<br>>25 (Trend): 适合顺势<br><20 (Chop): 适合高抛低吸"), unsafe_allow_html=True)
    with mi2: st.markdown(factor_html("RSI (14d)", f"{factors['rsi']:.0f}", "O/B" if factors['rsi']>70 else ("O/S" if factors['rsi']<30 else "Neu"), 0, "强弱指标。<br>>70: 超买 (回调风险)<br><30: 超卖 (反弹机会)"), unsafe_allow_html=True)
    with mi3: st.markdown(factor_html("Implied Vol", f"{factors['vol_base']*100:.1f}%", "Risk", 0, "预期波动率。<br>数值越大，预测范围(喇叭口)<br>张得越开，风险越大。"), unsafe_allow_html=True)
    with mi4: st.markdown(factor_html("Exp. Drift", f"{drift_est*100:+.2f}%", "Day", drift_est, "预期漂移率。<br>模型综合所有因子后，<br>推算出的今日上涨惯性。"), unsafe_allow_html=True)

    # --- 宗师级推演 ---
    st.markdown("### ☁️ 宗师级推演 (P90-P50-P10)")
    
    vol = factors['vol_base']
    drift = drift_est
    if vix_val > 25: drift -= 0.005; vol *= 1.3
    if factors['rsi'] > 75: drift -= 0.003
    if factors['rsi'] < 25: drift += 0.003
    if factors['regime'] == "Chop": drift *= 0.5; vol *= 0.8
    
    simulations = 500; days_ahead = 5; paths = []
    current = btdr['price']
    
    for i in range(simulations):
        path = [current]; p = current
        for d in range(days_ahead):
            shock = np.random.normal(0, 1)
            change = (drift - 0.5 * vol**2) + vol * shock
            p = p * np.exp(change)
            path.append(p)
        paths.append(path)
        
    paths = np.array(paths)
    p90 = np.percentile(paths, 90, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p10 = np.percentile(paths, 10, axis=0)
    
    chart_data = []
    for d in range(days_ahead + 1):
        chart_data.append({
            "Day": d,
            "P90": round(p90[d], 2),
            "P50": round(p50[d], 2),
            "P10": round(p10[d], 2),
        })
    df_chart = pd.DataFrame(chart_data)
    
    base = alt.Chart(df_chart).encode(x=alt.X('Day:O', title='未来交易日'))
    area = base.mark_area(opacity=0.2, color='#4dabf7').encode(y=alt.Y('P10', title='价格预演 (USD)', scale=alt.Scale(zero=False)), y2='P90')
    l90 = base.mark_line(color='#0ca678', strokeDash=[5,5]).encode(y='P90')
    l50 = base.mark_line(color='#228be6', size=3).encode(y='P50')
    l10 = base.mark_line(color='#d6336c', strokeDash=[5,5]).encode(y='P10')
    
    nearest = alt.selection_point(nearest=True, on='mouseover', fields=['Day'], empty=False)
    selectors = base.mark_rule(opacity=0).encode(x='Day:O').add_params(nearest)
    points = base.mark_circle(size=60, color="black").encode(
        y='P50', opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip('Day', title='T+'), alt.Tooltip('P90', title='P90 (High)', format='.2f'), alt.Tooltip('P50', title='P50 (Median)', format='.2f'), alt.Tooltip('P10', title='P10 (Low)', format='.2f')]
    )
    
    st.altair_chart((area + l90 + l50 + l10 + selectors + points).properties(height=300).interactive(), use_container_width=True)
    st.caption(f"Engine: v9.6 Final Fix | Drift: {drift*100:.2f}% | Vol: {vol*100:.1f}%")

# --- 7. 主程序入口 ---
st.markdown("### ⚡ BTDR 领航员 v9.6 Final Fix")
show_live_dashboard()
