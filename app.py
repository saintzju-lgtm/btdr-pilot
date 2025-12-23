import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
import altair as alt
from datetime import datetime, time as dt_time
import pytz

# --- 1. 页面配置 & 样式分离 ---
st.set_page_config(page_title="BTDR Pilot v9.5 Optimized", layout="centered")

CUSTOM_CSS = """
<style>
    html { overflow-y: scroll; }
    .stApp > header { display: none; }
    .stApp { margin-top: -30px; background-color: #ffffff; }
    div[data-testid="stStatusWidget"] { visibility: hidden; }
    
    h1, h2, h3, div, p, span { 
        color: #212529 !important; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important; 
    }
    
    /* 锁定图表高度 */
    div[data-testid="stAltairChart"] {
        height: 320px !important; min-height: 320px !important;
        overflow: hidden !important; border: 1px solid #f8f9fa;
    }
    
    /* 统一卡片样式 */
    .metric-card {
        background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02); height: 95px; padding: 0 16px;
        display: flex; flex-direction: column; justify-content: center;
    }
    .metric-label { font-size: 0.75rem; color: #888; margin-bottom: 2px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #212529; line-height: 1.2; }
    .metric-delta { font-size: 0.9rem; font-weight: 600; margin-top: 2px; }
    
    /* 因子卡片 & Tooltips */
    .factor-box {
        background: #fff; border: 1px solid #eee; border-radius: 8px; padding: 6px; text-align: center;
        height: 75px; display: flex; flex-direction: column; justify-content: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02); position: relative; cursor: help; transition: transform 0.1s;
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
        border-width: 5px; border-style: solid;
        border-color: rgba(33, 37, 41, 0.95) transparent transparent transparent;
    }
    .factor-box:hover .tooltip-text { visibility: visible; opacity: 1; }
    
    /* 颜色定义 */
    .color-up { color: #0ca678; } .color-down { color: #d6336c; } .color-neutral { color: #adb5bd; }
    
    /* 状态小圆点 */
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

def factor_html(title, val, delta_str, delta_val, tooltip_text, reverse_color=False):
    is_positive = delta_val >= 0
    if reverse_color: is_positive = not is_positive
    color_class = "color-up" if is_positive else "color-down"
    return f"""
    <div class="factor-box">
        <div class="tooltip-text">{tooltip_text}</div>
        <div class="factor-title">{title}</div>
        <div class="factor-val">{val}</div>
        <div class="factor-sub {color_class}">{delta_str}</div>
    </div>
    """

# --- 3. 核心计算逻辑 ---
@st.cache_data(ttl=600) # 延长缓存时间，减少非必要重算
def run_grandmaster_analytics():
    # 默认值
    default_model = {
        "high": {"intercept": 4.29, "beta_open": 0.67, "beta_btc": 0.52}, 
        "low": {"intercept": -3.22, "beta_open": 0.88, "beta_btc": 0.42}, 
        "beta_sector": 0.25
    }
    default_factors = {"vwap": 0, "adx": 20, "regime": "Neutral", "beta_btc": 1.5, "beta_qqq": 1.2, "rsi": 50, "vol_base": 0.05}
    
    try:
        # 下载数据 (仅下载日线，分钟线在 realtime 处获取)
        # period='6mo' 足够计算 rolling(60)
        data = yf.download("BTDR BTC-USD QQQ ^VIX", period="6mo", interval="1d", group_by='ticker', threads=True, progress=False)
        
        if data.empty: return default_model, default_factors, "No Data"

        # 预处理：对齐索引
        btdr, btc, qqq = data['BTDR'].dropna(), data['BTC-USD'].dropna(), data['QQQ'].dropna()
        idx = btdr.index.intersection(btc.index).intersection(qqq.index)
        btdr, btc, qqq = btdr.loc[idx], btc.loc[idx], qqq.loc[idx]
        
        if len(btdr) < 60: return default_model, default_factors, "Insufficient Data"

        # 向量化计算收益率
        ret_btdr, ret_btc, ret_qqq = btdr['Close'].pct_change(), btc['Close'].pct_change(), qqq['Close'].pct_change()
        
        # Beta 计算
        cov_btc = ret_btdr.rolling(60).cov(ret_btc).iloc[-1]
        var_btc = ret_btc.rolling(60).var().iloc[-1]
        beta_btc = cov_btc / var_btc if var_btc > 1e-6 else 1.5
        
        cov_qqq = ret_btdr.rolling(60).cov(ret_qqq).iloc[-1]
        var_qqq = ret_qqq.rolling(60).var().iloc[-1]
        beta_qqq = cov_qqq / var_qqq if var_qqq > 1e-6 else 1.2
        
        # VWAP (Simple approximation)
        pv = (btdr['Close'] * btdr['Volume']) # 使用Close代替TP以简化计算，差异极小
        vwap_30d = pv.tail(30).sum() / btdr['Volume'].tail(30).sum()
        
        # ADX Calculation
        high, low, close = btdr['High'], btdr['Low'], btdr['Close']
        tr = np.maximum(high - low, np.abs(high - close.shift(1)))
        atr = tr.rolling(14).mean()
        
        up = high.diff()
        down = -low.diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        
        plus_di = 100 * (pd.Series(plus_dm, index=btdr.index).rolling(14).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm, index=btdr.index).rolling(14).mean() / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean().iloc[-1]
        adx = 20 if np.isnan(adx) else adx
        
        # RSI & Volatility
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        vol_base = ret_btdr.ewm(span=20).std().iloc[-1]

        factors = {
            "beta_btc": beta_btc, "beta_qqq": beta_qqq, "vwap": vwap_30d, 
            "adx": adx, "regime": "Trend" if adx > 25 else "Chop", 
            "rsi": rsi, "vol_base": vol_base
        }
        
        # 日内模型回归 (简单的 OLS)
        df_reg = btdr.tail(30).copy()
        df_reg['PrevClose'] = df_reg['Close'].shift(1)
        df_reg = df_reg.dropna()
        
        x = ((df_reg['Open'] - df_reg['PrevClose']) / df_reg['PrevClose'] * 100).values
        y_high = ((df_reg['High'] - df_reg['PrevClose']) / df_reg['PrevClose'] * 100).values
        y_low = ((df_reg['Low'] - df_reg['PrevClose']) / df_reg['PrevClose'] * 100).values
        
        # 简单的线性回归斜率: cov(x,y)/var(x)
        var_x = np.var(x, ddof=1)
        beta_h = np.cov(x, y_high)[0, 1] / var_x if var_x > 0 else 0.67
        beta_l = np.cov(x, y_low)[0, 1] / var_x if var_x > 0 else 0.88
        
        final_model = {
            "high": {
                "intercept": 0.7*4.29 + 0.3*(np.mean(y_high)-beta_h*np.mean(x)), 
                "beta_open": 0.7*0.67 + 0.3*np.clip(beta_h, 0.3, 1.2), 
                "beta_btc": 0.52
            },
            "low": {
                "intercept": 0.7*-3.22 + 0.3*(np.mean(y_low)-beta_l*np.mean(x)), 
                "beta_open": 0.7*0.88 + 0.3*np.clip(beta_l, 0.4, 1.5), 
                "beta_btc": 0.42
            },
            "beta_sector": 0.25
        }
        return final_model, factors, "Grandmaster"
        
    except Exception as e:
        print(f"Model Error: {e}")
        return default_model, default_factors, "Offline"

# --- 4. 实时数据 (优化 Fetch 逻辑) ---
def determine_market_state(now_ny):
    """根据纽约时间确定市场状态，返回 Tag 和 CSS Class"""
    weekday = now_ny.weekday() # 0=Mon, 6=Sun
    curr_min = now_ny.hour * 60 + now_ny.minute
    
    # 1. 周末判断
    if weekday == 5: return "Weekend", "dot-closed" # Sat
    if weekday == 6 and now_ny.hour < 20: return "Weekend", "dot-closed" # Sun before 8PM
    
    # 2. 交易日逻辑 (含周日夜盘)
    if 240 <= curr_min < 570: return "Pre-Mkt", "dot-pre"     # 04:00 - 09:30
    if 570 <= curr_min < 960: return "Mkt Open", "dot-reg"    # 09:30 - 16:00
    if 960 <= curr_min < 1200: return "Post-Mkt", "dot-post"  # 16:00 - 20:00
    
    return "Overnight", "dot-night" # 20:00 - 04:00

def get_realtime_data():
    tickers_list = "BTC-USD BTDR MARA RIOT CORZ CLSK IREN QQQ ^VIX"
    symbols = tickers_list.split()
    
    try:
        # OPTIMIZATION: 仅下载最近 2 天的分钟数据（减少延迟），日线维持 5 天
        # 使用 threading 提高并发速度
        daily = yf.download(tickers_list, period="5d", interval="1d", group_by='ticker', threads=True, progress=False)
        live = yf.download(tickers_list, period="2d", interval="1m", prepost=True, group_by='ticker', threads=True, progress=False)
        
        if daily.empty: raise ValueError("Empty Data")
        
        quotes = {}
        tz_ny = pytz.timezone('America/New_York')
        now_ny = datetime.now(tz_ny)
        today_date = now_ny.date()
        
        state_tag, state_css = determine_market_state(now_ny)

        for sym in symbols:
            try:
                # 获取数据切片
                df_day = daily[sym].dropna(subset=['Close']) if sym in daily else pd.DataFrame()
                df_min = live[sym].dropna(subset=['Close']) if sym in live else pd.DataFrame()
                
                # 1. 现价逻辑: 优先取最新的分钟线，没有则取日线
                if not df_min.empty:
                    current_price = df_min['Close'].iloc[-1]
                elif not df_day.empty:
                    current_price = df_day['Close'].iloc[-1]
                else:
                    current_price = 0.0

                # 2. 昨收逻辑 (Prev Close)
                prev_close = 1.0
                open_price = 0.0
                is_open_today = False
                
                if not df_day.empty:
                    last_day_date = df_day.index[-1].date()
                    # 如果日线数据的最后一天是“今天”
                    if last_day_date == today_date:
                        is_open_today = True
                        open_price = df_day['Open'].iloc[-1]
                        if len(df_day) >= 2:
                            prev_close = df_day['Close'].iloc[-2]
                        else:
                            prev_close = df_day['Open'].iloc[-1] # fallback
                    else:
                        # 还没开盘日线，昨收就是最后一条 Close
                        prev_close = df_day['Close'].iloc[-1]
                        open_price = prev_close # 暂定
                
                pct = ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0
                
                quotes[sym] = {
                    "price": current_price, "pct": pct, "prev": prev_close,
                    "open": open_price, "tag": state_tag, "css": state_css,
                    "is_open_today": is_open_today
                }
            except Exception:
                quotes[sym] = {"price": 0, "pct": 0, "prev": 1, "open": 0, "tag": "ERR", "css": "dot-closed", "is_open_today": False}
        
        # 外部 API 异步处理建议：为了简单起见，这里设置极短超时
        try:
            fng = int(requests.get("https://api.alternative.me/fng/", timeout=0.8).json()['data'][0]['value'])
        except:
            fng = 50
            
        return quotes, fng

    except Exception as e:
        return None, 50

# --- 5. Fragment 局部刷新 ---
@st.fragment(run_every=10) # 稍微增加间隔以减轻 API 负担
def show_live_dashboard():
    quotes, fng_val = get_realtime_data()
    ai_model, factors, ai_status = run_grandmaster_analytics()
    
    if not quotes:
        st.warning("📡 连接中 (Initializing)...")
        time.sleep(1)
        st.rerun()
        return

    # 数据解包
    btc = quotes.get('BTC-USD', {'pct': 0})
    qqq = quotes.get('QQQ', {'pct': 0})
    vix = quotes.get('^VIX', {'price': 20, 'pct': 0})
    btdr = quotes.get('BTDR', {'price': 0})
    
    # 顶部状态栏
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny).strftime('%H:%M:%S')
    regime_tag = factors['regime']
    badge_class = "badge-trend" if regime_tag == "Trend" else "badge-chop"
    st.markdown(f"<div class='time-bar'>美东 {now_ny} &nbsp;|&nbsp; 状态: <span class='{badge_class}'>{regime_tag}</span> &nbsp;|&nbsp; 引擎: v9.5 Optimized</div>", unsafe_allow_html=True)
    
    # 主要 Metrics
    c1, c2 = st.columns(2)
    with c1: st.markdown(card_html("BTC (24h)", f"{btc['pct']:+.2f}%", f"{btc['pct']:+.2f}%", btc['pct']), unsafe_allow_html=True)
    with c2: st.markdown(card_html("恐慌指数", f"{fng_val}", None, 0), unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    st.caption("⚒️ 矿股板块 Beta")
    
    cols = st.columns(5)
    peers = ["MARA", "RIOT", "CORZ", "CLSK", "IREN"]
    peers_pcts = []
    for i, p in enumerate(peers):
        val = quotes.get(p, {'pct': 0})['pct']
        peers_pcts.append(val)
        cols[i].markdown(card_html(p, f"{val:+.1f}%", f"{val:+.1f}%", val), unsafe_allow_html=True)
            
    st.markdown("---")
    
    # BTDR 核心面板
    c3, c4, c5 = st.columns(3)
    status_tag = f"<span class='status-dot {btdr['css']}'></span> <span style='font-size:0.6rem; color:#999'>{btdr['tag']}</span>"
    
    with c3: st.markdown(card_html("BTDR 现价", f"${btdr['price']:.2f}", f"{btdr['pct']:+.2f}%", btdr['pct'], status_tag), unsafe_allow_html=True)
    
    open_label = "今日开盘" if btdr['is_open_today'] else "预计开盘/昨收"
    open_extra = "" if btdr['is_open_today'] else "(Pending)"
    with c4: st.markdown(card_html(open_label, f"${btdr['open']:.2f}", None, 0, open_extra), unsafe_allow_html=True)

    dist_vwap = ((btdr['price'] - factors['vwap']) / factors['vwap']) * 100 if factors['vwap'] > 0 else 0
    with c5: st.markdown(card_html("机构成本 (VWAP)", f"${factors['vwap']:.2f}", f"{dist_vwap:+.1f}%", dist_vwap), unsafe_allow_html=True)

    # --- 预测模型计算 ---
    btdr_open_pct = ((btdr['open'] - btdr['prev']) / btdr['prev']) * 100 if btdr['prev'] else 0
    peers_avg = np.mean(peers_pcts) if peers_pcts else 0
    sector_alpha = peers_avg - btc['pct']
    sentiment_adj = (fng_val - 50) * 0.02
    
    # 提取模型参数
    mh, ml = ai_model['high'], ai_model['low']
    beta_sec = ai_model['beta_sector']
    
    # 计算预测涨跌幅
    pred_h_pct = mh['intercept'] + (mh['beta_open'] * btdr_open_pct) + (mh['beta_btc'] * btc['pct']) + (beta_sec * sector_alpha) + sentiment_adj
    pred_l_pct = ml['intercept'] + (ml['beta_open'] * btdr_open_pct) + (ml['beta_btc'] * btc['pct']) + (beta_sec * sector_alpha) + sentiment_adj
    
    p_high = btdr['prev'] * (1 + pred_h_pct / 100)
    p_low = btdr['prev'] * (1 + pred_l_pct / 100)
    
    st.markdown("### 🎯 日内阻力/支撑 (Intraday)")
    col_h, col_l = st.columns(2)
    
    # 样式动态调整
    h_bg = "#e6fcf5" if btdr['price'] < p_high else "#0ca678"; h_txt = "#087f5b" if btdr['price'] < p_high else "#ffffff"
    l_bg = "#fff5f5" if btdr['price'] > p_low else "#e03131"; l_txt = "#c92a2a" if btdr['price'] > p_low else "#ffffff"
    
    with col_h: st.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background-color: {h_bg}; color: {h_txt}; border: 1px solid #c3fae8;"><div style="font-size: 0.8rem; opacity: 0.8;">日内阻力 (High)</div><div style="font-size: 1.5rem; font-weight: bold;">${p_high:.2f}</div></div></div>""", unsafe_allow_html=True)
    with col_l: st.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background-color: {l_bg}; color: {l_txt}; border: 1px solid #ffc9c9;"><div style="font-size: 0.8rem; opacity: 0.8;">日内支撑 (Low)</div><div style="font-size: 1.5rem; font-weight: bold;">${p_low:.2f}</div></div></div>""", unsafe_allow_html=True)

    # --- 因子面板 ---
    st.markdown("---")
    st.markdown("### 🌍 宏观环境 (Macro)")
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.markdown(factor_html("QQQ (纳指)", f"{qqq['pct']:+.2f}%", "Market", qqq['pct'], "科技股大盘风向标。"), unsafe_allow_html=True)
    with m2: st.markdown(factor_html("VIX (恐慌)", f"{vix['price']:.1f}", f"{vix['pct']:+.1f}%", -vix['pct'], "市场恐慌指数，>25需警惕。", reverse_color=True), unsafe_allow_html=True)
    with m3: st.markdown(factor_html("Beta (BTC)", f"{factors['beta_btc']:.2f}", "Corr", 0, "相对 BTC 的波动系数。"), unsafe_allow_html=True)
    with m4: st.markdown(factor_html("Beta (QQQ)", f"{factors['beta_qqq']:.2f}", "Corr", 0, "相对纳指的波动系数。"), unsafe_allow_html=True)
    
    st.markdown("### 🔬 微观结构 (Micro)")
    mi1, mi2, mi3, mi4 = st.columns(4)
    
    drift_est = (btc['pct']/100 * factors['beta_btc'] * 0.4) + (qqq['pct']/100 * factors['beta_qqq'] * 0.4)
    if abs(dist_vwap) > 10: drift_est -= (dist_vwap/100) * 0.05
    
    rsi_val = factors['rsi']
    rsi_status = "O/B" if rsi_val > 70 else ("O/S" if rsi_val < 30 else "Neu")
    
    with mi1: st.markdown(factor_html("ADX (强度)", f"{factors['adx']:.1f}", factors['regime'], 1 if factors['adx']>25 else -1, "趋势强度指标，>25为趋势。"), unsafe_allow_html=True)
    with mi2: st.markdown(factor_html("RSI (14d)", f"{rsi_val:.0f}", rsi_status, 0, "强弱指标，>70超买，<30超卖。"), unsafe_allow_html=True)
    with mi3: st.markdown(factor_html("Implied Vol", f"{factors['vol_base']*100:.1f}%", "Risk", 0, "预测波动率 (基于 EWM Std)。"), unsafe_allow_html=True)
    with mi4: st.markdown(factor_html("Exp. Drift", f"{drift_est*100:+.2f}%", "Day", drift_est, "模型推算的今日上涨惯性。"), unsafe_allow_html=True)

    # --- 宗师级推演 (Vectorized Monte Carlo) ---
    st.markdown("### ☁️ 宗师级推演 (P90-P50-P10)")
    
    # 1. 参数调整
    vol = factors['vol_base']
    drift = drift_est
    if vix['price'] > 25: drift -= 0.005; vol *= 1.3
    if rsi_val > 75: drift -= 0.003
    if rsi_val < 25: drift += 0.003
    if factors['regime'] == "Chop": drift *= 0.5; vol *= 0.8
    
    # 2. 向量化蒙特卡洛 (速度优化关键)
    simulations = 1000 # 增加模拟次数以获得更平滑的曲线，因已优化速度
    days_ahead = 5
    dt = 1 # day step
    
    # 生成随机震荡矩阵 (simulations x days_ahead)
    random_shocks = np.random.normal(0, 1, (simulations, days_ahead))
    # 每日漂移项 (GBM Formula)
    daily_drift = (drift - 0.5 * vol**2) * dt
    daily_shock = vol * np.sqrt(dt) * random_shocks
    
    # 计算每日收益率乘数
    daily_returns = np.exp(daily_drift + daily_shock)
    
    # 累乘得到价格路径
    price_paths = np.zeros((simulations, days_ahead + 1))
    price_paths[:, 0] = btdr['price']
    
    # 累积乘积 (Cumprod)
    price_paths[:, 1:] = btdr['price'] * np.cumprod(daily_returns, axis=1)
    
    # 计算百分位
    percentiles = np.percentile(price_paths, [10, 50, 90], axis=0) # Shape: (3, 6)
    p10, p50, p90 = percentiles[0], percentiles[1], percentiles[2]
    
    # 构建图表数据
    chart_data = pd.DataFrame({
        "Day": np.arange(days_ahead + 1),
        "P90": np.round(p90, 2),
        "P50": np.round(p50, 2),
        "P10": np.round(p10, 2)
    })
    
    # Altair 绘图
    base = alt.Chart(chart_data).encode(x=alt.X('Day:O', title='未来交易日 (T+)'))
    area = base.mark_area(opacity=0.2, color='#4dabf7').encode(y=alt.Y('P10', title='价格预演 (USD)', scale=alt.Scale(zero=False)), y2='P90')
    l90 = base.mark_line(color='#0ca678', strokeDash=[5,5]).encode(y='P90')
    l50 = base.mark_line(color='#228be6', size=3).encode(y='P50')
    l10 = base.mark_line(color='#d6336c', strokeDash=[5,5]).encode(y='P10')
    
    nearest = alt.selection_point(nearest=True, on='mouseover', fields=['Day'], empty=False)
    selectors = base.mark_rule(opacity=0).encode(x='Day:O').add_params(nearest)
    points = base.mark_circle(size=60, color="black").encode(
        y='P50', opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip('Day'), alt.Tooltip('P90'), alt.Tooltip('P50'), alt.Tooltip('P10')]
    )
    
    st.altair_chart((area + l90 + l50 + l10 + selectors + points).properties(height=300).interactive(), use_container_width=True)
    st.caption(f"Engine: v9.5 Vectorized | Drift: {drift*100:.2f}% | Vol: {vol*100:.1f}% | Sims: {simulations}")

# --- 6. 主程序入口 ---
st.markdown("### ⚡ BTDR 领航员 v9.5 Optimized")
show_live_dashboard()
