import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
import altair as alt
from datetime import datetime, date
import pytz

# --- 1. 页面配置 & CSS ---
st.set_page_config(page_title="BTDR Pilot v9.5 Optimized", layout="centered")

# CSS 保持原有风格，合并部分选择器以减小体积
st.markdown("""
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
        height: 320px !important; min-height: 320px !important; overflow: hidden !important; border: 1px solid #f8f9fa;
    }
    canvas { transition: opacity 0.2s ease-in-out; }
    
    /* 卡片样式 */
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
    
    /* 颜色 & 状态点 */
    .color-up { color: #0ca678; } .color-down { color: #d6336c; } .color-neutral { color: #adb5bd; }
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
    """, unsafe_allow_html=True)

# --- 2. 核心逻辑 (分离计算) ---

@st.cache_data(ttl=3600)  # 历史分析缓存1小时，因为Beta/Regime不需要秒级更新
def fetch_historical_analysis():
    """计算基于日线的宏观指标 (Beta, Regime, VWAP, Support/Resist Models)"""
    default_model = {"high": {"intercept": 4.29, "beta_open": 0.67, "beta_btc": 0.52}, 
                     "low": {"intercept": -3.22, "beta_open": 0.88, "beta_btc": 0.42}, 
                     "beta_sector": 0.25}
    default_factors = {"vwap": 0, "adx": 0, "regime": "Neutral", "beta_btc": 1.5, 
                       "beta_qqq": 1.2, "rsi": 50, "vol_base": 0.05}

    try:
        # 下载较长周期数据用于分析
        tickers = "BTDR BTC-USD QQQ"
        data = yf.download(tickers, period="3mo", interval="1d", group_by='ticker', threads=True, progress=False)
        
        if data.empty: return default_model, default_factors, "No Data"

        # 处理 MultiIndex (兼容新旧版 yfinance)
        def get_close(sym):
            return data[sym]['Close'] if sym in data else pd.Series()
        
        btdr_c = get_close('BTDR').dropna()
        btc_c = get_close('BTC-USD').dropna()
        qqq_c = get_close('QQQ').dropna()
        
        # 对齐索引
        idx = btdr_c.index.intersection(btc_c.index).intersection(qqq_c.index)
        if len(idx) < 30: return default_model, default_factors, "Insuf Data"

        # 1. 计算 Beta
        ret_btdr = btdr_c.loc[idx].pct_change()
        ret_btc = btc_c.loc[idx].pct_change()
        ret_qqq = qqq_c.loc[idx].pct_change()

        def calc_beta(r_asset, r_bench, window=60):
            cov = r_asset.rolling(window).cov(r_bench).iloc[-1]
            var = r_bench.rolling(window).var().iloc[-1]
            return cov / var if var != 0 else 1.0

        beta_btc = calc_beta(ret_btdr, ret_btc)
        beta_qqq = calc_beta(ret_btdr, ret_qqq)

        # 2. 计算技术指标 (使用 BTDR 完整 OHLC)
        btdr_full = data['BTDR'].loc[idx]
        # VWAP (简易版 30天)
        tp = (btdr_full['High'] + btdr_full['Low'] + btdr_full['Close']) / 3
        vwap = (tp * btdr_full['Volume']).tail(30).sum() / btdr_full['Volume'].tail(30).sum()

        # ADX (简化计算)
        high, low, close = btdr_full['High'], btdr_full['Low'], btdr_full['Close']
        tr = np.maximum(high - low, np.abs(high - close.shift(1)))
        atr = tr.rolling(14).mean()
        up = high - high.shift(1)
        down = low.shift(1) - low
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        adx = (100 * pd.Series(np.abs(plus_dm - minus_dm)).rolling(14).mean() / atr).rolling(14).mean().iloc[-1]
        adx = 20 if np.isnan(adx) else adx
        regime = "Trend" if adx > 25 else "Chop"

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain/loss)).iloc[-1]
        
        # Volatility
        vol_base = ret_btdr.ewm(span=20).std().iloc[-1]

        # 3. 训练简单的预测模型 (High/Low Regression)
        # 此处简化为更新 intercept，保留 weights 以节省计算
        df_reg = btdr_full.tail(30).copy()
        df_reg['PrevClose'] = df_reg['Close'].shift(1)
        df_reg = df_reg.dropna()
        
        x = ((df_reg['Open'] - df_reg['PrevClose']) / df_reg['PrevClose'] * 100).values
        y_h = ((df_reg['High'] - df_reg['PrevClose']) / df_reg['PrevClose'] * 100).values
        y_l = ((df_reg['Low'] - df_reg['PrevClose']) / df_reg['PrevClose'] * 100).values
        
        # 简单均值修正 Intercept
        model = {
            "high": {"intercept": np.mean(y_h - 0.67*x), "beta_open": 0.67, "beta_btc": 0.52},
            "low": {"intercept": np.mean(y_l - 0.88*x), "beta_open": 0.88, "beta_btc": 0.42},
            "beta_sector": 0.25
        }
        
        factors = {
            "beta_btc": beta_btc, "beta_qqq": beta_qqq, "vwap": vwap, 
            "adx": adx, "regime": regime, "rsi": rsi, "vol_base": vol_base
        }
        return model, factors, "Online"

    except Exception as e:
        print(f"Stats Error: {e}")
        return default_model, default_factors, "Offline"

def determine_market_state(now_ny):
    """判断当前市场时段状态"""
    current_minutes = now_ny.hour * 60 + now_ny.minute
    
    if now_ny.weekday() == 5: return "Weekend" # Sat
    if now_ny.weekday() == 6 and now_ny.hour < 20: return "Weekend" # Sun < 8PM
    
    if 240 <= current_minutes < 570: return "Pre-Mkt"
    if 570 <= current_minutes < 960: return "Mkt Open"
    if 960 <= current_minutes < 1200: return "Post-Mkt"
    return "Overnight"

def get_realtime_quotes():
    """获取实时报价 (轻量级)"""
    tickers = ["BTC-USD", "BTDR", "MARA", "RIOT", "CORZ", "CLSK", "IREN", "QQQ", "^VIX"]
    
    # 尝试最多3次，减少超时等待
    for _ in range(3):
        try:
            # 只取最近2天数据以加快速度，间隔1分钟
            # 注意: yfinance 某些时候需要 period="5d" 才能跨越周末拿到 prev_close，但我们可以单独处理
            live = yf.download(tickers, period="5d", interval="1m", group_by='ticker', threads=True, progress=False)
            if live.empty: continue

            quotes = {}
            tz_ny = pytz.timezone('America/New_York')
            now_ny = datetime.now(tz_ny)
            market_state = determine_market_state(now_ny)
            today_date = now_ny.date()

            for sym in tickers:
                if sym not in live: continue
                df = live[sym].dropna(subset=['Close'])
                if df.empty: continue
                
                # 1. 现价
                curr = df['Close'].iloc[-1]
                
                # 2. 昨收 & 开盘逻辑
                # 重新采样日线以获得准确的昨收 (不用再次下载，直接 resample)
                df_daily = df.resample('D').agg({'Open':'first', 'Close':'last'}).dropna()
                
                prev_close = curr # 默认
                open_price = curr # 默认
                is_open = False
                
                if len(df_daily) >= 1:
                    last_day = df_daily.index[-1].date()
                    if last_day == today_date:
                        open_price = df_daily['Open'].iloc[-1]
                        is_open = True
                        if len(df_daily) >= 2:
                            prev_close = df_daily['Close'].iloc[-2]
                        else:
                            # 只有今天数据，昨收可能是开盘价或需要额外逻辑，此处简化
                            prev_close = df_daily['Open'].iloc[-1] 
                    else:
                        prev_close = df_daily['Close'].iloc[-1]
                        open_price = prev_close # 还没开盘

                pct = ((curr - prev_close) / prev_close * 100) if prev_close > 0 else 0
                
                quotes[sym] = {
                    "price": curr, "pct": pct, "prev": prev_close, 
                    "open": open_price, "tag": market_state, "is_open_today": is_open
                }
            
            # FNG Index (Fear and Greed)
            try:
                fng = int(requests.get("https://api.alternative.me/fng/", timeout=1).json()['data'][0]['value'])
            except:
                fng = 50
                
            return quotes, fng, now_ny

        except Exception:
            time.sleep(0.2)
            
    return None, 50, datetime.now(pytz.timezone('America/New_York'))

# --- 3. 辅助 HTML 生成 ---
def card_html(label, value_str, delta_str=None, delta_val=0, extra_tag=""):
    delta_html = ""
    if delta_str:
        color_class = "color-up" if delta_val >= 0 else "color-down"
        delta_html = f"<div class='metric-delta {color_class}'>{delta_str}</div>"
    return f"""<div class="metric-card"><div class="metric-label">{label} {extra_tag}</div><div class="metric-value">{value_str}</div>{delta_html}</div>"""

def factor_html(title, val, delta_str, delta_val, tooltip, reverse_color=False):
    color_class = "color-up" if delta_val >= 0 else "color-down"
    if reverse_color: color_class = "color-down" if delta_val >= 0 else "color-up"
    return f"""
    <div class="factor-box">
        <div class="tooltip-text">{tooltip}</div>
        <div class="factor-title">{title}</div>
        <div class="factor-val">{val}</div>
        <div class="factor-sub {color_class}">{delta_str}</div>
    </div>
    """

# --- 4. 界面渲染 Fragment ---
@st.fragment(run_every=5) 
def show_live_dashboard():
    # 获取数据
    quotes, fng_val, now_ny = get_realtime_quotes()
    model, factors, _ = fetch_historical_analysis() # 这个是缓存的，很快
    
    if not quotes or 'BTDR' not in quotes:
        st.warning("📡 连接市场数据中 (Syncing)...")
        return

    # 提取变量
    btdr = quotes['BTDR']
    btc = quotes.get('BTC-USD', {'pct': 0})
    qqq = quotes.get('QQQ', {'pct': 0})
    vix = quotes.get('^VIX', {'price': 20, 'pct': 0})
    
    # 顶部状态栏
    regime_tag = factors['regime']
    badge_class = "badge-trend" if regime_tag == "Trend" else "badge-chop"
    st.markdown(f"<div class='time-bar'>美东 {now_ny.strftime('%H:%M:%S')} &nbsp;|&nbsp; 状态: <span class='{badge_class}'>{regime_tag}</span> &nbsp;|&nbsp; 引擎: v9.5 Optimized</div>", unsafe_allow_html=True)
    
    # 第一行：主要指标
    c1, c2 = st.columns(2)
    with c1: st.markdown(card_html("BTC (24h)", f"{btc['pct']:+.2f}%", f"{btc['pct']:+.2f}%", btc['pct']), unsafe_allow_html=True)
    with c2: st.markdown(card_html("FNG 指数", f"{fng_val}", "Market", 0), unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    st.caption("⚒️ 矿股板块 Beta")
    
    # 矿股列表
    cols = st.columns(5)
    peers = ["MARA", "RIOT", "CORZ", "CLSK", "IREN"]
    peers_pct_sum = 0
    valid_peers = 0
    for i, p in enumerate(peers):
        if p in quotes:
            val = quotes[p]['pct']
            cols[i].markdown(card_html(p, f"{val:+.1f}%", f"{val:+.1f}%", val), unsafe_allow_html=True)
            peers_pct_sum += val
            valid_peers += 1
            
    st.markdown("---")
    
    # BTDR 核心面板
    c3, c4, c5 = st.columns(3)
    
    state_map = {"Overnight": "dot-night", "Pre-Mkt": "dot-pre", "Mkt Open": "dot-reg", 
                 "Post-Mkt": "dot-post", "Weekend": "dot-closed"}
    dot_class = state_map.get(btdr['tag'], 'dot-closed')
    status_html = f"<span class='status-dot {dot_class}'></span> <span style='font-size:0.6rem; color:#999'>{btdr['tag']}</span>"
    
    with c3: st.markdown(card_html("BTDR 现价", f"${btdr['price']:.2f}", f"{btdr['pct']:+.2f}%", btdr['pct'], status_html), unsafe_allow_html=True)
    
    open_lbl = "今日开盘" if btdr['is_open_today'] else "预计开盘/昨收"
    open_extra = "" if btdr['is_open_today'] else "(Pending)"
    with c4: st.markdown(card_html(open_lbl, f"${btdr['open']:.2f}", None, 0, open_extra), unsafe_allow_html=True)

    dist_vwap = ((btdr['price'] - factors['vwap']) / factors['vwap']) * 100 if factors['vwap'] > 0 else 0
    with c5: st.markdown(card_html("机构成本 (VWAP)", f"${factors['vwap']:.2f}", f"{dist_vwap:+.1f}%", dist_vwap), unsafe_allow_html=True)

    # 预测逻辑
    open_change = ((btdr['open'] - btdr['prev']) / btdr['prev']) * 100
    sector_alpha = (peers_pct_sum/valid_peers - btc['pct']) if valid_peers > 0 else 0
    sent_adj = (fng_val - 50) * 0.02
    
    # 使用计算好的模型参数
    def predict_target(key):
        return (model[key]['intercept'] + 
               (model[key]['beta_open'] * open_change) + 
               (model[key]['beta_btc'] * btc['pct']) + 
               (model['beta_sector'] * sector_alpha) + sent_adj)

    pred_h_price = btdr['prev'] * (1 + predict_target('high') / 100)
    pred_l_price = btdr['prev'] * (1 + predict_target('low') / 100)

    st.markdown("### 🎯 日内阻力/支撑 (Intraday)")
    col_h, col_l = st.columns(2)
    
    # 颜色逻辑：突破显示绿色/红色
    h_style = ("#e6fcf5", "#087f5b", "#c3fae8") if btdr['price'] < pred_h_price else ("#0ca678", "#fff", "#0ca678")
    l_style = ("#fff5f5", "#c92a2a", "#ffc9c9") if btdr['price'] > pred_l_price else ("#e03131", "#fff", "#e03131")
    
    with col_h: st.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background:{h_style[0]}; color:{h_style[1]}; border:1px solid {h_style[2]};"><div style="font-size:0.8rem;opacity:0.8;">阻力 (High)</div><div style="font-size:1.5rem;font-weight:bold;">${pred_h_price:.2f}</div></div></div>""", unsafe_allow_html=True)
    with col_l: st.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background:{l_style[0]}; color:{l_style[1]}; border:1px solid {l_style[2]};"><div style="font-size:0.8rem;opacity:0.8;">支撑 (Low)</div><div style="font-size:1.5rem;font-weight:bold;">${pred_l_price:.2f}</div></div></div>""", unsafe_allow_html=True)

    # --- 因子面板 ---
    st.markdown("---")
    st.markdown("### 🌍 宏观 & 微观因子")
    m1, m2, m3, m4 = st.columns(4)
    
    drift_est = (btc['pct']/100 * factors['beta_btc'] * 0.4) + (qqq['pct']/100 * factors['beta_qqq'] * 0.4)
    if abs(dist_vwap) > 10: drift_est -= (dist_vwap/100) * 0.05
    
    with m1: st.markdown(factor_html("QQQ (纳指)", f"{qqq['pct']:+.2f}%", "Mkt", qqq['pct'], "科技股风向标"), unsafe_allow_html=True)
    with m2: st.markdown(factor_html("VIX (恐慌)", f"{vix['price']:.1f}", "Risk", -vix['pct'], "恐慌指数", reverse_color=True), unsafe_allow_html=True)
    with m3: st.markdown(factor_html("ADX (强度)", f"{factors['adx']:.1f}", factors['regime'], 1 if factors['adx']>25 else -1, "趋势强度 >25=Trend"), unsafe_allow_html=True)
    with m4: st.markdown(factor_html("Exp Drift", f"{drift_est*100:+.2f}%", "Day", drift_est, "今日预期漂移率"), unsafe_allow_html=True)

    # --- 宗师级推演 (Monte Carlo) ---
    st.markdown("### ☁️ 概率推演 (Monte Carlo)")
    
    # 动态调整波动率参数
    vol = factors['vol_base']
    drift = drift_est
    if vix['price'] > 25: drift -= 0.005; vol *= 1.3
    if factors['regime'] == "Chop": drift *= 0.5; vol *= 0.8
    
    # 固定随机种子以防止图表在自动刷新时疯狂抖动
    np.random.seed(int(time.time() / 60)) # 每分钟变一次种子，而不是每5秒
    
    days_ahead = 5
    simulations = 500
    dt = 1
    paths = np.zeros((simulations, days_ahead + 1))
    paths[:, 0] = btdr['price']
    
    for t in range(1, days_ahead + 1):
        shock = np.random.normal(0, 1, simulations)
        paths[:, t] = paths[:, t-1] * np.exp((drift - 0.5 * vol**2) * dt + vol * shock)
        
    p90 = np.percentile(paths, 90, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p10 = np.percentile(paths, 10, axis=0)
    
    df_chart = pd.DataFrame({
        'Day': list(range(days_ahead + 1)),
        'P90': np.round(p90, 2), 'P50': np.round(p50, 2), 'P10': np.round(p10, 2)
    })
    
    base = alt.Chart(df_chart).encode(x=alt.X('Day:O', title='未来 T+N 日'))
    area = base.mark_area(opacity=0.2, color='#4dabf7').encode(y=alt.Y('P10', scale=alt.Scale(zero=False), title='Price Forecast'), y2='P90')
    line_50 = base.mark_line(color='#228be6', size=3).encode(y='P50')
    line_90 = base.mark_line(color='#0ca678', strokeDash=[5,5]).encode(y='P90')
    line_10 = base.mark_line(color='#d6336c', strokeDash=[5,5]).encode(y='P10')
    
    # 交互点
    nearest = alt.selection_point(nearest=True, on='mouseover', fields=['Day'], empty=False)
    selectors = base.mark_rule(opacity=0).encode(x='Day:O').add_params(nearest)
    points = base.mark_circle(size=60, color="black").encode(
        y='P50', opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        tooltip=['Day', 'P90', 'P50', 'P10']
    )
    
    st.altair_chart((area + line_90 + line_50 + line_10 + selectors + points).properties(height=300).interactive(), use_container_width=True)

# --- Main ---
st.markdown("### ⚡ BTDR Pilot v9.5")
show_live_dashboard()
