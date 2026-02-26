import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import altair as alt
from datetime import datetime
import pytz
import os
import shutil

# ==========================================
# 1. 页面配置与缓存清理 (必须在最前面)
# ==========================================
st.set_page_config(page_title="BTDR Pilot v9.1+ (V7.4 Core)", layout="centered")

# 启动时清理一次缓存，防止旧数据残留导致的 nan 问题
if 'init_v91_plus' not in st.session_state:
    try:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "yfinance")
        if os.path.exists(cache_dir): shutil.rmtree(cache_dir)
    except: pass
    st.session_state.clear()
    st.session_state['init_v91_plus'] = True

# ==========================================
# 2. V9.1 原版 CSS 样式 (防抖 + 悬停提示)
# ==========================================
st.markdown("""
    <style>
    html { overflow-y: scroll; }
    .stApp > header { display: none; }
    .stApp { margin-top: -30px; background-color: #ffffff; }
    div[data-testid="stStatusWidget"] { visibility: hidden; }
    
    h1, h2, h3, div, p, span { 
        color: #212529 !important; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important; 
    }
    
    /* 锁定图表高度，防止刷新时页面上下抖动 */
    div[data-testid="stAltairChart"] {
        height: 320px !important;
        min-height: 320px !important;
        overflow: hidden !important;
        border: 1px solid #f8f9fa;
    }
    canvas { transition: opacity 0.2s ease-in-out; }
    
    /* 统一卡片样式 */
    .metric-card {
        background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02); height: 95px; padding: 0 16px;
        display: flex; flex-direction: column; justify-content: center;
    }
    .metric-label { font-size: 0.75rem; color: #888; margin-bottom: 2px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #212529; line-height: 1.2; }
    .metric-delta { font-size: 0.9rem; font-weight: 600; margin-top: 2px; }
    
    /* --- 核心修改：带悬停提示的因子卡片 --- */
    .factor-box {
        background: #fff; border: 1px solid #eee; border-radius: 8px; padding: 6px; text-align: center;
        height: 75px; display: flex; flex-direction: column; justify-content: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        position: relative; /* 为绝对定位的 tooltip 做参照 */
        cursor: help; /* 鼠标变成问号，提示可悬停 */
        transition: transform 0.1s;
    }
    
    .factor-box:hover {
        border-color: #ced4da;
        transform: translateY(-1px);
    }

    .factor-title { font-size: 0.65rem; color: #999; text-transform: uppercase; letter-spacing: 0.5px; }
    .factor-val { font-size: 1.1rem; font-weight: bold; color: #495057; margin: 2px 0; }
    .factor-sub { font-size: 0.7rem; font-weight: 600; }
    
    /* Tooltip 文本样式 */
    .tooltip-text {
        visibility: hidden;
        width: 180px;
        background-color: rgba(33, 37, 41, 0.95);
        color: #fff !important;
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 999;
        bottom: 110%; /* 显示在卡片上方 */
        left: 50%;
        margin-left: -90px; /* 居中 */
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.7rem !important;
        font-weight: normal;
        line-height: 1.4;
        pointer-events: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .tooltip-text::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: rgba(33, 37, 41, 0.95) transparent transparent transparent;
    }

    /* 悬停触发 */
    .factor-box:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }
    
    /* 颜色定义 */
    .color-up { color: #0ca678; } .color-down { color: #d6336c; } .color-neutral { color: #adb5bd; }
    .status-dot { height: 6px; width: 6px; border-radius: 50%; display: inline-block; margin-left: 6px; }
    .dot-reg { background-color: #0ca678; } .dot-closed { background-color: #adb5bd; }
    
    .pred-container-wrapper { height: 110px; width: 100%; display: block; margin-top: 5px; }
    .pred-box { padding: 0 10px; border-radius: 12px; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center; }
    
    .time-bar { font-size: 0.75rem; color: #999; text-align: center; margin-bottom: 20px; padding: 6px; background: #fafafa; border-radius: 6px; }
    .badge-trend { background:#fd7e14; color:white; padding:1px 4px; border-radius:3px; font-size:0.6rem; }
    .badge-chop { background:#868e96; color:white; padding:1px 4px; border-radius:3px; font-size:0.6rem; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3. 辅助组件函数
# ==========================================
def safe_float(val, default=0.0):
    """强力数据清洗，彻底屏蔽 NaN 和 Inf 报错"""
    try:
        if val is None: return default
        if hasattr(val, "iloc"):
            if val.empty: return default
            val = val.iloc[-1]
        f = float(val)
        if np.isnan(f) or np.isinf(f): return default
        return f
    except: return default

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

# ==========================================
# 4. 数据获取核心引擎 (V7.4 全时段抓取逻辑)
# ==========================================
def fetch_all_market_data():
    # 默认兜底值
    quotes = {}
    model = {"high": {"intercept": 4.29, "beta_open": 0.67, "beta_btc": 0.52}, "low": {"intercept": -3.22, "beta_open": 0.88, "beta_btc": 0.42}, "beta_sector": 0.25}
    factors = {"vwap": 10.0, "adx": 20.0, "regime": "Neutral", "beta_btc": 1.5, "beta_qqq": 1.2, "rsi": 50, "vol_base": 0.05}
    fng = 50
    
    try:
        tickers = "BTDR BTC-USD QQQ ^VIX MARA RIOT CORZ CLSK IREN"
        
        # 1. 获取日线 (用于计算各种因子和昨收)
        hist = yf.download(tickers, period="6mo", interval="1d", group_by='ticker', threads=False, progress=False)
        
        # 2. 获取分钟线 (极其关键：加上 prepost=True，实现盘前、盘后、夜盘全时段覆盖)
        live = yf.download(tickers, period="1d", interval="1m", prepost=True, group_by='ticker', threads=False, progress=False)
        
        today_ny = datetime.now(pytz.timezone('America/New_York')).date()
        syms = tickers.split()
        
        # --- A. 行情数据处理 (V7.4 逻辑) ---
        for s in syms:
            try:
                df_d = hist[s] if s in hist else pd.DataFrame()
                df_m = live[s] if s in live else pd.DataFrame()
                
                # 价格获取：优先分钟线，兜底日线
                price = 0.0
                state = "ERR"
                if not df_m.empty:
                    val = safe_float(df_m['Close'])
                    if val > 0: 
                        price = val
                        state = "REG"
                
                if price == 0 and not df_d.empty:
                    price = safe_float(df_d['Close'])
                    state = "CLOSED"
                
                # 昨收 & 开盘 判断逻辑
                prev = 0.0
                open_p = 0.0
                if not df_d.empty:
                    last_dt = df_d.index[-1].date()
                    if last_dt == today_ny and len(df_d) > 1:
                        # 当前是交易日
                        prev = safe_float(df_d['Close'].iloc[-2]) 
                        open_p = safe_float(df_d['Open'].iloc[-1])
                    else:
                        # 盘前或未正式开盘
                        prev = safe_float(df_d['Close'].iloc[-1])
                        # 开盘价暂用最新价填充，防止没数据
                        open_p = price if price > 0 else prev
                
                # 最终清洗，防止任何情况出现 0
                if price <= 0.01: price = 10.0 
                if prev <= 0.01: prev = price
                if open_p <= 0.01: open_p = price
                
                pct = ((price - prev) / prev) * 100
                quotes[s] = {"price": price, "pct": pct, "prev": prev, "open": open_p, "tag": state}
            except:
                quotes[s] = {"price": 10.0, "pct": 0.0, "prev": 10.0, "open": 10.0, "tag": "ERR"}

        # --- B. 计算因子 (拆分写法，防止 SyntaxError) ---
        btdr = hist['BTDR'].dropna()
        btc = hist['BTC-USD'].dropna()
        qqq = hist['QQQ'].dropna()
        
        idx = btdr.index.intersection(btc.index).intersection(qqq.index)
        
        if len(idx) > 30:
            btdr = btdr.loc[idx]
            btc = btc.loc[idx]
            qqq = qqq.loc[idx]
            
            # Beta
            rb = btdr['Close'].pct_change()
            rc = btc['Close'].pct_change()
            rq = qqq['Close'].pct_change()
            
            beta_btc = safe_float((rb.rolling(60).cov(rc)/rc.rolling(60).var()).iloc[-1], 1.5)
            beta_qqq = safe_float((rb.rolling(60).cov(rq)/rq.rolling(60).var()).iloc[-1], 1.2)
            
            # VWAP
            btdr['TP'] = (btdr['High']+btdr['Low']+btdr['Close'])/3
            btdr['PV'] = btdr['TP']*btdr['Volume']
            vwap = safe_float(btdr['PV'].tail(30).sum() / btdr['Volume'].tail(30).sum(), quotes['BTDR']['price'])
            
            # RSI & Vol
            delta = btdr['Close'].diff()
            gain = (delta.where(delta>0, 0)).rolling(14).mean()
            loss = (-delta.where(delta<0, 0)).rolling(14).mean()
            rsi = safe_float(100 - (100/(1 + gain/loss)).iloc[-1], 50.0)
            vol_base = safe_float(rb.ewm(span=20).std().iloc[-1], 0.05)
            
            # ADX
            high = btdr['High']
            low = btdr['Low']
            close = btdr['Close']
            tr = np.maximum(high-low, np.abs(high-close.shift(1)))
            atr = tr.rolling(14).mean()
            p_dm = (high-high.shift(1)).clip(lower=0)
            m_dm = (low.shift(1)-low).clip(lower=0)
            p_di = 100 * p_dm.rolling(14).mean() / atr
            m_di = 100 * m_dm.rolling(14).mean() / atr
            dx = 100 * np.abs(p_di-m_di)/(p_di+m_di)
            adx = safe_float(dx.rolling(14).mean().iloc[-1], 20.0)
            
            factors = {"beta_btc": beta_btc, "beta_qqq": beta_qqq, "vwap": vwap, "adx": adx, "regime": "Trend" if adx>25 else "Chop", "rsi": rsi, "vol_base": vol_base}
            
            # 回归模型计算 (多行写法，绝对安全)
            df_r = btdr.tail(30).copy()
            df_r['Prev'] = df_r['Close'].shift(1)
            df_r.dropna(inplace=True)
            
            x = ((df_r['Open']-df_r['Prev'])/df_r['Prev']*100).values
            yh = ((df_r['High']-df_r['Prev'])/df_r['Prev']*100).values
            yl = ((df_r['Low']-df_r['Prev'])/df_r['Prev']*100).values
            
            ch = np.cov(x, yh)
            bh = safe_float(ch[0,1]/ch[0,0], 0.67)
            
            cl = np.cov(x, yl)
            bl = safe_float(cl[0,1]/cl[0,0], 0.88)
            
            model = {
                "high": {"intercept": 0.7*4.29 + 0.3*(np.mean(yh)-bh*np.mean(x)), "beta_open": 0.7*0.67+0.3*bh, "beta_btc": 0.52},
                "low": {"intercept": 0.7*-3.22 + 0.3*(np.mean(yl)-bl*np.mean(x)), "beta_open": 0.7*0.88+0.3*bl, "beta_btc": 0.42},
                "beta_sector": 0.25
            }
            
        try: fng = int(requests.get("https://api.alternative.me/fng/", timeout=1).json()['data'][0]['value'])
        except: fng = 50
        
        return quotes, fng, model, factors
    except:
        return quotes, 50, model, factors

# ==========================================
# 5. UI 渲染引擎 (局部防抖刷新)
# ==========================================
@st.fragment(run_every=5) 
def show_live_dashboard():
    # 1. 抓取数据
    quotes, fng_val, model, factors = fetch_all_market_data()
    
    if not quotes or 'BTDR' not in quotes:
        st.warning("📡 正在连接交易所数据流...")
        return

    # 2. 数据解包
    btc_chg = quotes['BTC-USD']['pct']
    qqq_chg = quotes.get('QQQ', {'pct': 0})['pct']
    vix_val = quotes.get('^VIX', {'price': 20})['price']
    vix_chg = quotes.get('^VIX', {'pct': 0})['pct']
    btdr = quotes['BTDR']
    
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny).strftime('%H:%M:%S')
    
    regime_tag = "Trend" if factors['regime'] == "Trend" else "Chop"
    badge_class = "badge-trend" if regime_tag == "Trend" else "badge-chop"
    st.markdown(f"<div class='time-bar'>美东 {now_ny} &nbsp;|&nbsp; 状态: <span class='{badge_class}'>{regime_tag}</span> &nbsp;|&nbsp; 引擎: v9.1+ (V7.4 Core)</div>", unsafe_allow_html=True)
    
    # 3. 宏观顶栏
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
    
    # 4. BTDR 核心三栏 (实时价 | 开盘价 | VWAP)
    c_live, c_open, c_vwap = st.columns(3)
    
    state_map = {"PRE": "dot-reg", "REG": "dot-reg", "POST": "dot-reg", "CLOSED": "dot-closed"}
    dot_class = state_map.get(btdr.get('tag', 'CLOSED'), 'dot-closed')
    status_tag = f"<span class='status-dot {dot_class}'></span>"
    
    # 实时价 (包含盘前盘后)
    with c_live: st.markdown(card_html("BTDR 实时", f"${btdr['price']:.2f}", f"{btdr['pct']:+.2f}%", btdr['pct'], status_tag), unsafe_allow_html=True)
    
    # 开盘价 (相对于昨收的涨幅)
    open_pct = ((btdr['open'] - btdr['prev']) / btdr['prev']) * 100 if btdr['prev'] > 0 else 0
    with c_open: st.markdown(card_html("计算用开盘", f"${btdr['open']:.2f}", f"{open_pct:+.2f}%", open_pct), unsafe_allow_html=True)
    
    # VWAP
    dist_vwap = ((btdr['price'] - factors['vwap']) / factors['vwap']) * 100 if factors['vwap'] > 0 else 0
    with c_vwap: st.markdown(card_html("机构成本 (VWAP)", f"${factors['vwap']:.2f}", f"{dist_vwap:+.1f}% Prem.", dist_vwap), unsafe_allow_html=True)

    # 5. 日内预测 (基于真实的开盘涨幅进行测算)
    peers_avg = sum(quotes[p]['pct'] for p in peers if p in quotes) / 5
    sector_alpha = peers_avg - btc_chg
    sentiment_adj = (fng_val - 50) * 0.02
    
    # 重点：使用 open_pct 驱动
    pred_high_pct = (model['high']['intercept'] + (model['high']['beta_open'] * open_pct) + (model['high']['beta_btc'] * btc_chg) + (model['beta_sector'] * sector_alpha) + sentiment_adj)
    pred_low_pct = (model['low']['intercept'] + (model['low']['beta_open'] * open_pct) + (model['low']['beta_btc'] * btc_chg) + (model['beta_sector'] * sector_alpha) + sentiment_adj)
    
    pred_high_price = btdr['prev'] * (1 + pred_high_pct / 100)
    pred_low_price = btdr['prev'] * (1 + pred_low_pct / 100)
    
    st.markdown("### 🎯 日内阻力/支撑 (Intraday)")
    col_h, col_l = st.columns(2)
    h_bg = "#e6fcf5" if btdr['price'] < pred_high_price else "#0ca678"; h_txt = "#087f5b" if btdr['price'] < pred_high_price else "#ffffff"
    l_bg = "#fff5f5" if btdr['price'] > pred_low_price else "#e03131"; l_txt = "#c92a2a" if btdr['price'] > pred_low_price else "#ffffff"
    with col_h: st.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background-color: {h_bg}; color: {h_txt}; border: 1px solid #c3fae8;"><div style="font-size: 0.8rem; opacity: 0.8;">日内阻力 (High)</div><div style="font-size: 1.5rem; font-weight: bold;">${pred_high_price:.2f}</div></div></div>""", unsafe_allow_html=True)
    with col_l: st.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background-color: {l_bg}; color: {l_txt}; border: 1px solid #ffc9c9;"><div style="font-size: 0.8rem; opacity: 0.8;">日内支撑 (Low)</div><div style="font-size: 1.5rem; font-weight: bold;">${pred_low_price:.2f}</div></div></div>""", unsafe_allow_html=True)

    # 6. 因子面板 (保留 V9.1 的悬停提示)
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

    # 7. 宗师级推演图表
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
    st.caption(f"Engine: v9.1+ (V7.4 Data Core) | Drift: {drift*100:.2f}% | Vol: {vol*100:.1f}%")

# ==========================================
# 6. 主程序入口 (必须在所有函数定义之后)
# ==========================================
if __name__ == "__main__":
    st.markdown("### ⚡ BTDR 领航员 v9.1+")
    show_live_dashboard()
