import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
import pytz
import shutil
import os
import requests

# ==========================================
# 1. 基础配置 & 缓存清理
# ==========================================
st.set_page_config(page_title="BTDR Pilot v10.0", layout="centered")

# 启动时清理一次缓存，防止旧的错误数据残留
if 'init_v10' not in st.session_state:
    try:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "yfinance")
        if os.path.exists(cache_dir): shutil.rmtree(cache_dir)
    except: pass
    st.session_state.clear()
    st.session_state['init_v10'] = True

# ==========================================
# 2. CSS 样式 (V9.1 风格：防抖 + 悬停)
# ==========================================
st.markdown("""
    <style>
    /* 全局设置 */
    html { overflow-y: scroll; }
    .stApp > header { display: none; }
    .stApp { margin-top: -30px; background-color: #ffffff; }
    
    h1, h2, h3, div, p, span { 
        color: #212529 !important; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important; 
    }
    
    /* 图表容器锁定 (防抖) */
    div[data-testid="stAltairChart"] {
        height: 300px !important; min-height: 300px !important; overflow: hidden !important; border: 1px solid #f8f9fa;
    }
    canvas { transition: none !important; animation: none !important; }
    
    /* 指标卡片 */
    .metric-card {
        background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 12px;
        height: 95px; padding: 0 16px; display: flex; flex-direction: column; justify-content: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    .metric-label { font-size: 0.75rem; color: #888; margin-bottom: 2px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #212529; line-height: 1.1; }
    .metric-delta { font-size: 0.85rem; font-weight: 600; margin-top: 2px; }
    
    /* 因子卡片 (带 Tooltip) */
    .factor-box {
        background: #fff; border: 1px solid #eee; border-radius: 8px; padding: 6px; text-align: center;
        height: 75px; display: flex; flex-direction: column; justify-content: center;
        position: relative; cursor: help;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    .factor-box:hover { border-color: #adb5bd; transform: translateY(-1px); }
    .factor-title { font-size: 0.65rem; color: #999; text-transform: uppercase; }
    .factor-val { font-size: 1.1rem; font-weight: bold; color: #495057; }
    .factor-sub { font-size: 0.7rem; font-weight: 600; }
    
    /* Tooltip 悬浮窗 */
    .tooltip-text {
        visibility: hidden; width: 150px; background-color: rgba(0,0,0,0.9); color: #fff !important;
        text-align: center; border-radius: 4px; padding: 6px; position: absolute; z-index: 1000;
        bottom: 115%; left: 50%; margin-left: -75px; opacity: 0; transition: opacity 0.2s;
        font-size: 0.7rem !important; pointer-events: none; font-weight: normal; line-height: 1.4;
    }
    .tooltip-text::after {
        content: ""; position: absolute; top: 100%; left: 50%; margin-left: -5px;
        border-width: 5px; border-style: solid; border-color: rgba(0,0,0,0.9) transparent transparent transparent;
    }
    .factor-box:hover .tooltip-text { visibility: visible; opacity: 1; }
    
    .color-up { color: #0ca678; } .color-down { color: #d6336c; } .color-neutral { color: #adb5bd; }
    .status-dot { height: 6px; width: 6px; border-radius: 50%; display: inline-block; margin-left: 6px; }
    .dot-reg { background-color: #0ca678; } .dot-closed { background-color: #adb5bd; }
    
    .pred-container-wrapper { height: 100px; width: 100%; display: block; margin-top: 5px; }
    .pred-box { padding: 0 10px; border-radius: 12px; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center; }
    .time-bar { font-size: 0.75rem; color: #999; text-align: center; margin-bottom: 15px; padding: 4px; background: #fafafa; border-radius: 4px; }
    .badge-trend { background:#fd7e14; color:white; padding:1px 4px; border-radius:3px; font-size:0.6rem; }
    .badge-chop { background:#868e96; color:white; padding:1px 4px; border-radius:3px; font-size:0.6rem; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3. 辅助函数
# ==========================================
def safe_float(val, default=0.0):
    """V7.4 核心：强力数据清洗，拒绝 NaN"""
    try:
        if val is None: return default
        # 处理 Series/DataFrame
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
        c = "color-up" if delta_val >= 0 else "color-down"
        delta_html = f"<div class='metric-delta {c}'>{delta_str}</div>"
    return f"""<div class="metric-card"><div class="metric-label">{label} {extra_tag}</div><div class="metric-value">{value_str}</div>{delta_html}</div>"""

def factor_html(title, val, delta_str, delta_val, tooltip, rev=False):
    c = "color-up" if delta_val >= 0 else "color-down"
    if rev: c = "color-down" if delta_val >= 0 else "color-up"
    return f"""<div class="factor-box"><div class="tooltip-text">{tooltip}</div><div class="factor-title">{title}</div><div class="factor-val">{val}</div><div class="factor-sub {c}">{delta_str}</div></div>"""

# ==========================================
# 4. 数据获取引擎 (V7.4 核心逻辑移植)
# ==========================================
def fetch_data_v74_core():
    # 默认兜底
    quotes = {}
    model = {"high": {"intercept": 4.29, "beta_open": 0.67, "beta_btc": 0.52}, "low": {"intercept": -3.22, "beta_open": 0.88, "beta_btc": 0.42}, "beta_sector": 0.25}
    factors = {"vwap": 10.0, "adx": 20.0, "regime": "Neutral", "beta_btc": 1.5, "beta_qqq": 1.2, "rsi": 50, "vol_base": 0.05}
    fng = 50
    
    try:
        tickers = "BTDR BTC-USD QQQ ^VIX MARA RIOT CORZ CLSK IREN"
        
        # 1. 抓取数据 - 启用 prepost=True 以获取盘前盘后
        # 历史数据 (1y) -> 用于因子计算和昨收判断
        hist = yf.download(tickers, period="1y", interval="1d", group_by='ticker', threads=False, progress=False)
        # 实时数据 (1d, 1m) -> 用于实时价格，必须开启 prepost
        live = yf.download(tickers, period="1d", interval="1m", prepost=True, group_by='ticker', threads=False, progress=False)
        
        today_ny = datetime.now(pytz.timezone('America/New_York')).date()
        syms = tickers.split()
        
        # 2. 处理行情 (V7.4 逻辑：双重保险)
        for s in syms:
            try:
                df_d = hist[s] if s in hist else pd.DataFrame()
                df_m = live[s] if s in live else pd.DataFrame()
                
                # --- A. 实时价格 (Price) ---
                price = 0.0
                state = "ERR"
                
                # 优先取分钟线 (含盘前盘后)
                if not df_m.empty:
                    val = safe_float(df_m['Close'])
                    if val > 0: 
                        price = val
                        state = "REG"
                
                # 如果分钟线挂了 (比如刚开盘瞬间)，取日线最新
                if price == 0 and not df_d.empty:
                    price = safe_float(df_d['Close'])
                    state = "CLOSED"
                
                # --- B. 昨收 (Prev) & 开盘 (Open) ---
                prev = 0.0
                open_p = 0.0
                
                if not df_d.empty:
                    last_dt = df_d.index[-1].date()
                    
                    if last_dt == today_ny and len(df_d) > 1:
                        # 情况1: 日线已更新到今天
                        prev = safe_float(df_d['Close'].iloc[-2]) # 昨收是倒数第二根
                        open_p = safe_float(df_d['Open'].iloc[-1]) # 今开是最后一根Open
                    else:
                        # 情况2: 日线还停留在昨天 (盘前/未开盘)
                        prev = safe_float(df_d['Close'].iloc[-1]) # 昨收是最后一根
                        # 还没正式开盘，Open 暂时用 Price 或 Prev 代替，避免 nan
                        open_p = price if price > 0 else prev
                
                # --- C. 最终兜底 (防止除零) ---
                if price == 0: price = 10.0
                if prev == 0: prev = price
                if open_p == 0: open_p = price
                
                pct = ((price - prev) / prev) * 100
                quotes[s] = {"price": price, "pct": pct, "prev": prev, "open": open_p, "tag": state}
            except:
                quotes[s] = {"price": 10.0, "pct": 0.0, "prev": 10.0, "open": 10.0, "tag": "ERR"}

        # 3. 计算因子 (V9 逻辑)
        btdr = hist['BTDR'].dropna(); btc = hist['BTC-USD'].dropna(); qqq = hist['QQQ'].dropna()
        idx = btdr.index.intersection(btc.index).intersection(qqq.index)
        
        if len(idx) > 20:
            btdr = btdr.loc[idx]; btc = btc.loc[idx]; qqq = qqq.loc[idx]
            
            rb = btdr['Close'].pct_change()
            rc = btc['Close'].pct_change()
            rq = qqq['Close'].pct_change()
            
            beta_btc = safe_float((rb.rolling(30).cov(rc)/rc.rolling(30).var()).iloc[-1], 1.5)
            beta_qqq = safe_float((rb.rolling(30).cov(rq)/rq.rolling(30).var()).iloc[-1], 1.2)
            
            # VWAP
            btdr['TP'] = (btdr['High']+btdr['Low']+btdr['Close'])/3
            btdr['PV'] = btdr['TP']*btdr['Volume']
            vwap = safe_float(btdr['PV'].tail(20).sum() / btdr['Volume'].tail(20).sum(), quotes['BTDR']['price'])
            
            # RSI & Vol
            delta = btdr['Close'].diff()
            gain = (delta.where(delta>0, 0)).rolling(14).mean()
            loss = (-delta.where(delta<0, 0)).rolling(14).mean()
            rsi = safe_float(100 - (100/(1 + gain/loss)).iloc[-1], 50.0)
            vol_base = safe_float(rb.ewm(span=20).std().iloc[-1], 0.05)
            
            # ADX
            high = btdr['High']; low = btdr['Low']; close = btdr['Close']
            tr = np.maximum(high-low, np.abs(high-close.shift(1)))
            atr = tr.rolling(14).mean()
            p_dm = (high-high.shift(1)).clip(lower=0)
            m_dm = (low.shift(1)-low).clip(lower=0)
            p_di = 100 * p_dm.rolling(14).mean() / atr
            m_di = 100 * m_dm.rolling(14).mean() / atr
            dx = 100 * np.abs(p_di-m_di)/(p_di+m_di)
            adx = safe_float(dx.rolling(14).mean().iloc[-1], 20.0)
            
            factors = {"beta_btc": beta_btc, "beta_qqq": beta_qqq, "vwap": vwap, "adx": adx, "regime": "Trend" if adx>25 else "Chop", "rsi": rsi, "vol_base": vol_base}
            
            # Regression
            df_r = btdr.tail(30).copy()
            df_r['Prev'] = df_r['Close'].shift(1); df_r.dropna(inplace=True)
            x = ((df_r['Open']-df_r['Prev'])/df_r['Prev']*100).values
            yh = ((df_r['High']-df_r['Prev'])/df_r['Prev']*100).values
            yl = ((df_r['Low']-df_r['Prev'])/df_r['Prev']*100).values
            ch = np.cov(x, yh); bh = safe_float(ch[0,1]/ch[0,0], 0.7)
            cl = np.cov(x, yl); bl = safe_float(cl[0,1]/cl[0,0], 0.9)
            
            model = {
                "high": {"intercept": 0.7*4.29 + 0.3*(np.mean(yh)-bh*np.mean(x)), "beta_open": 0.7*0.67+0.3*bh, "beta_btc": 0.52},
                "low": {"intercept": 0.7*-3.22 + 0.3*(np.mean(yl)-bl*np.mean(x)), "beta_open": 0.7*0.88+0.3*bl, "beta_btc": 0.42},
                "beta_sector": 0.25
            }
        else:
            model = default_model
            factors = default_factors
            
        try: fng = int(requests.get("https://api.alternative.me/fng/", timeout=1).json()['data'][0]['value'])
        except: fng = 50
        
        return quotes, fng, model, factors
    except:
        return quotes, 50, default_model, default_factors

# ==========================================
# 5. UI 主程序 (Fragment 局部刷新)
# ==========================================
@st.fragment(run_every=5)
def dashboard():
    # 1. 抓取数据
    quotes, fng_val, model, factors = fetch_data_v74_core()
    
    if not quotes or 'BTDR' not in quotes:
        st.info("⌛ 正在建立数据连接 (V7.4 Core)...")
        return

    # 2. 变量解包
    btdr = quotes['BTDR']
    btc = quotes['BTC-USD']
    qqq = quotes.get('QQQ', {'pct': 0, 'price': 0})
    vix = quotes.get('^VIX', {'pct': 0, 'price': 20})
    
    tz_ny = pytz.timezone('America/New_York')
    now_str = datetime.now(tz_ny).strftime('%H:%M:%S')
    badge_c = "#fd7e14" if factors['regime']=="Trend" else "#868e96"
    
    # 3. 状态栏
    st.markdown(f"<div class='time-bar'>美东 {now_str} &nbsp;|&nbsp; 状态: <span style='background:{badge_c};color:white;padding:1px 3px;border-radius:3px;font-size:0.6rem'>{factors['regime']}</span> &nbsp;|&nbsp; 引擎: v10.0 (V7.4-Core)</div>", unsafe_allow_html=True)
    
    # 4. 第一行
    c1, c2 = st.columns(2)
    c1.markdown(card_html("BTC (全时段)", f"{btc['pct']:+.2f}%", f"{btc['pct']:+.2f}%", btc['pct']), unsafe_allow_html=True)
    c2.markdown(card_html("恐慌指数", f"{fng_val}", None, 0), unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    st.caption("⚒️ 矿股板块")
    cols = st.columns(5)
    peers = ["MARA", "RIOT", "CORZ", "CLSK", "IREN"]
    for i, p in enumerate(peers):
        if p in quotes:
            v = quotes[p]['pct']
            cols[i].markdown(card_html(p, f"{v:+.1f}%", f"{v:+.1f}%", v), unsafe_allow_html=True)
            
    st.markdown("---")
    
    # 5. BTDR 核心 (三栏布局: 实时 | 开盘 | VWAP)
    c_live, c_open, c_vwap = st.columns(3)
    
    # 实时 (含盘前盘后)
    state_dot = f"<span class='status-dot dot-reg'></span>"
    c_live.markdown(card_html("BTDR 实时", f"${btdr['price']:.2f}", f"{btdr['pct']:+.2f}%", btdr['pct'], state_dot), unsafe_allow_html=True)
    
    # 开盘 (计算涨幅: Open vs Prev)
    # 因为有了开盘价，日内阻力支撑就有数了
    op_pct = ((btdr['open'] - btdr['prev']) / btdr['prev']) * 100 if btdr['prev'] > 0 else 0
    c_open.markdown(card_html("计算用开盘", f"${btdr['open']:.2f}", f"{op_pct:+.2f}%", op_pct), unsafe_allow_html=True)
    
    # VWAP
    vwap = factors['vwap']
    v_diff = ((btdr['price'] - vwap)/vwap)*100 if vwap>0 else 0
    c_vwap.markdown(card_html("机构成本 (VWAP)", f"${vwap:.2f}", f"{v_diff:+.1f}% Prem.", v_diff), unsafe_allow_html=True)
    
    # 6. 日内预测 (因为 btdr['open'] 有数，所以这里不会 nan)
    peers_avg = sum(quotes[p]['pct'] for p in peers if p in quotes)/5
    sec_alpha = peers_avg - btc['pct']
    sent_adj = (fng_val - 50) * 0.02
    
    pred_h_pct = model['high']['intercept'] + model['high']['beta_open']*op_pct + model['high']['beta_btc']*btc['pct'] + 0.25*sec_alpha + sent_adj
    pred_l_pct = model['low']['intercept'] + model['low']['beta_open']*op_pct + model['low']['beta_btc']*btc['pct'] + 0.25*sec_alpha + sent_adj
    
    p_h = btdr['prev'] * (1 + pred_h_pct/100)
    p_l = btdr['prev'] * (1 + pred_l_pct/100)
    
    st.markdown("### 🎯 日内阻力/支撑")
    ch, cl = st.columns(2)
    h_bg = "#e6fcf5" if btdr['price'] < p_h else "#0ca678"
    h_tx = "#087f5b" if btdr['price'] < p_h else "#fff"
    l_bg = "#fff5f5" if btdr['price'] > p_l else "#e03131"
    l_tx = "#c92a2a" if btdr['price'] > p_l else "#fff"
    
    ch.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background:{h_bg};color:{h_tx};border:1px solid #c3fae8"><div style="font-size:0.8rem;opacity:0.8">日内阻力 (High)</div><div style="font-size:1.5rem;font-weight:bold">${p_h:.2f}</div></div></div>""", unsafe_allow_html=True)
    cl.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background:{l_bg};color:{l_tx};border:1px solid #ffc9c9"><div style="font-size:0.8rem;opacity:0.8">日内支撑 (Low)</div><div style="font-size:1.5rem;font-weight:bold">${p_l:.2f}</div></div></div>""", unsafe_allow_html=True)
    
    # 7. 因子
    st.markdown("---")
    st.markdown("### 🌍 宏观 & 微观")
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(factor_html("QQQ (纳指)", f"{qqq_chg:+.2f}%", "Market", qqq_chg, "美股科技风向标"), unsafe_allow_html=True)
    m2.markdown(factor_html("VIX (恐慌)", f"{vix_val:.1f}", "Risk", -1 if vix_val>20 else 1, "恐慌指数", True), unsafe_allow_html=True)
    m3.markdown(factor_html("ADX (强度)", f"{factors['adx']:.1f}", factors['regime'], 1 if factors['adx']>25 else -1, "趋势强度"), unsafe_allow_html=True)
    
    drift = (btc['pct']/100 * factors['beta_btc'] * 0.5)
    if abs(v_diff) > 10: drift -= (v_diff/100)*0.05
    m4.markdown(factor_html("Exp. Drift", f"{drift*100:+.2f}%", "Day", drift, "当日预期漂移"), unsafe_allow_html=True)
    
    # 8. 图表
    st.markdown("### ☁️ 宗师级推演 (P90-P50-P10)")
    
    vol = factors['vol_base']
    if vix_val > 25: drift -= 0.005; vol *= 1.3
    if factors['regime'] == "Chop": drift *= 0.5; vol *= 0.8
    
    sims = 300; days = 5
    curr = btdr['price']
    paths = []
    
    for i in range(sims):
        path = [curr]
        p = curr
        for d in range(days):
            shock = np.random.normal(0, 1)
            change = (drift - 0.5 * vol**2) + vol * shock
            p = p * np.exp(change)
            path.append(p)
        paths.append(path)
        
    paths = np.array(paths)
    p90 = np.percentile(paths, 90, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p10 = np.percentile(paths, 10, axis=0)
    
    cdata = []
    for d in range(days + 1):
        cdata.append({"Day": d, "P90": round(p90[d], 2), "P50": round(p50[d], 2), "P10": round(p10[d], 2)})
    df_c = pd.DataFrame(cdata)
    
    base = alt.Chart(df_c).encode(x=alt.X('Day:O', title='T+ Days'))
    area = base.mark_area(opacity=0.2, color='#4dabf7').encode(y=alt.Y('P10', scale=alt.Scale(zero=False)), y2='P90')
    l90 = base.mark_line(color='#0ca678', strokeDash=[5,5]).encode(y='P90')
    l50 = base.mark_line(color='#228be6', size=3).encode(y='P50')
    l10 = base.mark_line(color='#d6336c', strokeDash=[5,5]).encode(y='P10')
    
    near = alt.selection_point(nearest=True, on='mouseover', fields=['Day'], empty=False)
    sel = base.mark_rule(opacity=0).encode(x='Day:O').add_params(near)
    pts = base.mark_circle(size=60, color='black').encode(
        y='P50', opacity=alt.condition(near, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip('Day'), alt.Tooltip('P90'), alt.Tooltip('P50'), alt.Tooltip('P10')]
    )
    
    st.altair_chart((area+l90+l50+l10+sel+pts).properties(height=300).interactive(), use_container_width=True)
    st.caption(f"Engine: v10.0 Final | Drift: {drift*100:.2f}% | Vol: {vol*100:.1f}%")

# ==========================================
# 6. 执行入口
# ==========================================
if __name__ == "__main__":
    st.markdown("### ⚡ BTDR 领航员 v10.0")
    dashboard()
