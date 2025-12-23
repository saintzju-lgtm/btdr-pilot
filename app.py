import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
import pytz
import shutil
import os

# --- 0. 启动前清理缓存 (防止旧的错误数据残留) ---
try:
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "yfinance")
    if os.path.exists(cache_dir): shutil.rmtree(cache_dir)
except: pass

# --- 1. 页面配置 ---
st.set_page_config(page_title="BTDR Pilot v9.4", layout="centered")

# CSS: 视觉锁定 + 悬停提示
st.markdown("""
    <style>
    /* 全局设置 */
    html { overflow-y: scroll; }
    .stApp > header { display: none; }
    .stApp { margin-top: -30px; background-color: #ffffff; }
    div[data-testid="stStatusWidget"] { visibility: hidden; }
    
    h1, h2, h3, div, p, span { 
        color: #212529 !important; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important; 
    }
    
    /* 图表容器强力锁定 (防抖) */
    div[data-testid="stAltairChart"] {
        height: 320px !important; min-height: 320px !important; overflow: hidden !important; border: 1px solid #f8f9fa;
    }
    canvas { transition: none !important; animation: none !important; }
    
    /* 指标卡片 */
    .metric-card {
        background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02); height: 95px; padding: 0 16px;
        display: flex; flex-direction: column; justify-content: center;
    }
    .metric-label { font-size: 0.75rem; color: #888; margin-bottom: 2px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #212529; line-height: 1.2; }
    .metric-delta { font-size: 0.9rem; font-weight: 600; margin-top: 2px; }
    
    /* 因子卡片 (带悬停) */
    .factor-box {
        background: #fff; border: 1px solid #eee; border-radius: 8px; padding: 6px; text-align: center;
        height: 75px; display: flex; flex-direction: column; justify-content: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02); position: relative; cursor: help;
    }
    .factor-box:hover { transform: translateY(-1px); border-color: #ced4da; }
    .factor-title { font-size: 0.65rem; color: #999; text-transform: uppercase; letter-spacing: 0.5px; }
    .factor-val { font-size: 1.1rem; font-weight: bold; color: #495057; margin: 2px 0; }
    .factor-sub { font-size: 0.7rem; font-weight: 600; }
    
    /* Tooltip 悬浮窗 */
    .tooltip-text {
        visibility: hidden; width: 160px; background-color: rgba(0,0,0,0.9); color: #fff !important;
        text-align: center; border-radius: 6px; padding: 8px; position: absolute; z-index: 1000;
        bottom: 115%; left: 50%; margin-left: -80px; opacity: 0; transition: opacity 0.2s;
        font-size: 0.7rem !important; pointer-events: none; font-weight: normal; line-height: 1.4;
    }
    .factor-box:hover .tooltip-text { visibility: visible; opacity: 1; }
    
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
    return f"""<div class="factor-box"><div class="tooltip-text">{tooltip_text}</div><div class="factor-title">{title}</div><div class="factor-val">{val}</div><div class="factor-sub {color_class}">{delta_str}</div></div>"""

# --- 3. 核心数据获取 (Robust Fetch) ---
@st.cache_data(ttl=300) # 5分钟缓存，避免频繁请求被封，通过fragment前端刷新感官
def fetch_robust_data():
    # 默认兜底数据 (防止 nan)
    default_model = {"high": {"intercept": 4.29, "beta_open": 0.67, "beta_btc": 0.52}, "low": {"intercept": -3.22, "beta_open": 0.88, "beta_btc": 0.42}, "beta_sector": 0.25}
    default_factors = {"vwap": 10.0, "adx": 20.0, "regime": "Neutral", "beta_btc": 1.5, "beta_qqq": 1.2, "rsi": 50, "vol_base": 0.05}
    
    try:
        # 1. 抓取数据 (1年日线用于因子，1天分钟线用于实时)
        # 增加 'threads=False' 提高稳定性
        tickers = "BTDR BTC-USD QQQ ^VIX MARA RIOT CORZ CLSK IREN"
        data_day = yf.download(tickers, period="1y", interval="1d", group_by='ticker', threads=False, progress=False)
        data_min = yf.download(tickers, period="1d", interval="1m", prepost=True, group_by='ticker', threads=False, progress=False)
        
        quotes = {}
        symbols = tickers.split()
        
        # 2. 处理行情 (Quotes)
        for sym in symbols:
            try:
                # 尝试获取分钟线
                df_m = data_min[sym] if sym in data_min else pd.DataFrame()
                # 尝试获取日线
                df_d = data_day[sym] if sym in data_day else pd.DataFrame()
                
                # --- 价格熔断机制 ---
                price = 0
                state = "ERR"
                
                # 优先：分钟线最新价
                if not df_m.empty and not pd.isna(df_m['Close'].iloc[-1]):
                    price = df_m['Close'].iloc[-1]
                    state = "REG"
                # 兜底：日线最新价 (防止盘前/盘后空数据)
                elif not df_d.empty and not pd.isna(df_d['Close'].iloc[-1]):
                    price = df_d['Close'].iloc[-1]
                    state = "CLOSED"
                
                # 昨收 & 开盘
                prev = 0; open_p = 0
                if not df_d.empty:
                    # 昨收取倒数第二根 (如果是今天) 或 最后一根 (如果是昨天)
                    last_dt = df_d.index[-1].date()
                    now_dt = datetime.now(pytz.timezone('America/New_York')).date()
                    
                    if last_dt == now_dt and len(df_d) > 1:
                        prev = df_d['Close'].iloc[-2]
                        open_p = df_d['Open'].iloc[-1]
                    else:
                        prev = df_d['Close'].iloc[-1]
                        open_p = price # 还没开盘，暂用当前价
                
                # 再次兜底
                if prev == 0: prev = price if price > 0 else 10.0
                if open_p == 0: open_p = price if price > 0 else 10.0
                if price == 0: price = prev # 实在没数了，就用昨收
                
                pct = ((price - prev) / prev) * 100
                
                quotes[sym] = {"price": price, "pct": pct, "prev": prev, "open": open_p, "tag": state}
            except:
                quotes[sym] = {"price": 10.0, "pct": 0.0, "prev": 10.0, "open": 10.0, "tag": "ERR"}

        # 3. 计算因子 (Factors)
        btdr = data_day['BTDR'].dropna(); btc = data_day['BTC-USD'].dropna(); qqq = data_day['QQQ'].dropna()
        # 对齐
        idx = btdr.index.intersection(btc.index).intersection(qqq.index)
        if len(idx) > 30:
            btdr = btdr.loc[idx]; btc = btc.loc[idx]; qqq = qqq.loc[idx]
            
            # Beta
            rb = btdr['Close'].pct_change(); rc = btc['Close'].pct_change(); rq = qqq['Close'].pct_change()
            beta_btc = (rb.rolling(60).cov(rc)/rc.rolling(60).var()).iloc[-1]
            beta_qqq = (rb.rolling(60).cov(rq)/rq.rolling(60).var()).iloc[-1]
            
            # VWAP & ADX & RSI
            btdr['TP'] = (btdr['High']+btdr['Low']+btdr['Close'])/3
            vwap = (btdr['TP']*btdr['Volume']).tail(30).sum() / btdr['Volume'].tail(30).sum()
            
            delta = btdr['Close'].diff()
            rsi = 100 - (100/(1 + (delta.clip(lower=0)).rolling(14).mean()/(-delta.clip(upper=0)).rolling(14).mean())).iloc[-1]
            
            vol = rb.ewm(span=20).std().iloc[-1]
            
            # ADX 简化计算 (防止nan)
            tr = np.maximum(btdr['High'] - btdr['Low'], np.abs(btdr['High'] - btdr['Close'].shift(1)))
            atr = tr.rolling(14).mean()
            dm_p = (btdr['High'] - btdr['High'].shift(1)).clip(lower=0)
            dm_m = (btdr['Low'].shift(1) - btdr['Low']).clip(lower=0)
            di_p = 100 * dm_p.rolling(14).mean() / atr
            di_m = 100 * dm_m.rolling(14).mean() / atr
            dx = 100 * np.abs(di_p - di_m) / (di_p + di_m)
            adx = dx.rolling(14).mean().iloc[-1]
            
            # 清洗 nan
            if np.isnan(beta_btc): beta_btc = 1.5
            if np.isnan(vwap): vwap = quotes['BTDR']['price']
            if np.isnan(adx): adx = 25
            if np.isnan(vol): vol = 0.05
            
            factors = {"beta_btc": beta_btc, "beta_qqq": beta_qqq, "vwap": vwap, "adx": adx, "regime": "Trend" if adx>25 else "Chop", "rsi": rsi, "vol_base": vol}
            
            # 回归模型
            df_reg = btdr.tail(30).copy()
            df_reg['PrevClose'] = df_reg['Close'].shift(1); df_reg = df_reg.dropna()
            x = ((df_reg['Open']-df_reg['PrevClose'])/df_reg['PrevClose']*100).values
            y_h = ((df_reg['High']-df_reg['PrevClose'])/df_reg['PrevClose']*100).values
            y_l = ((df_reg['Low']-df_reg['PrevClose'])/df_reg['PrevClose']*100).values
            cov_h = np.cov(x, y_h); b_h = cov_h[0,1]/cov_h[0,0] if cov_h[0,0]!=0 else 0.7
            cov_l = np.cov(x, y_l); b_l = cov_l[0,1]/cov_l[0,0] if cov_l[0,0]!=0 else 0.9
            
            model = {
                "high": {"intercept": 0.7*4.29 + 0.3*(np.mean(y_h)-b_h*np.mean(x)), "beta_open": 0.7*0.67 + 0.3*b_h, "beta_btc": 0.52},
                "low": {"intercept": 0.7*-3.22 + 0.3*(np.mean(y_l)-b_l*np.mean(x)), "beta_open": 0.7*0.88 + 0.3*b_l, "beta_btc": 0.42},
                "beta_sector": 0.25
            }
            return quotes, 50, model, factors
            
        else:
            return quotes, 50, default_model, default_factors
            
    except: return None, 50, default_model, default_factors

# --- 4. 局部刷新显示 ---
@st.fragment(run_every=5)
def show_dashboard():
    quotes, fng_val, model, factors = fetch_robust_data()
    
    if not quotes:
        st.warning("正在连接交易所数据...")
        return

    # 变量提取
    btdr = quotes['BTDR']
    btc_chg = quotes['BTC-USD']['pct']
    qqq_chg = quotes.get('QQQ', {'pct': 0})['pct']
    vix_val = quotes.get('^VIX', {'price': 20})['price']
    vix_chg = quotes.get('^VIX', {'pct': 0})['pct']
    
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny).strftime('%H:%M:%S')
    badge_color = "#fd7e14" if factors['regime'] == "Trend" else "#868e96"
    
    st.markdown(f"<div class='time-bar'>美东 {now_ny} &nbsp;|&nbsp; 状态: <span style='background:{badge_color};color:white;padding:1px 4px;border-radius:3px;font-size:0.6rem'>{factors['regime']}</span> &nbsp;|&nbsp; 引擎: v9.4 (Robust)</div>", unsafe_allow_html=True)
    
    # 核心
    c1, c2 = st.columns(2)
    with c1: st.markdown(card_html("BTC (全时段)", f"{btc_chg:+.2f}%", f"{btc_chg:+.2f}%", btc_chg), unsafe_allow_html=True)
    with c2: st.markdown(card_html("恐慌指数", f"{fng_val}", None, 0), unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    st.caption("⚒️ 矿股板块 Beta")
    cols = st.columns(5)
    peers = ["MARA", "RIOT", "CORZ", "CLSK", "IREN"]
    for i, p in enumerate(peers):
        val = quotes[p]['pct']
        cols[i].markdown(card_html(p, f"{val:+.1f}%", f"{val:+.1f}%", val), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 三栏布局
    c_live, c_open, c_vwap = st.columns(3)
    status_tag = f"<span class='status-dot dot-reg'></span>"
    with c_live: st.markdown(card_html("BTDR 实时", f"${btdr['price']:.2f}", f"{btdr['pct']:+.2f}%", btdr['pct'], status_tag), unsafe_allow_html=True)
    
    open_pct = ((btdr['open'] - btdr['prev']) / btdr['prev']) * 100 if btdr['prev'] > 0 else 0
    with c_open: st.markdown(card_html("计算用开盘", f"${btdr['open']:.2f}", f"{open_pct:+.2f}%", open_pct), unsafe_allow_html=True)
    
    vwap_val = factors['vwap']
    dist_vwap = ((btdr['price'] - vwap_val) / vwap_val) * 100 if vwap_val > 0 else 0
    with c_vwap: st.markdown(card_html("机构成本 (VWAP)", f"${vwap_val:.2f}", f"{dist_vwap:+.1f}% Prem.", dist_vwap), unsafe_allow_html=True)
    
    # 日内预测
    peers_avg = sum(quotes[p]['pct'] for p in peers) / 5
    sector_alpha = peers_avg - btc_chg
    sentiment_adj = (fng_val - 50) * 0.02
    
    pred_h_pct = (model['high']['intercept'] + (model['high']['beta_open'] * open_pct) + (model['high']['beta_btc'] * btc_chg) + (model['beta_sector'] * sector_alpha) + sentiment_adj)
    pred_l_pct = (model['low']['intercept'] + (model['low']['beta_open'] * open_pct) + (model['low']['beta_btc'] * btc_chg) + (model['beta_sector'] * sector_alpha) + sentiment_adj)
    pred_h = btdr['prev'] * (1 + pred_h_pct/100)
    pred_l = btdr['prev'] * (1 + pred_l_pct/100)
    
    st.markdown("### 🎯 日内阻力/支撑 (Intraday)")
    col_h, col_l = st.columns(2)
    h_bg = "#e6fcf5" if btdr['price'] < pred_h else "#0ca678"; h_txt = "#087f5b" if btdr['price'] < pred_h else "#ffffff"
    l_bg = "#fff5f5" if btdr['price'] > pred_l else "#e03131"; l_txt = "#c92a2a" if btdr['price'] > pred_l else "#ffffff"
    with col_h: st.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background-color: {h_bg}; color: {h_txt}; border: 1px solid #c3fae8;"><div style="font-size: 0.8rem; opacity: 0.8;">日内阻力 (High)</div><div style="font-size: 1.5rem; font-weight: bold;">${pred_h:.2f}</div></div></div>""", unsafe_allow_html=True)
    with col_l: st.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background-color: {l_bg}; color: {l_txt}; border: 1px solid #ffc9c9;"><div style="font-size: 0.8rem; opacity: 0.8;">日内支撑 (Low)</div><div style="font-size: 1.5rem; font-weight: bold;">${pred_l:.2f}</div></div></div>""", unsafe_allow_html=True)
    
    # 因子面板
    st.markdown("---")
    st.markdown("### 🌍 宏观环境 (Macro)")
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.markdown(factor_html("QQQ (纳指)", f"{qqq_chg:+.2f}%", "Market", qqq_chg, "美股科技股风向标"), unsafe_allow_html=True)
    with m2: st.markdown(factor_html("VIX (恐慌)", f"{vix_val:.1f}", f"{vix_chg:+.1f}%", -vix_chg, "市场恐慌指数，越高越危险", reverse_color=True), unsafe_allow_html=True)
    with m3: st.markdown(factor_html("Beta (BTC)", f"{factors['beta_btc']:.2f}", "Corr", 0, "与比特币的联动性"), unsafe_allow_html=True)
    with m4: st.markdown(factor_html("Beta (QQQ)", f"{factors['beta_qqq']:.2f}", "Corr", 0, "与美股大盘的联动性"), unsafe_allow_html=True)
    
    st.markdown("### 🔬 微观结构 (Micro)")
    mi1, mi2, mi3, mi4 = st.columns(4)
    drift = (btc_chg/100 * factors['beta_btc'] * 0.4) + (qqq_chg/100 * factors['beta_qqq'] * 0.4)
    if abs(dist_vwap) > 10: drift -= (dist_vwap/100) * 0.05
    
    with mi1: st.markdown(factor_html("ADX (强度)", f"{factors['adx']:.1f}", factors['regime'], 1 if factors['adx']>25 else -1, "趋势强度 >25 为趋势"), unsafe_allow_html=True)
    with mi2: st.markdown(factor_html("RSI (14d)", f"{factors['rsi']:.0f}", "Neu", 0, "强弱指标 >70超买 <30超卖"), unsafe_allow_html=True)
    with mi3: st.markdown(factor_html("Implied Vol", f"{factors['vol_base']*100:.1f}%", "Risk", 0, "潜在波动率"), unsafe_allow_html=True)
    with mi4: st.markdown(factor_html("Exp. Drift", f"{drift*100:+.2f}%", "Day", drift, "当日预期漂移率"), unsafe_allow_html=True)
    
    # 宗师推演
    st.markdown("### ☁️ 宗师级推演 (P90-P50-P10)")
    
    vol = factors['vol_base']
    if vix_val > 25: drift -= 0.005; vol *= 1.3
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
        chart_data.append({"Day": d, "P90": round(p90[d], 2), "P50": round(p50[d], 2), "P10": round(p10[d], 2)})
    df_chart = pd.DataFrame(chart_data)
    
    base = alt.Chart(df_chart).encode(x=alt.X('Day:O', title='未来交易日'))
    area = base.mark_area(opacity=0.2, color='#4dabf7').encode(y=alt.Y('P10', title='价格预演 (USD)', scale=alt.Scale(zero=False)), y2='P90')
    l90 = base.mark_line(color='#0ca678', strokeDash=[5,5]).encode(y='P90')
    l50 = base.mark_line(color='#228be6', size=3).encode(y='P50')
    l10 = base.mark_line(color='#d6336c', strokeDash=[5,5]).encode(y='P10')
    nearest = alt.selection_point(nearest=True, on='mouseover', fields=['Day'], empty=False)
    selectors = base.mark_rule(opacity=0).encode(x='Day:O').add_params(nearest)
    points = base.mark_circle(size=60, color="black").encode(y='P50', opacity=alt.condition(nearest, alt.value(1), alt.value(0)), tooltip=[alt.Tooltip('Day', title='T+'), alt.Tooltip('P90', title='P90 (High)', format='.2f'), alt.Tooltip('P50', title='P50 (Median)', format='.2f'), alt.Tooltip('P10', title='P10 (Low)', format='.2f')])
    
    st.altair_chart((area + l90 + l50 + l10 + selectors + points).properties(height=300).interactive(), use_container_width=True)
    st.caption(f"Engine: v9.4 Robust | Drift: {drift*100:.2f}% | Vol: {vol*100:.1f}%")

# --- 7. 主程序 ---
st.markdown("### ⚡ BTDR 领航员 v9.4")
show_dashboard()

with st.expander("📖 参数详解手册"):
    st.markdown("""
    * **计算用开盘**: 今日开盘价。若无数据则自动回溯昨收。
    * **VWAP**: 机构成本线。价格高于此线说明强势。
    * **P90/P10**: 预测价格的置信区间 (90%概率会落在此区间内)。
    """)
