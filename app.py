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
# 1. 基础配置 & 强力清缓存
# ==========================================
st.set_page_config(page_title="BTDR Pilot v10.4", layout="centered")

# 每次部署或重启时，强行删除 yfinance 缓存文件夹
if 'init_v104' not in st.session_state:
    try:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "yfinance")
        if os.path.exists(cache_dir): shutil.rmtree(cache_dir)
    except: pass
    st.session_state.clear()
    st.session_state['init_v104'] = True

# ==========================================
# 2. CSS 样式 (V9.1 风格：防抖 + 悬停 + 三栏)
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
# 3. 辅助函数 (强力清洗数据)
# ==========================================
def safe_float(val, default=0.0):
    """
    数据清洗核弹：
    无论输入是 NaN, Inf, None, Empty Series 还是 String，
    统统强制转为 float。如果失败，返回 default。
    """
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
# 4. 数据获取引擎 (V7.4 Core + Triple Lock)
# ==========================================
def fetch_data_v10_core():
    # 默认兜底
    quotes = {}
    model = {"high": {"intercept": 4.29, "beta_open": 0.67, "beta_btc": 0.52}, "low": {"intercept": -3.22, "beta_open": 0.88, "beta_btc": 0.42}, "beta_sector": 0.25}
    factors = {"vwap": 10.0, "adx": 20.0, "regime": "Neutral", "beta_btc": 1.5, "beta_qqq": 1.2, "rsi": 50, "vol_base": 0.05}
    fng = 50
    
    try:
        tickers = "BTDR BTC-USD QQQ ^VIX MARA RIOT CORZ CLSK IREN"
        
        # 1. 抓取数据 (Threads=False 是为了 Streamlit Cloud 的稳定性)
        # 历史数据 (1y) -> 用于因子计算和昨收判断
        hist = yf.download(tickers, period="1y", interval="1d", group_by='ticker', threads=False, progress=False)
        # 实时数据 (1d, 1m) -> 用于实时价格，必须开启 prepost
        live = yf.download(tickers, period="1d", interval="1m", prepost=True, group_by='ticker', threads=False, progress=False)
        
        today_ny = datetime.now(pytz.timezone('America/New_York')).date()
        syms = tickers.split()
        
        # 2. 处理行情 (三层兜底逻辑)
        for s in syms:
            try:
                df_d = hist[s] if s in hist else pd.DataFrame()
                df_m = live[s] if s in live else pd.DataFrame()
                
                # --- A. 实时价格 (Price) ---
                price = 0.0
                state = "ERR"
                
                # 优先1: 分钟线最新值 (最准)
                if not df_m.empty:
                    val = safe_float(df_m['Close'])
                    if val > 0: 
                        price = val
                        state = "REG"
                
                # 优先2: 如果分钟线为空(盘前/后没成交), 取日线最新值 (兜底)
                if price == 0 and not df_d.empty:
                    price = safe_float(df_d['Close'])
                    state = "CLOSED"
                
                # --- B. 昨收 (Prev) & 开盘 (Open) ---
                prev = 0.0
                open_p = 0.0
                
                if not df_d.empty:
                    last_dt = df_d.index[-1].date()
                    
                    if last_dt == today_ny and len(df_d) > 1:
                        # 正常交易日：昨收是倒数第二根
                        prev = safe_float(df_d['Close'].iloc[-2]) 
                        open_p = safe_float(df_d['Open'].iloc[-1])
                    else:
                        # 盘前或未开盘：昨收是最后一根
                        prev = safe_float(df_d['Close'].iloc[-1])
                        # 还没正式开盘数据，Open 暂时用 Price 或 Prev 代替，避免 nan
                        open_p = price if price > 0 else prev
                
                # --- C. 最终兜底 (防止除零或 NaN) ---
                if price <= 0.01: price = 10.0 # 极端口袋
                if prev <= 0.01: prev = price
                if open_p <= 0.01: open_p = price
                
                pct = ((price - prev) / prev) * 100
                quotes[s] = {"price": price, "pct": pct, "prev": prev, "open": open_p, "tag": state}
            except:
                # 发生任何错误，返回安全值，不报错
                quotes[s] = {"price": 10.0, "pct": 0.0, "prev": 10.0, "open": 10.0, "tag": "ERR"}

        # 3. 计算因子 (V9.0 逻辑，加了 safe_float 保护)
        btdr = hist['BTDR'].dropna(); btc = hist['BTC-USD'].dropna(); qqq = hist['QQQ'].dropna()
        idx = btdr.index.intersection(btc.index).intersection(qqq.index)
        
        if len(idx) > 20:
            btdr = btdr.loc[idx]; btc = btc.loc[idx]; qqq = qqq.loc[idx]
            
            rb = btdr['Close'].pct_change()
            rc = btc['Close'].pct_change()
            rq = qqq['Close'].pct_change()
            
            beta_btc = safe_float((rb.rolling(30).cov(rc)/rc.rolling(30).var()).iloc[-1], 1.5)
            beta_qqq = safe_float((rb.rolling(30).cov(rq)/rq.rolling(30).var()).iloc[-1], 1.2)
            
            btdr['TP'] = (btdr['High']+btdr['Low']+btdr['Close'])/3
            btdr['PV'] = btdr['TP']*btdr['Volume']
            vwap = safe_float(btdr['PV'].tail(20).sum() / btdr['Volume'].tail(20).sum(), quotes['BTDR']['price'])
            
            delta = btdr['Close'].diff()
            gain = (delta.where(delta>0, 0)).rolling(14).mean()
            loss = (-delta.where(delta<0, 0)).rolling(14).mean()
            rsi = safe_float(100 - (100/(1 + gain/loss)).iloc[-1], 50.0)
            
            high = btdr['High']; low = btdr['Low']; close = btdr['Close']
            tr = np.maximum(high-low, np.abs(high-close.shift(1)))
            atr = tr.rolling(14).mean()
            p_dm = (high-high.shift(1)).clip(lower=0)
            m_dm = (low.shift(1)-low).clip(lower=0)
            p_di = 100 * p_dm.rolling(14).mean() / atr
            m_di = 100 * m_dm.rolling(14).mean() / atr
            dx = 100 * np.abs(p_di-m_di)/(p_di+m_di)
            adx = safe_float(dx.rolling(14).mean().iloc[-1], 20.0)
            
            vol_base = safe_float(rb.ewm(span=20).std().iloc[-1], 0.05)
            
            factors = {"beta_btc": beta_btc, "beta_qqq": beta_qqq, "vwap": vwap, "adx": adx, "regime": "Trend" if adx>25 else "Chop", "rsi": rsi, "vol_base": vol_base}
            
            # Regression
            df_r = btdr.tail(30).copy()
            df_r['Prev'] = df_r['Close'].shift(1); df_r.dropna(inplace=True)
            x = ((df_r['Open']-df_r['Prev'])/df_r['Prev']*100).values
            yh = ((df_r['High']-df_r['Prev'])/df_r['Prev']*100).values
            yl = ((df_r['Low']-df_r['Prev'])/df_r['Prev']*100).values
            ch = np.cov(x, yh); bh = safe_float(ch[0,1]/ch[0,0], 0.7)
            cl = np.cov(x, yl); bl
