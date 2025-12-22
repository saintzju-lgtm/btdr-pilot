import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import pytz
from streamlit_autorefresh import st_autorefresh

# --- 1. 页面配置 ---
st.set_page_config(page_title="BTDR Pilot v7.3", layout="centered")

# 5秒刷新
st_autorefresh(interval=5000, limit=None, key="realtime_counter")

# CSS 样式
st.markdown("""
    <style>
    html { overflow-y: scroll; }
    .stApp > header { display: none; }
    .stApp { margin-top: -30px; background-color: #ffffff; }
    div[data-testid="stStatusWidget"] { visibility: hidden; }
    
    h1, h2, h3, div, p, span { 
        color: #212529 !important; 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important; 
    }
    
    div[data-testid="stMetric"] {
        background-color: #f8f9fa !important;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        height: 90px;
        display: flex; flex-direction: column; justify-content: center;
        overflow: hidden;
    }
    
    .btdr-box {
        height: 95px;
        background-color: #fff;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 0 16px;
        display: flex; flex-direction: column; justify-content: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    }

    .pred-container-wrapper {
        height: 110px; width: 100%; display: block;
    }
    .pred-box {
        padding: 0 10px; border-radius: 12px; text-align: center;
        height: 100%; display: flex; flex-direction: column; justify-content: center;
        transition: all 0.3s ease;
    }
    
    .time-bar {
        font-size: 0.75rem; color: #999; text-align: center;
        margin-bottom: 20px; padding: 6px; 
        background: #fafafa; border-radius: 6px;
    }
    
    .status-dot {
        height: 6px; width: 6px; border-radius: 50%; display: inline-block; 
        margin-left: 6px; vertical-align: middle; margin-bottom: 2px;
    }
    .dot-pre { background-color: #f59f00; box-shadow: 0 0 4px #f59f00; }
    .dot-reg { background-color: #0ca678; box-shadow: 0 0 4px #0ca678; }
    .dot-post { background-color: #1c7ed6; box-shadow: 0 0 4px #1c7ed6; }
    .dot-closed { background-color: #adb5bd; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 状态管理与热修复 ---
if 'data_cache' not in st.session_state:
    st.session_state['data_cache'] = None

# 【v7.3 关键修复】检查缓存版本
# 如果缓存存在，但里面没有 'model' 字段（说明是旧版缓存），强制清空
if st.session_state['data_cache'] is not None:
    if 'model' not in st.session_state['data_cache']:
        st.session_state['data_cache'] = None
        st.rerun() # 立即重启脚本，防止报错

# 标题
st.markdown("### ⚡ BTDR 领航员 v7.3")

# --- 3. UI 骨架 ---
ph_time = st.empty()

c1, c2 = st.columns(2)
with c1: ph_btc = st.empty()
with c2: ph_fng = st.empty()

st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
st.caption("⚒️ 矿股板块 Beta")
cols = st.columns(5)
ph_peers = [col.empty() for col in cols]

st.markdown("---")

c3, c4 = st.columns(2)
with c3: ph_btdr_price = st.empty()
with c4: ph_btdr_open = st.empty()

st.markdown("### 🎯 AI 托管预测")
col_h, col_l = st.columns(2)
with col_h: ph_pred_high = st.empty()
with col_l: ph_pred_low = st.empty()

st.markdown("---")
ph_footer = st.empty()

# --- 4. 核心逻辑：AI 自动调参 ---
@st.cache_data(ttl=3600)
def auto_tune_model():
    default_model = {
        "high": {"intercept": 4.29, "beta_open": 0.67, "beta_btc": 0.52},
        "low":  {"intercept": -3.22, "beta_open": 0.88, "beta_btc": 0.42},
        "beta_sector": 0.25
    }
    
    try:
        df = yf.download("BTDR", period="1mo", interval="1d", progress=False)
        if len(df) < 10: return default_model, "数据不足"
        
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs('BTDR', axis=1, level=1)
            
        df = df.dropna()
        df['PrevClose'] = df['Close'].shift(1)
        df = df.dropna()
        
        x = ((df['Open'] - df['PrevClose']) / df['PrevClose'] * 100).values
        y_high = ((df['High'] - df['PrevClose']) / df['PrevClose'] * 100).values
        y_low = ((df['Low'] - df['PrevClose']) / df['PrevClose'] * 100).values
        
        # High Model
        cov_h = np.cov(x, y_high)
        beta_h = cov_h[0, 1] / cov_h[0, 0] if cov_h[0, 0] != 0 else 0.67
        intercept_h = np.mean(y_high) - beta_h * np.mean(x)
        
        # Low Model
        cov_l = np.cov(x, y_low)
        beta_l = cov_l[0, 1] / cov_l[0, 0] if cov_l[0, 0] != 0 else 0.88
        intercept_l = np.mean(y_low) - beta_l * np.mean(x)
        
        # Safety Clip
        beta_h = np.clip(beta_h, 0.3, 1.2)
        beta_l = np.clip(beta_l, 0.4, 1.5)
        
        final_model = {
            "high": {
                "intercept": 0.7 * 4.29 + 0.3 * intercept_h,
                "beta_open": 0.7 * 0.67 + 0.3 * beta_h,
                "beta_btc": 0.52
            },
            "low": {
                "intercept": 0.7 * -3.22 + 0.3 * intercept_l,
                "beta_open": 0.7 * 0.88 + 0.3 * beta_l,
                "beta_btc": 0.42
            },
            "beta_sector": 0.25
        }
        return final_model, "已自适应"
        
    except Exception:
        return default_model, "默认参数"

# --- 5. 渲染函数 ---
def render_ui(data):
    if not data: return
    
    # 【v7.3 修复】再次进行安全检查，确保字段存在
    if 'model' not in data: return 
    
    quotes = data['quotes']
    fng_val = data['fng']
    model_params = data['model']
    model_status = data['model_status']
    
    btc_chg = quotes['BTC-USD']['pct']
    btdr = quotes['BTDR']
    
    tz_bj = pytz.timezone('Asia/Shanghai')
    tz_ny = pytz.timezone('America/New_York')
    now_bj = datetime.now(tz_bj).strftime('%H:%M:%S')
    now_ny = datetime.now(tz_ny).strftime('%H:%M:%S')
    
    ph_time.markdown(f"<div class='time-bar'>北京 {now_bj} &nbsp;|&nbsp; 美东 {now_ny} &nbsp;|&nbsp; AI {model_status}</div>", unsafe_allow_html=True)
    
    ph_btc.metric("BTC (全时段)", f"{btc_chg:+.2f}%")
    ph_fng.metric("恐慌指数", f"{fng_val}")
    
    peers = ["MARA", "RIOT", "CORZ", "CLSK", "IREN"]
    for i, p in enumerate(peers):
        if p in quotes: ph_peers[i].metric(p, f"{quotes[p]['pct']:+.1f}%")
            
    valid_peers = [p for p in peers if quotes[p]['price'] > 0]
    peers_avg = sum(quotes[p]['pct'] for p in valid_peers) / len(valid_peers) if valid_peers else 0
    sector_alpha = peers_avg - btc_chg
    sentiment_adj = (fng_val - 50) * 0.02
    
    pred_high_price, pred_low_price, pred_high_pct, pred_low_pct, btdr_open_pct = 0,0,0,0,0
    
    if btdr['price'] > 0:
        btdr_open_pct = ((btdr['open'] - btdr['prev']) / btdr['prev']) * 100
        
        MODEL = model_params
        pred_high_pct = (MODEL['high']['intercept'] + (MODEL['high']['beta_open'] * btdr_open_pct) + (MODEL['high']['beta_btc'] * btc_chg) + (MODEL['beta_sector'] * sector_alpha) + sentiment_adj)
        pred_low_pct = (MODEL['low']['intercept'] + (MODEL['low']['beta_open'] * btdr_open_pct) + (MODEL['low']['beta_btc'] * btc_chg) + (MODEL['beta_sector'] * sector_alpha) + sentiment_adj)
        
        pred_high_price = btdr['prev'] * (1 + pred_high_pct / 100)
        pred_low_price = btdr['prev'] * (1 + pred_low_pct / 100)

    state_map = {"PRE": "dot-pre", "REG": "dot-reg", "POST": "dot-post", "CLOSED": "dot-closed"}
    dot_class = state_map.get(btdr.get('tag', 'CLOSED'), 'dot-closed')
    state_text = btdr.get('tag', 'CLOSED')
    
    ph_btdr_price.markdown(f"""
    <div class="btdr-box">
        <div style='font-size:0.75rem; color:#888; display:flex; align-items:center;'>
            BTDR 实时 <span class='status-dot {dot_class}'></span> <span style='margin-left:4px; font-size:0.7rem;'>{state_text}</span>
        </div>
        <div style='font-size:1.8rem; font-weight:700; color:#212529; margin: 2px 0;'>${btdr['price']:.2f}</div>
        <div style='font-size:0.9rem; color:{'#0ca678' if btdr['pct']>=0 else '#d6336c'}; font-weight:600;'>{btdr['pct']:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    ph_btdr_open.metric("计算用开盘", f"${btdr['open']:.2f}", f"{btdr_open_pct:+.2f}%")
    
    h_bg = "#e6fcf5" if btdr['price'] < pred_high_price else "#0ca678"; h_txt = "#087f5b" if btdr['price'] < pred_high_price else "#ffffff"
    l_bg = "#fff5f5" if btdr['price'] > pred_low_price else "#e03131"; l_txt = "#c92a2a" if btdr['price'] > pred_low_price else "#ffffff"

    ph_pred_high.markdown(f"""
    <div class="pred-container-wrapper">
        <div class="pred-box" style="background-color: {h_bg}; color: {h_txt}; border: 1px solid #c3fae8;">
            <div style="font-size: 0.8rem; opacity: 0.8;">阻力位 (High)</div>
            <div style="font-size: 1.5rem; font-weight: bold;">${pred_high_price:.2f}</div>
            <div style="font-size: 0.75rem; opacity: 0.9;">预期: {pred_high_pct:+.2f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    ph_pred_low.markdown(f"""
    <div class="pred-container-wrapper">
        <div class="pred-box" style="background-color: {l_bg}; color: {l_txt}; border: 1px solid #ffc9c9;">
            <div style="font-size: 0.8rem; opacity: 0.8;">支撑位 (Low)</div>
            <div style="font-size: 1.5rem; font-weight: bold;">${pred_low_price:.2f}</div>
            <div style="font-size: 0.75rem; opacity: 0.9;">预期: {pred_low_pct:+.2f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    ph_footer.caption(f"Update: {now_ny} ET | Auto-Tuned by AI")

# --- 6. 数据与模型获取 ---
@st.cache_data(ttl=5)
def get_data_v72():
    tickers_list = "BTC-USD BTDR MARA RIOT CORZ CLSK IREN"
    try:
        daily = yf.download(tickers_list, period="5d", interval="1d", group_by='ticker', threads=True, progress=False)
        live = yf.download(tickers_list, period="1d", interval="1m", prepost=True, group_by='ticker', threads=True, progress=False)
        
        quotes = {}
        symbols = tickers_list.split()
        today_ny = datetime.now(pytz.timezone('America/New_York')).date()
        
        for sym in symbols:
            try:
                df_day = daily[sym] if sym in daily else pd.DataFrame()
                if not df_day.empty: df_day = df_day.dropna(subset=['Close'])
                
                df_min = live[sym] if sym in live else pd.DataFrame()
                if not df_min.empty: df_min = df_min.dropna(subset=['Close'])
                
                # A. 实时价
                state = "CLOSED"
                if not df_min.empty:
                    current_price = df_min['Close'].iloc[-1]
                    state = "REG" 
                elif not df_day.empty:
                    current_price = df_day['Close'].iloc[-1]
                else:
                    current_price = 0
                
                # B. 昨收
                prev_close = 1.0
                if not df_day.empty:
                    last_date = df_day.index[-1].date()
                    if last_date == today_ny:
                        if len(df_day) >= 2: prev_close = df_day['Close'].iloc[-2]
                        elif not df_day.empty: prev_close = df_day['Open'].iloc[-1]
                    else:
                        prev_close = df_day['Close'].iloc[-1]
                
                # C. 涨跌
                pct = ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0
                
                # D. 开盘
                if not df_day.empty and df_day.index[-1].date() == today_ny:
                     open_price = df_day['Open'].iloc[-1]
                else:
                     open_price = current_price

                quotes[sym] = {"price": current_price, "pct": pct, "prev": prev_close, "open": open_price, "tag": state}
            except:
                quotes[sym] = {"price": 0, "pct": 0, "prev": 0, "open": 0, "tag": "ERR"}
        return quotes
    except:
        return None

# --- 7. 执行流 ---

# 如果有有效缓存，先渲染
if st.session_state['data_cache'] and 'model' in st.session_state['data_cache']: 
    render_ui(st.session_state['data_cache'])
else:
    ph_time.info("📡 正在升级 AI 内核...")

# 获取数据
new_quotes = get_data_v72()
ai_model, ai_status = auto_tune_model()

if new_quotes:
    try: fng = int(requests.get("https://api.alternative.me/fng/", timeout=1).json()['data'][0]['value'])
    except: fng = 50
    
    new_data = {
        'quotes': new_quotes, 
        'fng': fng, 
        'model': ai_model,
        'model_status': ai_status
    }
    st.session_state['data_cache'] = new_data
    render_ui(new_data)
