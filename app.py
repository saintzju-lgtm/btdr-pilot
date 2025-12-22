import streamlit as st
import requests
import pandas as pd
import time
from streamlit_autorefresh import st_autorefresh # 引入自动刷新组件

# --- 1. 页面配置 ---
st.set_page_config(page_title="BTDR Pilot Live", layout="centered")

# 【核心功能】每 5000 毫秒 (5秒) 自动刷新一次
# key 用于保持组件状态，避免重复加载
count = st_autorefresh(interval=5000, limit=None, key="realtime_counter")

# 强制覆盖 CSS：纯白底色，极简风格，适配手机
st.markdown("""
    <style>
    .stApp {background-color: #ffffff;}
    h1, h2, h3, div, p, span {color: #212529 !important; font-family: -apple-system, BlinkMacSystemFont, sans-serif !important;}
    
    /* 隐藏 Streamlit 默认的汉堡菜单和页脚，看起来像原生 App */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 卡片样式 */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa !important;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
        border-radius: 10px;
    }
    
    /* 数值大字体 */
    [data-testid="stMetricValue"] {
        font-weight: 700; 
        font-size: 1.4rem !important;
        color: #212529 !important;
    }
    
    /* 预测框动效 */
    .pred-box {
        padding: 15px; border-radius: 12px; margin-top: 10px; text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    
    /* 闪烁指示灯 (可选，增加实时感) */
    .live-indicator {
        height: 10px; width: 10px; background-color: #00e676;
        border-radius: 50%; display: inline-block; margin-right: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("### ⚡ BTDR 实盘监控")

# --- 2. 黄金参数 (插件同款) ---
MODEL = {
    "high": {"intercept": 4.29, "beta_open": 0.67, "beta_btc": 0.52},
    "low":  {"intercept": -3.22, "beta_open": 0.88, "beta_btc": 0.42},
    "beta_sector": 0.25
}

# --- 3. 数据获取 (原生接口) ---
def fetch_yahoo_raw(symbol):
    try:
        t = int(time.time() * 1000)
        # 随机 User-Agent 防止被 Yahoo 认为是爬虫
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&useYfid=true&_={t}"
        
        # 3秒超时，为了实时性，如果卡顿宁可直接失败
        resp = requests.get(url, headers=headers, timeout=3)
        data = resp.json()
        
        meta = data['chart']['result'][0]['meta']
        current = meta['regularMarketPrice']
        prev_close = meta['chartPreviousClose']
        
        # 盘前盘后逻辑：如果 regularMarketOpen 存在则用它，否则用 current
        open_price = meta.get('regularMarketOpen')
        if open_price is None: open_price = current 

        pct = ((current - prev_close) / prev_close) * 100
        
        return {"price": current, "pct": pct, "prev": prev_close, "open": open_price}
    except:
        return {"price": 0, "pct": 0, "prev": 0, "open": 0}

# 使用缓存装饰器，但 TTL 设为 0 或极短，确保每次刷新都真抓取
# 这里不加 @st.cache_data，因为我们希望每次 autorefresh 触发时都重新执行网络请求
def get_all_data():
    btc_data = fetch_yahoo_raw("BTC-USD") # 必须用 Yahoo 源以保持算法一致
    
    # 情绪接口稍微慢点，为了不拖慢整体速度，设短超时
    try:
        fng = requests.get("https://api.alternative.me/fng/", timeout=1).json()
        fng_val = int(fng['data'][0]['value'])
    except:
        fng_val = 50 

    tickers = ["BTDR", "MARA", "RIOT", "CORZ", "CLSK", "IREN"]
    quotes = {}
    for t in tickers:
        quotes[t] = fetch_yahoo_raw(t)
            
    return btc_data['pct'], fng_val, quotes

# --- 4. 核心计算 ---
btc_chg, fng_val, quotes = get_all_data()

# 板块 Alpha
peers = ["MARA", "RIOT", "CORZ", "CLSK", "IREN"]
valid_peers = [p for p in peers if quotes[p]['price'] > 0]
peers_avg = sum(quotes[p]['pct'] for p in valid_peers) / len(valid_peers) if valid_peers else 0

sector_alpha = peers_avg - btc_chg
sentiment_adj = (fng_val - 50) * 0.02

# BTDR 预测
btdr = quotes['BTDR']
pred_high_price, pred_low_price, pred_high_pct, pred_low_pct = 0, 0, 0, 0

if btdr['price'] > 0 and btdr['prev'] > 0:
    btdr_open_pct = ((btdr['open'] - btdr['prev']) / btdr['prev']) * 100
    
    pred_high_pct = (MODEL['high']['intercept'] + (MODEL['high']['beta_open'] * btdr_open_pct) + (MODEL['high']['beta_btc'] * btc_chg) + (MODEL['beta_sector'] * sector_alpha) + sentiment_adj)
    pred_low_pct = (MODEL['low']['intercept'] + (MODEL['low']['beta_open'] * btdr_open_pct) + (MODEL['low']['beta_btc'] * btc_chg) + (MODEL['beta_sector'] * sector_alpha) + sentiment_adj)
    
    pred_high_price = btdr['prev'] * (1 + pred_high_pct / 100)
    pred_low_price = btdr['prev'] * (1 + pred_low_pct / 100)
else:
    btdr_open_pct = 0

# --- 5. 渲染界面 (极简布局) ---

# 状态栏
t = time.strftime('%H:%M:%S')
st.caption(f"<div class='live-indicator'></div>Live Feed | 更新于 {t}", unsafe_allow_html=True)

# 核心指标
c1, c2 = st.columns(2)
c1.metric("BTC (Yahoo)", f"{btc_chg:+.2f}%")
c2.metric("恐慌指数", f"{fng_val}")

# 板块微缩图
st.markdown("##### ⚒️ 矿股板块 Beta")
cols = st.columns(5)
for i, p in enumerate(peers):
    cols[i].metric(p, f"{quotes[p]['pct']:+.1f}%")

st.markdown("---")

# BTDR 数据
c3, c4 = st.columns(2)
c3.metric("BTDR 现价", f"${btdr['price']:.2f}", f"{btdr['pct']:+.2f}%")
c4.metric("今日开盘", f"${btdr['open']:.2f}", f"{btdr_open_pct:+.2f}%")

# 预测结果
st.markdown("### 🎯 AI 实时预测")
col_h, col_l = st.columns(2)

# 动态配色：如果现价已经突破阻力位，显示高亮提示
high_style = "border: 2px solid #00e676;" if btdr['price'] >= pred_high_price else "border: 1px solid #badbcc;"
low_style = "border: 2px solid #ff1744;" if btdr['price'] <= pred_low_price else "border: 1px solid #f5c2c7;"

with col_h:
    st.markdown(f"""
    <div class="pred-box" style="background-color: #d1e7dd; color: #0f5132; {high_style}">
        <div style="font-size: 0.8rem;">阻力位 (High)</div>
        <div style="font-size: 1.5rem; font-weight: bold;">${pred_high_price:.2f}</div>
        <div style="font-size: 0.8rem;">预期: {pred_high_pct:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col_l:
    st.markdown(f"""
    <div class="pred-box" style="background-color: #f8d7da; color: #842029; {low_style}">
        <div style="font-size: 0.8rem;">支撑位 (Low)</div>
        <div style="font-size: 1.5rem; font-weight: bold;">${pred_low_price:.2f}</div>
        <div style="font-size: 0.8rem;">预期: {pred_low_pct:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
