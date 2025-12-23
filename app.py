import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
import pytz
import requests

# --- 1. 页面配置 & 缓存强制清洗 ---
st.set_page_config(page_title="BTDR Pilot v9.0", layout="centered")

# 【关键】版本控制：如果版本号变了，强制清除所有缓存，解决 KeyError
if 'version' not in st.session_state or st.session_state['version'] != '9.0':
    st.session_state.clear()
    st.session_state['version'] = '9.0'

# CSS: 视觉锁定
st.markdown("""
    <style>
    /* 全局样式 */
    html { overflow-y: scroll; }
    .stApp > header { display: none; }
    .stApp { margin-top: -30px; background-color: #ffffff; }
    
    /* 字体优化 */
    h1, h2, h3, div, p, span, li { 
        color: #212529 !important; 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important; 
    }
    
    /* 图表容器锁定：防止高度塌陷 */
    div[data-testid="stAltairChart"] {
        height: 320px !important;
        min-height: 320px !important;
        overflow: hidden !important;
        border: 1px solid #f8f9fa;
    }
    
    /* 指标卡片 */
    .metric-card {
        background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02); height: 95px; padding: 0 16px;
        display: flex; flex-direction: column; justify-content: center;
    }
    .metric-label { font-size: 0.75rem; color: #888; margin-bottom: 2px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #212529; line-height: 1.2; }
    .metric-delta { font-size: 0.9rem; font-weight: 600; margin-top: 2px; }
    
    /* 因子卡片 */
    .factor-box {
        background: #fff; border: 1px solid #eee; border-radius: 8px; padding: 6px; text-align: center;
        height: 75px; display: flex; flex-direction: column; justify-content: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .factor-title { font-size: 0.65rem; color: #999; text-transform: uppercase; letter-spacing: 0.5px; }
    .factor-val { font-size: 1.1rem; font-weight: bold; color: #495057; margin: 2px 0; }
    .factor-sub { font-size: 0.7rem; font-weight: 600; }
    
    /* 颜色定义 */
    .color-up { color: #0ca678; } .color-down { color: #d6336c; } .color-neutral { color: #adb5bd; }
    .status-dot { height: 6px; width: 6px; border-radius: 50%; display: inline-block; margin-left: 6px; }
    .dot-reg { background-color: #0ca678; } .dot-closed { background-color: #adb5bd; }
    
    /* 日内预测框 */
    .pred-container-wrapper { height: 110px; width: 100%; display: block; margin-top: 5px; }
    .pred-box { padding: 0 10px; border-radius: 12px; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center; }
    
    /* 状态栏 */
    .time-bar { font-size: 0.75rem; color: #999; text-align: center; margin-bottom: 20px; padding: 6px; background: #fafafa; border-radius: 6px; }
    .badge-trend { background:#fd7e14; color:white; padding:1px 4px; border-radius:3px; font-size:0.6rem; }
    .badge-chop { background:#868e96; color:white; padding:1px 4px; border-radius:3px; font-size:0.6rem; }
    
    /* Expander */
    .streamlit-expanderHeader { font-size: 0.8rem; color: #666; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. HTML 辅助函数 ---
def card_html(label, value_str, delta_str=None, delta_val=0, extra_tag=""):
    delta_html = ""
    if delta_str:
        color_class = "color-up" if delta_val >= 0 else "color-down"
        delta_html = f"<div class='metric-delta {color_class}'>{delta_str}</div>"
    return f"""<div class="metric-card"><div class="metric-label">{label} {extra_tag}</div><div class="metric-value">{value_str}</div>{delta_html}</div>"""

def factor_html(title, val, delta_str, delta_val, reverse_color=False):
    color_class = "color-up" if delta_val >= 0 else "color-down"
    if reverse_color: color_class = "color-down" if delta_val >= 0 else "color-up"
    return f"""<div class="factor-box"><div class="factor-title">{title}</div><div class="factor-val">{val}</div><div class="factor-sub {color_class}">{delta_str}</div></div>"""

# --- 3. 数据获取与计算 (Data Logic) ---
# 注意：不缓存此函数，确保每次 Fragment 运行时都拿最新数据
def fetch_and_calculate():
    # 默认兜底数据
    default_model = {"high": {"intercept": 4.29, "beta_open": 0.67, "beta_btc": 0.52}, "low": {"intercept": -3.22, "beta_open": 0.88, "beta_btc": 0.42}, "beta_sector": 0.25}
    default_factors = {"vwap": 0, "adx": 20, "regime": "Neutral", "beta_btc": 1.5, "beta_qqq": 1.2, "rsi": 50, "vol_base": 0.05}
    
    try:
        # 1. 延长到 1y 以修复 ADX nan
        data = yf.download("BTDR BTC-USD QQQ ^VIX", period="1y", interval="1d", group_by='ticker', threads=True, progress=False)
        live = yf.download("BTDR BTC-USD QQQ ^VIX", period="1d", interval="1m", prepost=True, group_by='ticker', threads=True, progress=False)
        
        # 2. 处理实时行情
        quotes = {}
        symbols = ["BTDR", "BTC-USD", "QQQ", "^VIX", "MARA", "RIOT", "CORZ", "CLSK", "IREN"]
        
        # 重新获取板块实时数据 (上面只拿了核心宏观)
        peers_live = yf.download("MARA RIOT CORZ CLSK IREN", period="1d", interval="1m", prepost=True, group_by='ticker', threads=True, progress=False)
        
        # 整合 Quotes
        for sym in symbols:
            try:
                # 确定数据源
                if sym in ["MARA", "RIOT", "CORZ", "CLSK", "IREN"]:
                    df_min = peers_live[sym] if sym in peers_live else pd.DataFrame()
                    df_day = pd.DataFrame() # 板块不需要日线历史
                else:
                    df_day = data[sym].dropna()
                    df_min = live[sym] if sym in live else pd.DataFrame()
                
                # 提取价格
                price = 0
                if not df_min.empty:
                    price = df_min['Close'].iloc[-1]
                    state = "REG"
                elif not df_day.empty:
                    price = df_day['Close'].iloc[-1]
                    state = "CLOSED"
                else:
                    price = 0
                    state = "ERR"
                
                # 计算涨跌幅
                prev = 0
                if not df_day.empty:
                    prev = df_day['Close'].iloc[-1]
                    # 如果是当天，取昨收
                    if df_day.index[-1].date() == datetime.now(pytz.timezone('America/New_York')).date():
                         if len(df_day) > 1: prev = df_day['Close'].iloc[-2]
                
                if prev == 0 and price > 0: prev = price # 防止除零
                
                pct = ((price - prev) / prev) * 100 if prev > 0 else 0
                
                # 特殊处理 BTDR Open
                open_p = 0
                if sym == 'BTDR' and not df_day.empty:
                    open_p = df_day['Open'].iloc[-1]
                if open_p == 0: open_p = price
                
                quotes[sym] = {"price": price, "pct": pct, "prev": prev, "open": open_p, "tag": state}
            except:
                quotes[sym] = {"price": 0, "pct": 0, "prev": 0, "open": 0, "tag": "ERR"}

        # 3. 计算因子
        btdr = data['BTDR'].dropna()
        btc = data['BTC-USD'].dropna()
        qqq = data['QQQ'].dropna()
        
        # 对齐索引
        common = btdr.index.intersection(btc.index).intersection(qqq.index)
        btdr = btdr.loc[common]
        btc = btc.loc[common]
        qqq = qqq.loc[common]
        
        # Beta
        ret_btdr = btdr['Close'].pct_change()
        ret_btc = btc['Close'].pct_change()
        ret_qqq = qqq['Close'].pct_change()
        
        beta_btc = (ret_btdr.rolling(60).cov(ret_btc) / ret_btc.rolling(60).var()).iloc[-1]
        beta_qqq = (ret_btdr.rolling(60).cov(ret_qqq) / ret_qqq.rolling(60).var()).iloc[-1]
        
        # VWAP
        btdr['TP'] = (btdr['High'] + btdr['Low'] + btdr['Close']) / 3
        btdr['PV'] = btdr['TP'] * btdr['Volume']
        vwap_30d = btdr['PV'].tail(30).sum() / btdr['Volume'].tail(30).sum()
        
        # ADX (修复 nan)
        high = btdr['High']; low = btdr['Low']; close = btdr['Close']
        tr = np.maximum(high - low, np.abs(high - close.shift(1)))
        atr = tr.rolling(14).mean()
        up = high - high.shift(1); down = low.shift(1) - low
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        plus_di = 100 * (pd.Series(plus_dm, index=btdr.index).rolling(14).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm, index=btdr.index).rolling(14).mean() / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean().iloc[-1]
        if np.isnan(adx): adx = 25.0 # 兜底值
        regime = "Trend" if adx > 25 else "Chop"
        
        # RSI
        delta = btdr['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        vol_base = ret_btdr.ewm(span=20).std().iloc[-1]
        
        factors = {"beta_btc": beta_btc, "beta_qqq": beta_qqq, "vwap": vwap_30d, "adx": adx, "regime": regime, "rsi": rsi, "vol_base": vol_base}
        
        # 模型参数 (Regression)
        # 简化处理，使用最近30天
        df_reg = btdr.tail(30).copy()
        df_reg['PrevClose'] = df_reg['Close'].shift(1)
        df_reg = df_reg.dropna()
        x = ((df_reg['Open'] - df_reg['PrevClose']) / df_reg['PrevClose'] * 100).values
        y_high = ((df_reg['High'] - df_reg['PrevClose']) / df_reg['PrevClose'] * 100).values
        y_low = ((df_reg['Low'] - df_reg['PrevClose']) / df_reg['PrevClose'] * 100).values
        cov_h = np.cov(x, y_high); beta_h = cov_h[0, 1] / cov_h[0, 0] if cov_h[0, 0] != 0 else 0.67
        cov_l = np.cov(x, y_low); beta_l = cov_l[0, 1] / cov_l[0, 0] if cov_l[0, 0] != 0 else 0.88
        
        model = {
            "high": {"intercept": 0.7*4.29 + 0.3*(np.mean(y_high)-beta_h*np.mean(x)), "beta_open": 0.7*0.67 + 0.3*np.clip(beta_h,0.3,1.2), "beta_btc": 0.52},
            "low": {"intercept": 0.7*-3.22 + 0.3*(np.mean(y_low)-beta_l*np.mean(x)), "beta_open": 0.7*0.88 + 0.3*np.clip(beta_l,0.4,1.5), "beta_btc": 0.42},
            "beta_sector": 0.25
        }
        
        try: fng = int(requests.get("https://api.alternative.me/fng/", timeout=1).json()['data'][0]['value'])
        except: fng = 50
        
        return quotes, fng, model, factors
        
    except Exception as e:
        # print(e) # Debug
        return None, 50, default_model, default_factors

# --- 4. 局部刷新容器 (The Fragment) ---
@st.fragment(run_every=5)
def dashboard():
    # 1. 加载数据
    quotes, fng_val, model, factors = fetch_and_calculate()
    
    if not quotes:
        st.warning("正在初始化数据引擎...")
        return

    # 2. 准备变量
    btdr = quotes['BTDR']
    btc_chg = quotes['BTC-USD']['pct']
    qqq_chg = quotes['QQQ']['pct']
    vix_val = quotes['^VIX']['price']
    vix_chg = quotes['^VIX']['pct']
    
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny).strftime('%H:%M:%S')
    
    badge_color = "#fd7e14" if factors['regime'] == "Trend" else "#868e96"
    st.markdown(f"<div class='time-bar'>美东 {now_ny} &nbsp;|&nbsp; 状态: <span style='background:{badge_color};color:white;padding:1px 4px;border-radius:3px;font-size:0.6rem'>{factors['regime']}</span> &nbsp;|&nbsp; 引擎: v9.0 (Stable)</div>", unsafe_allow_html=True)
    
    # 3. 核心指标
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
    
    # 4. BTDR & VWAP
    c3, c4 = st.columns(2)
    state_map = {"PRE": "dot-reg", "REG": "dot-reg", "POST": "dot-reg", "CLOSED": "dot-closed"}
    dot_class = state_map.get(btdr.get('tag', 'CLOSED'), 'dot-closed')
    status_tag = f"<span class='status-dot {dot_class}'></span>"
    
    with c3: st.markdown(card_html("BTDR 实时", f"${btdr['price']:.2f}", f"{btdr['pct']:+.2f}%", btdr['pct'], status_tag), unsafe_allow_html=True)
    
    dist_vwap = ((btdr['price'] - factors['vwap']) / factors['vwap']) * 100 if factors['vwap'] > 0 else 0
    with c4: st.markdown(card_html("机构成本 (VWAP)", f"${factors['vwap']:.2f}", f"{dist_vwap:+.1f}% Prem.", dist_vwap), unsafe_allow_html=True)
    
    # 5. 日内阻力/支撑
    btdr_open_pct = ((btdr['open'] - btdr['prev']) / btdr['prev']) * 100
    # 简单计算板块均值
    valid_peers = [p for p in peers if quotes[p]['price'] > 0]
    peers_avg = sum(quotes[p]['pct'] for p in valid_peers) / len(valid_peers) if valid_peers else 0
    sector_alpha = peers_avg - btc_chg
    sentiment_adj = (fng_val - 50) * 0.02
    
    pred_high_pct = (model['high']['intercept'] + (model['high']['beta_open'] * btdr_open_pct) + (model['high']['beta_btc'] * btc_chg) + (model['beta_sector'] * sector_alpha) + sentiment_adj)
    pred_low_pct = (model['low']['intercept'] + (model['low']['beta_open'] * btdr_open_pct) + (model['low']['beta_btc'] * btc_chg) + (model['beta_sector'] * sector_alpha) + sentiment_adj)
    pred_high = btdr['prev'] * (1 + pred_high_pct/100)
    pred_low = btdr['prev'] * (1 + pred_low_pct/100)
    
    st.markdown("### 🎯 日内阻力/支撑 (Intraday)")
    col_h, col_l = st.columns(2)
    h_bg = "#e6fcf5" if btdr['price'] < pred_high else "#0ca678"; h_txt = "#087f5b" if btdr['price'] < pred_high else "#ffffff"
    l_bg = "#fff5f5" if btdr['price'] > pred_low else "#e03131"; l_txt = "#c92a2a" if btdr['price'] > pred_low else "#ffffff"
    
    with col_h: st.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background-color: {h_bg}; color: {h_txt}; border: 1px solid #c3fae8;"><div style="font-size: 0.8rem; opacity: 0.8;">日内阻力 (High)</div><div style="font-size: 1.5rem; font-weight: bold;">${pred_high:.2f}</div></div></div>""", unsafe_allow_html=True)
    with col_l: st.markdown(f"""<div class="pred-container-wrapper"><div class="pred-box" style="background-color: {l_bg}; color: {l_txt}; border: 1px solid #ffc9c9;"><div style="font-size: 0.8rem; opacity: 0.8;">日内支撑 (Low)</div><div style="font-size: 1.5rem; font-weight: bold;">${pred_low:.2f}</div></div></div>""", unsafe_allow_html=True)
    
    # 6. 因子面板
    st.markdown("---")
    st.markdown("### 🌍 宏观环境 (Macro)")
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.markdown(factor_html("QQQ (纳指)", f"{qqq_chg:+.2f}%", "Market", qqq_chg), unsafe_allow_html=True)
    with m2: st.markdown(factor_html("VIX (恐慌)", f"{vix_val:.1f}", f"{vix_chg:+.1f}%", -vix_chg, reverse_color=True), unsafe_allow_html=True)
    with m3: st.markdown(factor_html("Beta (BTC)", f"{factors['beta_btc']:.2f}", "Corr", 0), unsafe_allow_html=True)
    with m4: st.markdown(factor_html("Beta (QQQ)", f"{factors['beta_qqq']:.2f}", "Corr", 0), unsafe_allow_html=True)
    
    st.markdown("### 🔬 微观结构 (Micro)")
    mi1, mi2, mi3, mi4 = st.columns(4)
    
    drift_est = (btc_chg/100 * factors['beta_btc'] * 0.4) + (qqq_chg/100 * factors['beta_qqq'] * 0.4)
    if abs(dist_vwap) > 10: drift_est -= (dist_vwap/100) * 0.05
    
    with mi1: st.markdown(factor_html("ADX (强度)", f"{factors['adx']:.1f}", factors['regime'], 1 if factors['adx']>25 else -1), unsafe_allow_html=True)
    with mi2: st.markdown(factor_html("RSI (14d)", f"{factors['rsi']:.0f}", "O/B" if factors['rsi']>70 else ("O/S" if factors['rsi']<30 else "Neu"), 0), unsafe_allow_html=True)
    with mi3: st.markdown(factor_html("Implied Vol", f"{factors['vol_base']*100:.1f}%", "Risk", 0), unsafe_allow_html=True)
    with mi4: st.markdown(factor_html("Exp. Drift", f"{drift_est*100:+.2f}%", "Day", drift_est), unsafe_allow_html=True)
    
    # 7. 宗师级图表
    st.markdown("### ☁️ 宗师级推演 (P90-P50-P10)")
    
    # 模拟计算
    vol = factors['vol_base']
    drift = drift_est
    if vix_val > 25: drift -= 0.005; vol *= 1.3
    if factors['rsi'] > 75: drift -= 0.003
    if factors['rsi'] < 25: drift += 0.003
    if factors['regime'] == "Chop": drift *= 0.5; vol *= 0.8
    
    simulations = 500
    days_ahead = 5
    current = btdr['price']
    if current == 0: current = 10 # 兜底防止报错
    
    paths = []
    for i in range(simulations):
        path = [current]
        p = current
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
    
    # 构建数据框
    chart_data = []
    for d in range(days_ahead + 1):
        chart_data.append({
            "Day": d,
            "P90": round(p90[d], 2),
            "P50": round(p50[d], 2),
            "P10": round(p10[d], 2),
        })
    df_chart = pd.DataFrame(chart_data)
    
    # 绘图
    base = alt.Chart(df_chart).encode(x=alt.X('Day:O', title='未来交易日'))
    
    area = base.mark_area(opacity=0.2, color='#4dabf7').encode(
        y=alt.Y('P10', title='价格预演 (USD)', scale=alt.Scale(zero=False)),
        y2='P90'
    )
    
    l90 = base.mark_line(color='#0ca678', strokeDash=[5,5]).encode(y='P90')
    l50 = base.mark_line(color='#228be6', size=3).encode(y='P50')
    l10 = base.mark_line(color='#d6336c', strokeDash=[5,5]).encode(y='P10')
    
    nearest = alt.selection_point(nearest=True, on='mouseover', fields=['Day'], empty=False)
    
    selectors = base.mark_rule(opacity=0).encode(
        x='Day:O'
    ).add_params(nearest)
    
    points = base.mark_circle(size=60, color="black").encode(
        y='P50',
        opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        tooltip=[
            alt.Tooltip('Day', title='T+'),
            alt.Tooltip('P90', title='P90 (High)', format='.2f'),
            alt.Tooltip('P50', title='P50 (Median)', format='.2f'),
            alt.Tooltip('P10', title='P10 (Low)', format='.2f')
        ]
    )
    
    st.altair_chart((area + l90 + l50 + l10 + selectors + points).properties(height=300).interactive(), use_container_width=True)
    st.caption(f"Engine: v9.0 Stable | Drift: {drift*100:.2f}% | Vol: {vol*100:.1f}%")

# --- 主程序入口 ---
st.markdown("### ⚡ BTDR 领航员 v9.0")
# 调用局部刷新函数
dashboard()

# --- 参数说明书 (静态部分，不刷新) ---
with st.expander("📖 参数详解手册 (看不懂点这里)"):
    st.markdown("""
    #### 🌍 宏观环境 (看天吃饭)
    * **QQQ (纳指)**: 科技股大盘风向标。如果 QQQ 大跌，BTDR 很难独善其身。    * **VIX (恐慌)**: 市场的恐惧指数。**数值越高越危险**。
        * VIX > 20: 市场紧张。
        * VIX > 30: 极度恐慌 (崩盘风险)。
    * **Beta (联动)**: BTDR 跟随大哥的程度。
        * Beta(BTC) = 1.5: BTC 涨 1%，BTDR 往往能涨 1.5%。
    
    #### 🔬 微观结构 (自身体质)
    * **VWAP (机构成本)**: 过去30天主力资金的平均成本线。
        * 价格 > VWAP: 强势 (但太高会有回调压力)。
        * 价格 < VWAP: 弱势 (但也可能是抄底机会)。
    * **ADX (趋势强度)**: 判断现在是“单边行情”还是“垃圾时间”。
        * ADX > 25 (Trend): 趋势明显，适合顺势操作。
        * ADX < 20 (Chop): 震荡市，适合高抛低吸。
    * **RSI (强弱指标)**: 是否涨过头了？
        * RSI > 70: 超买 (警惕回调)。
        * RSI < 30: 超卖 (可能反弹)。
    * **Implied Vol (波动率)**: 预期的震荡幅度。数值越大，预测的喇叭口张得越开。
    * **Exp. Drift (预期漂移)**: 模型综合所有因子后，算出来的**当日预期涨跌惯性**。
    """)
