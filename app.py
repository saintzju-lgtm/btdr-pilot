import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 动态数据与回归引擎 ---
@st.cache_data(ttl=60)
def get_refreshed_data():
    ticker = "BTDR"
    # 获取 60 天日线用于实时拟合刷新
    hist = yf.download(ticker, period="60d", interval="1d")
    if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
    
    live_1m = yf.download(ticker, period="1d", interval="1m", prepost=True)
    if isinstance(live_1m.columns, pd.MultiIndex): live_1m.columns = live_1m.columns.get_level_values(0)
    
    float_shares = 118000000 
    df = hist.copy()
    df['昨收'] = df['Close'].shift(1)
    df['今开比例'] = (df['Open'] - df['昨收']) / df['昨收']
    df['最高比例'] = (df['High'] - df['昨收']) / df['昨收']
    df['最低比例'] = (df['Low'] - df['昨收']) / df['昨收']
    df['换手率'] = (df['Volume'] / float_shares) * 100
    df['5日均值'] = df['Close'].rolling(5).mean()
    df = df.dropna()

    # --- 实时执行线性回归拟合 ---
    X = df[['今开比例']].values
    # 动态训练最高比例模型
    model_h = LinearRegression().fit(X, df['最高比例'].values)
    # 动态训练最低比例模型
    model_l = LinearRegression().fit(X, df['最低比例'].values)
    
    # 提取动态系数
    slopes = {"h": model_h.coef_[0], "l": model_l.coef_[0]}
    intercepts = {"h": model_h.intercept_, "l": model_l.intercept_}
    r_squared = {"h": model_h.score(X, df['最高比例'].values), "l": model_l.score(X, df['最低比例'].values)}
    
    return df, live_1m, slopes, intercepts, r_squared

# --- 2. 界面与逻辑 ---
try:
    hist_df, live_df, slope, inter, r2 = get_refreshed_data()
    last_row = hist_df.iloc[-1]
    
    # 价格与开盘逻辑
    curr_p = live_df['Close'].iloc[-1]
    live_df.index = live_df.index.tz_convert('America/New_York')
    today_open = live_df.between_time('09:30', '16:00')['Open'].iloc[0] if not live_df.between_time('09:30', '16:00').empty else live_df['Open'].iloc[-1]
    today_open_ratio = (today_open - last_row['Close']) / last_row['Close']

    # 基于动态刷新的公式进行场景推算
    def calc_scenarios(o_ratio, base_p, s_h, i_h, s_l, i_l):
        mid_h_r = i_h + s_h * o_ratio
        mid_l_r = i_l + s_l * o_ratio
        mid_h = base_p * (1 + mid_h_r)
        mid_l = base_p * (1 + mid_l_r)
        return mid_h, mid_l, mid_h_r, mid_l_r

    p_h_mid, p_l_mid, h_r, l_r = calc_scenarios(today_open_ratio, last_row['Close'], slope['h'], inter['h'], slope['l'], inter['l'])

    # --- UI 渲染 (保持原布局) ---
    st.title("🏹 BTDR 动态拟合交易决策终端")
    
    # 指标卡
    c1, c2, c3 = st.columns(3)
    c1.metric("当前价", f"${curr_p:.2f}", f"{(curr_p/last_hist['Close']-1):.2%}" if 'last_hist' in locals() else None)
    c2.metric("动态压力 (中性)", f"${p_h_mid:.2f}")
    c3.metric("动态支撑 (中性)", f"${p_l_mid:.2f}")

    # 主图 K 线 (略，保持之前代码一致)

    # --- 底部数据表 (格式刷新) ---
    st.subheader("📋 历史数据明细 (最近10日)")
    show_df = hist_df.tail(10).copy()
    fmt_cols = {'换手率': '{:.2f}%', '今开比例': '{:.2%}', '最高比例': '{:.2%}', '最低比例': '{:.2%}'}
    for col, fmt in fmt_cols.items():
        show_df[col] = show_df[col].map(fmt.format)
    st.dataframe(show_df[['Open', 'High', 'Low', 'Close', '换手率', '今开比例', '最高比例', '最低比例', '5日均值']])

    # --- 场景拟合板块 (额外增加，刷新逻辑) ---
    st.divider()
    st.subheader("📈 实时自动刷新：拟合推算模型")
    
    l_col, r_col = st.columns([1, 2])
    with l_col:
        st.write("**当前动态回归方程：**")
        st.latex(f"High\_R = {inter['h']:.5f} + {slope['h']:.3f} \times Open\_R")
        st.latex(f"Low\_R = {inter['l']:.5f} + {slope['l']:.3f} \times Open\_R")
        st.caption(f"最高比例 R²: {r2['h']:.3f} | 最低比例 R²: {r2['l']:.3f}")

    with r_col:
        # 按照场景表逻辑输出
        sc_data = {
            "场景": ["中性场景", "乐观场景 (+6%)", "悲观场景 (-6%)"],
            "最高股价预测 (推算)": [
                f"{p_h_mid:.2f} (中值)", f"{p_h_mid * 1.06:.2f}", f"{p_h_mid * 0.94:.2f}"
            ],
            "最低股价预测 (推算)": [
                f"{p_l_mid:.2f} (中值)", f"{p_l_mid * 1.06:.2f}", f"{p_l_mid * 0.94:.2f}"
            ]
        }
        st.table(pd.DataFrame(sc_data))

except Exception as e:
    st.error(f"分析引擎刷新中... {e}")
