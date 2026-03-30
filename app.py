import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from scipy.optimize import newton

# --- 1. 期权 IV 计算引擎 (Black-Scholes 反推) ---
def bs_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def find_iv(market_price, S, K, T, r, option_type='call'):
    if market_price <= 0: return 0
    try:
        # 使用牛顿迭代法求解 sigma
        func = lambda sigma: bs_price(S, K, T, r, sigma, option_type) - market_price
        return newton(func, 0.5, maxiter=100)
    except:
        return 0

def get_realtime_iv(ticker_obj, current_price):
    try:
        expirations = ticker_obj.options
        if not expirations: return 0.5, 1.0 # 默认值
        
        # 获取最近一个到期日的看涨期权
        opt_chain = ticker_obj.option_chain(expirations[0]).calls
        # 筛选平值期权 (ATM)
        atm_option = opt_chain.iloc[(opt_chain['strike'] - current_price).abs().argsort()[:1]]
        
        mkt_p = atm_option['lastPrice'].values[0]
        strike = atm_option['strike'].values[0]
        # 距离到期天数 (年化)
        days_to_expiry = (pd.to_datetime(expirations[0]) - pd.Timestamp.now()).days / 365.0
        
        iv = find_iv(mkt_p, current_price, strike, max(days_to_expiry, 0.001), 0.04)
        return iv
    except:
        return 0.45 # 兜底值

# --- 2. 指标增强：布林带 + MFI ---
def add_technical_indicators(df):
    # 布林带计算
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Std20'] = df['Close'].rolling(20).std()
    df['Upper'] = df['MA20'] + (df['Std20'] * 2)
    df['Lower'] = df['MA20'] - (df['Std20'] * 2)
    
    # MFI 计算 (保持原逻辑)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    rmf = tp * df['Volume']
    pos_flow = np.where(tp > tp.shift(1), rmf, 0)
    neg_flow = np.where(tp < tp.shift(1), rmf, 0)
    mfr = pd.Series(pos_flow).rolling(14).sum() / pd.Series(neg_flow).rolling(14).sum()
    df['MFI'] = 100 - (100 / (1 + mfr.values))
    return df

# --- 3. 简单策略回测模块 ---
def run_backtest(df, reg_params):
    """回测逻辑：当价格跌至回归预测的支撑位时买入，持有1日后的胜率"""
    df = df.copy()
    df['prev_close'] = df['Close'].shift(1)
    df['pred_low'] = df['prev_close'] * (1 + (reg_params['inter_l'] + reg_params['slope_l'] * ((df['Open']-df['prev_close'])/df['prev_close'])))
    
    # 信号：今日最低价触及预测支撑位
    df['signal'] = df['Low'] <= df['pred_low']
    # 结果：次日收盘相对于今日支撑位的涨幅
    df['next_ret'] = df['Close'].shift(-1) / df['Close'] - 1
    
    trades = df[df['signal'] == True].dropna()
    if len(trades) == 0: return 0, 0
    
    win_rate = (trades['next_ret'] > 0).sum() / len(trades)
    avg_ret = trades['next_ret'].mean()
    return win_rate, avg_ret

# --- 4. Streamlit UI 核心增强 ---
st.set_page_config(layout="wide", page_title="BTDR Quant Pro")

# 模拟原有的 get_btdr_quant_engine 逻辑并注入新功能
@st.cache_data(ttl=300)
def get_enhanced_data():
    tk = yf.Ticker("BTDR")
    hist = tk.history(period="100d")
    curr_p = hist['Close'].iloc[-1]
    
    # 1. 计算 IV
    real_iv = get_realtime_iv(tk, curr_p)
    
    # 2. 计算指标
    hist = add_technical_indicators(hist)
    
    # 3. 回归计算 (简化展示)
    hist['last_close'] = hist['Close'].shift(1)
    train = hist.dropna().tail(60)
    X = ((train['Open'] - train['last_close']) / train['last_close']).values.reshape(-1, 1)
    m_l = LinearRegression().fit(X, (train['Low'] / train['last_close'] - 1).values)
    reg_params = {'slope_l': m_l.coef_[0], 'inter_l': m_l.intercept_}
    
    return hist, real_iv, reg_params

try:
    df, iv_val, reg = get_enhanced_data()
    last_row = df.iloc[-1]
    
    # --- 布局 ---
    col1, col2, col3 = st.columns(3)
    col1.metric("实时反推 IV", f"{iv_val:.2%}")
    
    win_rate, avg_ret = run_backtest(df, reg)
    col2.metric("回归支撑位回测胜率", f"{win_rate:.1%}")
    col3.metric("信号平均收益", f"{avg_ret:.2%}")

    # --- 图表：布林带与 MFI 共振 ---
    st.subheader("📈 指标共振分析 (BB + MFI)")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    
    # 主图：K线 + 布林带
    plot_df = df.tail(40)
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Upper'], line=dict(color='rgba(173, 216, 230, 0.5)'), name="BB Upper"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Lower'], line=dict(color='rgba(173, 216, 230, 0.5)'), fill='tonexty', name="BB Lower"), row=1, col=1)
    
    # 副图：MFI
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MFI'], name="MFI", line=dict(color='orange')), row=2, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.update_layout(height=700, template="plotly_white", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- 决策逻辑判定 ---
    st.subheader("🛡️ 综合决策矩阵")
    logic_cols = st.columns(2)
    with logic_cols[0]:
        if last_row['Close'] < last_row['Lower'] and last_row['MFI'] < 30:
            st.success("🚨 多头共振：股价触及布林下轨且 MFI 超卖，反弹概率高。")
        elif last_row['Close'] > last_row['Upper'] and last_row['MFI'] > 70:
            st.warning("⚠️ 空头共振：股价触及布林上轨且 MFI 超买，谨慎追高。")
        else:
            st.info("📊 震荡区间：暂无极值信号，关注回归支撑位表现。")

except Exception as e:
    st.error(f"分析失败: {e}")
