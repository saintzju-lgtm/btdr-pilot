import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# --- 1. 页面配置与样式 ---
st.set_page_config(page_title="BTDR 量化全功能终端", layout="wide")
st.markdown("""
    <style>
    .metric-card { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3d4251; }
    </style>
""", unsafe_allow_html=True)

# --- 2. 核心数据引擎 ---
@st.cache_data(ttl=60)
def get_integrated_data():
    ticker_symbol = "BTDR"
    # 获取历史数据用于回归 (60天)
    hist = yf.download(ticker_symbol, period="60d", interval="1d")
    if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
    
    # 获取今日 1m 逐分钟数据 (含盘前)
    today_1m = yf.download(ticker_symbol, period="1d", interval="1m", prepost=True)
    if isinstance(today_1m.columns, pd.MultiIndex): today_1m.columns = today_1m.columns.get_level_values(0)
    
    # 获取流通盘 (用于换手率)
    ticker_obj = yf.Ticker(ticker_symbol)
    float_shares = ticker_obj.info.get('floatShares', 35000000)
    prev_close = hist['Close'].iloc[-1]
    
    return hist, today_1m, float_shares, prev_close

# --- 3. 核心计算逻辑 ---
def process_logic(hist, today_1m, prev_close, float_shares):
    # (A) 回归模型训练 (功能1)
    df_reg = hist.copy()
    df_reg['Prev_Close'] = df_reg['Close'].shift(1)
    df_reg['Open_Ratio'] = (df_reg['Open'] - df_reg['Prev_Close']) / df_reg['Prev_Close']
    df_reg['Max_Ratio'] = (df_reg['High'] - df_reg['Prev_Close']) / df_reg['Prev_Close']
    df_reg['Min_Ratio'] = (df_reg['Low'] - df_reg['Prev_Close']) / df_reg['Prev_Close']
    df_reg = df_reg.dropna()
    
    X = df_reg[['Open_Ratio']].values
    model_h = LinearRegression().fit(X, df_reg['Max_Ratio'].values)
    model_l = LinearRegression().fit(X, df_reg['Min_Ratio'].values)
    
    # (B) 实时数据处理
    today_1m.index = today_1m.index.tz_convert('America/New_York')
    pre_market = today_1m.between_time('04:00', '09:29')
    regular_market = today_1m.between_time('09:30', '16:00')
    
    curr_p = today_1m['Close'].iloc[-1]
    # 确定今日开盘价 (若未开盘则用盘前最后价)
    today_open = regular_market['Open'].iloc[0] if not regular_market.empty else today_1m['Open'].iloc[-1]
    
    # (C) 波动范围预测
    o_ratio = (today_open - prev_close) / prev_close
    pred_h = prev_close * (1 + model_h.predict([[o_ratio]])[0])
    pred_l = prev_close * (1 + model_l.predict([[o_ratio]])[0])
    
    return pre_market, regular_market, curr_p, pred_h, pred_l, o_ratio

# --- 4. 界面渲染 ---
try:
    hist, today_1m, float_shares, prev_close = get_integrated_data()
    pre_df, reg_df, curr_p, p_high, p_low, o_ratio = process_logic(hist, today_1m, prev_close, float_shares)

    st.title("🏹 BTDR 实时量化交易终端")

    # --- 顶层核心指标 ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("当前实时价", f"${curr_p:.2f}", f"{(curr_p/prev_close-1):.2%}")
    m2.metric("开盘涨幅", f"{o_ratio:.2%}")
    m3.metric("预测最高 (压力)", f"${p_high:.2f}", "回归边界", delta_color="inverse")
    m4.metric("预测最低 (支撑)", f"${p_low:.2f}", "回归边界")

    st.divider()

    col_main, col_side = st.columns([2, 1])

    with col_main:
        # --- K线图与预测区间 ---
        st.subheader("🕯️ 实时走势与波动预测带")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # K线
        fig.add_trace(go.Candlestick(x=today_1m.index, open=today_1m['Open'], high=today_1m['High'],
                                     low=today_1m['Low'], close=today_1m['Close'], name="1m K线"), row=1, col=1)
        # 成交量
        fig.add_trace(go.Bar(x=today_1m.index, y=today_1m['Volume'], name="成交量", marker_color='gray', opacity=0.3), row=2, col=1)
        
        # 预测线 (红绿虚线)
        fig.add_hline(y=p_high, line_dash="dash", line_color="#FF4B4B", annotation_text="预测压力位", row=1, col=1)
        fig.add_hline(y=p_low, line_dash="dash", line_color="#00FF00", annotation_text="预测支撑位", row=1, col=1)
        
        fig.update_layout(xaxis_rangeslider_visible=False, height=600, template="plotly_dark", margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_side:
        # --- 策略决策引擎 ---
        st.subheader("🤖 决策矩阵")
        
        # 1. 换手率监控 (功能要求)
        total_vol = today_1m['Volume'].sum()
        turnover = (total_vol / float_shares) * 100
        
        if turnover >= 20:
            st.error(f"🔴 换手率红色预警: {turnover:.2f}% (极度过热)")
        elif turnover >= 10:
            st.warning(f"🟠 换手率黄色预警: {turnover:.2f}% (活跃度高)")
        else:
            st.success(f"🟢 换手率正常: {turnover:.2f}%")

        # 2. 盘前背离识别
        st.write("**实时形态分析：**")
        if not pre_df.empty and len(pre_df) > 10:
            p_move = pre_df['Close'].iloc[-1] - pre_df['Close'].iloc[0]
            v_move = pre_df['Volume'].iloc[-5:].mean() - pre_df['Volume'].iloc[:5].mean()
            if p_move > 0 and v_move < 0:
                st.markdown(":red[⚠️ 发现盘前【价涨量缩】背离，警惕诱多回落！]")

        # 3. 综合操作建议 (基于位置与形态)
        st.divider()
        st.markdown("### 💡 操作建议")
        
        advice = "持仓观望"
        reason = "价格处于波动中轴，未触及极值区间。"
        
        if curr_p >= p_high * 0.98:
            advice = "分批减仓"
            reason = "股价进入回归预测高位区间，且伴随换手率放大。"
        elif curr_p <= p_low * 1.02:
            advice = "博弈做多"
            reason = "触及预测支撑位，若未放量跌破可尝试抢反弹。"
        
        st.info(f"**建议动作：{advice}**")
        st.caption(f"理由：{reason}")

        # 4. K线形态微观分析
        last_body = today_1m['Close'].iloc[-1] - today_1m['Open'].iloc[-1]
        if (today_1m['High'].iloc[-1] - max(today_1m['Close'].iloc[-1], today_1m['Open'].iloc[-1])) > abs(last_body):
            st.write("📍 检测到长上影线：上方抛压较重")

    # 历史参考
    with st.expander("查看回归模型参考数据 (最近10日)"):
        st.table(hist.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume']])

except Exception as e:
    st.error(f"数据获取中或发生错误: {e}")
    st.info("提示：若在盘前时段，部分成交指标可能较小。")
