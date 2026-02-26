import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 数据引擎 ---
@st.cache_data(ttl=60)
def get_btdr_final_data():
    ticker = "BTDR"
    hist = yf.download(ticker, period="60d", interval="1d")
    if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
    
    live_1m = yf.download(ticker, period="1d", interval="1m", prepost=True)
    if isinstance(live_1m.columns, pd.MultiIndex): live_1m.columns = live_1m.columns.get_level_values(0)
    
    float_shares = 118000000 
    hist['昨收'] = hist['Close'].shift(1)
    hist['今开比例'] = (hist['Open'] - hist['昨收']) / hist['昨收']
    hist['最高比例'] = (hist['High'] - hist['昨收']) / hist['昨收']
    hist['最低比例'] = (hist['Low'] - hist['昨收']) / hist['昨收']
    hist['换手率'] = (hist['Volume'] / float_shares) * 100
    hist['5日均值'] = hist['Close'].rolling(5).mean()
    
    fit_df = hist.dropna()
    X = fit_df[['今开比例']].values
    m_h = LinearRegression().fit(X, fit_df['最高比例'].values)
    m_l = LinearRegression().fit(X, fit_df['最低比例'].values)
    
    reg_params = {'slope_h': m_h.coef_[0], 'inter_h': m_h.intercept_, 'slope_l': m_l.coef_[0], 'inter_l': m_l.intercept_}
    return fit_df, live_1m, float_shares, reg_params

# --- 2. 界面显示 ---
st.set_page_config(layout="wide", page_title="BTDR 深度量化终端")
st.title("🏹 BTDR 深度形态量化终端")

try:
    hist_df, live_df, float_shares, reg = get_btdr_final_data()
    last_hist = hist_df.iloc[-1]
    curr_p = live_df['Close'].iloc[-1]
    
    live_df.index = live_df.index.tz_convert('America/New_York')
    regular_market = live_df.between_time('09:30', '16:00')
    today_open = regular_market['Open'].iloc[0] if not regular_market.empty else live_df['Open'].iloc[-1]
    today_open_ratio = (today_open - last_hist['Close']) / last_hist['Close']

    # 核心计算：中性/乐观/悲观
    p_h_mid = last_hist['Close'] * (1 + (reg['inter_h'] + reg['slope_h'] * today_open_ratio))
    p_l_mid = last_hist['Close'] * (1 + (reg['inter_l'] + reg['slope_l'] * today_open_ratio))
    today_turnover = (live_df['Volume'].sum() / float_shares) * 100

    # 场景识别
    def get_market_scene(p, h, l, vol):
        if p >= h * 1.005 and vol >= 10: return "乐观场景", "#00FF00", "价格突破统计边界，量能配合强劲。"
        elif p <= l * 0.995 and vol >= 10: return "悲观场景", "#FF4B4B", "价格跌破统计支撑，恐慌盘正在释放。"
        else: return "中性场景", "#1E90FF", "处于统计区间内，波动受回归线锚定。"

    s_name, s_color, s_desc = get_market_scene(curr_p, p_h_mid, p_l_mid, today_turnover)

    # --- 板块 1：场景预测位 (位置上移 & 恢复最高最低逻辑) ---
    st.markdown(f"""
        <div style="background-color: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border-left: 10px solid {s_color}; margin-bottom: 20px;">
            <h2 style="margin:0;">当前定位：<span style="color:{s_color};">{s_name}</span></h2>
            <p style="margin:5px 0 0 0; color:#888;">{s_desc}</p>
        </div>
    """, unsafe_allow_html=True)

    col_target, col_metric = st.columns([1.5, 1])
    with col_target:
        st.subheader("📍 不同场景股价预测目标")
        # 恢复最高股价、最低股价预测逻辑
        scenario_table = pd.DataFrame({
            "场景": ["中性场景 (回归)", "乐观场景 (+6%)", "悲观场景 (-6%)"],
            "最高股价预测": [p_h_mid, p_h_mid * 1.06, p_h_mid * 0.94],
            "最低股价预测": [p_l_mid, p_l_mid * 1.06, p_l_mid * 0.94]
        })
        st.table(scenario_table.style.format(precision=2))
    
    with col_metric:
        st.subheader("📊 实时状态")
        m1, m2 = st.columns(2)
        m1.metric("当前成交价", f"${curr_p:.2f}", f"{(curr_p/last_hist['Close']-1):.2%}")
        t_status = "red" if today_turnover >= 20 else "orange" if today_turnover >= 10 else "green"
        m2.markdown(f"**当日实时换手**\n### :{t_status}[{today_turnover:.2f}%]")

    st.divider()

    # --- 板块 2：深度形态解析 (结合股价与换手率) ---
    col_text, col_radar = st.columns([1.5, 1])

    with col_text:
        st.subheader("🤖 深度形态分析结论")
        analysis_points = []
        
        # 核心逻辑：股价位置 + 换手率结合
        if curr_p >= p_h_mid * 0.98:
            if today_turnover > 15:
                analysis_points.append(f"🔥 **高位放量压力**：股价触及预测上限 `${p_h_mid:.2f}` 且换手剧增。这通常是日内情绪的高点，若量能无法持续封板，需警惕冲高回落风险。")
            else:
                analysis_points.append(f"🟡 **缩量触压**：股价逼近压力位 `${p_h_mid:.2f}` 但动能衰减，大概率将转为高位横盘或小幅回调。")
        
        elif curr_p <= p_l_mid * 1.02:
            if today_turnover > 15:
                analysis_points.append(f"⚠️ **恐慌支撑考验**：股价跌至 `${p_l_mid:.2f}` 附近伴随恐慌盘放量。若此处换手能稳住，则是剧烈洗盘后的博弈点；若守不住，则形态走坏。")
            else:
                analysis_points.append(f"🟢 **缩量踩支**：缩量回踩预测底线 `${p_l_mid:.2f}`，属于健康的技术性回调，支撑力度较强。")

        # 趋势与MA5结合
        if curr_p > last_hist['5日均值']:
            analysis_points.append(f"📈 **多头趋势**：站稳MA5（${last_hist['5日均值']:.2f}）上方，重心持续上移。配合当前换手率，属于典型的趋势性持筹区间。")
        else:
            analysis_points.append(f"📉 **重心下移**：运行于MA5下方，每一轮反弹如无放量支持，预测最高价 `${p_h_mid:.2f}` 均构成强力反压。")

        for pt in analysis_points: st.markdown(f"> {pt}")

    with col_radar:
        st.subheader("🎯 实时评分雷达")
        mom = min(max(((curr_p / today_open - 1) + 0.05) / 0.1 * 100, 0), 100)
        trd = min(max(((curr_p / last_hist['5日均值'] - 1) + 0.05) / 0.1 * 100, 0), 100)
        trn = min((today_turnover / 20) * 100, 100)
        sup = min(max((1 - abs(curr_p - p_l_mid) / p_l_mid) * 100, 0), 100)
        
        radar_fig = go.Figure(data=go.Scatterpolar(
            r=[mom, sup, trn, trd], theta=['动能(MOM)', '支撑(SUP)', '换手(TRN)', '趋势(TRD)'],
            fill='toself', fillcolor=f'rgba{tuple(int(s_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.3,)}',
            line=dict(color=s_color, width=2)
        ))
        radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=300, margin=dict(l=40, r=40, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(radar_fig, use_container_width=True)
        
        # 雷达图解读
        st.caption(f"🔸 **动能**: {'强势爆发' if mom > 75 else '震荡平缓' if mom > 40 else '动能匮乏'}")
        st.caption(f"🔸 **支撑**: {'极度贴合统计底' if sup > 85 else '中立区域'}")
        st.caption(f"🔸 **换手**: {'主力博弈剧烈' if trn > 75 else '常规成交'}")

    # --- 板块 3：可视化图表 ---
    st.subheader("🕒 趋势监控 (垂直 MM/DD 标签)")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    plot_df = hist_df.tail(20).copy()
    plot_df['label'] = plot_df.index.strftime('%m/%d')

    fig.add_trace(go.Candlestick(x=plot_df['label'], open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name="日K"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df['label'], y=plot_df['5日均值'], name="MA5", line=dict(color='yellow')), row=1, col=1)
    fig.add_hline(y=p_h_mid, line_dash="dash", line_color="cyan", annotation_text="预测最高", row=1, col=1)
    fig.add_hline(y=p_l_mid, line_dash="dash", line_color="cyan", annotation_text="预测最低", row=1, col=1)
    fig.add_trace(go.Bar(x=plot_df['label'], y=plot_df['换手率'], name="换手率", marker_color=['red' if x >= 20 else 'orange' if x >= 10 else 'gray' for x in plot_df['换手率']]), row=2, col=1)
    fig.update_xaxes(tickangle=-90, dtick=1, row=1, col=1)
    fig.update_xaxes(tickangle=-90, dtick=1, row=2, col=1)
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"分析引擎刷新中: {e}")
