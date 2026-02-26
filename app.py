import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time

# --- 页面基本配置 ---
st.set_page_config(page_title="BTDR 实时监控与分析引擎", layout="wide", page_icon="📈")

# --- 定义自动刷新逻辑 (Streamlit 1.37+ 原生支持) ---
# 利用 st.fragment 实现局部每 60 秒自动刷新
@st.experimental_fragment(run_every=60)
def live_dashboard():
    # 获取实时数据 (过去15个交易日用于计算均线)
    ticker = yf.Ticker("BTDR")
    df = ticker.history(period="15d")
    
    if df.empty:
        st.error("数据获取失败，请检查网络或 API 状态。")
        return

    # --- 数据处理与计算 ---
    df['Prev Close'] = df['Close'].shift(1)
    df['5 MA'] = df['Close'].rolling(window=5).mean()
    df['Typical Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Turnover'] = df['Typical Price'] * df['Volume']
    
    # 提取最近一天的实时数据和前一天数据
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 按照用户要求构建表格字段
    outstanding_shares = 11800 # 预设流通股(万股)，与截图一致
    
    data = {
        "日期": df.index[-1].strftime('%Y/%m/%d'),
        "今开": latest['Open'],
        "最高": latest['High'],
        "最低": latest['Low'],
        "今收": latest['Close'], # 盘中为实时价格
        "昨收": latest['Prev Close'],
        "成交量(万股)": latest['Volume'] / 10000,
        "成交额(万)": latest['Turnover'] / 10000,
        "交易成本": latest['Typical Price'], # 采用典型价格近似日内VWAP
        "流通股(万股)": outstanding_shares,
        "换手率": (latest['Volume'] / 10000) / outstanding_shares,
        "最高比例": (latest['High'] - latest['Prev Close']) / latest['Prev Close'],
        "最低比例": (latest['Low'] - latest['Prev Close']) / latest['Prev Close'],
        "今开比例": (latest['Open'] - latest['Prev Close']) / latest['Prev Close'],
        "今收/昨收": (latest['Close'] - latest['Prev Close']) / latest['Prev Close'],
        "5日均值": latest['5 MA']
    }
    
    result_df = pd.DataFrame([data])
    
    # --- UI 顶部概览 ---
    st.title("📈 BTDR (Bitdeer) 实时量价监测与分析引擎")
    st.caption(f"最后更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (每60秒自动刷新)")
    
    price_change = data['今收'] - data['昨收']
    price_pct = data['今收/昨收'] * 100
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("最新价格 (今收)", f"${data['今收']:.2f}", f"{price_change:.2f} ({price_pct:.2f}%)")
    col2.metric("日内交易成本 (VWAP)", f"${data['交易成本']:.2f}")
    col3.metric("实时换手率", f"{data['换手率']*100:.2f}%")
    col4.metric("5日均价", f"${data['5日均值']:.2f}")
    
    # --- 格式化表格输出 ---
    st.subheader("📊 核心交易数据表 (实时)")
    
    # 设置显示格式
    format_dict = {
        '今开': '${:.2f}', '最高': '${:.2f}', '最低': '${:.2f}', '今收': '${:.2f}', '昨收': '${:.2f}',
        '成交量(万股)': '{:.2f}', '成交额(万)': '{:.2f}', '交易成本': '${:.2f}', 
        '换手率': '{:.2%}', '最高比例': '{:.2%}', '最低比例': '{:.2%}', 
        '今开比例': '{:.2%}', '今收/昨收': '{:.2%}', '5日均值': '${:.2f}'
    }
    
    styled_df = result_df.style.format(format_dict).applymap(
        lambda x: 'color: red' if pd.notnull(x) and isinstance(x, float) and x > 0 else ('color: green' if pd.notnull(x) and isinstance(x, float) and x < 0 else ''),
        subset=['最高比例', '最低比例', '今开比例', '今收/昨收']
    )
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # --- 动态形态分析与操作建议 ---
    st.divider()
    st.subheader("🧠 AI 形态深度分析 & 实时建议")
    
    # 动态逻辑判断
    trend_status = "多头掌控" if data['今收'] > data['5日均值'] else "空头承压"
    volume_status = "缩量" if data['成交量(万股)'] < (prev['Volume']/10000) else "放量"
    vwap_status = "站稳成本线" if data['今收'] >= data['交易成本'] else "受制于解套抛压"
    
    analysis_text = f"""
    ### 📈 形态与量价分析
    1. **趋势研判 ({trend_status})**：当前价格为 **${data['今收']:.2f}**，5日均线位于 **${data['5日均值']:.2f}**。{'股价站上5日线，短期反弹通道成型，多头开始掌控局部主动权。' if data['今收'] > data['5日均值'] else '股价低于5日线，形态处于弱势寻底阶段，需等待右侧企稳信号。'}
    2. **量价配合 ({volume_status})**：今日实时成交量为 **{data['成交量(万股)']:.2f}万股**，换手率为 **{data['换手率']*100:.2f}%**。相比昨日呈现{volume_status}状态。{('底部的缩量说明抛压出清，主力锁仓良好。' if volume_status == '缩量' else '放量说明多空分歧加大，有资金在积极换手。')}
    3. **日内博弈 ({vwap_status})**：今日盘中主力资金的平均成本约为 **${data['交易成本']:.2f}**。目前股价{vwap_status}。
    
    ### 🎯 操作建议
    * **📍 关键点位**：
        * **强支撑位**：**${data['最低']:.2f}** (日内低点) / **${data['5日均值']:.2f}** (5日线)
        * **强压力位**：**${data['交易成本']:.2f}** (日内均价线) / **${data['最高']:.2f}** (日内高点)
    
    * **💡 执行策略**：
        1. **持仓者**：{'建议持股待涨，依托5日线作为防守位。' if trend_status == '多头掌控' else '控制仓位，若跌破今日低点建议暂时规避风险。'}
        2. **空仓/加仓者**：日内最佳潜伏区在 **${max(data['最低'], data['5日均值'] - 0.2):.2f} - ${data['5日均值']:.2f}** 附近，切忌在冲高至 **${data['交易成本']:.2f}** 以上时追高。
        3. **日内T+0**：若冲高触及 **${data['最高']:.2f}** 附近出现滞涨，可做T减仓；回落至 **${data['交易成本']:.2f}** 下方寻找接回机会。
    """
    st.markdown(analysis_text)

# 运行主程序
if __name__ == "__main__":
    live_dashboard()
