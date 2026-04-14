import streamlit as st
import pandas as pd
import akshare as ak
import time
import datetime
import json
import os

# ================= 1. 基础配置 =================
st.set_page_config(page_title="A股专业量化监控看板", page_icon="📈", layout="wide")
CONFIG_FILE = "stocks_config.json"

DEFAULT_STOCKS = {
    "601899": "紫金矿业",
    "002156": "通富微电",
    "300274": "阳光电源",
    "300124": "汇川技术",
    "601138": "工业富联"
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_STOCKS

def save_config(stocks_dict):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(stocks_dict, f, ensure_ascii=False, indent=4)

@st.cache_data(ttl=86400)
def get_stock_list():
    try:
        df = ak.stock_info_a_code_name()
        df['display'] = df['code'] + " - " + df['name']
        return df['display'].tolist()
    except Exception:
        return []

# ================= 2. 专业数据与策略逻辑 =================
@st.cache_data(ttl=10)
def fetch_realtime_data_pro(stock_map):
    """抓取全量深度数据，并计算大盘情绪"""
    try:
        df_all = ak.stock_zh_a_spot_em()
        
        # 1. 计算大盘情绪（上涨家数 vs 下跌家数）
        up_count = len(df_all[df_all['涨跌幅'] > 0])
        down_count = len(df_all[df_all['涨跌幅'] < 0])
        total_valid = up_count + down_count
        sentiment_ratio = up_count / total_valid if total_valid > 0 else 0.5
        
        # 情绪打分标签
        if sentiment_ratio > 0.7: market_mood = "🔥 情绪高潮 (普涨)"
        elif sentiment_ratio > 0.5: market_mood = "☀️ 情绪偏暖 (多头占优)"
        elif sentiment_ratio > 0.3: market_mood = "☁️ 情绪分化 (震荡分化)"
        else: market_mood = "❄️ 情绪冰点 (普跌退潮)"

        # 2. 过滤目标股票并提取专业指标
        df_target = df_all[df_all['代码'].isin(stock_map.keys())].copy()
        df_target['名称'] = df_target['代码'].map(stock_map)
        
        # 保留更多量价特征列
        df_target = df_target[['名称', '代码', '最新价', '涨跌幅', '量比', '换手率', '振幅']]
        df_target.columns = ['股票名称', '代码', '实时价格', '涨跌幅(%)', '量比', '换手率(%)', '振幅(%)']
        
        return df_target.reset_index(drop=True), market_mood, sentiment_ratio, True
    except Exception:
        return pd.DataFrame(), "⚠️ 获取失败", 0.5, False

def generate_pro_advice(row, sentiment_ratio):
    """基于多因子（量价+情绪+振幅）生成专业操作建议"""
    pct = row['涨跌幅(%)']
    vol_ratio = row['量比']
    turnover = row['换手率(%)']
    amplitude = row['振幅(%)']
    
    advice = []
    
    # 1. 大盘情绪环境判定
    if sentiment_ratio < 0.3 and pct > 0:
        advice.append("逆势抗跌，")
    elif sentiment_ratio > 0.7 and pct < 0:
        advice.append("逆势走弱，")
        
    # 2. 量价形态核心判定
    if pct >= 3.0:
        if vol_ratio > 1.5:
            advice.append("放量大涨，资金介入明显。")
            if turnover > 10: advice.append("但高换手提示筹码分歧加大，切勿无脑追高，持筹者可沿均线定止盈。")
            else: advice.append("筹码锁定良好，建议顺势持有。")
        elif vol_ratio < 0.8:
            advice.append("缩量上涨，抛压较小但追买意愿不足，注意提防冲高回落风险。")
        else:
            advice.append("量价配合健康，趋势良好，继续持股。")
            
    elif 0 < pct < 3.0:
        if amplitude > 5:
            advice.append("宽幅震荡收红，日内洗盘/分歧加剧，建议观察企稳情况。")
        elif vol_ratio > 1.5:
            advice.append("放量滞涨，警惕上方抛压，可能面临变盘。")
        else:
            advice.append("温和震荡向上，按原有节奏持仓观望。")
            
    elif -2.0 <= pct <= 0:
        if vol_ratio < 0.8:
            advice.append("缩量整理，属于良性调整，重点关注下方核心均线支撑。")
        elif vol_ratio > 1.5:
            advice.append("放量微跌，资金博弈激烈，短线有向下破位隐患，注意风控。")
        else:
            advice.append("弱势震荡，暂无明显方向，多看少动。")
            
    else: # pct < -2.0
        if vol_ratio > 1.5:
            advice.append("放量杀跌，机构/大户出逃明显，**严禁轻易抄底**，坚决执行止损纪律。")
        else:
            if amplitude > 6:
                advice.append("缩量宽幅下跌，情绪性杀跌为主，可能存在错杀，等待右侧止跌企稳信号。")
            else:
                advice.append("阴跌走势，趋势破位风险大，建议逢反弹减仓。")

    return "".join(advice)

# ================= 3. 状态初始化与侧边栏 =================
if "stocks" not in st.session_state:
    st.session_state.stocks = load_config()

st.sidebar.header("🕹️ 操作控制")
if st.sidebar.button("🔄 立即刷新数据", use_container_width=True):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("🛠️ 股票池配置")

current_stocks = st.session_state.stocks.copy()
for code, name in list(current_stocks.items()):
    col1, col2 = st.sidebar.columns([3, 1])
    col1.text(f"{name} ({code})")
    if col2.button("删除", key=f"del_{code}"):
        del st.session_state.stocks[code]
        save_config(st.session_state.stocks)
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("添加新股票")
all_stocks_list = get_stock_list()

if all_stocks_list:
    selected_stock = st.sidebar.selectbox(
        "搜索股票 (支持输入代码或名称)", 
        options=all_stocks_list, 
        index=None,
        placeholder="例如: 600325 或 宏和..."
    )

    if st.sidebar.button("确认添加"):
        if selected_stock:
            new_code, new_name = selected_stock.split(" - ", 1)
            if new_code in st.session_state.stocks:
                st.sidebar.warning(f"{new_name} 已在列表中！")
            else:
                st.session_state.stocks[new_code] = new_name
                save_config(st.session_state.stocks)
                st.rerun()
        else:
            st.sidebar.error("请先从下拉菜单中选择股票")

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("开启 1分钟 自动刷新", value=True)

# ================= 4. 主界面渲染 =================
st.title("📊 A股专业日内量化监控看板")

placeholder = st.empty()
with placeholder.container():
    df, market_mood, sentiment_ratio, success = fetch_realtime_data_pro(st.session_state.stocks)
    
    current_time = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    
    # 顶部状态栏：展示时间和实时大盘情绪
    col1, col2 = st.columns([1, 1])
    col1.info(f"🕒 快照时间: {current_time} (60秒更新)")
    col2.info(f"🧭 当前大盘情绪: {market_mood} (上涨率: {sentiment_ratio:.1%})")
    
    if success and not df.empty:
        # 生成专业分析建议
        df['量化交易内参'] = df.apply(lambda row: generate_pro_advice(row, sentiment_ratio), axis=1)
        
        # 格式化美化
        df_display = df.copy()
        df_display['涨跌幅(%)'] = df_display['涨跌幅(%)'].apply(lambda x: f"{x:+.2f}%")
        df_display['换手率(%)'] = df_display['换手率(%)'].apply(lambda x: f"{x:.2f}%")
        df_display['振幅(%)'] = df_display['振幅(%)'].apply(lambda x: f"{x:.2f}%")
        df_display['量比'] = df_display['量比'].apply(lambda x: f"{x:.2f}")
        df_display['实时价格'] = df_display['实时价格'].apply(lambda x: f"¥ {x:.2f}")

        # 使用高亮逻辑：给量比等核心指标加颜色提醒
        st.dataframe(
            df_display, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "量化交易内参": st.column_config.TextColumn(width="large"),
            }
        )
    elif df.empty and success:
        st.warning("监控列表为空，请在左侧添加股票。")
    else:
        st.error("数据抓取异常，请点击左上角手动刷新或检查网络。")

# ================= 5. 自动刷新逻辑 =================
if auto_refresh:
    time.sleep(60)
    st.rerun()
