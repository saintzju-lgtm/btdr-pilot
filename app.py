import streamlit as st
import pandas as pd
import akshare as ak
import time
import datetime
from datetime import timezone, timedelta
import json
import os
import random

# ================= 1. 基础配置与 CSS 魔法 =================
st.set_page_config(page_title="A股专业量化监控看板", page_icon="📈", layout="wide")
CONFIG_FILE = "stocks_config.json"

# 注入 CSS 强制消除右上角 Running 状态和灰屏动画，实现真正无感
st.markdown(
    """
    <style>
        /* 隐藏右上角的部署、Running 和 Stop 状态指示器 */
        [data-testid="stStatusWidget"] {
            display: none !important;
        }
        /* 隐藏顶部红色的加载进度条 */
        .stApp > header {
            display: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

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

@st.cache_data(ttl=86400, show_spinner=False)
def get_stock_list():
    try:
        df = ak.stock_info_a_code_name()
        df['display'] = df['code'] + " - " + df['name']
        return df['display'].tolist()
    except Exception:
        return []

# ================= 2. 专业数据与技术形态计算 =================
# 核心优化：单只股票独立缓存，且缓存12小时！因为日线级别一天只用算一次
@st.cache_data(ttl=43200, show_spinner=False) 
def fetch_single_stock_tech(code):
    """单独抓取单只股票的历史数据，最大程度防封禁"""
    start_date = (datetime.datetime.now() - datetime.timedelta(days=100)).strftime("%Y%m%d")
    
    for attempt in range(3):
        try:
            time.sleep(random.uniform(0.3, 0.8)) # 极短的随机休眠，模拟真人
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, adjust="qfq")
            
            if df is not None and not df.empty and len(df) >= 30:
                df['MA20'] = df['收盘'].rolling(window=20).mean()
                exp1 = df['收盘'].ewm(span=12, adjust=False).mean()
                exp2 = df['收盘'].ewm(span=26, adjust=False).mean()
                df['DIF'] = exp1 - exp2
                df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
                df['MACD'] = 2 * (df['DIF'] - df['DEA'])
                
                latest, prev = df.iloc[-1], df.iloc[-2]
                form_labels = []
                
                # MA20 判定
                if latest['收盘'] > latest['MA20'] and prev['收盘'] <= prev['MA20']:
                    form_labels.append("🔥 强势突破 MA20")
                elif latest['收盘'] < latest['MA20'] and prev['收盘'] >= prev['MA20']:
                    form_labels.append("⚠️ 跌破 MA20 支撑")
                elif latest['收盘'] > latest['MA20']:
                    form_labels.append("🟢 站稳 MA20 之上")
                else:
                    form_labels.append("🔴 受压于 MA20 之下")
                    
                # MACD 判定
                dif, dea, prev_dif, prev_dea = latest['DIF'], latest['DEA'], prev['DIF'], prev['DEA']
                loc = "零上" if dif > 0 else "零下"
                if dif > dea and prev_dif <= prev_dea:
                    form_labels.append(f"📈 MACD {loc}金叉")
                elif dif < dea and prev_dif >= prev_dea:
                    form_labels.append(f"📉 MACD {loc}死叉")
                elif dif > dea:
                    form_labels.append("☀️ MACD 多头排列")
                else:
                    form_labels.append("🌧️ MACD 空头排列")
                    
                return " | ".join(form_labels)
        except Exception:
            time.sleep(1)
            continue
            
    return None # 只有3次都失败才返回None，交由外层处理

@st.cache_data(ttl=10, show_spinner=False) # 盘口实时数据缓存 10 秒
def fetch_realtime_data_pro(stock_map):
    """只抓取快照数据，速度极快"""
    try:
        df_all = ak.stock_zh_a_spot_em()
        
        up_count = len(df_all[df_all['涨跌幅'] > 0])
        total_valid = up_count + len(df_all[df_all['涨跌幅'] < 0])
        sentiment_ratio = up_count / total_valid if total_valid > 0 else 0.5
        
        if sentiment_ratio > 0.7: market_mood = "🔥 情绪高潮 (普涨)"
        elif sentiment_ratio > 0.5: market_mood = "☀️ 情绪偏暖 (多头占优)"
        elif sentiment_ratio > 0.3: market_mood = "☁️ 情绪分化 (震荡分化)"
        else: market_mood = "❄️ 情绪冰点 (普跌退潮)"

        df_target = df_all[df_all['代码'].isin(stock_map.keys())].copy()
        df_target['名称'] = df_target['代码'].map(stock_map)
        df_target = df_target[['名称', '代码', '最新价', '涨跌幅', '量比', '换手率', '振幅']]
        df_target.columns = ['股票名称', '代码', '实时价格', '涨跌幅(%)', '量比', '换手率(%)', '振幅(%)']
        
        return df_target.reset_index(drop=True), market_mood, sentiment_ratio, True
    except Exception:
        return pd.DataFrame(), "⚠️ 获取失败", 0.5, False

def generate_pro_advice(row, sentiment_ratio):
    pct, vol_ratio, tech_form = row['涨跌幅(%)'], row['量比'], row.get('日线技术形态', '')
    advice = []
    if "突破 MA20" in tech_form: advice.append("日线级别突破，")
    elif "跌破 MA20" in tech_form: advice.append("中线生命线失守，")
        
    if pct >= 3.0:
        if vol_ratio > 1.5: advice.append("放量大涨，若筹码分歧可逢高定止盈。")
        else: advice.append("缩量上涨，追高意愿不足，防冲高回落。")
    elif 0 < pct < 3.0:
        if vol_ratio > 1.5: advice.append("放量滞涨，警惕抛压。")
        else: advice.append("温和震荡，按原有节奏持仓。")
    elif -2.0 <= pct <= 0:
        if vol_ratio < 0.8: advice.append("缩量整理，良性调整。")
        elif vol_ratio > 1.5: advice.append("放量微跌，短线有破位隐患。")
        else: advice.append("弱势震荡，多看少动。")
    else: 
        if vol_ratio > 1.5: advice.append("放量杀跌，严禁抄底。")
        else: advice.append("情绪杀跌，等待右侧企稳信号。")
    return "".join(advice)

# ================= 3. 状态初始化与侧边栏 =================
if "stocks" not in st.session_state: st.session_state.stocks = load_config()
if "tech_cache" not in st.session_state: st.session_state.tech_cache = {} # 记忆兜底池

st.sidebar.header("🕹️ 操作控制")
if st.sidebar.button("🔄 立即刷新实时数据", width="stretch"):
    # 绝对不能再清空 cache_data 缓存了！只触发 rerun 即可瞬间拉取最新快照
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("🛠️ 股票池配置")

current_stocks = st.session_state.stocks.copy()
for code, name in list(current_stocks.items()):
    col1, col2 = st.sidebar.columns([3, 1])
    col1.text(f"{name} ({code})")
    if col2.button("删除", key=f"del_{code}"):
        del st.session_state.stocks[code]
        if code in st.session_state.tech_cache: del st.session_state.tech_cache[code]
        save_config(st.session_state.stocks)
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("添加新股票")
all_stocks_list = get_stock_list()

if all_stocks_list:
    selected_stock = st.sidebar.selectbox("搜索股票", options=all_stocks_list, index=None, placeholder="例如: 600325")
    if st.sidebar.button("确认添加"):
        if selected_stock:
            new_code, new_name = selected_stock.split(" - ", 1)
            if new_code not in st.session_state.stocks:
                st.session_state.stocks[new_code] = new_name
                save_config(st.session_state.stocks)
                st.rerun()
            else:
                st.sidebar.warning(f"{new_name} 已在列表中！")

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("开启 1分钟自动无感刷新", value=True)

# ================= 4. 主界面渲染 =================
st.title("📊 A股专业量化监控看板 (盘口+日线共振)")

refresh_interval = 60 if auto_refresh else None

@st.fragment(run_every=refresh_interval)
def render_dashboard():
    # 1. 组装历史技术形态 (带兜底记忆)
    for code in st.session_state.stocks.keys():
        tech_result = fetch_single_stock_tech(code)
        if tech_result is not None:
            st.session_state.tech_cache[code] = tech_result # 更新最新记忆
        elif code not in st.session_state.tech_cache:
            st.session_state.tech_cache[code] = "⏳ 数据采集中" # 只有第一次死活抓不到才显示这个

    # 2. 抓取实时盘口数据
    df, market_mood, sentiment_ratio, success = fetch_realtime_data_pro(st.session_state.stocks)
    
    beijing_tz = timezone(timedelta(hours=8))
    current_time = datetime.datetime.now(beijing_tz).strftime("%Y.%m.%d %H:%M:%S")
    
    col1, col2 = st.columns([1, 1])
    col1.info(f"🕒 快照时间 (北京时间): {current_time} (无感静默更新)")
    col2.info(f"🧭 当前大盘情绪: {market_mood} (上涨率: {sentiment_ratio:.1%})")
    
    if success and not df.empty:
        # 将记忆中的技术形态映射给表格
        df['日线技术形态'] = df['代码'].map(st.session_state.tech_cache)
        
        cols = df.columns.tolist()
        cols.insert(4, cols.pop(cols.index('日线技术形态')))
        df = df[cols]
        
        df['量化交易内参'] = df.apply(lambda row: generate_pro_advice(row, sentiment_ratio), axis=1)
        
        df_display = df.copy()
        df_display['涨跌幅(%)'] = df_display['涨跌幅(%)'].apply(lambda x: f"{x:+.2f}%")
        df_display['换手率(%)'] = df_display['换手率(%)'].apply(lambda x: f"{x:.2f}%")
        df_display['振幅(%)'] = df_display['振幅(%)'].apply(lambda x: f"{x:.2f}%")
        df_display['量比'] = df_display['量比'].apply(lambda x: f"{x:.2f}")
        df_display['实时价格'] = df_display['实时价格'].apply(lambda x: f"¥ {x:.2f}")

        st.dataframe(
            df_display, width="stretch", hide_index=True,
            column_config={
                "日线技术形态": st.column_config.TextColumn(width="medium"),
                "量化交易内参": st.column_config.TextColumn(width="large"),
            }
        )
    elif df.empty and success:
        st.warning("监控列表为空，请在左侧添加股票。")
    else:
        st.error("实时盘口抓取失败，正在尝试重连...")

render_dashboard()
