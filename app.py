import streamlit as st
import pandas as pd
import akshare as ak
import time
import datetime
import json
import os

# ================= 配置文件管理 =================
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

# ================= 页面配置 =================
st.set_page_config(page_title="A股自定义监控看板", page_icon="📈", layout="wide")

if "stocks" not in st.session_state:
    st.session_state.stocks = load_config()

# ================= 侧边栏 =================
st.sidebar.header("🕹️ 操作控制")

# 1. 手动刷新按钮 (点击即触发全页重新运行)
if st.sidebar.button("🔄 立即刷新数据", use_container_width=True):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("🛠️ 股票池配置")

# 2. 修改与管理
current_stocks = st.session_state.stocks.copy()
for code, name in list(current_stocks.items()):
    col1, col2 = st.sidebar.columns([3, 1])
    col1.text(f"{name} ({code})")
    if col2.button("删除", key=f"del_{code}"):
        del st.session_state.stocks[code]
        save_config(st.session_state.stocks)
        st.rerun()

st.sidebar.markdown("---")
new_code = st.sidebar.text_input("股票代码")
new_name = st.sidebar.text_input("股票名称")
if st.sidebar.button("确认添加"):
    if new_code and new_name:
        st.session_state.stocks[new_code] = new_name
        save_config(st.session_state.stocks)
        st.rerun()

st.sidebar.markdown("---")
# 3. 自动刷新逻辑开关
auto_refresh = st.sidebar.checkbox("开启 1分钟 自动刷新", value=True)
if auto_refresh:
    st.sidebar.caption(f"下次自动刷新将在 60秒 后...")

# ================= 数据抓取逻辑 =================
@st.cache_data(ttl=30) # 缓存30秒，防止手动点击过快导致接口被封
def fetch_realtime_data(stock_map):
    try:
        df_all = ak.stock_zh_a_spot_em()
        df_target = df_all[df_all['代码'].isin(stock_map.keys())].copy()
        df_target['名称'] = df_target['代码'].map(stock_map)
        df_target = df_target[['名称', '代码', '最新价', '涨跌幅']]
        df_target.columns = ['股票名称', '代码', '实时价格', '当日涨跌幅(%)']
        return df_target.reset_index(drop=True), True
    except:
        return pd.DataFrame(), False

def generate_advice(pct_change):
    if pct_change >= 3.0: return "⚠️ 涨幅较大，建议逢高止盈"
    elif 1.0 <= pct_change < 3.0: return "📈 走势良好，建议持有"
    elif -1.0 <= pct_change < 1.0: return "⏸️ 震荡整理，观望为主"
    else: return "📉 走势偏弱，注意风控"

# ================= 主界面渲染 =================
st.title("📊 A股自定义实时监控看板")

# 顶部时间显示
current_time = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
st.info(f"数据快照时间: {current_time} (每60秒更新)")

placeholder = st.empty()

with placeholder.container():
    df, success = fetch_realtime_data(st.session_state.stocks)
    
    if success and not df.empty:
        df['后续持仓建议'] = df['当日涨跌幅(%)'].apply(generate_advice)
        
        # 格式化美化
        df_display = df.copy()
        df_display['当日涨跌幅(%)'] = df_display['当日涨跌幅(%)'].apply(lambda x: f"{x:+.2f}%")
        df_display['实时价格'] = df_display['实时价格'].apply(lambda x: f"¥{x:.2f}")

        st.dataframe(df_display, use_container_width=True, hide_index=True)
    elif df.empty and success:
        st.info("列表为空，请在侧边栏添加股票。")
    else:
        st.error("无法连接行情数据接口，请重试。")

# ================= 1分钟自动刷新逻辑 =================
if auto_refresh:
    time.sleep(60) # 等待 60 秒
    st.rerun()
