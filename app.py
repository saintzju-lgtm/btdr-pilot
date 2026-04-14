import streamlit as st
import pandas as pd
import akshare as ak
import time
import datetime
import json
import os

# ================= 1. 基础配置 =================
st.set_page_config(page_title="A股实时监控看板", page_icon="📈", layout="wide")
CONFIG_FILE = "stocks_config.json"

# 默认截图中的核心股票
DEFAULT_STOCKS = {
    "601899": "紫金矿业",
    "002156": "通富微电",
    "300274": "阳光电源",
    "300124": "汇川技术",
    "601138": "工业富联"
}

# ================= 2. 核心功能函数 =================
def load_config():
    """从本地读取配置"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_STOCKS

def save_config(stocks_dict):
    """保存配置到本地"""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(stocks_dict, f, ensure_ascii=False, indent=4)

@st.cache_data(ttl=86400) # 缓存24小时！极大地提升刷新速度
def get_stock_list():
    """获取A股全市场股票列表用于模糊搜索"""
    try:
        df = ak.stock_info_a_code_name()
        df['display'] = df['code'] + " - " + df['name']
        return df['display'].tolist()
    except Exception:
        return []

@st.cache_data(ttl=10) # 行情数据缓存10秒，防止快速点击时卡顿或被封IP
def fetch_realtime_data(stock_map):
    """抓取选定股票的实时行情"""
    try:
        df_all = ak.stock_zh_a_spot_em()
        # 过滤并重命名列
        df_target = df_all[df_all['代码'].isin(stock_map.keys())].copy()
        df_target['名称'] = df_target['代码'].map(stock_map)
        df_target = df_target[['名称', '代码', '最新价', '涨跌幅']]
        df_target.columns = ['股票名称', '代码', '实时价格', '当日涨跌幅(%)']
        return df_target.reset_index(drop=True), True
    except Exception:
        return pd.DataFrame(), False

def generate_advice(pct_change):
    """生成操作建议"""
    if pct_change >= 3.0: return "⚠️ 涨幅较大，建议逢高分批止盈"
    elif 1.0 <= pct_change < 3.0: return "📈 走势良好，建议继续持有"
    elif -1.0 <= pct_change < 1.0: return "⏸️ 震荡整理，暂不操作观望"
    elif -3.0 < pct_change <= -1.0: return "📉 出现回调，注意下方支撑"
    else: return "💡 跌幅较大，观察企稳位置寻找低吸机会"

# ================= 3. 状态初始化 =================
if "stocks" not in st.session_state:
    st.session_state.stocks = load_config()

# ================= 4. 侧边栏 UI =================
st.sidebar.header("🕹️ 操作控制")

if st.sidebar.button("🔄 立即刷新数据", use_container_width=True):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("🛠️ 股票池配置")

# 4.1 管理当前股票
current_stocks = st.session_state.stocks.copy()
for code, name in list(current_stocks.items()):
    col1, col2 = st.sidebar.columns([3, 1])
    col1.text(f"{name} ({code})")
    if col2.button("删除", key=f"del_{code}"):
        del st.session_state.stocks[code]
        save_config(st.session_state.stocks)
        st.rerun()

st.sidebar.markdown("---")

# 4.2 下拉框模糊搜索添加股票
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
            # 拆分出代码和名称
            new_code, new_name = selected_stock.split(" - ", 1)
            
            if new_code in st.session_state.stocks:
                st.sidebar.warning(f"{new_name} 已在列表中！")
            else:
                st.session_state.stocks[new_code] = new_name
                save_config(st.session_state.stocks)
                st.rerun()
        else:
            st.sidebar.error("请先从下拉菜单中选择股票")
else:
    st.sidebar.error("基础股票列表加载中或网络异常...")

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("开启 1分钟 自动刷新", value=True)

# ================= 5. 主界面渲染 =================
st.title("📊 A股核心市场数据切片")

# 顶部时间与状态
current_time = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
st.info(f"实时快照 (Real-time Snapshot): {current_time} | 每 60 秒自动更新")

# 数据抓取与展示
placeholder = st.empty()
with placeholder.container():
    df, success = fetch_realtime_data(st.session_state.stocks)
    
    if success and not df.empty:
        df['后续持仓建议'] = df['当日涨跌幅(%)'].apply(generate_advice)
        
        # 格式化美化 (处理正负号和保留两位小数)
        df_display = df.copy()
        df_display['当日涨跌幅(%)'] = df_display['当日涨跌幅(%)'].apply(lambda x: f"{x:+.2f}%")
        df_display['实时价格'] = df_display['实时价格'].apply(lambda x: f"¥ {x:.2f}")

        # 使用 column_config 美化列宽
        st.dataframe(
            df_display, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "后续持仓建议": st.column_config.TextColumn(width="large")
            }
        )
    elif df.empty and success:
        st.warning("监控列表为空，请在左侧添加股票。")
    else:
        st.error("数据抓取异常，请点击左上角手动刷新或检查网络。")

# ================= 6. 自动刷新逻辑 =================
# 放在页面最后，确保先渲染完 UI 再进行阻塞挂起
if auto_refresh:
    time.sleep(60)
    st.rerun()
