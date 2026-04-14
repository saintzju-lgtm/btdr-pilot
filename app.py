import streamlit as st
import pandas as pd
import akshare as ak
import time
import datetime
import random

# ================= 配置页面 =================
st.set_page_config(page_title="A股核心市场数据切片", page_icon="📊", layout="wide")

# ================= 目标股票池 =================
# 对应截图中的股票名称和代码
TARGET_STOCKS = {
    "601899": "紫金矿业",
    "002156": "通富微电",
    "300274": "阳光电源",
    "300124": "汇川技术",
    "601138": "工业富联"
}

# ================= 核心功能函数 =================
@st.cache_data(ttl=5) # 设置5秒缓存，防止极端情况下的接口风暴
def fetch_realtime_data():
    """使用 akshare 获取 A股 实时数据"""
    try:
        # 获取东方财富 A股 实时行情
        df_all = ak.stock_zh_a_spot_em()
        
        # 过滤出我们需要的目标股票
        df_target = df_all[df_all['代码'].isin(TARGET_STOCKS.keys())].copy()
        
        # 提取需要的列并重命名
        df_target = df_target[['名称', '代码', '最新价', '涨跌幅']]
        df_target.columns = ['股票名称', '代码', '实时价格', '当日涨跌幅(%)']
        
        # 按照自定义顺序排序（可选）
        df_target['sort_idx'] = df_target['代码'].map({k: i for i, k in enumerate(TARGET_STOCKS.keys())})
        df_target = df_target.sort_values('sort_idx').drop('sort_idx', axis=1).reset_index(drop=True)
        
        return df_target, True
    except Exception as e:
        # 如果接口请求失败，生成模拟数据进行展示兜底
        mock_data = []
        for code, name in TARGET_STOCKS.items():
            price = random.uniform(30.0, 150.0)
            pct_change = random.uniform(-5.0, 5.0)
            mock_data.append([name, code, round(price, 2), round(pct_change, 2)])
        
        df_mock = pd.DataFrame(mock_data, columns=['股票名称', '代码', '实时价格', '当日涨跌幅(%)'])
        return df_mock, False

def generate_advice(pct_change):
    """基于涨跌幅生成简单的持仓建议"""
    if pct_change >= 3.0:
        return "⚠️ 涨幅较大，建议逢高分批止盈，锁定利润"
    elif 1.0 <= pct_change < 3.0:
        return "📈 走势良好，建议继续持有观望"
    elif -1.0 <= pct_change < 1.0:
        return "⏸️ 震荡整理，暂不操作，观察为主"
    elif -3.0 < pct_change <= -1.0:
        return "📉 出现回调，注意下方支撑位风险"
    else:
        return "💡 跌幅较大，不宜盲目杀跌，可寻找企稳位置低吸"

# ================= UI 布局与渲染 =================
st.title("📊 A股核心市场数据切片及操作建议")

# 侧边栏控制刷新
st.sidebar.header("控制面板")
auto_refresh = st.sidebar.checkbox("开启 10s 自动刷新", value=True)
st.sidebar.caption("提示：关闭复选框可暂停自动刷新。")

# 占位符，用于每次刷新时覆盖原有内容
placeholder = st.empty()

with placeholder.container():
    # 获取数据
    df, is_real = fetch_realtime_data()
    
    # 顶部时间及状态提示
    current_time = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    st.subheader(f"Real-time Snapshot: {current_time}")
    
    if not is_real:
        st.warning("⚠️ 网络或 API 异常，当前展示为本地模拟兜底数据。")
    
    # 计算并添加建议列
    df['持仓建议'] = df['当日涨跌幅(%)'].apply(generate_advice)
    
    # 格式化显示（添加 % 符号和保留小数）
    df_display = df.copy()
    df_display['当日涨跌幅(%)'] = df_display['当日涨跌幅(%)'].apply(lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%")
    df_display['实时价格'] = df_display['实时价格'].apply(lambda x: f"¥ {x:.2f}")

    # 使用 Markdown 表格或 st.dataframe 进行精美展示
    st.dataframe(
        df_display, 
        use_container_width=True,
        hide_index=True,
        column_config={
            "股票名称": st.column_config.TextColumn("股票名称", width="medium"),
            "代码": st.column_config.TextColumn("代码", width="medium"),
            "实时价格": st.column_config.TextColumn("实时价格", width="medium"),
            "当日涨跌幅(%)": st.column_config.TextColumn("当日涨跌幅", width="medium"),
            "持仓建议": st.column_config.TextColumn("后续持仓建议", width="large")
        }
    )

    st.markdown("---")
    st.caption("免责声明：本工具生成的持仓建议仅基于设定的简单涨跌幅阈值生成，不构成任何实际投资建议。股市有风险，投资需谨慎。")

# ================= 自动刷新逻辑 =================
if auto_refresh:
    time.sleep(10)
    st.rerun()
