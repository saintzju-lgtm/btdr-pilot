import streamlit as st
import pandas as pd
import akshare as ak
import time
import datetime
from datetime import timezone, timedelta
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

@st.cache_data(ttl=86400, show_spinner=False)
def get_stock_list():
    try:
        df = ak.stock_info_a_code_name()
        df['display'] = df['code'] + " - " + df['name']
        return df['display'].tolist()
    except Exception:
        return []

# ================= 2. 专业数据与技术形态计算 =================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_technical_analysis(stock_codes):
    """抓取历史K线并计算 MA20 和 MACD（加入防限流机制）"""
    tech_forms = {}
    # 核心优化1：只取最近100天的历史数据，极大减轻接口压力和传输时间
    start_date = (datetime.datetime.now() - datetime.timedelta(days=100)).strftime("%Y%m%d")
    
    for code in stock_codes:
        try:
            # 核心优化2：加入 0.2 秒的微小延迟，防止并发请求被东方财富防火墙拦截
            time.sleep(0.2) 
            
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, adjust="qfq")
            if df is None or df.empty or len(df) < 30:
                tech_forms[code] = "数据不足/停牌"
                continue
                
            # 计算 MA20
            df['MA20'] = df['收盘'].rolling(window=20).mean()
            
            # 计算 MACD
            exp1 = df['收盘'].ewm(span=12, adjust=False).mean()
            exp2 = df['收盘'].ewm(span=26, adjust=False).mean()
            df['DIF'] = exp1 - exp2
            df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
            df['MACD'] = 2 * (df['DIF'] - df['DEA'])
            
            # 取最近两天的指标来判断突破和交叉
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            form_labels = []
            
            # 1. MA20 判定
            if latest['收盘'] > latest['MA20'] and prev['收盘'] <= prev['MA20']:
                form_labels.append("🔥 强势突破 MA20")
            elif latest['收盘'] < latest['MA20'] and prev['收盘'] >= prev['MA20']:
                form_labels.append("⚠️ 跌破 MA20 支撑")
            elif latest['收盘'] > latest['MA20']:
                form_labels.append("🟢 站稳 MA20 之上")
            else:
                form_labels.append("🔴 受压于 MA20 之下")
                
            # 2. MACD 判定
            dif, dea = latest['DIF'], latest['DEA']
            prev_dif, prev_dea = prev['DIF'], prev['DEA']
            
            loc = "零上" if dif > 0 else "零下"
            if dif > dea and prev_dif <= prev_dea:
                form_labels.append(f"📈 MACD {loc}金叉")
            elif dif < dea and prev_dif >= prev_dea:
                form_labels.append(f"📉 MACD {loc}死叉")
            elif dif > dea:
                form_labels.append("☀️ MACD 多头排列")
            else:
                form_labels.append("🌧️ MACD 空头排列")
                
            tech_forms[code] = " | ".join(form_labels)
        except Exception as e:
            # 如果依然失败，输出具体的错误代码方便排查
            tech_forms[code] = "接口限流/失败"
            
    return tech_forms

@st.cache_data(ttl=10, show_spinner=False)
def fetch_realtime_data_pro(stock_map):
    """抓取全量深度数据并合并技术形态"""
    try:
        df_all = ak.stock_zh_a_spot_em()
        
        up_count = len(df_all[df_all['涨跌幅'] > 0])
        down_count = len(df_all[df_all['涨跌幅'] < 0])
        total_valid = up_count + down_count
        sentiment_ratio = up_count / total_valid if total_valid > 0 else 0.5
        
        if sentiment_ratio > 0.7: market_mood = "🔥 情绪高潮 (普涨)"
        elif sentiment_ratio > 0.5: market_mood = "☀️ 情绪偏暖 (多头占优)"
        elif sentiment_ratio > 0.3: market_mood = "☁️ 情绪分化 (震荡分化)"
        else: market_mood = "❄️ 情绪冰点 (普跌退潮)"

        df_target = df_all[df_all['代码'].isin(stock_map.keys())].copy()
        df_target['名称'] = df_target['代码'].map(stock_map)
        
        # 提取关键字段
        df_target = df_target[['名称', '代码', '最新价', '涨跌幅', '量比', '换手率', '振幅']]
        df_target.columns = ['股票名称', '代码', '实时价格', '涨跌幅(%)', '量比', '换手率(%)', '振幅(%)']
        
        # 聚合日线技术形态
        tech_dict = fetch_technical_analysis(list(stock_map.keys()))
        df_target['日线技术形态'] = df_target['代码'].map(tech_dict)
        
        return df_target.reset_index(drop=True), market_mood, sentiment_ratio, True
    except Exception:
        return pd.DataFrame(), "⚠️ 获取失败", 0.5, False

def generate_pro_advice(row, sentiment_ratio):
    """多因子操作建议（结合了技术形态）"""
    pct = row['涨跌幅(%)']
    vol_ratio = row['量比']
    tech_form = row.get('日线技术形态', '')
    
    advice = []
    
    if "突破 MA20" in tech_form: advice.append("日线级别右侧突破启动，")
    elif "跌破 MA20" in tech_form: advice.append("中线生命线失守，趋势转弱，")
        
    if pct >= 3.0:
        if vol_ratio > 1.5: advice.append("放量大涨。若筹码出现分歧可逢高定止盈；若封板或走势稳健则顺势持有。")
        else: advice.append("缩量上涨，追高意愿不足，谨防冲高回落。")
    elif 0 < pct < 3.0:
        if vol_ratio > 1.5: advice.append("放量滞涨，警惕上方抛压。")
        else: advice.append("温和震荡向上，按原有节奏持仓。")
    elif -2.0 <= pct <= 0:
        if vol_ratio < 0.8: advice.append("缩量整理，属良性调整，继续观察。")
        elif vol_ratio > 1.5: advice.append("放量微跌，资金博弈激烈，短线有破位隐患。")
        else: advice.append("弱势震荡，暂无明显方向。")
    else: 
        if vol_ratio > 1.5: advice.append("放量杀跌，严禁轻易抄底，严格执行止损纪律。")
        else: advice.append("缩量阴跌，情绪性杀跌为主，等待右侧企稳信号。")

    return "".join(advice)

# ================= 3. 状态初始化与侧边栏 =================
if "stocks" not in st.session_state:
    st.session_state.stocks = load_config()

st.sidebar.header("🕹️ 操作控制")
if st.sidebar.button("🔄 立即刷新数据", use_container_width=True):
    st.cache_data.clear() # 手动点击时清除一下缓存，强制拉取最新
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
st.title("📊 A股专业量化监控看板 (盘口+日线共振)")

# 设置局部刷新的时间间隔（如果关闭则为 None）
refresh_interval = 60 if auto_refresh else None

# 核心优化3：使用 @st.fragment 隔离刷新区域。这使得整个页面不再灰屏！
@st.fragment(run_every=refresh_interval)
def render_dashboard():
    df, market_mood, sentiment_ratio, success = fetch_realtime_data_pro(st.session_state.stocks)
    
    beijing_tz = timezone(timedelta(hours=8))
    current_time = datetime.datetime.now(beijing_tz).strftime("%Y.%m.%d %H:%M:%S")
    
    col1, col2 = st.columns([1, 1])
    col1.info(f"🕒 快照时间 (北京时间): {current_time} (无感静默更新)")
    col2.info(f"🧭 当前大盘情绪: {market_mood} (上涨率: {sentiment_ratio:.1%})")
    
    if success and not df.empty:
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
            df_display, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "日线技术形态": st.column_config.TextColumn(width="medium"),
                "量化交易内参": st.column_config.TextColumn(width="large"),
            }
        )
    elif df.empty and success:
        st.warning("监控列表为空，请在左侧添加股票。")
    else:
        st.error("数据抓取异常，请点击左侧手动刷新或检查网络。")

# 调用执行局部刷新容器
render_dashboard()
