# [修改点 1]: 计算 Kalman 预测 (保持不变)
    pred_h_kalman = mh['intercept'] + (mh['beta_gap'] * current_gap_pct) + (mh['beta_btc'] * btc_pct_factor) + (mh['beta_vol'] * vol_state_factor)
    pred_l_kalman = ml['intercept'] + (ml['beta_gap'] * current_gap_pct) + (ml['beta_btc'] * btc_pct_factor) + (ml['beta_vol'] * vol_state_factor)
    
    # [修改点 2]: 改进 AI 波动逻辑 - 锁定锚点
    # 原逻辑: 基于当前价格 (price) 浮动 -> 导致追涨杀跌
    # 新逻辑: 基于开盘价 (open) 或 VWAP 加上 3 倍波动率 -> 形成真正的"压力墙"
    anchor_price = btdr['open'] if btdr['open'] > 0 else btdr['prev']
    # 如果开盘价不可用，使用 Prev Close。放大系数从 2.5 提升到 3.0 以捕捉极端行情
    ai_upper_bound_pct = (anchor_price * (1 + 3.0 * live_vol_pct) - btdr['prev']) / btdr['prev']
    ai_lower_bound_pct = (anchor_price * (1 - 3.0 * live_vol_pct) - btdr['prev']) / btdr['prev']
    
    # [修改点 3]: 调整权重 - 降低实时 AI 噪音，增加历史统计权重
    # 原权重: w_ai 0.5 太高
    w_kalman = 0.4; w_hist = 0.25; w_mom = 0.15; w_ai = 0.2 
    
    final_h_ret = (w_kalman * pred_h_kalman) + (w_hist * ai_model['ensemble_hist_h']) + (w_mom * ai_model['ensemble_mom_h']) + (w_ai * ai_upper_bound_pct)
    final_l_ret = (w_kalman * pred_l_kalman) + (w_hist * ai_model['ensemble_hist_l']) + (w_mom * ai_model['ensemble_mom_l']) + (w_ai * ai_lower_bound_pct)
    
    sentiment_adj = (fng_val - 50) * 0.0005
    final_h_ret += sentiment_adj; final_l_ret += sentiment_adj
    
    # 计算原始目标位
    raw_p_high = btdr['prev'] * (1 + final_h_ret)
    raw_p_low = btdr['prev'] * (1 + final_l_ret)

    # [修改点 4]: Session State 平滑滤波 - 彻底解决数字跳动
    if 'smooth_high' not in st.session_state: st.session_state.smooth_high = raw_p_high
    if 'smooth_low' not in st.session_state: st.session_state.smooth_low = raw_p_low

    # 仅当变化超过 0.5% 或为了平滑时更新 (EMA 滤波)
    # 0.95 的平滑系数意味着新数值只占 5% 的权重，数字会非常稳定
    smoothing_factor = 0.95
    st.session_state.smooth_high = (st.session_state.smooth_high * smoothing_factor) + (raw_p_high * (1 - smoothing_factor))
    st.session_state.smooth_low = (st.session_state.smooth_low * smoothing_factor) + (raw_p_low * (1 - smoothing_factor))

    # 赋值给最终显示变量
    p_high = st.session_state.smooth_high
    p_low = st.session_state.smooth_low
