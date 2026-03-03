def get_ai_analysis_and_scores(data_summary):
    if "DEEPSEEK_API_KEY" not in st.secrets:
        return "⚠️ 未配置 Key", [50, 50, 50, 50]
    
    # 新增计算字段：偏离度
    dev_sup = ((data_summary['curr_p'] / data_summary['p_l_mid']) - 1) * 100
    
    try:
        client = OpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
        prompt = f"""
        你是一名严格的量化策略主理人。请审计 BTDR 实时盘面并给出硬核指令：
        
        【核心数据】
        - 实时成交: ${data_summary['curr_p']:.2f} (MA5: ${data_summary['ma5']:.2f})
        - 支撑位偏离度: {dev_sup:.2f}% (正值代表在支撑位上方，负值代表已跌破)
        - 换手状态: {data_summary['turnover']}
        
        【强制任务】
        1. 空间审计：当前价在回归模型的哪个位置？属于“安全区”还是“高压区”？
        2. 异常预警：换手率与价格走势是否背离？
        3. 战术指令：必须给出明确的操作建议（如：当前位分批低吸，止损设在$8.35）。
        4. 评分输出：SCORES: 动能=X, 支撑=X, 换手=X, 趋势=X
        """
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": "你只说真话，不带情绪，逻辑严密，回答字数控制在200字内。"}, {"role": "user", "content": prompt}]
        )
        full_text = response.choices[0].message.content
        
        # 提取分值（保持逻辑不变）
        scores = [50, 50, 50, 50]
        score_line = re.findall(r"SCORES:.*", full_text)
        if score_line:
            nums = re.findall(r"\d+", score_line[0])
            if len(nums) == 4: scores = [int(n) for n in nums]
        
        return re.sub(r"SCORES:.*", "", full_text).strip(), scores
    except Exception as e:
        return f"❌ 调用失败: {str(e)}", [0, 0, 0, 0]
