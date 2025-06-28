import requests
import json
import time

# 使用SiliconFlow API
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = "sk-qseennfhdprismchczwnkzpohyjmuwgpiaywuclsisgugfvo"  # 替换为您的API密钥

# 更新模型列表（只保留支持的模型）
# 更新为SiliconFlow支持的模型名称
AVAILABLE_MODELS = {
    "1": "deepseek-ai/DeepSeek-R1",  # DeepSeek-R1
    "2": "Qwen/Qwen2.5-72B-Instruct-128K",  # Qwen2.5-7B
    "3": "THUDM/glm-4-9b-chat"  # GLM-4-9B (注意大写THUDM)
}


def get_medical_advice(input_data, model_name="deepseek-ai/deepseek-r1", params=None):
    """获取医疗建议的API调用函数"""
    if params is None:
        params = {}
    
    # 设置默认参数
    default_params = {
        'temperature': 0.5,
        'top_p': 0.9,
        'max_tokens': 1500,
        'frequency_penalty': 0.2,
        'presence_penalty': 0.1,
        'system_weight': 1.2,
        'repetition_penalty': 1.2,
        'top_k': 50,
        'length_penalty': 1.0,
        'typical_p': 0.95,
        'beam_size': 1,
        'min_p': 0.05,
        'temperature_schedule': "constant"
    }
    # 合并用户参数和默认参数
    final_params = {**default_params, **params}
    
    # 增强的系统提示词
    system_content = '''你是一位专业的全科医生，拥有10年临床经验。请根据患者症状提供专业、准确的医疗建议。
                     
                    重要指导原则:
                    1. 首先评估症状的紧急程度（1-5级）
                    2. 提供2-3个最可能的诊断（按可能性排序）
                    3. 推荐最合适的就诊科室
                    4. 对于轻微症状，可推荐非处方药（OTC）并说明用法
                    5. 提供实用的自我护理建议
                    6. 特别注意患者提供的既有疾病、当前用药和过敏史
                    7. 语言要专业但易懂，避免过度医学术语'''
    
    # 应用系统权重
    if final_params['system_weight'] > 1.0:
        system_content += f"\n\n[系统强调] 请严格遵守医疗规范和输出格式要求（权重:{final_params['system_weight']}x)"
    
    system_prompt = {
        "role": "system",
        "content": system_content
    }
    
    # 构建用户消息
    user_message = {
        "role": "user",
        "content": f"患者信息: {input_data}\n\n请提供专业医疗建议并严格遵守输出格式要求。"
    }
    
    # 准备请求
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    payload = {
        "model": model_name,
        "messages": [system_prompt, user_message],
        "temperature": final_params['temperature'],
        "top_p": final_params['top_p'],
        "max_tokens": final_params['max_tokens'],
        "frequency_penalty": final_params['frequency_penalty'],
        "presence_penalty": final_params['presence_penalty'],
        "response_format": {"type": "json_object"}
    }
    
    # 添加新参数
    if final_params['repetition_penalty'] != default_params['repetition_penalty']:
        payload['repetition_penalty'] = final_params['repetition_penalty']
    
    if final_params['top_k'] != default_params['top_k']:
        payload['top_k'] = final_params['top_k']
    
    if final_params['length_penalty'] != default_params['length_penalty']:
        payload['length_penalty'] = final_params['length_penalty']
    
    if final_params['typical_p'] != default_params['typical_p']:
        payload['typical_p'] = final_params['typical_p']
    
    if final_params['beam_size'] != default_params['beam_size'] and final_params['beam_size'] > 1:
        payload['beam_size'] = final_params['beam_size']
    
    if final_params['min_p'] != default_params['min_p']:
        payload['min_p'] = final_params['min_p']
    
    # 温度调度
    if final_params['temperature_schedule'] != "constant":
        payload['temperature'] = {
            "schedule": final_params['temperature_schedule'],
            "start": final_params.get('temperature_start', 0.7),
            "end": final_params.get('temperature_end', 0.3)
        }
    
    try:
        # 发送请求 - 添加超时和重试机制
        for attempt in range(3):  # 最多重试3次
            try:
                response = requests.post(
                    API_URL, 
                    headers=headers, 
                    json=payload, 
                    timeout=(10, 30)  # 连接超时10秒，读取超时30秒
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                elif response.status_code == 429:
                    # 处理速率限制
                    retry_after = int(response.headers.get('Retry-After', 10))
                    print(f"⚠️ 达到速率限制，{retry_after}秒后重试...")
                    time.sleep(retry_after)
                    continue
                else:
                    return f"API错误 ({response.status_code}): {response.text}"
                
            except (requests.Timeout, requests.ConnectionError):
                if attempt < 2:  # 如果不是最后一次尝试
                    print(f"⚠️ 连接超时，5秒后重试 ({attempt+1}/3)...")
                    time.sleep(5)
                else:
                    return "请求超时，请检查网络连接后重试"
        
        return "请求失败：超过最大重试次数"
    
    except Exception as e:
        return f"请求失败: {str(e)}"

def get_advanced_parameters():
    """获取高级参数设置 - 添加更多参数"""
    params = {}
    
    print("\n" + "=" * 70)
    print("高级参数设置".center(70))
    print("=" * 70)
    
    # 温度参数
    print("\n🌡️ 温度 (Temperature) - 控制输出的随机性:")
    print("  0.0-0.3: 保守诊断 (推荐严重症状)")
    print("  0.4-0.7: 平衡模式 (默认)")
    print("  0.8-1.0: 创意建议 (生活护理)")
    try:
        temp = float(input("  输入值 (0.0-1.0, 默认0.5): ") or "0.5")
        params['temperature'] = max(0.0, min(1.0, temp))
    except:
        print("  输入无效，使用默认值0.5")
        params['temperature'] = 0.5
    
    # Top-p 采样
    print("\n🎯 Top-p - 控制词汇采样的范围:")
    print("  0.5-0.8: 更集中的建议 (推荐诊断)")
    print("  0.8-0.95: 平衡模式 (默认)")
    print("  0.95-1.0: 更多样化的建议")
    try:
        top_p = float(input("  输入值 (0.5-1.0, 默认0.9): ") or "0.9")
        params['top_p'] = max(0.5, min(1.0, top_p))
    except:
        print("  输入无效，使用默认值0.9")
        params['top_p'] = 0.9
    
    # 最大长度
    print("\n📏 最大长度 (Max Tokens) - 控制响应详细程度:")
    print("  500-800: 简明建议")
    print("  1000-1500: 详细报告 (默认)")
    print("  1500-3000: 非常详细的指导")
    try:
        max_tokens = int(input("  输入值 (500-3000, 默认1500): ") or "1500")
        params['max_tokens'] = max(500, min(3000, max_tokens))
    except:
        print("  输入无效，使用默认值1500")
        params['max_tokens'] = 1500
    
    # 频率惩罚
    print("\n🔄 频率惩罚 (Frequency Penalty) - 减少重复内容:")
    print("  0.0-0.3: 允许适当重复 (医学术语)")
    print("  0.3-0.6: 平衡模式 (默认)")
    print("  0.6-1.0: 严格避免重复")
    try:
        freq_penalty = float(input("  输入值 (0.0-1.0, 默认0.2): ") or "0.2")
        params['frequency_penalty'] = max(0.0, min(1.0, freq_penalty))
    except:
        print("  输入无效，使用默认值0.2")
        params['frequency_penalty'] = 0.2
    
    # 存在惩罚
    print("\n💬 存在惩罚 (Presence Penalty) - 鼓励新话题:")
    print("  0.0-0.2: 保持主题集中 (推荐诊断)")
    print("  0.2-0.5: 平衡模式 (默认)")
    print("  0.5-1.0: 鼓励更多相关主题")
    try:
        pres_penalty = float(input("  输入值 (0.0-1.0, 默认0.1): ") or "0.1")
        params['presence_penalty'] = max(0.0, min(1.0, pres_penalty))
    except:
        print("  输入无效，使用默认值0.1")
        params['presence_penalty'] = 0.1
    
    # 系统权重
    print("\n⚖️ 系统权重 (System Weight) - 强调医疗规范:")
    print("  0.5-1.0: 更自由的建议 (生活护理)")
    print("  1.0-1.5: 平衡模式 (默认)")
    print("  1.5-2.0: 严格遵守医疗规范 (推荐诊断)")
    try:
        system_weight = float(input("  输入值 (0.5-2.0, 默认1.2): ") or "1.2")
        params['system_weight'] = max(0.5, min(2.0, system_weight))
    except:
        print("  输入无效，使用默认值1.2")
        params['system_weight'] = 1.2
    
    # 新增参数
    print("\n" + "=" * 70)
    print("专家级参数设置".center(70))
    print("=" * 70)
    
    # 重复惩罚
    print("\n🔄 重复惩罚 (Repetition Penalty) - 减少内容重复:")
    print("  1.0: 无惩罚")
    print("  1.1-1.3: 轻度惩罚 (推荐)")
    print("  1.3-2.0: 严格惩罚")
    try:
        rep_penalty = float(input("  输入值 (1.0-2.0, 默认1.2): ") or "1.2")
        params['repetition_penalty'] = max(1.0, min(2.0, rep_penalty))
    except:
        print("  输入无效，使用默认值1.2")
        params['repetition_penalty'] = 1.2
    
    # Top-K 采样
    print("\n🎯 Top-K - 限制考虑的词汇数量:")
    print("  10-30: 高度集中 (诊断)")
    print("  30-70: 平衡模式 (默认)")
    print("  70-100: 多样化选择")
    try:
        top_k = int(input("  输入值 (10-100, 默认50): ") or "50")
        params['top_k'] = max(10, min(100, top_k))
    except:
        print("  输入无效，使用默认值50")
        params['top_k'] = 50
    
    # 长度惩罚
    print("\n📏 长度惩罚 (Length Penalty) - 控制响应长度:")
    print("  0.0-0.5: 鼓励长响应")
    print("  0.5-1.0: 中性 (默认)")
    print("  1.0-2.0: 鼓励短响应")
    try:
        len_penalty = float(input("  输入值 (0.0-2.0, 默认1.0): ") or "1.0")
        params['length_penalty'] = max(0.0, min(2.0, len_penalty))
    except:
        print("  输入无效，使用默认值1.0")
        params['length_penalty'] = 1.0
    
    # 典型采样
    print("\n🎲 典型采样 (Typical P) - 提高输出多样性:")
    print("  0.7-0.9: 平衡多样性")
    print("  0.9-1.0: 高度多样化")
    try:
        typical_p = float(input("  输入值 (0.7-1.0, 默认0.95): ") or "0.95")
        params['typical_p'] = max(0.7, min(1.0, typical_p))
    except:
        print("  输入无效，使用默认值0.95")
        params['typical_p'] = 0.95
    
    # 束搜索
    print("\n🔍 束搜索 (Beam Size) - 提高输出质量:")
    print("  1: 贪心解码 (默认)")
    print("  2-5: 提高质量 (增加计算时间)")
    try:
        beam_size = int(input("  输入值 (1-5, 默认1): ") or "1")
        params['beam_size'] = max(1, min(5, beam_size))
    except:
        print("  输入无效，使用默认值1")
        params['beam_size'] = 1
    
    # 最小概率
    print("\n🎯 最小概率 (Min P) - 过滤低概率词汇:")
    print("  0.01-0.1: 严格过滤")
    print("  0.1-0.3: 平衡过滤")
    try:
        min_p = float(input("  输入值 (0.01-0.3, 默认0.05): ") or "0.05")
        params['min_p'] = max(0.01, min(0.3, min_p))
    except:
        print("  输入无效，使用默认值0.05")
        params['min_p'] = 0.05
    
    # 温度调度
    print("\n📊 温度调度 (Temperature Schedule) - 动态调整随机性:")
    print("  1. 恒定 (constant)")
    print("  2. 递减 (decreasing)")
    print("  3. 递增 (increasing)")
    schedule_choice = input("  选择调度策略 (1-3, 默认1): ") or "1"
    
    if schedule_choice == "2":
        params['temperature_schedule'] = "decreasing"
        try:
            temp_start = float(input("  起始温度 (0.1-1.0, 默认0.7): ") or "0.7")
            temp_end = float(input("  结束温度 (0.1-1.0, 默认0.3): ") or "0.3")
            params['temperature_start'] = max(0.1, min(1.0, temp_start))
            params['temperature_end'] = max(0.1, min(1.0, temp_end))
        except:
            print("  输入无效，使用默认值")
            params['temperature_start'] = 0.7
            params['temperature_end'] = 0.3
    elif schedule_choice == "3":
        params['temperature_schedule'] = "increasing"
        try:
            temp_start = float(input("  起始温度 (0.1-1.0, 默认0.3): ") or "0.3")
            temp_end = float(input("  结束温度 (0.1-1.0, 默认0.7): ") or "0.7")
            params['temperature_start'] = max(0.1, min(1.0, temp_start))
            params['temperature_end'] = max(0.1, min(1.0, temp_end))
        except:
            print("  输入无效，使用默认值")
            params['temperature_start'] = 0.3
            params['temperature_end'] = 0.7
    else:
        params['temperature_schedule'] = "constant"
    
    return params

def main():
    """主程序 - 完整的参数调节系统"""
    print("=" * 70)
    print("AI医疗咨询助手".center(70))
    print("=" * 70)
    
    # 收集患者信息
    print("\n请详细描述患者情况:")
    symptoms = input("✍️ 主要症状及持续时间: ")
    age = input("👶 年龄: ")
    gender = input("👤 性别: ")
    conditions = input("🏥 既有疾病(如高血压、糖尿病等, 没有请留空): ") or "无"
    medications = input("💊 当前用药(没有请留空): ") or "无"
    allergies = input("⚠️ 过敏史(药物/食物等, 没有请留空): ") or "无"
    
    # 构建输入数据
    input_data = (
        f"症状: {symptoms}\n"
        f"年龄: {age}岁\n"
        f"性别: {gender}\n"
        f"既有疾病: {conditions}\n"
        f"当前用药: {medications}\n"
        f"过敏史: {allergies}"
    )
    
    # 模型选择
    print("\n" + "-" * 70)
    print("请选择AI模型 (输入数字):")
    print("1. DeepSeek-R1 (推理能力强)")
    print("2. Qwen2.5-7B (对话流畅)")
    print("3. GLM-4-9B (中文优化)")
    
    while True:
        model_choice = input("选择模型 (1-3, 默认1): ").strip() or "1"
        if model_choice in AVAILABLE_MODELS:
            selected_model = AVAILABLE_MODELS[model_choice]
            break
        print("⚠️ 无效选择，请重新输入")
    
    # 询问是否使用高级参数
    use_advanced = input("\n是否配置高级参数? (y/n, 默认n): ").lower().strip() == 'y'
    
    if use_advanced:
        params = get_advanced_parameters()
    else:
        # 使用默认参数
        params = {
            'temperature': 0.5,
            'top_p': 0.9,
            'max_tokens': 1500,
            'frequency_penalty': 0.2,
            'presence_penalty': 0.1,
            'system_weight': 1.2,
            'repetition_penalty': 1.2,
            'top_k': 50,
            'length_penalty': 1.0,
            'typical_p': 0.95,
            'beam_size': 1,
            'min_p': 0.05,
            'temperature_schedule': "constant"
        }
        print("\n使用默认参数设置:")
        print(f"  温度: {params['temperature']}, Top-p: {params['top_p']}, 最大长度: {params['max_tokens']}")
    
    print(f"\n使用 {selected_model} 分析中，请稍候...")
    
    # 显示加载动画
    for i in range(5):
        print("▰" * (i+1) + "▱" * (5-i), end="\r")
        time.sleep(0.3)
    
    # 获取医疗建议
    advice_json = get_medical_advice(input_data, selected_model, params)
    
    # 解析并显示结果
    print("\n" + "=" * 70)
    print("医疗建议报告".center(70))
    print("=" * 70)
    print(f"模型: {selected_model}")
    print(f"参数: T={params['temperature']}, P={params['top_p']}, L={params['max_tokens']}, RP={params.get('repetition_penalty', 1.2)}")
    print("-" * 70)
    
    try:
        if advice_json.startswith("API错误") or advice_json.startswith("请求失败"):
            print(f"\n⚠️ {advice_json}")
            print("\n建议解决方案:")
            print("1. 检查API密钥是否正确")
            print("2. 确保网络连接正常")
            print("3. 尝试更换模型或简化查询")
            print("4. 稍后再试或联系API提供商")
        else:
            advice = json.loads(advice_json)
            
            # 显示诊断
            print("\n🔍 [诊断分析]")
            for i, diagnosis in enumerate(advice.get("possible_diagnoses", []), 1):
                print(f"  {i}. {diagnosis}")
            
            # 紧急程度
            urgency_level = advice.get("urgency_level", 3)
            urgency_desc = {
                1: "🆘 紧急就医 (需立即前往急诊)",
                2: "⚠️ 高度紧急 (24小时内就医)",
                3: "❗ 中度紧急 (2-3天内就医)",
                4: "ℹ️ 轻度紧急 (1周内就医)",
                5: "✅ 可在家观察 (注意症状变化)"
            }
            print(f"\n🚨 [紧急程度] {urgency_level}级 - {urgency_desc.get(urgency_level, '中度紧急')}")
            print(f"  原因: {advice.get('urgency_reason', '基于症状评估')}")
            
            # 推荐科室
            print(f"\n🏥 [推荐科室] {advice.get('recommended_department', '全科')}")
            print(f"  理由: {advice.get('department_reason', '最适合处理此症状')}")
            
            # OTC药品推荐
            if advice.get("otc_recommendations"):
                print("\n💊 [药品建议]")
                for med in advice["otc_recommendations"]:
                    print(f"  - {med.get('name', '未命名药品')}")
                    print(f"    用法: {med.get('dose', '请遵医嘱')}")
                    if "warning" in med:
                        print(f"    注意: {med['warning']}")
            else:
                print("\n💊 [药品建议] 无适用OTC推荐")
            
            # 自我护理建议
            if advice.get("self_care_advice"):
                print("\n🏡 [自我护理]")
                for i, tip in enumerate(advice["self_care_advice"], 1):
                    print(f"  {i}. {tip}")
            
            # 随访建议
            if "follow_up_instructions" in advice:
                print(f"\n📅 [随访建议] {advice['follow_up_instructions']}")
            
            # 诊断信心
            if "confidence_level" in advice:
                confidence = advice["confidence_level"]
                confidence_desc = {
                    "high": "💯 高信心 (基于明确症状)",
                    "medium": "🆗 中等信心 (需进一步检查)",
                    "low": "❓ 低信心 (症状不明确)"
                }
                print(f"\n🔬 [诊断信心] {confidence_desc.get(confidence, confidence)}")
    
    except json.JSONDecodeError:
        print("\n⚠️ 无法解析建议格式，原始API响应:")
        print(advice_json)
        print("\n提示: 可能是模型未正确遵循输出格式要求，请尝试：")
        print("  1. 降低温度参数(Temperature)")
        print("  2. 增加系统权重(System Weight)")
        print("  3. 使用支持JSON格式的模型")
    
    print("\n" + "=" * 70)
    print("免责声明: 本建议仅供参考，不能替代专业医疗诊断".center(70))
    print("=" * 70)
    
    # 保存报告
    save = input("\n是否保存报告到文件? (y/n): ").lower()
    if save == 'y':
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"medical_report_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("AI医疗咨询报告\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"使用模型: {selected_model}\n")
                f.write(f"参数设置: temperature={params['temperature']}, top_p={params['top_p']}, "
                        f"max_tokens={params['max_tokens']}, repetition_penalty={params.get('repetition_penalty', 1.2)}\n\n")
                f.write("[患者信息]\n")
                f.write(input_data + "\n\n")
                f.write("[医疗建议]\n")
                
                if 'advice' in locals():
                    # 保存结构化的建议
                    f.write("诊断分析:\n")
                    for diag in advice.get("possible_diagnoses", []):
                        f.write(f"  - {diag}\n")
                    
                    f.write(f"\n紧急程度: {advice.get('urgency_level', 3)}级\n")
                    f.write(f"原因: {advice.get('urgency_reason', '')}\n")
                    
                    f.write(f"\n推荐科室: {advice.get('recommended_department', '')}\n")
                    f.write(f"理由: {advice.get('department_reason', '')}\n")
                    
                    if advice.get("otc_recommendations"):
                        f.write("\n药品建议:\n")
                        for med in advice["otc_recommendations"]:
                            f.write(f"  - {med.get('name', '')}: {med.get('dose', '')}\n")
                            if "warning" in med:
                                f.write(f"    注意: {med['warning']}\n")
                    
                    if advice.get("self_care_advice"):
                        f.write("\n自我护理建议:\n")
                        for tip in advice["self_care_advice"]:
                            f.write(f"  - {tip}\n")
                    
                    if "follow_up_instructions" in advice:
                        f.write(f"\n随访建议: {advice['follow_up_instructions']}\n")
                    
                    if "confidence_level" in advice:
                        f.write(f"\n诊断信心: {advice['confidence_level']}\n")
                else:
                    f.write(advice_json)
                
                f.write("\n\n" + "=" * 70 + "\n")
                f.write("免责声明: 本建议仅供参考，不能替代专业医疗诊断")
            
            print(f"✅ 报告已保存到: {filename}")
        except Exception as e:
            print(f"⚠️ 保存失败: {str(e)}")

if __name__ == "__main__":
    main()