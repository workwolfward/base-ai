import requests
import json

# API配置
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = "sk-qseennfhdprismchczwnkzpohyjmuwgpiaywuclsisgugfvo"  # 替换为您的API密钥

# 可用模型列表
AVAILABLE_MODELS = {
    "1": "Qwen/QwQ-32B",
    "2": "deepseek-ai/DeepSeek-R1",
    "3": "THUDM/GLM-4-9B-0414",
    "4": "Qwen/Qwen2.5-7B-Instruct",
    "5": "THUDM/chatglm3-6b",
    "6": "Baidu/ERNIE-Bot-4-128K"
}

def get_medical_advice(input_data, model_name="Qwen/QwQ-32B", temperature=0.7):
    """获取医疗建议的API调用函数"""
    # 构建系统提示词
    system_prompt = {
        "role": "system",
        "content": ('''你是一位专业的全科医生。请根据患者症状提供医疗建议，
                    注意分辨日常和紧急严重需求，如果症状较轻是生活中单发的可以不做太严谨的推测而是推荐一些日常非处方药
                    建议尽量偏向生活化一点 减少术语
                    输出必须是以下JSON格式：'''
                   '{"possible_diagnoses": ["诊断1", "诊断2"], "urgency_level": 1-5, '
                   '"recommended_department": "科室名称", "otc_recommendations": ['
                   '{"name": "药品名", "dose": "用法用量"}], "self_care_advice": ["建议1", "建议2"]}')
    }
    
    # 构建用户消息
    user_message = {
        "role": "user",
        "content": f"患者信息: {input_data}"
    }
    
    # 准备请求
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [system_prompt, user_message],
        "temperature": temperature,
        "response_format": {"type": "json_object"}
    }
    
    try:
        # 发送请求
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            return f"API错误: {response.status_code} - {response.text}"
    
    except Exception as e:
        return f"请求失败: {str(e)}"

def main():
    """主程序 - 提供用户交互界面"""
    print("=" * 60)
    print("医疗咨询助手".center(60))
    print("=" * 60)
    
    # 收集患者信息
    print("\n请输入患者信息:")
    symptoms = input("症状描述: ")
    age = input("年龄: ")
    gender = input("性别: ")
    conditions = input("既有疾病(可选): ")
    medications = input("当前用药(可选): ")
    allergies = input("过敏史(可选): ")
    
    # 构建输入数据
    input_data = f"症状: {symptoms}, 年龄: {age}, 性别: {gender}"
    if conditions: input_data += f", 既有疾病: {conditions}"
    if medications: input_data += f", 当前用药: {medications}"
    if allergies: input_data += f", 过敏史: {allergies}"
    
    # 选择模型
    print("\n请选择AI模型:")
    for key, model in AVAILABLE_MODELS.items():
        print(f"{key}. {model}")
    
    model_choice = input("输入选项 (1-6, 默认1): ").strip() or "1"
    selected_model = AVAILABLE_MODELS.get(model_choice, "Qwen/QwQ-32B")
    
    # 设置temperature
    try:
        temperature = float(input("设置temperature (0.0-1.0, 默认0.7): ") or "0.7")
        temperature = max(0.0, min(1.0, temperature))
    except:
        print("输入无效，使用默认值0.7")
        temperature = 0.7
    
    print(f"\n使用{selected_model}分析中...")
    
    # 获取医疗建议
    advice_json = get_medical_advice(input_data, selected_model, temperature)
    
    # 解析并显示结果
    print("\n" + "=" * 60)
    print("医疗建议报告".center(60))
    print("=" * 60)
    
    try:
        advice = json.loads(advice_json)
        
        # 显示诊断
        print("\n[诊断分析]")
        for i, diagnosis in enumerate(advice.get("possible_diagnoses", []), 1):
            print(f"  {i}. {diagnosis}")
        
        # 紧急程度
        urgency_level = advice.get("urgency_level", 3)
        urgency_desc = {
            1: "紧急就医 (需立即前往急诊)",
            2: "高度紧急 (24小时内就医)",
            3: "中度紧急 (2-3天内就医)",
            4: "轻度紧急 (1周内就医)",
            5: "可在家观察 (注意症状变化)"
        }
        print(f"\n[紧急程度] {urgency_level}级 - {urgency_desc.get(urgency_level, '中度紧急')}")
        
        # 推荐科室
        print(f"[推荐科室] {advice.get('recommended_department', '全科')}")
        
        # OTC药品推荐
        if "otc_recommendations" in advice and advice["otc_recommendations"]:
            print("\n[药品建议]")
            for med in advice["otc_recommendations"]:
                print(f"  - {med.get('name', '未命名药品')}: {med.get('dose', '用法未指定')}")
        else:
            print("\n[药品建议] 无适用推荐")
        
        # 自我护理建议
        if "self_care_advice" in advice and advice["self_care_advice"]:
            print("\n[自我护理]")
            for i, tip in enumerate(advice["self_care_advice"], 1):
                print(f"  {i}. {tip}")
        
    except json.JSONDecodeError:
        print("\n无法解析建议格式，原始API响应:")
        print(advice_json)
    
    print("\n" + "=" * 60)
    print("免责声明: 本建议仅供参考，不能替代专业医疗诊断".center(60))
    print("=" * 60)

if __name__ == "__main__":
    main()