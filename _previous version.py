import sys
import json
import requests
import re
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QLabel, QLineEdit, QComboBox, QSlider, QPushButton, QTextEdit,
                             QSplitter, QFrame, QTabWidget, QScrollArea, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QIcon

# API配置
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = "sk-qseennfhdprismchczwnkzpohyjmuwgpiaywuclsisgugfvo"  # 替换为您的API密钥

# 可用模型列表及JSON支持情况
AVAILABLE_MODELS = {
    "DeepSeek-R1": {"id": "deepseek-ai/DeepSeek-R1", "json_support": False},
    "GLM-4-9B": {"id": "THUDM/GLM-4-9B-0414", "json_support": True},
    "Qwen2.5-7B": {"id": "Qwen/Qwen2.5-7B-Instruct", "json_support": True}
}

class MedicalAPIThread(QThread):
    """处理API调用的线程"""
    update_signal = pyqtSignal(str, bool)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, input_data, model_info, temperature):
        super().__init__()
        self.input_data = input_data
        self.model_info = model_info
        self.temperature = temperature
        self.running = True

    def run(self):
        """执行API调用并处理响应"""
        # 构建系统提示词 - 根据文档优化提示词格式
        system_prompt = {
            "role": "system",
            "content": ('''你是一位专业的全科医生。请根据患者症状提供医疗建议，
                        注意分辨日常和紧急严重需求，如果症状较轻是生活中单发的可以不做太严谨的推测而是推荐一些日常非处方药
                        建议尽量偏向生活化一点 减少术语
                        紧急程度必须是1/2/3/4/5之中的一个数,越紧急数越小
                        输出必须是以下JSON格式：
                        {
                          "possible_diagnoses": ["诊断1", "诊断2"],
                          "urgency_level": 1-5,
                          "recommended_department": "科室名称",
                          "otc_recommendations": [
                            {"name": "药品名", "dose": "用法用量"}
                          ],
                          "self_care_advice": ["建议1", "建议2"]
                        }''')
        }
        
        # 构建用户消息 - 根据文档优化消息格式
        user_message = {
            "role": "user",
            "content": self.input_data  # 直接使用输入数据，不添加额外前缀
        }
        
        # 准备请求 - 根据文档设置正确的headers和payload
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json"  # 明确要求JSON响应
        }
        
        payload = {
            "model": self.model_info["id"],
            "messages": [system_prompt, user_message],
            "temperature": self.temperature,
            "stream": True  # 启用流式响应
        }
        
        # 仅对支持JSON模式的模型添加响应格式要求
        if self.model_info["json_support"]:
            payload["response_format"] = {"type": "json_object"}
        
        try:
            # 发送流式请求 - 根据文档设置超时参数
            response = requests.post(API_URL, headers=headers, json=payload, stream=True, timeout=30)
            
            # 检查响应状态码 - 根据文档处理错误响应
            if response.status_code != 200:
                try:
                    error_data = response.json()
                    error_msg = f"API错误 {error_data.get('code', response.status_code)}: {error_data.get('message', '未知错误')}"
                except:
                    error_msg = f"API错误 {response.status_code}: {response.text[:200]}"
                self.error_signal.emit(error_msg)
                return
            
            full_response = ""
            for line in response.iter_lines():
                if not self.running:
                    break
                    
                if line:
                    try:
                        decoded_line = line.decode('utf-8').strip()
                        if not decoded_line:
                            continue
                            
                        # 根据文档处理SSE格式
                        if decoded_line.startswith('data: '):
                            event_data = decoded_line[6:]
                            if event_data == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(event_data)
                                if 'choices' in data and len(data['choices']) > 0:
                                    delta = data['choices'][0].get('delta', {})
                                    if 'content' in delta and delta['content']:
                                        content = delta['content']
                                        full_response += content
                                        self.update_signal.emit(content, False)
                            except json.JSONDecodeError:
                                # 处理部分JSON响应
                                continue
                    except Exception as e:
                        self.error_signal.emit(f"解析错误: {str(e)}")
                        continue
            
            # 尝试解析最终响应 - 根据文档处理可能的非JSON响应
            try:
                # 尝试直接解析为JSON
                advice = json.loads(full_response)
                self.finished_signal.emit(advice)
            except json.JSONDecodeError:
                # 尝试提取可能的JSON部分
                json_match = re.search(r'\{.*\}', full_response, re.DOTALL)
                if json_match:
                    try:
                        advice = json.loads(json_match.group(0))
                        self.finished_signal.emit(advice)
                    except:
                        # 对于不支持JSON的模型，尝试解析为文本
                        self.handle_non_json_response(full_response)
                else:
                    self.error_signal.emit("无法解析API响应")
                    self.update_signal.emit(full_response, True)
                
        except requests.exceptions.Timeout:
            self.error_signal.emit("请求超时，请稍后再试")
        except Exception as e:
            self.error_signal.emit(f"请求失败: {str(e)}")
    
    def handle_non_json_response(self, response_text):
        """处理非JSON格式的响应"""
        try:
            # 尝试从文本中提取结构化数据
            advice = {
                "possible_diagnoses": [],
                "urgency_level": 3,
                "recommended_department": "全科",
                "otc_recommendations": [],
                "self_care_advice": []
            }
            
            # 提取可能的诊断
            diagnoses_match = re.search(r'诊断分析[：:]\s*(.+)', response_text)
            if diagnoses_match:
                diagnoses = diagnoses_match.group(1).split('、') if '、' in diagnoses_match.group(1) else diagnoses_match.group(1).split(',')
                advice["possible_diagnoses"] = [d.strip() for d in diagnoses]
            
            # 提取紧急程度
            urgency_match = re.search(r'紧急程度[：:]\s*(\d)', response_text)
            if urgency_match:
                try:
                    advice["urgency_level"] = int(urgency_match.group(1))
                except:
                    pass
            
            # 提取推荐科室
            dept_match = re.search(r'推荐科室[：:]\s*(\w+)', response_text)
            if dept_match:
                advice["recommended_department"] = dept_match.group(1)
            
            # 提取药品建议
            med_matches = re.findall(r'药品建议[：:]\s*(.+?)(?=\n|$)', response_text)
            if med_matches:
                for med in med_matches:
                    name_dose = med.split('，')
                    if len(name_dose) >= 2:
                        advice["otc_recommendations"].append({
                            "name": name_dose[0].strip(),
                            "dose": name_dose[1].strip()
                        })
            
            # 提取自我护理建议
            advice_matches = re.findall(r'自我护理建议[：:]\s*(.+?)(?=\n|$)', response_text)
            if advice_matches:
                advice["self_care_advice"] = [a.strip() for a in advice_matches[0].split('、')]
            
            self.finished_signal.emit(advice)
        except:
            self.error_signal.emit("无法解析API响应")
            self.update_signal.emit(response_text, True)
    
    def stop(self):
        """停止线程"""
        self.running = False

class MedicalConsultationApp(QMainWindow):
    """医疗咨询助手主应用"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI医疗咨询助手")
        self.setGeometry(100, 100, 1200, 800)
        
        # 设置应用图标（如果有）
        try:
            self.setWindowIcon(QIcon("medical_icon.png"))
        except:
            pass
        
        # 初始化UI
        self.init_ui()
        
        # 当前状态
        self.api_thread = None
        self.current_advice = None

    def init_ui(self):
        """初始化用户界面"""
        # 主布局
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # 标题
        title_label = QLabel("AI医疗咨询助手")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("background-color: #2c3e50; color: white; padding: 15px;")
        main_layout.addWidget(title_label)
        
        # 主分割器
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # 左侧输入面板
        input_panel = QWidget()
        input_layout = QVBoxLayout()
        input_panel.setLayout(input_layout)
        input_panel.setMaximumWidth(400)
        
        # 患者信息组 - 根据医疗场景优化字段
        patient_group = QGroupBox("患者基本信息")
        patient_layout = QVBoxLayout()
        
        # 基本信息
        info_layout = QHBoxLayout()
        self.age_edit = QLineEdit()
        self.age_edit.setPlaceholderText("年龄")
        self.age_edit.setMaximumWidth(80)
        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["男", "女", "其他", "未提供"])
        self.gender_combo.setCurrentIndex(3)  # 默认选择"未提供"
        
        info_layout.addWidget(QLabel("年龄:"))
        info_layout.addWidget(self.age_edit)
        info_layout.addStretch()
        info_layout.addWidget(QLabel("性别:"))
        info_layout.addWidget(self.gender_combo)
        
        # 症状描述 - 改为多行文本输入
        self.symptoms_edit = QTextEdit()
        self.symptoms_edit.setPlaceholderText("请详细描述症状、持续时间、部位等信息...")
        self.symptoms_edit.setMaximumHeight(100)
        
        # 病史信息
        self.conditions_edit = QLineEdit()
        self.conditions_edit.setPlaceholderText("既有疾病 (如高血压、糖尿病)")
        self.medications_edit = QLineEdit()
        self.medications_edit.setPlaceholderText("当前用药")
        self.allergies_edit = QLineEdit()
        self.allergies_edit.setPlaceholderText("药物/食物过敏史")
        
        patient_layout.addLayout(info_layout)
        patient_layout.addWidget(QLabel("症状描述:"))
        patient_layout.addWidget(self.symptoms_edit)
        patient_layout.addWidget(QLabel("既有疾病:"))
        patient_layout.addWidget(self.conditions_edit)
        patient_layout.addWidget(QLabel("当前用药:"))
        patient_layout.addWidget(self.medications_edit)
        patient_layout.addWidget(QLabel("过敏史:"))
        patient_layout.addWidget(self.allergies_edit)
        
        patient_group.setLayout(patient_layout)
        
        # AI设置组
        ai_group = QGroupBox("AI设置")
        ai_layout = QVBoxLayout()
        
        # 模型选择
        self.model_combo = QComboBox()
        self.model_combo.addItems(AVAILABLE_MODELS.keys())
        self.model_combo.setCurrentText("DeepSeek-R1")  
        
        # 添加JSON支持状态提示
        self.model_support_label = QLabel("当前模型支持JSON输出")
        self.model_support_label.setStyleSheet("color: #27ae60;")
        self.model_combo.currentTextChanged.connect(self.update_model_support)
        
        # 温度参数
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setRange(0, 100)  # 0.0 - 1.0 映射为 0-100
        self.temp_slider.setValue(70)  # 默认0.7
        self.temp_label = QLabel("Temperature: 0.7")
        
        ai_layout.addWidget(QLabel("选择AI模型:"))
        ai_layout.addWidget(self.model_combo)
        ai_layout.addWidget(self.model_support_label)
        ai_layout.addSpacing(15)
        ai_layout.addWidget(QLabel("AI创造力参数:"))
        ai_layout.addWidget(self.temp_label)
        ai_layout.addWidget(self.temp_slider)
        ai_layout.addWidget(QLabel("较低值 → 更准确保守\n较高值 → 更有创造力"))
        
        ai_group.setLayout(ai_layout)
        
        # 提交按钮
        self.submit_btn = QPushButton("获取医疗建议")
        self.submit_btn.setStyleSheet(
            "background-color: #3498db; color: white; font-weight: bold; padding: 10px; border-radius: 5px;"
        )
        self.submit_btn.clicked.connect(self.get_medical_advice)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 不确定进度
        self.progress_bar.setVisible(False)
        
        # 添加到左侧面板
        input_layout.addWidget(patient_group)
        input_layout.addWidget(ai_group)
        input_layout.addWidget(self.submit_btn)
        input_layout.addWidget(self.progress_bar)
        input_layout.addStretch()
        
        # 右侧输出面板
        output_tabs = QTabWidget()
        
        # 流式输出标签页
        stream_tab = QWidget()
        stream_layout = QVBoxLayout()
        self.stream_output = QTextEdit()
        self.stream_output.setReadOnly(True)
        self.stream_output.setFont(QFont("Consolas", 10))
        stream_layout.addWidget(QLabel("AI分析过程 (流式输出):"))
        stream_layout.addWidget(self.stream_output)
        stream_tab.setLayout(stream_layout)
        
        # 结构化结果标签页
        result_tab = QWidget()
        result_layout = QVBoxLayout()
        
        # 结果容器
        result_container = QScrollArea()
        result_container.setWidgetResizable(True)
        result_widget = QWidget()
        self.result_layout = QVBoxLayout()
        self.result_layout.setAlignment(Qt.AlignTop)
        result_widget.setLayout(self.result_layout)
        result_container.setWidget(result_widget)
        
        result_layout.addWidget(QLabel("结构化医疗建议:"))
        result_layout.addWidget(result_container)
        result_tab.setLayout(result_layout)
        
        # 添加标签页
        output_tabs.addTab(stream_tab, "实时分析")
        output_tabs.addTab(result_tab, "医疗报告")
        
        # 添加到分割器
        main_splitter.addWidget(input_panel)
        main_splitter.addWidget(output_tabs)
        main_splitter.setSizes([300, 900])
        
        # 状态栏
        self.statusBar().showMessage("就绪")
        
        # 连接信号
        self.temp_slider.valueChanged.connect(self.update_temp_label)
        
        # 初始化模型支持状态
        self.update_model_support(self.model_combo.currentText())

    def update_model_support(self, model_name):
        """更新模型JSON支持状态显示"""
        model_info = AVAILABLE_MODELS.get(model_name)
        if model_info:
            if model_info["json_support"]:
                self.model_support_label.setText("✅ 当前模型支持JSON输出")
                self.model_support_label.setStyleSheet("color: #27ae60;")
            else:
                self.model_support_label.setText("⚠️ 当前模型不支持JSON输出，将尝试解析文本")
                self.model_support_label.setStyleSheet("color: #e67e22;")

    def update_temp_label(self):
        """更新温度参数标签"""
        value = self.temp_slider.value() / 100.0
        self.temp_label.setText(f"Temperature: {value:.1f}")

    def get_medical_advice(self):
        """获取医疗建议"""
        # 收集患者信息
        symptoms = self.symptoms_edit.toPlainText().strip()
        if not symptoms:
            self.statusshowMessage("错误：请输入症状描述")
            return
            
        age = self.age_edit.text().strip() or "未提供"
        gender = self.gender_combo.currentText() or "未提供"
        conditions = self.conditions_edit.text().strip() or "无"
        medications = self.medications_edit.text().strip() or "无"
        allergies = self.allergies_edit.text().strip() or "无"
        
        # 构建输入数据 - 根据API文档优化输入格式
        input_data = (
            f"患者年龄: {age}\n"
            f"患者性别: {gender}\n"
            f"主要症状: {symptoms}\n"
            f"既有疾病: {conditions}\n"
            f"当前用药: {medications}\n"
            f"过敏史: {allergies}"
        )
        
        # 获取模型信息
        model_name = self.model_combo.currentText()
        model_info = AVAILABLE_MODELS.get(model_name, {"id": "Qwen/QwQ-32B", "json_support": True})
        temperature = self.temp_slider.value() / 100.0
        
        # 更新UI状态
        self.submit_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.stream_output.clear()
        self.clear_results()
        self.statusBar().showMessage(f"使用 {model_name} 分析中...")
        
        # 停止之前的线程（如果存在）
        if self.api_thread and self.api_thread.isRunning():
            self.api_thread.stop()
            self.api_thread.wait()
            
        # 启动API线程
        self.api_thread = MedicalAPIThread(input_data, model_info, temperature)
        self.api_thread.update_signal.connect(self.update_stream_output)
        self.api_thread.finished_signal.connect(self.handle_finished)
        self.api_thread.error_signal.connect(self.handle_error)
        self.api_thread.finished.connect(self.thread_finished)
        self.api_thread.start()

    def update_stream_output(self, content, is_error):
        """更新流式输出区域"""
        # 确保内容是字符串
        if content is None:
            content = ""
            
        cursor = self.stream_output.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(content)
        self.stream_output.setTextCursor(cursor)
        self.stream_output.ensureCursorVisible()
        
        if is_error:
            self.stream_output.append("\n\n⚠️ 无法解析为JSON格式，显示原始输出")

    def handle_finished(self, advice):
        """处理成功的API响应"""
        self.current_advice = advice
        self.display_results(advice)
        self.statusBar().showMessage("分析完成！")

    def handle_error(self, error_msg):
        """处理API错误"""
        if error_msg is None:
            error_msg = "未知错误"
        self.stream_output.append(f"\n\n❌ {error_msg}")
        self.statusBar().showMessage("发生错误")

    def thread_finished(self):
        """线程结束时清理UI状态"""
        self.submit_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def clear_results(self):
        """清除之前的分析结果"""
        # 移除所有结果部件
        while self.result_layout.count():
            item = self.result_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def display_results(self, advice):
        """显示结构化结果"""
        self.clear_results()
        
        # 确保advice是字典类型
        if not isinstance(advice, dict):
            self.result_layout.addWidget(QLabel("无法解析医疗建议"))
            return
        
        # 诊断分析
        diagnoses = advice.get("possible_diagnoses", [])
        if diagnoses:
            group = QGroupBox("诊断分析")
            layout = QVBoxLayout()
            
            for i, diagnosis in enumerate(diagnoses, 1):
                # 确保诊断是字符串
                label = QLabel(f"{i}. {str(diagnosis)}")
                layout.addWidget(label)
            
            group.setLayout(layout)
            self.result_layout.addWidget(group)
        
        # 紧急程度
        urgency_level = advice.get("urgency_level", 3)
        if not isinstance(urgency_level, int):
            try:
                urgency_level = int(urgency_level)
            except:
                urgency_level = 3
                
        urgency_desc = {
            1: ("紧急就医 (需立即前往急诊)", "#e74c3c"),
            2: ("高度紧急 (24小时内就医)", "#e67e22"),
            3: ("中度紧急 (2-3天内就医)", "#f39c12"),
            4: ("轻度紧急 (1周内就医)", "#2ecc71"),
            5: ("可在家观察 (注意症状变化)", "#27ae60")
        }
        
        desc, color = urgency_desc.get(urgency_level, ("中度紧急", "#f39c12"))
        group = QGroupBox("紧急程度")
        layout = QVBoxLayout()
        
        label = QLabel(f"<b>等级 {urgency_level}:</b> {desc}")
        label.setStyleSheet(f"color: {color}; font-size: 14px;")
        layout.addWidget(label)
        
        # 添加紧急程度说明
        urgency_info = QLabel()
        if urgency_level == 1:
            urgency_info.setText("立即拨打急救电话或前往急诊科！")
            urgency_info.setStyleSheet("color: #e74c3c; font-weight: bold;")
        elif urgency_level == 2:
            urgency_info.setText("24小时内就医，不要延误！")
            urgency_info.setStyleSheet("color: #e67e22; font-weight: bold;")
        layout.addWidget(urgency_info)
        
        group.setLayout(layout)
        self.result_layout.addWidget(group)
        
        # 推荐科室
        department = advice.get("recommended_department", "全科")
        group = QGroupBox("推荐科室")
        layout = QVBoxLayout()
        
        label = QLabel(f"<b>{str(department)}</b>")
        label.setStyleSheet("font-size: 16px; color: #2980b9;")
        layout.addWidget(label)
        
        group.setLayout(layout)
        self.result_layout.addWidget(group)
        
        # 药品建议
        otc_recommendations = advice.get("otc_recommendations", [])
        if otc_recommendations:
            group = QGroupBox("药品建议")
            layout = QVBoxLayout()
            
            for med in otc_recommendations:
                # 确保药品信息是字典
                if not isinstance(med, dict):
                    continue
                    
                frame = QFrame()
                frame.setFrameShape(QFrame.StyledPanel)
                frame.setStyleSheet("background-color: #f8f9fa; border-radius: 5px; padding: 10px;")
                med_layout = QVBoxLayout()
                
                name = med.get('name', '未命名药品') or "未命名药品"
                dose = med.get('dose', '用法未指定') or "用法未指定"
                
                name_label = QLabel(f"<b>{str(name)}</b>")
                name_label.setStyleSheet("font-size: 14px;")
                med_layout.addWidget(name_label)
                
                dose_label = QLabel(f"用法用量: {str(dose)}")
                dose_label.setStyleSheet("color: #7f8c8d;")
                med_layout.addWidget(dose_label)
                
                frame.setLayout(med_layout)
                layout.addWidget(frame)
            
            group.setLayout(layout)
            self.result_layout.addWidget(group)
        else:
            group = QGroupBox("药品建议")
            layout = QVBoxLayout()
            layout.addWidget(QLabel("无适用非处方药推荐"))
            group.setLayout(layout)
            self.result_layout.addWidget(group)
        
        # 自我护理建议
        self_care_advice = advice.get("self_care_advice", [])
        if self_care_advice:
            group = QGroupBox("自我护理建议")
            layout = QVBoxLayout()
            
            for i, tip in enumerate(self_care_advice, 1):
                frame = QFrame()
                frame.setFrameShape(QFrame.StyledPanel)
                frame.setStyleSheet("background-color: #f8f9fa; border-radius: 5px; padding: 10px;")
                tip_layout = QHBoxLayout()
                
                num_label = QLabel(f"{i}.")
                num_label.setStyleSheet("font-weight: bold; color: #3498db; font-size: 16px;")
                tip_layout.addWidget(num_label)
                
                tip_label = QLabel(str(tip))  # 确保tip是字符串
                tip_label.setWordWrap(True)
                tip_layout.addWidget(tip_label, 1)
                
                frame.setLayout(tip_layout)
                layout.addWidget(frame)
            
            group.setLayout(layout)
            self.result_layout.addWidget(group)
        
        # 免责声明
        disclaimer = QLabel(
            "<i>免责声明: 本建议由AI生成，仅供参考，不能替代专业医疗诊断。"
            "如有紧急情况请立即就医。用药前请咨询医生或药师。</i>"
        )
        disclaimer.setStyleSheet("color: #95a5a6; padding: 15px; background: #f8f9fa;")
        disclaimer.setAlignment(Qt.AlignCenter)
        disclaimer.setWordWrap(True)
        self.result_layout.addWidget(disclaimer)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QMainWindow {
            background-color: #ecf0f1;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #bdc3c7;
            border-radius: 5px;
            margin-top: 1.5em;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QTextEdit {
            background-color: white;
            border: 1px solid #bdc3c7;
            border-radius: 5px;
            padding: 10px;
        }
        QLineEdit {
            padding: 5px;
            border: 1px solid #bdc3c7;
            border-radius: 3px;
        }
        QPushButton:disabled {
            background-color: #bdc3c7;
        }
        QFrame {
            border: none;
        }
    """)
    
    window = MedicalConsultationApp()
    window.show()
    sys.exit(app.exec_())




