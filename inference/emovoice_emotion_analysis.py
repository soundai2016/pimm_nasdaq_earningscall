import json
import openai
import requests
from typing import List, Dict, Optional
import time
import os
import datetime
from dataclasses import dataclass

@dataclass
class EmotionResult:
    """情感分析结果"""
    line: int
    text: str
    emotion: str
    confidence: float
    reasoning: str

class LLMEmotionAnalyzer:
    def __init__(self, file_path: str, emovoice_api_config: Dict):
        self.file_path = file_path
        self.emovoice_api_config = emovoice_api_config
        self.data = []
        self.chunk_size = 1000
        self.api_requests_log = []
        
        # 修改输出逻辑：直接保存在原始JSON文件所在目录
        input_dir = os.path.dirname(os.path.abspath(file_path))
        input_basename = os.path.splitext(os.path.basename(file_path))[0]
        
        # 不再创建单独的emotion_analysis文件夹，直接在同级目录下生成文件
        self.output_dir = input_dir  # 直接使用输入文件所在目录
        
        # 设置输出文件路径（添加_emotion_analysis后缀以区分）
        self.output_json_file = os.path.join(self.output_dir, f'{input_basename}_emotion_analysis_detailed.json')
        self.output_summary_file = os.path.join(self.output_dir, f'{input_basename}_emotion_analysis_summary.txt')
        self.api_log_file = os.path.join(self.output_dir, f'{input_basename}_emotion_analysis_api_log.json')
        
        print(f"输出目录: {self.output_dir}")
        print(f"结果文件: {self.output_json_file}")
        print(f"摘要文件: {self.output_summary_file}")
        print(f"API日志文件: {self.api_log_file}")
    
    def load_data(self):
        """加载JSON数据，提取text字段和行号"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 从results字段中提取数据
            if 'results' in data:
                for item in data['results']:
                    text_content = item.get('text', '').strip()
                    if text_content:  # 只保存有文本内容的记录
                        self.data.append({
                            'line': item.get('line', 0),
                            'text': text_content,
                            'speaker': item.get('speaker', 'Unknown')
                        })
            else:
                print("错误：JSON文件中没有找到'results'字段")
                return False
                
            print(f"已加载 {len(self.data)} 条有效文本记录")
            return True
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False
    
    def log_api_request(self, request_data: Dict, response_data: Dict, chunk_info: Dict):
        """记录API请求和响应"""
        log_entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "chunk_info": chunk_info,
            "request": request_data,
            "response": response_data
        }
        self.api_requests_log.append(log_entry)
    
    def save_api_logs(self):
        """保存API请求日志"""
        try:
            with open(self.api_log_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "total_requests": len(self.api_requests_log),
                    "requests": self.api_requests_log
                }, f, ensure_ascii=False, indent=2)
            print(f"API请求日志已保存到: {self.api_log_file}")
        except Exception as e:
            print(f"保存API日志失败: {e}")
    
    def call_deepseek_api(self, prompt: str, chunk_info: Dict) -> str:
        """调用DeepSeek API"""
        try:
            print("正在初始化DeepSeek客户端...")
            
            base_url = self.emovoice_api_config.get('deepseek_base_url', 'https://api.deepseek.com')
            print(f"使用API地址: {base_url}")
            
            client = openai.OpenAI(
                api_key=self.emovoice_api_config['deepseek_api_key'],
                base_url=base_url
            )
            
            model = self.emovoice_api_config.get('deepseek_model', 'deepseek-chat')
            print(f"使用模型: {model}")
            print(f"提示词长度: {len(prompt)} 字符")
            print("开始发送API请求...")
            
            # 准备请求数据
            messages = [
                {"role": "system", "content": "你是一个专业的情感分析专家。请仔细分析提供的美股财报电话会议文本，识别每句话的情感。请用英文回答。"},
                {"role": "user", "content": prompt}
            ]
            
            request_data = {
                "model": model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 4000,
                "stream": False,
                "timeout": 120
            }
            
            start_time = time.time()
            
            response = client.chat.completions.create(**request_data)
            
            end_time = time.time()
            response_content = response.choices[0].message.content
            
            print(f"API调用成功，耗时: {end_time - start_time:.2f} 秒")
            print(f"响应长度: {len(response_content)} 字符")
            
            # 记录API请求和响应
            response_data = {
                "success": True,
                "response_time": end_time - start_time,
                "content": response_content,
                "content_length": len(response_content),
                "usage": {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                    "total_tokens": getattr(response.usage, 'total_tokens', 0)
                } if hasattr(response, 'usage') else None
            }
            
            self.log_api_request(request_data, response_data, chunk_info)
            
            return response_content
        except Exception as e:
            print(f"DeepSeek API调用失败: {type(e).__name__}: {e}")
            
            # 记录失败的API请求
            error_response = {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
            
            self.log_api_request(request_data if 'request_data' in locals() else {}, error_response, chunk_info)
            
            return None
    
    def call_openai_api(self, prompt: str, chunk_info: Dict, model: str = "gpt-4") -> str:
        """调用OpenAI API"""
        try:
            client = openai.OpenAI(
                api_key=self.emovoice_api_config['openai_api_key'],
                base_url=self.emovoice_api_config.get('openai_base_url')
            )
            
            messages = [
                {"role": "system", "content": "You are a professional emotion analysis expert. Please carefully analyze the provided US earnings call text and identify the emotion in each sentence."},
                {"role": "user", "content": prompt}
            ]
            
            request_data = {
                "model": model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 4000
            }
            
            start_time = time.time()
            
            response = client.chat.completions.create(**request_data)
            
            end_time = time.time()
            response_content = response.choices[0].message.content
            
            # 记录API请求和响应
            response_data = {
                "success": True,
                "response_time": end_time - start_time,
                "content": response_content,
                "content_length": len(response_content),
                "usage": {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                    "total_tokens": getattr(response.usage, 'total_tokens', 0)
                } if hasattr(response, 'usage') else None
            }
            
            self.log_api_request(request_data, response_data, chunk_info)
            
            return response_content
        except Exception as e:
            print(f"OpenAI API调用失败: {e}")
            
            # 记录失败的API请求
            error_response = {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
            
            self.log_api_request(request_data if 'request_data' in locals() else {}, error_response, chunk_info)
            
            return None
    
    def call_api(self, prompt: str, chunk_info: Dict) -> str:
        """根据配置调用相应的API"""
        api_type = self.emovoice_api_config.get('api_type', 'deepseek')
        print(f"当前API类型: {api_type}")
        
        if api_type == 'openai':
            print("调用OpenAI API...")
            return self.call_openai_api(prompt, chunk_info)
        elif api_type == 'deepseek':
            print("调用DeepSeek API...")
            return self.call_deepseek_api(prompt, chunk_info)
        else:
            print(f"不支持的API类型: {api_type}")
            return None
    
    def generate_emotion_analysis_prompt(self, chunk_data: List[Dict]) -> str:
        """生成情感分析提示词"""
        prompt = f"""你是一个专业的情感分析专家。请分析以下文本中每句话的情感色彩。

需要分析的文本（共{len(chunk_data)}行）：
"""
        
        # 添加文本内容
        for item in chunk_data:
            prompt += f"\n第{item['line']}行: {item['text']}"
        
        prompt += f"""\n\n任务：分析每行文本的情感。对于每一行，从以下情感类别中识别主要情感：
- happiness（快乐）: 积极、乐观、自信、满意的情绪
- sadness（悲伤）: 失望、担忧、遗憾、忧郁的情绪
- anger（愤怒）: 挫折、恼怒、批评、强烈反对的情绪
- fear（恐惧）: 担心、焦虑、不确定、对风险的谨慎
- surprise（惊讶）: 意外的结果、震惊、惊奇
- disgust（厌恶）: 强烈的不赞成、拒绝、反感
- none（中性）: 中性、事实性或没有明显情感色彩

重要提醒：每个行号必须且只能出现在一个情感类别中。不要在不同情感中重复行号。

输出格式（必须严格按照以下格式，按情感分组）：
happiness: 行号用逗号分隔
sadness: 行号用逗号分隔
anger: 行号用逗号分隔
fear: 行号用逗号分隔
surprise: 行号用逗号分隔
disgust: 行号用逗号分隔
none: 行号用逗号分隔

示例：
happiness: 13,14,15,16
sadness: 12,18,19
anger: 4,5,20
fear: 10,17
surprise: 8
disgust: 9,11
none: 1,2,3,6,7

请客观分析每一行，将每个行号分配到恰好一个情感类别中。确保从第{chunk_data[0]['line']}行到第{chunk_data[-1]['line']}行的所有行号都被包含，且没有行号出现两次。"""
        
        return prompt
    
    def parse_emotion_response(self, response_text: str, chunk_data: List[Dict]) -> List[Dict]:
        """解析API响应的情感分析结果"""
        try:
            response_text = response_text.strip()
            
            # 解析情感结果
            import re
            emotion_results = []
            used_line_numbers = set()  # 跟踪已使用的行号
            
            # 创建行号到数据的映射
            line_to_data = {item['line']: item for item in chunk_data}
            
            # 解析按情绪分组的格式
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        emotion = parts[0].strip().lower()
                        line_numbers_str = parts[1].strip()
                        
                        # 验证情感类型
                        valid_emotions = ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'none']
                        if emotion not in valid_emotions:
                            continue
                        
                        # 解析行号
                        if line_numbers_str:  # 如果不为空
                            try:
                                line_numbers = [int(x.strip()) for x in line_numbers_str.split(',') if x.strip()]
                                for line_num in line_numbers:
                                    # 检查重复
                                    if line_num in used_line_numbers:
                                        print(f"⚠️  警告：行号 {line_num} 重复出现，跳过后续出现")
                                        continue
                                    
                                    if line_num in line_to_data:
                                        emotion_results.append({
                                            "line": line_num,
                                            "text": line_to_data[line_num]['text'],
                                            "speaker": line_to_data[line_num]['speaker'],
                                            "emotion": emotion
                                        })
                                        used_line_numbers.add(line_num)
                            except ValueError:
                                # 如果解析行号失败，跳过这一行
                                continue
            
            # 检查是否有遗漏的行号
            expected_lines = set(item['line'] for item in chunk_data)
            missing_lines = expected_lines - used_line_numbers
            if missing_lines:
                print(f"⚠️  警告：以下行号未被分析：{sorted(missing_lines)}")
                # 将遗漏的行号归类为 'none'
                for line_num in missing_lines:
                    emotion_results.append({
                        "line": line_num,
                        "text": line_to_data[line_num]['text'],
                        "speaker": line_to_data[line_num]['speaker'],
                        "emotion": "none"
                    })
            
            print(f"成功解析 {len(emotion_results)} 条情感分析结果")
            return emotion_results
            
        except Exception as e:
            print(f"解析API响应失败: {e}")
            return []
    
    def generate_emotion_summary(self, results: List[Dict]) -> Dict:
        """生成按情绪分组的摘要"""
        emotion_groups = {
            'happiness': [],
            'sadness': [],
            'anger': [],
            'fear': [],
            'surprise': [],
            'disgust': [],
            'none': []
        }
        
        # 按情绪分组
        for result in results:
            emotion = result['emotion']
            if emotion in emotion_groups:
                emotion_groups[emotion].append(result['line'])
        
        # 排序行号
        for emotion in emotion_groups:
            emotion_groups[emotion].sort()
        
        return emotion_groups
    
    def save_emotion_summary(self, emotion_groups: Dict):
        """保存情绪摘要到文本文件"""
        try:
            with open(self.output_summary_file, 'w', encoding='utf-8') as f:
                f.write("=== 情感分析摘要 ===\n\n")
                
                for emotion, lines in emotion_groups.items():
                    if lines:  # 只显示有数据的情绪
                        line_str = ','.join(map(str, lines))
                        f.write(f"{emotion}: {line_str}\n")
                    else:
                        f.write(f"{emotion}: 无\n")
                
                f.write(f"\n总计分析行数: {sum(len(lines) for lines in emotion_groups.values())}\n")
            
            print(f"情绪摘要已保存到: {self.output_summary_file}")
            
        except Exception as e:
            print(f"保存情绪摘要失败: {e}")
    
    def run_emotion_analysis(self) -> Dict:
        """运行情感分析"""
        try:
            # 1. 加载数据
            if not self.load_data():
                return None
            
            if not self.data:
                print("❌ 没有加载到有效数据")
                return None
            
            # 2. 判断是否需要分块处理
            if len(self.data) <= self.chunk_size:
                print(f"数据量 {len(self.data)} 行，无需分块处理")
                total_chunks = 1
                chunks = [self.data]
            else:
                print(f"数据量 {len(self.data)} 行，超过 {self.chunk_size} 行，开始分块处理")
                total_chunks = (len(self.data) + self.chunk_size - 1) // self.chunk_size
                chunks = [self.data[i:i + self.chunk_size] for i in range(0, len(self.data), self.chunk_size)]
            
            # 3. 处理数据块
            all_emotion_results = []
            
            for chunk_idx, chunk_data in enumerate(chunks):
                chunk_num = chunk_idx + 1
                chunk_info = {
                    "chunk_number": chunk_num,
                    "total_chunks": total_chunks,
                    "chunk_size": len(chunk_data),
                    "start_line": chunk_data[0]['line'],
                    "end_line": chunk_data[-1]['line']
                }
                
                print(f"\n处理第 {chunk_num}/{total_chunks} 块 (行 {chunk_info['start_line']}-{chunk_info['end_line']})")
                
                # 生成提示词
                prompt = self.generate_emotion_analysis_prompt(chunk_data)
                
                # 调用API
                api_response = self.call_api(prompt, chunk_info)
                if not api_response:
                    print(f"❌ 第 {chunk_num} 块API调用失败")
                    continue
                
                # 解析响应
                chunk_results = self.parse_emotion_response(api_response, chunk_data)
                if chunk_results:
                    all_emotion_results.extend(chunk_results)
                    print(f"✅ 第 {chunk_num} 块处理完成，获得 {len(chunk_results)} 条结果")
                else:
                    print(f"❌ 第 {chunk_num} 块解析失败")
                
                # 添加延迟避免API限流
                if chunk_num < total_chunks:
                    print("等待2秒避免API限流...")
                    time.sleep(2)
            
            # 4. 生成情绪摘要
            emotion_groups = self.generate_emotion_summary(all_emotion_results)
            
            # 5. 整理最终结果
            final_result = {
                "total_lines": len(self.data),
                "processed_lines": len(all_emotion_results),
                "api_type": self.emovoice_api_config.get('api_type', 'unknown'),
                "processing_method": "chunked_emotion_analysis",
                "chunk_size": self.chunk_size,
                "total_chunks": total_chunks,
                "emotion_summary": emotion_groups,
                "results": sorted(all_emotion_results, key=lambda x: x['line'])
            }
            
            # 6. 保存结果
            with open(self.output_json_file, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)
            
            # 7. 保存情绪摘要
            self.save_emotion_summary(emotion_groups)
            
            # 8. 保存API日志
            self.save_api_logs()
            
            print(f"\n✅ 情感分析完成！")
            print(f"📊 总计处理: {final_result['processed_lines']}/{final_result['total_lines']} 行")
            print(f"📁 结果已保存到: {self.output_json_file}")
            print(f"📁 摘要已保存到: {self.output_summary_file}")
            print(f"📁 API日志已保存到: {self.api_log_file}")
            
            # 9. 打印情绪摘要
            self.print_emotion_summary(emotion_groups)
            
            return final_result
            
        except Exception as e:
            print(f"❌ 情感分析过程中发生错误: {e}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            return None
    
    def print_emotion_summary(self, emotion_groups: Dict):
        """打印情感分析摘要"""
        print("\n=== 情感分析摘要 ===")
        total_lines = sum(len(lines) for lines in emotion_groups.values())
        
        for emotion, lines in emotion_groups.items():
            if lines:
                line_str = ','.join(map(str, lines))
                percentage = (len(lines) / total_lines) * 100 if total_lines > 0 else 0
                print(f"{emotion}: {line_str} ({len(lines)}条, {percentage:.1f}%)")
            else:
                print(f"{emotion}: 无")

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) != 2:
        print("使用方法: python emotion_analysis.py <json_file_path>")
        print("示例: python emotion_analysis.py '/path/to/speaker_results.json'")
        return
    
    json_file_path = sys.argv[1]
    
    # 检查文件是否存在
    if not os.path.exists(json_file_path):
        print(f"错误：文件不存在 {json_file_path}")
        return
    
    # 加载API配置
    config_file = 'emovoice_api_config.json'
    if not os.path.exists(config_file):
        print(f"错误：API配置文件不存在 {config_file}")
        return
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            emovoice_api_config = json.load(f)
    except Exception as e:
        print(f"错误：无法加载API配置文件 {e}")
        return
    
    # 创建分析器并运行
    analyzer = LLMEmotionAnalyzer(json_file_path, emovoice_api_config)
    result = analyzer.run_emotion_analysis()
    
    if result:
        print("\n🎉 情感分析完成！")
    else:
        print("❌ 情感分析失败")

if __name__ == "__main__":
    main()
