import json
import argparse
import os
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from datetime import datetime
from emotion_analysis import LLMEmotionAnalyzer

class BatchEmotionProcessor:
    def __init__(self, config, max_workers=4):
        self.config = config
        self.max_workers = max_workers
        self.lock = threading.Lock()
        self.processed_count = 0
        self.success_count = 0
        self.failed_count = 0
        self.reprocessed_count = 0  # 新增：重新处理的文件计数
        self.start_time = None
        
    def load_config(self, config_file='emovoice_api_config.json'):
        """加载API配置"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"配置文件 {config_file} 不存在，请创建配置文件")
            return None
        except json.JSONDecodeError:
            print(f"配置文件 {config_file} 格式错误")
            return None
    
    def check_emotion_analysis_success(self, detailed_file_path):
        """检查情绪分析结果文件是否处理成功
        
        Args:
            detailed_file_path: 详细结果JSON文件路径
            
        Returns:
            bool: True表示处理成功，False表示需要重新处理
        """
        try:
            if not detailed_file_path.exists():
                return False
                
            with open(detailed_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 检查processed_lines是否为0
            processed_lines = data.get('processed_lines', 0)
            total_lines = data.get('total_lines', 0)
            
            # 如果processed_lines为0，说明处理失败
            if processed_lines == 0:
                print(f"🔄 发现失败的分析文件（processed_lines=0）: {detailed_file_path.name}")
                return False
                
            # 如果results数组为空，也认为是失败
            results = data.get('results', [])
            if not results or len(results) == 0:
                print(f"🔄 发现空结果的分析文件: {detailed_file_path.name}")
                return False
                
            return True
            
        except Exception as e:
            print(f"⚠️  检查文件时出错 {detailed_file_path}: {e}")
            return False
    
    def find_all_json_files(self, root_folder, pattern="*.json"):
        """查找指定文件夹结构中的json文件
        
        新规则：
        - 扫描 root_folder/子文件夹/azero_full*_analysis/ 中的json文件
        - 跳过已经是情绪分析结果的文件
        """
        json_files = []
        root_path = Path(root_folder)
        
        # 遍历根文件夹下的直接子文件夹（如 test1/1, test1/2）
        for level1_item in root_path.iterdir():
            if level1_item.is_dir():
                # 直接在一级子文件夹中查找以 azero_full 开头、_analysis 结尾的文件夹
                for level2_item in level1_item.iterdir():
                    if (level2_item.is_dir() and 
                        level2_item.name.startswith('azero_full') and 
                        level2_item.name.endswith('_analysis')):
                        # 在 analysis 文件夹中查找 json 文件
                        for json_file in level2_item.glob(pattern):
                            if json_file.is_file():
                                # 跳过已经是情绪分析结果的文件
                                if "_emotion_analysis" not in json_file.name:
                                    json_files.append(json_file)
                                    print(f"找到文件: {json_file.relative_to(root_path)}")
        
        return json_files
    
    def get_output_filename(self, input_path, suffix="_emotion_analysis"):
        """根据输入文件生成输出文件名 - 输出到同一文件夹"""
        input_path = Path(input_path)
        base_name = input_path.stem
        output_dir = input_path.parent  # 直接使用json文件所在的文件夹
        
        return {
            'summary_file': output_dir / f"{base_name}{suffix}_summary.txt",
            'detailed_file': output_dir / f"{base_name}{suffix}_detailed.json"
        }
    
    def is_large_file(self, file_path, max_lines=1000):
        """检查JSON文件是否包含过多行数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return len(data) > max_lines
                elif isinstance(data, dict) and 'lines' in data:
                    return len(data['lines']) > max_lines
                else:
                    return False
        except Exception:
            return False
    
    def process_single_file(self, file_path, thread_id=1):
        """处理单个文件"""
        try:
            with self.lock:
                self.processed_count += 1
                current_count = self.processed_count
            
            print(f"[线程{thread_id}] [{current_count}] 开始处理: {file_path.name}")
            
            # 检查输出文件是否已存在且处理成功
            output_files = self.get_output_filename(file_path)
            
            # 检查是否需要重新处理
            need_reprocess = False
            if output_files['summary_file'].exists():
                # 文件存在，检查是否处理成功
                if self.check_emotion_analysis_success(output_files['detailed_file']):
                    print(f"[线程{thread_id}] [{current_count}] ⏭️  跳过（已成功处理）: {file_path.name}")
                    with self.lock:
                        self.success_count += 1
                    return True, f"已存在且成功: {file_path}"
                else:
                    need_reprocess = True
                    with self.lock:
                        self.reprocessed_count += 1
                    print(f"[线程{thread_id}] [{current_count}] 🔄 重新处理失败的文件: {file_path.name}")
            
            # 创建情绪分析器
            analyzer = LLMEmotionAnalyzer(
                file_path=str(file_path),
                emovoice_api_config=self.config
            )
            
            # 运行情绪分析
            result = analyzer.run_emotion_analysis()
            
            if result:
                with self.lock:
                    self.success_count += 1
                status_msg = "重新处理完成" if need_reprocess else "完成"
                print(f"[线程{thread_id}] [{current_count}] ✅ {status_msg}: {file_path.name}")
                return True, f"成功: {file_path}"
            else:
                with self.lock:
                    self.failed_count += 1
                status_msg = "重新处理失败" if need_reprocess else "失败"
                print(f"[线程{thread_id}] [{current_count}] ❌ {status_msg}: {file_path.name}")
                return False, f"失败: {file_path}"
                
        except Exception as e:
            with self.lock:
                self.failed_count += 1
            error_msg = f"异常: {file_path} - {str(e)}"
            print(f"[线程{thread_id}] [{current_count}] 💥 {error_msg}")
            return False, error_msg
    
    def print_progress(self, total_files):
        """打印进度信息"""
        while self.processed_count < total_files:
            with self.lock:
                processed = self.processed_count
                success = self.success_count
                failed = self.failed_count
                reprocessed = self.reprocessed_count
            
            if self.start_time:
                elapsed = time.time() - self.start_time
                if processed > 0:
                    avg_time = elapsed / processed
                    remaining = (total_files - processed) * avg_time
                    eta = datetime.now().timestamp() + remaining
                    eta_str = datetime.fromtimestamp(eta).strftime('%H:%M:%S')
                else:
                    eta_str = "计算中..."
            else:
                eta_str = "计算中..."
            
            progress = (processed / total_files) * 100 if total_files > 0 else 0
            print(f"\r📊 进度: {processed}/{total_files} ({progress:.1f}%) | ✅{success} ❌{failed} 🔄{reprocessed} | ETA: {eta_str}", end="", flush=True)
            time.sleep(2)
    
    def batch_process(self, root_folder, pattern="*.json", skip_existing=True):
        """批量处理文件夹中的所有json文件"""
        print(f"🔍 扫描文件夹: {root_folder}")
        print(f"📋 文件模式: {pattern}")
        print(f"🧵 线程数: {self.max_workers}")
        
        # 查找所有json文件
        json_files = self.find_all_json_files(root_folder, pattern)
        
        if not json_files:
            print(f"❌ 在 {root_folder} 中未找到匹配 {pattern} 的文件")
            return
        
        print(f"📁 找到 {len(json_files)} 个文件")
        
        # 如果跳过已存在的文件，先过滤（但要检查是否处理成功）
        if skip_existing:
            filtered_files = []
            for file_path in json_files:
                output_files = self.get_output_filename(file_path)
                # 文件不存在或者处理失败的都需要处理
                if (not output_files['summary_file'].exists() or 
                    not self.check_emotion_analysis_success(output_files['detailed_file'])):
                    filtered_files.append(file_path)
            
            skipped_count = len(json_files) - len(filtered_files)
            if skipped_count > 0:
                print(f"⏭️  跳过 {skipped_count} 个已成功处理的文件")
            
            json_files = filtered_files
        
        if not json_files:
            print("✅ 所有文件都已成功处理完成！")
            return
        
        print(f"🚀 开始处理 {len(json_files)} 个文件...")
        
        # 重置计数器
        self.processed_count = 0
        self.success_count = 0
        self.failed_count = 0
        self.reprocessed_count = 0
        self.start_time = time.time()
        
        # 启动进度显示线程
        progress_thread = threading.Thread(
            target=self.print_progress, 
            args=(len(json_files),),
            daemon=True
        )
        progress_thread.start()
        
        # 使用线程池处理文件
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(self.process_single_file, file_path, i % self.max_workers + 1): file_path 
                for i, file_path in enumerate(json_files)
            }
            
            # 收集结果
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    success, message = future.result()
                    results.append((success, message))
                except Exception as e:
                    results.append((False, f"线程异常: {file_path} - {str(e)}"))
        
        # 等待进度线程结束
        time.sleep(1)
        print()  # 换行
        
        # 统计结果
        total_time = time.time() - self.start_time
        
        print(f"\n📊 批量情绪分析完成！")
        print(f"⏱️  总耗时: {total_time:.1f} 秒")
        print(f"📁 总文件数: {len(json_files)}")
        print(f"✅ 成功: {self.success_count}")
        print(f"❌ 失败: {self.failed_count}")
        print(f"🔄 重新处理: {self.reprocessed_count}")
        print(f"📈 成功率: {(self.success_count/len(json_files)*100):.1f}%")
        
        # 保存处理日志
        log_file = f"batch_emotion_analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"批量情绪分析日志 - {datetime.now()}\n")
            f.write(f"根文件夹: {root_folder}\n")
            f.write(f"文件模式: {pattern}\n")
            f.write(f"线程数: {self.max_workers}\n")
            f.write(f"总耗时: {total_time:.1f} 秒\n")
            f.write(f"成功: {self.success_count}/{len(json_files)}\n")
            f.write(f"重新处理: {self.reprocessed_count}\n\n")
            
            f.write("详细结果:\n")
            for success, message in results:
                status = "✅" if success else "❌"
                f.write(f"{status} {message}\n")
        
        print(f"📝 处理日志已保存到: {log_file}")

def main():
    parser = argparse.ArgumentParser(description='多线程批量情绪分析工具')
    parser.add_argument('root_folder', 
                       help='根文件夹路径（包含多个子文件夹）')
    parser.add_argument('--config', '-c', 
                       default='emovoice_api_config.json',
                       help='API配置文件路径（默认: emovoice_api_config.json）')
    parser.add_argument('--api-type', 
                       choices=['openai', 'deepseek', 'claude', 'custom'],
                       help='指定API类型（覆盖配置文件中的设置）')
    parser.add_argument('--pattern', '-p',
                       default='*.json',
                       help='文件匹配模式（默认: *.json）')
    parser.add_argument('--threads', '-t',
                       type=int,
                       default=4,
                       help='线程数（默认: 4）')
    parser.add_argument('--no-skip-existing',
                       action='store_true',
                       help='不跳过已存在的输出文件')
    
    args = parser.parse_args()
    
    # 检查根文件夹是否存在
    if not os.path.exists(args.root_folder):
        print(f"❌ 根文件夹不存在: {args.root_folder}")
        return
    
    if not os.path.isdir(args.root_folder):
        print(f"❌ 路径不是文件夹: {args.root_folder}")
        return
    
    # 创建处理器实例
    processor = BatchEmotionProcessor(config=None, max_workers=args.threads)
    
    # 加载配置
    config = processor.load_config(args.config)
    if not config:
        return
    
    processor.config = config
    
    # 如果命令行指定了API类型，覆盖配置文件中的设置
    if args.api_type:
        config['api_type'] = args.api_type
    
    # 检查API密钥
    api_type = config.get('api_type', 'openai')
    if api_type == 'openai' and not config.get('openai_api_key'):
        print("❌ 请在配置文件中设置OpenAI API密钥")
        return
    elif api_type == 'deepseek' and not config.get('deepseek_api_key'):
        print("❌ 请在配置文件中设置DeepSeek API密钥")
        return
    elif api_type == 'claude' and not config.get('claude_api_key'):
        print("❌ 请在配置文件中设置Claude API密钥")
        return
    
    print(f"🔧 配置文件: {args.config}")
    print(f"🚀 使用 {api_type.upper()} API")
    
    # 开始批量处理
    processor.batch_process(
        root_folder=args.root_folder,
        pattern=args.pattern,
        skip_existing=not args.no_skip_existing
    )

if __name__ == "__main__":
    main()
