import json
import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from datetime import datetime

class BatchEmotionMergeProcessor:
    def __init__(self, max_workers=4):
        self.max_workers = max_workers
        self.lock = threading.Lock()
        self.processed_count = 0
        self.success_count = 0
        self.failed_count = 0
        self.start_time = None

    def load_emotion_results(self, emotion_results_file):
        """Load emotion analysis results"""
        try:
            with open(emotion_results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Create mapping from line number to emotion
                emotion_map = {}
                for result in data.get('results', []):
                    line_num = result.get('line')
                    emotion = result.get('emotion', 'none')
                    emotion_map[line_num] = emotion
                return emotion_map
        except Exception as e:
            print(f"ERROR: Failed to load emotion analysis results file: {e}")
            return None

    def merge_emotion_to_original(self, original_file, emotion_results_file, output_file=None):
        """Merge emotion information into original file"""
        
        # Load emotion analysis results
        emotion_map = self.load_emotion_results(emotion_results_file)
        if not emotion_map:
            return False
        
        # If no output file specified, add _with_emotion suffix to original filename
        if not output_file:
            original_path = Path(original_file)
            output_file = original_path.parent / f"{original_path.stem}_with_emotion{original_path.suffix}"
        
        try:
            processed_lines = 0
            matched_lines = 0
            
            with open(original_file, 'r', encoding='utf-8') as infile, \
                 open(output_file, 'w', encoding='utf-8') as outfile:
                
                for line_num, line in enumerate(infile, 1):
                    line = line.strip()
                    if line:
                        try:
                            # Parse JSON
                            data = json.loads(line)
                            
                            # Add emotion information
                            if line_num in emotion_map:
                                data['text_emotion'] = emotion_map[line_num]
                                matched_lines += 1
                            else:
                                data['text_emotion'] = 'none'
                            
                            # Write new JSON line
                            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                            processed_lines += 1
                            
                        except json.JSONDecodeError:
                            # If not valid JSON, write original line
                            outfile.write(line + '\n')
                            processed_lines += 1
                    else:
                        # Write empty line
                        outfile.write('\n')
            
            return True, processed_lines, matched_lines, str(output_file)
            
        except Exception as e:
            print(f"ERROR: Failed to process file: {e}")
            return False, 0, 0, ""

    def find_emotion_results_file(self, original_file):
        """Automatically find corresponding emotion analysis results file"""
        original_path = Path(original_file)
        base_name = original_path.stem
        
        # Ensure it's not a _with_emotion file
        if base_name.endswith('_with_emotion'):
            # If it's a _with_emotion file, remove suffix to get original filename
            base_name = base_name[:-13]  # Remove '_with_emotion'
        
        # If it's a _with_speaker file, remove suffix to get original filename
        if base_name.endswith('_with_speaker'):
            base_name = base_name[:-12]  # Remove '_with_speaker'
        
        # Build analysis subdirectory path
        analysis_dir = original_path.parent / f"{base_name}analysis"
        
        # Possible emotion analysis result file name patterns (in priority order)
        possible_patterns = [
            # 1. Standard emotion analysis file naming (in analysis subdirectory)
            f"{base_name}speaker_results_emotion_analysis_detailed.json",
            # 2. Simplified emotion analysis file naming
            f"{base_name}_emotion_analysis.json",
            # 3. Default filename
            "emotion_analysis_results.json"
        ]
        
        # First search in analysis subdirectory
        if analysis_dir.exists():
            for pattern in possible_patterns:
                candidate_file = analysis_dir / pattern
                if candidate_file.exists():
                    print(f"Found emotion analysis file: {candidate_file}")
                    return str(candidate_file)
        
        # If analysis directory doesn't exist or not found, search in original directory (backward compatibility)
        for pattern in possible_patterns:
            candidate_file = original_path.parent / pattern
            if candidate_file.exists():
                print(f"Found emotion analysis file: {candidate_file}")
                return str(candidate_file)
        
        # If not found, print debug information
        print(f"Failed to find emotion analysis file, tried paths:")
        if analysis_dir.exists():
            print(f"  Analysis directory: {analysis_dir}")
            for pattern in possible_patterns:
                candidate_file = analysis_dir / pattern
                print(f"    - {candidate_file} (exists: {candidate_file.exists()})")
        else:
            print(f"  Analysis directory does not exist: {analysis_dir}")
        
        print(f"  Original file directory: {original_path.parent}")
        for pattern in possible_patterns:
            candidate_file = original_path.parent / pattern
            print(f"    - {candidate_file} (exists: {candidate_file.exists()})")
        
        return None

    def find_all_txt_files(self, root_folder, pattern="*_with_speaker.txt"):
        """Find txt files in direct subfolders of root folder
        
        Rules:
        - Don't read txt files in root folder itself
        - Only read txt files in direct subfolders of root folder
        - Don't read txt files in sub-subfolders
        - Default search for files ending with _with_speaker
        - Exclude already processed _with_emotion files
        """
        txt_files = []
        root_path = Path(root_folder)
        
        # Traverse direct subfolders of root folder
        for item in root_path.iterdir():
            if item.is_dir():  # Only process subfolders
                # Find txt files in each subfolder (non-recursive)
                for txt_file in item.glob(pattern):
                    if txt_file.is_file():
                        # Exclude already processed _with_emotion files
                        if not txt_file.stem.endswith('_with_emotion'):
                            txt_files.append(txt_file)
                            print(f"Found file: {txt_file.relative_to(root_path)}")
                        else:
                            print(f"Skipping processed file: {txt_file.relative_to(root_path)}")
        
        return txt_files

    def get_output_filename(self, input_path, suffix="_with_emotion"):
        """Generate output filename based on input file"""
        input_path = Path(input_path)
        base_name = input_path.stem
        output_dir = input_path.parent
        
        return output_dir / f"{base_name}{suffix}{input_path.suffix}"

    def process_single_file(self, file_path, thread_id):
        """Process single file (thread-safe)"""
        try:
            with self.lock:
                self.processed_count += 1
                current_count = self.processed_count
            
            print(f"[Thread{thread_id}] [{current_count}] Processing: {file_path.name}")
            
            # Find corresponding emotion analysis results file
            emotion_results_file = self.find_emotion_results_file(file_path)
            if not emotion_results_file:
                with self.lock:
                    self.failed_count += 1
                error_msg = f"Emotion analysis results file not found: {file_path}"
                print(f"[Thread{thread_id}] [{current_count}] WARNING: {error_msg}")
                return False, error_msg
            
            # Check if output file already exists
            output_file = self.get_output_filename(file_path)
            if output_file.exists():
                print(f"[Thread{thread_id}] [{current_count}] SKIP (exists): {file_path.name}")
                with self.lock:
                    self.success_count += 1
                return True, f"Already exists: {file_path}"
            
            # Execute merge
            result = self.merge_emotion_to_original(
                str(file_path), 
                emotion_results_file, 
                str(output_file)
            )
            
            if result[0]:  # Success
                with self.lock:
                    self.success_count += 1
                processed_lines, matched_lines, output_path = result[1], result[2], result[3]
                print(f"[Thread{thread_id}] [{current_count}] SUCCESS: {file_path.name} (processed {processed_lines} lines, matched {matched_lines} lines)")
                return True, f"Success: {file_path} -> {output_path}"
            else:
                with self.lock:
                    self.failed_count += 1
                print(f"[Thread{thread_id}] [{current_count}] FAILED: {file_path.name}")
                return False, f"Failed: {file_path}"
                
        except Exception as e:
            with self.lock:
                self.failed_count += 1
            error_msg = f"Exception: {file_path} - {str(e)}"
            print(f"[Thread{thread_id}] [{current_count}] EXCEPTION: {error_msg}")
            return False, error_msg

    def print_progress(self, total_files):
        """Print progress information"""
        while self.processed_count < total_files:
            with self.lock:
                processed = self.processed_count
                success = self.success_count
                failed = self.failed_count
            
            if self.start_time:
                elapsed = time.time() - self.start_time
                if processed > 0:
                    avg_time = elapsed / processed
                    remaining = (total_files - processed) * avg_time
                    eta = datetime.now().timestamp() + remaining
                    eta_str = datetime.fromtimestamp(eta).strftime('%H:%M:%S')
                else:
                    eta_str = "Calculating..."
            else:
                eta_str = "Calculating..."
            
            progress = (processed / total_files) * 100 if total_files > 0 else 0
            print(f"\rProgress: {processed}/{total_files} ({progress:.1f}%) | Success:{success} Failed:{failed} | ETA: {eta_str}", end="", flush=True)
            time.sleep(2)

    def batch_process(self, root_folder, pattern="*_with_speaker.txt", skip_existing=True):
        """Batch process all txt files in folder"""
        print(f"Scanning folder: {root_folder}")
        print(f"File pattern: {pattern}")
        print(f"Thread count: {self.max_workers}")
        
        # Find all txt files
        txt_files = self.find_all_txt_files(root_folder, pattern)
        
        if not txt_files:
            print(f"ERROR: No files matching {pattern} found in {root_folder}")
            return
        
        print(f"Found {len(txt_files)} files")
        
        # Filter out existing files if skip_existing is True
        if skip_existing:
            filtered_files = []
            for file_path in txt_files:
                output_file = self.get_output_filename(file_path)
                if not output_file.exists():
                    filtered_files.append(file_path)
            
            skipped_count = len(txt_files) - len(filtered_files)
            if skipped_count > 0:
                print(f"Skipping {skipped_count} already processed files")
            
            txt_files = filtered_files
        
        if not txt_files:
            print("SUCCESS: All files have been processed!")
            return
        
        print(f"Starting to process {len(txt_files)} files...")
        
        # Reset counters
        self.processed_count = 0
        self.success_count = 0
        self.failed_count = 0
        self.start_time = time.time()
        
        # Start progress display thread
        progress_thread = threading.Thread(
            target=self.print_progress, 
            args=(len(txt_files),),
            daemon=True
        )
        progress_thread.start()
        
        # Use thread pool to process files
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_single_file, file_path, i % self.max_workers + 1): file_path 
                for i, file_path in enumerate(txt_files)
            }
            
            # Collect results
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    success, message = future.result()
                    results.append((success, message))
                except Exception as e:
                    results.append((False, f"Thread exception: {file_path} - {str(e)}"))
        
        # Wait for progress thread to finish
        time.sleep(1)
        print()  # New line
        
        # Statistics
        total_time = time.time() - self.start_time
        
        print(f"\nBatch processing completed!")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Total files: {len(txt_files)}")
        print(f"Success: {self.success_count}")
        print(f"Failed: {self.failed_count}")
        print(f"Success rate: {(self.success_count/len(txt_files)*100):.1f}%")
        
        # Save processing log
        log_file = f"emotion_merge_processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Emotion merge batch processing log - {datetime.now()}\n")
            f.write(f"Root folder: {root_folder}\n")
            f.write(f"File pattern: {pattern}\n")
            f.write(f"Thread count: {self.max_workers}\n")
            f.write(f"Total time: {total_time:.1f} seconds\n")
            f.write(f"Success: {self.success_count}/{len(txt_files)}\n\n")
            
            f.write("Detailed results:\n")
            for success, message in results:
                status = "SUCCESS" if success else "FAILED"
                f.write(f"{status} {message}\n")
        
        print(f"Processing log saved to: {log_file}")

# Keep original single file processing functions for backward compatibility
def load_emotion_results(emotion_results_file):
    """Load emotion analysis results"""
    processor = BatchEmotionMergeProcessor()
    return processor.load_emotion_results(emotion_results_file)

def merge_emotion_to_original(original_file, emotion_results_file, output_file=None):
    """Merge emotion information into original file"""
    processor = BatchEmotionMergeProcessor()
    result = processor.merge_emotion_to_original(original_file, emotion_results_file, output_file)
    
    if result[0]:  # Success
        processed_lines, matched_lines, output_path = result[1], result[2], result[3]
        print(f"SUCCESS: Processing completed!")
        print(f"Total processed lines: {processed_lines}")
        print(f"Lines with emotion matched: {matched_lines}")
        print(f"Output file: {output_path}")
        return True
    else:
        return False

def find_emotion_results_file(original_file):
    """Automatically find corresponding emotion analysis results file"""
    processor = BatchEmotionMergeProcessor()
    result = processor.find_emotion_results_file(original_file)
    if result:
        print(f"Found emotion analysis results file: {result}")
    return result

def batch_process_directory(directory_path, auto_find=True):
    """Batch process all txt files in directory (keep original interface)"""
    processor = BatchEmotionMergeProcessor(max_workers=1)  # Single thread mode for compatibility
    
    directory = Path(directory_path)
    if not directory.exists() or not directory.is_dir():
        print(f"ERROR: Directory does not exist or is not a valid directory: {directory_path}")
        return
    
    # Find all txt files
    txt_files = list(directory.glob("*.txt"))
    if not txt_files:
        print(f"ERROR: No txt files found in directory {directory_path}")
        return
    
    print(f"Found {len(txt_files)} txt files in directory {directory_path}")
    
    success_count = 0
    for txt_file in txt_files:
        print(f"\nProcessing file: {txt_file.name}")
        
        if auto_find:
            emotion_results_file = processor.find_emotion_results_file(txt_file)
            if not emotion_results_file:
                print(f"WARNING: Corresponding emotion analysis results file not found, skipping: {txt_file.name}")
                continue
        else:
            print(f"WARNING: Need to manually specify emotion analysis results file, skipping: {txt_file.name}")
            continue
        
        # Execute merge
        success = merge_emotion_to_original(str(txt_file), emotion_results_file)
        if success:
            success_count += 1
    
    print(f"\nBatch processing completed! Successfully processed {success_count}/{len(txt_files)} files")

def main():
    parser = argparse.ArgumentParser(description='Merge emotion analysis results into original files (supports batch processing)')
    parser.add_argument('path', help='Original text file path or directory path')
    parser.add_argument('--emotion-results', '-e', help='Emotion analysis results JSON file path')
    parser.add_argument('--output', '-o', help='Output file path (optional)')
    parser.add_argument('--auto-find', '-a', action='store_true', default=True,
                       help='Automatically find corresponding emotion analysis results file (default: enabled)')
    parser.add_argument('--batch', '-b', action='store_true',
                       help='Batch process all txt files in directory')
    parser.add_argument('--threads', '-t', type=int, default=4,
                       help='Number of threads for batch processing (default: 4)')
    parser.add_argument('--pattern', '-p', default='*_with_speaker.txt',
                       help='File matching pattern for batch processing (default: *_with_speaker.txt)')
    parser.add_argument('--no-skip-existing', action='store_true',
                       help='Do not skip existing output files')
    
    args = parser.parse_args()
    
    # Check if path exists
    if not os.path.exists(args.path):
        print(f"ERROR: Path does not exist: {args.path}")
        return
    
    # Determine batch processing or single file processing
    if args.batch or os.path.isdir(args.path):
        # Batch process directory
        processor = BatchEmotionMergeProcessor(max_workers=args.threads)
        processor.batch_process(
            args.path, 
            pattern=args.pattern,
            skip_existing=not args.no_skip_existing
        )
        return
    
    # Single file processing
    if not os.path.isfile(args.path):
        print(f"ERROR: Not a valid file: {args.path}")
        return
    
    # Determine emotion analysis results file
    emotion_results_file = args.emotion_results
    
    if args.auto_find or not emotion_results_file:
        print("Automatically searching for emotion analysis results file...")
        emotion_results_file = find_emotion_results_file(args.path)
        if not emotion_results_file:
            print("ERROR: Corresponding emotion analysis results file not found")
            print("Please use --emotion-results parameter to specify file path")
            return
    
    # Check if emotion analysis results file exists
    if not os.path.exists(emotion_results_file):
        print(f"ERROR: Emotion analysis results file does not exist: {emotion_results_file}")
        return
    
    print(f"Original file: {args.path}")
    print(f"Emotion analysis results file: {emotion_results_file}")
    
    # Execute merge
    success = merge_emotion_to_original(args.path, emotion_results_file, args.output)
    if success:
        print("Processing completed successfully!")
    else:
        print("Processing failed!")

if __name__ == '__main__':
    main()
