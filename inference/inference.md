# Earnings Call Data Processing with PIMM Model

This document outlines the process for handling audio/video files and performing automated speech recognition (ASR) and emotion analysis for earnings call data using the **Physics-Informed Acoustic Model (PIMM)** framework.

## 1. Batch ASR Processing for Audio/Video Files

### Command Format
```bash
nohup python3 emovoice_batch_asr_result.py /data/algo_financial_data 1 > poll_run_3.log 2>&1 &
```

### Parameters
- `<audio_root_dir>`: Path to the root directory containing audio files. The script recursively searches for audio files in this directory.
- `[max_workers]`: Optional. Number of parallel processing threads (default: 5).

### Supported Audio Formats
- `.wav`: Processed directly.
- `.mp3`, `.mp4`, `.m4a`: Automatically converted to 16kHz mono WAV format.

### Output
- For each audio file, an ASR result file is generated in the same directory: `azero_full_results_{audio_filename}.txt`.
- The result file is in JSON format, with each line containing a JSON object with the following fields:
  - `timestamp`: Timestamp of the audio segment.
  - `mode`: Recognition mode.
  - `language`: Language type.
  - `text`: Transcribed text.
  - `is_final`: Indicates if the result is final.
  - `emotion`: Emotion information (if enabled).
  - `events`: Event information.
  - Other metadata fields.

### ASR Service Configuration
- Server IP: `172.16.150.18`

## 2. Batch Emotion Analysis for Text with PIMM

### Command Format
```bash
nohup python3 emovoice_batch_emotion_analysis.py /data/algo_financial_data --threads 5 --api-type deepseek > batch_emotion_analysis.log 2>&1 &
```

### Parameters
- `root_folder`: Path to the root directory containing JSON files to process.
- `--threads, -t`: Number of parallel processing threads (default: 4).
- `--api-type`: API type. Options: `openai`, `deepseek`, `claude`, `custom`.
- `--config, -c`: Path to the API configuration file (default: `emovoice_api_config.json`).
- `--pattern, -p`: File matching pattern (default: `*.json`).
- `--no-skip-existing`: Process all files, including those with existing output files.

### Output
For each processed JSON file, two output files are generated:
1. `{original_filename}_emotion_analysis_summary.txt`: Summary of emotion analysis using the PIMM model.
2. `{original_filename}_emotion_analysis_detailed.json`: Detailed emotion analysis results.

### Processing Logs
- Real-time progress display: `Progress: X/Y (percentage%) | ✅ Successes ❌ Failures 🔄 Retries | ETA: Estimated completion time`.
- Log file: `batch_emotion_analysis_log_YYYYMMDD_HHMMSS.txt`.

## 3. Merging Emotion Analysis Results with Original Text

### Command Format
```bash
python3 emovoice_merge_emotion_to_original.py --batch /data/algo_financial_data
```

### Parameters
- `path`: Path to the input text file or directory.
- `--batch, -b`: Process all `.txt` files in the specified directory.
- `--threads, -t`: Number of parallel processing threads (default: 4).
- `--pattern, -p`: File matching pattern (default: `*_with_speaker.txt`).
- `--emotion-results, -e`: Path to the emotion analysis JSON results.
- `--output, -o`: Output file path (optional).
- `--auto-find, -a`: Automatically locate corresponding emotion analysis result files (enabled by default).
- `--no-skip-existing`: Process all files, including those with existing output files.

### Output
- A new file with merged emotion data: `{original_filename}_with_emotion.txt`.
- The original ASR results are augmented with emotion analysis fields from the PIMM model.
- Processing log: `emotion_merge_processing_log_YYYYMMDD_HHMMSS.txt`.

### Processing Statistics
- Total processing time.
- Number of successful/failed files and success rate.
- Number of lines matched with emotion data.

## API Configuration File

The `emovoice_api_config.json` file must include the necessary API keys for the PIMM model integration:

```json
{
  "api_type": "deepseek",
  "deepseek_api_key": "your_deepseek_api_key",
  "deepseek_model": "deepseek-chat",
  "openai_api_key": "your_openai_api_key",
  "claude_api_key": "your_claude_api_key"
}
```

## Notes

1. **Dependencies**: Ensure `ffmpeg` is installed for audio format conversion.
2. **Storage**: ASR and emotion analysis generate large result files; ensure sufficient storage space.
3. **API Limits**: Be aware of rate limits and quotas imposed by API providers.
4. **Error Handling**: The scripts automatically skip processed files and support resuming from interruptions.
5. **Log Monitoring**: Regularly check log files to track progress and identify errors.