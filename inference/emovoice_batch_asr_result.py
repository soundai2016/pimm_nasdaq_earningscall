import glob
import subprocess
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import datetime

if len(sys.argv) < 2:
    print("Usage: python poll_all_audio_result.py <audio_root_dir> [max_workers]")
    sys.exit(1)

audio_root = sys.argv[1]
max_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 5
audio_files = sorted(
    glob.glob(f'{audio_root}/*/*.wav') +
    glob.glob(f'{audio_root}/*/*.mp3') +
    glob.glob(f'{audio_root}/*/*.mp4') +
    glob.glob(f'{audio_root}/*/*.m4a')
)

def process_audio(audio, idx=None):
    thread_name = threading.current_thread().name
    def log(msg):
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{now}][{thread_name}] {msg}")
    ext = os.path.splitext(audio)[1].lower()
    base = os.path.splitext(os.path.basename(audio))[0]
    parent_dir = os.path.dirname(audio)
    result_path = os.path.join(parent_dir, f"azero_full_results_{base}.txt")
    if os.path.exists(result_path):
        log(f"Skip {audio}, result exists: {result_path}")
        return
    log(f"Start processing: {audio}")
    if ext in ['.mp3', '.mp4', '.m4a']:
        wav_path = os.path.splitext(audio)[0] + '.wav'
        if not os.path.exists(wav_path):
            log(f"Converting {audio} to {wav_path}")
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-i', audio,
                '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', '-sample_fmt', 's16', wav_path
            ]
            subprocess.run(ffmpeg_cmd, check=True)
        audio = wav_path
    cmd = [
        sys.executable, 'asr_stream.py',
        '--server-ip', '172.16.150.18',
        '--wav-path', audio,
        '--nbest', '3',
        '--language', 'en',
        '--result-path', result_path
    ]
    log(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
    log(f"Finished: {audio}")

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    for idx, audio in enumerate(audio_files):
        futures.append(executor.submit(process_audio, audio, idx))
    for future in as_completed(futures):
        future.result()
