"""
Author: chenxiaoliang <chenxiaoliang@soundai.com>
Date: 2025-03-31 17:41:31 (Original)
LastEditors: Refactored by AI for clarity and improved handling
LastEditTime: 2025-03-31 (Refactored for concurrent send/receive focus)
FilePath: asr_stream.py
Description: AzeroOmni Test Client (Refactored for clarity, paced file sending, immediate receive, and graceful shutdown,
             emphasizing independent concurrent execution of send and receive tasks)
"""

import os
import ssl
import json
import wave
import argparse
import asyncio
import datetime
import logging
from multiprocessing import Process
import websockets

def print_with_time(*args, **kwargs):
    now = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S.%f]")[:-3]
    print(now, *args, **kwargs)



# Print websockets version for diagnostics
try:
    print_with_time(f"INFO: websockets library version: {websockets.__version__}")
except AttributeError:
    print_with_time("WARNING: Could not determine websockets library version (websockets.__version__ not found).")

# Import State enum based on common locations
try:
    from websockets.enums import State # Common in v10+
    print_with_time("INFO: Imported 'State' from 'websockets.enums'")
except ImportError:
    try:
        from websockets.connection import State # Common in v8, v9; also available in v10, v11 as an alias
        print_with_time("INFO: Imported 'State' from 'websockets.connection'")
    except ImportError:
        # This is a fallback if State cannot be imported from common locations.
        print_with_time("CRITICAL: Failed to import 'State' from 'websockets.enums' or 'websockets.connection'. "
              "Connection state checks will likely fail or behave unexpectedly. "
              "Please check your websockets library installation and version.")
        # Define a minimal dummy State object to allow the script to load,
        # though it's unlikely to function correctly if state checks are critical.
        class _MinimalDummyState:
            def __init__(self):
                self.OPEN = "OPEN_DUMMY_STATE" # Using unique string to avoid accidental matches
                # Add other states (e.g., CLOSED, CONNECTING, CLOSING) if they are explicitly
                # checked against in the code, otherwise this minimal version might suffice
                # if only `State.OPEN` is ever checked.
        State = _MinimalDummyState()
        print_with_time(f"WARNING: Using a dummy 'State' object: {State.OPEN}")


# Configure logging level
logging.basicConfig(level=logging.ERROR) # Use logging.INFO for more verbose websockets library logs

# Constants
DEFAULT_HOST_IP = "127.0.0.1"
DEFAULT_PORT = 2035
DEFAULT_NBEST = 3
DEFAULT_DENOISE = 0
WEBSOCKET_RECEIVE_TIMEOUT_AFTER_FILE_SEND = 10.0 # Seconds to wait for final messages after sending a file

# Audio stream constants
AUDIO_RATE = 16000  # Sample rate in Hz
MICROPHONE_CHUNK_MS = 40  # Chunk size in milliseconds
AUDIO_SAMPLE_WIDTH = 2  # Bytes per sample for 16-bit PCM

# Global WebSocket connection (managed within ws_client_session)
websocket: websockets.WebSocketClientProtocol | None = None # Type hint uses deprecated name, but this is fine for runtime.

# Parse command-line arguments
parser = argparse.ArgumentParser(description="AzeroOmni gASR Streaming Test Client")
parser.add_argument("--server-ip", type=str, default=DEFAULT_HOST_IP, help="Server IP address")
parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
parser.add_argument("--wav-path", type=str, default=None, help="Path to WAV file. If None, use microphone.")
parser.add_argument("--hotword", type=str, default="hotwords/hotwords.json", help="Path to hotword JSON file")
parser.add_argument("--language", type=str, default="zh", help="Source language code (e.g., zh, en)")
parser.add_argument("--target", type=str, default="en", help="Target language code for translation (e.g., en, zh)")
parser.add_argument("--nbest", type=int, default=DEFAULT_NBEST, help="Number of n-best hypotheses to return")
parser.add_argument("--denoise", type=int, default=DEFAULT_DENOISE, help="Denoise option (implementation specific)")
parser.add_argument("--result-path", type=str, default=None, help="Path to result file. If None, print to console.")
args = parser.parse_args()


class EchoMindResponse:
    """Model for responses from EchoMind API."""
    def __init__(self, is_final=False, mode="azero-part", text="", text_tts="",
                 translate="", translate_tts="", wav_name="", embedding=None, timestamp="",
                 language="auto", target="auto", events="", emotion="", think=""):
        self.is_final = is_final
        self.mode = mode
        self.text = text
        self.text_tts = text_tts
        self.translate = translate
        self.translate_tts = translate_tts
        self.wav_name = wav_name
        self.embedding = embedding if embedding else [0.0, 0.0]
        self.timestamp = timestamp
        self.language = language
        self.target = target
        self.events = events
        self.emotion = emotion
        self.think = think
        self.reserved = ""

    def update_from_dict(self, response_dict: dict):
        """Update the response attributes from a dictionary."""
        for key, value in response_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # print_with_time(f"Warning: Unknown key '{key}' in response.")
                self.reserved += f"{key}:{value};"


class EchoMindRequest:
    """Model for requests to EchoMind API."""
    def __init__(self, name="echomind", signal="start", nbest=3,
                 wav_name="microphone", hotwords="{}", language="auto",
                 denoise=0, spkname="", spkpara="", target="en", others=""):
        self.name = name
        self.signal = signal
        self.nbest = nbest
        self.wav_name = wav_name
        self.hotwords = hotwords
        self.language = language
        self.denoise = denoise
        self.spkname = spkname
        self.spkpara = spkpara
        self.target = target
        self.others = others

    def to_dict(self) -> dict:
        """Convert request object to dictionary format for sending."""
        return self.__dict__


def load_hotwords(filepath: str) -> str:
    """Load hotwords from JSON file and return as JSON string."""
    if not filepath or not os.path.exists(filepath):
        print_with_time(f"Hotword file not found or path is empty: {filepath}. Using empty hotwords.")
        return "{}"
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.dumps(json.load(f))
    except json.JSONDecodeError:
        print_with_time(f"Error decoding hotword JSON from {filepath}. Please check file format. Using empty hotwords.")
    except Exception as e:
        print_with_time(f"Error loading hotword file {filepath}: {e}. Using empty hotwords.")
    return "{}"


def prepare_request_message(request: EchoMindRequest) -> str:
    """Construct the JSON message for the WebSocket session from EchoMindRequest."""
    return json.dumps(request.to_dict())


def initialize_microphone_stream():
    """Initialize a PyAudio stream for microphone input."""
    try:
        import pyaudio
    except ImportError:
        print_with_time("Error: PyAudio module not found. Please install it to use microphone input (pip install PyAudio).")
        raise
    
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=AUDIO_RATE,
                    input=True,
                    frames_per_buffer=int(AUDIO_RATE / 1000 * MICROPHONE_CHUNK_MS))
    print_with_time("Microphone stream initialized.")
    return p, stream


async def send_initial_request(wav_name_param: str, hotword_msg: str):
    """Send the 'start' signal message to the server."""
    global websocket
    if not websocket or not hasattr(websocket, 'state') or websocket.state != State.OPEN:
        msg = "WebSocket connection not established or not open. Cannot send initial request."
        print_with_time(msg)
        # Ensure State.OPEN is compared correctly, even if State is the dummy object
        if not hasattr(State, 'OPEN'): 
             raise websockets.exceptions.InvalidState("Dummy State object missing OPEN attribute.")
        raise websockets.exceptions.ConnectionClosed(None, msg)


    request = EchoMindRequest(
        signal="start", 
        wav_name=wav_name_param,
        hotwords=hotword_msg,
        language=args.language,
        target=args.target,
        nbest=args.nbest,
        denoise=args.denoise
    )
    init_msg = prepare_request_message(request)
    try:
        await websocket.send(init_msg)
        print_with_time(f"Sent initial request for {wav_name_param}")
    except websockets.exceptions.ConnectionClosed:
        print_with_time("WebSocket closed before sending initial request.")
        raise
    except Exception as e:
        print_with_time(f"Error sending initial request: {e}")
        raise


async def record_and_send_from_microphone():
    """Continuously read from microphone and send audio to server."""
    global websocket
    if not websocket or not hasattr(websocket, 'state') or websocket.state != State.OPEN:
        print_with_time("WebSocket connection not established or not open. Cannot record from microphone.")
        return

    pyaudio_instance, stream = None, None
    try:
        pyaudio_instance, stream = initialize_microphone_stream()
        await send_initial_request("microphone", load_hotwords(args.hotword))
        
        chunk_size_frames = int(AUDIO_RATE / 1000 * MICROPHONE_CHUNK_MS)

        print_with_time("Starting microphone recording and streaming...")
        while websocket and hasattr(websocket, 'state') and websocket.state == State.OPEN: 
            data = stream.read(chunk_size_frames, exception_on_overflow=False)
            await websocket.send(data)
            await asyncio.sleep(0.001) # Yield control briefly
    except websockets.exceptions.ConnectionClosed:
        print_with_time("WebSocket connection closed during microphone streaming.")
    except asyncio.CancelledError:
        print_with_time("Microphone sending task cancelled.")
    except Exception as e:
        print_with_time(f"Error during microphone recording or sending: {e}")
    finally:
        print_with_time("Stopping microphone stream...")
        if stream:
            stream.stop_stream()
            stream.close()
        if pyaudio_instance:
            pyaudio_instance.terminate()
        print_with_time("Microphone stream stopped and resources released.")


def get_audio_bytes_from_file(wav_path: str) -> bytes:
    """Read raw audio bytes from a .pcm or .wav file."""
    if not os.path.exists(wav_path):
        print_with_time(f"Audio file not found: {wav_path}")
        return b""

    try:
        if wav_path.lower().endswith(".pcm"):
            with open(wav_path, "rb") as f:
                audio_bytes = f.read()
                if len(audio_bytes) % AUDIO_SAMPLE_WIDTH != 0:
                    print_with_time(f"Warning: PCM file size ({len(audio_bytes)} bytes) is not a multiple "
                          f"of {AUDIO_SAMPLE_WIDTH} bytes (sample width).")
                return audio_bytes
        elif wav_path.lower().endswith(".wav"):
            with wave.open(wav_path, "rb") as wav_file:
                if wav_file.getframerate() != AUDIO_RATE:
                    print_with_time(f"Warning: WAV file sample rate ({wav_file.getframerate()}) "
                          f"does not match expected {AUDIO_RATE}. Results may be affected.")
                if wav_file.getsampwidth() != AUDIO_SAMPLE_WIDTH:
                    print_with_time(f"Warning: WAV file sample width ({wav_file.getsampwidth()} bytes) "
                          f"does not match expected {AUDIO_SAMPLE_WIDTH} bytes. Results may be affected.")
                if wav_file.getnchannels() != 1:
                    print_with_time(f"Warning: WAV file has {wav_file.getnchannels()} channels. "
                          "Expected 1 (mono). Results may be affected.")
                return wav_file.readframes(wav_file.getnframes())
        else:
            print_with_time(f"Unsupported audio file format: {wav_path}. Please use .pcm or .wav.")
            return b""
    except Exception as e:
        print_with_time(f"Error reading audio file {wav_path}: {e}")
        return b""

async def send_file_audio_in_chunks():
    """Read audio from file, send in paced chunks, then send 'end' signal."""
    global websocket
    if not websocket or not args.wav_path:
        print_with_time("WebSocket not ready or WAV path not specified for file streaming.")
        return
    if not hasattr(websocket, 'state') or websocket.state != State.OPEN: # Initial check
        print_with_time(f"WebSocket not open at start of send_file_audio_in_chunks for {args.wav_path}.")
        return


    wav_path = args.wav_path.strip()
    audio_bytes = get_audio_bytes_from_file(wav_path)
    
    await send_initial_request(os.path.basename(wav_path), load_hotwords(args.hotword))

    if not audio_bytes:
        print_with_time(f"No audio data read from {wav_path}. Only 'start' and 'end' signals will be sent.")
        await send_end_signal(os.path.basename(wav_path if wav_path else "unknown_empty_file"))
        return

    bytes_per_chunk = int(AUDIO_RATE / 1000 * MICROPHONE_CHUNK_MS) * AUDIO_SAMPLE_WIDTH
    sleep_duration_s = MICROPHONE_CHUNK_MS / 1100.0

    if bytes_per_chunk == 0:
        print_with_time("Error: Calculated bytes_per_chunk is zero. Check audio constants. Cannot send audio.")
        await send_end_signal(os.path.basename(wav_path))
        return

    total_bytes = len(audio_bytes)
    bytes_sent = 0
    chunk_index = 0

    print_with_time(f"Starting to send audio from {wav_path} in {MICROPHONE_CHUNK_MS}ms chunks ({bytes_per_chunk} bytes each).")
    try:
        while bytes_sent < total_bytes and websocket and hasattr(websocket, 'state') and websocket.state == State.OPEN:
            chunk = audio_bytes[bytes_sent : bytes_sent + bytes_per_chunk]
            if not chunk:
                break 
            
            await websocket.send(chunk)
            bytes_sent += len(chunk)
            chunk_index += 1
            
            if bytes_sent < total_bytes:
                 await asyncio.sleep(sleep_duration_s) # Pacing
            else: # Last chunk sent
                await asyncio.sleep(0.001) # Small yield before sending end signal

        print_with_time(f"Finished sending all audio data ({bytes_sent} bytes in {chunk_index} chunks) from {wav_path}.")
    except websockets.exceptions.ConnectionClosed: # This can be raised by send_initial_request or send_end_signal too
        print_with_time("WebSocket connection closed during file audio sending processing.")
    except asyncio.CancelledError:
        print_with_time("File audio sending task cancelled.")
    except Exception as e:
        print_with_time(f"Error sending audio data in chunks: {e}")
    finally:
        if websocket and hasattr(websocket, 'state') and websocket.state == State.OPEN:
            try:
                print_with_time('**************end start ********')
                await send_end_signal(os.path.basename(wav_path))
            except websockets.exceptions.ConnectionClosed:
                 print_with_time(f"WebSocket closed before 'end' signal could be sent for file {os.path.basename(wav_path)} (in finally block).")
            except Exception as e_final:
                print_with_time(f"Error sending 'end' signal in finally block for {os.path.basename(wav_path)}: {e_final}")
        elif websocket and hasattr(websocket, 'state'):
            print_with_time(f"WebSocket was not open (state: {websocket.state}) before 'end' signal could be sent for file in finally.")
        elif websocket: # hasattr is false, means no 'state' attribute
             print_with_time(f"WebSocket object (type: {type(websocket)}) missing 'state' attribute in finally block of send_file_audio_in_chunks.")
        else: # websocket is None
            print_with_time("WebSocket was None before 'end' signal could be sent for file in finally.")


async def send_end_signal(wav_name_param: str):
    """Send the 'end' signal to the server."""
    global websocket
    if not websocket or not hasattr(websocket, 'state') or websocket.state != State.OPEN:
        msg = f"WebSocket connection not established or not open. Cannot send end signal for {wav_name_param}."
        print_with_time(msg)
        if not hasattr(State, 'OPEN'):
             raise websockets.exceptions.InvalidState("Dummy State object missing OPEN attribute.")
        raise websockets.exceptions.ConnectionClosed(None, msg)
    
    request = EchoMindRequest(
        signal="end", 
        wav_name=wav_name_param,
        hotwords="{}", 
        language=args.language, 
        target=args.target,   
        nbest=args.nbest,
        denoise=args.denoise
    )
    end_msg = prepare_request_message(request)
    try:
        await websocket.send(end_msg)
        print_with_time(f"Sent 'end' signal for {wav_name_param}.")
    except websockets.exceptions.ConnectionClosed:
        print_with_time(f"WebSocket closed before sending 'end' signal for {wav_name_param}.")
        raise 
    except asyncio.CancelledError:
        print_with_time(f"Sending 'end' signal task for {wav_name_param} was cancelled.")
        raise 
    except Exception as e:
        print_with_time(f"Error sending 'end' signal for {wav_name_param}: {e}")
        raise 


async def receive_messages():
    """Continuously receive and process messages from the server."""
    global websocket
    if not websocket or not hasattr(websocket, 'state') or websocket.state != State.OPEN:
        print_with_time("WebSocket connection not established or not open. Cannot receive messages.")
        return True 

    print_with_time("Listening for messages from server...")
    raw_message_content_for_error_log = "" 
    try:
        while websocket and hasattr(websocket, 'state') and websocket.state == State.OPEN:
            raw_message = await websocket.recv()
            if not isinstance(raw_message, str):
                print_with_time(f"Received non-text message, ignoring: {type(raw_message)}")
                continue
            
            raw_message_content_for_error_log = raw_message

            msg_dict = json.loads(raw_message)
            response = EchoMindResponse()
            response.update_from_dict(msg_dict)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            print_with_time(f"{timestamp} | Mode: {response.mode} | Lang: {response.language} -> {response.target} | "
                  f"Text: '{response.text}' | TTS: '{response.text_tts}' | "
                  f"Translate: '{response.translate}' | TTS_Translate: '{response.translate_tts}' | "
                  f"Final: {response.is_final} | Emotion: {response.emotion} | Events: {response.events} | Think: {response.think}")

            # 新增：mode为azero-full时，写入一行json
            if response.mode == "azero-full":
                try:
                    # 构造所有字段的dict
                    record = {
                        "timestamp": timestamp,
                        "mode": response.mode,
                        "language": response.language,
                        "target": response.target,
                        "text": response.text,
                        "text_tts": response.text_tts,
                        "translate": response.translate,
                        "translate_tts": response.translate_tts,
                        "is_final": response.is_final,
                        "emotion": response.emotion,
                        "events": response.events,
                        "think": response.think,
                        "wav_name": getattr(response, "wav_name", None),
                        "embedding": getattr(response, "embedding", None),
                        "reserved": getattr(response, "reserved", None)
                    }
                    # 假设已通过 argparse 增加 --result-path 参数
                    result_path = args.result_path if hasattr(args, 'result_path') and args.result_path else "azero_full_results.txt"
                    with open(result_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception as e:
                    print_with_time(f"写入azero_full_results.txt失败: {e}")

            if response.is_final:
                print_with_time("Final message received. Message listener will now stop after this message.")
                return True 
    except websockets.exceptions.ConnectionClosedOK:
        print_with_time("WebSocket connection closed gracefully by server during receive.")
        return True 
    except websockets.exceptions.ConnectionClosedError as e:
        print_with_time(f"WebSocket connection closed with error during receive: {e}")
        return True 
    except json.JSONDecodeError as e:
        print_with_time(f"Error decoding JSON message from server: {e}. Message: '{raw_message_content_for_error_log}'")
    except asyncio.CancelledError:
        print_with_time("Receive messages task was cancelled.")
        return True 
    except Exception as e:
        print_with_time(f"Unexpected error in receive_messages(): {e}")
        return True


async def _handle_task_results(results, tasks, client_name, mode_name_log_prefix):
    """Helper to log task completion statuses and exceptions."""
    for i, res_or_exc in enumerate(results):
        task_name = tasks[i].get_name() if hasattr(tasks[i], 'get_name') and callable(tasks[i].get_name) else f"UnknownTask_{i}"
        
        if isinstance(res_or_exc, Exception) and not isinstance(res_or_exc, asyncio.CancelledError):
            log_msg = f"[{client_name}] {mode_name_log_prefix}: Task '{task_name}' finished with an exception: {res_or_exc}"
            logging.error(log_msg, exc_info=False) 
            print_with_time(log_msg)
        elif isinstance(res_or_exc, asyncio.CancelledError):
            print_with_time(f"[{client_name}] {mode_name_log_prefix}: Task '{task_name}' was cancelled.")


async def _manage_file_mode_session(client_name: str, send_task: asyncio.Task, receive_task: asyncio.Task):
    """Manages the lifecycle of send and receive tasks for file mode."""
    send_task_failed_or_cancelled = False
    send_task_name = send_task.get_name() if hasattr(send_task, 'get_name') and callable(send_task.get_name) else "send_task"
    receive_task_name = receive_task.get_name() if hasattr(receive_task, 'get_name') and callable(receive_task.get_name) else "receive_task"
    try:
        print_with_time(f"[{client_name}] File Mode: Send task ('{send_task_name}') is active. Receive task ('{receive_task_name}') is also active concurrently.")
        await send_task
        
        if send_task.done():
            if send_task.cancelled():
                print_with_time(f"[{client_name}] File Mode: Send task ('{send_task_name}') was cancelled.")
                send_task_failed_or_cancelled = True
            elif send_task.exception():
                exc = send_task.exception()
                print_with_time(f"[{client_name}] File Mode: Send task ('{send_task_name}') finished with error: {exc}")
                send_task_failed_or_cancelled = True
            else:
                 print_with_time(f"[{client_name}] File Mode: Send task ('{send_task_name}') has completed successfully.")
        else: 
            print_with_time(f"[{client_name}] File Mode: Send task ('{send_task_name}') finished but state is unexpected (not done).")
            send_task_failed_or_cancelled = True

    except asyncio.CancelledError:
        print_with_time(f"[{client_name}] File Mode: Send task ('{send_task_name}') was cancelled externally (while being awaited).")
        send_task_failed_or_cancelled = True
    except Exception as e: 
        print_with_time(f"[{client_name}] File Mode: Send task ('{send_task_name}') raised an exception during await: {e}")
        send_task_failed_or_cancelled = True
    
    if send_task_failed_or_cancelled:
        if not receive_task.done():
            print_with_time(f"[{client_name}] File Mode: Cancelling receive task ('{receive_task_name}') due to send task issue.")
            receive_task.cancel()
        await asyncio.gather(receive_task, return_exceptions=True)
        print_with_time(f"[{client_name}] File Mode: Session ending early due to send task issue.")
        return

    if not receive_task.done():
        print_with_time(f"[{client_name}] File Mode: Awaiting receive task ('{receive_task_name}') for final messages with timeout ({WEBSOCKET_RECEIVE_TIMEOUT_AFTER_FILE_SEND}s).")
        try:
            await asyncio.wait_for(receive_task, timeout=WEBSOCKET_RECEIVE_TIMEOUT_AFTER_FILE_SEND)
            if receive_task.done():
                if receive_task.cancelled():
                     print_with_time(f"[{client_name}] File Mode: Receive task ('{receive_task_name}') was cancelled during wait_for.")
                elif receive_task.exception():
                     print_with_time(f"[{client_name}] File Mode: Receive task ('{receive_task_name}') completed with error: {receive_task.exception()}")
                else:
                     print_with_time(f"[{client_name}] File Mode: Receive task ('{receive_task_name}') completed gracefully or was already done before timeout.")
        except asyncio.TimeoutError:
            print_with_time(f"[{client_name}] File Mode: Timeout waiting for receive task ('{receive_task_name}'). Cancelling.")
            if not receive_task.done():
                receive_task.cancel()
        except asyncio.CancelledError: 
            print_with_time(f"[{client_name}] File Mode: Receive task ('{receive_task_name}') was already cancelled when wait_for started.")
        except Exception as e_recv: 
            print_with_time(f"[{client_name}] File Mode: Receive task ('{receive_task_name}') failed during wait_for: {e_recv}")
    else: 
        if receive_task.cancelled():
            print_with_time(f"[{client_name}] File Mode: Receive task ('{receive_task_name}') was already cancelled when send_task finished.")
        elif receive_task.exception():
            print_with_time(f"[{client_name}] File Mode: Receive task ('{receive_task_name}') was already completed with error: {receive_task.exception()}")
        else:
            print_with_time(f"[{client_name}] File Mode: Receive task ('{receive_task_name}') was already completed successfully when send_task finished.")
    
    if not receive_task.done():
        await asyncio.gather(receive_task, return_exceptions=True)
    print_with_time(f"[{client_name}] File Mode: Session management complete.")


async def ws_client_session(client_name: str):
    """Establish WebSocket connection and manage send/receive tasks."""
    global websocket
    
    ssl_context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    uri = f"wss://{args.server_ip}:{args.port}/azeroasr/gasr/v1/streaming"
    print_with_time(f"[{client_name}] Connecting to WebSocket: {uri}")

    send_task: asyncio.Task | None = None
    receive_task: asyncio.Task | None = None

    try:
        async with websockets.connect(
            uri,
            subprotocols=["binary"],
            ping_interval=None,
            ping_timeout=None,
            ssl=ssl_context,
        ) as ws_connection:
            websocket = ws_connection
            ws_type_info = type(websocket).__name__
            ws_state_info = "unknown (no 'state' attr)"
            if hasattr(websocket, 'state'):
                ws_state_info = str(websocket.state)
            print_with_time(f"[{client_name}] WebSocket connection established. Type: {ws_type_info}, Initial State: {ws_state_info}")


            receive_task = asyncio.create_task(receive_messages(), name=f"{client_name}_receiver")
            
            if args.wav_path:
                send_task = asyncio.create_task(send_file_audio_in_chunks(), name=f"{client_name}_file_sender")
                await _manage_file_mode_session(client_name, send_task, receive_task)
            else:
                send_task = asyncio.create_task(record_and_send_from_microphone(), name=f"{client_name}_mic_sender")
                
                send_task_name = send_task.get_name() if hasattr(send_task, 'get_name') and callable(send_task.get_name) else "mic_sender"
                receive_task_name = receive_task.get_name() if hasattr(receive_task, 'get_name') and callable(receive_task.get_name) else "receiver"
                print_with_time(f"[{client_name}] Mic Mode: Running send ('{send_task_name}') and receive ('{receive_task_name}') tasks concurrently.")
                
                done, pending = await asyncio.wait(
                    [send_task, receive_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                done_task_names = [t.get_name() if hasattr(t, 'get_name') and callable(t.get_name) else "a_task" for t in done]
                print_with_time(f"[{client_name}] Mic Mode: One task group completed. Done tasks: {done_task_names}.")

                for task in pending:
                    task_name = task.get_name() if hasattr(task, 'get_name') and callable(task.get_name) else "pending_task"
                    if not task.done():
                        print_with_time(f"[{client_name}] Mic Mode: Cancelling pending task: {task_name}")
                        task.cancel()
                
                all_tasks = [send_task, receive_task]
                results = await asyncio.gather(*all_tasks, return_exceptions=True)
                _handle_task_results(results, all_tasks, client_name, "Mic Mode")

            print_with_time(f"[{client_name}] Primary task logic in ws_client_session completed.")

    except websockets.exceptions.InvalidURI:
        print_with_time(f"[{client_name}] Invalid WebSocket URI: {uri}")
    except ConnectionRefusedError:
        print_with_time(f"[{client_name}] Connection refused by server at {uri}.")
    except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.ConnectionClosedOK) as e:
        print_with_time(f"[{client_name}] WebSocket connection closed externally or during setup: {e}")
    except asyncio.CancelledError:
        print_with_time(f"[{client_name}] Client session task (ws_client_session) was cancelled.")
    except Exception as e:
        print_with_time(f"[{client_name}] Unexpected error in ws_client_session: {e}")
        logging.exception(f"[{client_name}] ws_client_session trace:", exc_info=e)
    finally:
        print_with_time(f"[{client_name}] Cleaning up ws_client_session...")
        tasks_to_finalize = []
        if send_task and not send_task.done():
            tasks_to_finalize.append(send_task)
        if receive_task and not receive_task.done():
            tasks_to_finalize.append(receive_task)
        
        if tasks_to_finalize:
            task_names_to_finalize = [t.get_name() if hasattr(t, 'get_name') and callable(t.get_name) else "a_task" for t in tasks_to_finalize]
            print_with_time(f"[{client_name}] Cancelling outstanding tasks: {task_names_to_finalize}")
            for task in tasks_to_finalize:
                task.cancel()
            await asyncio.gather(*tasks_to_finalize, return_exceptions=True)
            print_with_time(f"[{client_name}] Outstanding tasks processing complete after cancellation attempts.")
            
        if websocket and hasattr(websocket, 'state') and websocket.state == State.OPEN: 
            print_with_time(f"[{client_name}] WebSocket still open, attempting graceful close.")
            try:
                await websocket.close(code=1000, reason="Client shutting down")
            except Exception as e_close:
                print_with_time(f"[{client_name}] Error during explicit WebSocket close: {e_close}")
        
        websocket = None 
        print_with_time(f"[{client_name}] WebSocket client session finished.")


def run_client_process(client_name: str):
    """Runs the asyncio WebSocket client session."""
    try:
        asyncio.run(ws_client_session(client_name))
    except KeyboardInterrupt:
        print_with_time(f"\n[{client_name}] Process ({os.getpid()}) interrupted by user (KeyboardInterrupt in run_client_process).")
    except Exception as e:
        print_with_time(f"[{client_name}] Error in run_client_process ({os.getpid()}): {e}")
        logging.exception(f"[{client_name}] run_client_process trace:", exc_info=e)


if __name__ == '__main__':
    client_id_name = "mic_client" if args.wav_path is None else "file_client"
    print_with_time(f"Starting AzeroOmni EchoMind gASR Client ({client_id_name}, PID: {os.getpid()})...")
    
    if args.wav_path:
        print_with_time(f"Mode: Processing audio file: {args.wav_path}")
        if not os.path.exists(args.wav_path):
            print_with_time(f"Error: WAV file not found at {args.wav_path}. Exiting.")
            exit(1)
    else:
        print_with_time("Mode: Recording from microphone")
        try:
            import pyaudio 
        except ImportError:
            print_with_time("Error: PyAudio module not found. Please install it to use microphone input (e.g., pip install PyAudio).")
            exit(1)

    main_proc = Process(target=run_client_process, args=(client_id_name,))
    
    try:
        main_proc.start()
        main_proc.join() 
    except KeyboardInterrupt:
        print_with_time(f"\nMain process (PID: {os.getpid()}) interrupted by user (Ctrl+C). Terminating client process...")
        if main_proc.is_alive():
            main_proc.terminate() 
            main_proc.join(timeout=5) 
            if main_proc.is_alive():
                print_with_time("Client process did not terminate gracefully via SIGTERM, attempting SIGKILL.")
                main_proc.kill() 
                main_proc.join() 
        print_with_time("Client process terminated.")
    except Exception as e:
        print_with_time(f"An error occurred in the main execution block (PID: {os.getpid()}): {e}")
        logging.exception("Main execution block error", exc_info=e)
    finally:
        print_with_time(f"AzeroOmni EchoMind gASR Client (PID: {os.getpid()}) ended.")
