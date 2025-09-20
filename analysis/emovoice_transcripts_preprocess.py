"""
Author: Chen Xiaoliang <chenxiaoliang@soundai.com>
Date: 2025-07-30
Copyright (c) 2025 SoundAI Inc. All rights reserved.
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

# --- Configuration ---
SOURCE_DATA_DIR = '../../datasets/emovoice_nasdaq_earningscall_stockprice_transcripts'
OUTPUT_DIR = 'results'
OUTPUT_CSV_FILE = os.path.join(OUTPUT_DIR, 'emovoice_transcript_emotion_per_segment.csv')

# --- Mappings and Constants ---
# Using text labels for emotions for clarity and compatibility
EMOTION_TEXT_ORDER = ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']

# Comprehensive map to standardize all inputs to a clean text format
INPUT_TO_TEXT_MAP = {
    # Emoji to Text
    '😊': 'happiness', '😔': 'sadness', '😡': 'anger', '😰': 'fear',
    '😮': 'surprise', '🤢': 'disgust', '😒': 'disgust', '😐': 'neutral',
    # Text to Text (for consistency)
    'happiness': 'happiness', 'sadness': 'sadness', 'anger': 'anger', 'fear': 'fear',
    'surprise': 'surprise', 'disgust': 'disgust', 'none': 'neutral',
    # Handle empty/null cases
    '': 'neutral'
}

def standardize_emotion(value):
    """Safely standardizes various emotion inputs into a consistent text format using a map."""
    return INPUT_TO_TEXT_MAP.get(str(value).lower(), 'neutral') # Default to neutral for any unknown values

def classify_speaker(speaker_str):
    """Classifies a detailed speaker string into a broader category."""
    if not isinstance(speaker_str, str): return 'Other'
    s_lower = speaker_str.lower()
    if 'ceo' in s_lower or 'chief executive officer' in s_lower: return 'CEO'
    if 'cfo' in s_lower or 'chief financial officer' in s_lower: return 'CFO'
    if any(role in s_lower for role in ['officer', 'coo', 'cso', 'cmo', 'cto', 'president', 'executive']): return 'CXO'
    if 'operator' in s_lower or 'moderator' in s_lower or 'investor relations' in s_lower: return 'HOST'
    if 'analyst' in s_lower: return 'Analyst'
    return 'Other'

def parse_stock_data(price_file_path):
    """Parses the specific nested JSON format of the stock price file."""
    with open(price_file_path, 'r', encoding='utf-8') as f: data = json.load(f)
    time_series_data = data.get("Time Series (Daily)")
    if not time_series_data: return None
    stock_df = pd.DataFrame.from_dict(time_series_data, orient='index')
    stock_df.index = pd.to_datetime(stock_df.index)
    stock_df.columns = [col.split('. ')[1] for col in stock_df.columns]
    stock_df = stock_df.apply(pd.to_numeric)
    return stock_df.sort_index()

def get_price_on_or_before(date, df):
    """Finds the last available stock price on or before a given date."""
    subset = df.loc[:date]
    return subset.iloc[-1] if not subset.empty else None

def get_price_on_or_after(date, df):
    """Finds the first available stock price on or after a given date."""
    subset = df.loc[date:]
    return subset.iloc[0] if not subset.empty else None

def analyze_data():
    """Performs an expanded descriptive statistical analysis on the organized data."""
    print("--- Starting Expanded Descriptive Analysis ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Results will be saved in: {OUTPUT_DIR}")

    all_segments_data = []
    tickers = [d for d in os.listdir(SOURCE_DATA_DIR) if os.path.isdir(os.path.join(SOURCE_DATA_DIR, d))]
    print(f"Found {len(tickers)} tickers to process.")

    # Pre-scan to count the number of conference calls with transcripts
    conference_count = 0
    for ticker in tickers:
        ticker_path = os.path.join(SOURCE_DATA_DIR, ticker)
        index_file_path = os.path.join(ticker_path, f"{ticker}_Index.json")
        if not os.path.exists(index_file_path): continue
        with open(index_file_path, 'r', encoding='utf-8') as f:
            try:
                index_records = json.load(f)
                for index_record in index_records:
                    transcript_filename = index_record.get('transcript_sound')
                    if transcript_filename and os.path.exists(os.path.join(ticker_path, transcript_filename)):
                        conference_count += 1
            except json.JSONDecodeError:
                continue
    print(f"Found {conference_count} conference calls with transcripts to analyze.")

    for ticker in tqdm(tickers, desc="Processing Tickers"):
        ticker_path = os.path.join(SOURCE_DATA_DIR, ticker)
        index_file_path = os.path.join(ticker_path, f"{ticker}_Index.json")
        price_file_path = os.path.join(ticker_path, f"{ticker}_Nasdaq_Stockprice.json")
        if not os.path.exists(index_file_path): continue

        stock_df = None
        if os.path.exists(price_file_path):
            try: stock_df = parse_stock_data(price_file_path)
            except Exception as e: tqdm.write(f"  [Error] Could not process price file for {ticker}: {e}")
        
        with open(index_file_path, 'r', encoding='utf-8') as f: index_records = json.load(f)

        for index_record in index_records:
            transcript_filename = index_record.get('transcript_sound')
            if not transcript_filename: continue
            transcript_path = os.path.join(ticker_path, transcript_filename)
            if not os.path.exists(transcript_path): continue
            
            stock_metrics = {}
            event_time_str = index_record.get('event_start_et')
            if event_time_str and stock_df is not None:
                try:
                    event_date = datetime.strptime(event_time_str, '%Y-%m-%d %H:%M:%S').date()
                    day_t0_series = get_price_on_or_before(event_date, stock_df)
                    if day_t0_series is not None:
                        stock_metrics['open_t0'] = day_t0_series['open']
                        stock_metrics['close_t0'] = day_t0_series['close']
                        stock_metrics['volume_t0'] = day_t0_series['volume']
                        
                        # --- START: Added logic for past prices and returns ---
                        past_horizons = [1, 7, 30, 90]
                        for h in past_horizons:
                            past_date = event_date - timedelta(days=h)
                            day_th_series_past = get_price_on_or_before(past_date, stock_df)
                            if day_th_series_past is not None:
                                stock_metrics[f'open_t-{h}d'] = day_th_series_past['open']
                                stock_metrics[f'close_t-{h}d'] = day_th_series_past['close']
                                
                                # Calculate return from t-h to t0: (close_t0 - close_t-h) / close_t-h
                                close_past = day_th_series_past['close']
                                if close_past != 0:
                                    stock_metrics[f'return_t-{h}d'] = (day_t0_series['close'] - close_past) / close_past
                        # --- END: Added logic for past prices and returns ---

                        # --- Logic for future prices and returns (existing) ---
                        future_horizons = [1, 7, 30, 90]
                        for h in future_horizons:
                            future_date = event_date + timedelta(days=h)
                            day_th_series = get_price_on_or_after(future_date, stock_df)
                            if day_th_series is not None:
                                stock_metrics[f'open_t+{h}d'] = day_th_series['open']
                                stock_metrics[f'close_t+{h}d'] = day_th_series['close']
                                stock_metrics[f'return_t+{h}d'] = (day_th_series['close'] - day_t0_series['close']) / day_t0_series['close']
                        
                        # --- Logic for historical volatility (existing) ---
                        vol_end_date = event_date - timedelta(days=1)
                        vol_start_date = vol_end_date - timedelta(days=45)
                        vol_df = stock_df.loc[vol_start_date:vol_end_date]
                        if len(vol_df) > 1:
                            daily_returns = vol_df['close'].pct_change().dropna()
                            stock_metrics['historical_volatility_30d'] = daily_returns.std() * np.sqrt(252)
                except Exception as e:
                    tqdm.write(f"    [Warning] Ticker {ticker}: Stock data processing failed: {e}")

            with open(transcript_path, 'r', encoding='utf-8') as tf:
                for line in tf:
                    try:
                        segment = json.loads(line)
                        emotion = standardize_emotion(segment.get('emotion'))
                        text_emotion = standardize_emotion(segment.get('text_emotion'))
                        speaker = segment.get('speaker')
                        segment_text = segment.get('text', '')
                        
                        processed_segment = {
                            'ticker': ticker,
                            'uid': index_record.get('uid'),
                            "event_time": event_time_str,
                            'speaker': speaker,
                            'speaker_role': classify_speaker(speaker),
                            'transcripts': segment_text,
                            'language': index_record.get('language'),
                            'acoustic_segment_length': len(segment_text.split()),
                            'acoustic_emotion': emotion,
                            'textual_emotion': text_emotion,
                            'combinded_emotion': f"{emotion}|{text_emotion}",
                            'acoustic_events': segment.get('events')
                        }
                        processed_segment.update(stock_metrics)
                        all_segments_data.append(processed_segment)
                    except json.JSONDecodeError: continue

    if not all_segments_data:
        print("\nNo data processed. Exiting.")
        return
        
    df = pd.DataFrame(all_segments_data)
        
    # --- START: Reordered columns for better readability including new past data ---
    final_cols = [
        'ticker', 'uid', 'event_time', 'speaker', 'speaker_role',
        'transcripts', 'language', 'acoustic_segment_length',
        'acoustic_emotion', 'textual_emotion', 'combinded_emotion',
        'acoustic_events',
        # Past data columns
        'open_t-1d', 'close_t-1d', 'return_t-1d',
        'open_t-7d', 'close_t-7d', 'return_t-7d',
        'open_t-30d', 'close_t-30d', 'return_t-30d',
        'open_t-90d', 'close_t-90d', 'return_t-90d',
        # Event day (t0) data columns
        'open_t0', 'close_t0', 'volume_t0', 'historical_volatility_30d',
        # Future data columns
        'open_t+1d', 'close_t+1d', 'return_t+1d',
        'open_t+7d', 'close_t+7d', 'return_t+7d',
        'open_t+30d', 'close_t+30d', 'return_t+30d',
        'open_t+90d', 'close_t+90d', 'return_t+90d'
    ]
    # --- END: Reordered columns ---
    
    # Ensure all columns exist in the DataFrame before reordering
    df_cols = [col for col in final_cols if col in df.columns]
    df = df[df_cols]

    df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')
    print(f"\nFull descriptive data saved to: {OUTPUT_CSV_FILE}")
    print(f"Total segments processed: {len(df)}")

    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    analyze_data()