"""
Author: Chen Xiaoliang <chenxiaoliang@soundai.com>
Date: 2025-07-30
Copyright (c) 2025 SoundAI Inc. All rights reserved.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from wordcloud import WordCloud, STOPWORDS
import re
from collections import Counter
import os
import seaborn as sns

# --- NLTK Stopwords Setup ---
try:
    from nltk.corpus import stopwords
    STOPWORDS_SET = set(stopwords.words('english')).union(STOPWORDS)
    print("Successfully loaded NLTK stopwords for enhanced filtering.")
except (ImportError, LookupError):
    print("NLTK stopwords not found. Falling back to the default list from 'wordcloud'.")
    print("For better filtering, run: import nltk; nltk.download('stopwords')")
    STOPWORDS_SET = STOPWORDS

def set_publication_style():
    """Sets global matplotlib parameters for Nature journal-compliant plots."""
    print("\nSetting plot style to Nature journal standards (Arial font, 5-7 pt)...")
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times'],
        'font.size': 8,             # base font size for most elements
        'axes.labelsize': 8,        # axis labels slightly smaller than title
        'axes.titlesize': 9,       # figure title
        'xtick.labelsize': 7,       # tick labels smaller for visual hierarchy
        'ytick.labelsize': 7,
        'legend.fontsize': 7,       # small but readable legend
        'figure.titlesize': 10,     # overall figure title
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.linestyle': ':',
        'grid.color': 'grey',
        'pdf.fonttype': 42,         # ensure Type 1 fonts
        'ps.fonttype': 42,
        'text.usetex': False        # avoid LaTeX dependency unless necessary
    })

def _save_plot(figure, filename: str, output_dir: str):
    """
    Helper function to save a matplotlib figure in PNG (300 dpi) and EPS formats.
    """
    base_name, _ = os.path.splitext(filename)
    png_path = os.path.join(output_dir, f"{base_name}.png")
    eps_path = os.path.join(output_dir, f"{base_name}.eps")
    
    figure.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
    figure.savefig(eps_path, bbox_inches='tight', format='eps')
    
    print(f"Plot saved to: {png_path} and {eps_path}")

def generate_wordcloud(text_series: pd.Series, output_dir: str, custom_stopwords: set = None):
    """Generates a word cloud with a publication-ready color palette."""
    print("\n--- Generating Word Cloud ---")
    full_text = ' '.join(text_series.dropna().astype(str))
    if not full_text.strip():
        print("Text content is empty. Cannot generate word cloud.")
        return

    all_stopwords = STOPWORDS_SET.union(custom_stopwords or set())
    
    wordcloud = WordCloud(
        width=1200, height=600, background_color='white',
        stopwords=all_stopwords, min_font_size=10,
        colormap='cividis',
        collocations=False
    ).generate(full_text)

    fig, ax = plt.subplots(figsize=(3.5, 1.75))  # Single-column width: 89 mm ≈ 3.5 inches
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_axis_off()
    # fig.suptitle('Dominant Term Representation in Earnings Call')
    
    _save_plot(fig, "dominant_term_representation_in_earnings_call.png", output_dir)
    plt.close(fig)

def plot_zipf(text_series: pd.Series, output_dir: str, custom_stopwords: set = None):
    """Generates a Zipf's Law plot following Nature standards."""
    print("\n--- Generating Zipf's Law Plot ---")
    full_text = ' '.join(text_series.dropna().astype(str)).lower()
    words = re.findall(r'\b\w+\b', full_text)
    
    all_stopwords = STOPWORDS_SET.union(custom_stopwords or set())
    filtered_words = [word for word in words if word not in all_stopwords and not word.isdigit()]

    if not filtered_words:
        print("No words left after filtering. Cannot generate Zipf's plot.")
        return

    segment_counts = Counter(filtered_words)
    sorted_counts = sorted(segment_counts.values(), reverse=True)
    ranks = range(1, len(sorted_counts) + 1)
    
    fig, ax = plt.subplots(figsize=(3.5, 2.1))  # Single-column width
    ax.loglog(ranks, sorted_counts, marker=".", linestyle="none", color='#0072B2')
    # ax.set_title("Acoustic Segment Frequency Distribution in Earnings Call")
    ax.set_xlabel("Acoustic Segment Rank (Log Scale)")
    ax.set_ylabel("Frequency (Log Scale)")
    
    _save_plot(fig, "acoustic_segment_frequency_distribution_in_earnings_call.png", output_dir)
    plt.close(fig)

def plot_tfidf_bar(text_series: pd.Series, output_dir: str, custom_stopwords: set = None):
    """Generates a bar plot of top 20 words by mean TF-IDF score."""
    print("\n--- Generating TF-IDF Bar Plot (Top 20 Words) ---")
    documents = text_series.dropna().astype(str).tolist()
    if not documents:
        print("No text data available for TF-IDF analysis.")
        return

    all_stopwords = STOPWORDS_SET.union(custom_stopwords or set())
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words=list(all_stopwords), max_features=20)
    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
    except ValueError:
        print("Could not generate TF-IDF matrix. Vocabulary may be empty.")
        return

    mean_scores = tfidf_matrix.mean(axis=0).A1  # Use .A1 to convert to 1D array
    df_tfidf = pd.DataFrame({'word': feature_names, 'mean_tfidf': mean_scores})
    df_tfidf = df_tfidf.sort_values('mean_tfidf', ascending=True)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))  # Single-column width
    ax.barh(df_tfidf['word'], df_tfidf['mean_tfidf'], color='#009E73')
    ax.set_xlabel('Mean TF-IDF Score')
    ax.set_ylabel('Term')
    # ax.set_title('Top 20 Terms by TF-IDF Relevance')
    
    _save_plot(fig, "top_20_terms_by_TF-IDF_relevance.png", output_dir)
    plt.close(fig)

def plot_segment_count_stats(segment_count_series: pd.Series, output_dir: str):
    """Generates a combined histogram and boxplot for word count distribution."""
    print("\n--- Generating Word Count Descriptive Statistics Plot ---")
    segment_counts = segment_count_series.dropna().astype(int)
    if segment_counts.empty:
        print("No word count data available for analysis.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.5), gridspec_kw={'height_ratios': [3, 1]})  # Single-column width
    sns.histplot(segment_counts, bins=30, color='#009E73', ax=ax1)
    ax1.set_xlabel('')
    ax1.set_ylabel('Frequency')
    # ax1.set_title('Distribution of Acoustic Segment Length in Earnings Call')

    sns.boxplot(x=segment_counts, color='#009E73', ax=ax2)
    ax2.set_xlabel('Acoustic Segment Length')
    ax2.set_yticks([])

    plt.tight_layout()
    _save_plot(fig, "distribution_of_acoustic_segment_length_in_earnings_call.png", output_dir)
    plt.close(fig)

def main():
    """Main function to read data and call analysis functions."""
    data_dir = "results"
    output_dir = os.path.join(data_dir, "descriptive_segment_results")
    file_name = "emovoice_transcript_emotion_per_segment.csv"
    csv_file_path = os.path.join(data_dir, file_name)
    os.makedirs(output_dir, exist_ok=True)
    
    set_publication_style()

    print(f"\nAttempting to read file from: {csv_file_path}")
    try:
        df = pd.read_csv(csv_file_path, low_memory=False)
        print(f"Successfully read the file. Found {len(df)} text segments.")
    except FileNotFoundError:
        print(f"\nError: File not found at '{csv_file_path}'")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    text_data = df['transcripts']
    segment_count_data = df['acoustic_segment_length']
    custom_stopwords = {'abbvie', 'company', 'rob', 'jeff', 'scott', 'liz', 'thanks', 'question'}

    generate_wordcloud(text_data, output_dir=output_dir, custom_stopwords=custom_stopwords)
    plot_zipf(text_data, output_dir=output_dir, custom_stopwords=custom_stopwords)
    plot_tfidf_bar(text_data, output_dir=output_dir, custom_stopwords=custom_stopwords)
    plot_segment_count_stats(segment_count_data, output_dir=output_dir)

if __name__ == "__main__":
    main()