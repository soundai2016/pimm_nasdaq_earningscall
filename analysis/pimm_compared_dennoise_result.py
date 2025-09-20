import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
import os
import pandas as pd
import warnings
import librosa
import librosa.display
import matplotlib.gridspec as gridspec

# ------------------------------
# Paths & Constants
# ------------------------------
OUTPUT_DIR = 'results/pimm_compared_dennoise_result'
TABLEAU_COLORS = ['#1642df', '#FF6347', '#FBA400', '#008A89', '#892ADE', '#C25160']
AUDIO_BEFORE_PATH = 'results/before-pimm.mp3'
AUDIO_AFTER_PATH = 'results/after-pimm.wav'


# Helper function for exact figure sizing from reference
def mm2inch(mm: float) -> float: return mm / 25.4


# Adjusted figure size for the new 3-row layout
FIGSIZE_EXTENDED = (mm2inch(185), mm2inch(235))


# ------------------------------
# Plotting Style (from emovoice_segments_analysis.py)
# ------------------------------
def set_publication_style():
    """
    Sets the publication-ready plot style, exactly matching the reference script.
    """
    sns.set_theme(style='ticks')
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'STIXGeneral', 'Computer Modern Roman'],
        'mathtext.fontset': 'stix',
        'font.size': 8, 'figure.titlesize': 10, 'axes.titlesize': 9, 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'legend.fontsize': 7, 'legend.title_fontsize': 7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.linestyle': ':', 'grid.linewidth': 0.4, 'grid.color': '#AAAAAA',
        'axes.linewidth': 0.8, 'axes.titlelocation': 'left', 'axes.titlepad': 4.0, 'axes.labelpad': 2.0,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'lines.linewidth': 1.2, 'lines.markersize': 4.0,
        'legend.frameon': False, 'legend.handlelength': 1.6, 'legend.handletextpad': 0.6,
        'legend.borderaxespad': 0.2, 'legend.columnspacing': 1.2,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300
    })
    mpl.rcParams['axes.prop_cycle'] = cycler(color=TABLEAU_COLORS)
    sns.set_palette(TABLEAU_COLORS)


def _save_plot(fig, name: str, outdir: str):
    """Saves the plot in PNG, PDF, and EPS formats."""
    os.makedirs(outdir, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(outdir, f"{name}.pdf"), bbox_inches='tight')
        fig.savefig(os.path.join(outdir, f"{name}.eps"), bbox_inches='tight', format='eps')
    print(f"Saved figure to: {os.path.join(outdir, f'{name}.[png|pdf|eps]')}")


# ------------------------------
# Data & Configurations
# ------------------------------
data = {
    "Babble Noise": {"PIMM": [2.43, 3.02, 3.55, 3.91, 4.29], "RNNoise": [np.nan, 1.42, 1.8, 2.2, 2.8],
                     "MMSE": [np.nan, 1.35, 1.5, 1.7, 2.41], "Noisy": [np.nan, 1.35, 1.45, 1.6, 2.4]},
    "Car Noise": {"PIMM": [2.44, 3.01, 3.52, 3.89, 4.27], "RNNoise": [np.nan, 1.6, 2.1, 2.5, 3.4],
                  "MMSE": [np.nan, 1.5, 1.8, 2.09, 2.9], "Noisy": [np.nan, 1.3, 1.6, 1.9, 2.6]},
    "Street Noise": {"PIMM": [2.86, 3.38, 3.77, 4.06, 4.35], "RNNoise": [np.nan, 1.5, 1.8, 2.2, 3.0],
                     "MMSE": [np.nan, 1.3, 1.6, 1.75, 2.6], "Noisy": [np.nan, 1.2, 1.4, 1.65, 2.4]}
}
snr_levels = [-5, 0, 5, 10, 20]
methods_cfg = {
    "PIMM": {"color": TABLEAU_COLORS[0], "marker": "o", "ls": "-"},
    "RNNoise": {"color": TABLEAU_COLORS[1], "marker": "s", "ls": "--"},
    "MMSE": {"color": TABLEAU_COLORS[2], "marker": "^", "ls": "-."},
    "Noisy": {"color": TABLEAU_COLORS[3], "marker": "D", "ls": ":"},
}


# ------------------------------
# Subplot Generation Functions
# ------------------------------

def estimate_snr(y, frame_length=2048, hop_length=512):
    """
    Estimates the Signal-to-Noise Ratio (SNR) of an audio signal in dB.
    It assumes the quietest parts of the signal are noise and the loudest are signal.
    """
    # Calculate RMS energy for each frame
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
    power = rms**2

    # Use percentiles on the power values to find thresholds
    noise_thresh = np.percentile(power, 20)
    signal_thresh = np.percentile(power, 80)

    # Separate frames into signal and noise
    signal_frames = power[power >= signal_thresh]
    noise_frames = power[power <= noise_thresh]

    # Calculate average power, handling cases with no frames
    if len(signal_frames) == 0 or len(noise_frames) == 0:
        return np.nan

    avg_signal_power = np.mean(signal_frames)
    avg_noise_power = np.mean(noise_frames)

    # Calculate SNR, handling division by zero
    if avg_noise_power == 0:
        return np.inf

    snr = avg_signal_power / avg_noise_power
    snr_db = 10 * np.log10(snr)

    return snr_db


def plot_audio_analysis(fig, gs_spec):
    """
    Creates the audio analysis subplot (e) comparing before and after processing.
    Now includes estimated SNR calculation and display.
    """
    gs_audio = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs_spec, wspace=0.45, hspace=0.8)

    # Add overall panel label (e)
    ax_panel_label = fig.add_subplot(gs_spec)
    ax_panel_label.text(-0.08, 1.0, '(e)', transform=ax_panel_label.transAxes, fontsize=10, fontweight='bold', va='top')
    ax_panel_label.axis('off')

    try:
        y_before, sr_before = librosa.load(AUDIO_BEFORE_PATH, sr=None)
        y_after, sr_after = librosa.load(AUDIO_AFTER_PATH, sr=None)

        # --- MODIFIED: Calculate SNR ---
        snr_before = estimate_snr(y_before)
        snr_after = estimate_snr(y_after)

        audio_data = {
            'Before': {'y': y_before, 'sr': sr_before, 'row': 0, 'snr': snr_before},
            'After (PIMM)': {'y': y_after, 'sr': sr_after, 'row': 1, 'snr': snr_after}
        }

        for title, ad in audio_data.items():
            y, sr, row_idx, snr = ad['y'], ad['sr'], ad['row'], ad['snr']
            # --- MODIFIED: Prepare SNR text for title ---
            snr_text = f' (Est. SNR: {snr:.2f} dB @ {sr} Hz)' if not np.isnan(snr) else f' (SNR N/A @ {sr} Hz)'

            # Waveform
            ax = fig.add_subplot(gs_audio[row_idx, 0])
            librosa.display.waveshow(y, sr=sr, ax=ax)
            ax.set_title(f'{title} Waveform\n{snr_text}')
            ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude')

            # Spectrogram
            ax = fig.add_subplot(gs_audio[row_idx, 1])
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax, cmap='inferno')
            ax.set_ylim(top=8000)
            ax.set_title(f'{title} Spectrogram\n{snr_text}')
            ax.set_xlabel('Time (s)'); ax.set_ylabel('Frequency (Hz)')
            fig.colorbar(img, ax=ax, format='%+2.0f dB', pad=0.01)

            # Mel Spectrogram
            ax = fig.add_subplot(gs_audio[row_idx, 2])
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            S_db = librosa.power_to_db(S, ref=np.max)
            img_mel = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax, fmax=8000, cmap='inferno')
            ax.set_title(f'{title} Mel Spectrogram\n{snr_text}')
            ax.set_xlabel('Time (s)'); ax.set_ylabel('Mel Freq.')
            fig.colorbar(img_mel, ax=ax, format='%+2.0f dB', pad=0.01)

    except FileNotFoundError:
        ax_error = fig.add_subplot(gs_spec)
        error_msg = f"Audio files not found. Skipping plot (e).\nPlace '{os.path.basename(AUDIO_BEFORE_PATH)}' and\n'{os.path.basename(AUDIO_AFTER_PATH)}' in '{os.path.dirname(AUDIO_BEFORE_PATH)}/'"
        ax_error.text(0.5, 0.5, error_msg, ha='center', va='center', fontsize=9, style='italic', color='red')
        ax_error.set_xticks([]); ax_error.set_yticks([])


# ------------------------------
# Comprehensive Plot Generation
# ------------------------------
def plot_comprehensive_summary():
    """
    Generates a single, optimized figure summarizing all analyses, including audio comparison.
    """
    set_publication_style()

    # 1. Prepare data using pandas
    records = [
        {"noise": noise, "method": method, "snr": snr_levels[i], "score": score}
        for noise, methods in data.items()
        for method, scores in methods.items()
        for i, score in enumerate(scores)
    ]
    df = pd.DataFrame(records).dropna()

    # 2. Create the figure and GridSpec layout
    fig = plt.figure(figsize=FIGSIZE_EXTENDED)
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1.2])

    # --- Panel a: Average Performance Trend ---
    ax_a = fig.add_subplot(gs[0, 0])
    avg_trend = df.groupby(['method', 'snr'])['score'].mean().unstack(level='method')
    for method, cfg in methods_cfg.items():
        if method in avg_trend.columns:
            avg_trend[method].plot(ax=ax_a, **cfg)
    ax_a.set_title('Average Performance vs. SNR', fontweight='bold')
    ax_a.text(-0.18, 1.05, '(a)', transform=ax_a.transAxes, fontsize=10, fontweight='bold')
    ax_a.set_xlabel('SNR (dB)'); ax_a.set_ylabel('Average MOS-LQO Score')
    ax_a.set_xticks(snr_levels)
    ax_a.legend(loc='lower right', title='Method')

    # --- Panel b: Overall Performance Summary ---
    ax_b = fig.add_subplot(gs[0, 1])
    summary = df.groupby('method')['score'].agg(['mean', 'std']).reindex(list(methods_cfg.keys()))
    colors_b = [methods_cfg[m]['color'] for m in summary.index]
    bars_b = ax_b.bar(summary.index, summary['mean'], yerr=summary['std'], color=colors_b, capsize=2.5, alpha=0.9,
                      error_kw={'elinewidth': 0.8})
    ax_b.set_title('Overall Performance', fontweight='bold')
    ax_b.text(-0.18, 1.05, '(b)', transform=ax_b.transAxes, fontsize=10, fontweight='bold')
    ax_b.set_ylabel('Average MOS-LQO Score')
    ax_b.bar_label(bars_b, fmt='%.2f', fontsize=6, padding=2)
    ax_b.set_ylim(top=ax_b.get_ylim()[1] * 1.1)

    # --- Prepare data for gain calculation ---
    df_pivot = df.pivot_table(index=['noise', 'snr'], columns='method', values='score')
    for method in ['PIMM', 'RNNoise', 'MMSE']:
        if method in df_pivot.columns and 'Noisy' in df_pivot.columns:
            df_pivot[f'gain_{method}'] = df_pivot[method] - df_pivot['Noisy']
    df_gain = df_pivot.filter(like='gain_').reset_index()
    df_gain_melt = df_gain.melt(id_vars=['noise', 'snr'], var_name='method', value_name='gain')
    df_gain_melt['method'] = df_gain_melt['method'].str.replace('gain_', '')

    # --- Panel c: Average Denoising Gain ---
    ax_c = fig.add_subplot(gs[1, 0])
    gain_summary = df_gain_melt.groupby('method')['gain'].agg(['mean', 'std']).reindex(['PIMM', 'RNNoise', 'MMSE'])
    colors_c = [methods_cfg[m]['color'] for m in gain_summary.index]
    bars_c = ax_c.bar(gain_summary.index, gain_summary['mean'], yerr=gain_summary['std'], color=colors_c, capsize=2.5,
                      alpha=0.9, error_kw={'elinewidth': 0.8})
    ax_c.set_title('MOS Score Gain vs. Noisy Baseline', fontweight='bold')
    ax_c.text(-0.18, 1.05, '(c)', transform=ax_c.transAxes, fontsize=10, fontweight='bold')
    ax_c.set_ylabel('MOS Score Improvement')
    ax_c.bar_label(bars_c, fmt='%.2f', fontsize=6, padding=2)
    ax_c.set_ylim(top=ax_c.get_ylim()[1] * 1.1)

    # --- Panel d: Denoising Gain vs. SNR ---
    ax_d = fig.add_subplot(gs[1, 1])
    gain_vs_snr = df_gain_melt.groupby(['method', 'snr'])['gain'].mean().unstack(level='method')
    denoising_methods = ['PIMM', 'RNNoise', 'MMSE']
    for method in denoising_methods:
        if method in gain_vs_snr.columns:
            gain_vs_snr[method].plot(ax=ax_d, **methods_cfg[method])
    ax_d.set_title('Denoising Gain vs. SNR', fontweight='bold')
    ax_d.text(-0.18, 1.05, '(d)', transform=ax_d.transAxes, fontsize=10, fontweight='bold')
    ax_d.set_xlabel('SNR (dB)'); ax_d.set_ylabel('MOS Score Improvement')
    ax_d.set_xticks(snr_levels)
    ax_d.legend(denoising_methods, title='Method')

    # --- Panel e: Audio Analysis ---
    plot_audio_analysis(fig, gs[2, :])

    # --- Layout Adjustment ---
    fig.tight_layout(pad=1.0, h_pad=2.5, w_pad=2.5)
    
    _save_plot(fig, "fig_pimm_perf_summary", OUTPUT_DIR)
    plt.close(fig)


# ------------------------------
# Main Program
# ------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    print("Generating the final, extended 5-panel summary figure...")
    plot_comprehensive_summary()
    print("\nProcess finished.")