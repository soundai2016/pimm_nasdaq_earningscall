"""
Author: Chen Xiaoliang <chenxiaoliang@soundai.com>
Date: 2025-07-30
Copyright (c) 2025 SoundAI Inc. All rights reserved.
"""

import os, re, pickle, random, warnings, shutil
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

# Optional dependencies
try:
    import statsmodels.api as sm
except Exception:
    sm = None
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import yfinance as yf
except Exception:
    yf = None

from sklearn.metrics import mean_squared_error
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
from cycler import cycler

# ------------------------------ Paths & Globals ------------------------------
DATA_DIR   = 'results'
INPUT_CSV  = os.path.join(DATA_DIR, 'emovoice_transcript_emotion_per_segment.csv')

OUTPUT_DIR = os.path.join(DATA_DIR, 'emovoice_segments_analysis')
CACHE_DIR  = os.path.join(OUTPUT_DIR, 'cache')
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')
TABLE_DIR  = os.path.join(OUTPUT_DIR, 'tables')

ROLES_TO_ANALYZE = ['CEO', 'CFO', 'CXO']

TARGET_HORIZONS_VOL = [1, 5, 10, 15, 20, 25, 30]
TARGET_HORIZONS_CAR = [1, 5, 10, 15, 20, 25, 30]
PRIMARY_HORIZON     = 10  # matches paper's focal window

BASELINE_CONTROL    = 'historical_volatility_10d'  # sanity baseline
MARKET_BENCHMARK    = 'SPY'
EXTRA_BENCHMARKS    = ['QQQ', '^IXIC']

ESTIMATION_WINDOW   = 252
MIN_ESTIMATION_DAYS = 120
N_BOOTSTRAPS_IMPORTANCE = 100
N_BOOTSTRAPS_CI     = 500
N_BOOTSTRAPS_ABSORB = 1000
N_PERMUTATIONS_ABSORB = 2000
EPS                 = 1e-8
GLOBAL_SEED         = 42

N_Q_TIME = 4
N_Q_VOL  = 4
N_TOP_INDUSTRIES = 10
MIN_SUBSET_N     = 10

# Emotion sets for Masking Index
NEG_ACOUSTIC = {'anger','fear','sadness','disgust'}
POS_TEXT     = {'happiness','neutral'}

class CFG:
    """Analysis configuration toggles."""
    QA_ONLY: bool = False
    ORTHOGONALIZE_ACOUSTIC: bool = True
    ADD_INDUSTRY_INTERACTIONS: bool = False
    RUN_DM_AND_MCS: bool = True
    RUN_CONTRASTS: bool = True
    F4_PANELS = ['industry','time_quartiles','vol_quartiles']
    MAX_LABEL_CHARS = 28

# ------------------------------ Emotion mapping (V–A–D) ------------------------------
def _get_default_vad_map():
    # Valence, Arousal, Dominance anchors
    return {
        'happiness': (+0.80, +0.30, +0.25),
        'anger'    : (-0.70, +0.60, +0.20),
        'fear'     : (-0.75, +0.70, -0.50),
        'sadness'  : (-0.65, -0.30, -0.40),
        'disgust'  : (-0.60, +0.20, -0.20),
        'surprise' : ( 0.00, +0.65,  0.00),
        'neutral'  : ( 0.00,  0.00,  0.00),
    }

VAD_MAP = _get_default_vad_map()

# ------------------------------ Plot Style ------------------------------
# Okabe–Ito palette (colorblind-safe)
OKABE_ITO = ['#000000','#E69F00','#56B4E9','#009E73','#F0E442','#1642df','#D55E00','#CC79A7']
PRIMARY_COLOR = '#1642df'  # blue
ACCENT_COLORS = ['#D55E00', '#009E73', '#CC79A7']  # vermillion / bluish green / purple
TABLEAU_COLORS = ['#1642df', '#E69F00', '#56B4E9', '#009E73', '#CC79A7', '#000000', '#F0E442', '#D55E00']
HORIZON_COLORS = {1:'#009E73', 5:'#E69F00', 10:'#1642df', 15:'#CC79A7', 20:'#56B4E9', 25:'#000000', 30:'#D55E00'}

def mm2inch(mm: float) -> float: return mm / 25.4
FIGSIZE_SINGLE = (mm2inch(89),  mm2inch(70))
FIGSIZE_DOUBLE = (mm2inch(180), mm2inch(85))
FIGSIZE_TRIPLE = (mm2inch(180), mm2inch(95))
FIGSIZE_QUAD   = (mm2inch(180), mm2inch(150))

def set_publication_style():
    """Small-font, tight-layout style for paper-quality figures."""
    sns.set_theme(style='ticks')
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'mathtext.fontset': 'stixsans',
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
    """Save using LaTeX-aligned aliases only (keep exactly five figure names)."""
    os.makedirs(outdir, exist_ok=True)
    alias_map = {
        "fig1_descriptive_analysis": "fig_descriptive_analysis",
        "fig2_model_performance":    "fig_model_performance",
        "fig3_feature_importance":   "fig_feature_importance",
        "fig4_robustness_combined":  "fig_robustness_combined",
        "fig5_benchmarks_and_sensitivity": "fig_benchmarks_and_sensitivity"
    }
    save_name = alias_map.get(name, name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ext in ('png','pdf','eps'):
            fig.savefig(os.path.join(outdir, f"{save_name}.{ext}"), dpi=300, bbox_inches='tight', format=ext)
    print(f"Saved: {save_name}.[png|pdf|eps]")

def _wrap_label(text: str, max_len: int = 28) -> str:
    """Wrap long categorical labels for forest/violin plots."""
    s = str(text)
    if ' - ' in s and len(s) > max_len: s = s.replace(' - ', '\n', 1)
    if len(s) > max_len: s = s[:max_len-1] + '…'
    return s

# ------------------------------ Safe error bars ------------------------------
def _safe_err(center, lo, hi):
    """Return non-negative error lengths aligned to center."""
    c = np.asarray(center, dtype=float)
    l = np.asarray(lo, dtype=float)
    h = np.asarray(hi, dtype=float)
    l = np.where(np.isfinite(l), l, c)
    h = np.where(np.isfinite(h), h, c)
    l = np.minimum(l, c)
    h = np.maximum(h, c)
    left  = np.maximum(0.0, c - l)
    right = np.maximum(0.0, h - c)
    return [left, right]

# ------------------------------ Metrics ------------------------------
def _r2(y, yhat) -> float:
    y = np.asarray(y); yhat = np.asarray(yhat)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + EPS
    return 1.0 - ss_res / ss_tot

def qlike_vector(y_true_rv: np.ndarray, y_pred_rv: np.ndarray) -> np.ndarray:
    y_true = np.maximum(np.asarray(y_true_rv, dtype=float), EPS)
    y_pred = np.maximum(np.asarray(y_pred_rv, dtype=float), EPS)
    ratio = y_true / y_pred
    return ratio - np.log(ratio) - 1.0

def qlike_loss(y_true_rv: np.ndarray, y_pred_rv: np.ndarray) -> float:
    return float(np.mean(qlike_vector(y_true_rv, y_pred_rv)))

def _ci_percentiles(alpha=0.05):
    """Two-sided percentiles for (1-alpha) CIs."""
    pl = (alpha/2.0)*100.0
    pu = (1.0 - alpha/2.0)*100.0
    return pl, pu

def _cluster_bootstrap_ci_r2(y_true, y_pred, groups, B=N_BOOTSTRAPS_CI, alpha=0.05, seed=GLOBAL_SEED):
    """Grouped bootstrap CI for R^2 (cluster = ticker)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'y': y_true, 'yhat': y_pred, 'g': groups})
    uniq = df['g'].dropna().unique().tolist()
    if len(uniq) < 5: return (np.nan, np.nan)
    stats = []
    for _ in range(B):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        sub = df[df['g'].isin(gs)]
        if sub['y'].nunique() < 2: continue
        stats.append(_r2(sub['y'].values, sub['yhat'].values))
    if not stats: return (np.nan, np.nan)
    pl, pu = _ci_percentiles(alpha)
    return float(np.percentile(stats, pl)), float(np.percentile(stats, pu))

def _cluster_bootstrap_ci_corr(y_true, x_feat, groups, B=N_BOOTSTRAPS_CI, alpha=0.05, seed=GLOBAL_SEED):
    """Grouped bootstrap CI for correlations."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'y': y_true, 'x': x_feat, 'g': groups}).dropna()
    uniq = df['g'].dropna().unique().tolist()
    if len(uniq) < 5: return (float('nan'), float('nan'))
    stats = []
    for _ in range(B):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        sub = df[df['g'].isin(gs)]
        if sub['y'].nunique() < 2 or sub['x'].nunique() < 2: continue
        c = np.corrcoef(sub['y'].values, sub['x'].values)[0,1]
        if np.isfinite(c): stats.append(float(c))
    if not stats: return (float('nan'), float('nan'))
    pl, pu = _ci_percentiles(alpha)
    return float(np.percentile(stats, pl)), float(np.percentile(stats, pu))

def _cluster_bootstrap_ci_delta_r2(y_true, yhat_base, yhat_full, groups, B=N_BOOTSTRAPS_CI, alpha=0.05, seed=GLOBAL_SEED):
    """Grouped bootstrap CI for ΔR^2 = R^2(full) − R^2(base)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'y': y_true, 'y0': yhat_base, 'y1': yhat_full, 'g': groups})
    uniq = df['g'].dropna().unique().tolist()
    if len(uniq) < 5: return (np.nan, np.nan)
    stats = []
    for _ in range(B):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        sub = df[df['g'].isin(gs)]
        if sub['y'].nunique() < 2: continue
        stats.append(_r2(sub['y'], sub['y1']) - _r2(sub['y'], sub['y0']))
    if not stats: return (np.nan, np.nan)
    pl, pu = _ci_percentiles(alpha)
    return float(np.percentile(stats, pl)), float(np.percentile(stats, pu))

# ------------------------------ Fig.1: Descriptives ------------------------------
QNA_TRIGGERS    = ['q&a', 'question and answer', 'questions and answers', 'qa session']
ANALYST_HINTS   = ['analyst']
OPERATOR_HINTS  = ['operator']

def _is_question_like(text: str) -> bool:
    """Heuristic to detect question-like utterances for Q&A onset."""
    if not isinstance(text, str): return False
    t = text.strip().lower()
    return ('?' in t) or t.startswith(('what ','why ','how ','when ','where ','which ','could ','would ','can ','may '))

def create_descriptive_plots(df: pd.DataFrame, output_dir: str):
    """Figure 1: emotion distributions, cross-modal agreement, and V–A density/ellipses."""
    set_publication_style()
    print("\n--- Generating Figure 1 ---")

    import matplotlib.gridspec as gridspec
    roles = list(ROLES_TO_ANALYZE)

    fig_w, fig_h = mm2inch(180), mm2inch(112 if len(roles) == 3 else 100)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs_outer  = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[0.92, 1.18], hspace=0.70)
    gs_top    = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[0], wspace=0.60)
    gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[1], wspace=0.45)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.93, bottom=0.18)

    # (a) Role-wise emotion distributions (acoustic vs. text)
    ax_a = fig.add_subplot(gs_top[0, 0])
    emotions = ['anger','disgust','fear','happiness','neutral','sadness','surprise']
    df_roles = df[df['speaker_role'].isin(roles)].copy()
    text_counts = df_roles.groupby(['textual_emotion','speaker_role'], observed=True).size().unstack(fill_value=0)
    ac_counts   = df_roles.groupby(['acoustic_emotion','speaker_role'],  observed=True).size().unstack(fill_value=0)
    total = max(1, len(df_roles))
    text_pct = (text_counts / total * 100.0).reindex(index=emotions, columns=roles, fill_value=0.0)
    ac_pct   = (ac_counts   / total * 100.0).reindex(index=emotions, columns=roles, fill_value=0.0)
    x = np.arange(len(emotions)); width = 0.36
    b_text = np.zeros_like(x, dtype=float); b_ac = np.zeros_like(x, dtype=float)

    for j, r in enumerate(roles):
        ax_a.bar(x - width/2, ac_pct[r].values, width=width, bottom=b_ac, color=TABLEAU_COLORS[j], alpha=1.0, label=f'Acoustic - {r}')
        b_ac += ac_pct[r].values
        ax_a.bar(x + width/2, text_pct[r].values, width=width, bottom=b_text, color=TABLEAU_COLORS[j], alpha=0.5,
                 hatch='//', edgecolor='white', linewidth=0.5, label=f'Textual - {r}')
        b_text += text_pct[r].values

    ax_a.set_title('Emotion Distribution by Role', loc='left', pad=2.0)
    ax_a.text(-0.18, 1.03, '(a)', transform=ax_a.transAxes, fontweight='bold')
    ax_a.set_ylabel('Overall Percentage (%)'); ax_a.set_xlabel('Acoustic vs. Textual Emotion')
    ax_a.set_xticks(x, [e.capitalize() for e in emotions], rotation=30, ha='right')
    ax_a.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=6))
    ax_a.grid(axis='x', linestyle=':')

    # Legend: roles + modality
    leg_role = ax_a.legend(handles=[Patch(facecolor=TABLEAU_COLORS[j], label=r) for j, r in enumerate(roles)],
                           loc='upper left', bbox_to_anchor=(0.02, 1.02), borderaxespad=0.1, frameon=False)
    ax_a.add_artist(leg_role)
    ax_a.legend(handles=[Patch(facecolor='#8C8C8C', alpha=1.0, label='Acoustic'),
                         Patch(facecolor='#8C8C8C', alpha=0.95, hatch='//', label='Textual')],
                loc='upper right', bbox_to_anchor=(1.02, 1.02), frameon=False)

    # (b) Cross-modal agreement heatmap
    ax_b = fig.add_subplot(gs_top[0, 1])
    contingency = pd.crosstab(df['acoustic_emotion'], df['textual_emotion'])
    contingency = contingency.reindex(index=emotions, columns=emotions, fill_value=0)
    contingency_norm = contingency.div(contingency.sum(axis=1).replace(0, 1.0), axis=0)
    hm = sns.heatmap(contingency_norm, annot=True, fmt=".1%", cmap='viridis',
                     linewidths=.5, ax=ax_b, cbar_kws={'label': 'Row Proportion'}, annot_kws={'fontsize': 7})
    ax_b.set_title('Acoustic vs. Textual Agreement', loc='left', pad=2.0)
    ax_b.text(-0.18, 1.03, '(b)', transform=ax_b.transAxes, fontweight='bold')
    ax_b.set_xlabel('Textual Emotion'); ax_b.set_ylabel('Acoustic Emotion')
    plt.setp(ax_b.get_xticklabels(), rotation=40, ha='right', rotation_mode='anchor')
    ax_b.tick_params(axis='y', rotation=0)
    if hm.collections: hm.collections[0].colorbar.ax.tick_params(labelsize=7)

    # (c)-(e) V–A density/ellipse by role (acoustic vs text)
    try:
        from scipy.stats import gaussian_kde
        _has_kde = True
    except Exception:
        _has_kde = False

    def _plot_std_ellipse(ax, mean_xy, cov2x2, edgecolor, linestyle='-', facecolor=None, alpha=0.06):
        vals, vecs = np.linalg.eigh(cov2x2)
        if not np.isfinite(vals).all() or (vals <= 0).any(): return
        order = vals.argsort()[::-1]; vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 2.0*np.sqrt(vals[0]), 2.0*np.sqrt(vals[1])
        e = mpl.patches.Ellipse(xy=mean_xy, width=width, height=height, angle=float(theta),
                                edgecolor=edgecolor, facecolor=(facecolor if facecolor is not None else 'none'),
                                linestyle=linestyle, linewidth=1.0, alpha=(alpha if facecolor is not None else 1.0))
        ax.add_patch(e)

    df_va = df[df['speaker_role'].isin(roles)].copy()
    for i, dim in enumerate(['valence', 'arousal']):
        df_va[f'ac_{dim}'] = df_va['acoustic_emotion'].map(lambda x: VAD_MAP.get(x, (0,0,0))[i])
        df_va[f'tx_{dim}'] = df_va['textual_emotion'].map(lambda x: VAD_MAP.get(x, (0,0,0))[i])

    for j, r in enumerate(roles):
        ax = fig.add_subplot(gs_bottom[0, j])
        sub = df_va[df_va['speaker_role'] == r].copy()
        for mod, style in {'ac':('-', 'o'), 'tx':('--', 's')}.items():
            V = pd.to_numeric(sub[f'{mod}_valence'], errors='coerce').dropna()
            A = pd.to_numeric(sub[f'{mod}_arousal'], errors='coerce').dropna()
            idx = V.index.intersection(A.index); V, A = V.loc[idx].values, A.loc[idx].values
            if len(V) < 3: continue
            if _has_kde and len(V) >= 50:
                from numpy import linspace, meshgrid, vstack
                xy = vstack([V, A]); kde = gaussian_kde(xy)
                xi = yi = linspace(-1.05, 1.05, 120); X, Y = meshgrid(xi, yi)
                Z = kde(vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                levels = np.quantile(Z.ravel(), [0.20, 0.50])
                ax.contour(X, Y, Z, levels=levels, colors=TABLEAU_COLORS[j], linestyles=style[0], linewidths=1.0, alpha=0.9)
            else:
                ax.hexbin(V, A, gridsize=22, extent=(-1.05,1.05,-1.05,1.05), mincnt=1, linewidths=0.0,
                          alpha=0.30 if mod=='ac' else 0.22, cmap='Greys')
            mu = (float(np.mean(V)), float(np.mean(A)))
            ax.scatter([mu[0]],[mu[1]], s=55, marker=style[1], edgecolor='white', linewidth=0.5, color=TABLEAU_COLORS[j])
            cov = np.cov(np.vstack([V, A])); _plot_std_ellipse(ax, mu, cov, edgecolor=TABLEAU_COLORS[j], linestyle=style[0],
                                                               facecolor=TABLEAU_COLORS[j] if mod=='ac' else None, alpha=0.55)
        ax.axhline(0, color='#666', linewidth=0.6, linestyle='--'); ax.axvline(0, color='#666', linewidth=0.6, linestyle='--')
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_xticks([-1,0,1]); ax.set_yticks([-1,0,1])
        ax.tick_params(axis='both', labelsize=7, pad=1.0)
        ax.set_xlabel('Valence'); ax.set_ylabel('Arousal' if j==0 else '')
        ax.set_title(f'V–A (Role: {r})', loc='left', pad=1.2, fontsize=9)
        ax.text(-0.18, 1.03, f"({chr(ord('c') + j)})", transform=ax.transAxes, fontweight='bold')
        ax.set_aspect('equal', adjustable='box'); ax.grid(True, linestyle=':')

    legend_handles = [
        Line2D([0],[0], color='#8C8C8C', linestyle='-',  marker='o', label='Acoustic (Mean & 1-SD Ellipse)'),
        Line2D([0],[0], color='#8C8C8C', linestyle='--', marker='s', label='Textual (Mean & 1-SD Ellipse)')
    ]
    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.01),
               ncol=2, frameon=False, fontsize=9)

    _save_plot(fig, "fig1_descriptive_analysis", output_dir)
    plt.close(fig)

# ------------------------------ Feature Engineering ------------------------------
def label_call_sections(df: pd.DataFrame, min_confirm: int = 2) -> pd.DataFrame:
    """Heuristics to label within-call 'presentation' vs 'q&a' sections."""
    df = df.copy(); df['section'] = 'presentation'
    def determine_section(group: pd.DataFrame) -> pd.DataFrame:
        qna_started, recent_analyst_q, rows = False, 0, []
        for _, row in group.iterrows():
            text = str(row.get('transcripts', '')).lower()
            role = str(row.get('speaker_role', '')).lower()
            operator_ann    = (any(h in role for h in OPERATOR_HINTS) and any(k in text for k in QNA_TRIGGERS))
            analyst_question = (any(h in role for h in ANALYST_HINTS) and _is_question_like(text))
            if operator_ann: qna_started = True
            if analyst_question: recent_analyst_q += 1
            else: recent_analyst_q = max(0, recent_analyst_q - 0.25)
            if (recent_analyst_q >= min_confirm): qna_started = True
            rows.append('q&a' if qna_started else 'presentation')
        group = group.copy(); group['section'] = rows; return group
    return df.groupby('uid', group_keys=False, observed=True).apply(determine_section)

def engineer_vad_features(df: pd.DataFrame, vad_map: dict) -> pd.DataFrame:
    """Aggregate V–A–D (acoustic & text) per role × section; build Δ-features (Q&A − Pres.)."""
    df_roles = df[df['speaker_role'].isin(ROLES_TO_ANALYZE)].copy()
    for mod in ['acoustic', 'textual']:
        for i, dim in enumerate(['valence', 'arousal', 'dominance']):
            df_roles[f'{mod}_{dim}'] = df_roles[f'{mod}_emotion'].map(lambda x: vad_map.get(x, (0.0, 0.0, 0.0))[i])

    # Robust skewness/kurtosis with fallback when scipy is absent
    def skew_unbiased(x):
        s = pd.Series(x).dropna()
        if len(s) == 0: return 0.0
        try:
            from scipy.stats import skew
            return float(skew(s, bias=False, nan_policy='omit'))
        except Exception:
            return float(s.skew()) if len(s) > 2 else 0.0

    def kurt_unbiased(x):
        s = pd.Series(x).dropna()
        if len(s) == 0: return 0.0
        try:
            from scipy.stats import kurtosis
            return float(kurtosis(s, bias=False, nan_policy='omit'))
        except Exception:
            return float(s.kurt()) if len(s) > 3 else 0.0

    features_to_agg = [c for c in df_roles.columns if any(dim in c for dim in ['valence', 'dominance', 'arousal'])]
    agg_map = {feat: ['mean', 'std', skew_unbiased, kurt_unbiased] for feat in features_to_agg}

    grouped = df_roles.groupby(['uid', 'speaker_role', 'section'], observed=True).agg(agg_map)
    grouped.columns = [f"{c[0]}_{c[1]}" for c in grouped.columns.to_flat_index()]
    call_level = grouped.unstack(level=['speaker_role', 'section'])
    call_level.columns = [f"{col[1]}_{col[2]}_{col[0]}" for col in call_level.columns]

    # Build Δ-features: Q&A − Presentation per role × modality × dimension × stat
    for role in ROLES_TO_ANALYZE:
        for feat_type in ['acoustic', 'textual']:
            for dim in ['valence', 'dominance', 'arousal']:
                for stat in ['mean', 'std', 'skew', 'kurtosis']:
                    base = f'{feat_type}_{dim}_{stat}'
                    pres = f'{role}_presentation_{base}'
                    qna  = f'{role}_q&a_{base}'
                    if pres in call_level.columns and qna in call_level.columns:
                        call_level[f'{role}_delta_{feat_type}_{dim}_{stat}'] = call_level[qna] - call_level[pres]

    # Join back with event-level metadata and baseline controls
    base_cols = ['uid', 'ticker', 'event_time']
    if BASELINE_CONTROL in df.columns: base_cols.append(BASELINE_CONTROL)
    for optional in ['period', 'call_modality']:
        if optional in df.columns: base_cols.append(optional)
    stock_index = df[base_cols].drop_duplicates(subset=['uid']).set_index('uid')
    joined = stock_index.join(call_level, how='inner')
    return joined.reset_index()

# ------------------------------ Masking Index (MI) ------------------------------
def compute_masking_index_per_call(df_segments_labeled: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MI per call/section:
    MI = share[ acoustic ∈ NEG_ACOUSTIC & text ∈ POS_TEXT ]
    Output columns: ['uid','MI_presentation','MI_q&a','Delta_MI'].
    """
    df = df_segments_labeled.copy()
    df['is_masking'] = df['acoustic_emotion'].isin(NEG_ACOUSTIC) & df['textual_emotion'].isin(POS_TEXT)
    grp = df.groupby(['uid','section'], observed=True)['is_masking'].mean().unstack()
    for col in ['presentation','q&a']:
        if col not in grp.columns: grp[col] = np.nan
    grp = grp.rename(columns={'presentation':'MI_presentation', 'q&a':'MI_q&a'})
    grp['Delta_MI'] = grp['MI_q&a'] - grp['MI_presentation']
    return grp.reset_index()

# ------------------------------ Market & Industry ------------------------------
def _norm_ticker(t):
    t = str(t).strip().upper()
    if not t: return None
    if t.startswith('^'):   
        return t
    t = t.replace('/', '-').replace('.', '-')
    return re.sub(r'[^A-Z0-9\-\^]', '', t)

def get_financial_data(tickers, start, end, cache_path):
    """Download close prices with caching; fall back to offline parquet/csv if available."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                obj = pickle.load(f)
                if isinstance(obj, dict) and 'Close' in obj and isinstance(obj['Close'], pd.DataFrame):
                    return obj
        except Exception:
            pass
    if yf is None:
        print("yfinance unavailable; returning None.")
        return None

    start = pd.to_datetime(start).tz_localize(None)
    end   = pd.to_datetime(end).tz_localize(None)
    end   = min(end, pd.Timestamp.today().normalize() + pd.Timedelta(days=3))

    universe = [x for x in {_norm_ticker(t) for t in tickers if isinstance(t, str)} if x]
    if MARKET_BENCHMARK not in universe: universe.append(MARKET_BENCHMARK)

    print("Downloading financial data…")
    def _download_batch(T):
        try:
            return yf.download(T, start=start, end=end, interval='1d',
                               auto_adjust=True, progress=True, threads=True, group_by=None)
        except Exception:
            try:
                return yf.download(T, start=start, end=end, interval='1d',
                                   auto_adjust=True, progress=False)
            except Exception:
                return None

    def _extract_close_panel(df):
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            lev0 = df.columns.get_level_values(0)
            levN = df.columns.get_level_values(-1)
            if 'Close' in lev0:
                panel = df['Close'].copy()
                if isinstance(panel.columns, pd.MultiIndex):
                    panel.columns = [c[-1] for c in panel.columns]
                return panel
            if 'Adj Close' in lev0:
                panel = df['Adj Close'].copy()
                if isinstance(panel.columns, pd.MultiIndex):
                    panel.columns = [c[-1] for c in panel.columns]
                return panel
            if 'Close' in levN:
                return df.xs('Close', axis=1, level=-1, drop_level=True).copy()
            if 'Adj Close' in levN:
                return df.xs('Adj Close', axis=1, level=-1, drop_level=True).copy()

        cols = df.columns
        if 'Close' in cols:
            name = 'SINGLE'
            return df[['Close']].rename(columns={'Close': name})
        if 'Adj Close' in cols:
            name = 'SINGLE'
            return df[['Adj Close']].rename(columns={'Adj Close': name})
        return None

    raw = _download_batch(universe)
    close_df = _extract_close_panel(raw)

    if close_df is None or close_df.empty:
        print('[get_financial_data] Batch returned empty; falling back to chunked downloads.')
        close_df = pd.DataFrame()
        CHUNK = 64
        for i in range(0, len(universe), CHUNK):
            chunk = universe[i:i+CHUNK]
            dfc = _extract_close_panel(_download_batch(chunk))
            if dfc is not None and not dfc.empty:
                close_df = pd.concat([close_df, dfc], axis=1)
       
        missing = [t for t in universe if t not in getattr(close_df, 'columns', [])]
        if missing:
            print(f"[get_financial_data] Still missing {len(missing)} tickers; retry individually…")
        for t in missing:
            try:
                dt = yf.download(t, start=start, end=end, interval='1d',
                                 auto_adjust=True, progress=False, threads=False)
                if dt is not None and not dt.empty:
                    if 'Close' in dt.columns:     close_df[t] = dt['Close']
                    elif 'Adj Close' in dt.columns: close_df[t] = dt['Adj Close']
            except Exception:
                pass

    if close_df is None or close_df.empty:
        fallback_dir = os.path.dirname(cache_path)
        for p in ['close_prices_fallback.parquet', 'close_prices_fallback.csv']:
            p = os.path.join(fallback_dir, p)
            if os.path.exists(p):
                print(f'[get_financial_data] Using offline fallback: {p}')
                try:
                    if p.endswith('.parquet'):
                        close_df = pd.read_parquet(p)
                    else:
                        tmp = pd.read_csv(p)
                        if {'Date','ticker','Close'} <= set(tmp.columns):
                            close_df = (tmp.assign(Date=pd.to_datetime(tmp['Date']))
                                          .pivot(index='Date', columns='ticker', values='Close')
                                          .sort_index())
                        else:
                            close_df = (tmp.rename(columns={tmp.columns[0]:'Date'})
                                          .assign(Date=pd.to_datetime(tmp['Date']))
                                          .set_index('Date'))
                except Exception:
                    pass
                break

    if close_df is None or close_df.empty:
        print("[get_financial_data] No data downloaded at all.")
        return None

    close_df = close_df.sort_index().dropna(how='all', axis=1)
    if MARKET_BENCHMARK not in close_df.columns:
        try:
            spy = yf.download(MARKET_BENCHMARK, start=start, end=end, interval='1d',
                              auto_adjust=True, progress=False, threads=False)
            if spy is not None and not spy.empty:
                if 'Close' in spy.columns:     close_df[MARKET_BENCHMARK] = spy['Close']
                elif 'Adj Close' in spy.columns: close_df[MARKET_BENCHMARK] = spy['Adj Close']
        except Exception:
            print(f"[get_financial_data] WARNING: failed to fetch benchmark {MARKET_BENCHMARK}.")

    close_df = close_df.loc[:, ~close_df.columns.duplicated()]
    data_structured = {'Close': close_df}
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data_structured, f)
    except Exception:
        pass
    return data_structured

def get_industry_data(tickers, cache_path):
    """Fetch a coarse industry/sector mapping per ticker (best-effort, cached)."""
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f: return pickle.load(f)
    if yf is None:
        print("yfinance unavailable; returning generic 'N/A' industries.")
        return {t: 'N/A' for t in tickers}
    print("Downloading ticker industry data...")
    industry_map = {}
    for ticker in tqdm(tickers, desc="Industries"):
        try:
            tk = yf.Ticker(ticker)
            info = None
            try:
                info = tk.get_info()
            except Exception:
                try:
                    info = tk.info
                except Exception:
                    info = {}
            industry = info.get('industry') or info.get('sector') or 'N/A'
            industry_map[ticker] = industry
        except Exception:
            industry_map[ticker] = 'N/A'
    with open(cache_path, 'wb') as f: pickle.dump(industry_map, f)
    return industry_map

def calculate_market_outcomes(df_events: pd.DataFrame, financial_data):
    """Compute CAR, RV, log_RV for multiple horizons under a market-model abnormal return."""
    if sm is None:
        print("statsmodels unavailable; cannot compute market outcomes.")
        return pd.DataFrame()
    print("Calculating Market Outcomes (CAR, RV, log_RV) and 10d Baseline…")
    df = df_events.copy()
    df['event_date_normalized'] = pd.to_datetime(df['event_time']).dt.normalize()
    prices = financial_data['Close']
    mkt_ret = prices[MARKET_BENCHMARK].pct_change().rename('market_return')

    results = []
    horizons = sorted(set(TARGET_HORIZONS_VOL + TARGET_HORIZONS_CAR))
    skip_no_ticker = skip_insufficient = skip_no_future = 0

    for _, event in tqdm(df.iterrows(), total=len(df), desc="Events"):
        ticker = event['ticker']; event_date = event['event_date_normalized']
        if ticker not in prices.columns or prices[ticker].isnull().all():
            skip_no_ticker += 1; continue
        stock_ret = prices[ticker].pct_change().rename('stock_return')
        data = pd.merge(stock_ret, mkt_ret, left_index=True, right_index=True, how='left').dropna(subset=['stock_return'])

        # Estimation window for market-model beta
        estimation_end   = event_date - pd.Timedelta(days=1)
        estimation_start = estimation_end - pd.Timedelta(days=ESTIMATION_WINDOW)
        est = data.loc[estimation_start:estimation_end]
        if len(est) < MIN_ESTIMATION_DAYS: skip_insufficient += 1; continue

        X = sm.add_constant(est['market_return'].fillna(0.0))
        model = sm.OLS(est['stock_return'], X).fit()

        # Event window (future)
        ev_all = data.loc[data.index >= event_date]
        if ev_all.empty: skip_no_future += 1; continue
        ev = ev_all.iloc[:max(horizons)]
        Xe = sm.add_constant(ev['market_return'].fillna(0.0))
        expected = model.predict(Xe)
        ev = ev.copy(); ev['abnormal_return'] = ev['stock_return'] - expected

        row = event.to_dict()
        # baseline control: 10-day realized volatility before the event
        pre = data.loc[:estimation_end].copy()
        pre10 = pre['stock_return'].dropna().iloc[-10:]
        row[BASELINE_CONTROL] = float(np.sqrt(np.sum(np.square(pre10.values)))) if len(pre10) >= 3 else np.nan
        for h in horizons:
            if len(ev) >= h:
                car    = float(ev['abnormal_return'].iloc[0:h].sum())
                rv     = float(np.sum(np.square(ev['stock_return'].iloc[0:h])))
                log_rv = np.log(rv + EPS)
            else:
                car = rv = log_rv = np.nan
            row[f'car_t+{h}d']     = car
            row[f'rv_t+{h}d']      = rv
            row[f'log_rv_t+{h}d']  = log_rv
        results.append(row)

    print(f"Processed={len(results)}, NoTicker={skip_no_ticker}, InsufficientHist={skip_insufficient}, NoFuture={skip_no_future}")
    return pd.DataFrame(results)

# ------------------------------ Utilities ------------------------------
def add_quarter(df: pd.DataFrame) -> pd.DataFrame:
    """Quarter labels and coarse COVID period bins for heterogeneity cuts."""
    df = df.copy()
    df['event_time'] = pd.to_datetime(df['event_time'])
    df['quarter'] = df['event_time'].dt.to_period('Q').astype(str)
    dt = df['event_time']
    df['period'] = np.where(dt < pd.Timestamp('2020-03-01'), 'Pre-COVID',
                     np.where(dt < pd.Timestamp('2022-01-01'), 'COVID', 'Post-COVID'))
    return df

# ------------------------------ Feature sets & Models ------------------------------
def _make_feature_sets(cols) -> dict:
    """Define model feature bundles consistent with the paper's comparisons."""
    financial = [BASELINE_CONTROL] if BASELINE_CONTROL in cols else []
    text = [c for c in cols if 'textual_' in c]
    if CFG.QA_ONLY: text = [c for c in text if '_q&a_' in c]
    acoustic = [c for c in cols if 'acoustic_' in c]
    if CFG.QA_ONLY: acoustic = [c for c in acoustic if '_q&a_' in c or '_delta_acoustic_' in c]
    orth_acoustic = [c for c in cols if c.startswith('orth_') and 'acoustic_' in c]
    non_acoustic = financial + text
    acoustic_delta_only = [c for c in cols if '_delta_acoustic_' in c]
    base_acoustic = (list(dict.fromkeys(acoustic + orth_acoustic)) if (CFG.ORTHOGONALIZE_ACOUSTIC and orth_acoustic) else acoustic)
    multimodal = financial + text + base_acoustic
    return {
        "Financial-Only": financial,
        "Textual-Only": text,
        "Financial+Textual": non_acoustic,
        "Acoustic-Only": acoustic,
        "Acoustic Δ (Q&A − Pres.)": acoustic_delta_only,
        "Multimodal": multimodal
    }

def _train_xgb_with_val(X_tr, y_tr, X_val, y_val, seed=GLOBAL_SEED):
    """XGB with early stopping on validation."""
    if xgb is None: raise ImportError("xgboost is required but not installed.")
    model = xgb.XGBRegressor(objective='reg:squarederror', missing=np.nan,
                             n_estimators=1000, learning_rate=0.05, max_depth=3,
                             subsample=1.0, colsample_bytree=1.0, random_state=seed,
                             n_jobs=-1, early_stopping_rounds=50, tree_method='hist')
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model

def _group_exclusion(train_df, val_df, test_df, group_col='ticker'):
    """Exclude test/val firms from the training set to prevent leakage (grouped splits)."""
    exclude = set(pd.concat([val_df[group_col], test_df[group_col]]).unique().tolist())
    return train_df[~train_df[group_col].isin(exclude)].copy()

def _rolling_quarter_windows(df: pd.DataFrame):
    """Yield rolling (train≤t-2, val=t-1, test=t) quarterly slices."""
    qs = sorted(df['quarter'].unique().tolist())
    for i in range(2, len(qs)):
        q_test = qs[i]; q_val = qs[i-1]; q_trn_max = qs[i-2]
        yield q_test, df[df['quarter'] <= q_trn_max].copy(), df[df['quarter'] == q_val].copy(), df[df['quarter'] == q_test].copy()

# ------------------------------ Table 1 (pooled OOS; 10d) ------------------------------
def run_model_comparison(df: pd.DataFrame, table_dir: str):
    """Table 1 (single-horizon 10d) comparing feature bundles."""
    print("\n--- Table 1 (pooled OOS; target=10d log_RV) ---")
    os.makedirs(table_dir, exist_ok=True)
    target_log, target_rv = f'log_rv_t+{PRIMARY_HORIZON}d', f'rv_t+{PRIMARY_HORIZON}d'
    data = add_quarter(df.dropna(subset=[target_log, target_rv]).copy())
    feats = _make_feature_sets(data.columns)
    per_model_log, per_model_rv = defaultdict(list), defaultdict(list)

    for q_test, df_tr, df_val, df_te in _rolling_quarter_windows(data):
        if len(df_te) < 20 or len(df_val) < 20 or len(df_tr) < 100: continue
        df_tr = _group_exclusion(df_tr, df_val, df_te)
        for name, F in feats.items():
            if not F: continue
            X_tr, y_tr = df_tr[F], df_tr[target_log]
            X_val, y_val = df_val[F], df_val[target_log]
            X_te,  y_te  = df_te[F],  df_te[target_log]
            model = _train_xgb_with_val(X_tr, y_tr, X_val, y_val)
            yhat_log = model.predict(X_te); yhat_rv = np.exp(yhat_log)
            per_model_log[name].append(pd.DataFrame({'uid': df_te['uid'].values, 'ticker': df_te['ticker'].values,
                                                     'y_true_log': y_te.values, 'y_pred_log': yhat_log}))
            per_model_rv[name].append(pd.DataFrame({'uid': df_te['uid'].values, 'ticker': df_te['ticker'].values,
                                                    'y_true_rv': df_te[target_rv].values, 'y_pred_rv': yhat_rv}))

    rows = []
    for name in feats.keys():
        if not per_model_log[name]: continue
        df_log = pd.concat(per_model_log[name], ignore_index=True)
        df_rv  = pd.concat(per_model_rv[name], ignore_index=True)
        r2 = _r2(df_log['y_true_log'], df_log['y_pred_log'])
        mse = mean_squared_error(df_log['y_true_log'], df_log['y_pred_log'])
        ql  = qlike_loss(df_rv['y_true_rv'], df_rv['y_pred_rv'])
        lo, hi = _cluster_bootstrap_ci_r2(df_log['y_true_log'], df_log['y_pred_log'], df_log['ticker'])
        rows.append({'Model Configuration': name, f'OOS R^2 (log_RV, {PRIMARY_HORIZON}d)': r2,
                     '95% CI (lower)': lo, '95% CI (upper)': hi,
                     f'OOS MSE (log_RV, {PRIMARY_HORIZON}d)': mse, f'OOS QLIKE (RV, {PRIMARY_HORIZON}d)': ql})
    out = pd.DataFrame(rows).set_index('Model Configuration').sort_values(f'OOS R^2 (log_RV, {PRIMARY_HORIZON}d)', ascending=False)
    out.to_csv(os.path.join(TABLE_DIR, "table1_performance_summary.csv"))
    print(out.to_string(float_format="%.4f"))
    return out

# ------------------------------ Table 1 Multihorizon ------------------------------
def run_model_comparison_multihorizon(df: pd.DataFrame, target_kind: str, table_dir: str):
    """Multi-horizon Table 1 (log_rv or car)."""
    assert target_kind in ('log_rv','car')
    print(f"\n--- Table 1 (multi-horizon; target={target_kind}) ---")
    horizons = TARGET_HORIZONS_VOL if target_kind=='log_rv' else TARGET_HORIZONS_CAR
    targets  = [f'{("log_rv" if target_kind=="log_rv" else "car")}_t+{h}d' for h in horizons]
    data = add_quarter(df.copy()); feats = _make_feature_sets(data.columns)
    results = []

    for name, F in feats.items():
        if not F: continue
        pooled_true = {h: [] for h in horizons}
        pooled_pred = {h: [] for h in horizons}
        for q_test, df_tr, df_val, df_te in _rolling_quarter_windows(data):
            mask = df_te[targets].notna().all(axis=1)
            df_te_h = df_te.loc[mask].copy()
            if len(df_te_h) < 20 or len(df_val) < 20 or len(df_tr) < 100: continue
            df_tr = _group_exclusion(df_tr, df_val, df_te_h)
            for h in horizons:
                tcol = f'{("log_rv" if target_kind=="log_rv" else "car")}_t+{h}d'
                if df_tr[F].empty or df_tr[tcol].isna().all(): continue
                try:
                    m = _train_xgb_with_val(df_tr[F], df_tr[tcol], df_val[F], df_val[tcol])
                    yh = m.predict(df_te_h[F])
                    pooled_true[h].append(df_te_h[tcol].values)
                    pooled_pred[h].append(yh)
                except Exception:
                    continue
        r = {'Model Configuration': name}
        for h in horizons:
            if pooled_true[h]:
                y = np.concatenate(pooled_true[h]); p = np.concatenate(pooled_pred[h])
                r[f'OOS R^2 ({target_kind}, {h}d)'] = _r2(y, p)
            else:
                r[f'OOS R^2 ({target_kind}, {h}d)'] = np.nan
        results.append(r)
    out = pd.DataFrame(results).set_index('Model Configuration')
    out = out.sort_values(f'OOS R^2 ({target_kind}, {PRIMARY_HORIZON}d)', ascending=False)
    fname = "table1_performance_summary_multihorizon.csv" if target_kind=='log_rv' else "table1_performance_summary_car_multihorizon.csv"
    out.to_csv(os.path.join(TABLE_DIR, fname))
    print(out.to_string(float_format="%.4f"))
    return out

# ------------------------------ Multi-horizon perf & ΔR² ------------------------------
def run_multihorizon_performance_and_delta(final_df: pd.DataFrame):
    """Produce: (i) multi-horizon R^2 for vol/CAR; (ii) ΔR^2 curves for Textual/Multimodal/Acoustic Δ."""
    data = add_quarter(final_df.copy())
    feats = _make_feature_sets(data.columns)

    F_base = feats.get("Financial-Only", [])
    F_full_multimodal = feats.get("Multimodal", [])
    F_full_acdelta    = feats.get("Acoustic Δ (Q&A − Pres.)", [])
    F_full_text       = feats.get("Financial+Textual", [])

    perf_multi = {}      
    delta_curves = {}    

    # (i) multi-horizon R^2
    for h in TARGET_HORIZONS_VOL:
        target = f'log_rv_t+{h}d'
        d = data.dropna(subset=[target])
        y_all, yhat_all = [], []
        for q_test, df_tr, df_val, df_te in _rolling_quarter_windows(d):
            if len(df_te) < 20 or len(df_val) < 20 or len(df_tr) < 100: continue
            df_tr = _group_exclusion(df_tr, df_val, df_te)
            if not F_full_multimodal: continue
            m = _train_xgb_with_val(df_tr[F_full_multimodal], df_tr[target], df_val[F_full_multimodal], df_val[target])
            yhat_all.append(m.predict(df_te[F_full_multimodal])); y_all.append(df_te[target].values)
        if y_all:
            perf_multi[f'vol_t+{h}d'] = _r2(np.concatenate(y_all), np.concatenate(yhat_all))

    for h in TARGET_HORIZONS_CAR:
        target = f'car_t+{h}d'
        d = data.dropna(subset=[target])
        y_all, yhat_all = [], []
        for q_test, df_tr, df_val, df_te in _rolling_quarter_windows(d):
            if len(df_te) < 20 or len(df_val) < 20 or len(df_tr) < 100: continue
            df_tr = _group_exclusion(df_tr, df_val, df_te)
            if not F_full_multimodal: continue
            m = _train_xgb_with_val(df_tr[F_full_multimodal], df_tr[target], df_val[F_full_multimodal], df_val[target])
            yhat_all.append(m.predict(df_te[F_full_multimodal])); y_all.append(df_te[target].values)
        if y_all:
            perf_multi[f'car_t+{h}d'] = _r2(np.concatenate(y_all), np.concatenate(yhat_all))

    # (ii) ΔR^2 vs Financial-Only
    for label, F_full in [('Textual', F_full_text),   ('Multimodal', F_full_multimodal), ('Acoustic Δ', F_full_acdelta)]:
        rows = []
        for h in TARGET_HORIZONS_VOL:
            target = f'log_rv_t+{h}d'
            d = data.dropna(subset=[target])
            yh_f, yh_b, ys, gps = [], [], [], []
            for q_test, df_tr, df_val, df_te in _rolling_quarter_windows(d):
                if len(df_te) < 20 or len(df_val) < 20 or len(df_tr) < 100: continue
                df_tr = _group_exclusion(df_tr, df_val, df_te)
                if F_base:
                    mb = _train_xgb_with_val(df_tr[F_base], df_tr[target], df_val[F_base], df_val[target])
                    yb = mb.predict(df_te[F_base])
                else:
                    yb = np.repeat(df_tr[target].mean(), repeats=len(df_te))
                if not F_full: 
                    continue
                mf = _train_xgb_with_val(df_tr[F_full], df_tr[target], df_val[F_full], df_val[target])
                yf = mf.predict(df_te[F_full])
                yh_b.append(yb); yh_f.append(yf); ys.append(df_te[target].values); gps.append(df_te['ticker'].values)
            if ys:
                y_true = np.concatenate(ys); yb_all = np.concatenate(yh_b); yf_all = np.concatenate(yh_f); g_all = np.concatenate(gps)
                dr2 = _r2(y_true, yf_all) - _r2(y_true, yb_all)
                lo, hi = _cluster_bootstrap_ci_delta_r2(y_true, yb_all, yf_all, g_all)
                rows.append({'Horizon (days)': h, 'Delta R^2': dr2, 'CI (lower)': lo, 'CI (upper)': hi})
        delta_curves[label] = pd.DataFrame(rows).sort_values('Horizon (days)') if rows else pd.DataFrame(columns=['Horizon (days)','Delta R^2','CI (lower)','CI (upper)'])

    return perf_multi, delta_curves

def run_final_model_analysis(final_df: pd.DataFrame):
    """Return multi-horizon performance, feature importance (bootstrap), and ΔR² curves."""
    perf_multi, delta_curves = run_multihorizon_performance_and_delta(final_df)
    data = add_quarter(final_df.copy())
    feats = _make_feature_sets(data.columns)
    multi = feats.get("Multimodal", [])
    target = f'log_rv_t+{PRIMARY_HORIZON}d'
    d = data.dropna(subset=[target])
    last = None
    for w in _rolling_quarter_windows(d): last = w
    if last is None: return perf_multi, pd.DataFrame(), delta_curves
    _, df_tr, df_val, _ = last
    df_tr = _group_exclusion(df_tr, df_val, df_val)
    X, y = df_tr[multi], df_tr[target]
    boot = []
    for b in tqdm(range(N_BOOTSTRAPS_IMPORTANCE), desc="Bootstrap (feature importance)", leave=False):
        idx = X.sample(frac=1.0, replace=True, random_state=1000+b).index
        Xi, yi = X.loc[idx], y.loc[idx]
        m = xgb.XGBRegressor(objective='reg:squarederror', missing=np.nan, n_estimators=250, max_depth=3,
                             subsample=1.0, colsample_bytree=1.0, random_state=1000+b,
                             n_jobs=-1, tree_method='hist')
        m.fit(Xi, yi, verbose=False)

        try:
            gain_map = m.get_booster().get_score(importance_type='gain')
            series = pd.Series({k: gain_map.get(k, 0.0) for k in m.feature_names_in_})
        except Exception:
            series = pd.Series(m.feature_importances_, index=m.feature_names_in_)
        boot.append(series)
    imp = pd.concat(boot, axis=1)
    stats = pd.DataFrame({'mean_importance': imp.mean(axis=1), 'std_importance': imp.std(axis=1)}
                         ).sort_values('mean_importance', ascending=False)
    return perf_multi, stats, delta_curves

# ------------------------------ Robustness (Fig.4) ------------------------------
def run_robustness_analysis_fig4(final_df: pd.DataFrame):
    """Absolute R^2 by subsets (industry/time/volatility quartiles)."""
    print("\n--- Robustness (absolute R^2; 10d) ---")
    target = f'log_rv_t+{PRIMARY_HORIZON}d'
    d_all = add_quarter(final_df.dropna(subset=[target]).copy())
    feats = _make_feature_sets(d_all.columns)
    multi = feats.get("Multimodal", [])
    pooled = []
    for q_test, df_tr, df_val, df_te in _rolling_quarter_windows(d_all):
        if len(df_te) < 20 or len(df_val) < 20 or len(df_tr) < 100: continue
        df_tr = _group_exclusion(df_tr, df_val, df_te)
        m = _train_xgb_with_val(df_tr[multi], df_tr[target], df_val[multi], df_val[target])
        yhat = m.predict(df_te[multi])
        keep = ['uid', 'ticker', 'event_time', 'period', 'industry'] + ([BASELINE_CONTROL] if BASELINE_CONTROL in df_te.columns else [])
        chunk = df_te[keep].copy(); chunk['y'] = df_te[target].values; chunk['yhat'] = yhat
        pooled.append(chunk)
    if not pooled:
        return {'periods': pd.DataFrame(), 'time_quartiles': pd.DataFrame(), 'vol_quartiles': pd.DataFrame(), 'industry': pd.DataFrame()}
    pred = pd.concat(pooled, ignore_index=True); pred['event_time'] = pd.to_datetime(pred['event_time'])

    def eval_subset(mask, name):
        sub = pred[mask]; n = len(sub)
        if n < MIN_SUBSET_N or sub['y'].nunique() < 2:
            return {'Subset': name, 'R-squared': np.nan, 'CI (lower)': np.nan, 'CI (upper)': np.nan, 'N': n}
        r2 = _r2(sub['y'], sub['yhat']); lo, hi = _cluster_bootstrap_ci_r2(sub['y'], sub['yhat'], sub['ticker'])
        return {'Subset': name, 'R-squared': r2, 'CI (lower)': lo, 'CI (upper)': hi, 'N': n}

    periods_order = ['Pre-COVID', 'COVID', 'Post-COVID']
    df_periods = pd.DataFrame([eval_subset(pred['period'] == p, p) for p in periods_order])

    ranks = pred['event_time'].rank(method='first'); labels_time = [f'Time Q{i}' for i in range(1, N_Q_TIME+1)]
    pred['time_q'] = pd.qcut(ranks, q=N_Q_TIME, labels=labels_time)
    df_time = pd.DataFrame([eval_subset(pred['time_q'] == lab, lab) for lab in labels_time])

    if BASELINE_CONTROL in pred.columns:
        ranks_v = pred[BASELINE_CONTROL].rank(method='first'); labels_vol = [f'Vol. Q{i}' for i in range(1, N_Q_VOL+1)]
        pred['vol_q'] = pd.qcut(ranks_v, q=N_Q_VOL, labels=labels_vol)
        df_vol = pd.DataFrame([eval_subset(pred['vol_q'] == lab, lab) for lab in labels_vol])
    else:
        df_vol = pd.DataFrame(columns=['Subset','R-squared','CI (lower)','CI (upper)','N'])

    if 'industry' in pred.columns:
        pred['industry'] = pred['industry'].replace('N/A', np.nan)
        top_industries = pred['industry'].value_counts().nlargest(N_TOP_INDUSTRIES).index.tolist()
        df_ind = pd.DataFrame([eval_subset(pred['industry'] == ind, ind) for ind in top_industries])
    else:
        df_ind = pd.DataFrame(columns=['Subset','R-squared','CI (lower)','CI (upper)','N'])
    return {'periods': df_periods, 'time_quartiles': df_time, 'vol_quartiles': df_vol, 'industry': df_ind}

def run_robustness_analysis_fig4_delta(final_df: pd.DataFrame):
    """ΔR^2 by subsets (industry/time/volatility quartiles)."""
    print("\n--- Robustness (ΔR^2; 10d) ---")
    target = f'log_rv_t+{PRIMARY_HORIZON}d'
    d_all = add_quarter(final_df.dropna(subset=[target]).copy())
    feats = _make_feature_sets(d_all.columns)
    baseF = feats.get("Financial-Only", [])
    fullF = feats.get("Multimodal", [])
    pooled = []
    for q_test, df_tr, df_val, df_te in _rolling_quarter_windows(d_all):
        if len(df_te) < 20 or len(df_val) < 20 or len(df_tr) < 100: continue
        df_tr = _group_exclusion(df_tr, df_val, df_te)
        mb = _train_xgb_with_val(df_tr[baseF], df_tr[target], df_val[baseF], df_val[target]) if baseF else None
        mf = _train_xgb_with_val(df_tr[fullF], df_tr[target], df_val[fullF], df_val[target]) if fullF else None
        if mf is None: continue
        yhat_full = mf.predict(df_te[fullF])
        yhat_base = (mb.predict(df_te[baseF]) if mb is not None else np.repeat(df_tr[target].mean(), repeats=len(df_te)))
        keep = ['uid', 'ticker', 'event_time', 'period', 'industry'] + ([BASELINE_CONTROL] if BASELINE_CONTROL in df_te.columns else [])
        chunk = df_te[keep].copy(); chunk['y'] = df_te[target].values; chunk['yhat_base'] = yhat_base; chunk['yhat_full'] = yhat_full
        pooled.append(chunk)
    if not pooled:
        return {'periods': pd.DataFrame(), 'time_quartiles': pd.DataFrame(), 'vol_quartiles': pd.DataFrame(), 'industry': pd.DataFrame()}
    pred = pd.concat(pooled, ignore_index=True); pred['event_time'] = pd.to_datetime(pred['event_time'])

    def eval_subset(mask, name):
        sub = pred[mask]; n = len(sub)
        if n < MIN_SUBSET_N or sub['y'].nunique() < 2:
            return {'Subset': name, 'Delta R^2': np.nan, 'CI (lower)': np.nan, 'CI (upper)': np.nan, 'N': n}
        dr2 = _r2(sub['y'], sub['yhat_full']) - _r2(sub['y'], sub['yhat_base'])
        lo, hi = _cluster_bootstrap_ci_delta_r2(sub['y'], sub['yhat_base'], sub['yhat_full'], sub['ticker'])
        return {'Subset': name, 'Delta R^2': dr2, 'CI (lower)': lo, 'CI (upper)': hi, 'N': n}

    periods_order = ['Pre-COVID', 'COVID', 'Post-COVID']
    df_periods = pd.DataFrame([eval_subset(pred['period'] == p, p) for p in periods_order])

    ranks = pred['event_time'].rank(method='first'); labels_time = [f'Time Q{i}' for i in range(1, N_Q_TIME+1)]
    pred['time_q'] = pd.qcut(ranks, q=N_Q_TIME, labels=labels_time)
    df_time = pd.DataFrame([eval_subset(pred['time_q'] == lab, lab) for lab in labels_time])

    if BASELINE_CONTROL in pred.columns:
        ranks_v = pred[BASELINE_CONTROL].rank(method='first'); labels_vol = [f'Vol. Q{i}' for i in range(1, N_Q_VOL+1)]
        pred['vol_q'] = pd.qcut(ranks_v, q=N_Q_VOL, labels=labels_vol)
        df_vol = pd.DataFrame([eval_subset(pred['vol_q'] == lab, lab) for lab in labels_vol])
    else:
        df_vol = pd.DataFrame(columns=['Subset','Delta R^2','CI (lower)','CI (upper)','N'])

    if 'industry' in pred.columns:
        pred['industry'] = pred['industry'].replace('N/A', np.nan)
        top_industries = pred['industry'].value_counts().nlargest(N_TOP_INDUSTRIES).index.tolist()
        df_ind = pd.DataFrame([eval_subset(pred['industry'] == ind, ind) for ind in top_industries])
    else:
        df_ind = pd.DataFrame(columns=['Subset','Delta R^2','CI (lower)','CI (upper)','N'])
    return {'periods': df_periods, 'time_quartiles': df_time, 'vol_quartiles': df_vol, 'industry': df_ind}

# ------------------------------ Δ-Feature Correlations (Fig.3b) ------------------------------
def compute_delta_feature_correlations(final_df: pd.DataFrame, topK: int = 12):
    """Compute clustered correlations between Δ-features and volatility across horizons."""
    def _build_core_delta_indices(df: pd.DataFrame) -> pd.DataFrame:
        df = add_quarter(df.copy())
        keep = [c for c in df.columns if any(k in c for k in ['_delta_acoustic_', '_q&a_acoustic_', '_presentation_acoustic_'])]
        cols = ['uid','ticker','event_time','quarter','industry'] + keep
        df = df[cols].copy()
        def _agg(pattern):
            cols = [c for c in df.columns if c.endswith(pattern) and any(r in c for r in ['CEO_','CFO_','CXO_'])]
            return df[cols].mean(axis=1) if cols else None
        out = df[['uid','ticker','event_time','quarter','industry']].copy()
        out['delta_dominance_mean'] = _agg('delta_acoustic_dominance_mean')
        out['delta_dominance_std']  = _agg('delta_acoustic_dominance_std')
        out['delta_dominance_kurt'] = _agg('delta_acoustic_dominance_kurtosis') if _agg('delta_acoustic_dominance_kurtosis') is not None else _agg('delta_acoustic_dominance_kurt')
        out['delta_valence_mean']   = _agg('delta_acoustic_valence_mean')
        out['delta_valence_skew']   = _agg('delta_acoustic_valence_skew')
        return out

    core = _build_core_delta_indices(final_df.copy())
    targets = {h: f'log_rv_t+{h}d' for h in sorted(set(TARGET_HORIZONS_VOL)) if f'log_rv_t+{h}d' in final_df.columns}
    if not targets: return pd.DataFrame(), pd.DataFrame()
    h0 = 10 if 10 in targets else max(targets.keys())
    tcol0 = targets[h0]

    rows_anchor = []
    for f in [c for c in core.columns if c.startswith('delta_')]:
        sub = final_df[['uid','ticker','event_time', tcol0]].merge(core[['uid','ticker','event_time', f]],
               on=['uid','ticker','event_time'], how='inner').dropna()
        if sub.empty or sub[f].nunique() < 3 or sub[tcol0].nunique() < 3: continue
        y = sub[tcol0].values; x = sub[f].values; g = sub['ticker'].values
        corr = float(np.corrcoef(y, x)[0,1]) if np.isfinite(np.corrcoef(y, x)[0,1]) else np.nan
        lo, hi = _cluster_bootstrap_ci_corr(y, x, g)
        rows_anchor.append({'Feature': f, 'Corr': corr, 'CI_low': lo, 'CI_high': hi})
    corr_anchor = pd.DataFrame(rows_anchor).dropna(subset=['Corr'])
    if corr_anchor.empty: return pd.DataFrame(), pd.DataFrame()
    corr_anchor = corr_anchor.reindex(corr_anchor['Corr'].abs().sort_values(ascending=False).index).head(topK)

    keep_feats = corr_anchor['Feature'].tolist()
    mh = []
    for h, tcol in targets.items():
        for f in keep_feats:
            sub = final_df[['uid','ticker','event_time', tcol]].merge(core[['uid','ticker','event_time', f]],
                   on=['uid','ticker','event_time'], how='inner').dropna()
            if sub.empty or sub[f].nunique() < 3 or sub[tcol].nunique() < 3:
                mh.append({'Feature': f, 'Horizon (days)': h, 'Corr': np.nan, 'CI_low': np.nan, 'CI_high': np.nan}); continue
            y = sub[tcol].values; x = sub[f].values; g = sub['ticker'].values
            corr = float(np.corrcoef(y, x)[0,1]) if np.isfinite(np.corrcoef(y, x)[0,1]) else np.nan
            lo, hi = _cluster_bootstrap_ci_corr(y, x, g)
            mh.append({'Feature': f, 'Horizon (days)': h, 'Corr': corr, 'CI_low': lo, 'CI_high': hi})
    corr_multi = pd.DataFrame(mh)
    return corr_anchor, corr_multi

def _pretty_feature_label(x: str) -> str:
    return (str(x).replace('delta_', 'Δ ')
            .replace('_dominance_', ' Dominance ')
            .replace('_valence_', ' Valence ')
            .replace('_mean', ' (mean)').replace('_std', ' (std)')
            .replace('_skew', ' (skew)').replace('_kurt', ' (kurtosis)')
            .replace('_', ' ').strip())

def _plot_forest_dynamic(ax, coef_tbl: pd.DataFrame, dyn_tbl: pd.DataFrame, title: str, order=None):
    """Forest plot with dynamic multi-horizon overlays."""
    if coef_tbl is None or coef_tbl.empty:
        ax.text(0.5,0.5,'Not Available',ha='center',va='center',fontsize=9,color='gray')
        ax.set_title(title, fontsize=9, loc='left'); ax.set_xticks([]); ax.set_yticks([]); return
    d = coef_tbl.copy()
    if order: d = d[d['Feature'].isin(order)]
    d['label'] = d['Feature'].map(_pretty_feature_label)
    d = d.set_index('Feature')
    if order: d = d.reindex(order)
    d = d.reset_index()
    y = np.arange(len(d))

    xerr = _safe_err(d['Corr'], d['CI_low'], d['CI_high'])
    ax.errorbar(d['Corr'], y, xerr=xerr, fmt='o', color=PRIMARY_COLOR, capsize=2.5, ecolor='#5B5B5B', elinewidth=0.9, label=f'{PRIMARY_HORIZON}d')
    ax.set_yticks(y, d['label'], fontsize=6)
    ax.axvline(0, color='black', linewidth=0.6, linestyle='--')
    ax.set_xlabel('Correlation (ticker-cluster CI)'); ax.set_title(title, fontsize=9, loc='left'); ax.grid(axis='x', linestyle=':')

    if dyn_tbl is not None and not dyn_tbl.empty:
        sub = dyn_tbl.dropna(subset=['Corr']).copy()
        sub = sub[sub['Feature'].isin(d['Feature']) & sub['Horizon (days)'].isin(HORIZON_COLORS.keys())]
        ymap = dict(zip(d['Feature'], y))
        for h, group in sub.groupby('Horizon (days)'):
            if int(h) == int(PRIMARY_HORIZON): continue
            g = group.copy()
            g['yy'] = g['Feature'].map(ymap)
            g = g.dropna(subset=['yy'])
            if g.empty: continue
            c = HORIZON_COLORS.get(int(h), ACCENT_COLORS[0])
            xerr_g = _safe_err(g['Corr'], g['CI_low'], g['CI_high'])
            offset = (int(h) - int(PRIMARY_HORIZON)) * 0.02
            ax.errorbar(g['Corr'], g['yy'] + offset, xerr=xerr_g, fmt='.', color=c, capsize=1.5, elinewidth=0.7, markersize=5,
                        label=f'{int(h)}d')
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    if uniq:
        ax.legend(uniq.values(), uniq.keys(), loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=7)

# ------------------------------ Fig.2/3/4 Assembly ------------------------------
def create_fig4_combined(robust_abs, robust_delta, output_dir):
    """Figure 4: three panels—industry / time quartiles / volatility quartiles."""
    set_publication_style()
    panels = CFG.F4_PANELS
    fig, axes = plt.subplots(1, len(panels), figsize=(mm2inch(180), mm2inch(95)), gridspec_kw={'wspace': 0.70})
    if len(panels) == 1: axes = np.array([axes])

    # Define a mapping for industry name abbreviations to ensure uniqueness and clarity
    industry_short_names = {
        'Specialty Business Services': 'Spec. Biz Svcs',
        'Insurance - Property & Casualty': 'Insurance (P&C)',
        'Drug Manufacturers - General': 'Drug Mfg. (Gen)',
        'Semiconductor Equipment & Materials': 'Semicon. Equip.',
        'Utilities - Regulated Electric': 'Regulated Electric',
        'Software - Application': 'Software (App)',
        'Health Information Services': 'Health Info Svcs',
        'Specialty Industrial Machinery': 'Spec. Machinery',
        'Financial Conglomerates': 'Fin. Conglomer.',
        'Asset Management': 'Asset Mgmt.',
        'Specialty Retail': 'Specialty Retail',
        'Medical Devices': 'Medical Devices',
        'Banks - Diversified': 'Banks (Divers.)',
        'Aerospace & Defense': 'Aerosp. & Defense',
        'Oil & Gas Integrated': 'Oil & Gas (Int.)'
    }

    # Apply the mapping to the 'Subset' column of the industry dataframes
    if 'industry' in panels:
        for data_dict in [robust_abs, robust_delta]:
            if 'industry' in data_dict and not data_dict['industry'].empty:
                df_ind = data_dict['industry']
                df_ind['Subset'] = df_ind['Subset'].map(industry_short_names).fillna(df_ind['Subset'])

    def _panel(ax, df_abs, df_delta, title, label_transform=None):
        if (df_abs is None or df_abs.empty) and (df_delta is None or df_delta.empty):
            ax.text(0.5,0.5,'Not Available',ha='center',va='center',fontsize=9,color='gray'); ax.set_title(title, fontsize=9, loc='left'); return
        if df_abs is not None and not df_abs.empty:
            da = df_abs.dropna(subset=['R-squared']).copy().sort_values('R-squared'); order = da['Subset'].astype(str).tolist()
        else:
            dd = df_delta.dropna(subset=['Delta R^2']).copy().sort_values('Delta R^2'); order = dd['Subset'].astype(str).tolist()

        A = df_abs.set_index('Subset') if df_abs is not None and not df_abs.empty else pd.DataFrame().set_index(pd.Index([]))
        D = df_delta.set_index('Subset') if df_delta is not None and not df_delta.empty else pd.DataFrame().set_index(pd.Index([]))
        y = np.arange(len(order)); labels = []
        for name in order:
            n_abs = A.loc[name, 'N'] if name in A.index and 'N' in A.columns and pd.notna(A.loc[name,'N']) else None
            n_delta = D.loc[name, 'N'] if name in D.index and 'N' in D.columns and pd.notna(D.loc[name,'N']) else None
            if callable(label_transform):
                lab = label_transform(name, n_abs, n_delta); lab = _wrap_label(lab, CFG.MAX_LABEL_CHARS)
            else:
                base = _wrap_label(name, CFG.MAX_LABEL_CHARS); n_show = int(n_abs if n_abs is not None else (n_delta if n_delta is not None else 0))
                lab = f"{base} (n={n_show})" if n_show > 0 else base
            labels.append(lab)

        if len(A) and 'R-squared' in A.columns:
            xa  = np.array([A.loc[name,'R-squared'] if name in A.index else np.nan for name in order])
            loa = np.array([A.loc[name,'CI (lower)'] if name in A.index else np.nan for name in order])
            hia = np.array([A.loc[name,'CI (upper)'] if name in A.index else np.nan for name in order])
            ax.errorbar(xa, y+0.16, xerr=_safe_err(xa, loa, hia),
                        fmt='o', color=PRIMARY_COLOR, capsize=2.5, elinewidth=0.9, label='Absolute $R^2$')
        if len(D) and 'Delta R^2' in D.columns:
            xd  = np.array([D.loc[name,'Delta R^2'] if name in D.index else np.nan for name in order])
            lod = np.array([D.loc[name,'CI (lower)'] if name in D.index else np.nan for name in order])
            hid = np.array([D.loc[name,'CI (upper)'] if name in D.index else np.nan for name in order])
            ax.errorbar(xd, y-0.16, xerr=_safe_err(xd, lod, hid),
                        fmt='s', color=ACCENT_COLORS[0], capsize=2.5, elinewidth=0.9, label='Incremental $\\Delta R^2$')
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6)
        for t in ax.get_yticklabels(): t.set_wrap(True); t.set_ha('right'); t.set_va('center')
        ax.margins(y=0.05); ax.set_title(title, fontsize=9, loc='left'); ax.set_xlabel('Out-of-Sample $R^2$ / $\\Delta R^2$')
        ax.axvline(0, color='black', linewidth=0.6, linestyle='--'); ax.grid(axis='x', linestyle=':')

    titles = {'industry': f'(a) Top-{N_TOP_INDUSTRIES} Industries', 'time_quartiles': '(b) Time Quartiles', 'vol_quartiles':  '(c) Volatility Quartiles (10d)'}
    for i, key in enumerate(panels):
        _panel(axes[i], robust_abs.get(key, pd.DataFrame()), robust_delta.get(key, pd.DataFrame()),
               titles[key], label_transform=None)

    fig.subplots_adjust(left=0.30, right=0.98, top=0.92, bottom=0.22, wspace=0.70)
    handles = [mpl.lines.Line2D([], [], color=PRIMARY_COLOR, marker='o', linestyle='None', label='Absolute $R^2$'),
               mpl.lines.Line2D([], [], color=ACCENT_COLORS[0], marker='s', linestyle='None', label='Incremental $\\Delta R^2$')]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.6, 0.02), bbox_transform=fig.transFigure)
    _save_plot(fig, "fig4_robustness_combined", output_dir); plt.close(fig)

def format_feature_label(label):
    label = label.replace('_', ' ')
    replacements = {
        'acoustic': 'Acou.',
        'dominance': 'Dom.',
        'arousal': 'Arou.',
        'valence': 'Val.',
        'presentation': 'Pres.',
        'textual': 'Txt.',
        'unbiased': 'unb.',
        'skewness': 'skew',
        'kurtosis': 'kurt',
        'historical': 'Hist.',
        'financial': 'Fin.',
        'fundamental': 'Fund.',
        'volatility': 'Vol.',
        'returns': 'Ret.',
        'multimodal': 'Multi.',
        'financial-only': 'Fin.-Only',
        'textual-only': 'Txt.-Only',
        'acoustic-only': 'Acou.-Only',
        'multimodal': 'Multi.',
        'financial+textual': 'Fin.+Txt.',
        'acoustic Δ (q&a − pres.)': 'Acou. Δ (Q&A − Pres.)',
        'q&a': 'Q&A',
        'std': '(σ)',
        'mean': '(μ)',
        'skew': '(skew)',
        'kurt': '(kurt)',
        'delta': 'Δ',
    }
    for old, new in replacements.items():
        label = label.replace(old, new)

    if label.startswith('ceo '): label = 'CEO ' + label[4:]
    if label.startswith('cfo '): label = 'CFO ' + label[4:]
    if label.startswith('cxo '): label = 'CXO ' + label[4:]

    return label.strip() 

def create_final_plots(model_results_tuple, ablation_results, robustness_results, robustness_delta_results, final_df, output_dir):
    """Figure 2 & 3 & 4 assembly."""
    perf_multi, importance_stats, delta_curves = model_results_tuple
    set_publication_style()

    # ---------------- (Figure 2) ----------------
    fig2, axes2 = plt.subplots(2, 2, figsize=FIGSIZE_QUAD, gridspec_kw={'wspace': 0.35, 'hspace': 0.6})
    perf_items = []
    for k, v in perf_multi.items():
        typ = 'Return (CAR)' if k.startswith('car_t+') else 'Risk (Volatility)'
        horizon = k.split('+')[-1].replace('d', '')
        perf_items.append({'Target': k, 'R-squared': v, 'Type': typ, 'Horizon': horizon})
    perf_df = pd.DataFrame(perf_items); order_h = ['1','5','10','15','20','25','30']
    if not perf_df.empty:
        sns.barplot(x='Horizon', y='R-squared', hue='Type', data=perf_df, ax=axes2[0,0], order=order_h, dodge=True, width=0.75)
        axes2[0,0].set_title('(a) Predictive Power for Volatility vs. Returns', fontsize=9, loc='left'); axes2[0,0].tick_params(axis='x', rotation=0)
        axes2[0,0].set_ylabel('Out-of-Sample $R^2$'); axes2[0,0].set_xlabel('Prediction Horizon (days)')
        axes2[0,0].axhline(0, color='black', linewidth=0.6, linestyle='--')
        handles, labels = axes2[0,0].get_legend_handles_labels()
        axes2[0,0].legend(handles, labels, frameon=False, loc='lower left', bbox_to_anchor=(0.1, 0.02))
    else:
        axes2[0,0].text(0.5,0.5,'Not Available',ha='center',va='center',fontsize=9,color='gray'); axes2[0,0].set_title('(a) Predictive Power for Volatility vs. Returns', fontsize=9, loc='left')

    short_map = {'Financial-Only':'Financial','Textual-Only':'Textual','Acoustic-Only':'Acoustic','Multimodal':'Multimodal','Financial+Textual':'Financial+Textual','Acoustic Δ (Q&A − Pres.)':'Acoustic Δ'}
    plot_data_ablation = ablation_results.copy().reset_index()
    plot_data_ablation['Model Short'] = plot_data_ablation['Model Configuration'].map(short_map)
    plot_data_ablation = plot_data_ablation.dropna(subset=['Model Short'])
    ycol = f'OOS R^2 (log_RV, {PRIMARY_HORIZON}d)'
    if not plot_data_ablation.empty and ycol in plot_data_ablation.columns:
        sns.barplot(x='Model Short', y=ycol, data=plot_data_ablation, ax=axes2[0,1], order=['Financial', 'Textual', 'Financial+Textual', 'Acoustic', 'Multimodal'])
        axes2[0,1].set_title(f'(b) Multimodal vs. Baselines ({PRIMARY_HORIZON}d Vol.)', fontsize=9, loc='left')
        axes2[0,1].set_ylabel('Out-of-Sample $R^2$'); axes2[0,1].set_xlabel('Prediction Model')
        # MODIFICATION START: Correctly rotate and align x-axis labels
        axes2[0,1].tick_params(axis='x', rotation=30)
        plt.setp(axes2[0,1].get_xticklabels(), ha='right')
        # MODIFICATION END
    else:
        axes2[0,1].text(0.5,0.5,'Not Available',ha='center',va='center',fontsize=9,color='gray'); axes2[0,1].set_title(f'(b) Multimodal vs. Baselines ({PRIMARY_HORIZON}d Vol.)', fontsize=9, loc='left')

    # (c) Incremental ΔR^2 bars at 10d
    axc = axes2[1,0]
    d_text  = delta_curves.get('Textual', pd.DataFrame())
    d_acdel = delta_curves.get('Acoustic Δ', pd.DataFrame())
    d_multi = delta_curves.get('Multimodal', pd.DataFrame())

    bars, lows, highs, labels = [], [], [], []

    if isinstance(d_text, pd.DataFrame) and not d_text.empty and PRIMARY_HORIZON in d_text['Horizon (days)'].values:
        r = d_text.loc[d_text['Horizon (days)'] == PRIMARY_HORIZON].iloc[0]
        bars.append(float(r['Delta R^2'])); lows.append(float(r['CI (lower)'])); highs.append(float(r['CI (upper)']))
        labels.append('Textual')

    if isinstance(d_acdel, pd.DataFrame) and not d_acdel.empty and PRIMARY_HORIZON in d_acdel['Horizon (days)'].values:
        r = d_acdel.loc[d_acdel['Horizon (days)'] == PRIMARY_HORIZON].iloc[0]
        bars.append(float(r['Delta R^2'])); lows.append(float(r['CI (lower)'])); highs.append(float(r['CI (upper)']))
        labels.append('Acoustic')

    if isinstance(d_multi, pd.DataFrame) and not d_multi.empty and PRIMARY_HORIZON in d_multi['Horizon (days)'].values:
        r = d_multi.loc[d_multi['Horizon (days)'] == PRIMARY_HORIZON].iloc[0]
        bars.append(float(r['Delta R^2'])); lows.append(float(r['CI (lower)'])); highs.append(float(r['CI (upper)']))
        labels.append('Multimodal')

    if bars:
        x = np.arange(len(bars))
        yerr = _safe_err(bars, lows, highs)
        axc.bar(x, bars, yerr=yerr, capsize=2.5, ecolor='#5B5B5B', alpha=0.95)
        axc.set_xticks(x, labels, rotation=0)
        axc.set_ylabel('Incremental $\\Delta R^2$ vs. Financial')
        axc.set_title(f'(c) Incremental $\Delta R^2$ vs. Financial ({PRIMARY_HORIZON}d)', fontsize=9, loc='left')
        axc.axhline(0, color='black', linewidth=0.6, linestyle='--'); axc.grid(axis='y', linestyle=':')
    else:
        axc.text(0.5,0.5,'Not Available',ha='center',va='center',fontsize=9,color='gray')
        axc.set_title(f'(c) IIncremental $\Delta R^2$ vs. Financial ({PRIMARY_HORIZON}d)', fontsize=9, loc='left')

    # (d) Multi-horizon ΔR^2 curve (volatility)
    if isinstance(d_multi, pd.DataFrame) and not d_multi.empty:
        ax = axes2[1,1]; h = d_multi['Horizon (days)'].values; y = d_multi['Delta R^2'].values; lo = d_multi['CI (lower)'].values; hi = d_multi['CI (upper)'].values
        yerr = _safe_err(y, lo, hi)
        ax.errorbar(h, y, yerr=yerr, fmt='o-', capsize=2.5, elinewidth=0.9, linewidth=1.2, color=PRIMARY_COLOR)
        ax.set_xticks([1,5,10,15,20,25,30]); ax.set_xlabel('Prediction Horizon (days)'); ax.set_ylabel(r'Incremental $\Delta R^2$ vs. Financial')
        ax.set_title(r'(d) Multi-horizon Incremental $\Delta R^2$ (Volatility)', fontsize=9, loc='left'); ax.axhline(0, color='black', linewidth=0.6, linestyle='--'); ax.grid(True, linestyle=':')
    else:
        axes2[1,1].text(0.5,0.5,'Not Available',ha='center',va='center',fontsize=9,color='gray'); axes2[1,1].set_title(r'(d) Multi-horizon Incremental $\Delta R^2$ (Volatility)', fontsize=9, loc='left')

    _save_plot(fig2, "fig2_model_performance", output_dir); plt.close(fig2)

    # ---------------- (Figure 3) ----------------
    corr_anchor, corr_multi = compute_delta_feature_correlations(final_df)
    fig3, axes3 = plt.subplots(1, 2, figsize=(mm2inch(180), mm2inch(95)), gridspec_kw={'wspace': 0.35})

    if importance_stats is not None and not importance_stats.empty:
        imp_top = importance_stats.head(15).sort_values('mean_importance')
        imp_top = imp_top.copy(); 
        imp_top.index = [format_feature_label(idx) for idx in imp_top.index] 
        # imp_top.index = [_wrap_label(idx, max_len=36) for idx in imp_top.index]
        imp_top['is_delta'] = [('_delta_' in idx) for idx in imp_top.index]
        colors = [ACCENT_COLORS[0] if dlt else PRIMARY_COLOR for dlt in imp_top['is_delta']]
        axes3[0].barh(imp_top.index, imp_top['mean_importance'], xerr=imp_top['std_importance'], color=colors, capsize=2.5, ecolor='#5B5B5B', error_kw={'elinewidth': 0.9})
        axes3[0].set_title(f'(a) Top 15 Stable Features for {PRIMARY_HORIZON}d Volatility', fontsize=9, loc='left')
        axes3[0].set_xlabel('Mean Gini Importance (bootstrap)'); axes3[0].set_ylabel('Feature'); axes3[0].grid(axis='x', linestyle=':'); axes3[0].tick_params(axis='y', labelsize=6)
        axes3[0].legend(handles=[Patch(facecolor=PRIMARY_COLOR, label='Level features'),
                                 Patch(facecolor=ACCENT_COLORS[0], label='Δ-features')], loc='lower right', frameon=False)
    else:
        axes3[0].text(0.5,0.5,'Not Available',ha='center',va='center',fontsize=9,color='gray'); axes3[0].set_title(f'(a) Top 15 Stable Features for {PRIMARY_HORIZON}d Volatility', fontsize=9, loc='left')

    if isinstance(corr_anchor, pd.DataFrame) and not corr_anchor.empty:
        coef_tbl = corr_anchor[['Feature','Corr','CI_low','CI_high']].copy()
        order = coef_tbl.sort_values('Corr')['Feature'].tolist()
        dyn_tbl = corr_multi[['Feature','Horizon (days)','Corr','CI_low','CI_high']] if isinstance(corr_multi, pd.DataFrame) and not corr_multi.empty else pd.DataFrame()
        _plot_forest_dynamic(axes3[1], coef_tbl, dyn_tbl, f'(b) Δ-features vs. Volatility: Clustered correlation', order=order)
    else:
        axes3[1].text(0.5,0.5,'Not Available',ha='center',va='center',fontsize=9,color='gray'); axes3[1].set_title('(b) Δ-features vs. Volatility: Clustered correlation', fontsize=9, loc='left')

    fig3.tight_layout()
    _save_plot(fig3, "fig3_feature_importance", output_dir); plt.close(fig3)

    # ---------------- (Figure 4) ----------------
    create_fig4_combined(robustness_results, robustness_delta_results, output_dir)
    
# ------------------------------ Fig.5: Benchmarks + Sensitivity + Masking ------------------------------
def _ewma_next_variance(stock_ret: pd.Series, lam=0.94, min_init=30):
    """EWMA next-step variance proxy; used to build h-day log-RV forecast."""
    r2 = stock_ret.dropna()**2
    if len(r2) < min_init+5: return pd.Series(index=stock_ret.index, dtype=float)
    s2 = np.empty(len(r2)); s2[:] = np.nan
    s2[min_init-1] = r2.iloc[:min_init].mean()
    for t in range(min_init, len(r2)):
        s2[t] = lam * s2[t-1] + (1-lam) * r2.iloc[t-1]
    out = pd.Series(s2, index=r2.index)
    return out.reindex(stock_ret.index)

def _har_direct_forecast(stock_ret: pd.Series, event_date: pd.Timestamp, h: int, window=252):
    """HAR-RV direct forecast using log of daily/weekly/monthly realized vol levels."""
    if sm is None: return np.nan
    r = stock_ret.dropna()
    if r.empty: return np.nan
    rv = (r**2).rename('rv')
    end = r.index.searchsorted(event_date) - 1
    if end <= 22: return np.nan
    start = max(0, end - window)
    rv_tr = rv.iloc[start:end+1]
    idx = rv_tr.index
    rv_d = rv_tr.copy()
    rv_w = rv_tr.rolling(5, min_periods=5).mean()
    rv_m = rv_tr.rolling(22, min_periods=22).mean()
    X = pd.DataFrame({'d': np.log(rv_d+EPS), 'w': np.log(rv_w+EPS), 'm': np.log(rv_m+EPS)}, index=idx).dropna()
    y = []
    for t in range(len(rv)-1):
        if rv.index[t] not in X.index: continue
        t_start = t+1; t_end = t+h
        if t_end >= len(rv): break
        y.append((rv.index[t], np.log(rv.iloc[t_start:t_end+1].sum() + EPS)))
    if not y: return np.nan
    y = pd.Series(dict(y)).reindex(X.index).dropna()
    X_tr = X.loc[X.index <= rv.index[end]].dropna()
    y_tr = y.loc[y.index.isin(X_tr.index)]
    if len(y_tr) < 30 or X_tr.isna().any().any(): return np.nan
    try:
        X1 = sm.add_constant(X_tr[['d','w','m']])
        model = sm.OLS(y_tr.values, X1.values).fit()
        x_end = X_tr[['d','w','m']].iloc[-1]; x_end = sm.add_constant(x_end, has_constant='add').values
        return float(model.predict(x_end).item())
    except Exception:
        return np.nan

def _compute_benchmark_r2_across_windows(final_df: pd.DataFrame, financial_data, method: str, target_kind: str):
    """Compute OOS R^2 for simple benchmarks (EWMA/HAR) for log-RV."""
    assert method in ('ewma','har'); assert target_kind in ('log_rv','car')
    data = add_quarter(final_df.copy())
    horizons = TARGET_HORIZONS_VOL if target_kind=='log_rv' else TARGET_HORIZONS_CAR
    targets  = [f'log_rv_t+{h}d' if target_kind=='log_rv' else f'car_t+{h}d' for h in horizons]
    prices = financial_data['Close']
    res = {}
    if target_kind == 'car':
        for h in horizons: res[h] = np.nan
        return res
    pooled_true = {h: [] for h in horizons}; pooled_pred = {h: [] for h in horizons}
    for _, df_tr, df_val, df_te in _rolling_quarter_windows(data.dropna(subset=targets, how='any')):
        for _, row in df_te.iterrows():
            ticker = row['ticker']; event_time = pd.to_datetime(row['event_time']).normalize()
            if ticker not in prices.columns: continue
            stock_ret = prices[ticker].pct_change()
            for h in horizons:
                tcol = f'log_rv_t+{h}d'; y_true = row.get(tcol, np.nan)
                if not np.isfinite(y_true): continue
                if method == 'ewma':
                    s2 = _ewma_next_variance(stock_ret, lam=0.94, min_init=30)
                    idx = s2.index.searchsorted(event_time) - 1
                    if idx < 1: continue
                    sigma2 = s2.iloc[idx]
                    y_hat = np.log(h * float(sigma2) + EPS) if np.isfinite(sigma2) else np.nan
                else:
                    y_hat = _har_direct_forecast(stock_ret, event_time, h, window=ESTIMATION_WINDOW)
                if np.isfinite(y_hat):
                    pooled_true[h].append(y_true); pooled_pred[h].append(y_hat)
    for h in horizons:
        if pooled_true[h]:
            res[h] = _r2(np.array(pooled_true[h]), np.array(pooled_pred[h]))
        else:
            res[h] = np.nan
    return res

def _compute_multihorizon_perf_for_sets(final_df: pd.DataFrame, set_names: list, target_kind: str):
    """Model curves for multiple feature sets across horizons (used in Fig.5b)."""
    assert target_kind in ('log_rv','car')
    data = add_quarter(final_df.copy())
    feats = _make_feature_sets(data.columns)
    horizons = TARGET_HORIZONS_VOL if target_kind=='log_rv' else TARGET_HORIZONS_CAR
    perf = {name: {h: np.nan for h in horizons} for name in set_names}
    for name in set_names:
        F = feats.get(name, [])
        if not F: continue
        for h in horizons:
            tcol = f'log_rv_t+{h}d' if target_kind=='log_rv' else f'car_t+{h}d'
            y_all, yhat_all = [], []
            for _, df_tr, df_val, df_te in _rolling_quarter_windows(data.dropna(subset=[tcol])):
                if len(df_te) < 20 or len(df_val) < 20 or len(df_tr) < 100: continue
                df_tr = _group_exclusion(df_tr, df_val, df_te)
                m = _train_xgb_with_val(df_tr[F], df_tr[tcol], df_val[F], df_val[tcol])
                yhat_all.append(m.predict(df_te[F])); y_all.append(df_te[tcol].values)
            if y_all:
                perf[name][h] = _r2(np.concatenate(y_all), np.concatenate(yhat_all))
    return perf

def create_fig5_benchmarks_and_sensitivity(final_df: pd.DataFrame,
                                           outcomes: pd.DataFrame,
                                           financial_data,
                                           ablation_results: pd.DataFrame,
                                           sensitivity_results: dict,
                                           df_segments_labeled: pd.DataFrame,
                                           output_dir: str):
    """
    Figure 5:
      (a) VAD ±20% perturbation robustness (distribution of OOS R^2)
      (b) Benchmarks vs Models (volatility; EWMA/HAR vs Textual/Financial+Textual/Multimodal)
      (c) Masking Index by Section (Presentation vs Q&A)
      (d) ΔMI vs Volatility (clustered CI across horizons)
    """
    set_publication_style()
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(mm2inch(180), mm2inch(100)))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.55, hspace=0.60)
    ax_a = fig.add_subplot(gs[0, 0])  # (a) VAD ±20%
    ax_b = fig.add_subplot(gs[0, 1])  # (b) Benchmarks vs Models
    ax_c = fig.add_subplot(gs[1, 0])  # (c) Masking Index by Section
    ax_d = fig.add_subplot(gs[1, 1])  # (d) ΔMI vs Volatility

    # (a) VAD ±20% perturbation
    r2_vol = sensitivity_results.get('vol', ([], pd.DataFrame()))[0] if isinstance(sensitivity_results, dict) else []
    r2_car = sensitivity_results.get('car', ([], pd.DataFrame()))[0] if isinstance(sensitivity_results, dict) else []
    data_box = []
    if r2_vol: data_box.append(pd.DataFrame({'Target': 'log-RV', 'OOS R^2': r2_vol}))
    if r2_car: data_box.append(pd.DataFrame({'Target': 'CAR',    'OOS R^2': r2_car}))
    data_box = pd.concat(data_box, ignore_index=True) if data_box else pd.DataFrame({'Target': [], 'OOS R^2': []})
    if not data_box.empty:
        palette = [TABLEAU_COLORS[0], TABLEAU_COLORS[7]]
        sns.boxplot(x='Target', y='OOS R^2', data=data_box, ax=ax_a, palette=palette)
        sns.stripplot(x='Target', y='OOS R^2', data=data_box, ax=ax_a, size=3, alpha=0.35, jitter=0.2, palette=palette)        
    else:
        ax_a.text(0.5, 0.5, 'Not Available', ha='center', va='center', fontsize=9, color='gray')
    ax_a.set_xlabel('Target'); ax_a.set_ylabel('OOS $R^2$')
    ax_a.set_title('(a) V-A-D ±20% Perturbation', fontsize=9, loc='left')
    ax_a.grid(True, axis='y', linestyle=':')

    # (b) Benchmarks vs Models（Volatility）
    horizons = TARGET_HORIZONS_VOL
    perf_sets = _compute_multihorizon_perf_for_sets(final_df,
                                                    ['Multimodal', 'Textual-Only', 'Financial+Textual'],
                                                    target_kind='log_rv')
    line_ewma = _compute_benchmark_r2_across_windows(final_df, financial_data, method='ewma', target_kind='log_rv')
    line_har  = _compute_benchmark_r2_across_windows(final_df, financial_data, method='har',  target_kind='log_rv')

    for name, style in [('Multimodal','o-'),
                        ('Textual-Only','s-'),
                        ('Financial+Textual','^-')]:
        vals = [perf_sets.get(name, {}).get(h, np.nan) for h in horizons]
        ax_b.plot(horizons, vals, style, label=name.split(' (')[0])
    ax_b.plot(horizons, [line_ewma.get(h, np.nan) for h in horizons], 'D--', label='EWMA 0.94')
    ax_b.plot(horizons, [line_har.get(h, np.nan)  for h in horizons], 'P--', label='HAR-RV')
    ax_b.set_xticks(horizons)
    ax_b.set_xlabel('Prediction Horizon (days)'); ax_b.set_ylabel('Out-of-Sample $R^2$')
    ax_b.set_title('(b) Benchmarks vs Models (Volatility)', fontsize=9, loc='left')
    ax_b.axhline(0, color='black', linewidth=0.6, linestyle='--')
    ax_b.grid(True, linestyle=':')
    ax_b.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=7)

    # (c)+(d) Masking Index
    mi = compute_masking_index_per_call(df_segments_labeled)
    targets = [f'log_rv_t+{h}d' for h in TARGET_HORIZONS_VOL]
    df_mi = mi.merge(outcomes[['uid'] + targets], on='uid', how='inner').dropna(subset=['MI_presentation','MI_q&a'])

    # (c) MI distribution by section
    if df_mi.empty:
        ax_c.text(0.5, 0.5, 'Not Available', ha='center', va='center', fontsize=9, color='gray')
        ax_c.set_title('(c) Masking Index by Section', loc='left')
    else:
        data_long = df_mi[['MI_presentation','MI_q&a']].rename(columns={'MI_presentation':'Presentation','MI_q&a':'Q&A'}).melt(var_name='Section', value_name='MI')
        sns.violinplot(x='Section', y='MI', data=data_long, ax=ax_c, inner=None, cut=0)
        sns.stripplot(x='Section', y='MI', data=data_long, ax=ax_c, size=2, alpha=0.25, jitter=0.2)
        ax_c.set_title('(c) Masking Index by Section', loc='left')
        ax_c.set_ylabel('Masking Index (share of segments)'); ax_c.set_xlabel('')
        m1, m2 = df_mi['MI_presentation'].mean(), df_mi['MI_q&a'].mean()
        ax_c.text(0.5, 0.95, f'Δ mean = {m2-m1:+.3f}', transform=ax_c.transAxes, ha='center', va='top')

    # (d) ΔMI vs logRV: clustered correlation across horizons
    rows = []
    if not df_mi.empty:
        dmi = df_mi.copy()
        dmi['Delta_MI'] = dmi['MI_q&a'] - dmi['MI_presentation']
        for h in TARGET_HORIZONS_VOL:
            tcol = f'log_rv_t+{h}d'
            tmp = outcomes[['uid','ticker']].merge(dmi[['uid','Delta_MI', tcol]], on='uid', how='inner').dropna()
            if tmp.empty or tmp['Delta_MI'].nunique() < 3 or tmp[tcol].nunique() < 3:
                rows.append({'Horizon (days)': h, 'Corr': np.nan, 'CI_low': np.nan, 'CI_high': np.nan})
                continue
            r = float(np.corrcoef(tmp['Delta_MI'], tmp[tcol])[0, 1])
            lo, hi = _cluster_bootstrap_ci_corr(tmp[tcol].values, tmp['Delta_MI'].values, tmp['ticker'].values)
            rows.append({'Horizon (days)': h, 'Corr': r, 'CI_low': lo, 'CI_high': hi})

    tbl = pd.DataFrame(rows) if rows else pd.DataFrame(columns=['Horizon (days)','Corr','CI_low','CI_high'])
    if not tbl.empty:
        try:
            os.makedirs(TABLE_DIR, exist_ok=True)
            tbl.to_csv(os.path.join(TABLE_DIR, 'table_masking_index_corr.csv'), index=False)
        except Exception:
            pass

    if tbl.dropna(subset=['Corr']).empty:
        ax_d.text(0.5, 0.5, 'Not Available', ha='center', va='center', fontsize=9, color='gray')
        ax_d.set_title('(d) ΔMI vs. Volatility: Clustered correlation', loc='left')
    else:
        d = tbl.dropna(subset=['Corr']).copy()
        y = np.arange(len(d))
        xerr = _safe_err(d['Corr'], d['CI_low'], d['CI_high'])
        ax_d.errorbar(d['Corr'], y, xerr=xerr, fmt='o', color=PRIMARY_COLOR, capsize=2.5, elinewidth=0.9)
        ax_d.set_yticks(y, [f'{int(h)}d' for h in d['Horizon (days)']])
        ax_d.axvline(0, color='black', linewidth=0.6, linestyle='--')
        ax_d.grid(axis='x', linestyle=':')
        ax_d.set_xlabel('Correlation (ticker-cluster CI)')
        ax_d.set_title('(d) ΔMI vs. Volatility: Clustered correlation', loc='left')

    _save_plot(fig, "fig5_benchmarks_and_sensitivity", output_dir)
    plt.close(fig)

# ------------------------------ Sensitivity → Table ------------------------------
def merge_sensitivity_into_table1(ablation_df: pd.DataFrame, sensitivity: dict, table_dir: str) -> pd.DataFrame:
    """Append sensitivity stats (VAD perturbation, best prototype K) to Table 1."""
    try:
        vad_r2, k_df = sensitivity.get('vol', ([], pd.DataFrame()))
        vad_median = float(np.median(vad_r2)) if isinstance(vad_r2, (list, np.ndarray)) and len(vad_r2) else np.nan
        vad_q75 = float(np.percentile(vad_r2, 75)) if isinstance(vad_r2, (list, np.ndarray)) and len(vad_r2) else np.nan
        vad_q25 = float(np.percentile(vad_r2, 25)) if isinstance(vad_r2, (list, np.ndarray)) and len(vad_r2) else np.nan
        vad_iqr = vad_q75 - vad_q25 if np.isfinite(vad_q75) and np.isfinite(vad_q25) else np.nan
        if isinstance(k_df, pd.DataFrame) and not k_df.empty:
            k_best_row = k_df.sort_values('R-squared', ascending=False).head(1)
            k_best = int(k_best_row['K'].iloc[0]); k_best_r2 = float(k_best_row['R-squared'].iloc[0])
        else:
            k_best = np.nan; k_best_r2 = np.nan
        out = ablation_df.copy()
        out['V-A-D Perturbation R2 (median)'] = vad_median
        out['V-A-D Perturbation R2 (IQR)'] = vad_iqr
        out['Prototype K*'] = k_best
        out['R2 @ K*'] = k_best_r2
        os.makedirs(table_dir, exist_ok=True)
        out.to_csv(os.path.join(table_dir, 'table1_performance_summary_with_sensitivity.csv'))
        if isinstance(vad_r2, (list, np.ndarray)): pd.DataFrame({'oos_r2': vad_r2}).to_csv(os.path.join(table_dir, 'table_sensitivity_vad_perturbation.csv'), index=False)
        if isinstance(k_df, pd.DataFrame): k_df.to_csv(os.path.join(table_dir, 'table_sensitivity_prototypeK.csv'), index=False)
        return out
    except Exception as e:
        print('[merge_sensitivity_into_table1] failed:', e)
        return ablation_df

# ------------------------------ DM (HAC) & MCS ------------------------------
def _dm_grouped_ttest(lossA: np.ndarray, lossB: np.ndarray, groups: np.ndarray):
    """Grouped t-test over firm-averaged loss differences (MCS helper)."""
    df = pd.DataFrame({'g': groups, 'd': lossA - lossB}).dropna()
    if df.empty: return np.nan, np.nan, np.nan
    grp = df.groupby('g')['d'].mean(); m = grp.mean(); s = grp.std(ddof=1); n = grp.size
    if n < 5 or s == 0: return float(m), np.nan, np.nan
    t = float(m / (s / np.sqrt(n)))
    try:
        from scipy.stats import t as tdist
        p = float(2*(1 - tdist.cdf(abs(t), df=n-1)))
    except Exception:
        p = np.nan
    return float(m), float(t), float(p)

def _dm_test_hac(lossA: np.ndarray, lossB: np.ndarray, dates: np.ndarray, h: int = 1, alpha: float = 0.05):
    """
    Diebold–Mariano test with Newey–West/HAC variance on the loss differential d_t = lA - lB.
    Returns (mean_diff, t_stat, p_value, ci_lower, ci_upper). Truncation lag m=h-1.
    """
    import numpy as np
    from math import sqrt
    try:
        import pandas as pd
    except Exception:
        pd = None
    try:
        from scipy.stats import t as tdist
        from scipy.stats import norm
    except Exception:
        tdist = None
        norm = None

    d = np.asarray(lossA - lossB, dtype=float)
    n = int(len(d))
    if n < 10 or not np.isfinite(d).all():
        return float(np.nan), float(np.nan), float(np.nan), float(np.nan), float(np.nan)

    # Keep original order; sort by dates if sortable
    if pd is not None:
        try:
            order = np.argsort(pd.to_datetime(dates).values)
            d = d[order]
        except Exception:
            pass

    mean_d = float(np.mean(d))
    # Newey–West/HAC with truncation lag m = h-1
    bw = max(1, int(h) - 1)
    gamma0 = np.sum((d - mean_d)**2) / n
    s = gamma0
    for lag in range(1, bw+1):
        w = 1.0 - lag/(bw+1)
        cov = np.sum((d[lag:] - mean_d) * (d[:-lag] - mean_d)) / n
        s += 2.0 * w * cov
    var_hat = s / n
    if var_hat <= 0:
        return mean_d, float('nan'), float('nan'), float('nan'), float('nan')

    se = sqrt(var_hat)
    t_stat = mean_d / se

    if tdist is not None:
        p_val = float(2*(1 - tdist.cdf(abs(t_stat), df=n-1)))
        critical_value = tdist.ppf(1.0 - alpha / 2.0, df=n - 1)
    elif norm is not None:
        p_val = float(2.0 * (1.0 - norm.cdf(abs(t_stat))))
        critical_value = norm.ppf(1.0 - alpha / 2.0)
    else: # fallback
        p_val = float('nan')
        critical_value = 1.96

    ci_lower = mean_d - critical_value * se
    ci_upper = mean_d + critical_value * se

    return mean_d, float(t_stat), p_val, float(ci_lower), float(ci_upper)

def _collect_predictions_all_models(data: pd.DataFrame,
                                    feats: dict,
                                    models_to_run: list,
                                    tcol: str):
    """
    Collect aligned predictions for all models on a single target column.
    Returns: dict[name] -> {'uid','y_true','y_pred','groups','dates'} (all np.ndarray)
    """
    predictions = {name: {'uid': [], 'y_true': [], 'y_pred': [], 'groups': [], 'dates': []}
                   for name in models_to_run}
    for q_test, df_tr, df_val, df_te in _rolling_quarter_windows(data.dropna(subset=[tcol])):
        if len(df_te) < 20 or len(df_val) < 20 or len(df_tr) < 100:
            continue
        df_tr_clean = _group_exclusion(df_tr, df_val, df_te)
        for name in models_to_run:
            F = feats.get(name, [])
            if not F:
                continue
            try:
                model = _train_xgb_with_val(df_tr_clean[F], df_tr_clean[tcol], df_val[F], df_val[tcol])
                y_hat = model.predict(df_te[F])
                predictions[name]['uid'].append(df_te['uid'].values)
                predictions[name]['y_true'].append(df_te[tcol].values)
                predictions[name]['y_pred'].append(y_hat)
                predictions[name]['groups'].append(df_te['ticker'].values)
                predictions[name]['dates'].append(df_te['event_time'].values)
            except Exception as e:
                print(f"[DM Collector] Skipped {name} @ {q_test} for {tcol}: {e}")
                continue
    # concat
    for name in list(predictions.keys()):
        if predictions[name]['y_true']:
            for k in ['uid','y_true','y_pred','groups','dates']:
                predictions[name][k] = np.concatenate(predictions[name][k])
        else:
            predictions[name] = None
    return predictions


def run_dm_and_mcs(final_df: pd.DataFrame,
                   table_dir: str,
                   horizons: list = (1, 5, 10, 20, 30),
                   targets: tuple = ('log_rv', 'car')):
    """
    Extended DM (Diebold–Mariano) test across multiple horizons and targets.
    For each target in {'log_rv','car'} and each h in horizons, we compare each model
    against the Financial-Only baseline using squared-error loss and HAC variance
    with truncation lag m = h - 1 (as in Methods §4.7 and Table 1 notes). Outputs
    a wide CSV with p-values for direct merging into Table 1.
    Returns: (dm_table_all, mcs_table_10d_logrv)
    """
    os.makedirs(table_dir, exist_ok=True)
    data_all = add_quarter(final_df.copy())
    feats = _make_feature_sets(data_all.columns)

    models_to_run = [
        'Financial-Only',
        'Textual-Only',
        'Financial+Textual',
        'Acoustic-Only',
        'Multimodal'
    ]
    baseline = 'Financial-Only'

    dm_rows = []
    # --- loop over targets and horizons ---
    for tgt in targets:
        for h in horizons:
            tcol = f'{"log_rv" if tgt=="log_rv" else "car"}_t+{h}d'
            if tcol not in data_all.columns:
                continue
            data = data_all.dropna(subset=[tcol]).copy()
            if data.empty:
                continue
            preds = _collect_predictions_all_models(data, feats, models_to_run, tcol)
            if preds.get(baseline) is None:
                print(f"[DM] Missing baseline predictions for {tgt}@{h}d; skipping.")
                continue

            # build baseline frame for alignment by uid
            base_df = pd.DataFrame({
                'uid':   preds[baseline]['uid'],
                'y':     preds[baseline]['y_true'],
                'yhat0': preds[baseline]['y_pred'],
                'g':     preds[baseline]['groups'],
                'date':  preds[baseline]['dates'],
            })

            for name in models_to_run:
                if name == baseline:
                    continue
                if preds.get(name) is None:
                    continue

                mod_df = pd.DataFrame({
                    'uid':   preds[name]['uid'],
                    'y_m':   preds[name]['y_true'],
                    'yhat1': preds[name]['y_pred'],
                    'g_m':   preds[name]['groups'],
                    'date_m':preds[name]['dates'],
                })
                # inner-join by uid to guarantee alignment
                join = base_df.merge(mod_df, on='uid', how='inner')
                if len(join) < 30:
                    continue
                # prefer baseline y (should equal y_m)
                loss0 = (join['y'] - join['yhat0'])**2
                loss1 = (join['y'] - join['yhat1'])**2
                diff, t, p, lo, hi = _dm_test_hac(loss1.values, loss0.values,
                                                  dates=join['date'].values, h=int(h))
                dm_rows.append({
                    'Target': ('log_RV' if tgt=='log_rv' else 'CAR'),
                    'Horizon (days)': int(h),
                    'Model': name,
                    'Comparison vs Financial-Only': f'{name} - {baseline}',
                    'Mean loss diff': float(diff),
                    't-stat': float(t) if np.isfinite(t) else np.nan,
                    'p-value': float(p) if np.isfinite(p) else np.nan,
                    '95% CI (lower)': float(lo) if np.isfinite(lo) else np.nan,
                    '95% CI (upper)': float(hi) if np.isfinite(hi) else np.nan,
                    'N': int(len(join))
                })

    dm_table_all = pd.DataFrame(dm_rows).sort_values(['Target','Horizon (days)','Model'])
    try:
        dm_table_all.to_csv(os.path.join(table_dir, 'table_dm_test_extended.csv'), index=False)
        if not dm_table_all.empty:
            print('\n[Extended DM test across horizons]\n',
                  dm_table_all.head(20).to_string(index=False, float_format="%.4f"))
    except Exception as e:
        print('[DM] write failed:', e)

    # --- MCS on 10d log_RV (unchanged default behavior) ---
    mcs_table = pd.DataFrame()
    try:
        tcol10 = f'log_rv_t+{PRIMARY_HORIZON}d'
        data10 = data_all.dropna(subset=[tcol10]).copy()
        preds10 = _collect_predictions_all_models(data10, feats, models_to_run, tcol10)
        perfs = {}
        for name, pred in preds10.items():
            if pred is None:
                continue
            y_true = pred['y_true']; y_pred = pred['y_pred']; groups = pred['groups']
            perfs[name] = {'mse': float(np.mean((y_true - y_pred)**2)),
                           'groups': groups, 'err': y_true - y_pred}
        mcs_set = set(perfs.keys()); alpha = 0.05
        while len(mcs_set) > 1:
            cur = {k:v for k,v in perfs.items() if k in mcs_set}
            worst = max(cur.items(), key=lambda kv: kv[1]['mse'])[0]
            best  = min(cur.items(), key=lambda kv: kv[1]['mse'])[0]
            e_best = cur[best]['err']; e_worst = cur[worst]['err']; g = cur[best]['groups']
            _, t, p = _dm_grouped_ttest(e_worst**2, e_best**2, g)
            if np.isnan(p) or p > alpha:
                break
            mcs_set.remove(worst)
        mcs_table = pd.DataFrame({'Model in MCS (alpha=0.05)': sorted(list(mcs_set))})
        mcs_table.to_csv(os.path.join(table_dir, 'table_mcs.csv'), index=False)
        if not mcs_table.empty:
            print('\n[MCS set @ 10d log-RV]\n', mcs_table)
    except Exception as e:
        print('[MCS] failed:', e)

    return dm_table_all, mcs_table 

# ------------------------------ Absorb (primary endpoint) ------------------------------
def _absorb_from_dr2(dr2_map: dict) -> float:
    """Absorb = 0.5*(ΔR2_10 + ΔR2_15) − ΔR2_1."""
    if 1 not in dr2_map or 10 not in dr2_map or 15 not in dr2_map:
        return float('nan')
    return 0.5*(dr2_map[10] + dr2_map[15]) - dr2_map[1]

def _delta_r2_from_arrays(y, yb, yf) -> float:
    return _r2(y, yf) - _r2(y, yb)

def _collect_predictions_by_horizon(final_df: pd.DataFrame,
                                    base_set: str,
                                    full_set: str,
                                    horizons=(1, 10, 15)) -> dict:
    """
    Collect pooled OOS predictions for (base, full) across requested horizons.
    Returns: {h: {'y':..., 'yb':..., 'yf':..., 'g':...}}
    """
    data = add_quarter(final_df.copy())
    feats = _make_feature_sets(data.columns)
    F_base = feats.get(base_set, [])
    F_full = feats.get(full_set, [])
    out = {}
    for h in horizons:
        tcol = f'log_rv_t+{h}d'
        d = data.dropna(subset=[tcol])
        y_list, yb_list, yf_list, g_list = [], [], [], []
        for _, df_tr, df_val, df_te in _rolling_quarter_windows(d):
            if len(df_te) < 20 or len(df_val) < 20 or len(df_tr) < 100: continue
            df_tr = _group_exclusion(df_tr, df_val, df_te)
            # Base model
            try:
                if F_base:
                    mb = _train_xgb_with_val(df_tr[F_base], df_tr[tcol], df_val[F_base], df_val[tcol])
                    yb = mb.predict(df_te[F_base])
                else:
                    yb = np.repeat(np.nanmean(df_tr[tcol].values), repeats=len(df_te))
            except Exception:
                yb = np.repeat(np.nanmean(df_tr[tcol].values), repeats=len(df_te))
            # Full model
            if not F_full: 
                continue
            try:
                mf = _train_xgb_with_val(df_tr[F_full], df_tr[tcol], df_val[F_full], df_val[tcol])
                yf = mf.predict(df_te[F_full])
            except Exception:
                continue
            y = df_te[tcol].values; g = df_te['ticker'].values
            y_list.append(y); yb_list.append(yb); yf_list.append(yf); g_list.append(g)
        if y_list:
            out[h] = {'y': np.concatenate(y_list), 'yb': np.concatenate(yb_list),
                      'yf': np.concatenate(yf_list), 'g': np.concatenate(g_list)}
    return out

def compute_absorb_and_write_inline(final_df: pd.DataFrame, outdir_tables: str,
                                    base_set='Financial-Only', full_set='Multimodal',
                                    horizons=(1,10,15,20,25,30), alpha=0.05,
                                    B=N_BOOTSTRAPS_ABSORB, P=N_PERMUTATIONS_ABSORB, seed=GLOBAL_SEED):
    """Compute Absorb with clustered bootstrap CI and permutation p-value; export tex/json snippets."""
    os.makedirs(outdir_tables, exist_ok=True)
    preds = _collect_predictions_by_horizon(final_df, base_set, full_set, horizons=horizons)
    if not set(horizons).issubset(preds.keys()):
        print(r'\emph{Absorb could not be computed (insufficient predictions).}')
        return

    # point estimate
    dr2 = {h: _delta_r2_from_arrays(preds[h]['y'], preds[h]['yb'], preds[h]['yf']) for h in horizons}
    absorb_hat = _absorb_from_dr2(dr2)

    # clustered bootstrap over groups (ticker as firm proxy)
    rng = np.random.default_rng(seed)
    groups_union = np.unique(np.concatenate([preds[h]['g'] for h in horizons]))
    boot_vals = []
    for _ in range(B):
        gs = rng.choice(groups_union, size=len(groups_union), replace=True)
        dr2_b = {}
        ok = True
        for h in horizons:
            mask = np.isin(preds[h]['g'], gs)
            y, yb, yf = preds[h]['y'][mask], preds[h]['yb'][mask], preds[h]['yf'][mask]
            if len(y) < 30 or np.unique(y).size < 2:
                ok = False; break
            dr2_b[h] = _delta_r2_from_arrays(y, yb, yf)
        if ok and all(k in dr2_b for k in horizons):
            boot_vals.append(_absorb_from_dr2(dr2_b))
    if boot_vals:
        pl, pu = _ci_percentiles(alpha); lo = float(np.percentile(boot_vals, pl)); hi = float(np.percentile(boot_vals, pu))
    else:
        lo = hi = float('nan')

    # permutation test (shuffle full-model predictions within ticker clusters)
    perm = []
    for b in range(P):
        dr2_p = {}
        ok = True
        for h in horizons:
            y  = preds[h]['y']; yb = preds[h]['yb']; yf = preds[h]['yf'].copy(); g = preds[h]['g']
            for tk in np.unique(g):
                idx = np.where(g == tk)[0]
                if len(idx) > 1:
                    rng.shuffle(yf[idx])
            if np.unique(y).size < 2:
                ok = False; break
            dr2_p[h] = _delta_r2_from_arrays(y, yb, yf)
        if ok:
            perm.append(_absorb_from_dr2(dr2_p))
    if perm:
        pval = float((np.sum(np.abs(perm) >= abs(absorb_hat)) + 1.0) / (len(perm) + 1.0))
    else:
        pval = float('nan')

    # write inline artifacts
    try:
        import json
        with open(os.path.join(outdir_tables, 'absorb_values.json'), 'w', encoding='utf-8') as f:
            json.dump({'Absorb': absorb_hat, 'ci_lo': lo, 'ci_hi': hi, 'p_perm': pval,
                       'B': B, 'P': P, 'horizons': list(horizons)}, f, indent=2)
        print(f"Absorb = {absorb_hat:+.3f} (95% CI [{lo:+.3f}, {hi:+.3f}]; perm. p = {pval:.3f})")
    except Exception:
        pass

# ------------------------------ Main ------------------------------
def main():
    set_publication_style()
    print("="*64 + "\nEmoVoice Risk-Absorption Narrative; PRIMARY_HORIZON=10 \n" + "="*64)
    os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True); os.makedirs(TABLE_DIR, exist_ok=True)

    # Load segment-level data (or create a demo file if missing)
    try:
        df_raw = pd.read_csv(INPUT_CSV, encoding='utf-8-sig', low_memory=False)
        print(f"Loaded {len(df_raw)} segments.")
    except FileNotFoundError:
        print(f"Input not found: {INPUT_CSV}. Creating a dummy file for demonstration.")
        os.makedirs(DATA_DIR, exist_ok=True)
        dummy_data = {
            'uid': [f'call_{i}' for i in range(50) for _ in range(10)],
            'ticker': [f'TICK{i%5}' for i in range(50) for _ in range(10)],
            'event_time': [pd.to_datetime('2021-01-01') + pd.Timedelta(days=i*10) for i in range(50) for _ in range(10)],
            'speaker_role': random.choices(['CEO', 'CFO', 'CXO', 'Analyst', 'Operator'], k=500),
            'transcripts': ['hello world this is a test? what is the q&a session' for _ in range(500)],
            'textual_emotion': random.choices(['neutral', 'happiness', 'sadness', 'anger','disgust','surprise','fear'], k=500),
            'acoustic_emotion': random.choices(['neutral', 'happiness', 'sadness', 'anger','disgust','surprise','fear'], k=500)
        }
        pd.DataFrame(dummy_data).to_csv(INPUT_CSV, index=False)
        df_raw = pd.read_csv(INPUT_CSV, encoding='utf-8-sig', low_memory=False)

    # Fig.1
    create_descriptive_plots(df_raw, FIGURE_DIR)

    # Feature engineering
    print("\n--- Feature Engineering ---")
    df_raw['event_time'] = pd.to_datetime(df_raw['event_time'])
    df_labeled = label_call_sections(df_raw)
    call_level_features = engineer_vad_features(df_labeled, vad_map=VAD_MAP)

    # Events universe
    unique_events = call_level_features[['uid','ticker','event_time']].drop_duplicates()
    tickers = unique_events['ticker'].dropna().unique().tolist()
    if not tickers:
        print("No tickers in data."); return

    min_date = unique_events['event_time'].min().normalize() - pd.Timedelta(days=365)
    max_date = unique_events['event_time'].max().normalize() + pd.Timedelta(days=90)
    max_date = min(max_date, pd.Timestamp.today().normalize() + pd.Timedelta(days=3))

    fin_cache = os.path.join(CACHE_DIR, 'financial_data_master.pkl')
    fin = get_financial_data(tickers, min_date, max_date, fin_cache)
    if fin is None:
        print("Failed to download market data."); return

    ind_cache = os.path.join(CACHE_DIR, 'industry_data.pkl')
    industry_map = get_industry_data(tickers, ind_cache)
    industry_df = pd.DataFrame(industry_map.items(), columns=['ticker', 'industry'])

    # Outcomes (CAR / RV / log_RV)
    outcomes = calculate_market_outcomes(unique_events, fin)
    if outcomes.empty:
        print("No outcomes computed."); return

    # Merge to final call-level table
    final_df = call_level_features.merge(
        outcomes.drop(columns=['ticker','event_time','event_date_normalized'], errors='ignore'),
        on='uid', how='inner'
    ).merge(industry_df, on='ticker', how='left')
    final_df.to_csv(os.path.join(OUTPUT_DIR, 'final_engineered_features.csv'), index=False)

    # Orthogonalize acoustic vs Financial+Textual (optional, improves interpretability)
    if CFG.ORTHOGONALIZE_ACOUSTIC:
        feats_tmp = _make_feature_sets(final_df.columns)
        base_cols = feats_tmp.get('Financial+Textual', [])
        final_df = orthogonalize_acoustic_features(final_df, base_cols)
        final_df.to_csv(os.path.join(OUTPUT_DIR, 'final_engineered_features_orth.csv'), index=False)

    # Table 1 (+ multi-horizon variants)
    ablation_results = run_model_comparison(final_df, TABLE_DIR)
    run_model_comparison_multihorizon(final_df, 'log_rv', TABLE_DIR)
    run_model_comparison_multihorizon(final_df, 'car',     TABLE_DIR)

    # DM / MCS
    if CFG.RUN_DM_AND_MCS:
        dm_tbl, mcs_tbl = run_dm_and_mcs(final_df, TABLE_DIR, horizons=[1,5,10,20,30], targets=('log_rv','car'))
        if isinstance(dm_tbl, pd.DataFrame) and not dm_tbl.empty: print('\n[DM test]\n', dm_tbl)
        if isinstance(mcs_tbl, pd.DataFrame) and not mcs_tbl.empty: print('\n[MCS set]\n', mcs_tbl)

    # Figures 2/3/4
    perf_multi, importance_stats, delta_curves = run_final_model_analysis(final_df)
    perf_tuple = (perf_multi, importance_stats, delta_curves)
    robustness_abs   = run_robustness_analysis_fig4(final_df)
    robustness_delta = run_robustness_analysis_fig4_delta(final_df)
    final_df_with_quarter = add_quarter(final_df.copy())
    create_final_plots(perf_tuple, ablation_results, robustness_abs, robustness_delta, final_df_with_quarter, FIGURE_DIR)

    # Sensitivity (VAD ±20%, Prototype K) + Fig.5 (benchmarks + MI)
    sensitivity = run_sensitivity_analysis_fig5(df_labeled, outcomes, call_level_features)
    merge_sensitivity_into_table1(ablation_results, sensitivity, TABLE_DIR)
    create_fig5_benchmarks_and_sensitivity(final_df_with_quarter, outcomes, fin, ablation_results, sensitivity, df_labeled, FIGURE_DIR)

    # Primary endpoint: Absorb (focus on 1,10,15 days)
    compute_absorb_and_write_inline(final_df_with_quarter, TABLE_DIR)

    print("\n--- Done ---")

# ------------------------------ Orthogonalization ------------------------------
def orthogonalize_acoustic_features(df: pd.DataFrame, base_cols: list) -> pd.DataFrame:
    """
    Regress each acoustic feature on Financial+Textual features and keep residuals as orth_*
    to reduce collinearity and improve interpretability of acoustic contributions.
    """
    if not isinstance(base_cols, (list, tuple)): return df
    base_cols = [c for c in base_cols if c in df.columns]
    ac_cols = [c for c in df.columns if 'acoustic_' in c]
    if not ac_cols or not base_cols: return df
    df = df.copy()
    Xb = df[base_cols].astype(float); base_ok = ~Xb.isna().any(axis=1)
    try:
        from sklearn.linear_model import LinearRegression
        _use_sklearn = True
    except Exception:
        _use_sklearn = False
        import numpy as _np
    for col in ac_cols:
        y = pd.to_numeric(df[col], errors='coerce')
        valid = base_ok & y.notna()
        if int(valid.sum()) < 5 or float(np.nanstd(y.loc[valid].values)) < 1e-8: continue
        try:
            if _use_sklearn:
                lr = LinearRegression(); lr.fit(Xb.loc[valid].values, y.loc[valid].values)
                pred = lr.predict(Xb.loc[valid].values)
            else:
                X = _np.c_[_np.ones(int(valid.sum())), Xb.loc[valid].values]
                beta, *_ = _np.linalg.lstsq(X, y.loc[valid].values, rcond=None)
                pred = X @ beta
            resid = pd.Series(np.nan, index=df.index, dtype=float); resid.loc[valid] = y.loc[valid].values - pred
            df[f'orth_{col}'] = resid.values
        except Exception as e:
            print(f"[orthogonalize] skip {col}: {e}")
    return df

# ------------------------------ Sensitivity (helper) ------------------------------
def _build_segment_level_acoustic(df_segments: pd.DataFrame, vad_map: dict) -> pd.DataFrame:
    """Map segment emotions to VAD coordinates for prototype-based sensitivity analysis."""
    df = df_segments[['uid', 'ticker', 'event_time', 'speaker_role', 'acoustic_emotion']].copy()
    for i, dim in enumerate(['valence', 'arousal', 'dominance']):
        df[f'ac_{dim}'] = df['acoustic_emotion'].map(lambda x: vad_map.get(x, (0.0, 0.0, 0.0))[i])
    return df

def _kproto_histograms_per_call(kmeans, seg_df: pd.DataFrame, uids: np.ndarray, K: int) -> pd.DataFrame:
    """Histogram of prototype assignments per call, normalized to shares."""
    sub = seg_df[seg_df['uid'].isin(uids)].copy()
    if sub.empty:
        return pd.DataFrame({'uid': [], **{f'proto_{k}': [] for k in range(K)}})
    X = sub[['ac_valence', 'ac_dominance', 'ac_arousal']].values
    labels = kmeans.predict(X)
    hist = pd.DataFrame({'uid': sub['uid'].values, 'proto': labels}).groupby(['uid','proto']).size().unstack(fill_value=0)
    for k in range(K):
        if k not in hist.columns: hist[k] = 0
    hist = hist.sort_index(axis=1).div(hist.sum(axis=1).replace(0, 1.0), axis=0)
    hist.columns = [f'proto_{c}' for c in hist.columns]
    return hist.reset_index()

def run_sensitivity_analysis_fig5(df_segments_labeled: pd.DataFrame, events_with_outcomes: pd.DataFrame, final_call_level: pd.DataFrame):
    """
    Sensitivity diagnostics for Fig.5:
      - (a) V-A-D ±20% perturbation effect on OOS R^2 distributions
      - (b) Prototype-K stability across K (16..512)
    Returns: dict with keys 'vol' and 'car', each (list_of_R2_under_VAD_perturbation, DataFrame[K, R2])
    """
    print("\n--- Sensitivity Analysis (Fig.5) ---")
    call_df = add_quarter(final_call_level.copy())
    outcome_min = events_with_outcomes[['uid', BASELINE_CONTROL] + [f'log_rv_t+{h}d' for h in TARGET_HORIZONS_VOL] +
                                       [f'rv_t+{h}d' for h in TARGET_HORIZONS_VOL] +
                                       [f'car_t+{h}d' for h in TARGET_HORIZONS_CAR]].copy()
    call_df = call_df.merge(outcome_min, on='uid', how='inner')
    feats = _make_feature_sets(call_df.columns)
    multi = feats.get("Multimodal", [])
    t_vol = f'log_rv_t+{PRIMARY_HORIZON}d'
    t_car = f'car_t+{PRIMARY_HORIZON}d'

    # (a) VAD ±20%
    n_runs = 20; r2_vol, r2_car = [], []
    for _ in tqdm(range(n_runs), desc="V-A-D Perturbation"):
        perturbed_map = {emo: tuple(c * random.uniform(0.8, 1.2) for c in coords) for emo, coords in VAD_MAP.items()}
        call_level_pert = engineer_vad_features(df_segments_labeled, vad_map=perturbed_map)
        call_level_pert = add_quarter(call_level_pert)
        d = call_level_pert.merge(outcome_min, on='uid', how='inner')
        # For vol
        per_y, per_yhat = [], []
        for _, df_tr, df_val, df_te in _rolling_quarter_windows(d.dropna(subset=[t_vol])):
            if len(df_te) < 20 or len(df_val) < 20 or len(df_tr) < 80: continue
            df_tr = _group_exclusion(df_tr, df_val, df_te)
            m = _train_xgb_with_val(df_tr[multi], df_tr[t_vol], df_val[multi], df_val[t_vol])
            per_y.append(df_te[t_vol].values); per_yhat.append(m.predict(df_te[multi]))
        if per_y:
            yy = np.concatenate(per_y); pp = np.concatenate(per_yhat); r2_vol.append(_r2(yy, pp))
        # For CAR
        per_y, per_yhat = [], []
        for _, df_tr, df_val, df_te in _rolling_quarter_windows(d.dropna(subset=[t_car])):
            if len(df_te) < 20 or len(df_val) < 20 or len(df_tr) < 80: continue
            df_tr = _group_exclusion(df_tr, df_val, df_te)
            m = _train_xgb_with_val(df_tr[multi], df_tr[t_car], df_val[multi], df_val[t_car])
            per_y.append(df_te[t_car].values); per_yhat.append(m.predict(df_te[multi]))
        if per_y:
            yy = np.concatenate(per_y); pp = np.concatenate(per_yhat); r2_car.append(_r2(yy, pp))

    # (b) Prototype K
    from sklearn.cluster import KMeans
    seg_ac = _build_segment_level_acoustic(df_segments_labeled, VAD_MAP)
    seg_ac['event_time'] = pd.to_datetime(seg_ac['event_time'])
    call_df_k = add_quarter(call_df.copy())
    K_list = [16, 32, 64, 128, 256, 512]
    k_results_vol, k_results_car = [], []
    for K in K_list:
        # log-RV
        per_y, per_yhat = [], []
        for _, df_tr, df_val, df_te in _rolling_quarter_windows(call_df_k.dropna(subset=[t_vol])):
            if len(df_te) < 20 or len(df_val) < 20 or len(df_tr) < 100: continue
            df_tr = _group_exclusion(df_tr, df_val, df_te)
            u_tr, u_val, u_te = df_tr['uid'].unique(), df_val['uid'].unique(), df_te['uid'].unique()
            seg_tr = seg_ac[seg_ac['uid'].isin(u_tr)]
            if len(seg_tr) < K: continue
            km = KMeans(n_clusters=K, random_state=GLOBAL_SEED, n_init=10)
            km.fit(seg_tr[['ac_valence','ac_dominance','ac_arousal']].values)
            h_tr = _kproto_histograms_per_call(km, seg_ac, u_tr, K)
            h_val = _kproto_histograms_per_call(km, seg_ac, u_val, K)
            h_te  = _kproto_histograms_per_call(km, seg_ac, u_te, K)
            cols = ['uid', t_vol] + ([BASELINE_CONTROL] if BASELINE_CONTROL in call_df_k.columns else [])
            dtr = df_tr[cols].merge(h_tr, on='uid', how='left').fillna(0.0)
            dva = df_val[cols].merge(h_val, on='uid', how='left').fillna(0.0)
            dte = df_te[['uid', t_vol]].merge(h_te,  on='uid', how='left').fillna(0.0)
            common_cols = set(dtr.columns) & set(dva.columns) & set(dte.columns)
            proto_cols  = sorted([c for c in common_cols if c.startswith('proto_')])
            feat_cols = list(proto_cols) + ([BASELINE_CONTROL] if BASELINE_CONTROL in common_cols else [])
            if not feat_cols: continue
            m = _train_xgb_with_val(dtr[feat_cols], dtr[t_vol], dva[feat_cols], dva[t_vol])
            per_y.append(dte[t_vol].values); per_yhat.append(m.predict(dte[feat_cols]))
        if per_y:
            yy = np.concatenate(per_y); pp = np.concatenate(per_yhat)
            k_results_vol.append({'K': K, 'R-squared': _r2(yy, pp)})

        # CAR
        per_y, per_yhat = [], []
        for _, df_tr, df_val, df_te in _rolling_quarter_windows(call_df_k.dropna(subset=[t_car])):
            if len(df_te) < 20 or len(df_val) < 20 or len(df_tr) < 100: continue
            df_tr = _group_exclusion(df_tr, df_val, df_te)
            u_tr, u_val, u_te = df_tr['uid'].unique(), df_val['uid'].unique(), df_te['uid'].unique()
            seg_tr = seg_ac[seg_ac['uid'].isin(u_tr)]
            if len(seg_tr) < K: continue
            km = KMeans(n_clusters=K, random_state=GLOBAL_SEED, n_init=10)
            km.fit(seg_tr[['ac_valence','ac_dominance','ac_arousal']].values)
            h_tr = _kproto_histograms_per_call(km, seg_ac, u_tr, K)
            h_val = _kproto_histograms_per_call(km, seg_ac, u_val, K)
            h_te  = _kproto_histograms_per_call(km, seg_ac, u_te, K)
            cols = ['uid', t_car] + ([BASELINE_CONTROL] if BASELINE_CONTROL in call_df_k.columns else [])
            dtr = df_tr[cols].merge(h_tr, on='uid', how='left').fillna(0.0)
            dva = df_val[cols].merge(h_val, on='uid', how='left').fillna(0.0)
            dte = df_te[['uid', t_car]].merge(h_te,  on='uid', how='left').fillna(0.0)
            common_cols = set(dtr.columns) & set(dva.columns) & set(dte.columns)
            proto_cols  = sorted([c for c in common_cols if c.startswith('proto_')])
            feat_cols = list(proto_cols) + ([BASELINE_CONTROL] if BASELINE_CONTROL in common_cols else [])
            if not feat_cols: continue
            m = _train_xgb_with_val(dtr[feat_cols], dtr[t_car], dva[feat_cols], dva[t_car])
            per_y.append(dte[t_car].values); per_yhat.append(m.predict(dte[feat_cols]))
        if per_y:
            yy = np.concatenate(per_y); pp = np.concatenate(per_yhat)
            k_results_car.append({'K': K, 'R-squared': _r2(yy, pp)})

    return {
        'vol': (r2_vol, pd.DataFrame(k_results_vol) if k_results_vol else pd.DataFrame({'K': [], 'R-squared': []})),
        'car': (r2_car, pd.DataFrame(k_results_car) if k_results_car else pd.DataFrame({'K': [], 'R-squared': []}))
    }

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
