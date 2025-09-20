# -*- coding: utf-8 -*-
"""
EmoVoice Extended Experiments (Financials-Only baseline)
-------------------------------------------------------
This extended script provides the following enhancements without changing the API of the main analysis script:
  1) Use Financials-Only as the baseline to output the relative ΔR² curve and Absorb score for the Multimodal model.
  2) Fix the prediction collection function call to consistently use _collect_predictions_by_horizon from the main script.
  3) Define the Absorb score as per the paper: Absorb = 0.5*(ΔR²_10 + ΔR²_15) − ΔR²_1.
  4) Orthogonalize acoustic features against the Financials-Only baseline.
  5) Provide ΔR² decay (half-life) fitting and clustered bootstrap confidence intervals.
  6) Retain and clean up the QDI (Question Difficulty Index) and Glitch-IV (2SLS) estimations and exports.

Dependencies:
  - Reuses the following from the main analysis script emovoice_segments_analysis.py:
      * label_call_sections / engineer_vad_features
      * get_financial_data / calculate_market_outcomes / get_industry_data
      * _make_feature_sets / orthogonalize_acoustic_features
      * _collect_predictions_by_horizon / _r2
"""
import os
import re
import json
import warnings
import importlib.util
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

# ----------------------------- Paths and Constants -----------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
REF_FILE    = os.path.join(BASE_DIR, 'emovoice_segments_analysis.py')  # Main analysis script

DATA_DIR    = os.path.join('results')
INPUT_CSV   = os.path.join(DATA_DIR, 'emovoice_transcript_emotion_per_segment.csv')

EXT_OUTDIR = os.path.join(DATA_DIR, 'emovoice_extended_experiments')
TAB_DIR     = os.path.join(EXT_OUTDIR, 'tables')
CACHE_DIR   = os.path.join(EXT_OUTDIR, 'cache')

os.makedirs(TAB_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

PRIMARY_HORIZON = 10
HORIZONS = [1, 5, 10, 15, 20, 25, 30]

GLOBAL_SEED = 42
N_BOOTSTRAPS_ABSORB = 500
N_PERMUTATIONS_ABSORB = 500

# Standardized baseline set name
BASELINE_SET_NAME = 'Financials-Only'
FULL_SET_NAME     = 'Multimodal'

# Minimal financial negative word list (for QDI)
_LMD_NEG_MINI = {
    'loss','liability','impairment','uncertain','uncertainty','risk','risks','decline',
    'negative','downturn','default','breach','fraud','lawsuit','penalty','fine',
    'volatility','drawdown','write-down','writeoff','covenant','downgrade','delay',
    'shortage','disruption','headwind','exposure','weakness'
}

# Text patterns for fallback IV if signal quality fields are unavailable
_GLITCH_TEXT_PATTERNS = [
    r'\binaudible\b', r'\bindiscernible\b', r'\btechnical (issue|difficulties|problem)\b',
    r'\b(can you|could you|please) (repeat|say again)\b', r'\bwe (lost|lose) (the )?line\b',
    r'\b(connection|audio) (issue|problem)\b', r'\bbad line\b', r'\bstatic\b',
    r'\bec(h|ho)\b', r'\b(mute|unmute)\b', r'\bdropped\b', r'\breconnect(ing|ed)?\b'
]

# ----------------------------- Dynamically Import Main Analysis Module -----------------------------
def _import_ref_module(path: str):
    """Dynamically imports the main analysis module from a given path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot locate reference file: {path}")
    spec = importlib.util.spec_from_file_location('emo_ref', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

emo = _import_ref_module(REF_FILE)  # Main module (contains training/splitting/evaluation functions)
BASELINE_CONTROL = getattr(emo, 'BASELINE_CONTROL', 'historical_volatility_10d')

# ----------------------------- I/O and Utility Functions -----------------------------
def _safe_read_csv(path: str) -> pd.DataFrame:
    """Reads a CSV file safely, trying utf-8-sig first."""
    try:
        return pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
    except Exception:
        return pd.read_csv(path, low_memory=False)

def _exists(path: str) -> bool:
    """Checks if a file exists and is not empty."""
    return os.path.exists(path) and os.path.getsize(path) > 0

def _zscore(s: pd.Series) -> pd.Series:
    """Computes the z-score of a pandas Series."""
    s = pd.to_numeric(s, errors='coerce')
    mu = s.mean()
    sd = s.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - mu) / sd

def _coalesce_column(df: pd.DataFrame, base_name: str) -> pd.DataFrame:
    """
    Merges duplicate columns (e.g., with suffixes like _x, _y, .1 from merges).
    Prioritizes the left-most column and removes redundant ones.
    """
    pattern = re.compile(rf'^{re.escape(base_name)}($|(_x|_y)|(\.\d+)|(_\d+))')
    cands = [c for c in df.columns if pattern.match(str(c))]
    if not cands:
        df[base_name] = np.nan
        return df
    cands = sorted(cands, key=lambda c: (0 if c == base_name else 1, len(c)))
    tmp = pd.concat([pd.to_numeric(df[c], errors='coerce') for c in cands], axis=1)
    df[base_name] = tmp.bfill(axis=1).iloc[:, 0]
    drop_cols = [c for c in cands if c != base_name]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    return df

# ----------------------------- Assemble or Minimally Rebuild Call-Level Table -----------------------------
def build_or_load_call_level() -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Prioritizes reusing exported files from the main script:
      - results/emovoice_segments_analysis/final_engineered_features_orth.csv
      - results/emovoice_segments_analysis/final_engineered_features.csv
    Also returns the segment-level DataFrame (with section labels) for QDI/Glitch construction.
    """
    outdir_ref = os.path.join('results', 'emovoice_segments_analysis')
    f_orth = os.path.join(outdir_ref, 'final_engineered_features_orth.csv')
    f_raw  = os.path.join(outdir_ref, 'final_engineered_features.csv')

    seg_path = INPUT_CSV
    df_segments = _safe_read_csv(seg_path)
    df_segments['event_time'] = pd.to_datetime(df_segments['event_time'])
    if 'section' not in df_segments.columns:
        df_segments = emo.label_call_sections(df_segments)

    if _exists(f_orth) or _exists(f_raw):
        final_df = _safe_read_csv(f_orth if _exists(f_orth) else f_raw)
        if 'event_time' in final_df.columns:
            final_df['event_time'] = pd.to_datetime(final_df['event_time'])
        meta = {'baseline_col': BASELINE_CONTROL, 'horizons': HORIZONS, 'primary_h': PRIMARY_HORIZON}
        return final_df, df_segments, meta

    # --- Minimal Rebuild (only if upstream artifacts are missing) ---
    print("[Info] Rebuilding call-level features (minimal) ...")
    call_level = emo.engineer_vad_features(df_segments, vad_map=getattr(emo, 'VAD_MAP', {}))
    unique_events = call_level[['uid', 'ticker', 'event_time']].drop_duplicates()

    min_date = unique_events['event_time'].min().normalize() - pd.Timedelta(days=365)
    max_date = unique_events['event_time'].max().normalize() + pd.Timedelta(days=90)
    fin = emo.get_financial_data(unique_events['ticker'].dropna().unique().tolist(),
                                 min_date, max_date,
                                 os.path.join(CACHE_DIR, 'financial_data.pkl'))
    outcomes = emo.calculate_market_outcomes(unique_events, fin)

    industry_map = emo.get_industry_data(unique_events['ticker'].dropna().unique().tolist(),
                                         os.path.join(CACHE_DIR, 'industry_map.pkl'))
    industry_df = pd.DataFrame(industry_map.items(), columns=['ticker', 'industry'])

    final_df = (
        call_level.merge(
            outcomes.drop(columns=['ticker', 'event_time', 'event_date_normalized'], errors='ignore'),
            on='uid', how='inner'
        ).merge(industry_df, on='ticker', how='left')
    )

    # Orthogonalization: Use Financials-Only as the non-acoustic base
    feats_tmp = emo._make_feature_sets(final_df.columns)
    base_cols = feats_tmp.get('Financials-Only', [])
    final_df = emo.orthogonalize_acoustic_features(final_df, base_cols)

    meta = {'baseline_col': BASELINE_CONTROL, 'horizons': HORIZONS, 'primary_h': PRIMARY_HORIZON}
    return final_df, df_segments, meta

# ----------------------------- QDI (Question Difficulty Index) -----------------------------
def _tokenize(s: str) -> List[str]:
    """Simple tokenizer for QDI calculation."""
    if not isinstance(s, str) or not s:
        return []
    s = s.lower()
    s = re.sub(r'[^a-z0-9\?\.\,\:\;\-\%\$ ]+', ' ', s)
    toks = re.split(r'\s+', s.strip())
    return [t for t in toks if t]

def build_qdi(df_segments: pd.DataFrame, neg_words: Optional[set] = None) -> pd.DataFrame:
    """
    Calculates a Question Difficulty Index (QDI) for segments likely to be analyst questions in the Q&A section:
      - neg_density / num_density / avg_length / TTR / question_mark_ratio / clause_punct_ratio
      - The QDI is the mean of the z-scores of these components.
    """
    neg_words = set(neg_words) if neg_words else _LMD_NEG_MINI

    def _is_q_like(text: str) -> bool:
        """Heuristically checks if a text string looks like a question."""
        if not isinstance(text, str): return False
        t = text.strip().lower()
        return ('?' in t) or t.startswith(('what ', 'why ', 'how ', 'when ', 'where ', 'which ', 'could ', 'would ', 'can ', 'may '))

    df = df_segments.copy()
    df = df[df['section'].str.lower().eq('q&a')]
    if 'speaker_role' in df.columns:
        is_analyst = df['speaker_role'].astype(str).str.lower().str.contains('analyst')
        is_q = df['transcripts'].astype(str).apply(_is_q_like)
        df = df[is_analyst | is_q]

    if df.empty:
        return pd.DataFrame({'uid': [], 'QDI': []})

    rows = []
    for uid, g in df.groupby('uid', observed=True):
        toks_all = []
        neg_cnt = num_cnt = qmark_cnt = clause_marks = total_utter = total_words = 0
        for _, r in g.iterrows():
            txt = str(r.get('transcripts', ''))
            toks = _tokenize(txt)
            if not toks:
                continue
            total_utter += 1
            total_words += len(toks)
            toks_all += toks
            neg_cnt += sum(1 for t in toks if t in neg_words)
            num_cnt += sum(1 for t in toks if (t.isdigit() or re.match(r'^[\-\+]?\d+(\.\d+)?(%|m|bn|b|k|\$)?$', t)))
            if txt.strip().endswith('?'):
                qmark_cnt += 1
            clause_marks += txt.count(',') + txt.count(';') + txt.count(' and ')
        if total_words == 0:
            continue
        uniq = len(set(toks_all))
        ttr = uniq / max(1, len(toks_all))
        rows.append({
            'uid': uid,
            'neg_density': neg_cnt / total_words,
            'num_density': num_cnt / total_words,
            'len_words':   total_words / max(1, total_utter),
            'ttr':         ttr,
            'qmark_ratio': qmark_cnt / max(1, total_utter),
            'clause_ratio': clause_marks / max(1, total_utter)
        })

    q = pd.DataFrame(rows)
    if q.empty:
        return pd.DataFrame({'uid': [], 'QDI': []})
    for c in ['neg_density','num_density','len_words','ttr','qmark_ratio','clause_ratio']:
        q[f'z_{c}'] = _zscore(q[c])
    zcols = [c for c in q.columns if c.startswith('z_')]
    q['QDI'] = q[zcols].mean(axis=1)
    q.to_csv(os.path.join(TAB_DIR, 'qdi_components.csv'), index=False)
    return q[['uid','QDI','neg_density','num_density','len_words','ttr','qmark_ratio','clause_ratio']]

# ----------------------------- Glitch (IV) Construction -----------------------------
def detect_glitches(df_segments: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs instrumental variables (IV) from signal quality metrics (mos_lqo/snr_db/jitter/packet_loss) or text patterns:
      - Marks abnormal frames based on quantile thresholds and aggregates to the call level.
      - Outputs summaries like glitch_iv, glitch_any, glitch_burst.
    """
    df = df_segments.copy()
    quality_cols = [c for c in df.columns if c.lower() in {'mos_lqo','snr_db','jitter','packet_loss'}]
    has_signal = len(quality_cols) > 0

    if has_signal:
        mark = pd.Series(False, index=df.index)
        if 'mos_lqo' in df.columns:
            thr = np.nanpercentile(pd.to_numeric(df['mos_lqo'], errors='coerce'), 10)
            mark |= (pd.to_numeric(df['mos_lqo'], errors='coerce') < thr)
        if 'snr_db' in df.columns:
            thr = np.nanpercentile(pd.to_numeric(df['snr_db'], errors='coerce'), 10)
            mark |= (pd.to_numeric(df['snr_db'], errors='coerce') < thr)
        if 'jitter' in df.columns:
            thr = np.nanpercentile(pd.to_numeric(df['jitter'], errors='coerce'), 90)
            mark |= (pd.to_numeric(df['jitter'], errors='coerce') > thr)
        if 'packet_loss' in df.columns:
            thr = np.nanpercentile(pd.to_numeric(df['packet_loss'], errors='coerce'), 90)
            mark |= (pd.to_numeric(df['packet_loss'], errors='coerce') > thr)
        df['glitch_flag_signal'] = mark.values
    else:
        pat = re.compile("|".join(_GLITCH_TEXT_PATTERNS), flags=re.IGNORECASE)
        df['glitch_flag_text'] = df['transcripts'].astype(str).str.contains(pat)

    def _burst_len(flags: List[bool]) -> int:
        """Calculates the length of the longest consecutive run of True values."""
        best = cur = 0
        for f in flags:
            if f:
                cur += 1; best = max(best, cur)
            else:
                cur = 0
        return best

    rows = []
    for uid, g in df.groupby('uid', observed=True):
        g = g.sort_values('event_time')
        if has_signal:
            flags = g['glitch_flag_signal'].fillna(False).astype(bool).tolist()
        else:
            flags = g['glitch_flag_text'].fillna(False).astype(bool).tolist()
        rate = float(np.mean(flags)) if len(flags) else 0.0
        rows.append({
            'uid': uid,
            'glitch_rate_signal': (rate if has_signal else np.nan),
            'glitch_rate_text':   (rate if not has_signal else np.nan),
            'glitch_any': float(rate > 0.0),
            'glitch_burst': float(_burst_len(flags))
        })
    out = pd.DataFrame(rows)
    out['glitch_iv'] = np.where(out['glitch_rate_signal'].notna(), out['glitch_rate_signal'], out['glitch_rate_text'])
    out.to_csv(os.path.join(TAB_DIR, 'glitch_iv_summary.csv'), index=False)
    return out

# ----------------------------- Cross-Role Delta-Feature Compact Composites -----------------------------
def build_delta_composites(final_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates composite features by averaging the Δ-dominance and Δ-valence of CEO/CFO/CXO roles:
      - delta_dominance_mean
      - delta_valence_mean
    """
    df = final_df.copy()
    candidates = [c for c in df.columns if 'delta_acoustic_' in c and any(k in c for k in ['_dominance_', '_valence_'])]
    keep = ['uid','ticker','event_time']
    out = df[keep].drop_duplicates().copy()

    def _avg_cols(substring: str) -> pd.Series:
        """Averages columns containing a specific substring for executive roles."""
        cols = [c for c in candidates if substring in c and any(r in c for r in ['CEO_','CFO_','CXO_'])]
        if not cols:
            return pd.Series(np.nan, index=df.index)
        return df[cols].mean(axis=1)

    out = (
        out.merge(
            df.assign(
                delta_dominance_mean=_avg_cols('_dominance_mean'),
                delta_valence_mean=_avg_cols('_valence_mean')
            )[["uid","delta_dominance_mean","delta_valence_mean"]],
            on='uid', how='left'
        )
    )
    return out

# ----------------------------- OLS (with Ticker-Clustered Robust Variances) -----------------------------
def _ols_clustered(y, X, cluster):
    """
    Performs OLS regression using statsmodels with cluster-robust (at the ticker level) standard errors.
    Returns the fitted result object, or None if all data is dropped due to missing values.
    """
    try:
        import statsmodels.api as sm
    except Exception as e:
        raise ImportError("statsmodels is required for IV/OLS but not installed.") from e

    # Consistently handle and drop missing data
    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=X.index, name='y')
    else:
        y = y.rename('y')
    if not isinstance(cluster, pd.Series):
        cluster = pd.Series(cluster, index=X.index, name='cluster')
    else:
        cluster = cluster.rename('cluster')

    df_reg = pd.concat([y, X, cluster], axis=1)
    df_reg.dropna(how='any', inplace=True)
    if df_reg.empty:
        warnings.warn("No data left for OLS after dropping NaNs.", RuntimeWarning)
        return None

    y_clean = df_reg['y']
    cluster_clean = df_reg['cluster']
    X_clean = df_reg.drop(columns=['y', 'cluster'])
    X1 = sm.add_constant(X_clean, has_constant='add')
    model = sm.OLS(y_clean, X1)
    res = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_clean})
    return res

# ----------------------------- IV (2SLS): Using Glitch as an Instrumental Variable -----------------------------
def run_iv_2sls(final_df: pd.DataFrame, qdi: pd.DataFrame, glitches: pd.DataFrame,
                baseline_col: str, horizon: int = 10) -> Dict:
    """
    Performs a two-stage least squares (2SLS) regression:
      Stage 1: Δdom/Δval ~ glitch_iv (+ glitch_any + glitch_burst + baseline + QDI)
      Stage 2: log_rv_t+hd ~ predicted(Δdom, Δval) + baseline + QDI
    Uses ticker-clustered robust standard errors.
    """
    try:
        import statsmodels.api as sm  # noqa: F401
    except Exception as e:
        print("[IV] statsmodels not available:", e)
        return {}

    df = final_df.copy()

    if isinstance(qdi, pd.DataFrame) and 'QDI' not in df.columns:
        df = df.merge(qdi[['uid','QDI']], on='uid', how='left')
    if isinstance(glitches, pd.DataFrame):
        df = df.merge(glitches[['uid','glitch_iv','glitch_any','glitch_burst']], on='uid', how='left')

    for col in ['QDI','glitch_iv','glitch_any','glitch_burst']:
        df = _coalesce_column(df, col)

    deltas = build_delta_composites(df)
    if isinstance(deltas, pd.DataFrame) and not deltas.empty:
        df = df.merge(deltas[['uid','delta_dominance_mean','delta_valence_mean']], on='uid', how='left')
    else:
        for col in ['delta_dominance_mean','delta_valence_mean']:
            if col not in df.columns:
                df[col] = np.nan

    tcol = f'log_rv_t+{horizon}d'
    if tcol not in df.columns:
        print(f"[IV] Target column not found: {tcol}")
        return {}

    keep = ['uid','ticker', baseline_col, 'QDI',
            'glitch_iv','glitch_any','glitch_burst',
            'delta_dominance_mean','delta_valence_mean', tcol]
    keep = [c for c in keep if c in df.columns]
    dfm = df[keep].dropna(subset=[tcol, baseline_col], how='any')
    if dfm.empty:
        print("[IV] No data after merges/filters (check QDI/glitch availability).")
        return {}

    X_iv = dfm[['glitch_iv','glitch_any','glitch_burst', baseline_col]].copy()
    if 'QDI' in dfm.columns:
        X_iv['QDI'] = dfm['QDI']
    y1 = dfm['delta_dominance_mean']
    y2 = dfm['delta_valence_mean']
    g  = dfm['ticker'].astype(str)

    res1 = _ols_clustered(y1, X_iv, cluster=g)
    res2 = _ols_clustered(y2, X_iv, cluster=g)
    if res1 is None or res2 is None:
        print("[IV] First stage regression failed (likely due to missing data).")
        return {}

    fstat_dom = float(res1.fvalue) if res1.fvalue is not None else np.nan
    fstat_val = float(res2.fvalue) if res2.fvalue is not None else np.nan

    import statsmodels.api as sm
    X1 = sm.add_constant(X_iv, has_constant='add')
    dfm['_pred_delta_dom'] = np.asarray(res1.predict(X1))
    dfm['_pred_delta_val'] = np.asarray(res2.predict(X1))

    X2_cols = ['_pred_delta_dom','_pred_delta_val', baseline_col]
    if 'QDI' in dfm.columns:
        X2_cols.append('QDI')
    X2 = dfm[X2_cols].copy()
    y  = dfm[tcol].values
    res_stage2 = _ols_clustered(y, X2, cluster=g)
    if res_stage2 is None:
        print("[IV] Second stage regression failed.")
        return {}

    tbl = pd.DataFrame({
        'variable': res_stage2.params.index,
        'coef': res_stage2.params.values,
        'std_err(cluster)': res_stage2.bse.values,
        't': res_stage2.tvalues.values,
        'p': res_stage2.pvalues.values
    })
    tbl.to_csv(os.path.join(TAB_DIR, f'iv_stage2_coef_{horizon}d.csv'), index=False)

    info = {
        'n': int(res_stage2.nobs),
        'first_stage_F_dom': fstat_dom,
        'first_stage_F_val': fstat_val,
        'stage2_r2': float(res_stage2.rsquared),
        'stage2_adj_r2': float(res_stage2.rsquared_adj)
    }
    with open(os.path.join(TAB_DIR, f'iv_summary_{horizon}d.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)

    return {'stage1_dom': res1, 'stage1_val': res2, 'stage2': res_stage2, 'summary': info}

# ----------------------------- ΔR² Curve (vs. Financials-Only Baseline) -----------------------------
def compute_delta_curves_vs_baseline(final_df: pd.DataFrame,
                                     base_set: str = BASELINE_SET_NAME,
                                     full_set: str = FULL_SET_NAME,
                                     horizons: List[int] = HORIZONS) -> pd.DataFrame:
    """
    Uses the main script's _collect_predictions_by_horizon to gather (y, yb, yf, g) for each horizon.
    Calculates ΔR²(h) = R²_full(h) − R²_base(h) and returns a DataFrame sorted by horizon.
    """
    preds = emo._collect_predictions_by_horizon(final_df, base_set, full_set, horizons=tuple(horizons))
    rows = []
    for h in sorted(set(horizons)):
        if h not in preds:
            continue
        y  = preds[h]['y']; yb = preds[h]['yb']; yf = preds[h]['yf']
        dr2 = emo._r2(y, yf) - emo._r2(y, yb)
        rows.append({'Horizon (days)': int(h), 'Delta R^2': float(dr2)})
    curve = pd.DataFrame(rows).sort_values('Horizon (days)')
    return curve

# ----------------------------- Decay Fitting (Half-Life) -----------------------------
def _decay_func(h, alpha, beta, tau):
    """Exponential decay function."""
    return alpha - beta * np.exp(-h / np.maximum(1e-6, tau))

def _fit_decay_leastsq(h: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Lightweight least squares fitting: performs a grid search for tau∈[1,60],
    and for each tau, solves for (alpha, beta) analytically.
    """
    h = np.asarray(h, dtype=float)
    y = np.asarray(y, dtype=float)
    tau_grid = np.linspace(1.0, 60.0, 120)
    best = (np.nan, np.nan, np.nan, np.inf)
    for tau in tau_grid:
        X = np.vstack([np.ones_like(h), -np.exp(-h / tau)]).T
        try:
            beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
            yhat = X @ beta_hat
            rss = float(np.sum((y - yhat) ** 2))
            if rss < best[3]:
                best = (float(beta_hat[0]), float(beta_hat[1]), float(tau), rss)
        except Exception:
            continue
    return best[0], best[1], best[2]

def bootstrap_decay_ci_with_predictions(final_df: pd.DataFrame,
                                        base_set: str = BASELINE_SET_NAME,
                                        full_set: str = FULL_SET_NAME,
                                        horizons: List[int] = HORIZONS,
                                        B: int = 500,
                                        seed: int = 42) -> Dict:
    """
    Calculates ΔR²(h) from collected predictions, fits a decay curve,
    and computes confidence intervals using a clustered (by ticker) bootstrap.
    """
    rng = np.random.default_rng(seed)
    preds = emo._collect_predictions_by_horizon(final_df, base_set, full_set, horizons=tuple(horizons))
    if not preds:
        return {'alpha': np.nan, 'beta': np.nan, 'tau': np.nan,
                'ci': {'alpha': (np.nan, np.nan), 'beta': (np.nan, np.nan), 'tau': (np.nan, np.nan)}}

    # Full sample point estimate
    dr2_all = []
    H = []
    for h in sorted(set(horizons)):
        if h not in preds:
            continue
        y  = preds[h]['y']; yb = preds[h]['yb']; yf = preds[h]['yf']
        dr2_all.append(emo._r2(y, yf) - emo._r2(y, yb))
        H.append(h)
    H = np.array(H, dtype=float)
    Y = np.array(dr2_all, dtype=float)
    alpha_hat, beta_hat, tau_hat = _fit_decay_leastsq(H, Y)

    # Clustered bootstrap
    G_un = np.unique(np.concatenate([preds[h]['g'] for h in preds]))
    boot = []
    for _ in range(B):
        gs = rng.choice(G_un, size=len(G_un), replace=True)
        dr2_b = []
        H_b = []
        ok = True
        for h in sorted(set(horizons)):
            if h not in preds:
                ok = False; break
            mask = np.isin(preds[h]['g'], gs)
            if np.sum(mask) < 30:
                ok = False; break
            y, yb, yf = preds[h]['y'][mask], preds[h]['yb'][mask], preds[h]['yf'][mask]
            dr2_b.append(emo._r2(y, yf) - emo._r2(y, yb))
            H_b.append(h)
        if not ok or len(dr2_b) < 2:
            continue
        Hb = np.array(H_b, dtype=float)
        Yb = np.array(dr2_b, dtype=float)
        a, b, t = _fit_decay_leastsq(Hb, Yb)
        if all(map(np.isfinite, [a, b, t])):
            boot.append((a, b, t))

    if not boot:
        ci = {'alpha': (np.nan, np.nan), 'beta': (np.nan, np.nan), 'tau': (np.nan, np.nan)}
    else:
        A = np.array([b[0] for b in boot], dtype=float)
        BETA = np.array([b[1] for b in boot], dtype=float)
        T = np.array([b[2] for b in boot], dtype=float)
        ci = {
            'alpha': (float(np.nanpercentile(A, 2.5)), float(np.nanpercentile(A, 97.5))),
            'beta':  (float(np.nanpercentile(BETA, 2.5)), float(np.nanpercentile(BETA, 97.5))),
            'tau':   (float(np.nanpercentile(T, 2.5)), float(np.nanpercentile(T, 97.5)))
        }

    out = {'alpha': float(alpha_hat), 'beta': float(beta_hat), 'tau': float(tau_hat),
           'ci': ci, 'dr2_curve': [{'horizon': int(h), 'delta_r2': float(v)} for h, v in zip(H, Y)]}
    with open(os.path.join(TAB_DIR, 'decay_fit_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    return out

# ----------------------------- Absorb (as defined in the paper) -----------------------------
def _absorb_from_dr2(dr2_map: Dict[int, float]) -> float:
    """
    Paper's definition: Absorb = 0.5*(ΔR²_10 + ΔR²_15) − ΔR²_1
    (A more negative value indicates a "deepening penalty", consistent with the paper's narrative)
    """
    if not all(h in dr2_map for h in (1, 10, 15)):
        return float('nan')
    return 0.5*(dr2_map[10] + dr2_map[15]) - dr2_map[1]

def compute_absorb_and_write_inline(final_df: pd.DataFrame,
                                    outdir_tables: str = TAB_DIR,
                                    base_set: str = BASELINE_SET_NAME,
                                    full_set: str = FULL_SET_NAME,
                                    horizons: Tuple[int, ...] = (1, 10, 15),
                                    alpha: float = 0.05,
                                    B: int = N_BOOTSTRAPS_ABSORB,
                                    P: int = N_PERMUTATIONS_ABSORB,
                                    seed: int = GLOBAL_SEED) -> None:
    """
    Using Financials-Only as a baseline, calculates the Absorb score for horizons (1,10,15)d.
    Outputs the point estimate, clustered bootstrap CI, and permutation p-value.
    """
    os.makedirs(outdir_tables, exist_ok=True)
    preds = emo._collect_predictions_by_horizon(final_df, base_set, full_set, horizons=horizons)
    if not all(h in preds for h in horizons):
        print(r'\emph{Absorb could not be computed (insufficient predictions).}')
        return

    # Point estimate
    dr2 = {h: (emo._r2(preds[h]['y'], preds[h]['yf']) - emo._r2(preds[h]['y'], preds[h]['yb'])) for h in horizons}
    absorb_hat = _absorb_from_dr2(dr2)

    # Clustered bootstrap CI
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
            dr2_b[h] = emo._r2(y, yf) - emo._r2(y, yb)
        if ok and all(k in dr2_b for k in horizons):
            boot_vals.append(_absorb_from_dr2(dr2_b))
    if boot_vals:
        pl, pu = alpha/2*100, (1-alpha/2)*100
        lo = float(np.percentile(boot_vals, pl)); hi = float(np.percentile(boot_vals, pu))
    else:
        lo = hi = float('nan')

    # Permutation test (shuffling full model predictions within ticker clusters)
    perm = []
    for _ in range(P):
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
            dr2_p[h] = emo._r2(y, yf) - emo._r2(y, yb)
        if ok:
            perm.append(_absorb_from_dr2(dr2_p))
    if perm:
        pval = float((np.sum(np.abs(perm) >= abs(absorb_hat)) + 1.0) / (len(perm) + 1.0))
    else:
        pval = float('nan')

    with open(os.path.join(outdir_tables, 'absorb_values.json'), 'w', encoding='utf-8') as f:
        json.dump({'Absorb': absorb_hat, 'ci_lo': lo, 'ci_hi': hi, 'p_perm': pval,
                   'B': B, 'P': P, 'horizons': list(horizons)}, f, indent=2)
    print(f"[Absorb] Absorb = {absorb_hat:+.3f} (95% CI [{lo:+.3f}, {hi:+.3f}]; perm. p = {pval:.3f})")

# ----------------------------- Normalized Effects (Economic Magnitude) -----------------------------
def compute_normalized_effects(final_df: pd.DataFrame, baseline_col: str, horizon: int = 10) -> pd.DataFrame:
    """
    Regresses: log_rv_h ~ Δdom + Δval + baseline (+QDI).
    Reports the impact of a 1-SD change in ΔX on log(RV) and RV(%).
    """
    try:
        import statsmodels.api as sm  # noqa: F401
    except Exception as e:
        print("[Normalized Effects] statsmodels not available:", e)
        return pd.DataFrame({'variable': [], 'beta': [], 'beta_se': [], 'sd_x': [],
                             'd_logRV_per_1sd': [], 'pct_change_RV(%)': [], 'pct_of_preRV(%)': []})

    df = final_df.copy()
    df = _coalesce_column(df, 'QDI')

    deltas = build_delta_composites(df)
    df = df.merge(deltas[['uid','delta_dominance_mean','delta_valence_mean']], on='uid', how='left')
    tcol = f'log_rv_t+{horizon}d'
    keep = ['uid','ticker', baseline_col, tcol, 'delta_dominance_mean','delta_valence_mean']
    if 'QDI' in df.columns:
        keep.append('QDI')
    d = df[keep].dropna()
    if d.empty:
        return pd.DataFrame({'variable': [], 'beta': [], 'beta_se': [], 'sd_x': [],
                             'd_logRV_per_1sd': [], 'pct_change_RV(%)': [], 'pct_of_preRV(%)': []})

    X = d[['delta_dominance_mean','delta_valence_mean', baseline_col] + (['QDI'] if 'QDI' in d.columns else [])]
    y = d[tcol].values
    g = d['ticker'].astype(str).values
    res = _ols_clustered(y, X, cluster=g)
    if res is None:
        print("[Normalized Effects] Regression failed.")
        return pd.DataFrame()

    rows = []
    for v in ['delta_dominance_mean','delta_valence_mean']:
        if v not in X.columns:
            continue
        beta = float(res.params.get(v, np.nan))
        se   = float(res.bse.get(v, np.nan))
        sd_x = float(np.nanstd(X[v].values, ddof=1))
        dlog = beta * sd_x
        pct_RV = (np.exp(dlog) - 1.0) * 100.0
        base = np.nanmean(d[baseline_col].values)
        pct_of_pre = (pct_RV / (base * 100.0)) * 100.0 if np.isfinite(base) and base > 0 else np.nan
        rows.append({'variable': v, 'beta': beta, 'beta_se': se, 'sd_x': sd_x,
                     'd_logRV_per_1sd': dlog, 'pct_change_RV(%)': pct_RV, 'pct_of_preRV(%)': pct_of_pre})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TAB_DIR, f'normalized_effects_{horizon}d.csv'), index=False)
    return out

# ----------------------------- Optional: VAD Comparison for Enhancement -----------------------------
def compare_enhancement_vad_if_available(df_segments: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    If the segment-level data contains both pre- and post-enhancement emotion labels
    (e.g., acoustic_emotion_piam vs. acoustic_emotion_baseline), this function maps them
    to Valence/Arousal/Dominance and outputs a summary of the differences.
    """
    cand_pairs = [
        ('acoustic_emotion_piam', 'acoustic_emotion_base'),
        ('acoustic_emotion_piam', 'acoustic_emotion_baseline'),
    ]
    pair = None
    for a, b in cand_pairs:
        if a in df_segments.columns and b in df_segments.columns:
            pair = (a, b); break
    if pair is None:
        return None

    A, B = pair
    df = df_segments[['uid','ticker','event_time', A, B]].copy()
    def _map_em(x, i):
        """Maps an emotion label to its V/A/D coordinate."""
        coords = getattr(emo, 'VAD_MAP', {}).get(str(x).lower(), (0.0, 0.0, 0.0))
        return coords[i]
    for i, dim in enumerate(['V','A','D']):
        df[f'{A}_{dim}'] = df[A].apply(lambda x: _map_em(x, i))
        df[f'{B}_{dim}'] = df[B].apply(lambda x: _map_em(x, i))
        df[f'delta_{dim}'] = df[f'{A}_{dim}'] - df[f'{B}_{dim}']

    agg = df.groupby('uid', observed=True)[['delta_V','delta_A','delta_D']].mean().reset_index()
    agg.to_csv(os.path.join(TAB_DIR, 'enhancement_vad_compare.csv'), index=False)
    return agg

# ----------------------------- Main Workflow -----------------------------
def main():
    print("="*72)
    print("Extended experiments (baseline = Financials-Only): Decay-fit, Glitch-IV, QDI, Absorb, Normalized Effects")
    print("="*72)

    # 1) Load or minimally rebuild data
    final_df, df_segments, meta = build_or_load_call_level()
    baseline_col = meta['baseline_col']

    # 2) Build QDI
    print("[Step] Building QDI ...")
    qdi = build_qdi(df_segments)
    if not qdi.empty:
        final_df = final_df.merge(qdi[['uid','QDI']], on='uid', how='left')
        final_df = _coalesce_column(final_df, 'QDI')

    # 3) Detect Glitches for IV
    print("[Step] Detecting glitches (IV) ...")
    glitches = detect_glitches(df_segments)

    # 4) Run IV (2SLS) @ 10d
    print("[Step] Running IV (2SLS) @ 10d ...")
    iv_res = run_iv_2sls(final_df, qdi, glitches, baseline_col=baseline_col, horizon=PRIMARY_HORIZON)
    if iv_res and 'summary' in iv_res:
        print("[IV] Summary:", iv_res['summary'])

    # 5) ΔR² Curve (Multimodal vs Financials-Only) + Decay Fitting
    print("[Step] Computing ΔR^2 curve vs Financials-Only & Decay-fit ...")
    curve = compute_delta_curves_vs_baseline(final_df, base_set=BASELINE_SET_NAME, full_set=FULL_SET_NAME, horizons=HORIZONS)
    # Maintain compatibility with old and new filenames
    curve.to_csv(os.path.join(TAB_DIR, 'delta_r2_curve_multimodal_vs_fintext.csv'), index=False)
    curve.to_csv(os.path.join(TAB_DIR, 'delta_r2_curve_multimodal_vs_nonacoustic.csv'), index=False)

    decay = bootstrap_decay_ci_with_predictions(final_df, base_set=BASELINE_SET_NAME, full_set=FULL_SET_NAME, horizons=HORIZONS)
    if decay and isinstance(decay, dict) and 'tau' in decay:
        ci_tau = decay.get('ci', {}).get('tau', (np.nan, np.nan))
        print(f"[Decay] tau = {decay['tau']:.2f} days (vs Financials-Only); 95% CI = [{ci_tau[0]:.2f}, {ci_tau[1]:.2f}]")

    # 6) Absorb score (1/10/15 days, baseline = Financials-Only)
    # print("[Step] Computing Absorb (1/10/15 days, baseline=Financials-Only) ...")
    # compute_absorb_and_write_inline(final_df,
    #                                 outdir_tables=TAB_DIR,
    #                                 base_set=BASELINE_SET_NAME,
    #                                 full_set=FULL_SET_NAME,
    #                                 horizons=(1, 10, 15))

    # 7) Normalized Effects (Economic Magnitude)
    print("[Step] Computing normalized effect sizes ...")
    eff = compute_normalized_effects(final_df, baseline_col=baseline_col, horizon=PRIMARY_HORIZON)
    if not eff.empty:
        print(eff.to_string(index=False, float_format="%.4f"))

    # 8) Optional: VAD comparison for enhancer
    print("[Step] Optional: enhancement VAD comparison ...")
    evc = compare_enhancement_vad_if_available(df_segments)
    if evc is None:
        print("[Info] Enhancement comparison skipped (no parallel emotion columns).")
    else:
        print("[OK] enhancement_vad_compare.csv exported.")

    print("\n[Done] All extended artifacts saved under:", EXT_OUTDIR)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()