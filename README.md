# Physics-Informed Modeling Uncovers a Persistent Predictive Penalty from Vocal Affect in Markets

This repository contains the source code, analysis scripts, and links to the public dataset for the paper: *"Physics-Informed Modeling Uncovers a Persistent Predictive Penalty from Vocal Affect in Markets"*.

Our study explores how executives' non-verbal vocal cues during corporate earnings calls influence market behavior. We identify a "predictive penalty," where vocal affect degrades stock volatility forecast accuracy over a 30-day horizon, acting as a "thermometer" of behavioral risk rather than an alpha-generating signal.

## Key Contributions

1. **Predictive Penalty Discovery**: Vocal affect in earnings calls consistently worsens out-of-sample stock volatility predictions, intensifying over 10-15 trading days and persisting for at least 30 days.
2. **PIMM Framework**: We developed the Physics-Informed Multimodal Model (PIMM) to extract vocal cues from complex earnings call audio, integrating sound propagation principles with deep learning for robust denoising and feature extraction.

## The PIMM Framework

The Physics-Informed Multimodal Model (PIMM) processes far-field teleconference audio, performing:
- **Acoustic Signal Enhancement (ASE)**: Denoising, dereverberation, and echo cancellation.
- **Automatic Speech Recognition (ASR)**: Time-aligned transcript generation.
- **Acoustic Emotion Recognition (AER)**: Categorical emotion detection from vocal tone.
- **Acoustic Event Detection (AED)**: Identification of non-voice cues (e.g., glitches, overlapping speech).

The trained PIMM model is accessible via a public API. See [inference.md](inference.md) for detailed usage instructions.

## Repository Structure

```
├── LICENSE
├── README.md
├── requirements.txt
├── .gitignore
├── inference/
│   ├── inference.md                      # Detailed PIMM usage instructions
│   ├── emovoice_api_config.json          # API keys and model configuration
│   ├── emovoice_batch_asr_result.py      # Batch speech recognition
│   ├── emovoice_batch_emotion_analysis.py # Batch emotion analysis
│   ├── emovoice_merge_emotion_to_original.py # Merges emotion results with transcripts
│   ├── poll_run_3.log                    # Example ASR processing log
│   └── batch_emotion_analysis.log        # Example emotion analysis log
├── analysis/
│   ├── emovoice_segments_descriptive.py  # Descriptive statistics and figures (Supp. Figs S1-S4)
│   ├── emovoice_extended_experiments.py  # Extended analyses, including confounder controls (Tables S11-S13)
│   ├── run_forecasting_models.py         # Trains/evaluates XGBoost models (Table 1, Fig 2)
│   ├── run_robustness_checks.py          # Heterogeneity and sensitivity analyses (Fig 3, Tables S7-S9)
│   ├── generate_statistical_tests.py     # Statistical tests (e.g., Diebold-Mariano, Tables S5, S6)
│   └── generate_paper_visuals.py         # Generates manuscript plots and tables
```

## Data

The study uses a corpus of 1,795 quarterly earnings calls with aligned audio and transcripts. The 151GB dataset, analysis code, and derived features are available on Hugging Face for reproducibility.

**[Download the dataset from Hugging Face](https://huggingface.co/datasets/soundai2016/pimm_nasdaq_earningscall)**

## Reproducing the Analysis

### Step 1: Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/soundai2016/pimm_nasdaq_earningscall.git
   cd pimm_nasdaq_earningscall
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install `ffmpeg` for audio conversion.

### Step 2: Data Preprocessing

Use scripts in the `inference/` directory to process audio and generate features. Detailed instructions are in [inference.md](inference.md):
1. **Batch ASR**: Run `emovoice_batch_asr_result.py` to transcribe audio into time-aligned JSON files.
2. **Emotion Analysis**: Run `emovoice_batch_emotion_analysis.py` to derive Valence-Arousal-Dominance (V-A-D) emotion features.
3. **Merge Results**: Run `emovoice_merge_emotion_to_original.py` to combine transcripts with emotion features.

### Step 3: Analysis and Visualization

Use scripts in the `analysis/` directory to train models and generate results:
- **Predictive Penalty**: See Figure 2, Table 1, and Supplementary Tables S3-S5 for volatility/return predictions and statistical tests.
- **Uncertainty Absorption**: The `Absorb` metric quantifies penalty dynamics (Supplementary Table S1).
- **Robustness**: Stable across industries and volatility regimes (Figure 3, Supplementary Tables S7-S9).
- **Extended Analyses**:
  - Emotional masking tests (Supplementary Table S10).
  - Analyst question complexity controls (Supplementary Table S11).
  - Instrumental variable analysis (Supplementary Tables S12-S13).
  - Vocal vs. textual signal distinction (Supplementary Table S14).
  - Economic impact assessment (Supplementary Table S15).
- **Descriptive Statistics**: Visualizations in Supplementary Figures S1-S4.

## Citation

```bibtex
@article{chen2025physics,
  title={Physics-informed modeling uncovers a persistent predictive penalty from vocal affect in markets},
  author={Chen, Xiaoliang and Chang, Le and Yu, Xin and Huang, Yunhe and Jing, Teng and He, Jiashuai and Luo, Yangjun and Cai, Yixuan and Qiu, Yuebo and Sun, Peiwen},
  journal={Research Square},
  year={2025},
  doi={10.21203/rs.3.rs-7655247/v1},
  url={https://doi.org/10.21203/rs.3.rs-7655247/v1}
}
```

## Ethics and Reproducibility

This observational study uses public earnings call data with no personal identifying information. All data, code, and features are shared via this repository and Hugging Face for full reproducibility.