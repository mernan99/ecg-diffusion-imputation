# Conditional Diffusion Models for Missing ECG Reconstruction with Uncertainty Quantification


A research project investigating **missing-segment reconstruction in electrocardiogram (ECG) time series** using classical interpolation, deterministic neural networks, and conditional denoising diffusion models.

The project evaluates whether a conditional diffusion model can reconstruct missing ECG segments more accurately than conventional baselines while also providing **predictive uncertainty estimates** through repeated stochastic reconstructions. It additionally applies **Polynomial Chaos Expansion (PCE)** and **Sobol sensitivity analysis** to identify which experimental factors drive reconstruction error, waveform degradation, and uncertainty.

---

## Contents


- [Project Overview](https://github.com/mernan99/ecg-diffusion-imputation#project-overview)
- [Research Questions](https://github.com/mernan99/ecg-diffusion-imputation#research-questions)
- [Main Findings](https://github.com/mernan99/ecg-diffusion-imputation#main-findings)
- [Dataset](https://github.com/mernan99/ecg-diffusion-imputation#dataset)
- [Experimental Design](https://github.com/mernan99/ecg-diffusion-imputation#experimental-design)
- [Methods](https://github.com/mernan99/ecg-diffusion-imputation#methods) 
  - [Interpolation Baselines](https://github.com/mernan99/ecg-diffusion-imputation#1-interpolation-baselines)
  - [Basic Autoencoder](https://github.com/mernan99/ecg-diffusion-imputation#2-basic-autoencoder)
  - [Morphology-Aware 1D U-Net](https://github.com/mernan99/ecg-diffusion-imputation#3-morphology-aware-1d-u-net)
  - [Conditional Diffusion Model](https://github.com/mernan99/ecg-diffusion-imputation#4-conditional-diffusion-model)
  - [Predictive Uncertainty](https://github.com/mernan99/ecg-diffusion-imputation#5-predictive-uncertainty)
  - [PCE and Sobol Sensitivity Analysis](https://github.com/mernan99/ecg-diffusion-imputation#6-pce-and-sobol-sensitivity-analysis)
  - [Paired Statistical Analysis](https://github.com/mernan99/ecg-diffusion-imputation#7-paired-statistical-analysis)
- [Repository Structure](https://github.com/mernan99/ecg-diffusion-imputation#repository-structure)
- [Installation](https://github.com/mernan99/ecg-diffusion-imputation#installation)
- [Dataset Setup](https://github.com/mernan99/ecg-diffusion-imputation#dataset-setup)
- [Running the Project](https://github.com/mernan99/ecg-diffusion-imputation#running-the-project)
- [Evaluation Metrics](https://github.com/mernan99/ecg-diffusion-imputation#evaluation-metrics)
- [Results](https://github.com/mernan99/ecg-diffusion-imputation#results)
- [Uncertainty Quantification Results](https://github.com/mernan99/ecg-diffusion-imputation#uncertainty-quantification-results)
- [Reproducibility](https://github.com/mernan99/ecg-diffusion-imputation#reproducibility)
- [Generated Outputs](https://github.com/mernan99/ecg-diffusion-imputation#generated-outputs)
- [Limitations](https://github.com/mernan99/ecg-diffusion-imputation#limitations)
- [Future Work](https://github.com/mernan99/ecg-diffusion-imputation#future-work)
- [Research Interpretation](https://github.com/mernan99/ecg-diffusion-imputation#research-interpretation)
- [Acknowledgements](https://github.com/mernan99/ecg-diffusion-imputation#acknowledgements)

---

# Project Overview


Missing data are common in physiological time series because of sensor disconnection, electrode contact failure, transmission loss, motion artefact, temporary acquisition failure, or corrupted recording intervals.

Simple interpolation can reconstruct smooth trends, but ECG morphology contains sharp and highly structured temporal events such as QRS complexes. When a complete cardiac event is missing, connecting the samples immediately before and after the gap is often insufficient.

This project therefore compares several increasingly sophisticated approaches:

```
Linear / PCHIP / Cubic interpolation
                ↓
       Basic convolutional AE
                ↓
     Morphology-aware 1D U-Net
                ↓
   Conditional diffusion model
                ↓
   Predictive uncertainty analysis
                ↓
       PCE + Sobol sensitivity
                ↓
      Paired statistical testing

```

The central distinction is that the deterministic U-Net learns a single reconstruction,

$$
\hat{x}=f_\theta(x_{\mathrm{observed}},m),
$$

whereas the diffusion model attempts to represent a conditional distribution,

$$
p_\theta\left(x_{\mathrm{missing}}\mid x_{\mathrm{observed}},m\right),
$$

allowing multiple plausible missing segments to be generated.

---

# Research Questions


### RQ1 — Reconstruction


**Can a conditional diffusion model reconstruct missing ECG segments more accurately than interpolation and deterministic neural-network baselines?**

### RQ2 — Missing-segment difficulty


**How does model performance change as the duration of the missing ECG segment increases?**

The evaluated gaps are:

$$
0.25,\;0.50,\;1.00,\;2.00\;\text{seconds}
$$

### RQ3 — Uncertainty


**Can stochastic diffusion reconstructions provide useful predictive uncertainty estimates, and which factors drive that uncertainty?**

The global UQ analysis considers:

$$
X=[\text{gap duration},\text{gap position},\text{measurement noise},\text{baseline wander}].
$$

---

# Main Findings


The main experimental result is a **gap-length-dependent crossover** between diffusion and the deterministic U-Net.

| Missing gap | U-Net RMSE ↓ | Diffusion RMSE ↓ | U-Net correlation ↑ | Diffusion correlation ↑ |
|---:|---:|---:|---:|---:|
| 0.25 s                                                                                 | 0.0533     | **0.0418** | 0.760     | **0.816** |
| 0.50 s                                                                                 | 0.0666     | **0.0615** | 0.794     | **0.815** |
| 1.00 s                                                                                 | **0.0826** | 0.0846     | **0.737** | 0.714     |
| 2.00 s                                                                                 | **0.1032** | 0.1103     | **0.629** | 0.555     |

The paired RMSE comparison shows the same pattern:

| Missing gap | Fraction of cases where diffusion had lower RMSE |
|---:|---:|
| 0.25 s                                                        | \~76.8% |
| 0.50 s                                                        | \~67.1% |
| 1.00 s                                                        | \~50.2% |
| 2.00 s                                                        | \~38.6% |

This suggests that diffusion is strongest for **short gaps**, the two learned approaches are approximately competitive around **1 second**, and the deterministic U-Net becomes a stronger **point estimator** for **2-second gaps**. Diffusion additionally provides a distribution of plausible reconstructions and therefore supports uncertainty estimation.

The UQ analysis also indicates that:

- **gap duration** strongly influences reconstruction quality and waveform correlation;
- **gap duration and measurement noise** dominate predictive uncertainty magnitude;
- **baseline wander** has a strong influence on probabilistic quality and interval calibration;
- **gap position** contributes comparatively little over the evaluated range.

---

# Dataset


## PTB-XL


This project uses the **PTB-XL** electrocardiography dataset.

The dataset itself is **not included in this repository**.

Download PTB-XL version **1.0.3** from PhysioNet and extract it into:

```
data/
└── ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/

```

The experiments use the **100 Hz recordings** stored in:

```
records100/

```

The expected dataset directory therefore resembles:

```
data/
└── ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
    ├── ptbxl_database.csv
    ├── records100/
    ├── records500/
    └── ...

```

## ECG configuration


The current project uses:

- **Lead:** II
- **Sampling frequency:** 100 Hz
- **Duration:** 10 seconds
- **Samples per signal:** 1000

The model input is therefore a one-dimensional physiological time series.

---

# Experimental Design


## PTB-XL folds


The PTB-XL `strat_fold` field is used to create a fixed train/validation/test protocol:

| Fold(s) | Purpose |
|---|---|
| 1–8              | Training                     |
| 9                | Validation / model selection |
| 10               | Final held-out testing       |

No Fold-10 records are used during model training or validation.

## Training-set normalisation


Normalisation statistics are calculated using training folds only.

For Lead II, the training statistics used in the current experiments are approximately:

$$
\mu=-0.00149\;\mathrm{mV}
$$

and

$$
\sigma=0.16422\;\mathrm{mV}
$$

Signals are standardised as:

$$
x_{\mathrm{norm}}=\frac{x-\mu}{\sigma}.
$$

This avoids leaking information from validation or test folds into preprocessing.

## Missing-data simulation


Contiguous missing regions are synthetically introduced with durations:

```
0.25 seconds
0.50 seconds
1.00 second
2.00 seconds

```

At 100 Hz, these correspond to:

```
25 samples
50 samples
100 samples
200 samples

```

A margin is maintained around the artificial missing interval so that the gap does not begin or end at the edge of the ECG.

The observation mask is:

$$
m_i=\begin{cases}
1,& \text{sample observed}\\
0,& \text{sample missing}.
\end{cases}
$$

The corrupted input is produced by setting the missing region to zero in normalised space while supplying the binary mask as a separate channel.

---

# Fixed Fold-10 Benchmark


A major reproducibility step in this project is the use of:

```
results/fold10_mask_manifest.csv

```

Rather than allowing each model to generate its own random test gaps, this manifest permanently stores the exact ECG record, gap duration, trial, start sample, and end sample.

This ensures that:

```
Linear interpolation
Basic AE
U-Net
Diffusion

```

are all evaluated on **identical missing ECG regions**.

For each of the four gap lengths, five trials are generated per Fold-10 recording.

This produces approximately:

```
2,198 Fold-10 records
× 4 gap lengths
× 5 trials
=
43,960 fixed test gaps

```

This fixed benchmark is used for the principal deterministic and diffusion comparisons.

---

# Methods


## 1. Interpolation Baselines


Three conventional interpolation approaches are included.

### Linear interpolation


For two observed samples surrounding a missing region,

$$
(t_a,x_a)\quad\text{and}\quad(t_b,x_b),
$$

linear interpolation estimates:

$$
\hat{x}$t$=x_a+\frac{t-t_a}{t_b-t_a}(x_b-x_a).
$$

Linear interpolation is computationally inexpensive and stable but cannot recreate missing nonlinear cardiac morphology.

### Cubic spline interpolation


A cubic spline fits smooth piecewise cubic functions between observed points.

In this project, cubic spline interpolation becomes unstable for long missing ECG intervals and can substantially overshoot the physiological signal range.

### PCHIP


Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) is included as a shape-preserving alternative to conventional cubic splines.

It avoids some spline overshoot, although it still cannot infer missing cardiac events that are not implied by neighbouring samples.

---

## 2. Basic Autoencoder


The first learned baseline is a convolutional denoising autoencoder.

Conceptually:

```
corrupted ECG + mask
        ↓
1D convolutional encoder
        ↓
compressed representation
        ↓
1D decoder
        ↓
reconstructed ECG

```

The model receives two input channels:

```
Channel 1: corrupted ECG
Channel 2: binary observation mask

```

and predicts a one-channel ECG reconstruction.

The loss is evaluated on the missing region so that training focuses on imputation rather than simply reproducing samples already observed.

The basic autoencoder improves numerical reconstruction error over interpolation but tends to produce overly smooth predictions for difficult gaps. This motivated the use of a skip-connected U-Net architecture.

---

## 3. Morphology-Aware 1D U-Net


The stronger deterministic model is a one-dimensional U-Net designed to retain temporal morphology.

The architecture contains:

- 1D convolutional encoder blocks;
- temporal downsampling;
- bottleneck processing;
- decoder blocks;
- skip connections between matching encoder and decoder resolutions.

Skip connections allow fine-grained temporal information to bypass the compressed bottleneck.

Conceptually:

```
input
 │
 ├────────────── skip ───────────────┐
 ↓                                  │
encoder                              │
 ↓                                  │
latent / bottleneck                  │
 ↓                                  │
decoder ◄────────────────────────────┘
 ↓
reconstructed ECG

```

The U-Net proved substantially stronger than the basic autoencoder, particularly for recovering sharp ECG events.

---

## 4. Conditional Diffusion Model


The central generative model is a **conditional one-dimensional denoising diffusion probabilistic model (DDPM)**.

### Forward diffusion


A clean missing signal $x_0$ is progressively corrupted with Gaussian noise.

Define:

$$
\alpha_t=1-\beta_t
$$

and:

$$
\bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s
$$

The forward process can be sampled directly using:

$$
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,
$$

where:

$$
\epsilon\sim\mathcal N(0,I).
$$

The project uses a **cosine noise schedule**.

### Conditional noise prediction


The neural network predicts the noise added at diffusion step $t$.

The model input contains three channels:

```
1. Current diffusion state
2. Observed ECG with missing samples zero-filled
3. Observation mask

```

Therefore the learned function is conceptually:

$$
\epsilon_\theta(x_t,t,x_{\mathrm{observed}},m).
$$

A sinusoidal timestep embedding supplies the diffusion step to residual convolutional blocks.

### Missing-region loss


Noise-prediction loss is evaluated only on missing samples:

$$
\mathcal L=\frac{\sum_i(1-m_i)(\epsilon_i-\epsilon_{\theta,i})^2}{\sum_i(1-m_i)}.
$$

This prevents observed samples from dominating training.

### Conditional clamping


During reverse diffusion, observed samples are kept fixed:

$$
x_t=m\odot x_{\mathrm{observed}}+(1-m)\odot x_t.
$$

Therefore stochastic generation occurs only inside the missing region.

### Stable reverse sampling


The initial DDPM sampler occasionally produced rare, extremely large reconstruction values.

The sampler was therefore changed to explicitly estimate the predicted clean ECG:

$$
\hat{x}_0=\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t,t)}{\sqrt{\bar{\alpha}_t}}
$$

The predicted clean signal is clipped to a conservative range in normalised space before evaluating the reverse posterior.

The reverse step then uses:

$$
q(x_{t-1}\mid x_t,\hat{x}_0)=\mathcal N(\tilde{\mu}_t,\tilde{\beta}_t I).
$$

This substantially improved sampling stability without retraining the diffusion network.

### Exponential moving average


An exponential moving average of the network weights is maintained during training and used for diffusion sampling.

This improves sampling stability compared with using the raw optimisation weights directly.

---

## 5. Predictive Uncertainty


A deterministic reconstruction produces a single signal.

Diffusion instead permits repeated sampling:

$$
x^{(1)},x^{(2)},\ldots,x^{(K)}\sim p_\theta(x_{\mathrm{missing}}\mid x_{\mathrm{observed}}).
$$

For each missing time point, the predictive mean is:

$$
\hat{\mu}_t=\frac{1}{K}\sum_{k=1}^{K}x_t^{(k)}
$$

The sample variance is:

$$
\hat{\sigma}_t^2=\frac{1}{K-1}\sum_{k=1}^{K}(x_t^{(k)}-\hat{\mu}_t)^2
$$

Empirical prediction intervals are estimated from diffusion quantiles.

For a nominal 90% interval:

$$
[L_t,U_t]=[Q_{0.05},Q_{0.95}].
$$

The dedicated uncertainty experiment generates **20 stochastic diffusion reconstructions per selected missing segment**.

A reproducible stratified subset is used:

```
500 cases × 0.25 s
500 cases × 0.50 s
500 cases × 1.00 s
500 cases × 2.00 s
=
2,000 uncertainty cases

```

---

## 6. PCE and Sobol Sensitivity Analysis


The project additionally performs a global uncertainty/sensitivity study.

The uncertain input vector is:

$$
X=[G,P,N,B],
$$

where:

| Symbol | Factor |
|---|---|
| (G)            | Gap duration                         |
| (P)            | Relative gap position                |
| (N)            | Measurement-noise standard deviation |
| (B)            | Baseline-wander amplitude            |

The final UQ experiment uses:

```
256 UQ design points
8 held-out Fold-10 ECG records
10 diffusion samples per record/design point

```

The factors are sampled independently over controlled uniform ranges.

### Outputs of interest


Scalar quantities of interest include:

```
RMSE
Correlation
CRPS
PICP90
MPIW90
Mean diffusion sample standard deviation

```

### Polynomial Chaos Expansion


Because repeated diffusion inference is expensive, a Polynomial Chaos Expansion is fitted to the input-output relationship:

$$
Y\approx\hat{Y}=\sum_{\alpha}c_\alpha\Psi_\alpha(X).
$$

Because the uncertain inputs are modelled as independent uniform variables, the implementation uses **Legendre polynomial basis functions**.

The final analysis uses a **total-degree 2 PCE**.

With four uncertain factors, a degree-2 expansion contains:

$$
\binom{4+2}{2}=15
$$

basis terms.

PCE quality is evaluated using cross-validation before Sobol indices are interpreted.

### Sobol sensitivity indices


For model output $Y$, first-order Sobol index:

$$
S_i=\frac{V_i}{V$Y$}
$$

measures the fraction of output variance attributable to factor $X_i$ independently.

The total-effect index $S_{T_i}$ includes both its direct contribution and all interactions involving that factor.

The difference:

$$
S_{T_i}-S_i
$$

is therefore informative about interaction effects.

---

## 7. Paired Statistical Analysis


The final U-Net and diffusion predictions are compared on exactly matched Fold-10 test gaps.

For error metrics such as RMSE, diffusion advantage is defined as:

$$
A_i=RMSE_{\mathrm{U-Net},i}-RMSE_{\mathrm{Diffusion},i}.
$$

Therefore:

```
A > 0 → diffusion better
A < 0 → U-Net better

```

For higher-is-better metrics such as correlation:

$$
A_i=Correlation_{\mathrm{Diffusion},i}-Correlation_{\mathrm{U-Net},i}.
$$

The analysis includes:

- paired bootstrap 95% confidence intervals;
- Wilcoxon signed-rank test;
- Holm-Bonferroni correction;
- matched-pairs rank-biserial effect size;
- paired win rates.

Because the paired sample size is large, effect sizes and confidence intervals are considered more informative than p-values alone.

---

# Repository Structure


A curated repository should resemble:

```
ecg-diffusion-imputation/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md
│
├── src/
│   ├── autoencoder.py
│   ├── data.py
│   ├── diffusion.py
│   ├── diffusion_dataset.py
│   ├── interpolation.py
│   ├── mask_manifest.py
│   ├── masking.py
│   ├── metrics.py
│   ├── ptbxl_dataset.py
│   └── unet_autoencoder.py
│
├── experiments/
│   ├── analyze_uq_pce.py
│   ├── compare_unet_diffusion_stats.py
│   ├── compute_train_stats.py
│   ├── create_test_mask_manifest.py
│   ├── evaluate_autoencoder.py
│   ├── evaluate_diffusion.py
│   ├── evaluate_fixed_manifest.py
│   ├── evaluate_uncertainty.py
│   ├── evaluate_unet.py
│   ├── interpolation_baseline.py
│   ├── run_uq_design.py
│   ├── train_autoencoder.py
│   ├── train_diffusion.py
│   ├── train_unet.py
│   ├── visualize_diffusion.py
│   ├── visualize_reconstructions.py
│   └── visualize_unet.py
│
└── results/
    ├── final_model_comparison.csv
    ├── fixed_test_summary.csv
    ├── diffusion_test_summary.csv
    ├── fold10_mask_manifest.csv
    ├── uncertainty_summary.csv
    ├── uncertainty_calibration.csv
    ├── uq_pce_validation.csv
    ├── uq_sobol_indices.csv
    ├── paired_model_statistics.csv
    └── selected figures

```

Raw PTB-XL data, large intermediate result files, Python cache files and model checkpoints should normally remain outside Git history.

---

# Installation


## 1. Clone the repository


```
git clone https://github.com/mernan99/ecg-diffusion-imputation.git
cd ecg-diffusion-imputation
```

## 2. Create a virtual environment


Windows PowerShell:

```
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```
python -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies


If a `requirements.txt` file is provided:

```
pip install -r requirements.txt
```

Core dependencies include:

```
numpy
pandas
scipy
matplotlib
torch

```

CUDA-enabled PyTorch is recommended for diffusion training and inference.

---

# Dataset Setup


Download PTB-XL version 1.0.3.

Place the extracted folder at:

```
data/
└── ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/

```

The following file must exist:

```
data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv

```

and the 100 Hz waveform records should be available under:

```
records100/

```

---

# Running the Project


Commands below assume the working directory is the repository root:

```
C:\Users\<USER>\ecg-diffusion-imputation

```

Do not run commands such as:

```
python experiments\train_unet.py
```

while already inside the `experiments/` directory, because this would resolve to:

```
experiments\experiments\train_unet.py

```

Instead either return to the root directory or use:

```
python train_unet.py
```

when inside `experiments`.

## Step 1 — Compute training normalisation statistics


```
python experiments\compute_train_stats.py
```

Output:

```
results/train_stats.json

```

Statistics are calculated from PTB-XL folds 1–8 only.

## Step 2 — Run interpolation baselines


```
python experiments\interpolation_baseline.py
```

Typical outputs include:

```
results/interpolation_results.csv
results/interpolation_summary.csv

```

The full raw result CSV may be excluded from Git because of its size.

## Step 3 — Train the basic autoencoder


```
python experiments\train_autoencoder.py
```

Outputs:

```
results/autoencoder_best.pt
results/autoencoder_history.csv

```

## Step 4 — Evaluate the basic autoencoder


```
python experiments\evaluate_autoencoder.py
```

Outputs include:

```
results/autoencoder_test_results.csv
results/autoencoder_test_summary.csv

```

## Step 5 — Train the morphology-aware U-Net


```
python experiments\train_unet.py
```

Outputs include:

```
results/unet_autoencoder_best.pt

```

## Step 6 — Create the permanent Fold-10 test masks


```
python experiments\create_test_mask_manifest.py
```

Output:

```
results/fold10_mask_manifest.csv

```

This file should be preserved once created. Do not regenerate it with a different seed after reporting final results.

## Step 7 — Evaluate deterministic baselines on fixed masks


```
python experiments\evaluate_fixed_manifest.py
```

Outputs:

```
results/fixed_test_results.csv
results/fixed_test_summary.csv

```

## Step 8 — Train the conditional diffusion model


```
python experiments\train_diffusion.py
```

Default training configuration:

```
Training folds:       1–8
Validation fold:      9
Lead:                 II
Epochs:               30
Batch size:           64
Diffusion steps:      100
Learning rate:        2e-4
EMA decay:            0.995

```

Outputs:

```
results/diffusion_best.pt
results/diffusion_history.csv

```

The training loss is **noise-prediction MSE**, not ECG reconstruction RMSE. It should therefore not be compared directly with the final reconstruction metrics.

## Step 9 — Quick diffusion evaluation


```
python experiments\evaluate_diffusion.py --max-rows 500
```

This provides a rapid sampling sanity check.

## Step 10 — Full Fold-10 diffusion evaluation


The final analysis uses a five-sample diffusion mean:

```
python experiments\evaluate_diffusion.py --samples-per-input 5
```

Outputs:

```
results/diffusion_test_results.csv
results/diffusion_test_summary.csv
results/final_model_comparison.csv

```

## Step 11 — Visualise stochastic reconstructions


For a 1-second missing region:

```
python experiments\visualize_diffusion.py --gap-seconds 1 --samples 20
```

For a 2-second region:

```
python experiments\visualize_diffusion.py --gap-seconds 2 --samples 20
```

The figure shows ground truth, observed ECG, U-Net reconstruction, diffusion predictive mean, diffusion 5–95% interval, and the missing region.

## Step 12 — Predictive uncertainty analysis


```
python experiments\evaluate_uncertainty.py
```

Default uncertainty experiment:

```
500 fixed test cases per gap
4 gap lengths
2,000 total cases
20 diffusion samples per case

```

Outputs include:

```
results/uncertainty_subset_manifest.csv
results/uncertainty_results.csv
results/uncertainty_summary.csv
results/uncertainty_calibration.csv
results/uncertainty_coverage_plot.png
results/uncertainty_width_by_gap.png

```

## Step 13 — Global UQ design


Final UQ configuration:

```
python experiments\run_uq_design.py --design-points 256 --records 8 --diffusion-samples 10
```

Outputs:

```
results/uq_design_results.csv
results/uq_config.json
results/uq_record_manifest.csv

```

## Step 14 — Fit the PCE and compute Sobol indices


```
python experiments\analyze_uq_pce.py --degree 2
```

Outputs include:

```
results/uq_pce_validation.csv
results/uq_sobol_indices.csv
results/uq_pce_coefficients.csv
results/uq_sobol_*.png

```

## Step 15 — Paired U-Net vs diffusion statistics


```
python experiments\compare_unet_diffusion_stats.py
```

Outputs:

```
results/paired_unet_diffusion_results.csv
results/paired_model_statistics.csv
results/paired_rmse_win_rate.png
results/paired_rmse_advantage.png
results/paired_correlation_advantage.png

```

---

# Evaluation Metrics


## Mean Absolute Error


$$
MAE=\frac{1}{N}\sum_{i=1}^{N}|y_i-\hat{y}_i|.
$$

Lower is better.

## Root Mean Squared Error


$$
RMSE=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2}.
$$

Lower is better. RMSE places greater weight on large reconstruction errors than MAE.

## Pearson correlation


$$
r=\frac{\sum_i(y_i-\bar y)(\hat y_i-\bar{\hat y})}{\sqrt{\sum_i(y_i-\bar y)^2}\sqrt{\sum_i(\hat y_i-\bar{\hat y})^2}}.
$$

Higher is better.

## Peak-event recovery


A peak-event metric is used as an additional morphology-sensitive measure.

Because the detector used in this project is heuristic rather than a clinically validated ECG R-peak detector, these metrics should be interpreted as **peak-event recovery** rather than definitive clinical R-wave detection performance.

## Prediction Interval Coverage Probability


For lower and upper predictive bounds $L_i,U_i$:

$$
PICP=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}(L_i\le y_i\le U_i).
$$

For a nominal 90% predictive interval, well-calibrated uncertainty should ideally produce empirical coverage near 0.90.

## Mean Prediction Interval Width


$$
MPIW=\frac{1}{N}\sum_{i=1}^{N}(U_i-L_i).
$$

Narrow intervals are desirable only when adequate coverage is maintained.

## Continuous Ranked Probability Score


For a finite ensemble:

$$
CRPS=\frac{1}{K}\sum_k|x_k-y|-\frac{1}{2K^2}\sum_{k,l}|x_k-x_l|.
$$

Lower CRPS indicates better probabilistic predictions.

---

# Results


## Interpolation


Linear interpolation is considerably more stable than conventional cubic spline interpolation as gap duration increases.

Cubic splines can overshoot substantially across long ECG gaps.

PCHIP avoids extreme cubic-spline instability but behaves similarly to linear interpolation in terms of the fundamental inability to reconstruct unseen cardiac morphology.

## Basic autoencoder


The basic autoencoder improves reconstruction error relative to interpolation but often produces smooth, low-amplitude estimates for missing morphology.

Example 1-second results:

```
Basic AE RMSE        ≈ 0.136
Basic AE correlation ≈ 0.222

```

This motivated a skip-connected architecture.

## U-Net improvement


The U-Net substantially improves both amplitude reconstruction and waveform correlation.

For a 1-second gap:

```
Basic AE RMSE         ≈ 0.1365
U-Net RMSE            ≈ 0.0826

Basic AE correlation  ≈ 0.222
U-Net correlation     ≈ 0.737

```

The U-Net therefore forms the principal deterministic baseline for diffusion.

## Final diffusion comparison


| Gap | Method | MAE ↓ | RMSE ↓ | Correlation ↑ |
|---:|---|---:|---:|---:|
| 0.25 s                                 | U-Net         | 0.0388     | 0.0533     | 0.760     |
|                                        | **Diffusion** | **0.0287** | **0.0418** | **0.816** |
| 0.50 s                                 | U-Net         | 0.0440     | 0.0666     | 0.794     |
|                                        | **Diffusion** | **0.0372** | **0.0615** | **0.815** |
| 1.00 s                                 | **U-Net**     | 0.0505     | **0.0826** | **0.737** |
|                                        | Diffusion     | **0.0472** | 0.0846     | 0.714     |
| 2.00 s                                 | **U-Net**     | 0.0628     | **0.1032** | **0.629** |
|                                        | Diffusion     | **0.0622** | 0.1103     | 0.555     |

The diffusion model provides a clear advantage at 0.25 and 0.50 seconds.

At approximately 1 second the models become competitive.

At 2 seconds the deterministic U-Net becomes the stronger point estimator according to RMSE and correlation.

---

# Paired Model Comparison


The paired RMSE win-rate analysis gives approximately:

```
0.25 s → diffusion wins 76.8% of paired cases
0.50 s → diffusion wins 67.1%
1.00 s → diffusion wins 50.2%
2.00 s → diffusion wins 38.6%

```

This reveals a clear crossover.

At 1 second the paired win rate is effectively 50:50, meaning a statistically detectable mean difference should not automatically be interpreted as a large practical advantage.

This is why the project reports paired mean differences, bootstrap confidence intervals, win rates and rank-biserial effect sizes alongside significance tests.

---

# Uncertainty Quantification Results


## Predictive spread


The diffusion model produces structured uncertainty intervals rather than a constant-width band.

Uncertainty tends to widen around ambiguous missing morphology, particularly where sharp ECG events may plausibly occur.

The uncertainty band is derived entirely from repeated stochastic diffusion trajectories rather than from an externally imposed error model.

---

# Sobol Sensitivity Findings


The final UQ analysis uses:

```
256 design points
8 Fold-10 ECG records
10 diffusion samples per design
degree-2 Legendre PCE

```

## MPIW90


Approximate Sobol indices:

| Factor | First-order S1 | Total-effect ST |
|---|---:|---:|
| Measurement noise                              | **0.482** | **0.488** |
| Gap length                                     | **0.464** | **0.472** |
| Baseline wander                                | 0.035     | 0.039     |
| Gap position                                   | 0.009     | 0.011     |

This indicates that predictive interval width is primarily controlled by:

$$
\boxed{\text{measurement noise}+\text{missing-segment duration}}
$$

with comparatively little interaction.

## Sample uncertainty


A nearly identical pattern is obtained for mean diffusion sample standard deviation:

| Factor | First-order S1 | Total-effect ST |
|---|---:|---:|
| Gap length                                     | **0.482** | **0.491** |
| Measurement noise                              | **0.463** | **0.470** |
| Baseline wander                                | 0.033     | 0.038     |
| Gap position                                   | 0.010     | 0.013     |

The agreement between two independent measures of predictive spread strengthens the conclusion.

## PICP90


Calibration is influenced more strongly by several factors and interactions.

Approximate total-effect importance follows:

```
Baseline wander
Gap length
Measurement noise
Gap position

```

with gap position again contributing very little.

## Interpretation


The global UQ analysis suggests that different aspects of reconstruction behaviour are controlled by different input uncertainties:

| Model behaviour | Principal driver(s) |
|---|---|
| Point reconstruction error           | Gap length + baseline wander         |
| Waveform correlation                 | Gap length                           |
| Predictive interval width            | Gap length + measurement noise       |
| Diffusion sample variability         | Gap length + measurement noise       |
| Coverage calibration                 | Baseline wander + gap length + noise |
| Gap-position sensitivity             | Low                                  |

---

# PCE Validation


Sensitivity indices should not be interpreted without checking the accuracy of the PCE surrogate.

Approximate cross-validated $R^2$ values from the final 256-point degree-2 experiment were:

| Output | CV R² |
|---|---:|
| MPIW90             | **0.888** |
| Sample uncertainty | **0.879** |
| PICP90             | **0.729** |
| CRPS               | 0.551     |
| RMSE               | 0.508     |
| Correlation        | 0.456     |

Therefore:

- MPIW90 and sample-uncertainty Sobol indices are the most strongly supported;
- PICP90 is reasonably supported;
- RMSE, CRPS and correlation sensitivity rankings are useful but their exact numerical Sobol values should be interpreted more cautiously.

---

# Reproducibility


Several measures were taken to make the experiments reproducible.

## Fixed train/validation/test split


```
Fold 1–8 → training
Fold 9   → validation
Fold 10  → testing

```

## Training-only normalisation


Mean and standard deviation are estimated from training folds only.

## Fixed Fold-10 masks


```
fold10_mask_manifest.csv

```

stores exact missing intervals.

## Fixed random seeds


Experiment scripts use fixed seeds for mask generation, UQ subset selection, diffusion sampling, PCE cross-validation, measurement-noise templates and bootstrap statistics.

## Common random numbers in UQ


The UQ experiment deliberately reuses common stochastic noise templates across design points.

This reduces nuisance Monte Carlo variation and allows the PCE to model the influence of the experimental factors rather than random diffusion fluctuations.

---

# Generated Outputs


Typical final repository outputs include:

```
results/
├── final_model_comparison.csv
├── fixed_test_summary.csv
├── diffusion_test_summary.csv
├── fold10_mask_manifest.csv
│
├── uncertainty_summary.csv
├── uncertainty_calibration.csv
│
├── uq_pce_validation.csv
├── uq_sobol_indices.csv
│
├── paired_model_statistics.csv
│
├── diffusion_reconstruction_visualization.png
├── paired_rmse_win_rate.png
├── paired_rmse_advantage.png
├── paired_correlation_advantage.png
├── uq_sobol_mpiw90.png
├── uq_sobol_picp90.png
└── uq_sobol_sample_uncertainty.png

```

Large raw files such as:

```
interpolation_results.csv
fixed_test_results.csv
diffusion_test_results.csv
uncertainty_results.csv
uq_design_results.csv
paired_unet_diffusion_results.csv

```

may be retained locally but excluded from Git history.

---

# Files Normally Excluded from Git


The project `.gitignore` should exclude:

```
PTB-XL dataset
Python __pycache__
*.pyc files
virtual environments
large raw experiment outputs
model checkpoints
temporary quick-test results

```

Model checkpoints such as:

```
autoencoder_best.pt
unet_autoencoder_best.pt
diffusion_best.pt

```

can be distributed separately through a release, model repository, or archival service if required.

---

# Limitations


## Single ECG lead


The current model reconstructs **Lead II only**. PTB-XL contains 12-lead ECGs, meaning cross-lead information is not currently exploited.

## Synthetic missingness


Missing segments are synthetically generated contiguous blocks. Real-world sensor dropout may have irregular shapes, multiple separate gaps, intermittent missing points, or signal corruption rather than complete missingness.

## Dataset-specific evaluation


The models are trained and tested on PTB-XL. Generalisation to other ECG datasets or acquisition systems has not yet been established.

## Peak-event evaluation


The current peak-event detector is heuristic. The resulting precision, recall and F1 metrics should not be interpreted as clinically validated R-peak detection performance.

## Computational cost


Diffusion inference requires repeated neural-network evaluations across reverse-diffusion timesteps. It is therefore considerably more expensive than a single deterministic U-Net forward pass.

## Nature of uncertainty


The reported diffusion uncertainty is primarily **predictive uncertainty estimated from stochastic conditional diffusion samples**. It does not fully quantify epistemic uncertainty in the model parameters.

## PCE approximation quality


PCE surrogate quality differs by output. Sobol estimates should therefore always be interpreted alongside PCE validation metrics.

## Experimental perturbation ranges


Measurement-noise and baseline-wander ranges are controlled experimental choices. They should not automatically be interpreted as universally representative clinical ranges.

---

# Future Work


1. **Multi-lead ECG reconstruction** — extend from Lead II to all 12 leads and exploit inter-lead structure.
2. **Real missingness patterns** — evaluate on real sensor dropout or acquisition artefacts.
3. **Clinically validated morphology metrics** — incorporate validated R-peak, QRS, QT and ST-segment analysis.
4. **Downstream clinical evaluation** — test whether reconstruction preserves rhythm or diagnostic classification performance.
5. **Faster diffusion sampling** — investigate DDIM-style or other reduced-step samplers.
6. **Epistemic uncertainty** — train multiple independently initialised models and separate within-model from between-model uncertainty.
7. **Uncertainty calibration** — investigate post-hoc calibration or alternative probabilistic objectives.
8. **Alternative generative models** — compare with VAEs, masked transformers, state-space models, flow matching or autoregressive models.

---

# Research Interpretation


The project does **not** support the simplistic conclusion that diffusion is always better than a deterministic neural network.

Instead, the findings show a more interesting trade-off.

For short missing intervals, observed temporal context strongly constrains the missing segment and diffusion averaging produces highly accurate reconstructions.

As gaps become longer, the exact cardiac trajectory becomes increasingly underdetermined.

A deterministic U-Net can provide a strong point estimate, whereas diffusion represents multiple possible trajectories.

This means pointwise accuracy and generative uncertainty answer different questions:

```
U-Net:
"What single reconstruction minimises prediction error?"

Diffusion:
"What reconstructions are plausible given the observed ECG?"

```

The paired experiments show that these objectives become increasingly different as the missing interval grows.

This distinction is particularly relevant in physiological time-series reconstruction, where an apparently realistic generated waveform should not automatically be interpreted as the exact unobserved physiological truth.

---

# Suggested Final Project Summary


> This project investigates conditional diffusion modelling for missing ECG reconstruction using PTB-XL. Classical interpolation, a basic convolutional autoencoder, a morphology-aware 1D U-Net and a conditional DDPM are evaluated on fixed held-out missing segments ranging from 0.25 to 2 seconds. Diffusion provides the strongest point reconstruction for short gaps, while the U-Net becomes more accurate for longer gaps. Unlike the deterministic models, diffusion also produces multiple plausible reconstructions, enabling empirical predictive intervals and probabilistic evaluation. Polynomial Chaos Expansion and Sobol analysis are used to identify the factors driving reconstruction error and uncertainty, showing that missing-segment duration and measurement noise strongly influence predictive spread while baseline wander affects calibration and probabilistic quality. The project therefore combines deep generative modelling with explicit uncertainty quantification for physiological time-series imputation.

---

# Acknowledgements


This project uses the **PTB-XL** ECG dataset made publicly available through PhysioNet.

The project builds on established work in denoising diffusion probabilistic models, U-Net architectures, ECG time-series analysis, uncertainty quantification, Polynomial Chaos Expansion and Sobol global sensitivity analysis.

---

# Disclaimer


This project is a research and educational implementation.

It is **not a medical device** and should not be used for clinical diagnosis, treatment decisions, or replacement of original ECG measurements.