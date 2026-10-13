# Methods and decision log

## 1. fUSI run

One functional ultrasound (power Doppler) run: 117 × 56 × 50 voxels at 0.513 mm isotropic,
254 volumes, 14 blocks alternating "Speech" and "Audio only" (10–19 s each). Frame times come
from per-volume timestamps in `logs.parquet` (median interval 1.602 s, range 1.600–1.622 s);
the NIfTI header's TR (1.0 s) is wrong and is not used.

**Preprocessing** (`fusi/preprocess.py`, `fusi/glm.py`)

1. Censor global-signal spikes: detrended global signal > median + 4 × MAD (5 frames).
2. Rigid registration of every remaining frame to the median volume (SimpleITK, correlation
   metric, 3-level pyramid), masked to voxels finite in every frame.
3. Framewise displacement (Power et al., 50 mm radius); censor FD > 0.5 mm (none exceeded; max 0.37 mm).
4. Temporal boxcar smoothing (3 frames), Friston-24 motion regressors, cosine drift (0.01 Hz).
5. nilearn `FirstLevelModel`: Glover + temporal derivative HRF, AR(1) noise, 2 mm spatial
   smoothing, censored frames excluded with `sample_masks`.

**Timing analysis** (`fusi/timing.py`)

- *ROI selection without timing assumptions.* Finite-impulse-response (FIR) regressors — the
  block boxcar delayed by 0…14 TR — for half the blocks (alternating occurrences of each
  condition). A voxelwise F-test on those columns picks the 500 most responsive voxels. Nothing
  in that test prefers one latency over another, so the ROI cannot bias the timing estimates.
- *Onset-shift scan.* Variance explained (beyond nuisance) by Glover-convolved blocks, with all
  onsets shifted from −6 to +6 s in 0.2 s steps.
- *Gamma fit.* Gamma-shaped impulse response with free time-to-peak (1–14 s) and width
  (2–16 s), grid search. Reported as the response **centroid** (centre of mass), which block
  designs constrain much better than the peak.
- *Uncertainty.* Leave-one-block-out jackknife (each block plus 12 s after it removed).
- *Cross-validation.* Voxels are chosen on one half of the blocks and the shift scan is run only
  on frames outside that half's block windows (shared baseline kept), in both directions, for
  each half's ROI and for the voxels both halves select.

An unconstrained FIR deconvolution was tried first and abandoned: with block designs, adjacent
FIR lags are nearly collinear and the estimates zig-zag; the two halves of the data disagreed.

**What cannot be separated.** A late response relative to logged onsets can mean slow
hemodynamics *or* a late event clock (e.g. audio output latency). Only the latency relative to
logged onsets is measured.

## 2. In-silico experiment (TRIBE v2)

[TRIBE v2](https://github.com/facebookresearch/tribev2) predicts fMRI on the fsaverage5 cortical
surface (20,484 vertices, 1 Hz) from audio, transcribed text and video, using Wav2Vec-BERT 2.0
(audio), Llama 3.2-3B (text) and V-JEPA 2 (video) features. It is trained with fMRI sampled 5 s
after each stimulus second (`data.neuro.offset: 5.0`), so prediction row *t* is fMRI at *t* + 5 s;
the analysis puts rows back on the stimulus clock.

**Stimuli** (`insilico/stimuli.py`). Fourteen original ~65-word narrative passages
(`insilico/passages.py`), synthesised with macOS `say` (voice Samantha, 140 wpm, 16 kHz) and
placed on the exact block timing of the fUSI run. Speech blocks always hold intact speech; the
condition sets what fills the Audio-only blocks (one fixed set of seven passages, so conditions
are paired over passages):

| Condition | Manipulation | What it removes |
|---|---|---|
| intact | none | — (null check) |
| lowpass | 4 kHz low-pass | > 4 kHz energy |
| rotated | spectral inversion around 2 kHz (Blesser 1972), same low-pass | intelligibility, spectral shape |
| vocoded | 4-channel noise vocoder (Shannon et al. 1995) | spectral fine structure, pitch |
| reversed | time reversal | intelligibility, temporal order |
| silence | none | everything |

All segments are RMS-matched with 50 ms ramps.

**Retention** (`insilico/analysis.py`). Per block and ROI (Destrieux atlas regions), amplitude =
mean prediction from onset + 4 s to offset + 3 s, minus the mean over silent frames ≥ 8 s after
any block. Retention = mean control amplitude / mean speech amplitude within the same run.
Paired contrasts subtract a matched reference passage-by-passage (rotated − lowpass,
reversed − intact, vocoded − intact). Bootstrap intervals resample passages. TRIBE is
deterministic, so this is the only variability there is; conclusions are about this stimulus
set and this voice, not about brains. ROIs whose speech amplitude is < 0.15 (IFG in audio-only
mode; V1) are not interpreted.

## 3. Decision log

Things that changed the analysis, in the order they came up.

| # | Issue | Evidence | Resolution |
|---|---|---|---|
| 1 | Original notebook shifts onsets by −1 TR | Shift scan: R² 0.19 at −1.6 s vs 0.35 at 0 s and 0.53 at the best shift | Default `onset_shift_s: 0`; peak z for speech rises from ~9 to 11.2 |
| 2 | SimpleITK axis order in registration | `GetImageFromArray` reverses axes; spacing was passed un-reversed | Fixed; test with anisotropic voxels (`tests/test_preprocess.py`). No effect on this isotropic run |
| 3 | FIR deconvolution unusable | Halves disagree, zig-zag estimates | Replaced by gamma fit + shift scan |
| 4 | Rotated speech is band-limited to 4 kHz | Spectrogram check | Added the `lowpass` control. Low-pass alone cuts TRIBE's STS/MTG response by 30–40%; most of the apparent rotation "hierarchy gradient" was bandwidth |
| 5 | GLM-beta retention biased for silence (−0.1 to −0.2) | Silence blocks are physically identical to the gaps | Model-free window amplitudes; silence retention now 0 ± 0.07 |
| 6 | TRIBE extractors hard-code CUDA | Crash on Apple Silicon | Config override; extractors on CPU, brain model on MPS |
| 7 | macOS `say` is not bit-identical across runs | Checksums differ between syntheses | Passage audio cached; checksums in manifests and prediction files. TRIBE rerun on a second synthesis: paired effects correlate r = 0.995, all signs agree, significance agrees 14/15 |
| 8 | Spurious numpy matmul warnings | Raised on random well-conditioned matrices too; results equal `einsum` | Filtered in pytest config only |
| 9 | ROI chosen from half B fits poorly (R² 0.15 vs 0.57) | Shared voxels fit as well as ROI A (0.52); the other 238 carry a start-of-run transient and half-B blocks respond weakly even in ROI A. Dropping the first 10 s or FDR selection did not help | Kept the fixed-count ROI; added held-out timing checks: best shift +2.4 to +3.4 s for every well-fitting ROI, −1 TR worst in all four |
| 10 | TRIBE's transcription fails on CPU | `uvx whisperx --compute_type float16` is rejected by the CPU backend; `uvx` was not installed | `uv` added to the environment; `insilico/tribe.py` runs the same WhisperX call with int8 off CUDA |
| 11 | Llama 3.2 text features unusably slow on a 24 GB Mac | CPU: 4–12 min per batch; MPS: 45–75 s per batch with the system swapping (~60 h for six runs) | Full mode moved to a Colab T4 (`notebooks/run_full_mode_colab.ipynb`). On GPUs under 20 GB, and on MPS, Llama runs in float16 (neuralset's own `accelerate` precision); TRIBE's released config computes text features in float32 on CUDA |

## 4. Known limitations

- One fUSI run from one session: no claims about populations or test–retest reliability.
- The fUSI field of view is a small volume that is not registered to any atlas, so fUSI and
  TRIBE are compared on timing and selectivity, not spatially.
- The real stimuli for the fUSI run were not available; the in-silico runs reuse its timing with
  synthetic speech, and the exact content of the "Audio only" condition is unknown.
- AR(1) noise with 3-frame temporal smoothing likely understates temporal autocorrelation, so
  the GLM z-values are probably inflated. Treat them as descriptive.
- TRIBE predicts an *average subject*; retention describes the model, not people.
