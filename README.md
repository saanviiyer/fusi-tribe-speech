# Speech in ultrasound and in silico

**Does a brain foundation model know what speech is, and does functional ultrasound respond
the way fMRI-trained tools assume?**

This project takes one functional ultrasound (fUSI) recording of auditory cortex during a
speech vs. non-speech block experiment and does two things with it:

1. **Re-analyses the real data** with an open, tested pipeline, and measures the response timing
   from the data instead of assuming the fMRI convention.
2. **Replays the same experiment in silico** through [TRIBE v2](https://github.com/facebookresearch/tribev2),
   Meta's multimodal fMRI encoding model, using synthetic speech and classic acoustic controls
   to ask what the model's "speech areas" actually respond to.

**[Explore the interactive results →](https://claude.ai/artifact/3DKQrjnk1do8h6GyYjMaRN)**
Drag the onset shift to fit the ultrasound signal, compare response shapes, and switch acoustic
controls to see what TRIBE treats as speech.

## Key findings

**1. The fUSI response is later and broader than the fMRI model assumes. The original
pipeline shifted it the wrong way.**
The analysis notebook this project started from shifted every stimulus onset one frame *earlier*
(−1.6 s). Scanning shifts from −6 to +6 s, the standard (Glover) response model fits best at
**+2.8 ± 0.6 s**. Variance explained is 0.53 there, versus 0.35 with no shift and 0.19 with the
original shift. Fitting the response shape directly, its centre of mass falls **7.3 ± 0.6 s**
after the logged onset, against 5.3 s for the fMRI model. Removing the shift raises the peak
z-score for speech from about 9 to **11.2**.

The timing holds on blocks the voxel selection never saw. Choosing voxels from one half of the
blocks and measuring timing only on the other half gives best shifts of **+2.4 to +3.4 s** for
every well-fitting ROI. In all four held-out checks, the original −1 TR shift fits worst of the
three options.

| Voxels chosen on | Timing measured on | Best shift | R² best / no shift / −1 TR |
|---|---|---|---|
| half A (500 voxels) | half B blocks | +3.4 ± 1.4 s | 0.47 / 0.26 / 0.11 |
| half A ∩ half B (262 voxels) | half B blocks | +3.2 ± 1.3 s | 0.51 / 0.26 / 0.09 |
| half A ∩ half B (262 voxels) | half A blocks | +2.4 ± 0.9 s | 0.51 / 0.42 / 0.30 |
| half B (500 voxels) | half A blocks | +5.0 ± 1.6 s | 0.09 / 0.00 / 0.00 |

The half-B ROI fits poorly because the half-B blocks drive weaker responses, even in the half-A
voxels. Its 238 voxels outside the shared set carry a start-of-run transient instead of a block
response (R² 0.23 against 0.52 for the shared voxels). Discarding the first 10 s or selecting
voxels by false-discovery rate instead of a fixed count did not change this.

![Timing comparison](results/timing_comparison.png)

**2. TRIBE v2's audio pathway responds to how speech sounds, not to whether it is intelligible.**
With the text pathway off, time-reversed speech (unintelligible, but with the same spectrum)
drives every auditory and language region *more* than real speech does (+8 to +21% across two
independent syntheses of the stimuli). Lateral
STG is the region most sensitive to destroying spectral structure: noise vocoding cuts its
response by 44%, and spectral rotation by 22% beyond a bandwidth-matched control.

![TRIBE retention](results/insilico/audio_only/retention.png)

**3. The right control changed the conclusion.** Spectral rotation seemed to produce a neat
gradient along the auditory hierarchy, with the response falling from primary auditory cortex
to STS. A bandwidth-matched low-pass control showed that most of that gradient comes from
removing energy above 4 kHz, which alone cuts TRIBE's STS and MTG responses by 30–40%. Only
the residual effect, concentrated in STG and STS, is attributable to rotation.

| Region | rotated − lowpass | reversed − intact | vocoded − intact |
|---|---|---|---|
| HG (primary auditory) | −0.07 [−0.10, −0.04] | **+0.12** [+0.10, +0.13] | −0.29 [−0.31, −0.26] |
| PT | +0.03 [−0.02, +0.09] | **+0.21** [+0.19, +0.24] | −0.04 [−0.12, +0.03] |
| STG | **−0.22** [−0.25, −0.18] | **+0.11** [+0.09, +0.13] | **−0.44** [−0.46, −0.41] |
| STS | **−0.18** [−0.26, −0.10] | **+0.13** [+0.11, +0.15] | −0.24 [−0.28, −0.21] |
| MTG | −0.05 [−0.18, +0.08] | **+0.16** [+0.12, +0.19] | −0.06 [−0.14, +0.00] |

*Change in retention (control response as a fraction of the speech response), paired over the
same seven passages; 95% bootstrap intervals over passages.*

**Robustness to the stimulus synthesis.** macOS text-to-speech is not bit-identical between runs,
so the full TRIBE experiment was repeated on a second synthesis of the same passages
(`configs/tts_variant.yaml`). The 15 paired effects correlate at r = 0.995 between the two runs;
every effect keeps its sign, and significance agrees for 14 of 15. The exception is vocoded − intact
in MTG, borderline in both. Tables: [results/robustness_tts_variant/](results/robustness_tts_variant/insilico/audio_only/).

## How it works

```
data/ (one fUSI run)                        synthetic speech (macOS TTS)
   │                                                │
   ▼                                                ▼
fusi-glm      QC · censoring · rigid      make-stimuli   same block timing as the fUSI run;
              registration · GLM                         control blocks hold lowpass /
   │                                                     rotated / vocoded / reversed /
   ▼                                                     intact / silence
fusi-timing   timing-agnostic ROI ·                │
              onset-shift scan · gamma fit          ▼
   │                                       run-tribe      TRIBE v2 → 20,484 cortical
   │                                                      vertices × 1 Hz
   │                                                │
   │                                                ▼
   │                                       analyze-tribe  retention per ROI ·
   │                                                      paired contrasts · timing
   └──────────────────────┬─────────────────────────┘
                          ▼
                       report  → results/timing_comparison.png
```

Methods, the full decision log and the limitations are in [docs/methods.md](docs/methods.md).

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
make install          # package + TRIBE v2 + pytest
make test             # 32 unit tests, no data needed (~10 s)
make all              # needs the run in data/ (see data/README.md); ~1.5 h on an M-series Mac
```

Or run single steps: `fusi-tribe fusi-glm`, `fusi-tribe fusi-timing`, `fusi-tribe make-stimuli`,
`fusi-tribe run-tribe --mode audio_only`, `fusi-tribe analyze-tribe --mode audio_only`,
`fusi-tribe report`. All settings are in [configs/default.yaml](configs/default.yaml).

Requirements: Python ≥ 3.11. Stimulus synthesis uses macOS `say`. TRIBE v2's audio pathway runs
on CUDA, on Apple Silicon (MPS, with feature extractors on CPU), or on CPU. The full text pathway
needs access to the gated [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)
and, in practice, an NVIDIA GPU; the [Colab notebook](notebooks/run_full_mode_colab.ipynb) runs it
on a free T4.

### What the tests check

The tests plant known effects in synthetic data and check that the analysis recovers them:

- an onset shift of −2, 0 or +3 s, recovered by the shift scan
- a gamma response peaking at 8 s, recovered by the grid fit
- a 1 mm translation with anisotropic voxels, recovered by registration (this guards the
  axis-order bug fixed from the original code)
- retention: a silent control gives 0 and a scaled response gives its scale, regardless of
  baseline offset
- the stimuli: rotation maps *f* → 4 kHz − *f*, low-pass and rotation share a band, the vocoder
  keeps the envelope but removes harmonics

## Limitations

- **One fUSI run.** No claims about other sessions, people or test–retest reliability.
- **Latency vs. clock.** A response that is late relative to logged onsets could reflect slow
  blood-volume dynamics *or* a late event clock. This data cannot separate the two.
- **No spatial comparison.** The fUSI volume is not registered to an atlas, so fUSI and TRIBE
  are compared on timing and selectivity only.
- **Model, not brains.** The TRIBE results describe the model's predictions for one synthetic
  voice and 14 passages. They are hypotheses about cortex, not measurements of it.
- **Original stimuli unavailable.** The in-silico runs reuse the fUSI run's timing with new
  speech; what the real "Audio only" blocks contained is unknown.

## In progress

- **Text pathway (full TRIBE).** In full mode TRIBE transcribes the audio with WhisperX and feeds
  the words to Llama 3.2. The transcripts are already in
  ([results/insilico/transcription.tsv](results/insilico/transcription.tsv)):

  | Control | Words recovered from the real passage |
  |---|---|
  | lowpass | 100% |
  | vocoded (4 channels) | 92% |
  | reversed | 10% |
  | rotated | 4% |

  So the text pathway will receive the real words for vocoded speech, but mostly noise for
  reversed and rotated speech. The prediction: adding text should raise vocoded retention in
  STS, MTG and IFG and leave reversed and rotated retention low there, with little change in HG.
  Llama 3.2 is too slow and memory-hungry for a 24 GB laptop, so this step runs on a free Colab
  GPU: [notebooks/run_full_mode_colab.ipynb](notebooks/run_full_mode_colab.ipynb)
  ([open in Colab](https://colab.research.google.com/github/saanviiyer/fusi-tribe-speech/blob/main/notebooks/run_full_mode_colab.ipynb)).

## Repository layout

```
src/fusi_tribe/
  fusi/preprocess.py   QC, registration, smoothing, motion regressors
  fusi/glm.py          first-level GLM pipeline
  fusi/timing.py       ROI selection, shift scan, gamma fit, jackknife
  insilico/stimuli.py  speech synthesis and acoustic manipulations
  insilico/tribe.py    TRIBE v2 runner (CUDA / MPS / CPU)
  insilico/analysis.py retention, paired contrasts, TRIBE timing
  report.py            cross-modality figures
  cli.py               `fusi-tribe` command
configs/   tests/   docs/   results/ (committed figures and tables)
```

## Data and attribution

The fUSI recording and the original preprocessing helpers (`fusiproc_utils3.py`, adapted in
`fusi/preprocess.py`) were provided as lab sample data and are not redistributed here. TRIBE v2
is © Meta Platforms, Inc. and used under its own licence. The speech passages in
`insilico/passages.py` were written for this project.

## License

MIT for the code written for this project; see [LICENSE](LICENSE). `src/fusi_tribe/fusi/preprocess.py`
adapts lab-provided helper code, and those portions remain subject to whatever terms the original
authors set. TRIBE v2 and its model weights are under their own licences.
