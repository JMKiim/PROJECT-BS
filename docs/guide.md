# Technical Guide

[Overview](../README.md) · [Inputs](#inputs) · [Execution](#execution) · [Methods](#methods) · [Research](#research)

## Configuration

Paths are configured in `configs/paths.local.json`, relative to the repository root. Use `bs --config "<config.json>" <stage>` or `BS_CONFIG` to select another file. Executable names resolve through `PATH`; `wavelet_toolbox_dir` points to ASToolbox2018's `Functions` directory.

## Inputs

### Video and event features

```text
raw_video_dir/{semester}/{group}/{group}_{week}.mp4
cropped_video_dir/{semester}/{group}/{group}_{week}/{group}_{week}_T{n}_P{p}.mp4
features_dir/{semester}/{group}/{week}/T{n}/{group}_{week}_T{n}_P{p}.csv
```

`{n}` and `{p}` are timeline and participant numbers. Motion and augmented files append `_grayscaled` and `_augmented` to the CSV stem. Keep participant numbers consistent across clips. Split versions append `(1)`, `(2)`, etc. to the week identifier.

Event features use aligned 15 fps rows. OpenFace inputs need 68 `x_i`/`y_i` landmark pairs, pose rotations, AU presence columns, and optional `success`; motion inputs need `frame`, `timestamp`, and `ME`. Both files must describe the same frames.

Timeline metadata uses these UTF-8 CSV headers:

| Setting | Columns |
| :--- | :--- |
| `timeline_metadata` | `학기,그룹명,주차,파일버전,타임라인인덱스,시작시간,종료시간,인원수` |
| `session_metadata` | `학기,그룹명,주차,파일버전,분기시간,세션` |

Times use `hh:mm:ss`; `없음` denotes no file-version suffix. Cropping supports predefined 3–7-person screen layouts. Check the layout and timeline metadata before processing.

### Paired Head and MEA signals

```text
head_dir/{group}/{group}_P{p}_S{s}_slim.csv
mea_dir/{group}/{group}_P{p}_S{s}_mea.csv
```

Paired analysis uses P1/P2 and 25 Hz inputs. Head signals are extracted separately. MEA clips end in `_P{p}_S{s}.mp4`.

| Input | Columns |
| :--- | :--- |
| Head | `frame`, `timestamp`, `success`, `mp_confidence`, `pose_Rx/Ry/Rz`, `pose_Tx/Ty/Tz`, `face_distance` |
| MEA | `frame`, `timestamp`, `mea_z` |

MEA frames run from `0` to `N-1`, with `timestamp = frame/25`. P1/P2 Head time axes must match; Head columns cannot be entirely missing. MEA must be finite and nonconstant.

Set expected lengths in the JSON file selected by `wavelet_manifest`. This example uses fictional values:

```json
{
  "groups": ["demo"],
  "sessions": ["S1", "S2"],
  "sample_counts": {"demo": {"S1": 250, "S2": 500}},
  "distinct_sessions": [["demo", "S1", "S2"]]
}
```

Use each recording's actual length. `distinct_sessions` identifies pairs that must not contain identical signals. Group/session identifiers start with a letter and contain letters, digits, or underscores.

Optional `roi_config` rectangles exclude video overlays. For example, `{"_default": {"exclude": [[0, 0, 100, 20]]}}` excludes a 100×20-pixel region. Coordinates are `[x1, y1, x2, y2]`, with the right and bottom edges excluded.

### Assessment workbooks

Enrichment expects semester-specific `*_평가_STEP` sheets; normalization uses `*_평가_WEEK`. Required keys are `TEAM`, `WEEK`, score columns, and `STEP` for session scores. Adapt these importers for other assessment formats.

## Execution

### Event analysis

Run commands in order, starting from your available inputs:

| Command | Output |
| :--- | :--- |
| `bs crop` | Participant clips |
| `bs openface`, then `bs motion` | Facial features and motion CSVs |
| `bs augment`, then `bs statistics` | Combined features and reference statistics |
| `bs events` | `sync_counts.xlsx`, `sync_mask.xlsx` |
| `bs weeks`, then `bs sessions` | Weekly and session workbooks |
| `bs summary`, then `bs enrich` | Synchrony summary and assessment-enriched workbook |

Complete crops and extracted CSVs are skipped; augmentation, statistics, and reporting regenerate outputs. Event analysis skips a timeline only when both workbooks exist. Use `--force` to recalculate, then rerun downstream stages after changing indicators or statistics.

For a video preview, append `--force --video --start 0 --end 30` to the [timeline command](../README.md#minimal-example). Rendering requires matching tracked videos. Time bounds affect the preview only; workbook statistics cover the full timeline.

### Paired analysis

```powershell
bs mea
bs align --validate-only
bs align
bs wavelet --check
bs wavelet
```

Skip `mea` if `mea_z` inputs already exist. Alignment validates lengths, time axes, and duplicate-session rules. Before rebuilding, move the existing prepared directory or configure a new one. `wavelet --check` verifies inputs, MATLAB, and toolbox paths without running the calculation.

`bs normalize` scales session counts to 50 minutes and joins weekly scores; `normalization_input` selects its source. `bs signals` exports participant raw/Z-score tables, subject to the limits below.

## Methods

### Event synchrony

The twelve default indicators cover head rotation and velocity, face area and distance, motion energy and its log transform, and positive/negative AU labels.

| Measurement | Definition |
| :--- | :--- |
| Rotation / velocity | `pose_Rx/Ry/Rz`; first difference divided by 1/15 second |
| Face area / distance | Bounding rectangle of 68 landmarks; `pose_Tz` |
| Motion | Mean absolute grayscale frame difference; `log(1 + ME)` |
| Valence | First matching AU rule; happiness and surprise map to positive |

Reference means and population SDs use successfully tracked frames within each participant/recording-week folder. Split versions have separate statistics. Zero SD becomes one; fewer than two valid values use mean zero and SD one.

Threshold activation uses a strict comparison, optionally after Z-standardization; the default is 1 SD. Velocity events require a positive crossing followed by a negative crossing within 0.6 seconds and mark the cycle's completion frame.

Activations require successful tracking and extend forward over the configured window. At 15 fps, a one-second window includes `t-15` through `t`. `any` counts the union of positive/negative activations; `same` uses the larger directional count; `positive` and `negative` count only that direction.

`RawMask` stores signed activations, `PerFrameLevels` encodes participant counts, and `LevelCounts` sums them. Weekly merging concatenates masks without extending windows across clips. Session summaries retain levels 0–5, at-least-half and at-least-two counts, with weights `k`, `k²`, or `10^(k-1)`.

### Paired preprocessing and coherence

Head gaps use linear interpolation with nearest-value endpoints. Tracking success does not mask additional rows; no further Head smoothing or standardization is applied. MEA processing uses:

1. Sum of grayscale pixel differences greater than 12, excluding ROI rectangles.
2. Outlier detection beyond 10 SD, excluding the initial zero; expansion by about 0.5 seconds on each side and interpolation.
3. A centered 13-frame moving average, then participant/session Z-standardization.

MEA joins Head by row after length checks; Head supplies the output time axis.

| Wavelet parameter | Value |
| :--- | :--- |
| Sampling | 25 Hz → 10 Hz |
| Unit / band | Whole phase / 0.22–0.49 Hz |
| Wavelet | AWCOG, Morlet, beta 6 |
| Scale / smoothing | `dj = 1/30`, `wt_size = 5`, `ws_size = 10` |
| Surrogates / boundary | `n_sur = 0`; exclude `periods > coi` |

Resampling uses normalized endpoint grids and `floor(N × 10/25)` samples. Frequency aggregation takes the magnitude of mean **complex** coherence. Time points require all selected scales to be valid; their mean gives the phase score.

Pitch/Yaw/Roll and four-signal averages combine raw scores. Fisher Z follows averaging, clipped at `1 - 10^-6`.

### Limits

- Event rows must be aligned and equal-length. Activations can persist through failed tracking until the window expires; only an event's completion frame is tracking-masked.
- Session summaries support at most five participants. Metadata durations determine session boundaries; check missing clips and the `UsedTimelines` sheet.
- The `2023-2` adapter uses split times directly on the merged axis and assumes four members in summaries. Other cohorts map splits through timeline metadata.
- `bs signals` uses sample SD, includes failed-tracking values, and applies only the first split to the concatenated signal. Review it before using split recordings or multiple boundaries.
- Head interpolation has no maximum-gap limit. AU labels and synchrony scores measure operational behavior, not learning outcomes or emotional states directly.

## Research

**A Pilot Study on Collaborative Learning Performance Assessment Using Computer Vision**

Jaemyeong Kim, Youngwug Cho, Yoonhee Shin, and Kwanguk (Kenny) Kim — HCI Korea, Best Paper Award.

The pilot measured forward leaning from MediaPipe FaceMesh face width. The README uses its poster's webcam diagram and the paper's Figure 1. The current package extends this work with different signals and scoring rules.

Paired Head/MEA processing was developed through educational-technology collaboration. Its wavelet band, boundary handling, aggregation, and Fisher Z settings align with collaborator-led fNIRS analysis. fNIRS code and data are outside this repository.

AWCOG and MeanGAIN are from ASToolbox2018 by L. Aguiar-Conraria and M. J. Soares. Preserve external software licenses and attribution. The startup check verifies MeanGAIN's `abs(mean(...))` behavior with NaN propagation.

## Development

Run `python -m unittest discover -s tests -v` and `python -m pip check`. Tests use synthetic inputs. `requirements.lock.txt` records the Python environment used for these checks, not historical study dependencies.
