# Project Structure & Data Pipeline Documentation

This document outlines the **active file structure** and **data processing workflow** of the project as of February 2026.

## 1. Data Processing Pipeline

The 8-step process from raw videos to final analytical Excel files.

### Step 1: Video Cropping (Prep)
*   **Input**: Raw videos + `timeline_info.csv`
*   **Executable**: `all_video_crop.py`
*   **Output**: `D:\2025신윤희영상정렬\` (MP4 videos cropped by individual/timeline)
*   **Description**: Crops long videos into smaller, analyzable segments based on metadata (`timeline_info.csv`).

### Step 2: Feature Extraction (Feature & Motion)
*   **Input**: Cropped MP4 videos
*   **Executables**:
    1.  `run_all_videos.py` (calls `process_single_video.py` internally) → **OpenFace Analysis**
    2.  `compute_motion_energy.py` → **Motion Energy (ME) Calculation**
*   **Output**: OpenFace Result CSVs, `_grayscaled.csv` (ME Data)

### Step 3: Data Merge & Augmentation (Augment)
*   **Input**: OpenFace CSVs + Motion Energy CSVs
*   **Executable**: `csv_preprocess.py`
*   **Output**: **`_augmented.csv`** (The core baseline data for analysis)
*   **Description**: Merges basic features with derived variables (BBox size, Emotion labels, etc.) into a single file.

### Step 4: Global Statistics Calculation (Stats)
*   **Input**: `_augmented.csv`
*   **Executable**: `compute_global_stats_all_timeline.py`
*   **Output**: `global_stats.json` (Generated inside each timeline folder)
*   **Description**: Pre-calculates the means and standard deviations based on the entire week for Z-score normalization during synchrony analysis.

### Step 5: Synchrony Analysis (Sync Analysis)
*   **Input**: `_augmented.csv` + `global_stats.json`
*   **Executable**: `batch_visualize.py` (calls `single_visualizer.py` internally)
*   **Output**:
    *   `sync_counts.xlsx`: Simple concurrency counts
    *   **`sync_mask.xlsx`**: Detailed synchrony masks per frame/level (Source data for merging)

### Step 6: Week Integration (Week Merge)
*   **Input**: `sync_mask.xlsx` from each timeline
*   **Executable**: `week_merge.py`
*   **Output**: `sync_{semester}_{group}_{week}.xlsx` (e.g., `sync_24-2_F_W1.xlsx`)
*   **Description**: Merges scattered timeline files into a single consolidated weekly file.

### Step 7: Session Splitting (Session Splitting)
*   **Input**: Consolidated weekly files + **`session_timeline.csv`**
*   **Executable**: `split_weeks_to_sessions.py`
*   **Output**: `..._session1.xlsx`, `..._session2.xlsx`
*   **Description**: Splits weekly data into distinct sessions (Session 1, 2, etc.) based on the boundary timestamps defined in `session_timeline.csv`.

### Step 8: Master File Generation (Master Summary)
*   **Input**: Session-level Excel files
*   **Executables**: `masterfile_generator.py` → `master_postprocess.py`
*   **Output**: `master_enriched.xlsx` (Final raw data matrix)

---

## 2. Special Pipeline

### Wide Format Data Generation
An independent process to generate Wide Format data specifically requested for statistical analysis (e.g., SPSS).
*   **Executable**: `Requested_Longformat.py`
*   **Input**: `_augmented.csv`, `timeline_info.csv`, **`session_timeline.csv`**
*   **Output**: `D:\2025EE_Final_Output_Wide_Optimized\{semester}\{group}\Integrated_...xlsx`
*   **Note**: Bypasses the standard pipeline (Steps 5-8) and generates directly from augmented data (Step 3). Requires `session_timeline.csv`.

---

## 3. Core Files Summary

| Category | Filename | Role |
| :--- | :--- | :--- |
| **Metadata** | **`timeline_info.csv`** | Video segment boundaries (Project-wide basis) |
| | **`session_timeline.csv`** | Session split points (Crucial for Phase 1/2 distinction) |
| | **`config_indicators.json`** | Analysis indicators setup and threshold values |
| **Executables** | `csv_preprocess.py` | "Data Factory" - Prepares raw data for analysis |
| | `single_visualizer.py` | Core synchrony calculation logic |
| | `week_merge.py` | Merges Timelines → Weeks |
| | `split_weeks_to_sessions.py` | Splits Weeks → Sessions |
| | `Requested_Longformat.py` | Wide format converter for statistical analysis |

---

## 4. Data Directory Map

```text
D:\2025신윤희Data\MediaPipe\  (Top-level raw data root)
└── {Semester} \ {Group} \ {Week} \ {Timeline}
    ├── {Video}_augmented.csv      <-- [CORE] Step 3 output (Augmented Data)
    ├── {Video}_grayscaled.csv     <-- Motion Energy data
    ├── global_stats.json          <-- Statistics cache
    ├── sync_counts.xlsx           <-- Simple sync counts
    └── sync_mask.xlsx             <-- Synchrony map (Step 5 output)
```
