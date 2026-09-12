"""Validate and align paired Head/MEA signals using a private session manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


from bs.settings import path, load_manifest

BASE_DIR = path('prepared_dir').parent
HEAD_DIR = path('head_dir')
MEA_DIR = path('mea_dir')
OUTPUT_DIR = path('prepared_dir')
SOURCE_HZ = 25
GROUPS, PARTICIPANTS, SESSIONS = [], ['P1', 'P2'], []
SAMPLE_COUNTS = {}

def configure_manifest():
    global GROUPS, SESSIONS, SAMPLE_COUNTS
    manifest = load_manifest()
    GROUPS, SESSIONS = manifest['groups'], manifest['sessions']
    SAMPLE_COUNTS = manifest['sample_counts']

HEAD_SIGNAL_COLUMNS = [
    "pose_Rx",
    "pose_Ry",
    "pose_Rz",
    "pose_Tx",
    "pose_Ty",
    "pose_Tz",
    "face_distance",
]
HEAD_REQUIRED_COLUMNS = {
    "frame",
    "timestamp",
    "success",
    "mp_confidence",
    *HEAD_SIGNAL_COLUMNS,
}
MEA_REQUIRED_COLUMNS = {"frame", "timestamp", "mea_z"}


def expected_rows(group: str, session: str) -> int:
    return SAMPLE_COUNTS[group][session]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and jointly preprocess all Head and MEA files."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run all source checks without creating preprocessed files.",
    )
    return parser.parse_args()


def source_paths(group: str, participant: str, session: str) -> tuple[Path, Path]:
    stem = f"{group}_{participant}_{session}"
    head_path = HEAD_DIR / group / f"{stem}_slim.csv"
    mea_path = MEA_DIR / group / f"{stem}_mea.csv"
    return head_path, mea_path


def output_path(group: str, participant: str, session: str) -> Path:
    stem = f"{group}_{participant}_{session}"
    return OUTPUT_DIR / group / f"{stem}_slim.csv"


def numeric_array(series: pd.Series, *, label: str) -> np.ndarray:
    try:
        values = pd.to_numeric(series, errors="raise").to_numpy(dtype="float64")
    except Exception as exc:
        raise ValueError(f"{label} is not numeric: {exc}") from exc
    return values


def validate_head(path: Path, expected: int) -> tuple[pd.DataFrame, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"missing Head file: {path}")

    columns = list(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)
    missing = sorted(HEAD_REQUIRED_COLUMNS - set(columns))
    if missing:
        raise ValueError(f"missing Head columns {missing}: {path}")
    if "mea_z" in columns:
        raise ValueError(f"raw Head file unexpectedly contains mea_z: {path}")

    check_columns = ["frame", "timestamp", "success", *HEAD_SIGNAL_COLUMNS]
    data = pd.read_csv(path, usecols=check_columns, encoding="utf-8-sig")
    if len(data) != expected:
        raise ValueError(f"Head rows {len(data):,} != {expected:,}: {path}")

    frame = numeric_array(data["frame"], label=f"{path} frame")
    timestamp = numeric_array(data["timestamp"], label=f"{path} timestamp")
    success = numeric_array(data["success"], label=f"{path} success")

    if not np.isfinite(frame).all() or not np.equal(frame, np.floor(frame)).all():
        raise ValueError(f"Head frame has non-finite or non-integer values: {path}")
    if len(np.unique(frame)) != expected or np.any(np.diff(frame) <= 0):
        raise ValueError(f"Head frame is duplicated or non-increasing: {path}")
    if not np.isfinite(timestamp).all() or np.any(np.diff(timestamp) <= 0):
        raise ValueError(f"Head timestamp is non-finite or non-increasing: {path}")
    if not np.isin(success, [0, 1]).all():
        raise ValueError(f"Head success contains values other than 0/1: {path}")

    signal_nan_count = 0
    for column in HEAD_SIGNAL_COLUMNS:
        values = numeric_array(data[column], label=f"{path} {column}")
        signal_nan_count += int(np.isnan(values).sum())
        if not np.isfinite(values).any():
            raise ValueError(f"Head signal is entirely missing ({column}): {path}")
        if np.isinf(values).any():
            raise ValueError(f"Head signal contains Inf ({column}): {path}")

    axes = pd.DataFrame(
        {"frame": frame.astype("int64"), "timestamp": timestamp}
    )
    stats = {
        "head_rows": int(len(data)),
        "head_success_rate": float(np.mean(success == 1)),
        "head_signal_nan_before": signal_nan_count,
        "head_columns": columns,
    }
    return axes, stats


def validate_mea(path: Path, expected: int) -> tuple[pd.DataFrame, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"missing MEA file: {path}")

    columns = set(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)
    missing = sorted(MEA_REQUIRED_COLUMNS - columns)
    if missing:
        raise ValueError(f"missing MEA columns {missing}: {path}")

    data = pd.read_csv(
        path, usecols=["frame", "timestamp", "mea_z"], encoding="utf-8-sig"
    )
    if len(data) != expected:
        raise ValueError(f"MEA rows {len(data):,} != {expected:,}: {path}")

    frame = numeric_array(data["frame"], label=f"{path} frame")
    timestamp = numeric_array(data["timestamp"], label=f"{path} timestamp")
    mea_z = numeric_array(data["mea_z"], label=f"{path} mea_z")
    expected_frame = np.arange(expected, dtype="float64")
    expected_timestamp = expected_frame / SOURCE_HZ

    if not np.array_equal(frame, expected_frame):
        raise ValueError(f"MEA frame must be the local sequence 0..N-1: {path}")
    if not np.allclose(timestamp, expected_timestamp, rtol=0, atol=1e-10):
        raise ValueError(f"MEA timestamp is not an exact local 25 Hz axis: {path}")
    if not np.isfinite(mea_z).all():
        raise ValueError(f"MEA mea_z contains NaN or Inf: {path}")
    if np.ptp(mea_z) == 0:
        raise ValueError(f"MEA mea_z is constant: {path}")

    stats = {
        "mea_rows": int(len(data)),
        "mea_z_mean": float(np.mean(mea_z)),
        "mea_z_std": float(np.std(mea_z)),
    }
    return data, stats


def validate_distinct_sessions() -> None:
    """Check explicitly listed session pairs for duplicated source signals."""
    for item in load_manifest().get('distinct_sessions', []):
        group, left_session, right_session = item
        for participant in PARTICIPANTS:
            h1, m1 = source_paths(group, participant, left_session)
            h2, m2 = source_paths(group, participant, right_session)
            for left, right, columns in [(h1, h2, HEAD_SIGNAL_COLUMNS), (m1, m2, ['mea_z'])]:
                if left.is_file() and right.is_file():
                    a = pd.read_csv(left, usecols=columns).to_numpy()
                    b = pd.read_csv(right, usecols=columns).to_numpy()
                    if np.array_equal(a, b, equal_nan=True):
                        raise ValueError(f"Duplicate session signals: {left.name}, {right.name}")


def preflight() -> list[dict]:
    """Validate all source pairs before any output directory is created."""
    errors: list[str] = []
    records: list[dict] = []

    for group in GROUPS:
        for session in SESSIONS:
            expected = expected_rows(group, session)
            pair_axes: dict[str, pd.DataFrame] = {}

            for participant in PARTICIPANTS:
                stem = f"{group}_{participant}_{session}"
                head_path, mea_path = source_paths(group, participant, session)
                try:
                    axes, head_stats = validate_head(head_path, expected)
                    _, mea_stats = validate_mea(mea_path, expected)
                    pair_axes[participant] = axes
                    records.append(
                        {
                            "group": group,
                            "participant": participant,
                            "session": session,
                            "rows": expected,
                            "seconds": expected / SOURCE_HZ,
                            **head_stats,
                            **mea_stats,
                        }
                    )
                except Exception as exc:
                    errors.append(f"{stem}: {exc}")

            if set(pair_axes) == set(PARTICIPANTS):
                p1 = pair_axes["P1"]
                p2 = pair_axes["P2"]
                if not np.array_equal(p1["frame"].to_numpy(), p2["frame"].to_numpy()):
                    errors.append(f"{group} {session}: Head P1/P2 frame mismatch")
                if not np.allclose(
                    p1["timestamp"].to_numpy(),
                    p2["timestamp"].to_numpy(),
                    rtol=0,
                    atol=1e-9,
                ):
                    errors.append(f"{group} {session}: Head P1/P2 timestamp mismatch")

    try:
        validate_distinct_sessions()
    except Exception as exc:
        errors.append(str(exc))

    expected_count = len(GROUPS) * len(PARTICIPANTS) * len(SESSIONS)
    if len(records) != expected_count:
        errors.append(f"validated pair count {len(records)} != {expected_count}")

    if errors:
        raise RuntimeError(
            "Unified preprocessing preflight failed. No output was written.\n  - "
            + "\n  - ".join(errors)
        )
    return records


def process_one(group: str, participant: str, session: str) -> dict:
    expected = expected_rows(group, session)
    head_path, mea_path = source_paths(group, participant, session)
    destination = output_path(group, participant, session)

    head = pd.read_csv(head_path, encoding="utf-8-sig")
    mea_z = pd.read_csv(mea_path, usecols=["mea_z"], encoding="utf-8-sig")[
        "mea_z"
    ]
    if len(head) != expected or len(mea_z) != expected:
        raise RuntimeError(f"source changed after preflight: {group}_{participant}_{session}")

    nan_before = int(head[HEAD_SIGNAL_COLUMNS].isna().sum().sum())
    for column in HEAD_SIGNAL_COLUMNS:
        numeric = pd.to_numeric(head[column], errors="raise")
        head[column] = numeric.interpolate(method="linear", limit_direction="both")
        head[column] = head[column].ffill().bfill()
    nan_after = int(head[HEAD_SIGNAL_COLUMNS].isna().sum().sum())
    if nan_after:
        raise RuntimeError(
            f"Head NaN remains after interpolation: {group}_{participant}_{session}"
        )

    head["mea_z"] = pd.to_numeric(mea_z, errors="raise").to_numpy(dtype="float64")
    analysis_columns = ["pose_Rx", "pose_Ry", "pose_Rz", "mea_z"]
    if not np.isfinite(head[analysis_columns].to_numpy(dtype="float64")).all():
        raise RuntimeError(
            f"non-finite analysis signal after preprocessing: "
            f"{group}_{participant}_{session}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    head.to_csv(destination, index=False)
    return {
        "group": group,
        "participant": participant,
        "session": session,
        "rows": int(len(head)),
        "nan_before": nan_before,
        "nan_after": nan_after,
        "output": str(destination),
    }


def validate_outputs() -> None:
    files = sorted(OUTPUT_DIR.glob("*/*_slim.csv"))
    expected_count = len(GROUPS) * len(PARTICIPANTS) * len(SESSIONS)
    errors: list[str] = []
    if len(files) != expected_count:
        errors.append(f"output file count {len(files)} != {expected_count}")

    required = {"frame", "timestamp", "pose_Rx", "pose_Ry", "pose_Rz", "mea_z"}
    for group in GROUPS:
        for session in SESSIONS:
            expected = expected_rows(group, session)
            pair_axes: dict[str, pd.DataFrame] = {}
            for participant in PARTICIPANTS:
                path = output_path(group, participant, session)
                if not path.is_file():
                    errors.append(f"missing output: {path}")
                    continue
                data = pd.read_csv(path)
                missing = sorted(required - set(data.columns))
                if missing:
                    errors.append(f"{path}: missing columns {missing}")
                    continue
                if len(data) != expected:
                    errors.append(f"{path}: rows {len(data)} != {expected}")
                values = data[["pose_Rx", "pose_Ry", "pose_Rz", "mea_z"]].to_numpy(
                    dtype="float64"
                )
                if not np.isfinite(values).all():
                    errors.append(f"{path}: non-finite analysis values")
                pair_axes[participant] = data[["frame", "timestamp"]]

            if set(pair_axes) == set(PARTICIPANTS):
                if not pair_axes["P1"].equals(pair_axes["P2"]):
                    errors.append(f"{group} {session}: output P1/P2 time-axis mismatch")

    if errors:
        raise RuntimeError("Output validation failed:\n  - " + "\n  - ".join(errors))


def write_report(preflight_records: list[dict], output_records: list[dict]) -> Path:
    report = {
        "pipeline": "unified_head_mea_preprocessing",
        "source_hz": SOURCE_HZ,
        "input_head_dir": str(HEAD_DIR),
        "input_mea_dir": str(MEA_DIR),
        "output_dir": str(OUTPUT_DIR),
        "file_count": len(output_records),
        "total_rows": sum(record["rows"] for record in output_records),
        "total_head_signal_nan_before": sum(
            record["nan_before"] for record in output_records
        ),
        "total_head_signal_nan_after": sum(
            record["nan_after"] for record in output_records
        ),
        "output_definition": (
            "all original Head columns with interpolated Head signals, plus mea_z"
        ),
        "timeline_rules": SAMPLE_COUNTS,
        "alignment": (
            "MEA local rows 0..N-1 are aligned by row order; output frame and "
            "timestamp come from the corresponding Head file"
        ),
        "preflight": preflight_records,
        "outputs": output_records,
    }
    path = OUTPUT_DIR / "preprocess_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    configure_manifest()
    print("=" * 72)
    print("Unified Head + MEA preprocessing")
    print("Head interpolation + MEA row alignment -> one preprocessed dataset")
    print("=" * 72)

    records = preflight()
    total_rows = sum(record["rows"] for record in records)
    print(f"[OK] source preflight: {len(records)} pairs / {total_rows:,} rows")
    if args.validate_only:
        print("[OK] validate-only completed; no output was written")
        return

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output directory already exists: {OUTPUT_DIR}\n"
            "Remove it explicitly before a clean rebuild."
        )

    output_records: list[dict] = []
    for index, group in enumerate(GROUPS, start=1):
        for participant in PARTICIPANTS:
            for session in SESSIONS:
                output_records.append(process_one(group, participant, session))
        print(f"[OK] processed {group} ({index}/{len(GROUPS)})", flush=True)

    validate_outputs()
    report_path = write_report(records, output_records)
    total_nan_before = sum(record["nan_before"] for record in output_records)

    print("=" * 72)
    print(f"[OK] outputs: {len(output_records)} files / {total_rows:,} rows")
    print(f"[OK] Head signal NaN interpolated: {total_nan_before:,} -> 0")
    print(f"[OK] report: {report_path}")
    print(f"[OK] output: {OUTPUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
