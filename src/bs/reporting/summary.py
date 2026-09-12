"""Build session-level synchrony summaries."""
from bs.settings import path, tool
import os
import re
import json
import pandas as pd
import numpy as np
ROOT_OUT = str(path('results_dir'))            # 세션 엑셀들이 있는 루트 (semester/group/…)
CONFIG_PATH = str(path('indicators'))
MASTER_OUT = os.path.join(ROOT_OUT, "master_sync_summary.xlsx")
FPS = 15

# 파일명 패턴: sync_{sem}_{grp}_{week}_session{n}.xlsx
SESSION_FILE_RE = re.compile(r"^sync_(?P<sem>[^_]+)_(?P<grp>[^_]+)_(?P<week>.+)_session(?P<phase>\d+)\.xlsx$")

WEEK_NUM_RE = re.compile(r"W\s*(\d+)", re.IGNORECASE)  # "W2" → 2
PARENS_RE    = re.compile(r"\(\d+\)")                  # "(1)" 제거

def normalize_week_to_int(week_label: str) -> int:
    """
    예시:
      "W2"         -> 2
      "W2(1)"      -> 2
      "W10(2)"     -> 10
      "2"          -> 2
    실패 시 0
    """
    if week_label is None:
        return 0
    s = str(week_label)
    # (1) 같은 파일버전 표기 제거
    s = PARENS_RE.sub("", s)
    s = s.strip()
    # "W10" → 10
    m = WEEK_NUM_RE.search(s)
    if m:
        try:
            return int(m.group(1))
        except:
            return 0
    # 그냥 숫자만 있었다면 그대로
    try:
        return int(s)
    except:
        return 0

def load_indicator_order_and_windows(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    order = list(cfg.keys())
    lookback = {k: float(v.get("sync_window", 0.3)) for k, v in cfg.items()}
    return order, lookback

def parse_session_filename(fname: str):
    """
    파일명에서 (sem, grp, week_raw, phase) 뽑기. week_raw는 문자열 그대로 반환.
    """
    m = SESSION_FILE_RE.match(fname)
    if not m:
        return None
    sem = m.group("sem")
    grp = m.group("grp")
    week_raw = m.group("week")      # "W2" 또는 "W2(1)" 등
    phase = int(m.group("phase"))
    return sem, grp, week_raw, phase

def read_metadata(path: str):
    """
    Metadata 시트를 dict로. 없으면 None.
    기대 컬럼: SEMESTER_TEAM_ID, WEEK, PHASE, n_members
    """
    try:
        df = pd.read_excel(path, sheet_name="Metadata")
        if not df.empty:
            row = df.iloc[0].to_dict()
            # 타입 보정
            row["PHASE"] = int(row.get("PHASE", 0)) if pd.notna(row.get("PHASE", None)) else 0
            row["n_members"] = int(row.get("n_members", 0)) if pd.notna(row.get("n_members", None)) else 0
            row["SEMESTER_TEAM_ID"] = str(row.get("SEMESTER_TEAM_ID", ""))
            row["WEEK"] = str(row.get("WEEK", ""))
            return row
    except Exception:
        pass
    return None

def read_levelcounts(path: str):
    """
    LevelCounts 시트 읽기 (index=indicator, columns=str(level)).
    """
    try:
        df = pd.read_excel(path, sheet_name="LevelCounts", index_col=0)
        if df is None or df.empty:
            return None
        df.columns = [str(c) for c in df.columns]
        df.index.name = "indicator"
        return df
    except Exception:
        return None

def semester_to_long(sem_short: str) -> str:
    # "24-1" -> "2024-1"
    parts = str(sem_short).split("-")
    if len(parts) != 2:
        return str(sem_short)
    return f"20{parts[0]}-{parts[1]}"

def build_master():
    indicator_order, lookback_map = load_indicator_order_and_windows(CONFIG_PATH)
    rows = []

    # ROOT_OUT/semester/group/sync_{sem}_{grp}_{week}_session{n}.xlsx
    for semester in sorted(os.listdir(ROOT_OUT)):
        sem_dir = os.path.join(ROOT_OUT, semester)
        if not os.path.isdir(sem_dir):
            continue

        for group in sorted(os.listdir(sem_dir)):
            grp_dir = os.path.join(sem_dir, group)
            if not os.path.isdir(grp_dir):
                continue

            for fname in sorted(os.listdir(grp_dir)):
                if "_session" not in fname:
                    continue
                if not fname.endswith(".xlsx"):
                    continue

                parsed = parse_session_filename(fname)
                if not parsed:
                    continue
                sem, grp, week_raw, phase = parsed
                fpath = os.path.join(grp_dir, fname)

                # Metadata
                meta = read_metadata(fpath)
                if not meta:
                    # 메타 없으면 파일명/기본값으로 구성
                    meta = {
                        "SEMESTER_TEAM_ID": f"{semester_to_long(sem)}-{grp}",
                        "WEEK": week_raw,
                        "PHASE": phase,
                        "n_members": 0
                    }

                # WEEK 정규화: 숫자만
                # (Metadata.WEEK 우선, 없으면 파일명 기반 week_raw)
                week_source = meta.get("WEEK", week_raw)
                week_num = normalize_week_to_int(week_source)

                # 23-2 학기면 n_members는 무조건 4
                if str(sem).startswith("23-2"):
                    meta["n_members"] = 4

                # LevelCounts
                lc = read_levelcounts(fpath)

                # 각 지표별로 한 행씩
                for m in indicator_order:
                    # 해당 지표가 없으면 0으로 채움
                    if lc is not None and m in lc.index:
                        counts = lc.loc[m].to_dict()  # {"0": n0, "1": n1, ...}
                    else:
                        counts = {}

                    # 최대 인원수
                    n_members = int(meta["n_members"])
                    if n_members > 5:
                        raise ValueError("Summary supports up to five members; retain level workbooks for larger groups")

                    # k0..k5 구성 (최대인원 초과 레벨은 0)
                    k_values = []
                    for k in range(0, 6):
                        if k > n_members:
                            v = 0
                        else:
                            v = int(counts.get(str(k), 0))
                        k_values.append(v)

                    # 파생지표
                    half_threshold = int(np.ceil(n_members / 2.0)) if n_members > 0 else 999  # n=0이면 합=0
                    frames_half = sum(k_values[k] for k in range(half_threshold, min(6, n_members+1)))
                    frames_duo  = sum(k_values[k] for k in range(2, min(6, n_members+1)))

                    row = {
                        "SEMESTER_TEAM_ID": meta["SEMESTER_TEAM_ID"],
                        "WEEK": week_num,                     # 숫자만 기록
                        "PHASE": int(meta["PHASE"]),
                        "measurement": m,
                        "FPS": FPS,
                        "lookback_sec": lookback_map.get(m, 0.3),
                        "n_members": n_members,
                        "frames_half": frames_half,
                        "frames_duo": frames_duo
                    }
                    for k in range(0, 6):
                        row[f"frames_k{k}"] = k_values[k]

                    rows.append(row)

    # Master DF
    master = pd.DataFrame(rows, columns=[
        "SEMESTER_TEAM_ID", "WEEK", "PHASE",
        "measurement", "FPS", "lookback_sec", "n_members",
        "frames_k0", "frames_k1", "frames_k2", "frames_k3", "frames_k4", "frames_k5",
        "frames_half", "frames_duo"
    ])

    # 정렬(가독성): 팀ID, WEEK(숫자), PHASE, measurement
    master = master.sort_values(by=["SEMESTER_TEAM_ID", "WEEK", "PHASE", "measurement"]).reset_index(drop=True)

    # 저장 (덮어쓰기)
    master.to_excel(MASTER_OUT, index=False)
    print(f"[OK] Master saved → {MASTER_OUT}")

if __name__ == "__main__":
    build_master()
