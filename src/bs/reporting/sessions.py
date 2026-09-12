"""Split weekly masks using private timeline metadata."""
from bs.settings import path, tool
import os, re, json
import pandas as pd
import numpy as np
ROOT_OUT = str(path('results_dir'))        # 주차 통합 엑셀들이 있는 루트 (sync_{sem}_{grp}_{week}.xlsx)
FPS = 15

# CSV 경로
TIMELINE_INFO_CSV    = str(path('timeline_metadata'))      # 학기,그룹명,주차,파일버전,타임라인인덱스,시작시간,종료시간,인원수
SESSION_TIMELINE_CSV = str(path('session_metadata'))   # 학기,그룹명,주차,파일버전,분기시간,세션  (타임라인인덱스 없음)

# 지표 순서 고정을 위해 사용
CONFIG_PATH = str(path('indicators'))

WEEK_VER_RE = re.compile(r"^\((\d+)\)$")   # "(1)" -> 1
PID_NUM_RE  = re.compile(r"(\d+)")

def parse_version(ver_str: str) -> int:
    if pd.isna(ver_str): return 0
    s = str(ver_str).strip()
    if s in ("없음",""): return 0
    m = WEEK_VER_RE.match(s)
    return int(m.group(1)) if m else 0

def hms_to_seconds(s: str) -> float:
    if pd.isna(s): return 0.0
    parts = [p.strip() for p in str(s).split(":")]
    # hh:mm:ss
    if len(parts) == 3:
        h,m,sec = parts
        return int(h)*3600 + int(m)*60 + float(sec)
    # mm:ss
    if len(parts) == 2:
        m,sec = parts
        return int(m)*60 + float(sec)
    # ss
    try:
        return float(parts[0])
    except:
        return 0.0

def extract_sem_grp_week(path: str):
    # 파일명: sync_{sem}_{grp}_{week}.xlsx
    base = os.path.splitext(os.path.basename(path))[0]
    try:
        _, sem, grp, week = base.split("_", 3)
        return sem, grp, week
    except:
        return None, None, None
def load_indicator_order(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # Python 3.7+ dict insertion-order 보존 가정
    return list(cfg.keys())

def pid_key(pid):
    if pd.isna(pid): return (1<<30)
    m = PID_NUM_RE.search(str(pid))
    return int(m.group(1)) if m else (1<<29)

def order_rawmask_columns(df: pd.DataFrame, indicator_order: list) -> pd.DataFrame:
    """RawMask: MultiIndex(indicator, pid) → config 순서 + pid 오름차순으로 정렬"""
    if df.empty or not isinstance(df.columns, pd.MultiIndex): return df
    lvl0 = df.columns.get_level_values(0)
    lvl1 = df.columns.get_level_values(1)
    present = set(lvl0)

    # config에 있는 것 중 존재하는 지표만 먼저
    cols = []
    for ind in indicator_order:
        if ind in present:
            pids = sorted(set(lvl1[lvl0==ind]), key=pid_key)
            cols.extend([(ind, p) for p in pids])

    # config에 없지만 엑셀에 있는 지표는 뒤에
    others = sorted([(i,p) for i,p in df.columns if i not in set(indicator_order)],
                    key=lambda x:(x[0], pid_key(x[1])))
    cols.extend(others)

    return df.reindex(columns=pd.MultiIndex.from_tuples(cols, names=df.columns.names))

def order_pfl_columns(df: pd.DataFrame, indicator_order: list) -> pd.DataFrame:
    """PerFrameLevels: MultiIndex(indicator, level) → config 순서 + level 숫자 오름차순"""
    if df.empty or not isinstance(df.columns, pd.MultiIndex): return df
    lvl0 = df.columns.get_level_values(0)
    lvl1 = df.columns.get_level_values(1)
    present = set(lvl0)

    cols = []
    for ind in indicator_order:
        if ind in present:
            levels = sorted(set(lvl1[lvl0==ind]), key=lambda x: int(str(x)))
            cols.extend([(ind, lv) for lv in levels])

    # 나머지 지표는 뒤에
    others = sorted([(i,l) for i,l in df.columns if i not in set(indicator_order)],
                    key=lambda x:(x[0], int(str(x[1]))))
    cols.extend(others)

    return df.reindex(columns=pd.MultiIndex.from_tuples(cols, names=df.columns.names))

def order_levelcounts(df: pd.DataFrame, indicator_order: list) -> pd.DataFrame:
    """LevelCounts: index=indicator, columns=level → config 순서 + level 숫자 오름차순"""
    if df.empty: return df
    idx_present = [ind for ind in indicator_order if ind in set(df.index)]
    others = sorted([i for i in df.index if i not in set(indicator_order)])
    df2 = df.reindex(index=idx_present + others)

    cols = [str(c) for c in df2.columns]
    df2.columns = cols
    df2 = df2.reindex(columns=sorted(cols, key=lambda x: int(x)))
    df2.index.name = "indicator"
    return df2

def levelcounts_from_perframelevels(pfl_slice: pd.DataFrame, indicator_order: list) -> pd.DataFrame:
    if pfl_slice.empty:
        return pd.DataFrame()
    sums = pfl_slice.sum(axis=0)  # frame 합계
    sums.index = pd.MultiIndex.from_tuples(sums.index, names=["indicator","level"])
    df = sums.unstack(level="level").fillna(0).astype(int)
    return order_levelcounts(df, indicator_order)
def compute_split_frames(semester: str, group: str, week_base: str,
                         df_info: pd.DataFrame, df_sess: pd.DataFrame):
    """
    반환: [frame_at_session2, (frame_at_session3)]  (to_session 오름차순)
    - 23-2: 분기시간(hh:mm:ss) × FPS → 전역 프레임으로 간주(Clamping만 수행)
    - 24-1 / 24-2: timeline_info.csv로 분기시간을 타임라인에 매핑(없으면 이후 첫 타임라인 시작, 그래도 없으면 마지막 끝)
    """
    # 세션 CSV 필터
    sess = df_sess[(df_sess['학기']==semester) &
                   (df_sess['그룹명']==group) &
                   (df_sess['주차'].eq(week_base))].copy()
    if sess.empty:
        return []

    sess['split_sec']  = sess['분기시간'].apply(hms_to_seconds)
    sess['to_session'] = sess['세션'].astype(int)
    sess = sess.sort_values(by=['to_session']).reset_index(drop=True)
    if str(semester).startswith("23-2"):
        out_frames = []
        for _, r in sess.iterrows():
            frame_idx = int(round(r['split_sec'] * FPS))
            out_frames.append(max(0, frame_idx))
        return out_frames
    info = df_info[(df_info['학기']==semester) &
                   (df_info['그룹명']==group) &
                   (df_info['주차'].eq(week_base))].copy()
    if info.empty:
        return []

    info['ver_int']   = info['파일버전'].apply(parse_version)
    info['start_sec'] = info['시작시간'].apply(hms_to_seconds)
    info['end_sec']   = info['종료시간'].apply(hms_to_seconds)
    info['tl_frames'] = ((info['end_sec'] - info['start_sec']) * FPS).round().astype(int).clip(lower=0)

    if '타임라인인덱스' in info.columns:
        info['tl_index'] = info['타임라인인덱스'].astype(int)
        info = info.sort_values(by=['ver_int','tl_index','start_sec']).reset_index(drop=True)
    else:
        info = info.sort_values(by=['ver_int','start_sec']).reset_index(drop=True)

    info['offset_frames'] = info['tl_frames'].shift(1, fill_value=0).cumsum().astype(int)
    info['tl_start'] = info['offset_frames']
    info['tl_end']   = info['offset_frames'] + info['tl_frames']   # half-open

    out_frames = []
    has_version = ('파일버전' in sess.columns)

    for _, r in sess.iterrows():
        ver_int = parse_version(r['파일버전']) if has_version else None
        cand = info[info['ver_int']==ver_int] if (has_version and ver_int is not None) else info

        # 1) 분기시간 포함 타임라인
        within = cand[(cand['start_sec'] <= r['split_sec']) & (r['split_sec'] <= cand['end_sec'])]
        if not within.empty:
            row = within.iloc[0]
            offset_sec = r['split_sec'] - row['start_sec']
            frame_idx = int(row['offset_frames'] + round(offset_sec * FPS))
            out_frames.append(max(0, frame_idx))
            continue

        # 2) 이후 첫 타임라인 시작으로 클램프
        later = cand[cand['start_sec'] >= r['split_sec']].sort_values('start_sec')
        if not later.empty:
            out_frames.append(int(later.iloc[0]['offset_frames']))
            continue

        # 3) 마지막 타임라인 끝으로 클램프
        if not cand.empty:
            last = cand.iloc[-1]
            out_frames.append(int(max(last['tl_end'] - 1, 0)))
        else:
            last = info.iloc[-1]
            out_frames.append(int(max(last['tl_end'] - 1, 0)))

    return out_frames
def semester_to_long(sem_short: str) -> str:
    # "24-1" -> "2024-1"
    parts = str(sem_short).split("-")
    if len(parts) != 2: return str(sem_short)
    return f"20{parts[0]}-{parts[1]}"

def compute_n_members_for_session(semester: str, group: str, week_base: str,
                                  df_info: pd.DataFrame, s0: int, s1: int) -> int:
    """
    세션 프레임 구간 [s0, s1)과 겹치는 타임라인들의 '인원수' 최댓값을 반환
    (timeline_info.csv 기반)
    """
    info = df_info[(df_info['학기']==semester) &
                   (df_info['그룹명']==group) &
                   (df_info['주차'].eq(week_base))].copy()
    if info.empty: return 0

    info['ver_int']   = info['파일버전'].apply(parse_version)
    info['start_sec'] = info['시작시간'].apply(hms_to_seconds)
    info['end_sec']   = info['종료시간'].apply(hms_to_seconds)
    info['tl_frames'] = ((info['end_sec'] - info['start_sec']) * FPS).round().astype(int).clip(lower=0)

    if '타임라인인덱스' in info.columns:
        info['tl_index'] = info['타임라인인덱스'].astype(int)
        info = info.sort_values(by=['ver_int','tl_index','start_sec']).reset_index(drop=True)
    else:
        info = info.sort_values(by=['ver_int','start_sec']).reset_index(drop=True)

    info['offset_frames'] = info['tl_frames'].shift(1, fill_value=0).cumsum().astype(int)
    info['tl_start'] = info['offset_frames']
    info['tl_end']   = info['offset_frames'] + info['tl_frames']

    overlap = info[(info['tl_end'] > s0) & (info['tl_start'] < s1)]
    if overlap.empty: return 0

    if '인원수' in overlap.columns:
        return int(pd.to_numeric(overlap['인원수'], errors='coerce').fillna(0).max())
    return 0
def split_week_file(week_path: str, df_info: pd.DataFrame, df_sess: pd.DataFrame, indicator_order: list):
    # 파일명에서 학기/그룹/주차 추출
    sem, grp, week = extract_sem_grp_week(week_path)
    if not sem:
        print(f"[SKIP] Bad filename: {week_path}")
        return

    # 주차 원본 시트 읽기
    try:
        raw = pd.read_excel(week_path, sheet_name="RawMask", header=[0,1], index_col=0)
    except Exception:
        raw = pd.DataFrame()

    try:
        pfl = pd.read_excel(week_path, sheet_name="PerFrameLevels", header=[0,1], index_col=0)
    except Exception:
        pfl = pd.DataFrame()

    if not pfl.empty:
        total_frames = int(pfl.index.max()) + 1
    elif not raw.empty:
        total_frames = int(raw.index.max()) + 1
    else:
        print(f"[SKIP] No RawMask/PerFrameLevels: {week_path}")
        return

    # 세션 분기 프레임 계산 (23-2 예외 포함)
    splits = compute_split_frames(sem, grp, week, df_info, df_sess)
    if not splits:
        print(f"[INFO] No session splits: {week_path}")
        return

    # 경계 프레임 (다음 세션 포함 규칙: [s0, s1))
    boundaries = [0] + [max(0, min(total_frames, s)) for s in splits] + [total_frames]
    boundaries = sorted(set(boundaries))
    n_sessions = len(boundaries) - 1
    if n_sessions < 2:
        print(f"[WARN] Less than 2 sessions for {week_path}: {boundaries}")

    out_dir = os.path.dirname(week_path)

    # 원본을 먼저 정렬 (config 기준)
    if not raw.empty:
        raw = order_rawmask_columns(raw, indicator_order)
    if not pfl.empty:
        pfl = order_pfl_columns(pfl, indicator_order)

    for i in range(n_sessions):
        s0, s1 = boundaries[i], boundaries[i+1]
        sess_no = i+1

        # 구간 슬라이스 + 0부터 재인덱스
        if not raw.empty:
            raw_slice = raw.loc[(raw.index >= s0)&(raw.index < s1)].copy()
            raw_slice.index = np.arange(len(raw_slice)); raw_slice.index.name = "frame"
            raw_slice = order_rawmask_columns(raw_slice, indicator_order)
        else:
            raw_slice = pd.DataFrame()

        if not pfl.empty:
            pfl_slice = pfl.loc[(pfl.index >= s0)&(pfl.index < s1)].copy()
            pfl_slice.index = np.arange(len(pfl_slice)); pfl_slice.index.name = "frame"
            pfl_slice = order_pfl_columns(pfl_slice, indicator_order)
        else:
            pfl_slice = pd.DataFrame()

        # 세션 LevelCounts 재집계
        lc_slice = levelcounts_from_perframelevels(pfl_slice, indicator_order)

        # Metadata 시트
        semester_long = semester_to_long(sem)
        sem_team_id = f"{semester_long}-{grp}"
        n_members = compute_n_members_for_session(sem, grp, week, df_info, s0, s1)

        meta = pd.DataFrame([{
            "SEMESTER_TEAM_ID": sem_team_id,
            "WEEK": week,
            "PHASE": sess_no,
            "n_members": n_members
        }])

        # 출력 (덮어쓰기 모드)
        out_name = f"sync_{sem}_{grp}_{week}_session{sess_no}.xlsx"
        out_path = os.path.join(out_dir, out_name)
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            (raw_slice if not raw_slice.empty else pd.DataFrame()).to_excel(
                writer, sheet_name="RawMask", index=not raw_slice.empty)
            (pfl_slice if not pfl_slice.empty else pd.DataFrame()).to_excel(
                writer, sheet_name="PerFrameLevels", index=not pfl_slice.empty)
            (lc_slice  if not lc_slice.empty   else pd.DataFrame()).to_excel(
                writer, sheet_name="LevelCounts", index=not lc_slice.empty)
            meta.to_excel(writer, sheet_name="Metadata", index=False)

        print(f"[OK] Saved → {out_path}")

def run_all():
    # CSV 로드
    df_info = pd.read_csv(TIMELINE_INFO_CSV, encoding="utf-8")
    df_sess = pd.read_csv(SESSION_TIMELINE_CSV, encoding="utf-8")

    # 지표 순서
    indicator_order = load_indicator_order(CONFIG_PATH)

    # 루트 탐색: ROOT_OUT/{semester}/{group}/sync_{sem}_{grp}_{week}.xlsx
    for semester in sorted(os.listdir(ROOT_OUT)):
        sem_dir = os.path.join(ROOT_OUT, semester)
        if not os.path.isdir(sem_dir): continue
        for group in sorted(os.listdir(sem_dir)):
            grp_dir = os.path.join(sem_dir, group)
            if not os.path.isdir(grp_dir): continue
            for fname in sorted(os.listdir(grp_dir)):
                if not (fname.startswith(f"sync_{semester}_{group}_") and fname.endswith(".xlsx")):
                    continue
                if "_session" in fname:       # 이미 세션 파일은 건너뜀
                    continue
                week_path = os.path.join(grp_dir, fname)
                split_week_file(week_path, df_info, df_sess, indicator_order)

if __name__ == "__main__":
    run_all()
