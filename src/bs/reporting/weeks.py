"""Concatenate timeline masks into weekly workbooks."""
from bs.settings import path, tool
import os
import re
import json
import numpy as np
import pandas as pd

ROOT_IN  = str(path('features_dir'))
ROOT_OUT = str(path('results_dir'))
CONFIG_PATH = str(path('indicators'))   # 지표 정렬용(있으면 사용)
FPS = 15

WEEK_RE = re.compile(r"^W(\d+)(?:\((\d+)\))?$")   # W2 / W2(1) / W2(2) ...
TL_RE   = re.compile(r"^T(\d+)$")                 # T1 / T2 ...

def parse_week_folder(name: str):
    m = WEEK_RE.match(name)
    if not m:
        return None
    week_num = int(m.group(1))
    ver = int(m.group(2)) if m.group(2) else 0
    base = f"W{week_num}"
    return base, week_num, ver

def parse_timeline_folder(name: str):
    m = TL_RE.match(name)
    if not m:
        return None
    return int(m.group(1))

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def read_sheet(xlsx_path: str, sheet: str, header_levels):
    """
    RawMask/PerFrameLevels: header=[0,1], index_col=0
    LevelCounts: header=0, index_col=0
    """
    if sheet in ("RawMask", "PerFrameLevels"):
        return pd.read_excel(xlsx_path, sheet_name=sheet, header=[0,1], index_col=0)
    elif sheet == "LevelCounts":
        return pd.read_excel(xlsx_path, sheet_name=sheet, header=0, index_col=0)
    else:
        raise ValueError(f"Unknown sheet: {sheet}")

def _sort_multi_cols(cols: pd.MultiIndex, indicator_order=None):
    # (indicator, pid/level) 순서 정렬
    if indicator_order is None:
        return pd.MultiIndex.from_tuples(sorted(cols.tolist(), key=lambda x: (str(x[0]), str(x[1]))), names=cols.names)
    def key(t):
        ind, second = str(t[0]), str(t[1])
        try:
            idx = indicator_order.index(ind)
        except ValueError:
            idx = len(indicator_order) + 999
        return (idx, ind, second)
    return pd.MultiIndex.from_tuples(sorted(cols.tolist(), key=key), names=cols.names)

def build_week_file(semester: str, group: str, week_base: str):
    in_group_dir  = os.path.join(ROOT_IN,  semester, group)
    out_group_dir = os.path.join(ROOT_OUT, semester, group)
    if not os.path.isdir(in_group_dir):
        print(f"[SKIP] Missing group: {in_group_dir}")
        return

    # (선택) 지표 정렬용 순서
    indicator_order = None
    if os.path.isfile(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        indicator_order = list(cfg.keys())

    # 해당 week_base의 모든 버전(Wk, Wk(1), Wk(2)...) 수집(버전순 정렬)
    week_candidates = []
    for name in os.listdir(in_group_dir):
        p = os.path.join(in_group_dir, name)
        if not os.path.isdir(p):
            continue
        parsed = parse_week_folder(name)
        if not parsed:
            continue
        base, _, ver = parsed
        if base == week_base:
            week_candidates.append((name, p, ver))
    if not week_candidates:
        print(f"[SKIP] No weeks for {semester}/{group}/{week_base}")
        return
    week_candidates.sort(key=lambda x: x[2])

    # 누적 컨테이너
    rawmask_dfs = []
    pfl_dfs = []
    used_rows = []
    rawmask_union = None
    pfl_union = None
    # LevelCounts 합산용 테이블(합집합 인덱스/컬럼으로 확장하며 누적)
    lc_sum = None
    # 프레임 오프셋
    offset_raw = 0
    offset_pfl = 0

    for week_name, week_path, ver in week_candidates:
        # 타임라인 정렬: T1, T2, ...
        timelines = []
        for tl_name in os.listdir(week_path):
            tl_path = os.path.join(week_path, tl_name)
            if not os.path.isdir(tl_path):
                continue
            tl_idx = parse_timeline_folder(tl_name)
            if tl_idx is None:
                continue
            timelines.append((tl_idx, tl_name, tl_path))
        timelines.sort(key=lambda x: x[0])

        for tl_idx, tl_name, tl_path in timelines:
            xlsx = os.path.join(tl_path, "sync_mask.xlsx")
            if not os.path.isfile(xlsx):
                print(f"[WARN] Missing sync_mask.xlsx: {tl_path}")
                continue
            try:
                df_raw = read_sheet(xlsx, "RawMask", header_levels=2)
                df_pfl = read_sheet(xlsx, "PerFrameLevels", header_levels=2)
                df_lc  = read_sheet(xlsx, "LevelCounts", header_levels=1)
            except Exception as e:
                print(f"[WARN] Read fail {xlsx}: {e}")
                continue
            n_raw = df_raw.shape[0]
            if n_raw > 0:
                rawmask_union = df_raw.columns if rawmask_union is None else rawmask_union.union(df_raw.columns)
                df_raw.index = df_raw.index + offset_raw
                rawmask_dfs.append(df_raw)
                offset_raw += n_raw
            n_pfl = df_pfl.shape[0]
            if n_pfl > 0:
                pfl_union = df_pfl.columns if pfl_union is None else pfl_union.union(df_pfl.columns)
                df_pfl.index = df_pfl.index + offset_pfl
                pfl_dfs.append(df_pfl)
                offset_pfl += n_pfl
            if df_lc.shape[0] > 0:
                if lc_sum is None:
                    lc_sum = df_lc.copy()
                else:
                    # 인덱스/컬럼 union 후 결측 0 채우고 더하기
                    new_index  = lc_sum.index.union(df_lc.index)
                    new_cols   = lc_sum.columns.union(df_lc.columns)
                    lc_sum = lc_sum.reindex(index=new_index, columns=new_cols, fill_value=0)
                    df_lc  = df_lc.reindex(index=new_index, columns=new_cols, fill_value=0)
                    lc_sum = lc_sum + df_lc
            # RawMask 기준 프레임 수로 기록 (PerFrameLevels와 동일해야 정상)
            frames = int(n_raw)
            dur_s  = round(frames / FPS, 2)
            used_rows.append({
                "semester": semester,
                "group": group,
                "week": week_base,          # base only (e.g., W2)
                "week_version": ver,        # 0(no suffix), 1,2,...
                "timeline_index": tl_idx,   # T number
                "frames": frames,
                "duration_sec": dur_s,
                "duration_min": round(dur_s / 60.0, 2),
                "path": tl_path
            })

    if not rawmask_dfs and not pfl_dfs and lc_sum is None:
        print(f"[SKIP] Nothing to merge: {semester}/{group}/{week_base}")
        return
    # RawMask
    if rawmask_dfs:
        rawmask_union = _sort_multi_cols(rawmask_union, indicator_order)
        rawmask_dfs = [df.reindex(columns=rawmask_union, fill_value=0) for df in rawmask_dfs]
        rawmask_merged = pd.concat(rawmask_dfs, axis=0)
        rawmask_merged.index.name = "frame"
    else:
        rawmask_merged = pd.DataFrame()

    # PerFrameLevels
    if pfl_dfs:
        pfl_union = _sort_multi_cols(pfl_union, indicator_order)
        pfl_dfs = [df.reindex(columns=pfl_union, fill_value=0) for df in pfl_dfs]
        pfl_merged = pd.concat(pfl_dfs, axis=0)
        pfl_merged.index.name = "frame"
    else:
        pfl_merged = pd.DataFrame()

    # LevelCounts (지표 정렬: config 순 → 나머지 알파벳, 레벨 컬럼은 0..max 정렬)
    if lc_sum is not None:
        # level 컬럼을 숫자 문자열로 통일 후 정렬
        lc_sum.columns = [str(c) for c in lc_sum.columns]
        # 지표 인덱스 정렬
        if indicator_order:
            ordered = [i for i in indicator_order if i in lc_sum.index]
            others  = sorted([i for i in lc_sum.index if i not in indicator_order])
            lc_sum = lc_sum.loc[ordered + others]
        # 레벨 컬럼 정렬(0..max)
        all_levels = sorted({int(x) for x in lc_sum.columns})
        lc_sum = lc_sum[[str(k) for k in all_levels]]
        lc_merged = lc_sum
    else:
        lc_merged = pd.DataFrame()

    used_df = pd.DataFrame(used_rows).sort_values(by=["week_version", "timeline_index"]).reset_index(drop=True)
    ensure_dir(out_group_dir)
    out_fname = f"sync_{semester}_{group}_{week_base}.xlsx"
    out_path  = os.path.join(out_group_dir, out_fname)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        if not rawmask_merged.empty:
            rawmask_merged.to_excel(writer, sheet_name="RawMask", index=True)
        else:
            pd.DataFrame().to_excel(writer, sheet_name="RawMask", index=False)

        if not pfl_merged.empty:
            pfl_merged.to_excel(writer, sheet_name="PerFrameLevels", index=True)
        else:
            pd.DataFrame().to_excel(writer, sheet_name="PerFrameLevels", index=False)

        if not lc_merged.empty:
            lc_merged.to_excel(writer, sheet_name="LevelCounts", index=True)
        else:
            pd.DataFrame().to_excel(writer, sheet_name="LevelCounts", index=False)

        used_df.to_excel(writer, sheet_name="UsedTimelines", index=False)

    print(f"[OK] Saved → {out_path}")

def build_all_weeks():
    # 스킵 로직 제거 (모든 학기/그룹 처리)
    skip_semesters = set()
    skip_groups    = set()
    skip_weeks     = set()

    for semester in sorted(os.listdir(ROOT_IN)):
        if semester in skip_semesters:
            print(f"[SKIP] {semester} 제외됨")
            continue

        sem_path = os.path.join(ROOT_IN, semester)
        if not os.path.isdir(sem_path):
            continue
        for group in sorted(os.listdir(sem_path)):
            if group in skip_groups:
                print(f"[SKIP] {semester}/{group} 제외됨")
                continue

            grp_path = os.path.join(sem_path, group)
            if not os.path.isdir(grp_path):
                continue

            # 그룹 내 base week 목록 수집 (Wk, Wk(1) → base=Wk)
            bases = set()
            for name in os.listdir(grp_path):
                wk_path = os.path.join(grp_path, name)
                if not os.path.isdir(wk_path):
                    continue
                parsed = parse_week_folder(name)
                if not parsed:
                    continue
                base, _, _ = parsed
                if base in skip_weeks:
                    continue

                bases.add(base)

            for base in sorted(bases, key=lambda b: int(b[1:])):  # 'W2' -> 2
                build_week_file(semester, group, base)

if __name__ == "__main__":
    # 전체 일괄 실행
    build_all_weeks()
    # 또는 특정만:
    # build_week_file("24-1", "A2", "W2")
