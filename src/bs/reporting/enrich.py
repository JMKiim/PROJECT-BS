"""Add synchrony weights and session assessment scores."""
from bs.settings import path, tool
import os
import re
import numpy as np
import pandas as pd
from typing import Optional
# 경로 설정
ROOT_OUT   = str(path('results_dir'))
MASTER_IN  = os.path.join(ROOT_OUT, "master_sync_summary.xlsx")
MASTER_OUT = os.path.join(ROOT_OUT, "master_enriched.xlsx")
CPS_PATH   = str(path('assessment_file'))

# frames_k 고정 세트 (0~5 수준 사용)
K_COLS = [f"frames_k{i}" for i in range(6)]
DERIVED_COLS = ["frames_half", "frames_duo"]  # 없으면 0으로 보완
# 유틸 함수
def ensure_cols(df: pd.DataFrame, cols, fill_val=0):
    for c in cols:
        if c not in df.columns:
            df[c] = fill_val
    return df

def to_int_safe(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(x)
    except:
        s = str(x)
        m = re.search(r"\d+", s)
        return int(m.group()) if m else default

def _to_int_week(x, default=0):
    return to_int_safe(x, default)

def _parse_step_to_phase(step_val):
    """
    'STEP 3.2' → (week=3, phase=2)
    '3.1' 같은 축약형 허용, 'STEP 2' → (2,1) 가정
    """
    if pd.isna(step_val):
        return None, None
    s = str(step_val)
    m = re.search(r"(\d+)\s*[\.\-]\s*(\d+)", s)
    if not m:
        m2 = re.search(r"(\d+)", s)
        if m2:
            return int(m2.group(1)), 1
        return None, None
    return int(m.group(1)), int(m.group(2))

def _normalize_team_to_semester_team_id(semester_str, team_val):
    if pd.isna(team_val):
        return None
    team = str(team_val).strip()
    team = re.sub(r"\s+", "", team)
    return f"{semester_str}-{team}"

def read_cps_scores_per_session(
    cps_path: str,
    sheet_name: str,
    semester_long: str,
    colmap_hint: Optional[dict] = None,  # 변경: Optional 사용
) -> pd.DataFrame:
    """
    세션 단위(CPS) 점수 파싱 → (SEMESTER_TEAM_ID, WEEK, PHASE) 점수 DF 반환
    - TEAM, WEEK, STEP, TOTAL/CRITICAL/CREATIVE 동적 탐지
    - TEAM/WEEK 빈 칸은 forward-fill
    - STEP을 기반으로 WEEK/PHASE 결정 (WEEK 문자열과 충돌 시 STEP 우선)
    """
    raw = pd.read_excel(cps_path, sheet_name=sheet_name)

    # 1) 컬럼명 정규화(대문자 + 공백 제거)
    norm_cols = {c: re.sub(r"\s+", "", str(c)).upper() for c in raw.columns}
    df = raw.rename(columns=norm_cols).copy()

    # 힌트 우선 적용
    targets = {"TEAM": None, "WEEK": None, "STEP": None, "TOTAL": None, "CRITICAL": None, "CREATIVE": None}
    if colmap_hint:
        for k, v in colmap_hint.items():
            if v in raw.columns and k in targets:
                targets[k] = v

    # 후보 자동 탐색
    def _find_one(candidates):
        for c in df.columns:
            if any(key in c for key in candidates):
                return c
        return None

    if targets["TEAM"] is None:
        targets["TEAM"] = _find_one(["TEAM"])
    if targets["WEEK"] is None:
        targets["WEEK"] = _find_one(["WEEK"])
    if targets["STEP"] is None:
        targets["STEP"] = _find_one(["STEP"])
    if targets["TOTAL"] is None:
        targets["TOTAL"] = _find_one(["TOTAL"])
    if targets["CRITICAL"] is None:
        targets["CRITICAL"] = _find_one(["CRITICAL"])
    if targets["CREATIVE"] is None:
        targets["CREATIVE"] = _find_one(["CREATIVE"])

    # 정규화된 이름으로 최종 바인딩
    bind = {}
    for k, v in targets.items():
        if v is None:
            bind[k] = None
            continue
        v_norm = re.sub(r"\s+", "", str(v)).upper()
        use = [c for c in df.columns if c == v_norm]
        bind[k] = use[0] if use else None

    # 필수 키 확인
    for req in ["TEAM", "WEEK", "STEP"]:
        if not bind.get(req):
            raise ValueError(f"CPS 시트({sheet_name})에서 '{req}' 컬럼을 찾지 못했습니다.")

    # 2) 필요한 컬럼 추출 및 ffill
    use_cols = [bind["TEAM"], bind["WEEK"], bind["STEP"], bind["TOTAL"], bind["CRITICAL"], bind["CREATIVE"]]
    use_cols = [c for c in use_cols if c is not None]
    sub = df[use_cols].copy()

    sub[bind["TEAM"]] = sub[bind["TEAM"]].ffill()
    sub[bind["WEEK"]] = sub[bind["WEEK"]].ffill()

    # 3) 키 파싱
    sub["WEEK_INT"] = sub[bind["WEEK"]].apply(_to_int_week)
    wk_from_step, phases = [], []
    for val in sub[bind["STEP"]]:
        w, p = _parse_step_to_phase(val)
        wk_from_step.append(w)
        phases.append(p)
    sub["WEEK_FROM_STEP"] = wk_from_step
    sub["PHASE"] = phases
    sub["WEEK_FINAL"] = np.where(pd.notna(sub["WEEK_FROM_STEP"]), sub["WEEK_FROM_STEP"], sub["WEEK_INT"])
    sub["WEEK_FINAL"] = sub["WEEK_FINAL"].fillna(0).astype(int)

    # 4) 점수 수치화
    for c_std, k in [("TOTAL", "TOTAL"), ("CRITICAL", "CRITICAL"), ("CREATIVE", "CREATIVE")]:
        col = bind.get(k)
        if col and col in sub.columns:
            sub[c_std] = pd.to_numeric(sub[col], errors="coerce")
        else:
            sub[c_std] = np.nan

    # 5) 최종 키 & SEMESTER_TEAM_ID
    sub["SEMESTER_TEAM_ID"] = sub[bind["TEAM"]].apply(lambda t: _normalize_team_to_semester_team_id(semester_long, t))
    out = sub[["SEMESTER_TEAM_ID", "WEEK_FINAL", "PHASE", "TOTAL", "CRITICAL", "CREATIVE"]].copy()
    out = out.rename(columns={"WEEK_FINAL": "WEEK"})
    out = out.dropna(subset=["SEMESTER_TEAM_ID", "WEEK", "PHASE"])
    out["WEEK"] = out["WEEK"].astype(int)
    out["PHASE"] = out["PHASE"].astype(int)

    # 중복 키 대비(마지막 값 유지)
    out = (out
           .sort_values(["SEMESTER_TEAM_ID", "WEEK", "PHASE"])
           .drop_duplicates(["SEMESTER_TEAM_ID", "WEEK", "PHASE"], keep="last"))
    return out

def read_all_cps_scores(cps_path: str) -> pd.DataFrame:
    """세 시트를 모두 읽어 concat"""
    sheets = [
        ("2023 2학기_평가_STEP", "2023-2", {
            "TEAM": "TEAM",
            "WEEK": "WEEK",
            "STEP": "STEP",
            "TOTAL": "STEP_TOTAL",
            "CRITICAL": "STEP_critical thinking",
            "CREATIVE": "STEP_creative thinking",
        }),
        ("2024 1학기_평가_STEP", "2024-1", {
            "TEAM": "TEAM",
            "WEEK": "WEEK",
            "STEP": "STEP",
            "TOTAL": "STEP_TOTAL",
            "CRITICAL": "STEP_critical thinking",
            "CREATIVE": "STEP_creative thinking",
        }),
        ("2024 2학기_평가_STEP", "2024-2", {
            "TEAM": "TEAM",
            "WEEK": "WEEK",
            "STEP": "STEP",
            "TOTAL": "STEP_TOTAL",
            "CRITICAL": "STEP_critical thinking",
            "CREATIVE": "STEP_creative thinking",
        }),
    ]
    parts = []
    for sheet_name, semester_long, hint in sheets:
        df_one = read_cps_scores_per_session(
            cps_path=cps_path,
            sheet_name=sheet_name,
            semester_long=semester_long,
            colmap_hint=hint,
        )
        parts.append(df_one)
    return pd.concat(parts, ignore_index=True)
# 1) 마스터 읽기
def main():
    df = pd.read_excel(MASTER_IN)

    # 정리: WEEK/PHASE 정수화, 필수 컬럼 보장
    df["WEEK"]  = df["WEEK"].apply(to_int_safe)
    df["PHASE"] = df["PHASE"].apply(to_int_safe)
    df = ensure_cols(df, K_COLS + DERIVED_COLS, fill_val=0)
    # 2) 가중합 파생 컬럼 (전체/half/duo)
    weights_linear = {f"frames_k{k}": k for k in range(1, 6)}           # 1..5
    weights_square = {f"frames_k{k}": (k**2) for k in range(1, 6)}       # 1,4,9,16,25
    weights_exp    = {f"frames_k{k}": (10**(k-1)) for k in range(1, 6)}  # 1,10,100,1000,10000

    def weighted_sum(row, weights):
        return sum(row.get(col, 0) * w for col, w in weights.items())

    def weighted_sum_kmin(row, weights, k_min):
        total = 0
        for k in range(max(1, k_min), 6):  # 1..5
            col = f"frames_k{k}"
            total += row.get(col, 0) * weights.get(col, 0)
        return total

    # 전체
    df["LINEAR"]      = df.apply(lambda r: weighted_sum(r, weights_linear), axis=1)
    df["SQUARE"]      = df.apply(lambda r: weighted_sum(r, weights_square), axis=1)
    df["EXPONENTIAL"] = df.apply(lambda r: weighted_sum(r, weights_exp), axis=1)

    # half (k >= ceil(n_members/2))
    def k_half_min(row):
        nm = row.get("n_members", 0)
        try:
            nm = int(nm)
        except:
            nm = 0
        kh = int(np.ceil(nm / 2.0))
        return min(max(kh, 1), 5)

    df["LINEAR_half"]      = df.apply(lambda r: weighted_sum_kmin(r, weights_linear, k_half_min(r)), axis=1)
    df["SQUARE_half"]      = df.apply(lambda r: weighted_sum_kmin(r, weights_square, k_half_min(r)), axis=1)
    df["EXPONENTIAL_half"] = df.apply(lambda r: weighted_sum_kmin(r, weights_exp,  k_half_min(r)), axis=1)

    # duo (k >= 2)
    df["LINEAR_duo"]      = df.apply(lambda r: weighted_sum_kmin(r, weights_linear, 2), axis=1)
    df["SQUARE_duo"]      = df.apply(lambda r: weighted_sum_kmin(r, weights_square, 2), axis=1)
    df["EXPONENTIAL_duo"] = df.apply(lambda r: weighted_sum_kmin(r, weights_exp,    2), axis=1)

    # 23-2 & PHASE==3 드롭 (병합 전이 깔끔)
    sem_col = df.get("SEMESTER_TEAM_ID")
    if sem_col is not None:
        df = df[~((df["PHASE"] == 3) & (df["SEMESTER_TEAM_ID"].astype(str).str.startswith("2023-2")))]
    # 3) CPS 점수 병합 (세션 단위)
    cps_all = read_all_cps_scores(CPS_PATH)

    df = df.drop(columns=["TOTAL","CRITICAL","CREATIVE"], errors="ignore")
    df = df.merge(cps_all, on=["SEMESTER_TEAM_ID", "WEEK", "PHASE"], how="left")
    # 4) 정렬 & 저장 (Master + measurement별 시트)
    df = df.sort_values(by=["SEMESTER_TEAM_ID", "WEEK", "PHASE", "measurement"]).reset_index(drop=True)

    with pd.ExcelWriter(MASTER_OUT, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Master", index=False)
        for m in df["measurement"].astype(str).unique():
            sub = df[df["measurement"].astype(str) == m].copy()
            sub.to_excel(writer, sheet_name=m[:31], index=False)  # 시트명 31자 제한

    print(f"[OK] Saved → {MASTER_OUT}")


if __name__ == "__main__":
    main()
