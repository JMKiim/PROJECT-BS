import os
import re
import numpy as np
import pandas as pd
from typing import Optional

# =========================
# 경로 설정
# =========================
ROOT_OUT   = r"D:/2025EE_Final_Output"
MASTER_IN  = os.path.join(ROOT_OUT, "temp_master.xlsx")          # 입력(세션 포함 Master)
MASTER_OUT = os.path.join(ROOT_OUT, "master_temprocess.xlsx")      # 출력(세션 통합/정규화/CPS/가중치)
CPS_PATH   = os.path.join(ROOT_OUT, "CPS_Score.xlsx")            # CPS 점수 파일(주차 단위 시트 사용)

# frames_k 고정 세트 (0~5 수준)
K_COLS = [f"frames_k{i}" for i in range(6)]
DERIVED_COLS = ["frames_half", "frames_duo"]  # 없으면 0으로 보완
TARGET_FRAMES = 3000 * 15  # 50분 * 60초 * 15fps = 45,000

# =========================
# 유틸 함수
# =========================
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

def _normalize_team_to_semester_team_id(semester_str, team_val):
    if pd.isna(team_val):
        return None
    team = str(team_val).strip()
    team = re.sub(r"\s+", "", team)
    return f"{semester_str}-{team}"

# =========================
# CPS(주차 단위) 읽기
# =========================
def read_cps_scores_per_week(
    cps_path: str,
    sheet_name: str,
    semester_long: str,
    colmap_hint: Optional[dict] = None,
) -> pd.DataFrame:
    """
    주차 단위(CPS) 점수 파싱 → (SEMESTER_TEAM_ID, WEEK) 점수 DF 반환
    - TEAM, WEEK, TOTAL/CRITICAL/CREATIVE 동적 탐지
    - TEAM/WEEK 빈 칸은 forward-fill
    - PHASE는 사용하지 않음(세션을 이미 통합했기 때문)
    """
    raw = pd.read_excel(cps_path, sheet_name=sheet_name)

    # 표준화용(대문자+공백제거) 뷰
    norm_name = {c: re.sub(r"\s+", "", str(c)).upper() for c in raw.columns}
    df = raw.rename(columns=norm_name).copy()

    # 힌트(있으면 우선 사용). 단, 힌트는 원본 컬럼명 기준이라 실패할 수 있으니 fallback도 둔다.
    targets = {"TEAM": None, "WEEK": None, "TOTAL": None, "CRITICAL": None, "CREATIVE": None}
    if colmap_hint:
        for k, v in colmap_hint.items():
            if v in raw.columns:  # 원본 이름으로 존재하면
                targets[k] = re.sub(r"\s+", "", str(v)).upper()

    # 자동 탐색 함수
    def _find_first(cands, prefer=None):
        cols = list(df.columns)
        # prefer(우선 키워드)가 있으면 그걸 포함하는 컬럼 우선
        if prefer:
            for c in cols:
                if all(p in c for p in prefer):
                    return c
        # 아니면 후보 중 하나라도 포함하면 OK
        for c in cols:
            if any(k in c for k in cands):
                return c
        return None

    # TEAM/WEEK 필수
    if targets["TEAM"] is None:
        targets["TEAM"] = _find_first(["TEAM"])
    if targets["WEEK"] is None:
        targets["WEEK"] = _find_first(["WEEK"])

    # 점수 컬럼: WEEK TOTAL을 우선적으로 잡고, 없으면 TOTAL 포함 아무거나
    if targets["TOTAL"] is None:
        targets["TOTAL"] = _find_first(["TOTAL"], prefer=["WEEK", "TOTAL"]) or _find_first(["TOTAL"])
    if targets["CRITICAL"] is None:
        targets["CRITICAL"] = _find_first(["CRITICAL"])
    if targets["CREATIVE"] is None:
        targets["CREATIVE"] = _find_first(["CREATIVE"])

    # 필수 키 확인
    for req in ["TEAM", "WEEK"]:
        if not targets.get(req):
            raise ValueError(f"CPS WEEK 시트({sheet_name})에서 '{req}' 컬럼을 찾지 못했습니다.")

    use_cols = [targets["TEAM"], targets["WEEK"], targets["TOTAL"], targets["CRITICAL"], targets["CREATIVE"]]
    use_cols = [c for c in use_cols if c is not None]
    sub = df[use_cols].copy()

    sub[targets["TEAM"]] = sub[targets["TEAM"]].ffill()
    sub[targets["WEEK"]] = sub[targets["WEEK"]].ffill()

    # 주차 정수화 ("WEEK 3" → 3 등)
    sub["WEEK"] = sub[targets["WEEK"]].apply(_to_int_week)

    # 점수 수치화
    def _num_or_nan(colname):
        if colname and colname in sub.columns:
            return pd.to_numeric(sub[colname], errors="coerce")
        return np.nan
    sub["TOTAL"]    = _num_or_nan(targets.get("TOTAL"))
    sub["CRITICAL"] = _num_or_nan(targets.get("CRITICAL"))
    sub["CREATIVE"] = _num_or_nan(targets.get("CREATIVE"))

    # 키 구성
    sub["SEMESTER_TEAM_ID"] = sub[targets["TEAM"]].apply(lambda t: _normalize_team_to_semester_team_id(semester_long, t))
    out = sub[["SEMESTER_TEAM_ID", "WEEK", "TOTAL", "CRITICAL", "CREATIVE"]].dropna(subset=["SEMESTER_TEAM_ID", "WEEK"])
    out["WEEK"] = out["WEEK"].astype(int)

    out = (out
           .sort_values(["SEMESTER_TEAM_ID", "WEEK"])
           .drop_duplicates(["SEMESTER_TEAM_ID", "WEEK"], keep="last"))
    return out

def read_all_cps_scores_week(cps_path: str) -> pd.DataFrame:
    """세 학기 주차 단위 시트를 모두 읽어 concat"""
    sheets = [
        ("2023 2학기_평가_WEEK", "2023-2", {
            "TEAM": "TEAM",
            "WEEK": "WEEK",
            "TOTAL": "WEEK TOTAL",                 # 공백 포함 케이스
            "CRITICAL": "WEEK_critical thinking",
            "CREATIVE": "WEEK_creative thinking",
        }),
        ("2024 1학기_평가_WEEK", "2024-1", {
            "TEAM": "TEAM",
            "WEEK": "WEEK",
            "TOTAL": "WEEK TOTAL",
            "CRITICAL": "WEEK_critical thinking",
            "CREATIVE": "WEEK_creative thinking",
        }),
        ("2024 2학기_평가_WEEK", "2024-2", {
            "TEAM": "TEAM",
            "WEEK": "WEEK",
            "TOTAL": "WEEK TOTAL",
            "CRITICAL": "WEEK_critical thinking",
            "CREATIVE": "WEEK_creative thinking",
        }),
    ]
    parts = []
    for sheet_name, semester_long, hint in sheets:
        parts.append(read_cps_scores_per_week(CPS_PATH, sheet_name, semester_long, hint))
    return pd.concat(parts, ignore_index=True)

# =========================
# 1) 마스터 읽기 (세션 포함)
# =========================
df = pd.read_excel(MASTER_IN)

# 기본 보정
df["WEEK"] = df["WEEK"].apply(to_int_safe)
df = ensure_cols(df, K_COLS + DERIVED_COLS, fill_val=0)

# =========================
# 2) 세션(phase) 머지: 같은 (학기+팀, 주차, 지표) 기준 합치기
# =========================
agg_map = {c: "sum" for c in K_COLS + DERIVED_COLS}  # frames_* 합산
if "n_members" in df.columns:
    agg_map["n_members"] = "max"                     # 세션 중 최대 인원
if "FPS" in df.columns:
    agg_map["FPS"] = "first"
if "lookback_sec" in df.columns:
    agg_map["lookback_sec"] = "first"

group_keys = ["SEMESTER_TEAM_ID", "WEEK", "measurement"]
present_keys = [k for k in group_keys if k in df.columns]
dfm = df.groupby(present_keys, as_index=False).agg(agg_map)

# 세션 통합 후 PHASE는 불필요 → 있으면 삭제
dfm = dfm.drop(columns=["PHASE"], errors="ignore")

# n_members 없으면 0으로
dfm = ensure_cols(dfm, ["n_members"], fill_val=0)

# =========================
# 3) 50분(=45,000 프레임) 기준 정규화
# =========================
sum_k = dfm[K_COLS].sum(axis=1).replace(0, np.nan)   # 0이면 NaN으로 두고 스케일 0 처리
scale = TARGET_FRAMES / sum_k
for c in K_COLS:
    dfm[c] = (dfm[c] * scale).fillna(0.0)

# 정규화된 frames_k 기반으로 frames_half/frames_duo 재계산
def frames_half_from_row(row):
    nm = row.get("n_members", 0)
    try:
        nm = int(nm)
    except:
        nm = 0
    kh = int(np.ceil(nm / 2.0))
    kh = min(max(kh, 1), 5)  # 1..5로 클램프
    return sum(row.get(f"frames_k{k}", 0.0) for k in range(kh, 6))

def frames_duo_from_row(row):
    return sum(row.get(f"frames_k{k}", 0.0) for k in range(2, 6))

dfm["frames_half"] = dfm.apply(frames_half_from_row, axis=1)
dfm["frames_duo"]  = dfm.apply(frames_duo_from_row,  axis=1)

# =========================
# 4) 주차 단위 CPS 점수 병합
# =========================
cps_week = read_all_cps_scores_week(CPS_PATH)

# 기존 점수 컬럼 있으면 제거 후 재병합
dfm = dfm.drop(columns=["TOTAL", "CRITICAL", "CREATIVE"], errors="ignore")
dfm = dfm.merge(cps_week, on=["SEMESTER_TEAM_ID", "WEEK"], how="left")

# =========================
# 5) 가중합 파생 칼럼 (정규화된 frames_k 기반)
# =========================
weights_linear = {f"frames_k{k}": k for k in range(1, 6)}           # 1,2,3,4,5
weights_square = {f"frames_k{k}": (k**2) for k in range(1, 6)}       # 1,4,9,16,25
weights_exp    = {f"frames_k{k}": (10**(k-1)) for k in range(1, 6)}  # 1,10,100,1000,10000

def weighted_sum(row, weights):
    return sum(row.get(col, 0.0) * w for col, w in weights.items())

def weighted_sum_kmin(row, weights, k_min):
    total = 0.0
    for k in range(max(1, k_min), 6):  # 1..5
        col = f"frames_k{k}"
        total += row.get(col, 0.0) * weights.get(col, 0.0)
    return total

# 전체
dfm["LINEAR"]      = dfm.apply(lambda r: weighted_sum(r, weights_linear), axis=1)
dfm["SQUARE"]      = dfm.apply(lambda r: weighted_sum(r, weights_square), axis=1)
dfm["EXPONENTIAL"] = dfm.apply(lambda r: weighted_sum(r, weights_exp), axis=1)

# half (k >= ceil(n_members/2))
def k_half_min(row):
    nm = row.get("n_members", 0)
    try:
        nm = int(nm)
    except:
        nm = 0
    kh = int(np.ceil(nm / 2.0))
    return min(max(kh, 1), 5)

dfm["LINEAR_half"]      = dfm.apply(lambda r: weighted_sum_kmin(r, weights_linear, k_half_min(r)), axis=1)
dfm["SQUARE_half"]      = dfm.apply(lambda r: weighted_sum_kmin(r, weights_square, k_half_min(r)), axis=1)
dfm["EXPONENTIAL_half"] = dfm.apply(lambda r: weighted_sum_kmin(r, weights_exp,  k_half_min(r)), axis=1)

# duo (k >= 2)
dfm["LINEAR_duo"]      = dfm.apply(lambda r: weighted_sum_kmin(r, weights_linear, 2), axis=1)
dfm["SQUARE_duo"]      = dfm.apply(lambda r: weighted_sum_kmin(r, weights_square, 2), axis=1)
dfm["EXPONENTIAL_duo"] = dfm.apply(lambda r: weighted_sum_kmin(r, weights_exp,    2), axis=1)

# =========================
# 6) 정렬 & 저장 (Master + measurement별 시트)
# =========================
dfm = dfm.sort_values(by=["SEMESTER_TEAM_ID", "WEEK", "measurement"]).reset_index(drop=True)

with pd.ExcelWriter(MASTER_OUT, engine="openpyxl") as writer:
    dfm.to_excel(writer, sheet_name="Master", index=False)
    for m in dfm["measurement"].astype(str).unique():
        sub = dfm[dfm["measurement"].astype(str) == m].copy()
        sub.to_excel(writer, sheet_name=m[:31], index=False)  # 시트명 31자 제한

print(f"[OK] Saved → {MASTER_OUT}")
