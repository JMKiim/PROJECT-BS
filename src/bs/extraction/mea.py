"""Extract thresholded motion energy with ROI exclusion, cleaning and scaling."""
import os
import re
import sys
import json
import time
import shutil
import argparse
from bs.settings import path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import cv2
import numpy as np
import pandas as pd

from bs.extraction.roi import (load_roi_config, get_exclude_rects, build_keep_mask,
                     rects_signature)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

RAW_COLS = ["frame", "timestamp", "mea", "mea_ppx", "n_active_px", "frac_active", "mean_luma"]
POSE_AXES = ["pose_Rx", "pose_Ry", "pose_Rz"]


def parse_args():
    p = argparse.ArgumentParser(
        description="MEA(Motion Energy Analysis; Ramseyer & Tschacher 2011 / rMEA) "
                    "전처리 파이프라인 [1]~[5]")
    p.add_argument("--src", default=None, help="크롭 루트 (기본: config의 cropped_root)")
    p.add_argument("--out", default=None, help="MEA 출력 루트 (기본: config의 mea_root)")
    p.add_argument("--slim-root", default=None,
                   help="head-pose slim csv 루트 (기본: <output_root>_slim)")
    p.add_argument("--group", "-g", default=None, nargs="+")
    p.add_argument("--participant", "-p", default=None)
    p.add_argument("--threshold", type=int, default=12,
                   help="[1] 프레임 절대차 노이즈 게이트 (0-255, 기본 12)")
    p.add_argument("--roi", default=None, help="ROI json 경로 (기본: config의 mea_roi)")
    p.add_argument("--sd-threshold", type=float, default=10.0,
                   help="[2] MEA 이상치 판정 SD 배수 (기본 10)")
    p.add_argument("--pose-sd-threshold", type=float, default=0.0,
                   help="[2] head-pose(Rx/Ry/Rz) 이상치 SD 배수 (0=사용 안 함, 기본 0)")
    p.add_argument("--smooth-sec", type=float, default=0.5,
                   help="[4] 이동평균 창 길이(초, 기본 0.5)")
    p.add_argument("--pad-sec", type=float, default=0.5,
                   help="[2] 이상치 프레임 앞뒤 확장(초, 기본 0.5)")
    p.add_argument("--zscore-scope", choices=["session", "participant"], default="session",
                   help="[5] z 표준화 단위 (기본 session = 참가자-세션)")
    p.add_argument("--reprocess", action="store_true",
                   help="영상 재디코드 없이 기존 raw csv 로 [2]~[5]만 다시 계산")
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


_SESSION_RE = re.compile(r"^(?P<base>.+)_(?P<participant>P\d+)_S(?P<session>\d+)$")


def find_session_clips(src_root, groups=None, participant=None):
    clips = []
    for group in sorted(os.listdir(src_root)):
        gdir = os.path.join(src_root, group)
        if not os.path.isdir(gdir):
            continue
        if groups and group not in groups:
            continue
        for fname in sorted(os.listdir(gdir)):
            stem, ext = os.path.splitext(fname)
            if ext.lower() != ".mp4":
                continue
            m = _SESSION_RE.match(stem)
            if not m:
                continue
            if participant and m.group("participant") != participant:
                continue
            clips.append((group, m.group("participant"), m.group("session"),
                          stem, os.path.join(gdir, fname)))
    return clips


def pipe_signature(sd_k, pose_sd, smooth_sec, pad_sec, zscope):
    return f"sd{sd_k:g}_psd{pose_sd:g}_sm{smooth_sec:g}_pad{pad_sec:g}_z-{zscope}"


# ── [1] Raw MEA — Motion Energy Analysis (grayscale frame differencing) ──────

def frame_differencing(video_path, threshold, exclude_rects=None):
    """[1] Motion Energy Analysis (Ramseyer & Tschacher, 2011).

    연속 프레임을 그레이스케일로 변환 → 절대차 D_t = |gray_t - gray_{t-1}| →
    노이즈 게이트(D_t <= threshold 는 0) → ROI(exclude_rects 제외 영역) 내 합이
    프레임 t 의 motion energy. mea_ppx 는 ROI 픽셀수로 나눈 값(rMEA 의 ROI 크기
    보정에 해당하나, 최종 [5] z 표준화가 같은 역할을 하므로 정보용).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상 열기 실패: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    exclude_rects = exclude_rects or []
    keep = build_keep_mask(h, w, exclude_rects)
    roi_npx = int(keep.sum())
    full_mask = keep.all()

    mea, n_active, luma = [], [], []
    prev = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        luma.append(float(gray.mean()) if full_mask else float(gray[keep].mean()))
        if prev is None:
            mea.append(0.0)
            n_active.append(0)
        else:
            d = cv2.absdiff(gray, prev)
            active = d > threshold
            if not full_mask:
                active &= keep
            d_gated = np.where(active, d, 0)
            mea.append(float(d_gated.sum(dtype=np.int64)))
            n_active.append(int(active.sum()))
        prev = gray
    cap.release()

    n = len(mea)
    idx = np.arange(n)
    mea = np.asarray(mea, float)
    df = pd.DataFrame({
        "frame": idx,
        "timestamp": (idx / fps).round(6),
        "mea": mea,
        "mea_ppx": (mea / max(roi_npx, 1)).round(6),
        "n_active_px": n_active,
        "frac_active": (np.asarray(n_active) / max(roi_npx, 1)).round(6),
        "mean_luma": np.round(luma, 4),
    })
    meta = {"width": w, "height": h, "npx": w * h, "roi_npx": roi_npx,
            "roi_exclude": [list(r) for r in exclude_rects],
            "roi_signature": rects_signature(exclude_rects),
            "fps": round(fps, 4), "nframes": n, "threshold": threshold}
    return df, meta


# ── MEA 프레임워크 연산 [2][4][5] ────────────────────────────────────────────
# 각 함수는 rMEA (Kleinbub & Ramseyer, 2021) 의 대응 연산과 같은 정의를 따른다.

def _sd_outlier_mask(x, k, valid=None):
    """valid(기본 전체)에서 mean/SD 계산 후 |x-mean| > k·SD 인 위치를 True 로.

    rMEA::MEAoutlier(threshold = function(x) k*sd(x)) 과 같은 기준. rMEA 기본
    direction="greater"(상측만) 와 달리 양측(|x-mean|)이나, motion energy 이상치는
    상측 스파이크라 실질 동일.
    """
    x = np.asarray(x, float)
    m = np.isfinite(x) if valid is None else (valid & np.isfinite(x))
    if m.sum() < 10:
        return np.zeros(len(x), bool)
    mu, sd = x[m].mean(), x[m].std()
    if not sd > 0:
        return np.zeros(len(x), bool)
    out = np.abs(x - mu) > k * sd
    out[~np.isfinite(x)] = False
    return out


def mea_outlier(mea, mask, fps, pad_sec):
    """[2] rMEA::MEAoutlier 대응 — 이상치 프레임 제거 후 선형보간 → mea_clean.

    mask    : 판정된 이상치 불리언 (pad 확장 전; MEA + head-pose 병합본)
    pad_sec : 이상치 앞뒤 확장 폭(초). rMEA 에 없는 확장(스파이크 여파 제거).
    반환    : (mea_clean, 최종 outlier 마스크)
    """
    mea = np.asarray(mea, float)
    pad = int(round(pad_sec * fps))
    if pad > 0 and mask.any():
        kern = np.ones(2 * pad + 1)
        mask = np.convolve(mask.astype(int), kern, mode="same") > 0

    clean = mea.copy()
    clean[mask] = np.nan                          # rMEA: replace = NA
    clean = pd.Series(clean).interpolate(limit_direction="both").to_numpy()
    if not np.isfinite(clean).all():              # 전부 NaN 등 예외
        fill = np.nanmean(mea) if np.isfinite(mea).any() else 0.0
        clean = np.nan_to_num(clean, nan=fill)
    return clean, mask


def mea_smooth(x, smooth_sec, fps):
    """[4] rMEA::MEAsmooth 대응 — 중심이동평균 (기본 창 0.5s = rMEA 기본값).
    반환: (mea_smooth, 실제 창 길이[프레임, 홀수])"""
    win = max(1, int(round(smooth_sec * fps)))
    if win % 2 == 0:
        win += 1
    sm = pd.Series(x).rolling(win, center=True, min_periods=1).mean().to_numpy()
    return sm, win


def mea_scale(x):
    """[5] rMEA::MEAscale 계열 — 여기서는 SD 표준화(z). 반환: (mea_z, mean, sd)"""
    mu, sd = float(np.nanmean(x)), float(np.nanstd(x))
    z = (x - mu) / sd if sd > 0 else np.zeros(len(x))
    return z, mu, sd


# ── [1]~[5] 오케스트레이션 ───────────────────────────────────────────────────

def process_pipeline(df, slim_path, fps, sd_k=10.0, pose_sd=0.0, smooth_sec=0.5,
                     pad_sec=0.5, zscope="session"):
    df = df[[c for c in RAW_COLS if c in df.columns]].copy()
    n = len(df)
    mea = df["mea"].to_numpy(float)

    # [2a] MEA 이상치 (0번 프레임은 인위적 0이므로 통계/판정에서 제외)
    core_valid = np.ones(n, bool)
    core_valid[0] = False
    mea_out = _sd_outlier_mask(mea, sd_k, valid=core_valid)

    # [2b] head-pose 이상치 — MEA 프레임워크에 없는 확장. pose_sd>0 일 때만,
    #      success=1 프레임에서만 판정 (기본 0 = 사용 안 함. MEA 와 관계있는 건
    #      각도 '위치'가 아니라 '속도'라 위치 SD 마스킹은 거의 무의미)
    pose_out = np.zeros(n, bool)
    success = np.full(n, -1, int)
    slim_ok = False
    if slim_path and os.path.isfile(slim_path):
        usecols = ["success"] + POSE_AXES
        sl = pd.read_csv(slim_path, usecols=lambda c: c in usecols, encoding="utf-8-sig")
        m = min(len(sl), n)
        if m > 0 and "success" in sl.columns:
            slim_ok = True
            succ_col = sl["success"].to_numpy()[:m]
            success[:m] = succ_col
            succ = succ_col == 1
            if pose_sd and pose_sd > 0:
                for ax in POSE_AXES:
                    if ax not in sl.columns:
                        continue
                    v = sl[ax].to_numpy(float)[:m]
                    bad = _sd_outlier_mask(v, pose_sd, valid=succ)
                    bad &= succ                  # 추적실패 프레임은 마스킹하지 않음
                    pose_out[:m] |= bad

    # [2c] 이상치 병합 → pad 확장 → NaN → 선형보간
    mea_clean, outlier = mea_outlier(mea, mea_out | pose_out, fps, pad_sec)

    # [4] 스무딩
    mea_smooth_arr, win = mea_smooth(mea_clean, smooth_sec, fps)

    # [5] z 표준화 (참가자-세션). participant 스코프는 메인에서 2차 패스로 덮어씀
    mea_z, zmu, zsd = mea_scale(mea_smooth_arr)

    df["success"] = success
    df["outlier"] = outlier.astype(int)
    df["mea_clean"] = np.round(mea_clean, 4)
    df["mea_smooth"] = np.round(mea_smooth_arr, 6)
    df["mea_z"] = np.round(mea_z, 6)

    stats = {
        "mea_out_pct": round(100 * mea_out.mean(), 3),
        "pose_out_pct": round(100 * pose_out.mean(), 3),
        "pose_sd_used": pose_sd if pose_sd and pose_sd > 0 else None,
        "outlier_pct": round(100 * outlier.mean(), 3),
        "slim_found": slim_ok,
        "success_rate": round(float((success == 1).mean()), 4) if slim_ok else None,
        "smooth_win": win,
        "z_scope": zscope,
        "z_mean_session": zmu,
        "z_std_session": zsd,
    }
    return df, stats


# ── 워커 ─────────────────────────────────────────────────────────────────────

def safe_process(task):
    (group, pid, sess, stem, path, out_root, slim_root, threshold, exclude_rects,
     sd_k, pose_sd, smooth_sec, pad_sec, zscope, reprocess) = task
    out_dir = os.path.join(out_root, group)
    out_csv = os.path.join(out_dir, f"{stem}_mea.csv")
    meta_path = os.path.join(out_dir, "mea_meta.json")
    want_roi = rects_signature(exclude_rects)
    want_pipe = pipe_signature(sd_k, pose_sd, smooth_sec, pad_sec, zscope)
    slim_path = os.path.join(slim_root, group, f"{stem}_slim.csv")

    prev_meta = {}
    if os.path.isfile(meta_path):
        try:
            prev_meta = json.load(open(meta_path, encoding="utf-8")).get(stem, {})
        except Exception:
            prev_meta = {}

    raw_ok = (os.path.isfile(out_csv) and os.path.getsize(out_csv) > 0
              and prev_meta.get("threshold") == threshold
              and (prev_meta.get("roi_signature") == want_roi
                   or (prev_meta.get("roi_signature") is None and not exclude_rects)))

    if (not reprocess and raw_ok
            and prev_meta.get("pipe_signature") == want_pipe):
        return ("skip", stem, None)

    try:
        if raw_ok:
            df = pd.read_csv(out_csv, encoding="utf-8-sig")
            fps = float(prev_meta.get("fps") or 25.0)
            meta = {k: prev_meta[k] for k in
                    ("width", "height", "npx", "roi_npx", "roi_exclude",
                     "roi_signature", "fps", "nframes", "threshold") if k in prev_meta}
            if "roi_signature" not in meta:
                meta["roi_signature"] = want_roi
        else:
            df, meta = frame_differencing(path, threshold, exclude_rects)
            fps = meta["fps"]

        proc, stats = process_pipeline(df, slim_path, fps, sd_k=sd_k, pose_sd=pose_sd,
                                       smooth_sec=smooth_sec, pad_sec=pad_sec,
                                       zscope=zscope)
        os.makedirs(out_dir, exist_ok=True)
        proc.to_csv(out_csv, index=False, encoding="utf-8-sig")

        meta.update({
            "pipe_signature": want_pipe,
            "sd_threshold": sd_k, "pose_sd_threshold": pose_sd,
            "smooth_sec": smooth_sec, "pad_sec": pad_sec,
            **stats,
        })
        return ("done", stem, meta)
    except Exception as e:
        import traceback
        return ("fail", f"{stem} → {e}\n{traceback.format_exc()}", None)


def _fmt_sec(sec):
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}시간 {m}분 {s}초" if h else (f"{m}분 {s}초" if m else f"{s}초")


def _write_group_meta(out_root, metas):
    by_group = defaultdict(dict)
    for name, meta in metas.items():
        mm = _SESSION_RE.match(name)
        if mm:
            by_group[mm.group("base")][name] = meta
    for group, entries in by_group.items():
        mp = os.path.join(out_root, group, "mea_meta.json")
        existing = {}
        if os.path.isfile(mp):
            try:
                existing = json.load(open(mp, encoding="utf-8"))
            except Exception:
                existing = {}
        existing.update(entries)
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)


def _apply_participant_zscore(out_root, clips):
    """[5] participant 스코프: 같은 (group, 참가자)의 모든 세션 mea_smooth 를 모아
    공통 mean/SD 로 mea_z 를 다시 계산해 각 csv 에 덮어씀."""
    bucket = defaultdict(list)
    for (g, p, s, stem, _path) in clips:
        c = os.path.join(out_root, g, f"{stem}_mea.csv")
        if os.path.isfile(c):
            bucket[(g, p)].append(c)
    for (g, p), csvs in sorted(bucket.items()):
        allv = np.concatenate([pd.read_csv(c, usecols=["mea_smooth"],
                                           encoding="utf-8-sig")["mea_smooth"].to_numpy()
                               for c in csvs])
        mu, sd = float(np.nanmean(allv)), float(np.nanstd(allv))
        for c in csvs:
            d = pd.read_csv(c, encoding="utf-8-sig")
            d["mea_z"] = np.round((d["mea_smooth"] - mu) / sd, 6) if sd > 0 else 0.0
            d.to_csv(c, index=False, encoding="utf-8-sig")
    print(f"[INFO] participant 스코프 z 적용: {len(bucket)}명 (참가자×그룹)")


def main():
    args = parse_args()
    src_root = args.src or str(path('mea_video_dir'))
    out_root = args.out or str(path('mea_dir'))
    slim_root = args.slim_root or str(path('head_dir'))
    num_workers = args.workers

    if not os.path.isdir(src_root):
        print(f"[오류] 크롭 루트 없음: {src_root}")
        sys.exit(1)
    if not os.path.isdir(slim_root):
        print(f"[경고] slim 루트 없음: {slim_root} — head-pose 마스킹 생략됨")

    os.makedirs(out_root, exist_ok=True)
    roi_path = args.roi or str(path('roi_config'))
    roi_cfg = load_roi_config(roi_path)
    if roi_cfg:
        print(f"[INFO] ROI 설정: {roi_path}")
    print(f"[INFO] slim(head-pose) 루트: {slim_root}")

    clips = find_session_clips(src_root, groups=args.group, participant=args.participant)
    if not clips:
        print(f"[경고] 대상 클립 없음: {src_root}")
        return

    tasks = [(g, p, s, stem, path, out_root, slim_root, args.threshold,
              get_exclude_rects(roi_cfg, g, s, p),
              args.sd_threshold, args.pose_sd_threshold, args.smooth_sec, args.pad_sec,
              args.zscore_scope, args.reprocess)
             for (g, p, s, stem, path) in clips]
    total = len(tasks)
    pose_txt = f"pose_sd={args.pose_sd_threshold}" if args.pose_sd_threshold > 0 else "pose_mask=off"
    print(f"[INFO] 파이프라인 [1]~[5] — {total}개 클립 | "
          f"th={args.threshold} sd={args.sd_threshold} {pose_txt} smooth={args.smooth_sec}s "
          f"pad={args.pad_sec}s z={args.zscore_scope} | 병렬 {num_workers}")

    done = skip = fail = 0
    metas = {}
    start = time.time()
    try:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(safe_process, t) for t in tasks]
            for fut in as_completed(futures):
                try:
                    status, name, meta = fut.result()
                except BrokenProcessPool:
                    print("[오류] 워커 강제 종료 — 중단", flush=True)
                    raise
                if status == "done":
                    done += 1
                    metas[name] = meta
                elif status == "skip":
                    skip += 1
                else:
                    fail += 1
                    print(f"[실패] {name}", flush=True)
                comp = done + skip + fail
                el = time.time() - start
                eta = (el / max(done, 1) * (total - comp)) if done else 0
                print(f"[진행] {comp}/{total} | 완료 {done} 스킵 {skip} 실패 {fail} "
                      f"| 경과 {_fmt_sec(el)} | 잔여 {_fmt_sec(eta)}", flush=True)
    except KeyboardInterrupt:
        print("[종료 요청됨]")
        sys.exit(1)

    _write_group_meta(out_root, metas)

    if args.zscore_scope == "participant":
        _apply_participant_zscore(out_root, clips)

    # 마스킹율 요약
    if metas:
        opct = np.array([m["outlier_pct"] for m in metas.values() if m.get("outlier_pct") is not None])
        if len(opct):
            print(f"[요약] outlier 마스킹율: 평균 {opct.mean():.2f}% / 중앙값 {np.median(opct):.2f}% "
                  f"/ 최대 {opct.max():.2f}%")
        worst = sorted(((m.get("outlier_pct", 0), n) for n, m in metas.items()), reverse=True)[:8]
        for pct, n in worst:
            print(f"        {n}: {pct:.2f}%")

    print(f"[완료] done={done} skip={skip} fail={fail} | 총 {_fmt_sec(time.time()-start)}")
    print(f"[출력] {out_root}")
    if fail:
        raise RuntimeError(f"{fail} MEA clip(s) failed")


if __name__ == "__main__":
    main()
