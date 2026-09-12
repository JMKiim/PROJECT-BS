"""Run event and threshold synchrony over selected timelines."""
from bs.settings import path, tool
import os
import argparse
from typing import Optional
import bs.visualization.timeline as timeline_renderer
from bs.visualization.timeline import visualize_timeline_optimized

MEDIA_PIPE_ROOT = str(path('features_dir'))  # 최상위 루트
DEFAULT_CONFIG   = str(path('indicators'))

def has_required_inputs(timeline_dir: str) -> bool:
    """ 필수 입력이 있는지 간단 체크: global_stats.json + *_augmented.csv 존재 """
    gs = os.path.join(timeline_dir, "global_stats.json")
    if not os.path.isfile(gs):
        return False
    any_aug = any(f.endswith("_augmented.csv") for f in os.listdir(timeline_dir))
    return any_aug

def already_done(timeline_dir: str) -> bool:
    """ 결과물 존재 여부 체크: sync_counts.xlsx + sync_mask.xlsx 둘 다 있으면 완료로 간주 """
    xlsx = os.path.join(timeline_dir, "sync_counts.xlsx")
    mask = os.path.join(timeline_dir, "sync_mask.xlsx")
    return os.path.isfile(xlsx) and os.path.isfile(mask)

def iter_timelines(root: str, semester: Optional[str], group: Optional[str],
                   week: Optional[str], timeline: Optional[str]):
    """
    MEDIA_PIPE_ROOT/SEM/GROUP/WEEK/TX 구조 순회.
    필터가 주어지면 해당 값만 매칭.
    """
    for sem in sorted(os.listdir(root)):
        if semester and sem != semester:
            continue
        sem_path = os.path.join(root, sem)
        if not os.path.isdir(sem_path):
            continue

        for grp in sorted(os.listdir(sem_path)):
            if group and grp != group:
                continue
            grp_path = os.path.join(sem_path, grp)
            if not os.path.isdir(grp_path):
                continue

            for wk in sorted(os.listdir(grp_path)):
                if week and wk != week:
                    continue
                wk_path = os.path.join(grp_path, wk)
                if not os.path.isdir(wk_path):
                    continue

                for tl in sorted(os.listdir(wk_path)):
                    if timeline and tl != timeline:
                        continue
                    tl_dir = os.path.join(wk_path, tl)
                    if os.path.isdir(tl_dir):
                        yield tl_dir

def main():
    ap = argparse.ArgumentParser(description="Batch visualizer for behavioral synchrony.")
    ap.add_argument("--root", default=MEDIA_PIPE_ROOT, help="MEDIA_PIPE_ROOT 경로")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="config_indicators.json 경로")
    ap.add_argument("--semester", help="Semester directory identifier")
    ap.add_argument("--group", help="예: A4")
    ap.add_argument("--week", help="Week directory identifier")
    ap.add_argument("--timeline", help="예: T1")
    ap.add_argument("--start", type=float, default=None, help="start_time(초). None이면 전체")
    ap.add_argument("--end", type=float, default=None, help="end_time(초). None이면 전체")
    ap.add_argument("--force", action="store_true", help="기존 결과가 있어도 강제 재생성")
    ap.add_argument("--video", action="store_true", help="Render participant video alongside graphs; output stays local")
    args = ap.parse_args()
    timeline_renderer.MAKE_VIDEO = args.video

    total = 0
    done  = 0
    skipped = 0
    missing = 0
    failed = 0

    for tl_dir in iter_timelines(args.root, args.semester, args.group, args.week, args.timeline):
        total += 1
        try:
            if not has_required_inputs(tl_dir):
                print(f"[SKIP] 입력 부족 → {tl_dir} (global_stats.json 또는 *_augmented.csv 없음)")
                missing += 1
                continue

            if (not args.force) and already_done(tl_dir):
                print(f"[SKIP] 이미 생성 완료 → {tl_dir}")
                skipped += 1
                continue


            print(f"[RUN] {tl_dir}")
            visualize_timeline_optimized(
                timeline_dir=tl_dir,
                config_path=args.config,
                start_time=args.start,
                end_time=args.end
            )
            done += 1

        except KeyboardInterrupt:
            print("\n[중단] 사용자 인터럽트")
            break
        except Exception as e:
            print(f"[FAIL] {tl_dir} → {e}")
            failed += 1

    print("\n===== SUMMARY =====")
    print(f"총 후보: {total}")
    print(f"완료: {done}")
    print(f"스킵(이미 존재): {skipped}")
    print(f"입력부족: {missing}")
    print(f"실패: {failed}")
    if failed:
        raise RuntimeError(f"{failed} timeline(s) failed")

if __name__ == "__main__":
    main()
