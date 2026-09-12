"""Run OpenFace on participant clips."""
from bs.settings import path, tool
import os
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from bs.extraction.openface import process_video

INPUT_ROOT = str(path('cropped_video_dir'))
OUTPUT_ROOT = str(path('features_dir'))
# 모든 학기 처리
VIDEO_EXTENSIONS = [".mp4", ".mov", ".mkv"]
NUM_WORKERS = 4

SKIP_LOG = []
FAIL_LOG = []
def find_all_timeline_videos():
    tasks = []
    for semester in sorted(os.listdir(INPUT_ROOT)):
        sem_path = os.path.join(INPUT_ROOT, semester)
        if not os.path.isdir(sem_path):
            continue

        for group in os.listdir(sem_path):
            group_path = os.path.join(sem_path, group)
            if not os.path.isdir(group_path):
                continue

            for week in os.listdir(group_path):
                week_path = os.path.join(group_path, week)
                if not os.path.isdir(week_path):
                    continue

                for file in os.listdir(week_path):
                    if not any(file.endswith(ext) for ext in VIDEO_EXTENSIONS):
                        continue

                    base = os.path.splitext(file)[0]
                    if "_T" not in base or "_P" not in base:
                        continue

                    try:
                        group_name, week_name, timeline, _ = base.split("_")
                        timeline_idx = timeline[1:]
                    except:
                        continue

                    input_path = os.path.join(week_path, file)
                    output_dir = os.path.join(OUTPUT_ROOT, semester, group_name, week_name, f"T{timeline_idx}")
                    tasks.append((input_path, output_dir))
    return tasks
def safe_process(task):
    input_path, output_dir = task
    # (결과물이 .csv 파일로 저장된다고 가정)
    input_basename = os.path.splitext(os.path.basename(input_path))[0]
    expected_output_file = os.path.join(output_dir, f"{input_basename}.csv")

    # 입력이 없으면 스킵
    if not os.path.isfile(input_path):
        SKIP_LOG.append(input_path)
        return
    if os.path.exists(expected_output_file):
        print(f"[스킵] 이미 처리된 파일: {expected_output_file}")
        return

    # 처리
    try:
        process_video((input_path, output_dir))
    except Exception as e:
        FAIL_LOG.append(input_path)
        raise RuntimeError(f"OpenFace failed: {input_path}") from e
# 명령행 인자
def parse_args():
    parser = argparse.ArgumentParser(description="Run OpenFace over cropped individual videos.")
    return parser.parse_args()
# 메인
if __name__ == "__main__":
    args = parse_args()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    tasks = find_all_timeline_videos()
    print(f"[INFO] 총 {len(tasks)}개 비디오 중 누락된 것만 처리합니다. (병렬 {NUM_WORKERS} workers)")

    try:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(safe_process, t) for t in tasks]
            for future in as_completed(futures):
                future.result()
    except KeyboardInterrupt:
        print("[종료 요청됨] Ctrl+C 감지")
        sys.exit(1)

    print("[전체 처리 완료]")
    if SKIP_LOG:
        print(f"[스킵된 {len(SKIP_LOG)}개 입력]")
    if FAIL_LOG:
        print(f"[실패한 {len(FAIL_LOG)}개 입력]")
