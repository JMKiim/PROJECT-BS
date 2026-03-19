import os
import shutil
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from process_single_video import process_video

INPUT_ROOT = "D:/2025신윤희영상정렬"
OUTPUT_ROOT = "D:/2025신윤희Data/MediaPipe"
# [수정] 모든 학기 처리
SEMESTERS = ["23-2", "24-1", "24-2"]
VIDEO_EXTENSIONS = [".mp4", ".mov", ".mkv"]
NUM_WORKERS = 4

SKIP_LOG = []
FAIL_LOG = []

# -------------------------------
# GPU 상태 확인
# -------------------------------
def check_gpu_status():
    try:
        import os
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"[INFO] GPU 사용 가능: {[gpu.name for gpu in gpus]}")
        else:
            print("[경고] GPU를 찾지 못함 → CPU fallback 가능성 있음")
    except Exception as e:
        print(f"[경고] GPU 상태 체크 실패: {e} (계속 진행)")

# -------------------------------
# 출력 폴더 초기화
# -------------------------------
def clear_output_folder():
    if os.path.exists(OUTPUT_ROOT):
        print(f"[INFO] 기존 출력 폴더 삭제: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

# -------------------------------
# 입력 영상 찾기
# -------------------------------
def find_all_timeline_videos():
    tasks = []
    for semester in SEMESTERS:
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

# -------------------------------
# 안전 처리: 누락된 경우에만 실행 (수정된 함수)
# -------------------------------
def safe_process(task):
    input_path, output_dir = task

    # 주요 수정 사항 1: 예상되는 '개별 출력 파일'의 전체 경로를 계산합니다.
    # (결과물이 .csv 파일로 저장된다고 가정)
    input_basename = os.path.splitext(os.path.basename(input_path))[0]
    expected_output_file = os.path.join(output_dir, f"{input_basename}.csv")

    # 입력이 없으면 스킵
    if not os.path.isfile(input_path):
        SKIP_LOG.append(input_path)
        return
    
    # 주요 수정 사항 2: 스킵 조건을 '폴더'가 아닌 '개별 파일'의 존재 여부로 변경합니다.
    if os.path.exists(expected_output_file):
        print(f"[스킵] 이미 처리된 파일: {expected_output_file}")
        return
    
    # 처리
    try:
        process_video((input_path, output_dir))
    except Exception as e:
        FAIL_LOG.append(input_path)
        print(f"[실패] {input_path} → {e}")

# -------------------------------
# 명령행 인자
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run OpenFace over cropped individual videos.")
    parser.add_argument("--clear", "-c", action="store_true",
                        help="If set, clear the output root before processing (default: keep existing).")
    return parser.parse_args()

# -------------------------------
# 메인
# -------------------------------
if __name__ == "__main__":
    args = parse_args()
    check_gpu_status()
    if args.clear:
        clear_output_folder()
    else:
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