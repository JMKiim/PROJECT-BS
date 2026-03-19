import os
import sys
import re
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------------------
# 설정
# ----------------------------
INPUT_ROOT = "D:/2025신윤희영상정렬"
OUTPUT_ROOT = "D:/2025신윤희Data/MediaPipe"
# [수정] 모든 학기 처리
SEMESTERS = ["23-2", "24-1", "24-2"]
VIDEO_EXTENSIONS = [".mp4", ".mov", ".mkv"]
NUM_WORKERS = 4

# Motion Energy 스크립트 옵션
SAVE_GRAYSCALE_VIDEO = True      # True로 설정하면 그레이스케일 영상(mp4)도 함께 저장
ENABLE_MOTION_COMP = False     # True로 설정하면 전역 이동 보정 적용
REPROCESS_EXISTING = True        # [수정] True로 설정하여 전체 재계산 (덮어쓰기)
VIDEO_CODEC = cv2.VideoWriter_fourcc(*'mp4v')

# ----------------------------
# 로그 저장용
# ----------------------------
SKIP_LOG = []
FAIL_LOG = []

# ----------------------------
# 처리할 비디오 목록 생성 (학기/그룹/주차(A4_W1) 폴더 구조 반영)
# 개인 영상(Px)만 포함
# ----------------------------
def find_all_videos():
    tasks = []
    for sem in SEMESTERS:
        sem_dir = os.path.join(INPUT_ROOT, sem)
        if not os.path.isdir(sem_dir):
            continue
        for group in os.listdir(sem_dir):
            group_dir = os.path.join(sem_dir, group)
            if not os.path.isdir(group_dir):
                continue
            for week_folder in os.listdir(group_dir):
                week_dir = os.path.join(group_dir, week_folder)
                if not os.path.isdir(week_dir):
                    continue
                parts = week_folder.split('_')
                if len(parts) != 2:
                    continue
                _, week = parts
                for fname in os.listdir(week_dir):
                    if not any(fname.endswith(ext) for ext in VIDEO_EXTENSIONS):
                        continue
                    base_name = os.path.splitext(fname)[0]
                    name_parts = base_name.split('_')
                    # 개인 비디오만 처리: 마지막 파트가 'P숫자' 패턴
                    if not re.match(r'^P\d+$', name_parts[-1]):
                        continue
                    if len(name_parts) < 3:
                        continue
                    timeline = name_parts[2]
                    input_path = os.path.join(week_dir, fname)
                    output_dir = os.path.join(OUTPUT_ROOT, sem, group, week, timeline)
                    tasks.append((input_path, output_dir))
    return tasks

# ----------------------------
# Motion Energy 계산 함수
# ----------------------------
def compute_me(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"[ERROR] 비디오 열기 실패: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    # 그레이스케일 영상 저장 준비
    writer = None
    if SAVE_GRAYSCALE_VIDEO:
        out_vid_path = os.path.join(output_dir, f"{base_name}_grayscaled.mp4")
        writer = cv2.VideoWriter(out_vid_path, VIDEO_CODEC, fps, (width, height), isColor=False)

    prev_gray = None
    me_values = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 전역 이동 보정 (옵션)
        if ENABLE_MOTION_COMP and prev_gray is not None:
            shift = cv2.phaseCorrelate(np.float32(prev_gray), np.float32(gray))
            dx, dy = shift[0]
            M = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)
            gray = cv2.warpAffine(gray, M, (width, height))

        # 그레이스케일 영상 저장
        if writer:
            writer.write(gray)

        # ME 계산 (첫 프레임에는 ME=0)
        if prev_gray is None:
            me = 0.0
        else:
            diff = cv2.absdiff(gray, prev_gray)
            # [수정] 해상도 의존성 제거: 픽셀 수로 나누어 '픽셀당 평균 변화량'으로 변경
            me = diff.sum() / (width * height)
        me_values.append(me)
        prev_gray = gray
        frame_count += 1

    cap.release()
    if writer:
        writer.release()

    # timestamp 계산 (frame 1의 timestamp=0)
    frames = list(range(1, frame_count + 1))
    timestamps = [(f - 1) / fps for f in frames]

    # DataFrame으로 저장
    df = pd.DataFrame({'frame': frames, 'timestamp': timestamps, 'ME': me_values})
    out_csv = os.path.join(output_dir, f"{base_name}_grayscaled.csv")
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved ME CSV: {out_csv}")

# ----------------------------
# 안전 처리 래퍼
# ----------------------------
def safe_process(task):
    video_path, output_dir = task
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_csv = os.path.join(output_dir, f"{base_name}_grayscaled.csv")
    # 입력 파일 확인
    if not os.path.isfile(video_path):
        SKIP_LOG.append(video_path)
        print(f"[스킵] 파일 없음: {video_path}")
        return
    # 이미 처리된 경우 처리 여부 판단
    if os.path.exists(out_csv):
        if REPROCESS_EXISTING:
            print(f"[재처리] 이미 처리된 파일: {video_path} (덮어쓰기)")
            # 기존 결과 삭제
            try:
                os.remove(out_csv)
                if SAVE_GRAYSCALE_VIDEO:
                    mp4_path = os.path.join(output_dir, f"{base_name}_grayscaled.mp4")
                    if os.path.exists(mp4_path): os.remove(mp4_path)
            except Exception:
                pass
        else:
            SKIP_LOG.append(video_path)
            print(f"[스킵] 이미 처리됨: {video_path}")
            return
    # 안전하게 ME 계산 수행
    try:
        compute_me(video_path, output_dir)
    except Exception as e:
        FAIL_LOG.append(video_path)
        print(f"[실패] {video_path} → {e}")

# ----------------------------
# 메인 실행
# ----------------------------
def main():
    tasks = find_all_videos()
    print(f"[INFO] 총 {len(tasks)}개 개인 비디오를 병렬({NUM_WORKERS}) 처리합니다.")
    try:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(safe_process, t) for t in tasks]
            for future in as_completed(futures):
                future.result()
    except KeyboardInterrupt:
        print("\n[종료 요청됨] Ctrl+C 입력 감지 → 즉시 중단")
        executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)

    print("\n[전체 처리 완료]")
    if SKIP_LOG:
        print(f"[스킵된 {len(SKIP_LOG)}개 개인 비디오]")
        for p in SKIP_LOG:
            print(" -", p)
    if FAIL_LOG:
        print(f"[실패한 {len(FAIL_LOG)}개 개인 비디오]")
        for p in FAIL_LOG:
            print(" -", p)

if __name__ == '__main__':
    main()
