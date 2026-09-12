"""Crop group recordings into timed participant clips."""
from bs.settings import path, tool
import os
import subprocess
import pandas as pd
import cv2
# 설정
ROOT_DIR = str(path('raw_video_dir'))
CROP_DIR = str(path('cropped_video_dir'))
CSV_PATH = str(path('timeline_metadata'))  # 같은 폴더에 있어야 함
VIDEO_EXTENSIONS = [".mp4", ".mov", ".mkv"]
# 유틸 함수

def get_video_resolution(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"비디오 열기 실패: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

def get_crop_coords(num_people, width, height):
    if num_people == 4:
        return [
            (0, 0, width // 2, height // 2),
            (width // 2, 0, width // 2, height // 2),
            (0, height // 2, width // 2, height // 2),
            (width // 2, height // 2, width // 2, height // 2)
        ]
    elif num_people == 3:
        half_w = width // 2
        half_h = height // 2
        return [
            (0, 0, half_w, half_h),                     # 좌상
            (half_w, 0, width - half_w, half_h),            # 우상
            ((width - half_w) // 2, half_h, half_w, height - half_h)  # 중앙 하단
        ]
    elif num_people == 5:
        w = int(width * 0.325)
        h = int(height * 0.325)
        top_left_w    = width/2 - (3*w)/2
        top_left_h    = height/2 - h
        bottom_left_w = width/2 - w
        mid_y         = height/2
        return [
            (int(top_left_w),           int(top_left_h),     w, h),  # P1: 상단 왼쪽
            (int(top_left_w) + w,       int(top_left_h),     w, h),  # P2: 상단 중앙
            (int(top_left_w) + 2 * w,   int(top_left_h),     w, h),  # P3: 상단 오른쪽
            (int(bottom_left_w),        int(mid_y),          w, h),  # P4: 하단 왼쪽
            (int(width/2),              int(mid_y),          w, h)   # P5: 하단 오른쪽
        ]
    elif num_people == 6:
        w = int(width * 0.325)
        h = int(height * 0.325)
        top_left_w    = width/2 - (3*w)/2
        top_left_h    = height/2 - h
        bottom_left_w = width/2 - w
        mid_y         = height/2
        return [
            (int(top_left_w),           int(top_left_h),     w, h),  # P1: 상단 왼쪽
            (int(top_left_w) + w,       int(top_left_h),     w, h),  # P2: 상단 중앙
            (int(top_left_w) + 2 * w,   int(top_left_h),     w, h),  # P3: 상단 오른쪽
            (int(top_left_w),           int(mid_y),          w, h),  # P4: 하단 왼쪽
            (int(top_left_w) + w,       int(mid_y),          w, h),  # P5: 하단 중앙
            (int(top_left_w) + 2 * w,   int(mid_y),          w, h)   # P6: 하단 오른쪽
        ]
    elif num_people == 7:
        w = int(width * 0.325)
        h = int(height * 0.325)
        top_left_w    = width/2 - (3*w)/2
        top_left_h    = height/2 - (3*h)/2
        bottom_left_w = width/2 - w
        mid_y         = height/2
        return [
            (int(top_left_w),           int(top_left_h),     w, h),  # P1: 상단 왼쪽
            (int(top_left_w) + w,       int(top_left_h),     w, h),  # P2: 상단 중앙
            (int(top_left_w) + 2 * w,   int(top_left_h),     w, h),  # P3: 상단 오른쪽
            (int(top_left_w),           int(top_left_h) + h, w, h),  # P4: 중단 왼쪽
            (int(top_left_w) + w,       int(top_left_h) + h, w, h),  # P5: 중단 중앙
            (int(top_left_w) + 2 * w,   int(top_left_h) + h, w, h),  # P6: 중단 오른쪽
            (int(top_left_w) + w,   int(top_left_h) + 2 * h, w, h)   # P7: 하단 중앙
        ]
    else:
        print(f"[경고] 인원 수 {num_people}명에 대한 crop 규칙이 정의되어 있지 않음")
        return None

def run_ffmpeg_cut(input_path, start_time, end_time, output_path):
    cmd = [
        tool('ffmpeg'), "-y",
        "-ss", start_time, "-to", end_time,
        "-i", input_path,
        "-r", "15",
        "-c:v", "libx264",
        "-c:a", "aac",
        output_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

def run_ffmpeg_crop(input_path, x, y, w, h, output_path):
    crop_filter = f"crop={w}:{h}:{x}:{y}"
    cmd = [
        tool('ffmpeg'), "-y",
        "-i", input_path,
        "-vf", crop_filter,
        "-c:v", "libx264",
        "-c:a", "aac",
        output_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

def ensure_timeline_and_crops(group, week, version, t_idx, n_people, input_path, start, end, out_folder):
    folder_name = f"{group}_{week}{'' if version == '없음' else version}"
    timeline_name = f"{folder_name}_T{t_idx}.mp4"
    timeline_path = os.path.join(out_folder, timeline_name)

    if not os.path.exists(timeline_path):
        run_ffmpeg_cut(input_path, start, end, timeline_path)

    # 2. crop 생성 (각 participant)
    try:
        width, height = get_video_resolution(input_path)
    except Exception as e:
        print(f"[에러] 해상도 얻기 실패: {e}")
        return
    coords = get_crop_coords(n_people, width, height)
    if not coords or len(coords) != n_people:
        print(f"[에러] crop 좌표 개수 불일치: 기대 {n_people}, 받은 {len(coords) if coords else 0}")
        return

    for pid, (x, y, w, h) in enumerate(coords, start=1):
        crop_name = f"{folder_name}_T{t_idx}_P{pid}.mp4"
        crop_path = os.path.join(out_folder, crop_name)
        print(f"[생성] crop 영상 만들기: {crop_name}")
        if not os.path.exists(crop_path):
            run_ffmpeg_crop(timeline_path, x, y, w, h, crop_path)
# 메인 처리 루프

def process_all_videos():
    df = pd.read_csv(CSV_PATH)
    for semester in sorted(df["학기"].dropna().astype(str).unique()):
        semester_path = os.path.join(ROOT_DIR, semester)
        if not os.path.isdir(semester_path):
            continue

        for group_name in os.listdir(semester_path):
            group_path = os.path.join(semester_path, group_name)
            if not os.path.isdir(group_path):
                continue

            for file in os.listdir(group_path):
                if not any(file.endswith(ext) for ext in VIDEO_EXTENSIONS):
                    continue

                base_name = os.path.splitext(file)[0]
                input_path = os.path.join(group_path, file)

                # 그룹명, 주차, 버전 추출
                parts = base_name.split("_")
                group = parts[0]
                week_with_version = parts[1] if len(parts) > 1 else ""
                if "(" in week_with_version:
                    week = week_with_version[:week_with_version.index("(")]
                    version = week_with_version[week_with_version.index("("):]
                else:
                    week = week_with_version
                    version = "없음"

                # 타임라인 필터링
                matched = df[
                    (df["학기"] == semester) &
                    (df["그룹명"] == group) &
                    (df["주차"] == week) &
                    (df["파일버전"] == version)
                ]

                if matched.empty:
                    print(f"[스킵] 타임라인 정보 없음: {file}")
                    continue

                print(f"[처리 중] {file}")
                # 출력 폴더 확보 (있어도 누락만 채움)
                folder_name = f"{group}_{week}{'' if version == '없음' else version}"
                out_folder = os.path.join(CROP_DIR, semester, group, folder_name)
                os.makedirs(out_folder, exist_ok=True)

                for _, row in matched.iterrows():
                    t_idx = row["타임라인인덱스"]
                    start = row["시작시간"]
                    end = row["종료시간"]
                    n_people = int(row["인원수"])

                    ensure_timeline_and_crops(group, week, version, t_idx, n_people, input_path, start, end, out_folder)

    print("[완료] 모든 비디오 처리 완료.")

# 실행
if __name__ == "__main__":
    process_all_videos()
