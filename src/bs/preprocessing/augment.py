"""Combine OpenFace features and motion energy; derive analysis columns."""
from bs.settings import path, tool
import os
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# FPS for pitch velocity calculation
FPS = 15
# 감정 규칙 (AU_c 기반, 감정별 개별 threshold)
AU_RULES = {
    'happy':     (['AU06_c', 'AU12_c'], 2),  # cheek raiser + lip corner puller
    'sad':       (['AU01_c', 'AU04_c', 'AU15_c'], 3),  # inner brow raiser, brow lowerer, lip corner depressor
    'anger':     (['AU04_c', 'AU07_c', 'AU23_c'], 3),  # brow lowerer, upper lid raiser, lid tightener, lip tightener
    'disgust':   (['AU09_c', 'AU10_c', 'AU15_c', 'AU16_c'], 3),  # nose wrinkler, lip corner depressor, lower lip depressor
    'fear':      (['AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU07_c', 'AU20_c', 'AU26_c'], 5),  # multiple AU thresholds
    'surprise':  (['AU01_c', 'AU02_c', 'AU05_c', 'AU26_c'], 3),  # inner brow raiser, outer brow raiser, upper lid raiser, jaw drop
}

VALENCE_MAPPING = {
    'happy': 'positive',
    'surprise': 'positive',
    'neutral': 'neutral',
    'sad': 'negative',
    'anger': 'negative',
    'disgust': 'negative',
    'fear': 'negative'
}
# 감정 추론 함수
def predict_emotion_by_rule(row):
    for emotion, (aus, req) in AU_RULES.items():
        active = sum(row.get(au, 0) for au in aus)
        if active >= req:
            return emotion
    return 'neutral'
# bbox 넓이 계산
def calculate_bbox_area(row, filename=None):
    try:
        xs = np.array([float(row[f'x_{i}']) for i in range(68)])
        ys = np.array([float(row[f'y_{i}']) for i in range(68)])
        return (xs.max() - xs.min()) * (ys.max() - ys.min())
    except Exception as e:
        if filename:
            print(f"[bbox 오류] {filename} -> {e}")
        else:
            print(f"[bbox 오류] {e}")
        return np.nan
# 개별 CSV 처리
def process_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        # Motion Energy 정보 추가 (_grayscaled.csv)
        gray_path = csv_path.replace('.csv', '_grayscaled.csv')
        if os.path.exists(gray_path):
            df_gray = pd.read_csv(gray_path)
            if len(df) != len(df_gray):
                raise ValueError("OpenFace and ME frame counts differ")
            overlap = set(df.columns) & set(df_gray.columns)
            df_gray_only = df_gray.drop(columns=list(overlap))
            df = pd.concat([df, df_gray_only], axis=1)
            print(f'[ME 추가] {os.path.basename(gray_path)} 병합 완료')
            # ME 컬럼이 'ME' 라는 이름일 때
            df['ME_log'] = np.log(df['ME'] + 1)
            print('[ME_LOG] ME_log 컬럼 추가됨')
        else:
            print(f'[경고] {os.path.basename(gray_path)} 파일 없음')

        # bbox 계산
        # df['bbox_area'] = df.apply(calculate_bbox_area, axis=1)
        df['bbox_area'] = df.apply(lambda row: calculate_bbox_area(row, os.path.basename(csv_path)), axis=1)


        # 감정 추론
        df['emotion'] = df.apply(predict_emotion_by_rule, axis=1)
        df['valence_label'] = df['emotion'].map(lambda e: VALENCE_MAPPING.get(e, 'neutral'))

        # pitch 속도 계산
        dt = 1.0 / FPS
        if 'pose_Rx' in df.columns:
            df['pitch_vel'] = df['pose_Rx'].diff() / dt
            df['pitch_vel'] = df['pitch_vel'].fillna(0)
        else:
            df['pitch_vel'] = np.nan

        # yaw 속도 계산
        if 'pose_Ry' in df.columns:
            df['yaw_vel'] = df['pose_Ry'].diff() / dt
            df['yaw_vel'] = df['yaw_vel'].fillna(0)
        else:
            df['yaw_vel'] = np.nan

        # roll 속도 계산
        if 'pose_Rz' in df.columns:
            df['roll_vel'] = df['pose_Rz'].diff() / dt
            df['roll_vel'] = df['roll_vel'].fillna(0)
        else:
            df['roll_vel'] = np.nan

        # 저장
        new_path = csv_path.replace('.csv', '_augmented.csv')
        df.to_csv(new_path, index=False)
        print(f'[완료] {os.path.basename(new_path)} 저장됨')

    except Exception as e:
        raise RuntimeError(f"Augmentation failed: {csv_path}") from e
# 전체 폴더 순회 병렬 처리
def process_folder(root_dir):
    csv_paths = []
    for folder, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.csv') and not (file.endswith('_augmented.csv') or file.endswith('_grayscaled.csv')):
                csv_paths.append(os.path.join(folder, file))

    print(f'[시작] 총 {len(csv_paths)}개 파일 처리 중...')
    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap(process_csv, csv_paths), total=len(csv_paths)))
# 실행
if __name__ == '__main__':
    # 전체 학기/그룹/주차를 처리하도록 루트 경로 설정
    ROOT = str(path('features_dir'))
    process_folder(ROOT)
