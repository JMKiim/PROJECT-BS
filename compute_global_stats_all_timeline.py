import os
import json
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# --------------------------
# 설정
# --------------------------
MEDIA_PIPE_ROOT = "D:/2025신윤희Data/MediaPipe"
CONFIG_PATH = "config_indicators.json"

# --------------------------
# 공통 config 불러오기
# --------------------------
with open(CONFIG_PATH, "r") as f:
    INDICATOR_CONFIG = json.load(f)

# --------------------------
# 참가자 통계 계산 함수
# --------------------------
def process_participant(args):
    timeline_dir, pid, df = args
    stats = {}
    for key, config in INDICATOR_CONFIG.items():
        # zscore가 True로 설정된 지표만 통계에 포함
        if config.get("zscore", False):
            col = config.get("column")
            try:
                values = df[col].astype(float)
                # 성공 프레임만 사용
                if "success" in df.columns:
                    values = values[df["success"] == 1]

                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                # [추가] 어디서 비었는지/샘플이 적은지 로깅
                n = values.dropna().shape[0]
                if n == 0:
                    print(f"[WARN][EMPTY] {timeline_dir} | pid={pid} | indicator={key} | column={col} | n=0")
                elif n < 2:
                    print(f"[WARN][LOW N] {timeline_dir} | pid={pid} | indicator={key} | column={col} | n={n}")
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                mean = np.nanmean(values)
                std = np.nanstd(values)
                stats[key] = {"mean": mean, "std": std}
            except KeyError:
                # 컬럼 자체가 없을 때도 찍어두면 추적 쉬움 (원하면 주석 해제)
                print(f"[WARN][MISSING COL] {timeline_dir} | pid={pid} | indicator={key} | column={col} 없음")
                continue
    return pid, stats

# --------------------------
# 타임라인별 처리 함수
# --------------------------
# --------------------------
# 타임라인별 처리가 아닌 주차(Week)별 통합 처리 함수
# --------------------------
def process_week_group(week_dir):
    # 1. 주차 내 모든 타임라인 폴더 수집
    timeline_folders = [
        os.path.join(week_dir, d) 
        for d in os.listdir(week_dir) 
        if os.path.isdir(os.path.join(week_dir, d))
    ]
    if not timeline_folders:
        return

    print(f"[주차 통합 처리] {week_dir} (총 {len(timeline_folders)}개 타임라인)")
    
    # 2. 모든 타임라인의 데이터를 메모리에 로드 (PID별로 통합)
    #    구조: aggregated_data = { 'P1': [df1, df2...], 'P2': [...] }
    aggregated_data = {}
    
    for t_dir in timeline_folders:
        csv_files = [f for f in os.listdir(t_dir) if f.endswith("_augmented.csv")]
        for file in csv_files:
            pid = file.replace("_augmented.csv", "").split("_")[-1]  # P1, P2...
            csv_path = os.path.join(t_dir, file)
            try:
                df = pd.read_csv(csv_path)
                if pid not in aggregated_data:
                    aggregated_data[pid] = []
                aggregated_data[pid].append(df)
            except Exception as e:
                print(f"[에러] 로드 실패: {csv_path} - {e}")

    if not aggregated_data:
        print(f"[스킵] 데이터 없음: {week_dir}")
        return

    # 3. PID별로 병합하여 통계 계산
    week_stats = {}
    
    # 병렬 처리 대신 단순 루프로 처리 (데이터 병합 오버헤드 고려)
    for pid, dfs in aggregated_data.items():
        if not dfs: continue
        full_df = pd.concat(dfs, ignore_index=True)
        
        # calculate_stats_for_df는 아래 helper 함수로 분리
        pid_stat = calculate_stats_for_df(full_df, pid, week_dir)
        week_stats[pid] = pid_stat

    # 4. 계산된 주차별 통계(week_stats)를 각 타임라인 폴더에 '배포'
    #    이렇게 하면 single_visualizer.py를 수정하지 않아도 됨
    for t_dir in timeline_folders:
        out_path = os.path.join(t_dir, "global_stats.json")
        with open(out_path, "w", encoding='utf-8') as f:
            json.dump(week_stats, f, indent=2, ensure_ascii=False)
        print(f"  -> 저장 완료: {out_path}")

def calculate_stats_for_df(df, pid, distinct_name):
    # 기존 process_participant 로직을 재사용
    stats = {}
    for key, config in INDICATOR_CONFIG.items():
        if config.get("zscore", False):
            col = config.get("column")
            try:
                values = df[col].astype(float)
                if "success" in df.columns:
                    values = values[df["success"] == 1]
                
                n = values.dropna().shape[0]
                if n < 2:
                    # 데이터 부족 시 기본값 처리
                    stats[key] = {"mean": 0.0, "std": 1.0}
                    continue

                mean = np.nanmean(values)
                std = np.nanstd(values)
                if std == 0: std = 1.0 # 분모 0 방지
                stats[key] = {"mean": float(mean), "std": float(std)}
            except KeyError:
                continue
    return stats

# --------------------------
# 전체 MediaPipe 폴더 순회
# --------------------------
def scan_all_timelines():
    for semester in os.listdir(MEDIA_PIPE_ROOT):
        sem_path = os.path.join(MEDIA_PIPE_ROOT, semester)
        if not os.path.isdir(sem_path): continue
        for group in os.listdir(sem_path):
            group_path = os.path.join(sem_path, group)
            if not os.path.isdir(group_path): continue
            for week in os.listdir(group_path):
                week_path = os.path.join(group_path, week)
                if not os.path.isdir(week_path): continue
                
                # [변경] 타임라인 단위가 아니라 'Week' 단위로 함수 호출
                process_week_group(week_path)

# --------------------------
# 실행
# --------------------------
if __name__ == "__main__":
    scan_all_timelines()
