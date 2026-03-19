import pandas as pd
import numpy as np
import os
import glob
import re
from openpyxl.utils import get_column_letter

# ==============================================================================
# [설정 영역]
# ==============================================================================
BASE_DIR = r"D:\2025신윤희Data\MediaPipe"
OUTPUT_DIR = r"D:\2025EE_Final_Output_Wide_Optimized" # 최종 폴더명

TIMELINE_INFO_PATH = r"D:\2025신윤희Code\timeline_info.csv"
SESSION_TIMELINE_PATH = r"D:\2025신윤희Code\session_timeline.csv"

# ★ [사용자 설정: 작업 대상 필터] ★
TARGET_SEMESTER = None
TARGET_GROUP    = None

# [변수 매핑]
VAR_MAPPING = {
    'face_pitch': 'pose_Rx',
    'face_yaw': 'pose_Ry',
    'face_roll': 'pose_Rz',
    'face_pitch_vel': 'pitch_vel',
    'face_yaw_vel': 'yaw_vel',
    'face_roll_vel': 'roll_vel',
    'motion_energy': 'ME',
    'motion_energy_log': 'ME_log',
    'bbox_area': 'bbox_area',
    'valence': 'valence_label'
}

# ==============================================================================
# [함수 정의]
# ==============================================================================

def time_str_to_seconds(t_str):
    if pd.isna(t_str) or str(t_str).strip() in ['없음', 'nan', '']:
        return None
    parts = str(t_str).strip().split(':')
    if len(parts) == 3:
        return int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
    elif len(parts) == 2:
        return int(parts[0])*60 + int(parts[1])
    return 0.0

def parse_version_str(ver_str):
    if pd.isna(ver_str): return 0
    s = str(ver_str).strip()
    if s in ['없음', '', 'nan']: return 0
    match = re.search(r'\((\d+)\)', s)
    return int(match.group(1)) if match else 0

def get_raw_file_path(base_dir, sem, grp, week, file_ver, t_idx, member_id):
    if pd.isna(file_ver) or str(file_ver).strip() in ['없음', 'nan', '']:
        week_folder = week
    else:
        week_folder = f"{week}{str(file_ver).strip()}"
        
    t_folder = os.path.join(base_dir, sem, grp, week_folder, f"T{t_idx}")
    if not os.path.exists(t_folder): return None
    
    pattern = os.path.join(t_folder, f"*P{member_id}_augmented.csv")
    files = glob.glob(pattern)
    return files[0] if files else None

def process_week_data_all_members(semester, group, week, df_timeline, split_time):
    # 23-2 예외 처리 (timeline_info.csv에 없으므로 가상 생성)
    if str(semester).startswith("23-2"):
        timelines = pd.DataFrame([{'타임라인인덱스': 1, '파일버전': '없음', '시작시간': '0:00:00', '종료시간': '24:00:00'}])
    else:
        # 1. 타임라인 정보 조회
        df_week_timelines = df_timeline[
            (df_timeline['학기'] == semester) & 
            (df_timeline['그룹명'] == group) & 
            (df_timeline['주차'] == week)
        ].copy()
        
        if df_week_timelines.empty: return None

        df_week_timelines['ver_sort'] = df_week_timelines['파일버전'].apply(parse_version_str)
        timelines = df_week_timelines.sort_values(by=['ver_sort', '타임라인인덱스'])

    # 2. Offset 계산 & 멤버 스캔
    timeline_configs = []
    current_offset = 0.0
    detected_members = set()

    for _, row in timelines.iterrows():
        t_idx = row['타임라인인덱스']
        f_ver = row['파일버전']
        s_sec = time_str_to_seconds(row['시작시간'])
        e_sec = time_str_to_seconds(row['종료시간'])
        duration = e_sec - s_sec
        
        timeline_configs.append({'t_idx': t_idx, 'file_ver': f_ver, 'offset': current_offset})
        
        chk_week = week if (pd.isna(f_ver) or str(f_ver) in ['없음','']) else f"{week}{str(f_ver)}"
        t_folder = os.path.join(BASE_DIR, semester, group, chk_week, f"T{t_idx}")
        if os.path.exists(t_folder):
            for f in os.listdir(t_folder):
                match = re.search(r'P(\d+)_augmented', f)
                if match: detected_members.add(int(match.group(1)))
        
        current_offset += duration

    all_members = sorted(list(detected_members))
    all_member_rows = []

    # 3. 멤버별 데이터 처리
    for mem_id in all_members:
        member_name = f"P{mem_id}"
        chunks = []
        
        for cfg in timeline_configs:
            raw_path = get_raw_file_path(BASE_DIR, semester, group, week, cfg['file_ver'], cfg['t_idx'], mem_id)
            if not raw_path: continue
            try:
                df_raw = pd.read_csv(raw_path)
                cols = {src: tgt for tgt, src in VAR_MAPPING.items() if src in df_raw.columns}
                if not cols: continue
                df_chunk = df_raw[list(cols.keys()) + ['timestamp']].copy()
                df_chunk = df_chunk.rename(columns=cols)
                df_chunk['timestamp'] = df_chunk['timestamp'] + cfg['offset']
                chunks.append(df_chunk)
            except: pass

        if not chunks: continue
        
        df_week = pd.concat(chunks, ignore_index=True).sort_values('timestamp')

        if 'valence' in df_week.columns:
            val_map = {'negative': -1, 'neutral': 0, 'positive': 1}
            df_week['valence'] = df_week['valence'].astype(str).str.lower().map(val_map).fillna(0)

        # Z-score 계산
        numeric_cols = df_week.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col == 'timestamp': continue
            mean_v, std_v = df_week[col].mean(), df_week[col].std()
            df_week[f"{col}_z"] = 0.0 if std_v == 0 else (df_week[col] - mean_v) / std_v

        # 세션 분리
        processed_dfs = []
        if split_time is None:
            df_week['phase'] = 1
            processed_dfs.append(df_week)
        else:
            df_p1 = df_week[df_week['timestamp'] < split_time].copy()
            if not df_p1.empty:
                df_p1['phase'] = 1
                processed_dfs.append(df_p1)
            
            df_p2 = df_week[df_week['timestamp'] >= split_time].copy()
            if not df_p2.empty:
                df_p2['phase'] = 2
                df_p2['timestamp'] = df_p2['timestamp'] - split_time
                processed_dfs.append(df_p2)

        # Long-format 변환
        for df_phase in processed_dfs:
            if df_phase.empty: continue
            measures = [c for c in VAR_MAPPING.keys() if c in df_phase.columns]
            
            df_long = df_phase.melt(
                id_vars=['phase', 'timestamp'], value_vars=measures,
                var_name='measurement', value_name='value_raw'
            )
            z_cols = [f"{c}_z" for c in measures]
            df_long_z = df_phase.melt(
                id_vars=['phase', 'timestamp'], value_vars=z_cols,
                var_name='measurement_z', value_name='value_z'
            )
            df_long_z['measurement'] = df_long_z['measurement_z'].str.replace(r'_z$', '', regex=True)
            
            df_final = pd.merge(df_long, df_long_z[['phase', 'timestamp', 'measurement', 'value_z']], 
                                on=['phase', 'timestamp', 'measurement'], how='left')
            
            df_final['semester_team_id'] = f"{semester}_{group}"
            df_final['week'] = int(re.search(r'\d+', week).group())
            df_final['member_id'] = member_name 
            
            all_member_rows.append(df_final)

    if not all_member_rows: return None
    return pd.concat(all_member_rows, ignore_index=True)


# ==============================================================================
# [실행부]
# ==============================================================================
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    df_info = pd.read_csv(TIMELINE_INFO_PATH)
    df_sess = pd.read_csv(SESSION_TIMELINE_PATH)
    
    # df_sess를 기준으로 전체 작업 테스크 추출 (23-2 포함)
    tasks = df_sess[['학기', '그룹명', '주차']].drop_duplicates()

    # 필터 적용
    if TARGET_SEMESTER: tasks = tasks[tasks['학기'] == TARGET_SEMESTER]
    if TARGET_GROUP: tasks = tasks[tasks['그룹명'] == TARGET_GROUP]

    if tasks.empty:
        print("조건에 맞는 데이터가 없습니다.")
        exit()

    print(f"총 {len(tasks)}개의 주차 작업 시작... (Target: {TARGET_SEMESTER}-{TARGET_GROUP})")
    print("구조: [Timestamp, Measurement] 유지 + [P1_raw...P4_raw] + [P1_z...P4_z] 정렬\n")

    for _, row in tasks.iterrows():
        sem, grp, week = row['학기'], row['그룹명'], row['주차']
        
        sess_row = df_sess[(df_sess['학기']==sem) & (df_sess['그룹명']==grp) & (df_sess['주차']==week)]
        split_t = time_str_to_seconds(sess_row.iloc[0]['분기시간']) if not sess_row.empty else None
        
        print(f"Processing: {sem}-{grp}-{week}...", end=" ")
        
        try:
            # 1. 데이터 수집
            df_long_all = process_week_data_all_members(sem, grp, week, df_info, split_t)
            
            if df_long_all is not None and not df_long_all.empty:
                
                # 2. Pivot 수행
                # Index: 고정될 컬럼들 (measurement 포함!)
                # Columns: 멤버 (P1, P2...)
                # Values: Raw 값과 Z 값
                df_pivot = df_long_all.pivot_table(
                    index=['semester_team_id', 'week', 'phase', 'timestamp', 'measurement'],
                    columns='member_id',
                    values=['value_raw', 'value_z']
                )
                
                # 3. 컬럼 재정의 및 정렬 (요청사항 반영)
                # 목표: P1_value_raw, P2_value_raw ... P1_value_z, P2_value_z ...
                
                raw_cols = []
                z_cols = []
                data_map = {}
                
                # 현재 Pivot 컬럼: (value_type, member_id)
                # 존재하는 멤버 목록 추출
                members = sorted(list(set([c[1] for c in df_pivot.columns])))
                
                for mem in members:
                    # Raw 값 먼저 수집
                    if ('value_raw', mem) in df_pivot.columns:
                        col_name = f"{mem}_value_raw"
                        data_map[col_name] = df_pivot[('value_raw', mem)]
                        raw_cols.append(col_name)
                    
                    # Z 값 수집
                    if ('value_z', mem) in df_pivot.columns:
                        col_name = f"{mem}_value_z"
                        data_map[col_name] = df_pivot[('value_z', mem)]
                        z_cols.append(col_name)
                
                # 순서 보장: Raw 그룹 -> Z 그룹
                new_columns = raw_cols + z_cols
                
                # DataFrame 재조립
                df_final = pd.DataFrame(data_map, index=df_pivot.index).reset_index()
                
                # 4. 최종 컬럼 순서 정리
                # semester_team_id, week, phase, timestamp, measurement, [P1_raw...], [P1_z...]
                meta_cols = ['semester_team_id', 'week', 'phase', 'timestamp', 'measurement']
                final_order = meta_cols + new_columns
                df_final = df_final[final_order]

                # 5. 정렬 (시간 -> measurement)
                df_final = df_final.sort_values(by=['phase', 'timestamp', 'measurement'])

                # 6. 엑셀 저장 (모두 하나의 'Data' 시트에 통합 + 폴더 구조화)
                save_dir = os.path.join(OUTPUT_DIR, sem, grp)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    
                file_name = f"Integrated_{sem}_{grp}_{week}.xlsx"
                file_path = os.path.join(save_dir, file_name)
                
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    df_final.to_excel(writer, sheet_name="Data", index=False)
                    
                    # 7. 열 너비 자동 조정
                    worksheet = writer.sheets["Data"]
                    for column in worksheet.columns:
                        col_letter = get_column_letter(column[0].column)
                        worksheet.column_dimensions[col_letter].width = 18 

                print(f"[OK] 저장됨 (Cols: {len(df_final.columns)})")
            else:
                print(f"[No Data]")
                
        except Exception as e:
            print(f"\n[Error] {e}")
            import traceback
            traceback.print_exc()

    print("\n모든 작업이 완료되었습니다.")