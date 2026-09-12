"""Threshold and pulse synchrony, independent of plotting and file I/O."""
import time
import numpy as np
FPS = 15

def detect_nod_events(arr, fall_thresh, rise_thresh, max_dur, min_cyc, fps):
    # stop_thresh 파라미터는 더 이상 사용하지 않음
    mask = np.zeros_like(arr, dtype=int)
    state = 'idle'
    cycle_count = 0
    event_start = None
    max_frames = int(max_dur * fps)
    for i, v in enumerate(arr):
        if state == 'idle':
            if v > fall_thresh:
                state = 'down'
                event_start = i
        elif state == 'down':
            if v < rise_thresh:
                cycle_count += 1
                if cycle_count >= min_cyc and event_start is not None and (i - event_start) <= max_frames:
                    mask[i] = 1  # 주기 완료 시점에 펄스 마킹
                    state = 'idle'
                    cycle_count = 0
                    event_start = None
        # 최대 지속시간 초과 시 초기화
        if event_start is not None and (i - event_start) > max_frames:
            state = 'idle'
            cycle_count = 0
            event_start = None
    return mask
# 동시성 마스크 계산
def calculate_synchrony_mask(data_dict, config, global_stats):
    start = time.time()
    masks = {}
    for name, cfg in config.items():
        ctype = cfg.get('type')
        win = int(cfg.get('sync_window', 1.0) * FPS)

        # Numeric & Categorical
        if ctype in ('numeric', 'categorical'):
            thr = cfg.get('threshold_std', 2.0)
            mode = cfg.get('sync_direction', 'same')
            mats_above, mats_below = [], []
            for pid, df in data_dict.items():
                if ctype == 'numeric':
                    raw = df[cfg['column']].astype(float).values
                else:
                    raw = df[cfg['column']].fillna('neutral') \
                          .map(cfg['mapping']).fillna(0).astype(float).values
                succ = (df['success'].astype(int).values == 1) if 'success' in df.columns else np.ones_like(raw, bool)
                if cfg.get('zscore', False):
                    m, s = global_stats[pid][name]['mean'], global_stats[pid][name]['std']
                    raw = (raw - m) / s
                mats_above.append((raw > thr) & succ)
                mats_below.append((raw < -thr) & succ)
            mats_above = np.vstack(mats_above)
            mats_below = np.vstack(mats_below)
            ext_above = np.zeros_like(mats_above, dtype=bool)
            ext_below = np.zeros_like(mats_below, dtype=bool)
            for idx in range(mats_above.shape[0]):
                r_ab, r_bl = mats_above[idx], mats_below[idx]
                ext_r_ab = np.zeros_like(r_ab)
                ext_r_bl = np.zeros_like(r_bl)
                for t in range(len(r_ab)):
                    start_t = max(0, t - win)
                    if r_ab[start_t:t+1].any(): ext_r_ab[t] = True
                    if r_bl[start_t:t+1].any(): ext_r_bl[t] = True
                ext_above[idx] = ext_r_ab
                ext_below[idx] = ext_r_bl
            if mode == 'any':
                # sync = ext_above.sum(axis=0) + ext_below.sum(axis=0)
                sync = np.logical_or(ext_above, ext_below).sum(axis=0)
            elif mode == 'same':
                sync = np.maximum(ext_above.sum(axis=0), ext_below.sum(axis=0))
            elif mode == 'positive':
                sync = ext_above.sum(axis=0)
            elif mode == 'negative':
                sync = ext_below.sum(axis=0)
            else:
                sync = np.zeros_like(ext_above.sum(axis=0), dtype=int)
            masks[name] = sync

        # Event
        elif ctype == 'event':
            params = cfg['event_params']
            fall = params.get('fall_z_thresh', 1.0)
            rise = params.get('rise_z_thresh', -1.0)
            md   = params.get('max_duration', 0.6)
            mc   = params.get('min_cycles', 1)
            mats = []
            for pid, df in data_dict.items():
                raw = df[cfg['column']].astype(float).values
                succ = (df['success'].astype(int).values == 1) if 'success' in df.columns else np.ones_like(raw, bool)
                if cfg.get('zscore', False):
                    m, s = global_stats[pid][name]['mean'], global_stats[pid][name]['std']
                    raw = (raw - m) / s
                ev = detect_nod_events(raw, fall, rise, md, mc, FPS).astype(bool)
                # success 적용
                ev = ev & succ
                mats.append(ev)
            mats = np.vstack(mats)
            ext = np.zeros_like(mats, dtype=bool)
            for idx in range(mats.shape[0]):
                r = mats[idx]
                ext_r = np.zeros_like(r)
                for t in range(len(r)):
                    start_t = max(0, t - win)
                    if r[start_t:t+1].any(): ext_r[t] = True
                ext[idx] = ext_r
            masks[name] = ext.sum(axis=0)
        else:
            continue

    print(f"[TIME] Synchrony mask calc: {time.time()-start:.2f}s")
    return masks
# 메인 시각화
