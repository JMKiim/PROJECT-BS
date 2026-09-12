"""
mea_roi.py
──────────
MEA 계산용 ROI(관심영역) 설정 로더/유틸.

groups 1~4처럼 Zoom UI(이름표, 팝업, 캡션 박스)가 프레임 안에 들어오는 경우,
해당 픽셀 사각형을 MEA 합산에서 제외하기 위한 "exclude 사각형" 목록을 관리한다.

설정 파일(JSON, 기본 scripts/mea_roi.json) 스키마:

    {
      "group_02": {
        "resolution": "600x400",           # (선택) 검증용. 실제와 다르면 경고
        "exclude": [[x1, y1, x2, y2], ...]  # 그룹 전 세션 공통 제외 사각형
      },
      "group_01": {
        "exclude": [[x1, y1, x2, y2]],      # 공통(예: 상단 팝업)
        "by_session": {
          "1": { "exclude": [[x1, y1, x2, y2]] },   # 세션별로 "추가"됨(합쳐짐)
          "3": { "resolution": "704x394", "exclude": [[...]] }
        }
      }
    }

좌표는 픽셀 정수, [x1, y1, x2, y2] 반열림(오른쪽/아래 경계 미포함).
그룹 exclude 와 세션 exclude 는 "concatenate"(둘 다 적용)된다.
"""
import os
import json

import numpy as np


def load_roi_config(path):
    if not path or not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _norm_rects(rects):
    out = []
    for r in rects or []:
        if len(r) != 4:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in r)
        out.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
    return out


def get_exclude_rects(cfg, group, session=None, participant=None):
    """(group, session, participant)에 적용할 exclude 사각형 목록을 반환.

    entry.exclude(공통) + by_session.<s>.exclude + by_participant.<p>.exclude 를 모두 합침.
    by_session.<s>.by_participant.<p>.exclude 도 지원.
    """
    if not cfg:
        return []
    entry = cfg.get(group)
    if entry is None:
        return _norm_rects(cfg.get("_default", {}).get("exclude", []))
    rects = list(entry.get("exclude", []))

    sess_entry = None
    if session is not None:
        sess_entry = entry.get("by_session", {}).get(str(session))
        if sess_entry:
            rects += list(sess_entry.get("exclude", []))

    if participant is not None:
        pid = str(participant)
        bp = entry.get("by_participant", {})
        if pid in bp:
            rects += list(bp[pid].get("exclude", []))
        if sess_entry:
            sbp = sess_entry.get("by_participant", {})
            if pid in sbp:
                rects += list(sbp[pid].get("exclude", []))

    return _norm_rects(rects)


def expected_resolution(cfg, group, session=None):
    """설정에 적힌 기대 해상도 문자열('WxH') 또는 None."""
    entry = (cfg or {}).get(group)
    if not entry:
        return None
    if session is not None:
        bs = entry.get("by_session", {})
        s = bs.get(str(session))
        if s and s.get("resolution"):
            return s["resolution"]
    return entry.get("resolution")


def build_keep_mask(height, width, exclude_rects):
    """제외 사각형을 뺀 boolean keep 마스크 (True=합산 대상)."""
    mask = np.ones((height, width), dtype=bool)
    for x1, y1, x2, y2 in _norm_rects(exclude_rects):
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = False
    return mask


def rects_signature(exclude_rects):
    """ROI 변경 감지용 안정적 문자열."""
    return json.dumps(_norm_rects(exclude_rects), sort_keys=True)
