"""Synthetic checks: no participant recordings or study outputs are used."""
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from bs.settings import ROOT, path, load_manifest
from bs.synchrony.core import calculate_synchrony_mask, detect_nod_events


class SynchronyTests(unittest.TestCase):
    def calculate(self, values, mode='any', window=0):
        config = {'x': {'type': 'numeric', 'column': 'x', 'threshold_std': 1,
                        'sync_direction': mode, 'sync_window': window}}
        frames = {f'P{i}': pd.DataFrame({'x': row, 'success': np.ones(len(row))})
                  for i, row in enumerate(values)}
        with contextlib.redirect_stdout(io.StringIO()):
            return calculate_synchrony_mask(frames, config, {})['x']

    def test_direction_modes(self):
        values = [[2, -2, 0], [-2, -2, 2]]
        for mode, expected in [('any', [2, 2, 1]), ('same', [1, 2, 1]),
                               ('positive', [1, 0, 1]), ('negative', [1, 2, 0])]:
            np.testing.assert_array_equal(self.calculate(values, mode), expected)

    def test_union_does_not_count_one_person_twice(self):
        np.testing.assert_array_equal(self.calculate([[2, -2, 0]], window=1), [1, 1, 1])

    def test_lookback_includes_both_endpoints(self):
        signal = np.zeros(18)
        signal[0] = 2
        actual = self.calculate([signal], window=1)
        np.testing.assert_array_equal(actual, [1] * 16 + [0, 0])

    def test_failed_tracking_cannot_start_activation(self):
        config = {'x': {'type': 'numeric', 'column': 'x', 'sync_window': 0, 'threshold_std': 1}}
        with contextlib.redirect_stdout(io.StringIO()):
            actual = calculate_synchrony_mask({'P1': pd.DataFrame({'x': [3, 3], 'success': [0, 1]})}, config, {})
        np.testing.assert_array_equal(actual['x'], [0, 1])

    def test_pulse_requires_reversal_in_time(self):
        np.testing.assert_array_equal(detect_nod_events(np.array([2, 0, -2]), 1, -1, .6, 1, 15), [0, 0, 1])
        long = np.array([2] + [0] * 10 + [-2])
        self.assertFalse(detect_nod_events(long, 1, -1, .6, 1, 15).any())


class ConfigurationTests(unittest.TestCase):
    def test_paths_ignore_working_directory(self):
        old = os.getcwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                self.assertEqual(path('indicators'), ROOT / 'configs/indicators.json')
            finally:
                os.chdir(old)

    def test_invalid_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            manifest = p / 'sessions.json'
            manifest.write_text(json.dumps({'groups': ['../private'], 'sessions': ['S1'], 'sample_counts': {}}))
            config = p / 'config.json'
            config.write_text(json.dumps({'wavelet_manifest': str(manifest)}))
            with patch.dict(os.environ, {'BS_CONFIG': str(config)}), self.assertRaises(ValueError):
                load_manifest()


class ExportTests(unittest.TestCase):
    def test_masks_weeks_sessions_summary_and_assessments(self):
        from bs.visualization import timeline
        from bs.reporting import weeks, sessions, summary, enrich, normalize
        config_path = ROOT / 'configs/indicators.json'
        config = json.loads(config_path.read_text())
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            features = root / 'features'
            results = root / 'results'
            tl = features / '24-1/Demo/W1/T1'
            tl.mkdir(parents=True)
            stats = {}
            for person in range(1, 5):
                t = np.arange(150) / 15
                frame = pd.DataFrame({cfg['column']: 2 * np.sin(t * 3 + person / 5)
                                      for cfg in config.values() if cfg['type'] != 'categorical'})
                frame['valence_label'] = np.where(np.arange(150) % 3 == 0, 'positive', 'neutral')
                frame['timestamp'] = t
                frame['success'] = 1
                frame.to_csv(tl / f'Demo_W1_T1_P{person}_augmented.csv', index=False)
                stats[f'P{person}'] = {key: {'mean': 0, 'std': 1} for key in config}
            (tl / 'global_stats.json').write_text(json.dumps(stats))
            timeline.visualize_timeline_optimized(str(tl), str(config_path))
            levels = pd.read_excel(tl / 'sync_mask.xlsx', sheet_name='LevelCounts', index_col=0)
            self.assertTrue((levels.sum(axis=1) == 150).all())
            from bs.synchrony.events import already_done
            self.assertTrue(already_done(str(tl)))
            with patch.multiple(weeks, ROOT_IN=str(features), ROOT_OUT=str(results), CONFIG_PATH=str(config_path)):
                weeks.build_week_file('24-1', 'Demo', 'W1')
            info = pd.DataFrame([{'학기': '24-1', '그룹명': 'Demo', '주차': 'W1', '파일버전': '없음',
                                  '타임라인인덱스': 1, '시작시간': '0:00:00', '종료시간': '0:00:10', '인원수': 4}])
            sess = pd.DataFrame([{'학기': '24-1', '그룹명': 'Demo', '주차': 'W1', '파일버전': '없음', '분기시간': '0:00:05', '세션': 2}])
            week_file = results / '24-1/Demo/sync_24-1_Demo_W1.xlsx'
            sessions.split_week_file(str(week_file), info, sess, list(config))
            master = results / 'master_sync_summary.xlsx'
            with patch.multiple(summary, ROOT_OUT=str(results), CONFIG_PATH=str(config_path), MASTER_OUT=str(master)):
                summary.build_master()
            table = pd.read_excel(master)
            self.assertEqual(len(table), 24)
            self.assertTrue((table.filter(regex=r'^frames_k').sum(axis=1) == 75).all())
            cps = root / 'assessment.xlsx'
            with pd.ExcelWriter(cps) as writer:
                for title in ('2023 2학기', '2024 1학기', '2024 2학기'):
                    pd.DataFrame({'TEAM': ['Demo', 'Demo'], 'WEEK': [1, 1], 'STEP': ['1.1', '1.2'],
                                  'STEP_TOTAL': [10, 20], 'STEP_critical thinking': [4, 8], 'STEP_creative thinking': [6, 12]}).to_excel(writer, sheet_name=title + '_평가_STEP', index=False)
                    pd.DataFrame({'TEAM': ['Demo'], 'WEEK': [1], 'WEEK TOTAL': [30],
                                  'WEEK_critical thinking': [12], 'WEEK_creative thinking': [18]}).to_excel(writer, sheet_name=title + '_평가_WEEK', index=False)
            enriched = root / 'enriched.xlsx'
            with patch.multiple(enrich, MASTER_IN=str(master), MASTER_OUT=str(enriched), CPS_PATH=str(cps)):
                enrich.main()
            self.assertTrue(pd.read_excel(enriched)['TOTAL'].notna().all())
            normalized = root / 'normalized.xlsx'
            with patch.multiple(normalize, MASTER_IN=str(master), MASTER_OUT=str(normalized), CPS_PATH=str(cps)):
                normalize.main()
            table = pd.read_excel(normalized)
            np.testing.assert_allclose(table.filter(regex=r'^frames_k').sum(axis=1), 45000)

    def test_week_one_does_not_match_week_ten(self):
        from bs.reporting.sessions import compute_split_frames
        info = pd.DataFrame([{'학기': '24-1', '그룹명': 'Demo', '주차': 'W10', '파일버전': '없음',
                              '타임라인인덱스': 1, '시작시간': '0:00:00', '종료시간': '0:00:10'}])
        sess = pd.DataFrame([{'학기': '24-1', '그룹명': 'Demo', '주차': 'W10', '파일버전': '없음', '분기시간': '0:00:05', '세션': 2}])
        self.assertEqual(compute_split_frames('24-1', 'Demo', 'W1', info, sess), [])

    def test_unequal_participant_lengths_rejected(self):
        from bs.visualization.timeline import open_timeline_data
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            (p / 'global_stats.json').write_text('{}')
            pd.DataFrame({'x': [1, 2]}).to_csv(p / 'P1_augmented.csv', index=False)
            pd.DataFrame({'x': [1]}).to_csv(p / 'P2_augmented.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'lengths differ'):
                open_timeline_data(str(p), str(ROOT / 'configs/indicators.json'), 0)


class AlignmentTests(unittest.TestCase):
    def test_paired_alignment_interpolates_head_and_keeps_mea(self):
        from bs.preprocessing import align
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            for kind in ('head', 'mea'):
                (root / kind / 'demo').mkdir(parents=True)
            t = np.arange(250) / 25
            wave = np.sin(t)
            for pid in ('P1', 'P2'):
                head = pd.DataFrame({col: wave.copy() for col in align.HEAD_SIGNAL_COLUMNS})
                head['frame'] = np.arange(250) + 100
                head['timestamp'] = t + 4
                head['success'] = 1
                head['mp_confidence'] = 1.0
                head.loc[10, align.HEAD_SIGNAL_COLUMNS] = np.nan
                head.to_csv(root / 'head/demo' / f'demo_{pid}_S1_slim.csv', index=False)
                pd.DataFrame({'frame': np.arange(250), 'timestamp': t, 'mea_z': wave}).to_csv(root / 'mea/demo' / f'demo_{pid}_S1_mea.csv', index=False)
            manifest = {'groups': ['demo'], 'sessions': ['S1'], 'sample_counts': {'demo': {'S1': 250}}}
            with patch.multiple(align, HEAD_DIR=root / 'head', MEA_DIR=root / 'mea', OUTPUT_DIR=root / 'prepared', BASE_DIR=root,
                                GROUPS=['demo'], SESSIONS=['S1'], SAMPLE_COUNTS=manifest['sample_counts']), patch.object(align, 'load_manifest', return_value=manifest):
                records = align.preflight()
                self.assertEqual(len(records), 2)
                outputs = [align.process_one('demo', pid, 'S1') for pid in ('P1', 'P2')]
                align.validate_outputs()
                align.write_report(records, outputs)
            actual = pd.read_csv(root / 'prepared/demo/demo_P1_S1_slim.csv')
            self.assertFalse(actual.isna().any().any())
            self.assertEqual(actual['frame'].iloc[0], 100)
            np.testing.assert_allclose(actual['mea_z'], wave, atol=1e-15)

    def test_mea_processing_has_expected_window_and_scale(self):
        from bs.extraction.mea import mea_smooth, mea_scale
        values, window = mea_smooth(np.arange(100, dtype=float), .5, 25)
        self.assertEqual(window, 13)
        z, _, _ = mea_scale(values)
        self.assertAlmostEqual(float(z.mean()), 0)
        self.assertAlmostEqual(float(z.std()), 1)


if __name__ == '__main__':
    unittest.main()
