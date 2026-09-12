"""One entry point for feature extraction, analysis, and reporting."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

from bs.settings import ROOT, configuration, path, tool, load_manifest

STAGES = {
    'crop': ('bs.extraction.crop', 'Cut recordings and crop participant views'),
    'openface': ('bs.extraction.openface_batch', 'Extract facial features with OpenFace'),
    'motion': ('bs.extraction.motion', 'Compute mean pixel motion energy'),
    'mea': ('bs.extraction.mea', 'Extract and standardize thresholded motion energy'),
    'augment': ('bs.preprocessing.augment', 'Combine features and derive analysis signals'),
    'statistics': ('bs.preprocessing.statistics', 'Calculate participant reference statistics'),
    'align': ('bs.preprocessing.align', 'Validate and align paired Head/MEA signals'),
    'events': ('bs.synchrony.events', 'Calculate threshold and event synchrony'),
    'weeks': ('bs.reporting.weeks', 'Merge timeline masks into weekly workbooks'),
    'sessions': ('bs.reporting.sessions', 'Split weekly workbooks into sessions'),
    'summary': ('bs.reporting.summary', 'Summarize synchrony levels per session'),
    'enrich': ('bs.reporting.enrich', 'Join assessment scores and weighted summaries'),
    'normalize': ('bs.reporting.normalize', 'Combine sessions and normalize to 50 minutes'),
    'signals': ('bs.reporting.signals', 'Export participant signals; see documented limitations'),
}
PARSERS = {'openface', 'mea', 'align', 'events'}


def doctor() -> None:
    print('PROJECT-BS | environment')
    print(f'Project: {ROOT}')
    for name in ('numpy', 'pandas', 'cv2', 'matplotlib', 'openpyxl', 'tqdm'):
        print(f'  {name:16} {"available" if importlib.util.find_spec(name) else "missing"}')
    for name in ('ffmpeg', 'openface_executable', 'matlab'):
        print(f'  {name:20} {"available" if shutil.which(tool(name)) else "not found"}')
    print('Local inputs are optional until the corresponding stage is run:')
    for key in ('raw_video_dir', 'features_dir', 'head_dir', 'mea_dir', 'wavelet_manifest', 'wavelet_toolbox_dir'):
        print(f'  {key:24} {"present" if path(key).exists() else "not configured / absent"}')


def wavelet(arguments: list[str]) -> None:
    parser = argparse.ArgumentParser(prog='bs wavelet', description='Run paired behavioral wavelet analysis in MATLAB.')
    parser.add_argument('--check', action='store_true', help='Check configuration and external dependencies without executing MATLAB')
    args = parser.parse_args(arguments)
    load_manifest()
    required = [path('prepared_dir'), path('wavelet_toolbox_dir') / 'WaveletTransforms/AWCOG.m', path('wavelet_toolbox_dir') / 'WaveletTransforms/MeanGAIN.m']
    for item in required:
        if not item.exists():
            raise FileNotFoundError(item)
    executable = shutil.which(tool('matlab'))
    if not executable:
        raise FileNotFoundError('MATLAB executable not found; configure matlab in paths.local.json')
    if args.check:
        print('Wavelet configuration and dependencies are available.')
        return
    env = os.environ.copy()
    env['BS_RESOLVED_CONFIG'] = json.dumps({key: str(path(key)) for key in ('prepared_dir', 'wavelet_results_dir', 'wavelet_manifest', 'wavelet_toolbox_dir')})
    script = ROOT / 'src/bs/synchrony/wavelet.m'
    expression = "run('" + script.as_posix().replace("'", "''") + "')"
    subprocess.run([executable, '-batch', expression], env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog='bs', description='Behavioral synchrony: signals, coordination, and analysis.')
    parser.add_argument('--config', help='Private path configuration; relative paths use the project root')
    parser.add_argument('stage', choices=['doctor', *STAGES, 'wavelet'], nargs='?')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    parser.epilog = '\n'.join(f'{name:12} {desc}' for name, (_, desc) in STAGES.items())
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    args = parser.parse_args()
    if args.config:
        os.environ['BS_CONFIG'] = args.config
    try:
        if not args.stage:
            parser.print_help()
        elif args.stage == 'doctor':
            doctor()
        elif args.stage == 'wavelet':
            wavelet(args.arguments)
        else:
            module, description = STAGES[args.stage]
            if args.stage not in PARSERS and args.arguments:
                if args.arguments == ['--help']:
                    print(f'bs {args.stage}: {description}\nUses configs/paths.local.json. See docs/guide.md.')
                    return
                parser.error(f'{args.stage} accepts no stage arguments')
            sys.argv = ['bs ' + args.stage, *args.arguments]
            runpy.run_module(module, run_name='__main__')
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
