"""Resolve project resources independently of the current working directory."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def configuration() -> dict:
    defaults = json.loads((ROOT / 'configs/paths.example.json').read_text(encoding='utf-8'))
    selected = os.environ.get('BS_CONFIG')
    local = Path(selected).expanduser() if selected else ROOT / 'configs/paths.local.json'
    if not local.is_absolute():
        local = ROOT / local
    if selected and not local.is_file():
        raise FileNotFoundError(f'Configuration not found: {local}')
    if local.is_file():
        override = json.loads(local.read_text(encoding='utf-8-sig'))
        unknown = set(override) - set(defaults)
        if unknown:
            raise ValueError(f'Unknown path settings: {sorted(unknown)}')
        defaults.update(override)
    return defaults


def path(key: str) -> Path:
    value = Path(os.path.expandvars(configuration()[key])).expanduser()
    return (ROOT / value).resolve() if not value.is_absolute() else value.resolve()


def tool(key: str) -> str:
    value = os.path.expandvars(configuration()[key])
    if '/' in value or '\\' in value:
        return str(path(key))
    return value  # A bare executable name is resolved through PATH.


def load_manifest() -> dict:
    location = path('wavelet_manifest')
    if not location.is_file():
        raise FileNotFoundError(f'Create a private session manifest at {location}; see docs/guide.md')
    manifest = json.loads(location.read_text(encoding='utf-8-sig'))
    for key in ('groups', 'sessions'):
        items = manifest.get(key, [])
        if not items or len(set(items)) != len(items):
            raise ValueError(f'{key} must be a nonempty list of unique identifiers')
        if any(not isinstance(x, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', x) for x in items):
            raise ValueError(f'Unsafe identifier in {key}')
    for group in manifest['groups']:
        for session in manifest['sessions']:
            count = manifest['sample_counts'][group][session]
            if type(count) is not int or count < 2:
                raise ValueError('Every group/session must have a positive integer sample count >= 2')
    for group, left, right in manifest.get('distinct_sessions', []):
        if group not in manifest['groups'] or left not in manifest['sessions'] or right not in manifest['sessions'] or left == right:
            raise ValueError('Invalid distinct_sessions entry')
    return manifest
