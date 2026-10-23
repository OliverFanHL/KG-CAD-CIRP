"""Portable data access and manifest verification for offline reproduction."""
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def lines(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    # Published JSON artifacts use CRLF on every platform.
    with Path(path).open('w', encoding='utf8', newline='\r\n') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_inputs(repo):
    """Check every final data file against the shipped content manifest."""
    repo = Path(repo).resolve()
    manifest = read(repo / 'data/manifest.json')
    for row in manifest['files']:
        path = (repo / row['path']).resolve()
        if not path.is_relative_to(repo):
            raise ValueError('Data path escapes repository: ' + row['path'])
        if not path.is_file():
            raise ValueError('Missing input; run python scripts/prepare_data.py: ' + row['path'])
        if ('bytes' in row and path.stat().st_size != row['bytes']) or sha(path) != row['sha256']:
            raise ValueError('Input differs from data manifest: ' + row['path'])
    return len(manifest['files'])
