"""Build an explicit public allowlist, never zip the whole repository."""
import argparse
import json
from pathlib import Path
import zipfile
from shared import HERE, REPO, digest


def package_files():
    final = [p for p in HERE.iterdir() if p.is_file() and p.suffix in ['.py', '.md', '.txt', '.json']]
    final += [HERE/'.gitattributes', HERE/'.gitignore']
    final += [p for directory in ['results', 'figures'] for p in (HERE/directory).glob('*')
              if p.is_file() and p.suffix in ['.json', '.csv', '.svg', '.txt']]
    paired = REPO/'validation/paired_evaluation'
    reused = [paired/name for name in ['alignment.py', 'metrics.py', 'paired_bootstrap.py', 'inventory.py']]
    reused += [paired/'results'/name for name in ['manifest.json', 'summary.json']]
    reused += [REPO/'reproduction/expected_metrics.json']
    return sorted(set(final+reused))


def build(output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Refusing to overwrite existing package')
    files = package_files()
    if any('audit_private' in p.parts or p.suffix in ['.parquet', '.joblib', '.pkl', '.ipynb'] for p in files):
        raise ValueError('Restricted file in public package')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Member timestamps are fixed, so archive bytes depend only on member bytes.
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for path in files:
            name = path.relative_to(REPO).as_posix()
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 24, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, path.read_bytes())
    return dict(file=output.name, sha256=digest(output), bytes=output.stat().st_size,
                members=len(files), policy='Explicit code/documentation/aggregate allowlist; no row-level artifacts or model checkpoints')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    print(json.dumps(build(p.parse_args().output), indent=2))
