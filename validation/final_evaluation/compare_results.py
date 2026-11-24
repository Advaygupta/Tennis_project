"""Compare scientific numeric results, excluding runtime/environment/prose metadata."""
import argparse
import json
from pathlib import Path
import math


def compare(reference, actual, tolerance=1e-10):
    errors = []

    def walk(a, b, path):
        if isinstance(a, dict):
            if not isinstance(b, dict) or set(a) != set(b):
                errors.append(path+': keys differ')
                return
            for key in a:
                walk(a[key], b[key], path+'/'+key)
        elif isinstance(a, list):
            if not isinstance(b, list) or len(a) != len(b):
                errors.append(path+': list shape differs')
                return
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, path+'/'+str(i))
        elif isinstance(a, (int, float)) and not isinstance(a, bool):
            if not isinstance(b, (int, float)) or not math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance):
                errors.append(path+': numeric mismatch')
        elif a != b:
            errors.append(path+': value mismatch')

    for key in ['n', 'calibration', 'logloss', 'uncertainty', 'subsets', 'scoring_only_sensitivity']:
        walk(reference[key], actual[key], key)
    if errors:
        raise ValueError('\n'.join(errors[:30]))
    return dict(status='Scientific results agree', absolute_and_relative_tolerance=tolerance)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, default=Path(__file__).resolve().parent/'results/summary.json')
    p.add_argument('--actual', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(compare(json.loads(args.reference.read_text(encoding='utf8')),
                             json.loads(args.actual.read_text(encoding='utf8'))), indent=2))
