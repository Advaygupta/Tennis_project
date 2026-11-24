"""Fail closed on artifact bytes before deserializing the trusted DataFrame."""
import argparse
import json
from pathlib import Path
from shared import HERE, REPO, PAIRED, digest

FILENAMES = ('v1_1_results.joblib', 'v2_1_test_match_probabilities.parquet', 'matches_v1_1_metadata.csv')


def verify_hash(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError('SHA-256 mismatch: '+str(path))
    return actual


def verify_inputs(artifacts, sidecar=None):
    previous = json.loads((PAIRED/'results/manifest.json').read_text(encoding='utf8'))
    expected = {Path(k.replace('\\', '/')).name: v for k, v in previous['inputs'].items()}
    paths = [Path(artifacts)/name for name in FILENAMES]
    if sidecar is not None:
        paths.append(Path(sidecar))
    verified = {p.name: dict(sha256=verify_hash(p, expected[p.name]), bytes=p.stat().st_size) for p in paths}
    # Reused executable helpers must also match the recorded Point 3 code hashes.
    code = {name: verify_hash(PAIRED/name, previous['code'][name])
            for name in ['alignment.py', 'metrics.py', 'paired_bootstrap.py', 'inventory.py']}
    return verified, code


def check_output(output, artifacts, inputs=()):
    output, artifacts = Path(output).resolve(), Path(artifacts).resolve()
    if output.exists():
        raise ValueError('Output already exists; choose a NEW directory')
    protected = [artifacts, REPO/'notebooks', REPO/'models', PAIRED,
                 HERE.parent/'scoring', HERE.parent/'temporal', HERE/'results', HERE/'figures']
    for root in protected:
        root = root.resolve()
        if output == root or root in output.parents or output in root.parents:
            raise ValueError('Output overlaps preserved artifacts, audits, or published results')
    for item in inputs:
        item = Path(item).resolve()
        if output == item or output in item.parents or item in output.parents:
            raise ValueError('Output overlaps an input')
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts', type=Path, default=REPO/'reproduction/artifacts')
    p.add_argument('--scoring-sidecar', type=Path)
    args = p.parse_args()
    inputs, code = verify_inputs(args.artifacts, args.scoring_sidecar)
    print(json.dumps(dict(inputs=inputs, reused_code=code,
        note='Hashes identify bytes, not provenance authenticity, correctness, or redistribution rights.'), indent=2))


if __name__ == '__main__':
    main()
