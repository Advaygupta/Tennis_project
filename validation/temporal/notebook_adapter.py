"""Load only feature definitions from committed notebooks; never execute workflow cells.

This adapter is the implementation UNDER TEST, not the independent reference.
"""
import ast
import contextlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FEATURE_NOTEBOOK = ROOT / 'notebooks/01_deduplication_feature_engineering.ipynb'
CELLS = (65, 72, 73, 74, 85, 94, 96, 98, 101, 105, 127)
FAMILIES = {
    'matchup': ('add_matchup_features', 'MATCHUP_FEATURE_COLUMNS'),
    'elo': ('add_batch_elo_features', 'ELO_FEATURE_COLUMNS'),
    'recent': ('add_recent_form_features', 'RECENT_FEATURE_COLUMNS'),
    'career_serve': ('add_career_serve_features', 'CAREER_SERVE_COLUMNS'),
    'matchup_serve': ('add_matchup_serve_features', 'MATCHUP_SERVE_COLUMNS'),
    'surface_serve': ('add_surface_serve_features', 'SURFACE_SERVE_COLUMNS'),
    'return': ('add_return_features', 'RETURN_FEATURE_COLUMNS'),
    'fatigue': ('add_fatigue_features', 'FATIGUE_FEATURE_COLUMNS'),
    'experience': ('add_experience_features', 'EXPERIENCE_FEATURE_COLUMNS'),
}


def source_cells(path):
    return [''.join(c['source']) for c in json.loads(path.read_text(encoding='utf-8'))['cells']]


def definitions():
    namespace = {}
    sources = source_cells(FEATURE_NOTEBOOK)
    for index in CELLS:
        nodes = []
        for node in ast.parse(sources[index]).body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = ([x.name.split('.')[0] for x in node.names]
                           if isinstance(node, ast.Import) else [node.module])
                if set(modules) <= {'collections', 'math', 'numpy', 'pandas'}:
                    nodes.append(node)
            elif isinstance(node, ast.FunctionDef):
                nodes.append(node)
            elif isinstance(node, ast.Assign) and all(
                isinstance(t, ast.Name) and t.id.isupper() for t in node.targets
            ):
                nodes.append(node)
        exec(compile(ast.Module(body=nodes, type_ignores=[]),
                     f'{FEATURE_NOTEBOOK.name}:cell{index}', 'exec'), namespace)
        if index == 85:
            # This list is built by a top-level loop in the original cell.
            namespace['RECENT_FEATURE_COLUMNS'] = [
                f'player{side}_{group}_{suffix}' for side in ('A', 'B')
                for group in ('career', 'surface_career', 'last10', 'last25',
                              'surface10', 'surface25', '52w')
                for suffix in ('wins', 'losses', 'matches', 'pct')]
    for function, columns in FAMILIES.values():
        if function not in namespace or not namespace[columns]:
            raise AssertionError(f'Notebook definitions changed: {function}/{columns}')
    return namespace


def feature_outputs(frame, namespace=None):
    ns = definitions() if namespace is None else namespace
    pieces = []
    with contextlib.redirect_stdout(io.StringIO()):
        for function, columns in FAMILIES.values():
            result = ns[function](frame.copy()).set_index('match_id')
            pieces.append(result[ns[columns]])
    import pandas as pd
    return pd.concat(pieces, axis=1).sort_index()


def cv_definition(notebook, symbol, train_dates):
    """Extract a CV function/class alone, without model imports or .fit calls."""
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import TimeSeriesSplit
    ns = dict(np=np, pd=pd, TimeSeriesSplit=TimeSeriesSplit, train_dates=train_dates)
    matches = []
    for cell in source_cells(ROOT / 'notebooks' / notebook):
        try:
            tree = ast.parse(cell)
        except SyntaxError:
            continue
        matches += [n for n in tree.body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == symbol]
    if len(matches) != 1:
        raise AssertionError(f'Expected one {symbol}, found {len(matches)}')
    exec(compile(ast.Module(body=matches, type_ignores=[]), notebook, 'exec'), ns)
    return ns[symbol]
