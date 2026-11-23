"""Read notebook JSON and execute only explicitly allowlisted function definitions.

No notebook cells, model loading, fitting, prediction, or artifact writes execute.
The match polynomial is transcribed from cell 217 and tested against the independent
server-state match recursion. Source and extraction hashes are recorded by audit.py.
"""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

FUNCTIONS=("game_win_probab","set_win_probab","probability_reach_6_6",
           "tiebreak_win_probab","deciding_set_win_probab","deciding_set_rule")


def extract(repo):
    path=Path(repo)/"notebooks/03_v1_1_model_training.ipynb"
    raw=path.read_bytes()
    notebook=json.loads(raw)
    found={}
    cell_indices={}
    for index,cell in enumerate(notebook["cells"]):
        if cell["cell_type"]!="code":continue
        source="".join(cell.get("source",[]))
        if "def deciding_set_win_probab" not in source:continue
        tree=ast.parse(source)
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name in FUNCTIONS:
                if node.decorator_list:raise ValueError("Unexpected function decorator")
                found[node.name]=node
                cell_indices[node.name]=index
    if set(found)!=set(FUNCTIONS):raise ValueError("Scoring definitions missing or changed")
    module=ast.Module(body=[found[n] for n in FUNCTIONS],type_ignores=[])
    namespace={"np":np,"pd":pd}
    exec(compile(module,str(path)+":allowlisted_functions","exec"),namespace)
    return namespace,{
        "notebook_sha256":hashlib.sha256(raw).hexdigest(),
        "function_ast_sha256":hashlib.sha256(ast.dump(module).encode()).hexdigest(),
        "cells_zero_based":cell_indices,
        "notebook_executed":False,
        "executed":list(FUNCTIONS),
    }


def match_probability(functions,p,q,best_of,rule):
    s=float(np.asarray(functions["set_win_probab"](p,q)).item())
    d=float(np.asarray(functions["deciding_set_win_probab"](p,q,rule,standard_set=s)).item())
    if best_of==3:return s*s+2*s*(1-s)*d
    if best_of==5:return s**3+3*s**3*(1-s)+6*s*s*(1-s)**2*d
    raise ValueError("Unsupported best_of")
