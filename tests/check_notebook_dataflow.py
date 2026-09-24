"""Check the notebook's mood-data handoff without downloading models."""

import ast
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


notebook = json.loads((Path(__file__).resolve().parents[1] / "src_code.ipynb").read_text())
cells = [ast.parse("".join(cell["source"])) for cell in notebook["cells"] if cell["cell_type"] == "code"]

producer = next(cell for cell in cells if "dataset_mood_adjusted.pkl" in ast.unparse(cell))
consumer = next(cell for cell in cells if "def parse_embedding" in ast.unparse(cell))
save_calls = [
    node for node in producer.body
    if isinstance(node, ast.Expr)
    and isinstance(node.value, ast.Call)
    and isinstance(node.value.func, ast.Attribute)
    and node.value.func.attr in {"to_pickle", "to_csv"}
]
saved_files = {node.value.args[0].value for node in save_calls}
assert {"dataset_mood_adjusted.pkl", "dataset_mood_adjusted.csv"} <= saved_files
assert "dataset_mood_adjusted.csv" in ast.unparse(consumer)

embedding = np.linspace(-1, 1, 384, dtype=np.float32)
namespace = {"df": pd.DataFrame({"sbert_embedding": [embedding], "mood_composite_str": ["Happy, Calm"]})}
parser = next(node for node in consumer.body if isinstance(node, ast.FunctionDef) and node.name == "parse_embedding")
exec(compile(ast.Module(body=[parser], type_ignores=[]), "src_code.ipynb", "exec"), {"np": np}, namespace)

with tempfile.TemporaryDirectory() as directory:
    old_cwd = Path.cwd()
    try:
        os.chdir(directory)
        exec(compile(ast.Module(body=save_calls, type_ignores=[]), "src_code.ipynb", "exec"), namespace)
        csv_row = pd.read_csv("dataset_mood_adjusted.csv").iloc[0]
        pickle_row = pd.read_pickle("dataset_mood_adjusted.pkl").iloc[0]
        parsed = namespace["parse_embedding"](csv_row["sbert_embedding"])
        assert parsed.size == embedding.size
        np.testing.assert_allclose(parsed, embedding, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(pickle_row["sbert_embedding"], embedding)
        assert csv_row["mood_composite_str"] == "Happy, Calm"
    finally:
        os.chdir(old_cwd)

print("Notebook mood-data handoff passed")
