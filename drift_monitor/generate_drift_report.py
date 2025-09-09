import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently import ColumnMapping

# ---- Inputs ----
REF_PATH = "data/reference_sample.csv"          # produced by train.py earlier
CUR_PATH = "data/current_sample.csv"            # optional; will synthesize if missing
OUT_HTML = "docs/drift_report.html"
OUT_JSON = "docs/drift_report.json"

# ---- Load reference ----
ref = pd.read_csv(REF_PATH)

# Try to load "current"; if not present, synthesize a slightly shifted sample
if os.path.exists(CUR_PATH):
    cur = pd.read_csv(CUR_PATH)
else:
    cur = ref.sample(min(len(ref), 500), random_state=42).copy()
    # small synthetic shift to demonstrate drift
    for col in cur.select_dtypes(include="number"):
        cur[col] = cur[col] * 1.03  # +3% shift

# Infer simple column mapping
target_col = "Attrition" if "Attrition" in ref.columns else None
num_cols = ref.select_dtypes(include="number").columns.tolist()
cat_cols = ref.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
if target_col in num_cols: num_cols.remove(target_col)
if target_col in cat_cols: cat_cols.remove(target_col)

mapping = ColumnMapping(
    target=target_col,
    prediction=None,
    numerical_features=num_cols,
    categorical_features=cat_cols,
)

# Build report
report = Report(metrics=[
    DataDriftPreset(),
    TargetDriftPreset() if target_col else DataDriftPreset()
])
report.run(reference_data=ref, current_data=cur, column_mapping=mapping)

# Save
os.makedirs("docs", exist_ok=True)
report.save_html(OUT_HTML)
with open(OUT_JSON, "w") as f:
    f.write(report.json())

print(f"Saved: {OUT_HTML}\nSaved: {OUT_JSON}")