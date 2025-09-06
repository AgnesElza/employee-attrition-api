import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric

os.makedirs("docs", exist_ok=True)

ref_path = "data/reference_sample.csv"
cur_path = "data/recent_requests.csv"  # optional; build this from logs/feedback

if not os.path.exists(ref_path):
    # Create a placeholder reference if missing
    pd.DataFrame({"placeholder":[0]}).to_csv(ref_path, index=False)

reference = pd.read_csv(ref_path)
current = pd.read_csv(cur_path) if os.path.exists(cur_path) else reference.copy()

report = Report(metrics=[DataDriftPreset(), DatasetDriftMetric()])
report.run(reference_data=reference, current_data=current)

html = report.as_html()
with open("docs/index.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Wrote Evidently drift report to docs/index.html")
