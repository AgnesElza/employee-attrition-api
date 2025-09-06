# Attrition Risk API · FastAPI + Docker + CI

End‑to‑end, zero‑cost demo of deploying a scikit‑learn model with **FastAPI**, **Docker**, and **GitHub Actions**.
Includes basic monitoring (Prometheus metrics) and weekly **Evidently** drift report published to GitHub Pages.

## Quickstart

```bash
# 1) (optional) create a venv
python -m venv .venv && source .venv/bin/activate

# 2) install deps
pip install -r requirements.txt

# 3) train a quick model (uses a CSV you provide; defaults assume IBM HR attrition format)
python -m src.train --data data/raw.csv --target Attrition --outdir models/

# 4) run the API locally
uvicorn app.main:app --reload --port 8080

# 5) docker build & run
docker build -t attrition-api:latest .
docker run -p 8080:8080 attrition-api:latest
```

**Swagger docs:** http://localhost:8080/docs  
**Healthcheck:** http://localhost:8080/health  
**Metrics (Prometheus):** http://localhost:8080/metrics

## Repo layout
```
src/               # training + utilities
app/               # FastAPI service
drift_monitor/     # weekly Evidently drift report → docs/index.html
.github/workflows/ # CI (tests + build + push image) and drift job
models/            # saved model + metadata.json
data/              # your data + reference_sample.csv (for drift)
docs/              # (auto-generated) GH Pages with latest drift report
```

## CI/CD
- **CI:** On every push/PR → run tests → build Docker → push to GitHub Container Registry (GHCR).
- **CD (simple):** Connect repo to **Render** or **Fly.io**; configure to deploy on push to `main`.
- **Drift report:** Weekly GitHub Action writes an **Evidently** report into `docs/` (enable GH Pages on the repo).

## Deploy tips
- Render: New Web Service → Docker → set Port = `8080`. Auto deploy on push.
- Fly.io: `fly launch` → `flyctl deploy` (it will use the Dockerfile).

---

_Generated: 2025-09-06T14:06:02.452966Z_
