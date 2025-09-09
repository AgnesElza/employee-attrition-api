# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and data needed for training + serving
COPY app/ app/
COPY src/ src/
COPY train.py .
COPY data/ data/

# === Train the model during the image build ===
# Adjust the data path/target if yours differ
RUN python train.py --data data/attrition.feather --target Attrition --outdir models/

# (Optional) list the model to verify in build logs
RUN ls -lh models/

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

