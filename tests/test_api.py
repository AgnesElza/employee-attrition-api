from fastapi.testclient import TestClient
from app.main import app
import os

client = TestClient(app)

def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert 'status' in r.json()

def test_predict_when_no_model():
    # If model not trained yet, API should still respond
    if not os.path.exists('models/model.joblib'):
        r = client.post('/predict', json={'features': {'Age': 29, 'MonthlyIncome': 5000, 'OverTime': 'No'}})
        assert r.status_code == 200
        body = r.json()
        assert 'probability' in body
