from typing import Dict, Any, Optional
from pydantic import BaseModel

class EmployeePayload(BaseModel):
    """Flexible payload: a dict of feature_name -> value.
    Replace with a typed schema when you freeze your feature list.
    """
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    probability: float
    risk: int
    top_features: Optional[Dict[str, float]] = None
