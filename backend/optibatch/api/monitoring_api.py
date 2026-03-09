"""
This file is responsible for monitoring API endpoints.
It is part of the api module and handles real-time monitoring functionality.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

from optibatch.monitoring.realtime_monitor import monitor_batch

router = APIRouter()

class MonitoringRequest(BaseModel):
    # Supports both:
    # - { "batch_parameters": { ... } } (canonical)
    # - { "temperature": 180, "pressure": 2.2, ... } (legacy/compat)
    batch_parameters: Optional[Dict[str, float]] = None

    class Config:
        extra = "allow"


def _as_batch_parameters(request: MonitoringRequest) -> Dict[str, float]:
    raw = request.dict(exclude_none=True)
    if "batch_parameters" in raw and isinstance(raw["batch_parameters"], dict):
        return raw["batch_parameters"]
    return {k: float(v) for k, v in raw.items() if k != "batch_parameters"}


def _normalize_predicted_metrics(predicted: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize backend prediction keys to the dashboard-friendly schema.
    - yield_percentage: 0-100
    - energy_consumption: kWh-ish
    - production_cost: USD-ish (if available)
    """
    predicted = predicted or {}

    # Prefer explicit keys if present, otherwise fall back to common variants.
    raw_yield = predicted.get("predicted_yield", predicted.get("yield", predicted.get("predicted_yield_percentage")))
    raw_energy = predicted.get("predicted_energy_consumption", predicted.get("predicted_energy", predicted.get("energy_consumption", predicted.get("energy"))))
    raw_cost = predicted.get("predicted_production_cost", predicted.get("production_cost"))
    raw_quality = predicted.get("predicted_quality", predicted.get("quality"))
    raw_perf = predicted.get("predicted_performance", predicted.get("performance"))

    yield_pct = None
    if raw_yield is not None:
        try:
            y = float(raw_yield)
            yield_pct = y * 100.0 if y <= 1.0 else y
        except Exception:
            yield_pct = None

    out: Dict[str, Any] = {}
    if yield_pct is not None:
        out["yield_percentage"] = yield_pct
    if raw_energy is not None:
        try:
            out["energy_consumption"] = float(raw_energy)
        except Exception:
            pass
    if raw_cost is not None:
        try:
            out["production_cost"] = float(raw_cost)
        except Exception:
            pass
    if raw_quality is not None:
        try:
            out["quality"] = float(raw_quality)
        except Exception:
            pass
    if raw_perf is not None:
        try:
            out["performance"] = float(raw_perf)
        except Exception:
            pass

    # Include original predicted_* keys for downstream consumers.
    for k, v in predicted.items():
        if k not in out:
            out[k] = v

    return out


def _build_dynamic_chart_data(batch_parameters: Dict[str, float], points: int = 60) -> List[Dict[str, float]]:
    """
    Generates dynamic, non-deterministic telemetry data for the dashboard chart.
    Uses the actual input parameters as the baseline and adds random noise to simulate a live stream.
    """
    import random
    
    t0 = float(batch_parameters.get("temperature", 180.0) or 180.0)
    golden = t0  # Golden batch baseline
    upper = t0 + 2.0
    lower = t0 - 2.0
    data: List[Dict[str, float]] = []

    # Start temperature slightly off and converge, adding random noise
    current_temp = t0 - 5.0 

    for i in range(points):
        # Converge towards target over the first 20 minutes
        if current_temp < t0:
            current_temp += random.uniform(0.1, 0.4)
            
        if current_temp > t0:
            current_temp -= random.uniform(0.1, 0.4)
            
        noise = random.uniform(-0.5, 0.5)
        measurement = current_temp + noise
        
        data.append(
            {
                "Time_Minutes": float(i),
                "Temperature_C": float(measurement),
                "Target_Upper": float(upper),
                "Target_Lower": float(lower),
            }
        )

    return data


@router.post("/monitor")
def monitor(request: MonitoringRequest):
    batch_parameters = _as_batch_parameters(request)

    try:
        report = monitor_batch(batch_parameters)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Monitoring pipeline error: {e}")

    if not isinstance(report, dict):
        raise HTTPException(status_code=500, detail="Monitoring pipeline returned invalid response")

    if report.get("error"):
        raise HTTPException(status_code=500, detail=report.get("error"))

    raw_predicted = report.get("predicted_metrics", {})
    predicted_metrics = _normalize_predicted_metrics(raw_predicted)
    chart_data = report.get("telemetryData") or report.get("chart_data") or _build_dynamic_chart_data(batch_parameters)

    # Extract trust/confidence metadata from raw predictions
    is_heuristic = raw_predicted.get("is_heuristic_fallback", False)
    prediction_confidence = float(raw_predicted.get("prediction_confidence", 0.0) or 0.0)

    # Use ML-generated historical context from monitor report
    raw_hist = report.get("historical_metrics", {})
    historical_metrics = {
        "yield": [round(v * 100 if v <= 1.0 else v, 1) for v in raw_hist.get("yield", [])],
        "energy": raw_hist.get("energy", []),
        "quality": raw_hist.get("quality", []),
    }

    return {
        "predicted_metrics": predicted_metrics,
        "closest_signature": report.get("closest_signature", "unknown"),
        "drift_detected": bool(report.get("drift_detected", False)),
        "drift_score": float(report.get("drift_score", 0.0) or 0.0),
        "batch_status": report.get("batch_status", "unknown"),
        "alert_message": report.get("alert_message", ""),
        "recommended_action": report.get("recommended_action", ""),
        "optimization_suggestion": report.get("optimization_suggestion", {}),
        "is_heuristic_fallback": is_heuristic,
        "prediction_confidence": prediction_confidence,
        "historical_metrics": historical_metrics,
        # Dashboard expects one of these keys for chart data
        "chart_data": chart_data,
    }
