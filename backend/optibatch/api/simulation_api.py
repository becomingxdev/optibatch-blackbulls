"""
This file is responsible for simulation API endpoints.
It is part of the api module and handles batch simulation requests.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any, List

from optibatch.simulation.batch_simulator import simulate_batch, run_parameter_sweep

router = APIRouter()

class SimulationRequest(BaseModel):
    batch_parameters: Dict[str, float]

class ParameterSweepRequest(BaseModel):
    parameter_ranges: Dict[str, List[float]]
    num_simulations: int = 100

@router.post("/simulate")
def api_simulate_batch(request: SimulationRequest):
    result = simulate_batch(request.batch_parameters)
    return result

@router.post("/sweep")
def api_parameter_sweep(request: ParameterSweepRequest):
    result = run_parameter_sweep(request.parameter_ranges, request.num_simulations)
    return result
