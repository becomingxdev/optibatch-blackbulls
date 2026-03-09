"""
This file is responsible for the main FastAPI application.
It is part of the api module and will serve as the system entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

# Import routers
from optibatch.api.prediction_api import router as prediction_router
from optibatch.api.optimization_api import router as optimization_router
from optibatch.api.monitoring_api import router as monitoring_router
from optibatch.api.simulation_api import router as simulation_router

app = FastAPI(
    title="OptiBatch Industrial AI API",
    description="Predictive and Prescriptive AI for Industrial Batch Optimization",
    version="1.0"
)

# Enable CORS (safe defaults for local dev + configurable via env)
default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

env_origins = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
allow_origins = (
    [o.strip() for o in env_origins.split(",") if o.strip()]
    if env_origins
    else default_origins
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(prediction_router, tags=["Prediction"])
app.include_router(optimization_router, tags=["Optimization"])
app.include_router(monitoring_router, tags=["Monitoring"])
app.include_router(simulation_router, tags=["Simulation"])

@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "OptiBatch API running"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("optibatch.api.app:app", host="0.0.0.0", port=8000, reload=True)
