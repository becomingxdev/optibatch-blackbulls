import json


def main() -> int:
    # NOTE: We intentionally avoid FastAPI TestClient here because it requires
    # the optional `httpx` dependency, which may not be present in minimal setups.
    # Instead, we exercise the actual endpoint callables + request models.
    from optibatch.api.app import app  # ensure routers register
    from optibatch.api.monitoring_api import MonitoringRequest, monitor
    from optibatch.api.optimization_api import OptimizationRequest, optimize
    from optibatch.api.simulation_api import ParameterSweepRequest, api_parameter_sweep
    from optibatch.api.prediction_api import PredictionRequest, predict

    batch_parameters = {
        "temperature": 180.0,
        "pressure": 2.2,
        "hold_time": 45.0,
        "catalyst_ratio": 1.2,
        "strategy": 70.0,
    }

    # 1) Monitoring dashboard request
    monitor_payload = MonitoringRequest(batch_parameters=batch_parameters)
    monitor_resp = monitor(monitor_payload)
    assert isinstance(monitor_resp, dict), monitor_resp
    monitor_json = json.loads(json.dumps(monitor_resp))
    assert "predicted_metrics" in monitor_json
    assert "chart_data" in monitor_json
    assert isinstance(monitor_json["chart_data"], list)

    # 2) Strategy optimization request
    optimize_payload = OptimizationRequest(
        batch_parameters=batch_parameters,
        predicted_metrics=monitor_json.get("predicted_metrics", {}),
    )
    optimize_resp = optimize(optimize_payload)
    optimize_json = json.loads(json.dumps(optimize_resp))
    assert "parameter_recommendations" in optimize_json
    assert "predicted_metrics" in optimize_json

    # 3) Parameter sweep simulation
    sweep_payload = ParameterSweepRequest(
        parameter_ranges={"temperature": [135, 155], "pressure": [1.8, 2.5]},
        num_simulations=10,
    )
    sweep_resp = api_parameter_sweep(sweep_payload)
    sweep_json = json.loads(json.dumps(sweep_resp))
    assert "best_simulated_batches" in sweep_json
    assert isinstance(sweep_json["best_simulated_batches"], list)

    # 4) Prediction request
    pred_payload = PredictionRequest(**batch_parameters)
    pred_resp = predict(pred_payload)
    pred_json = json.loads(json.dumps(pred_resp))
    assert "predicted_metrics" in pred_json
    assert "performance_class" in pred_json

    # Verify the app contains the expected routes (registration check)
    paths = sorted({getattr(r, "path", "") for r in app.routes})
    for required in ("/monitor", "/optimize", "/predict", "/sweep"):
        assert required in paths, f"Missing route: {required}"

    print("E2E smoke test PASSED")
    print(json.dumps({"monitor": monitor_json, "optimize": optimize_json, "sweep": sweep_json, "predict": pred_json}, indent=2)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

