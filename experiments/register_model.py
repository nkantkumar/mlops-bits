"""Register and promote champion models in the MLflow Model Registry."""

import argparse
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


def register_and_promote(
    experiment_name: str = "fraud-detection",
    model_name: str = "fraud-detector",
    tracking_uri: str = "sqlite:///mlflow.db",
    alias: str = "champion",
    metric_target: str = "p99_latency_ms",
    ascending: bool = True,
    specific_run_id: str | None = None,
) -> dict[str, Any]:
    """Register the best run from an experiment and assign a production alias."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    if specific_run_id:
        run = client.get_run(specific_run_id)
        chosen_run_id = run.info.run_id
        run_name = run.data.tags.get("mlflow.runName", "manual-selection")
        print(f"🎯 Registering specified run: {run_name} ({chosen_run_id})")
    else:
        exp = client.get_experiment_by_name(experiment_name)
        if not exp:
            raise ValueError(f"Experiment '{experiment_name}' not found.")

        order_dir = "ASC" if ascending else "DESC"
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=[f"metrics.{metric_target} {order_dir}"],
        )
        if not runs:
            raise ValueError(f"No runs found in experiment '{experiment_name}'.")

        best_run = runs[0]
        chosen_run_id = best_run.info.run_id
        run_name = best_run.data.tags.get("mlflow.runName", "best-run")
        metric_val = best_run.data.metrics.get(metric_target, 0.0)
        print(
            f"🏆 Best run by {metric_target} ({metric_val}): {run_name} (ID: {chosen_run_id})"
        )

    # 1. Register model
    model_uri = f"runs:/{chosen_run_id}/model"
    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = str(model_version.version)

    # 2. Assign alias (e.g., champion, challenger)
    client.set_registered_model_alias(name=model_name, alias=alias, version=version)
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="deployment_status",
        value=alias,
    )

    print("\n" + "=" * 70)
    print("✅ MODEL SUCCESSFULLY REGISTERED IN MLFLOW REGISTRY")
    print("=" * 70)
    print(f"Model Name    : {model_name}")
    print(f"Version       : {version}")
    print(f"Alias         : @{alias}  -> models:/{model_name}@{alias}")
    print(f"Source Run    : {run_name} ({chosen_run_id})")
    print("=" * 70)

    print("\n🚀 SERVING OPTIONS:")
    print("1. MLflow Built-In Server:")
    print(
        f'   uv run mlflow models serve -m "models:/{model_name}@{alias}" --port 8000 --env-manager local'
    )
    print("\n2. Enterprise FastAPI Service:")
    print("   uv run mlops-bits serve --host 0.0.0.0 --port 8000")
    print("\n3. Build Docker Container:")
    print(
        f'   uv run mlflow models build-docker -m "models:/{model_name}@{alias}" -n {model_name}:{alias}'
    )
    print("=" * 70 + "\n")

    return {
        "model_name": model_name,
        "version": version,
        "alias": alias,
        "run_id": chosen_run_id,
        "run_name": run_name,
    }


def main() -> None:
    """CLI runner for model registration."""
    parser = argparse.ArgumentParser(
        description="Register and promote model in MLflow Model Registry."
    )
    parser.add_argument(
        "--experiment-name", default="fraud-detection", help="MLflow experiment name"
    )
    parser.add_argument(
        "--model-name", default="fraud-detector", help="Registered model name"
    )
    parser.add_argument(
        "--tracking-uri", default="sqlite:///mlflow.db", help="MLflow tracking URI"
    )
    parser.add_argument(
        "--alias", default="champion", help="Deployment alias (@champion, @challenger)"
    )
    parser.add_argument(
        "--metric", default="p99_latency_ms", help="Metric to rank runs"
    )
    parser.add_argument("--run-id", default=None, help="Specific Run ID to register")

    args = parser.parse_args()
    ascending = not args.metric.startswith(("roc", "f1", "acc", "prec", "rec"))

    register_and_promote(
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        tracking_uri=args.tracking_uri,
        alias=args.alias,
        metric_target=args.metric,
        ascending=ascending,
        specific_run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
