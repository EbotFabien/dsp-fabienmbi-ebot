import pandas as pd
import joblib
from house_prices.preprocess import create_processed_dfs
import mlflow
import mlflow.sklearn
import os

# Hard-coded feature lists (must match training)
cont_features = ['GrLivArea', 'YearBuilt']
cat_nom_features = ['Neighborhood']
cat_ord_features = ['KitchenQual']
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def make_predictions(inference_df: pd.DataFrame, env="production", run_id=None) -> pd.Series:
    """
    Preprocesses inference data using MLflow-trained transformers and predicts house prices.

    Parameters
    ----------
    inference_df : pd.DataFrame
        The input data for prediction.
    env : str
        "production" or "test". Determines which MLflow experiment to use.
    run_id : str, optional
        Specific MLflow run to use. If None, the best run from the experiment is used.

    Returns
    -------
    predictions : np.ndarray
        Predicted house prices.
    """
    # Set MLflow experiment
    mlflow.set_tracking_uri("file:./mlruns")
    experiment_name = (
        "house_price_regression"
        if env == "production"
        else "house_price_regression_test"
    )
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Determine which run to use
    if run_id is None:
        runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.rmsle ASC"]
        )
        if runs_df.empty:
            raise ValueError(
                f"No runs found in MLflow experiment '{experiment_name}'"
            )
        run_id = runs_df.iloc[0].run_id

    print(f"Using MLflow run: {run_id} from experiment '{experiment_name}'")

    # Load model
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    # Helper to load transformers from artifacts
    def load_transformer(run_id: str, artifact_name: str):
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_name
        )
        return joblib.load(local_path)

    # Load transformers
    scaler = load_transformer(run_id, "standard_scaler.joblib")
    ohe = load_transformer(run_id, "one_hot_encoder.joblib")
    ordinal = load_transformer(run_id, "ordinal_encoder.joblib")

    # Preprocess input
    X_cont = scaler.transform(inference_df[cont_features])
    X_ohe = ohe.transform(inference_df[cat_nom_features])
    ohe_cols = list(ohe.get_feature_names_out(cat_nom_features))
    X_ord = ordinal.transform(inference_df[cat_ord_features])

    # Combine features
    X_final, _ = create_processed_dfs(
        X_train=inference_df,
        X_test=inference_df,  # ignored, only for signature
        X_train_cont=X_cont,
        X_test_cont=X_cont,
        cont_features=cont_features,
        X_train_ohe=X_ohe,
        X_test_ohe=X_ohe,
        ohe_cols=ohe_cols,
        X_train_ord=X_ord,
        X_test_ord=X_ord,
        cat_ord_features=cat_ord_features
    )

    # Predict
    predictions = model.predict(X_final.values)
    return predictions
