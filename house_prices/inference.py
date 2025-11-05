import pandas as pd
import joblib
from house_prices.preprocess import create_processed_dfs  
# reuse function from preprocess.py
import mlflow.sklearn
import mlflow

# Hard-coded feature lists
cont_features = ['GrLivArea', 'YearBuilt']
cat_nom_features = ['Neighborhood']
cat_ord_features = ['KitchenQual']

# Connect to MLflow
mlflow.set_tracking_uri("file:./mlruns")  # same as in train.py
experiment_name = "house_price_regression"
experiment = mlflow.get_experiment_by_name(experiment_name)

# Fetch all runs and pick the best (lowest RMSLE)
runs_df = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.rmsle ASC"]
)
best_run_id = runs_df.iloc[0].run_id
print(f"Using best run: {best_run_id}")

# Load model from MLflow
linear_regression_model_loaded = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")

# Load transformers from MLflow artifacts
def load_transformer(run_id: str, artifact_name: str):
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_name)
    return joblib.load(local_path)

standard_scaler_load = load_transformer(best_run_id, "standard_scaler.joblib")
one_hot_encoder_load = load_transformer(best_run_id, "one_hot_encoder.joblib")
ordinal_encoder_load = load_transformer(best_run_id, "ordinal_encoder.joblib")




def make_predictions(inference_df: pd.DataFrame) -> pd.Series:
    """
    Preprocesses inference data using fitted transformers and predicts house prices.
    """
    # --- Continuous features
    X_infer_cont = standard_scaler_load.transform(inference_df[cont_features])

    # --- Nominal features
    X_infer_ohe = one_hot_encoder_load.transform(inference_df[cat_nom_features])
    ohe_cols = list(one_hot_encoder_load.get_feature_names_out(cat_nom_features))

    # --- Ordinal features
    X_infer_ord = ordinal_encoder_load.transform(inference_df[cat_ord_features])

    # --- Combine all features using create_processed_dfs
    # Since create_processed_dfs expects X_train and X_test, we pass inference_df for both
    X_infer_final_df, _ = create_processed_dfs(
        X_train=inference_df,
        X_test=inference_df,  # ignored, just to satisfy function signature
        X_train_cont=X_infer_cont,
        X_test_cont=X_infer_cont,
        cont_features=cont_features,
        X_train_ohe=X_infer_ohe,
        X_test_ohe=X_infer_ohe,
        ohe_cols=ohe_cols,
        X_train_ord=X_infer_ord,
        X_test_ord=X_infer_ord,
        cat_ord_features=cat_ord_features
    )

    # --- Predict using loaded model
    predictions = linear_regression_model_loaded.predict(X_infer_final_df.values)
    #add fixes
    return predictions
