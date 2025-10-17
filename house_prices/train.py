import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression
from house_prices.preprocess import (
    split_data,
    scale_continuous_features,
    encode_nominal_features,
    encode_ordinal_features,
    create_processed_dfs
)


def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray, precision: int = 2) -> float:
    """Compute Root Mean Squared Logarithmic Error."""
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)

def build_model(data: pd.DataFrame) -> dict[str, float]:
    """
    Full data preprocessing + model training + evaluation pipeline.

    Parameters
    ----------
    data : pd.DataFrame
        Original dataset containing both features and target.

    Returns
    -------
    results : dict[str, float]
        Dictionary with model performance metrics (e.g., {'rmsle': 0.18}).
    """
    # --- Define feature sets
    cont_features = ['GrLivArea', 'YearBuilt']
    cat_nom_features = ['Neighborhood']
    cat_ord_features = ['KitchenQual']
    ord_categories = [['Po', 'Fa', 'TA', 'Gd', 'Ex']]

    # --- Split data
    X_train, X_test, y_train, y_test = split_data(
        data,
        cont_features,
        cat_nom_features,
        cat_ord_features,
        label_col='SalePrice'
    )

    # --- Scale continuous features
    X_train_cont, X_test_cont, _ = scale_continuous_features(X_train, X_test, cont_features)

    # --- Encode nominal features
    X_train_ohe, X_test_ohe, _, ohe_cols = encode_nominal_features(X_train, X_test, cat_nom_features)

    # --- Encode ordinal features
    X_train_ord, X_test_ord, _ = encode_ordinal_features(X_train, X_test, cat_ord_features, ord_categories)

    # --- Combine all processed features into final DataFrames
    X_train_final_df, X_test_final_df = create_processed_dfs(
        X_train, X_test,
        X_train_cont, X_test_cont, cont_features,
        X_train_ohe, X_test_ohe, ohe_cols,
        X_train_ord, X_test_ord, cat_ord_features
    )

    # --- Train Linear Regression model
    X_train_final = X_train_final_df.values
    X_test_final = X_test_final_df.values

    linear_regression_model = LinearRegression()
    linear_regression_model.fit(X_train_final, y_train)

    # --- Predict and evaluate
    y_pred = linear_regression_model.predict(X_test_final)
    rmsle = compute_rmsle(y_test.values, y_pred)

    # --- Return results
    print("RMSLE:", rmsle)
    return {'rmsle': rmsle}


def test():
    pass