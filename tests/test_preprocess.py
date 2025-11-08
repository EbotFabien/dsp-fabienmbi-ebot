import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import pytest
from house_prices.preprocess import (
    split_data,
    scale_continuous_features,
    encode_nominal_features,
    encode_ordinal_features,
    create_processed_dfs
)

# Sample test data
data = pd.DataFrame({
    'GrLivArea': [1000, 1500, 2000],
    'YearBuilt': [2000, 1990, 2010],
    'Neighborhood': ['A', 'B', 'A'],
    'KitchenQual': ['TA', 'Gd', 'Ex'],
    'SalePrice': [200000, 250000, 300000]
})

def test_split_data():
    X_train, X_test, y_train, y_test = split_data(
        data, ['GrLivArea', 'YearBuilt'], ['Neighborhood'], ['KitchenQual'], 'SalePrice', test_size=0.33, random_state=42
    )
    assert len(X_train) == 2
    assert len(X_test) == 1
    assert list(y_train.index) != list(y_test.index)  # Check split

def test_scale_continuous_features():
    X_train, X_test, _,_= split_data(data, ['GrLivArea', 'YearBuilt'], ['Neighborhood'], ['KitchenQual'],label_col='SalePrice')
    X_train_scaled, X_test_scaled, scaler = scale_continuous_features(X_train, X_test, ['GrLivArea', 'YearBuilt'],log_mlflow=False)
    assert X_train_scaled.shape == (2, 2)
    assert np.isclose(np.mean(X_train_scaled), 0, atol=1e-6)  # Scaled mean ~0

def test_encode_nominal_features():
    X_train, X_test, _, _ = split_data(
        data, ['GrLivArea', 'YearBuilt'], ['Neighborhood'], ['KitchenQual'], 'SalePrice'
    )
    X_train_ohe, X_test_ohe, encoder, ohe_cols = encode_nominal_features(X_train, X_test, ['Neighborhood'],log_mlflow=False)
    assert X_train_ohe.shape[1] == len(ohe_cols)
    assert encoder.get_feature_names_out(['Neighborhood']).tolist() == ohe_cols




def test_create_processed_dfs():
    X_train, X_test, _, _ = split_data(
        data, ['GrLivArea', 'YearBuilt'], ['Neighborhood'], ['KitchenQual'], 'SalePrice'
    )

    #  Apply all preprocessing
    X_train_scaled, X_test_scaled, scaler = scale_continuous_features(X_train, X_test, ['GrLivArea', 'YearBuilt'], log_mlflow=False)
    X_train_ohe, X_test_ohe, ohe_encoder, ohe_cols = encode_nominal_features(X_train, X_test, ['Neighborhood'], log_mlflow=False)
    X_train_ord, X_test_ord, ord_encoder = encode_ordinal_features(X_train, X_test, ['KitchenQual'], ord_categories=[['Po','Fa','TA','Gd','Ex']], log_mlflow=False)

    #  Combine into final DataFrames
    df_train_final, df_test_final = create_processed_dfs(
        X_train, X_test,
        X_train_scaled, X_test_scaled, ['GrLivArea', 'YearBuilt'],
        X_train_ohe, X_test_ohe, ohe_cols,
        X_train_ord, X_test_ord, ['KitchenQual']
    )

    #  Assertions
    # Shape
    expected_train_cols = 2 + len(ohe_cols) + 1  # continuous + one-hot + ordinal
    expected_test_cols = expected_train_cols
    assert df_train_final.shape == (2, expected_train_cols)
    assert df_test_final.shape == (1, expected_test_cols)

    # Column names
    assert all(col in df_train_final.columns for col in ['GrLivArea', 'YearBuilt', 'KitchenQual'])
    assert all(col in df_train_final.columns for col in ohe_cols)

    # Row indices
    assert list(df_train_final.index) == list(X_train.index)
    assert list(df_test_final.index) == list(X_test.index)
