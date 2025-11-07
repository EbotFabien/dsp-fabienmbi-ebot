import pandas as pd
from house_prices.train import build_model

# Small dummy dataset
data = pd.DataFrame({
    'GrLivArea': [1000, 1500, 2000],
    'YearBuilt': [2000, 1990, 2010],
    'Neighborhood': ['A', 'B', 'A'],
    'KitchenQual': ['TA', 'Gd', 'Ex'],
    'SalePrice': [200000, 250000, 300000]
})

def test_build_model_runs():
    results = build_model(data, env="test")
    assert 'rmsle' in results
    assert results['rmsle'] >= 0
