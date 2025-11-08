import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from house_prices.train import build_model
from house_prices.inference import make_predictions

# Small dataset
data = pd.DataFrame({
    'GrLivArea': [1000, 1500, 2000],
    'YearBuilt': [2000, 1990, 2010],
    'Neighborhood': ['A', 'B', 'A'],
    'KitchenQual': ['TA', 'Gd', 'Ex'],
    'SalePrice': [200000, 250000, 300000]
})

def test_end_to_end():
    # Train
    results = build_model(data, env="test")
    assert 'rmsle' in results

    # Predict
    predictions = make_predictions(data.drop(columns="SalePrice"), env="test")
    assert len(predictions) == len(data) #ML flow broken,For tests only
