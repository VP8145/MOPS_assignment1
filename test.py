import os

def test_model_exists():
    assert os.path.exists('best_rf_model.joblib')
    assert os.path.exists('scaler.joblib')
