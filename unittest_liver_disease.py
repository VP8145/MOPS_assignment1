import unittest
import os
import pandas as pd
from liver_disease import load_data, preprocess_data
from liver_disease import impute_missing_values
from liver_disease import split_data, train_model, save_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class TestLiverDiseaseModel(unittest.TestCase):

    def test_load_data(self):
        location = "liver_disease_1.csv"
        data = load_data(location)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)

    def test_preprocess_data(self):
        data = load_data("liver_disease_1.csv")
        processed_data = preprocess_data(data)
        self.assertIn('Dataset_Encoded', processed_data.columns)
        self.assertNotIn('Total_Bilirubin', processed_data.columns)
        self.assertNotIn('Alamine_Aminotransferase', processed_data.columns)
        self.assertNotIn('Total_Protiens', processed_data.columns)
        self.assertNotIn('Albumin_and_Globulin_Ratio', processed_data.columns)

    def test_impute_missing_values(self):
        data = load_data("liver_disease_1.csv")
        processed_data = preprocess_data(data)
        imputed_data = impute_missing_values(processed_data)
        self.assertFalse(imputed_data.isnull().values.any())

    def test_split_data(self):
        data = load_data("liver_disease_1.csv")
        processed_data = preprocess_data(data)
        imputed_data = impute_missing_values(processed_data)
        X_train, X_test, Y_train, Y_test = split_data(imputed_data)
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)
        self.assertGreater(len(Y_train), 0)
        self.assertGreater(len(Y_test), 0)

    def test_train_model(self):
        data = load_data("liver_disease_1.csv")
        processed_data = preprocess_data(data)
        imputed_data = impute_missing_values(processed_data)
        X_train, X_test, Y_train, Y_test = split_data(imputed_data)
        model, scaler = train_model(X_train, Y_train)
        self.assertIsInstance(model, RandomForestClassifier)
        self.assertIsInstance(scaler, StandardScaler)

    def test_save_model(self):
        data = load_data("liver_disease_1.csv")
        processed_data = preprocess_data(data)
        imputed_data = impute_missing_values(processed_data)
        X_train, X_test, Y_train, Y_test = split_data(imputed_data)
        model, scaler = train_model(X_train, Y_train)
        save_model(model, scaler, 'test_model.joblib', 'test_scaler.joblib')
        self.assertTrue(os.path.exists('test_model.joblib'))
        self.assertTrue(os.path.exists('test_scaler.joblib'))
        os.remove('test_model.joblib')
        os.remove('test_scaler.joblib')


if __name__ == '__main__':
    unittest.main()
