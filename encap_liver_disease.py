import pandas as pd
import joblib
import unittest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def load_data(location):
    data = pd.read_csv(location)
    return data

def preprocess_data(data):
    # Step 1: Sanity check
    print(data.head(2))

    # Count the number of samples in each class
    class_counts = data['Dataset'].value_counts()
    print(class_counts)

    # Label encoding
    label_encoder = LabelEncoder()
    data['Dataset_Encoded'] = label_encoder.fit_transform(data['Dataset'])
    data = data.drop(['Dataset'], axis=1)

    # Dropping columns
    columns_to_drop = ['Total_Bilirubin', 'Alamine_Aminotransferase', 'Total_Protiens', 'Albumin_and_Globulin_Ratio']
    data = data.drop(columns_to_drop, axis=1)
    print('---------- After dropping columns ----------')
    print(data.head(2))

    # Scaling data
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    return data

def impute_missing_values(data):
    df_class_0 = data.loc[data['Dataset_Encoded'] == 0]
    df_class_1 = data.loc[data['Dataset_Encoded'] == 1]

    print('df_class_0.shape ', df_class_0.shape)
    print('df_class_1.shape ', df_class_1.shape)

    knn_imputer = KNNImputer(n_neighbors=5)

    # Impute for class 0 and class 1
    df_class_0 = pd.DataFrame(knn_imputer.fit_transform(df_class_0), columns=data.columns)
    df_class_1 = pd.DataFrame(knn_imputer.fit_transform(df_class_1), columns=data.columns)

    df_imputed = pd.concat([df_class_0, df_class_1])

    return df_imputed

def split_data(df_imputed):
    Y = df_imputed['Dataset_Encoded']
    X = df_imputed.drop(['Dataset_Encoded'], axis=1)
    return train_test_split(X, Y, test_size=0.2, random_state=0)

def train_model(X_train, Y_train):
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Build and tune Random Forest classifier
    rf_classifier = RandomForestClassifier(random_state=42)
    param_grid = {'n_estimators': [5, 10, 20, 30], 'max_depth': [3, 5, 7]}
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, Y_train)

    print('Best hyperparameters:', grid_search.best_params_)
    best_rf_model = grid_search.best_estimator_

    return best_rf_model, scaler

def save_model(model, scaler, model_path='best_rf_model.joblib', scaler_path='scaler.joblib'):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print('Model and scaler saved')

def main(location):
    data = load_data(location)
    data = preprocess_data(data)
    data = impute_missing_values(data)
    X_train, X_test, Y_train, Y_test = split_data(data)
    model, scaler = train_model(X_train, Y_train)
    save_model(model, scaler)

if __name__ == '__main__':
    main("liver_disease_1.csv")
