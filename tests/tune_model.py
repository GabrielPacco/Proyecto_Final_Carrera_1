# tune_model.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from data.prepare_data import load_and_split_data, normalize_features
from analysis.evaluate_model import evaluate_model
from config.settings import FEATURES_DATA_PATH, MODEL_DATA_PATH
import joblib

def tune_hyperparameters(X_train, y_train):
    # Definir un rango de hiperparámetros para probar
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    # Inicializar el modelo
    model = RandomForestClassifier(random_state=42)

    # Inicializar GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')

    # Ajustar GridSearchCV a los datos de entrenamiento
    grid_search.fit(X_train, y_train)

    # Ver los mejores parámetros encontrados
    print(f'Mejores parámetros: {grid_search.best_params_}')
    
    # Devolver el mejor modelo
    return grid_search.best_estimator_

if __name__ == "__main__":
    # Suponiendo que 'features.csv' es tu archivo de características
    csv_filepath = f'{FEATURES_DATA_PATH}/features.csv'
    
    # Cargar y dividir los datos
    X_train, X_test, y_train, y_test = load_and_split_data(csv_filepath)

    # Normalizar las características
    X_train_scaled, X_test_scaled = normalize_features(X_train, X_test)

    # Ajustar hiperparámetros
    best_model = tune_hyperparameters(X_train_scaled, y_train)

    # Evaluar el modelo ajustado
    evaluate_model(best_model, X_test_scaled, y_test)

    # Guardar el modelo ajustado
    joblib.dump(best_model, f'{MODEL_DATA_PATH}/modelo_ajustado_rf.joblib')
