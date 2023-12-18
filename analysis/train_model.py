# train_model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from data.prepare_data import load_and_split_data
from config.settings import FEATURES_DATA_PATH, MODEL_DATA_PATH
import joblib
import os
import numpy as np

def train_and_save_model(csv_filepath, model_path):
    # Cargar y dividir los datos
    X_train, X_test, y_train, y_test, class_names = load_and_split_data(csv_filepath)

    # Crear y entrenar el modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Hacer predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(classification_report(y_test, y_pred))

    # Guardar el modelo entrenado para uso futuro
    joblib.dump(model, os.path.join(model_path, 'modelo_entrenado_rf.joblib'))

    # Devolver el modelo y los nombres de las clases
    return model, class_names

if __name__ == "__main__":
    # Definir la ruta al archivo CSV con las características
    csv_filepath = os.path.join(FEATURES_DATA_PATH, 'features.csv')

    # Entrenar el modelo y guardar
    model, class_names = train_and_save_model(csv_filepath, MODEL_DATA_PATH)

    # Opcional: imprimir los nombres de las clases
    print("Clases del modelo:", class_names)
