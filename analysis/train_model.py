# train_model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from config.settings import FEATURES_DATA_PATH, MODEL_DATA_PATH
import joblib
import numpy as np
import os

def load_data(npy_filepath):
    # Cargar los datos desde un archivo .npy
    data = np.load(npy_filepath)
    # Separar las características y las etiquetas
    features = data[:, :-1]
    labels = data[:, -1]
    return features, labels

def train_and_save_model(X, y, model_directory):
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Crear y entrenar el modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred))

    # Guardar el modelo entrenado para uso futuro
    model_file_path = os.path.join(model_directory, 'modelo_entrenado_rf.joblib')
    joblib.dump(model, model_file_path)

    return model

if __name__ == "__main__":
    # Definir la ruta al archivo .npy con las características
    npy_filepath = FEATURES_DATA_PATH + '/features.npy'

    # Definir la ruta al archivo donde se guardará el modelo
    model_path = MODEL_DATA_PATH + '/modelo_entrenado_rf.joblib'

    # Entrenar el modelo y guardarlo
    train_and_save_model(npy_filepath, model_path)
