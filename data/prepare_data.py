# prepare_data.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config.settings import FEATURES_DATA_PATH, TEST_DATA_PATH
import os
import numpy as np

def convert_labels_to_numeric(label):
    if label == 'Enferma':
        return 0
    elif label == 'Sana':
        return 1
    else:
        raise ValueError(f"Etiqueta no reconocida: {label}")


def load_and_split_data(csv_filepath):
    # Cargar el conjunto de datos desde el archivo CSV
    data = pd.read_csv(csv_filepath)

    # Separar las etiquetas de las características
    y = data['Label']
    X = data.drop(columns=['Label'])

    # Codificar las etiquetas si no son numéricas
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Obtener los nombres únicos de las clases (etiquetas)
    class_names = label_encoder.classes_

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, class_names

def normalize_features(X_train, X_test):
    scaler = StandardScaler()

    # Normalizar las características
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled

if __name__ == "__main__":
    csv_filepath = os.path.join(FEATURES_DATA_PATH, 'features.csv')
    
    X_train, X_test, y_train, y_test, class_names = load_and_split_data(csv_filepath)

    # Normalizar las características
    X_train_scaled, X_test_scaled = normalize_features(X_train, X_test)

    # Guardar los conjuntos de datos procesados
    np.save(os.path.join(TEST_DATA_PATH, 'X_train.npy'), X_train_scaled)
    np.save(os.path.join(TEST_DATA_PATH, 'X_test.npy'), X_test_scaled)
    np.save(os.path.join(TEST_DATA_PATH, 'y_train.npy'), y_train)
    np.save(os.path.join(TEST_DATA_PATH, 'y_test.npy'), y_test)
