from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from config.settings import FEATURES_DATA_PATH, MODEL_DATA_PATH, LABEL_DATA_PATH
import joblib
import numpy as np
import os

def load_test_data(features_path, labels_path):
    # Cargar las características y las etiquetas de prueba
    X_test = np.load(features_path)
    y_test = np.load(labels_path)
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Calcular métricas de rendimiento
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    # Mostrar la matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="g")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    # Define las rutas a tu modelo y datos de prueba
    model_path = os.path.join(MODEL_DATA_PATH, 'modelo_entrenado_rf.joblib')
    
    # Aquí actualizamos las rutas para apuntar al directorio correcto
    test_features_path = os.path.join(LABEL_DATA_PATH, 'test_features.npy')
    test_labels_path = os.path.join(LABEL_DATA_PATH, 'labels', 'labels.npy')  # Ruta actualizada para las etiquetas

    # Asegurarse de que el modelo y los datos de prueba existen
    if not os.path.exists(model_path) or not os.path.exists(test_features_path) or not os.path.exists(test_labels_path):
        raise FileNotFoundError("Modelo o datos de prueba no encontrados.")

    # Cargar el modelo y los datos de prueba
    model = joblib.load(model_path)
    X_test, y_test = load_test_data(test_features_path, test_labels_path)

    # Evaluar el modelo
    evaluate_model(model, X_test, y_test)
