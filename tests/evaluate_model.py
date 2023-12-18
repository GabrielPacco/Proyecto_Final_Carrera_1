# evaluate_model.py
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score
from config.settings import MODEL_DATA_PATH, TEST_DATA_PATH
import joblib
import time
import numpy as np

# Función para cargar el modelo y los conjuntos de datos de prueba
def load_model_and_data(model_path, test_data_path):
    # Cargar el modelo entrenado
    model = joblib.load(model_path)
    
    # Cargar los datos de prueba
    X_test = np.load(test_data_path + 'X_test.npy')
    y_test = np.load(test_data_path + 'y_test.npy')
    
    return model, X_test, y_test

# Función para evaluar el modelo
def evaluate_model(model, X_test, y_test):
    start_time = time.time()
    
    # Hacer predicciones
    y_pred = model.predict(X_test)
    
    end_time = time.time()
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calcular la matriz de confusión y luego calcular especificidad
    conf_matrix = confusion_matrix(y_test, y_pred)
    specificity = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1])
    
    # Imprimir métricas
    print(f'Accuracy: {accuracy}')
    print(f'Recall: {recall}')
    print(f'Specificity: {specificity}')
    print(f'F1-Score: {f1}')
    print(f'Tiempo de ejecución del modelo: {end_time - start_time} segundos')

    return accuracy, recall, specificity, f1

if __name__ == "__main__":
    # Define las rutas a tu modelo y datos de prueba
    model_path = f'{MODEL_DATA_PATH}/modelo_entrenado_rf.joblib'
    test_data_path = f'{TEST_DATA_PATH}/'  # Asegúrate de cerrar la cadena y agregar una barra al final

    # Cargar el modelo y los datos de prueba
    model, X_test, y_test = load_model_and_data(model_path, test_data_path)

    # Evaluar el modelo
    evaluate_model(model, X_test, y_test)
