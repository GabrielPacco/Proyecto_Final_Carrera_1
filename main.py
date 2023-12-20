import os
import numpy as np
import joblib
from data.data_manager import DataManager
from preprocessing.preprocessing import preprocess_image
from preprocessing.segmentation import segment_leaf
from preprocessing.feature_extraction import extract_all_features
from preprocessing.save_images import save_image
from preprocessing.save_features import save_features_to_npy
from analysis.train_model import train_and_save_model
from analysis.evaluate_model import evaluate_model, load_test_data
from analysis.resultados import plot_confusion_matrix, plot_roc_curve, plot_clusters, plot_feature_importance
from sklearn.model_selection import train_test_split
from config.settings import BASE_DATA_PATH, SEGMENTED_DATA_PATH, FEATURES_DATA_PATH, MODEL_DATA_PATH, LABEL_DATA_PATH

def main():
    # Inicializar el administrador de datos
    data_manager = DataManager(BASE_DATA_PATH)

    # Obtener todas las subcarpetas que corresponden a las categorías de hojas
    categories = [f.name for f in os.scandir(BASE_DATA_PATH) if f.is_dir()]

    # Preparar el conjunto de datos para las características y etiquetas
    feature_dataset = []
    label_dataset = []

    for category in categories:
        print(f"Procesando imágenes para la categoría: {category}")
        images, image_filenames = data_manager.load_images_from_folder(category)

        for image, filename in zip(images, image_filenames):
            preprocessed_image = preprocess_image(image)
            segmented_image, _ = segment_leaf(preprocessed_image, image)
            save_image(segmented_image, filename, category)

            features = extract_all_features(segmented_image, image)
            feature_dataset.append(features)
            label = 0 if category == 'Tomato___healthy' else 1
            label_dataset.append(label)
            #plot_clusters(features)

    # Convertir las listas a arrays de NumPy y guardarlas
    feature_dataset = np.array(feature_dataset)
    label_dataset = np.array(label_dataset)
    save_features_to_npy(feature_dataset, label_dataset, os.path.join(FEATURES_DATA_PATH, 'features.npy'))

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(feature_dataset, label_dataset, test_size=0.3, random_state=42)

    # Llamar a la función para entrenar y guardar el modelo
    train_and_save_model(X_train, y_train, MODEL_DATA_PATH)

    # Cargar el modelo entrenado
    model_directory = MODEL_DATA_PATH  # Solo el directorio, no el nombre del archivo
    model = train_and_save_model(feature_dataset, label_dataset, model_directory)

    # Ajustar el conjunto de prueba para que coincida con el número de características del conjunto de entrenamiento
    if X_test.shape[1] != X_train.shape[1]:
        print(f"Ajustando el número de características de prueba de {X_test.shape[1]} a {X_train.shape[1]}")
        X_test = X_test[:, :X_train.shape[1]]

    # Evaluar el modelo
    evaluate_model(model, X_test, y_test)

    # Visualizaciones de resultados
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(model, X_test)
    plot_roc_curve(y_test, y_pred)

if __name__ == "__main__":
    main()
