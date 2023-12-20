# test_data_preparation.py
import os
import numpy as np
from data.data_manager import DataManager
from preprocessing.preprocessing import preprocess_image
from preprocessing.segmentation import segment_leaf
from preprocessing.feature_extraction import extract_all_features
from preprocessing.save_features import save_features_to_npy
from config.settings import TRAIN_DATA_PATH, FEATURES_DATA_PATH

def prepare_test_data():
    # Inicializar el administrador de datos
    data_manager = DataManager(TRAIN_DATA_PATH)

    # Preparar el conjunto de datos para las características
    feature_dataset = []

    # Obtener todas las imágenes para procesar
    images, image_filenames = data_manager.load_all_images()

    for image in images:
        preprocessed_image = preprocess_image(image)
        segmented_image, _ = segment_leaf(preprocessed_image, image)
        features = extract_all_features(segmented_image, image)
        feature_dataset.append(features)

    # Convertir las listas a arrays de NumPy y guardarlas
    feature_dataset = np.array(feature_dataset)
    save_features_to_npy(feature_dataset, os.path.join(FEATURES_DATA_PATH, 'test_features.npy'))

if __name__ == "__main__":
    prepare_test_data()
