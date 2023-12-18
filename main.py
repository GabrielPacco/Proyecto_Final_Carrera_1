# main.py
import os
import logging
from data.data_manager import DataManager
from preprocessing.preprocessing import preprocess_image
from preprocessing.segmentation import segment_leaf
from preprocessing.feature_extraction import extract_all_features
from preprocessing.save_features import save_features
from analysis.train_model import train_and_save_model
from config.settings import BASE_DATA_PATH, MODEL_DATA_PATH, FEATURES_DATA_PATH

# Configurar el registro para incluir el tiempo.
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def process_images_from_directory(directory_path):
    # Inicializar el administrador de datos
    data_manager = DataManager(directory_path)

    # Procesar imágenes de cada subcarpeta
    for subfolder in os.listdir(directory_path):
        subfolder_path = os.path.join(directory_path, subfolder)
        if os.path.isdir(subfolder_path):
            logging.info(f"Procesando imágenes en la carpeta: {subfolder}")
            images, file_names = data_manager.load_images_from_folder(subfolder)

            # Determinar la etiqueta de la imagen basada en el nombre de la carpeta
            label = 'Sana' if 'healthy' in subfolder else 'Enferma'

            for image in images:
                preprocessed_image = preprocess_image(image)
                segmented_image, _ = segment_leaf(preprocessed_image)
                features = extract_all_features(segmented_image, preprocessed_image)
                save_features(features, label, subfolder)

    logging.info("Procesamiento de imágenes y guardado de características completado.")

def main():
    # Procesar imágenes y guardar características
    process_images_from_directory(BASE_DATA_PATH)

    # Entrenar el modelo con los datos procesados
    csv_filepath = os.path.join(FEATURES_DATA_PATH, 'features.csv')
    train_and_save_model(csv_filepath, MODEL_DATA_PATH)

    # A partir de aquí, puedes realizar ajustes de hiperparámetros o hacer predicciones con el modelo

if __name__ == "__main__":
    main()
