import os
from data.data_manager import DataManager
from preprocessing.preprocessing import preprocess_image
from preprocessing.segmentation import segment_leaf
from preprocessing.feature_extraction import extract_all_features
from analysis.classification import TomatoLeafClassifier
from config.settings import BASE_DATA_PATH, PREPROCESSED_DATA_PATH, FEATURES_DATA_PATH, CLASSIFIER_MODEL_PATH

def process_images_from_directory(directory_path):
    # Inicializar el administrador de datos
    data_manager = DataManager(directory_path)

    # Obtener todas las subcarpetas (cada enfermedad y hojas sanas)
    subfolders = [f.name for f in os.scandir(directory_path) if f.is_dir()]

    # Inicializar el clasificador (ajustar parámetros según sea necesario)
    classifier = TomatoLeafClassifier()

    # Procesar imágenes de cada subcarpeta
    for subfolder in subfolders:
        print(f"Procesando imágenes en la carpeta: {subfolder}")
        images, _ = data_manager.load_images_from_folder(subfolder)

        # Preprocesar, segmentar y extraer características de cada imagen
        for image in images:
            preprocessed_image = preprocess_image(image)
            segmented_image, _ = segment_leaf(preprocessed_image)
            features = extract_all_features(segmented_image, preprocessed_image)  # Asegúrate de pasar la imagen original también
            # Aquí podrías agregar las características a un conjunto de datos, etc.

            # Opcional: Guardar imagen preprocesada
            preprocessed_image_path = os.path.join(PREPROCESSED_DATA_PATH, subfolder)
            data_manager.save_preprocessed_image(preprocessed_image, 'preprocessed_image.jpg', preprocessed_image_path)

    # Aquí puedes añadir código para entrenar el clasificador con las características extraídas y las etiquetas correspondientes

    # Ejemplo: entrenar el clasificador (necesitarás las etiquetas reales de tus datos)
    # classifier.train(feature_dataset, label_dataset)

    # Opcional: Guardar el modelo entrenado
    # classifier.save_model(CLASSIFIER_MODEL_PATH)

    # Opcional: Cargar y usar el modelo para hacer predicciones
    # classifier.load_model(CLASSIFIER_MODEL_PATH)
    # predictions = classifier.predict(nuevas_caracteristicas)

    # Más código según sea necesario...

if __name__ == "__main__":
    process_images_from_directory(BASE_DATA_PATH)
