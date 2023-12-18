# make_predictions.py
import joblib
from preprocessing.feature_extraction import extract_all_features
from data.data_manager import load_all_images
from config.settings import MODEL_DATA_PATH, NEW_IMAGE_PATH

def load_model(model_path):
    # Cargar el modelo entrenado
    return joblib.load(model_path)

def make_prediction(model, features):
    # Hacer predicciones con el modelo
    return model.predict([features])

if __name__ == "__main__":
    # Cargar el modelo
    model = load_model(f'{MODEL_DATA_PATH}/modelo_ajustado_rf.joblib')

    # Cargar todas las nuevas imágenes para las cuales quieres hacer una predicción
    images, image_names = load_all_images(NEW_IMAGE_PATH)  # Asegúrate de que esta función devuelva las imágenes y sus nombres

    for image, name in zip(images, image_names):
        # Extraer características de la imagen actual
        features = extract_all_features(image)  # Asegúrate de que esta función es compatible con tus imágenes

        # Hacer predicciones
        prediction = make_prediction(model, features)

        # Imprimir o procesar la predicción
        print(f'Predicción para la imagen {name}:', prediction)
