import numpy as np
# Rutas de acceso a los directorios de datos
BASE_DATA_PATH = "C:/Users/gabri/Downloads/Proyecto_Final_Carrera_1/data/plantvillage"
PREPROCESSED_DATA_PATH = "C:/Users/gabri/Downloads/Proyecto_Final_Carrera_1/data/preprocessed"
SEGMENTED_DATA_PATH = "C:/Users/gabri/Downloads/Proyecto_Final_Carrera_1/data/segmented"
FEATURES_DATA_PATH = "C:/Users/gabri/Downloads/Proyecto_Final_Carrera_1/data/features"

# Parámetros para el preprocesamiento de imágenes
IMAGE_SIZE = (256, 256)  # Tamaño al que se redimensionarán las imágenes
GAUSSIAN_BLUR_KERNEL = (5, 5)  # Tamaño del kernel para el desenfoque gaussiano
GAUSSIAN_BLUR_SIGMA = 1  # Desviación estándar para el desenfoque gaussiano

# Parámetros para la segmentación de imágenes
K_MEANS_CLUSTERS = 3  # Número de clusters para la segmentación K-means
OTSU_THRESHOLD_METHOD = "binary"  # Método de umbralización (binary, binary_inv, etc.)

# Parámetros para la extracción de características
GLCM_DISTANCES = [1]  # Distancias para la matriz de co-ocurrencia de grises
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Ángulos para la matriz de co-ocurrencia de grises

# Parámetros para la clasificación
CLASSIFIER_MODEL_PATH = "C:/Users/gabri/Downloads/Proyecto_Final_Carrera_1/models/classifier.pkl"

# Otros parámetros globales
RANDOM_SEED = 42  # Semilla aleatoria para reproducibilidad en procesos aleatorios
