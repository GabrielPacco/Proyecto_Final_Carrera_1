# save_features.py
import csv
from config.settings import FEATURES_DATA_PATH
import os
import numpy as np

def initialize_features_file(num_color_features, num_texture_features, num_shape_features):
    # Generar los encabezados para el archivo CSV
    color_headers = ['Color_Feature_' + str(i) for i in range(num_color_features)]
    texture_headers = ['Texture_Feature_' + str(i) for i in range(num_texture_features)]
    shape_headers = ['Area', 'Perimeter']
    label_header = ['Label']  # Encabezado para la etiqueta de la imagen

    # Combinar todos los encabezados
    headers = color_headers + texture_headers + shape_headers + label_header

    # Escribir los encabezados en el archivo CSV
    features_file_path = os.path.join(FEATURES_DATA_PATH, 'features.csv')
    with open(features_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

def save_features(features, label, folder_name):
    # Asegurarse de que las características sean un array de NumPy y convertir la etiqueta y el nombre de la carpeta en array
    features_array = np.array(features)
    label_array = np.array([label])
    folder_name_array = np.array([folder_name])

    # Concatenar todas las características con la etiqueta y el nombre de la carpeta
    row = np.concatenate((features_array, label_array, folder_name_array))

    # Asegurarse de que el archivo de características exista, si no, inicializarlo
    features_file_path = os.path.join(FEATURES_DATA_PATH, 'features.csv')
    if not os.path.isfile(features_file_path):
        # Ajustar la cantidad de características de acuerdo a lo definido en la extracción de características
        initialize_features_file(len(features_array) - 2 - 1, 24, 2)

    with open(features_file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)
