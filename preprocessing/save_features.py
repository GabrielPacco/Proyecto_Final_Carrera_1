# save_features.py
from config.settings import FEATURES_DATA_PATH
import os
import numpy as np

label_mapping = {'Enferma': 1, 'Sana': 0}

def convert_label(label):
    return label_mapping.get(label, -1)  # Retorna -1 si la etiqueta no se encuentra

def save_features_to_npy(features_list, labels_list, filename="features.npy"):
    # Asegurarse de que las características sean un array de NumPy
    features_array = np.array(features_list)
    
    # Convertir etiquetas en un array de NumPy
    labels_array = np.array([convert_label(label) for label in labels_list])

    # Combinar características y etiquetas
    data_to_save = np.column_stack((features_array, labels_array))

    # Guardar este array en un archivo .npy
    features_file_path = os.path.join(FEATURES_DATA_PATH, filename)
    np.save(features_file_path, data_to_save)
    print(f"Características guardadas en: {features_file_path}")

# Uso de la función
# features_list = [...] # Lista con todas las características extraídas
# labels_list = [...]   # Lista con todas las etiquetas correspondientes
# save_features_to_npy(features_list, labels_list)
