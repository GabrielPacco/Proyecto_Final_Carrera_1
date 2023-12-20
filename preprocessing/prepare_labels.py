import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

def label_images(basedir):
    labels = []
    filenames = []
    categories = os.listdir(basedir)

    for category in categories:
        folder_path = os.path.join(basedir, category)
        images = os.listdir(folder_path)
        for img_filename in images:
            filenames.append(img_filename)
            labels.append(0 if category == 'Tomato___healthy' else 1)
    
    return filenames, labels

def save_labels(filenames, labels, output_dir):
    np.save(os.path.join(output_dir, 'filenames.npy'), filenames)
    np.save(os.path.join(output_dir, 'labels.npy'), labels)

if __name__ == "__main__":
    # Define el directorio de las imágenes de entrenamiento
    TRAIN_DATA_PATH = "C:/Users/gabri/Downloads/Proyecto_Final_Carrera_1/data/plantvillage/Prueba"
    # Define el directorio donde se guardarán las etiquetas
    LABELS_PATH = "C:/Users/gabri/Downloads/Proyecto_Final_Carrera_1/data/labels"

    # Asegúrate de que el directorio de salida existe
    os.makedirs(LABELS_PATH, exist_ok=True)

    # Etiqueta las imágenes y guarda los resultados
    filenames, labels = label_images(TRAIN_DATA_PATH)
    save_labels(filenames, labels, LABELS_PATH)

    print("Etiquetas y nombres de archivos guardados con éxito.")
