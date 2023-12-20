from config.settings import SEGMENTED_DATA_PATH
import cv2
import os

def ensure_dir(directory):
    """Asegura que el directorio exista. Si no existe, lo crea."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directorio creado: {directory}")

def save_image(image, filename, category):
    """Guarda la imagen en el directorio especificado con un nombre basado en el original."""
    # Construye la ruta completa para guardar la imagen con la categoría como subcarpeta
    save_dir = os.path.join(SEGMENTED_DATA_PATH, category)
    ensure_dir(save_dir)  # Asegúrate de que el subdirectorio de la categoría exista
    
    # Crea un nuevo nombre de archivo agregando un prefijo
    new_filename = f"segmented_{filename}"
    save_path = os.path.join(save_dir, new_filename)
    
    # Guarda la imagen en la ruta construida
    success = cv2.imwrite(save_path, image)

