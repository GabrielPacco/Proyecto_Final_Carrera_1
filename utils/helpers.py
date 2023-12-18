import cv2
import os
import numpy as np
import logging

def load_image(image_path):
    """
    Carga una imagen desde la ruta especificada.
    """
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"No se pudo cargar la imagen desde {image_path}")
    return image

def save_image(image, destination_path, file_name):
    """
    Guarda una imagen en la ruta especificada con el nombre de archivo dado.
    """
    try:
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
        cv2.imwrite(os.path.join(destination_path, file_name), image)
    except Exception as e:
        logging.error(f"Error al guardar la imagen en {os.path.join(destination_path, file_name)}: {e}")

def display_image(image, title="Image"):
    """
    Muestra una imagen en una ventana.
    """
    if image is not None:
        cv2.imshow(title, image)
        cv2.waitKey(0)  # Espera hasta que se presione una tecla
        cv2.destroyAllWindows()
    else:
        logging.error("Intentó mostrar una imagen nula")

def resize_image(image, target_size):
    """
    Cambia el tamaño de una imagen al tamaño objetivo.
    """
    if image is not None:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    else:
        logging.error("Intentó redimensionar una imagen nula")
        return None

def convert_to_grayscale(image):
    """
    Convierte una imagen a escala de grises.
    """
    if image is not None and len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        logging.error("Intentó convertir a escala de grises una imagen no válida")
        return None

def calculate_histogram(image, mask=None, bins=256, range=(0, 256)):
    """
    Calcula el histograma de una imagen. Opcionalmente, se puede aplicar una máscara.
    """
    if image is not None:
        hist = cv2.calcHist([image], [0], mask, [bins], range)
        return hist.flatten()
    else:
        logging.error("Intentó calcular el histograma de una imagen nula")
        return None

def normalize_histogram(hist):
    """
    Normaliza un histograma para que la suma de sus valores sea 1.
    """
    if hist is not None:
        return hist / sum(hist)
    else:
        logging.error("Intentó normalizar un histograma nulo")
        return None

def create_circular_mask(h, w, center=None, radius=None):
    """
    Crea una máscara circular para una imagen con altura h y anchura w.
    """
    if center is None:  
        center = (int(w/2), int(h/2))
    if radius is None:  
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    mask = dist_from_center <= radius
    return mask.astype(np.uint8)

def apply_mask(image, mask):
    """
    Aplica una máscara binaria a una imagen.
    """
    if image is not None and mask is not None:
        return cv2.bitwise_and(image, image, mask=mask)
    else:
        logging.error("Intentó aplicar una máscara a una imagen nula")
        return None
