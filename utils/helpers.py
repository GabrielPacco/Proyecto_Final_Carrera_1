import cv2
import os
import numpy as np

def load_image(image_path):
    """
    Carga una imagen desde la ruta especificada.
    """
    return cv2.imread(image_path)

def save_image(image, destination_path, file_name):
    """
    Guarda una imagen en la ruta especificada con el nombre de archivo dado.
    """
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    cv2.imwrite(os.path.join(destination_path, file_name), image)

def display_image(image, title="Image"):
    """
    Muestra una imagen en una ventana.
    """
    cv2.imshow(title, image)
    cv2.waitKey(0)  # Espera hasta que se presione una tecla
    cv2.destroyAllWindows()

def resize_image(image, target_size):
    """
    Cambia el tamaño de una imagen al tamaño objetivo.
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def convert_to_grayscale(image):
    """
    Convierte una imagen a escala de grises.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def calculate_histogram(image, mask=None, bins=256, range=(0, 256)):
    """
    Calcula el histograma de una imagen. Opcionalmente, se puede aplicar una máscara.
    """
    hist = cv2.calcHist([image], [0], mask, [bins], range)
    return hist.flatten()

def normalize_histogram(hist):
    """
    Normaliza un histograma para que la suma de sus valores sea 1.
    """
    return hist / sum(hist)

def create_circular_mask(h, w, center=None, radius=None):
    """
    Crea una máscara circular para una imagen con altura h y anchura w.
    """
    if center is None:  # Utiliza el centro de la imagen
        center = (int(w/2), int(h/2))
    if radius is None:  # Utiliza el radio más pequeño posible
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    mask = dist_from_center <= radius
    return mask.astype(np.uint8)

def apply_mask(image, mask):
    """
    Aplica una máscara binaria a una imagen.
    """
    return cv2.bitwise_and(image, image, mask=mask)
