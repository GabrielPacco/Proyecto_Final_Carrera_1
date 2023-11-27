import cv2
import numpy as np

def convert_to_lab(image):
    """
    Convierte una imagen de entrada en el espacio de color RGB al espacio de color L*a*b*.
    Esto es útil para la segmentación basada en color que será más resistente a las variaciones de iluminación.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    """
    Aplica un desenfoque gaussiano para suavizar la imagen y reducir el ruido y los detalles finos.
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)

def threshold_image(image, method=cv2.THRESH_BINARY_INV):
    """
    Aplica un umbral a la imagen para convertirla en una imagen binaria. El método por defecto es THRESH_BINARY_INV.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, method + cv2.THRESH_OTSU)
    return binary_image

def normalize_image(image):
    """
    Normaliza los valores de píxeles en la imagen para que estén en el rango [0, 1].
    """
    return image / 255.0

def resize_image(image, size=(256, 256)):
    """
    Redimensiona la imagen al tamaño especificado.
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def preprocess_image(image):
    """
    Aplica una serie de funciones de preprocesamiento a la imagen y la devuelve.
    """
    lab_image = convert_to_lab(image)
    blurred_image = apply_gaussian_blur(lab_image)
    binary_image = threshold_image(blurred_image)
    normalized_image = normalize_image(binary_image)
    resized_image = resize_image(normalized_image)
    return resized_image
