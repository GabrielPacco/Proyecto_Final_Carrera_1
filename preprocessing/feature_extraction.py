import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray

def extract_color_features(image):
    # Asegúrate de que la imagen sea de tipo float32 o uint8
    if image.dtype != np.uint8 and image.dtype != np.float32:
        image = image.astype(np.uint8)
    
    color_features = []
    if len(image.shape) == 2:  # Si la imagen está en escala de grises
        channel_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        color_features.extend(channel_hist.flatten())
    else:  # Si la imagen es multicanal
        for i in range(3):  # Canales RGB
            channel_hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            color_features.extend(channel_hist.flatten())
    
    return color_features

def extract_texture_features(image):

    # Asegúrate de que la imagen sea en color antes de convertirla a escala de grises
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Solo convierte a escala de grises si es una imagen en color
        gray_image = rgb2gray(image)
    else:
        # Si la imagen ya está en escala de grises, úsala directamente
        gray_image = image

    """
    Extrae características de textura utilizando matrices de co-ocurrencia de grises.
    """
    gray_image = rgb2gray(image)
    # Calcular la matriz de co-ocurrencia en 4 direcciones
    glcm = graycomatrix((gray_image * 255).astype('uint8'), distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    
    # Extraer propiedades de la GLCM
    texture_features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        texture_props = graycoprops(glcm, prop)
        texture_features.extend(texture_props.flatten())
    return texture_features

def extract_shape_features(image):
    """
    Extrae características de forma, como el área, perímetro, etc., de la región segmentada.
    """
    # Convertir imagen a escala de grises y binarizar
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Suponiendo que la mayor región contorneada es la hoja, extraemos sus características
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        return [area, perimeter]
    else:
        return [0, 0]

def extract_all_features(image):

    # Extrae características de la imagen segmentada y de la imagen original
    color_feats = extract_color_features(original_image)  # Usa la imagen original aquí
    texture_feats = extract_texture_features(segmented_image)  # Usa la imagen segmentada
    
    """
    Combina todas las características extraídas en un solo vector.
    """
    color_feats = extract_color_features(image)
    texture_feats = extract_texture_features(image)
    shape_feats = extract_shape_features(image)
    return color_feats + texture_feats + shape_feats
