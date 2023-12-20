import cv2
import numpy as np
import config.settings as SEGMENTED_DATA_PATH
import os

def apply_otsu_threshold(image):
    # Convierte la imagen a escala de grises si no lo está
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Asegúrate de que la imagen es de tipo uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    # Aplicar umbral de Otsu
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def find_contours(image):
    """
    Encuentra contornos en una imagen binaria.
    """
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def segment_leaf(preprocessed_image, original_color_image):
    """
    Segmenta la hoja de tomate del fondo aplicando la máscara a la imagen original en color.
    """
    # Aplicar umbralización para obtener la imagen binaria de la imagen preprocesada
    binary_image = apply_otsu_threshold(preprocessed_image)
    
    # Encontrar contornos y asumir que el contorno más grande es la hoja
    contours = find_contours(binary_image)

    if contours:
        # Ordenar los contornos por área y tomar el más grande
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        leaf_contour = contours[0]
        
        # Crear una máscara para la hoja
        mask = np.zeros_like(binary_image)
        cv2.drawContours(mask, [leaf_contour], -1, 255, thickness=cv2.FILLED)
        
        # Aplicar la máscara a la imagen original en color
        segmented_leaf = cv2.bitwise_and(original_color_image, original_color_image, mask=mask)
        return segmented_leaf, mask
    
    else:
        # En caso de no encontrar contornos, devolver la imagen original y una máscara vacía
        return original_color_image, np.zeros_like(original_color_image)

def segment_color_clusters(image, k=3):
    """
    Segmenta la imagen utilizando el algoritmo k-means para agrupar colores.
    """
    # Preparar los datos para k-means
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Definir los criterios de parada de k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # Aplicar k-means
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convertir los centros a un tipo de datos de 8 bits
    centers = np.uint8(centers)
    
    # Asignar cada pixel al centroide de color más cercano
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    return segmented_image, labels.reshape((image.shape[:-1]), centers)
