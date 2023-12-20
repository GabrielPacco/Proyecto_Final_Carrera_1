import cv2
import numpy as np
import matplotlib.pyplot as plt

def k_means_clustering(image, k=3, attempts=10):
    """
    Aplica el algoritmo K-Means para segmentar la imagen en k clusters.

    :param image: Imagen a segmentar.
    :param k: Número de clusters.
    :param attempts: Número de veces que el algoritmo se ejecutará con diferentes inicializaciones centroides.
    :return: La imagen segmentada con colores representando los diferentes clusters.
    """
    # Preparar los datos para k-means; los píxeles se convierten en una lista de puntos de datos de 2D.
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)

    # Definir criterios y aplicar k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, label, center = cv2.kmeans(Z, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

    # Convertir de nuevo en uint8 y hacer la imagen original de tamaño
    center = np.uint8(center)
    res = center[label.flatten()]
    segmented_image = res.reshape((image.shape))

    return segmented_image, label, center

def apply_mask(image, label, cluster_index):
    """
    Aplica una máscara a la imagen basada en el índice de cluster seleccionado.

    :param image: Imagen original a enmascarar.
    :param label: Etiquetas de clusters de la imagen.
    :param cluster_index: Índice del cluster que se utilizará para la máscara.
    :return: Imagen enmascarada mostrando solo el cluster seleccionado.
    """
    mask = label.reshape((image.shape[0], image.shape[1])) == cluster_index
    return cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))


def analyze_clusters(label, k):
    """
    Analiza los clusters resultantes y muestra la distribución de los píxeles en cada cluster.
    """
    unique, counts = np.unique(label, return_counts=True)
    cluster_info = dict(zip(unique, counts))

    print("Distribución de Clusters:")
    for cluster in cluster_info:
        print(f"Cluster {cluster}: {cluster_info[cluster]} píxeles")

def visualize_clusters(image, label, k):
    """
    Visualiza los clusters resultantes en la imagen.
    """
    plt.figure(figsize=(15, 10))
    for i in range(k):
        plt.subplot(1, k, i + 1)
        mask = label.reshape((image.shape[0], image.shape[1])) == i
        cluster_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
        plt.imshow(cluster_image)
        plt.title(f"Cluster {i}")
        plt.axis('off')
    plt.show()

# Ejemplo de uso
# image = cv2.imread('tu_imagen.jpg')
# segmented_image, label, _ = k_means_clustering(image, k=3)
# analyze_clusters(label, k=3)
# visualize_clusters(image, label, k=3)