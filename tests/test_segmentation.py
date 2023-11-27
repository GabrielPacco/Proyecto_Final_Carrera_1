import unittest
import numpy as np
import cv2
from preprocessing.segmentation import apply_otsu_threshold, find_contours, segment_leaf

class TestSegmentation(unittest.TestCase):

    def setUp(self):
        # Crear una imagen de prueba
        self.test_image = np.random.rand(100, 100, 3) * 255
        self.test_image = self.test_image.astype(np.uint8)

    def test_apply_otsu_threshold(self):
        # Prueba la función de umbralización de Otsu
        thresholded_image = apply_otsu_threshold(self.test_image)
        self.assertEqual(thresholded_image.shape, self.test_image.shape[:2])
        self.assertIn(np.unique(thresholded_image)[0], [0, 255])

    def test_find_contours(self):
        # Prueba la función de búsqueda de contornos
        binary_image = apply_otsu_threshold(self.test_image)
        contours = find_contours(binary_image)
        self.assertIsInstance(contours, list)

    def test_segment_leaf(self):
        # Prueba la función de segmentación de la hoja
        segmented_image, mask = segment_leaf(self.test_image)
        self.assertTrue(segmented_image is not None or mask is not None)
        

if __name__ == '__main__':
    unittest.main()
