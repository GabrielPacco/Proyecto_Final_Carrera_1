import unittest
import cv2
import numpy as np
from preprocessing import resize_image, convert_to_lab, apply_gaussian_blur

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Crear una imagen de prueba (puede ser una imagen negra, blanca o con patrones)
        self.test_image = np.random.rand(100, 100, 3) * 255
        self.test_image = self.test_image.astype(np.uint8)

    def test_resize_image(self):
        # Prueba la función de redimensionamiento de imagen
        target_size = (50, 50)
        resized_image = resize_image(self.test_image, target_size)
        self.assertEqual(resized_image.shape[:2], target_size)

    def test_convert_to_lab(self):
        # Prueba la función de conversión a espacio de color L*a*b*
        lab_image = convert_to_lab(self.test_image)
        self.assertEqual(lab_image.shape, self.test_image.shape)

    def test_apply_gaussian_blur(self):
        # Prueba la función de desenfoque gaussiano
        kernel_size = (5, 5)
        blurred_image = apply_gaussian_blur(self.test_image, kernel_size)
        self.assertEqual(blurred_image.shape, self.test_image.shape)

    # Puedes añadir más pruebas para otras funciones de preprocesamiento aquí

if __name__ == '__main__':
    unittest.main()
