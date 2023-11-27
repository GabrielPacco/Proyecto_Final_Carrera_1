import unittest
from analysis.classification import TomatoLeafClassifier
import numpy as np

class TestTomatoLeafClassifier(unittest.TestCase):

    def setUp(self):
        # Inicializa el clasificador para las pruebas
        self.classifier = TomatoLeafClassifier()
        # Generar algunos datos de prueba 
        self.test_features = np.random.rand(10, 4)  # 10 muestras, 4 características cada una
        self.test_labels = np.random.randint(0, 2, 10)  # Etiquetas binarias para las pruebas

    def test_train(self):
        # Prueba el método de entrenamiento
        self.classifier.train(self.test_features, self.test_labels)
        self.assertIsNotNone(self.classifier.classifier)

    def test_predict(self):
        # Prueba el método de predicción
        self.classifier.train(self.test_features, self.test_labels)
        predictions = self.classifier.predict(self.test_features[0])
        self.assertIsNotNone(predictions)

    def test_save_and_load_model(self):
        # Prueba guardar y cargar el modelo
        model_path = "test_model.pkl"
        self.classifier.train(self.test_features, self.test_labels)
        self.classifier.save_model(model_path)
        self.classifier.load_model(model_path)
        predictions = self.classifier.predict(self.test_features[0])
        self.assertIsNotNone(predictions)

if __name__ == '__main__':
    unittest.main()
