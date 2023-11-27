from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import numpy as np

# Suponiendo que 'features' es un array de NumPy con las características extraídas y
# 'labels' es un array con las etiquetas correspondientes (sano, enfermo, tipo de enfermedad, etc.)

class TomatoLeafClassifier:
    def __init__(self, n_estimators=100, random_state=42):
        """
        Inicializa el clasificador con los parámetros dados.
        """
        self.classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def train(self, features, labels):
        """
        Entrena el clasificador con las características y etiquetas proporcionadas.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
        self.classifier.fit(self.X_train, self.y_train)

    def evaluate(self):
        """
        Evalúa el rendimiento del clasificador en el conjunto de prueba.
        """
        y_pred = self.classifier.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred)}")

    def predict(self, features):
        """
        Predice las etiquetas para las características dadas.
        """
        return self.classifier.predict([features])

    def save_model(self, file_path):
        """
        Guarda el modelo entrenado en el archivo especificado.
        """
        joblib.dump(self.classifier, file_path)

    def load_model(self, file_path):
        """
        Carga un modelo entrenado desde el archivo especificado.
        """
        self.classifier = joblib.load(file_path)
