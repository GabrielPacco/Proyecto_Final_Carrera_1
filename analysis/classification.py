from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class TomatoLeafClassifier:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        """
        Inicializa el clasificador con los parámetros dados.
        """
        self.classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

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

        # Opcional: Mostrar la matriz de confusión
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True)
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.show()

    def cross_validate(self, features, labels, cv=5):
        """
        Realiza una validación cruzada del modelo.
        """
        cv_scores = cross_val_score(self.classifier, features, labels, cv=cv)
        print(f"Cross-Validation Scores: {cv_scores}")
        print(f"Mean CV Score: {np.mean(cv_scores)}")

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
