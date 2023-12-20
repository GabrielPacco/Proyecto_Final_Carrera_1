import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# Supongamos que estas funciones y variables están definidas en otro lugar de tu proyecto:
# X_test, y_test, model, features

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Matriz de Confusión")
    plt.ylabel('Verdaderos')
    plt.xlabel('Predicciones')
    plt.show()

def plot_feature_importance(model, X):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure()
    plt.title("Importancia de las Características")
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), indices)
    plt.show()

def plot_clusters(features):
    kmeans = KMeans(n_clusters=3)
    clusters = kmeans.fit_predict(features)
    plt.scatter(features[:, 0], features[:, 1], c=clusters, cmap='viridis')
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.title('Visualización de Clusters')
    plt.show()

def plot_roc_curve(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (área = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.show()

# Ejemplos de cómo utilizar estas funciones:
# Asegúrate de tener los datos de X_test, y_test, y tu modelo cargados antes de ejecutar estas funciones

# plot_confusion_matrix(y_test, model.predict(X_test))
# plot_feature_importance(model, X_test)
# plot_clusters(features)
# plot_roc_curve(y_test, model.predict(X_test))
