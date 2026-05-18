import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def detectar_anomalias_mahalanobis(
    datos: pd.DataFrame,
    n_componentes: int,
    umbral: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detecta anomalías en datos numéricos multivariados utilizando PCA y distancia de Mahalanobis.

    Argumentos:
    datos (pd.DataFrame): DataFrame que contiene únicamente variables numéricas.
    n_componentes (int): Número de componentes principales a utilizar en PCA.
    umbral (float): Valor umbral sobre la distancia de Mahalanobis para clasificar anomalías.

    Devuelve:
    tuple[np.ndarray, np.ndarray]: Una tupla con:
        - Un array de numpy con las etiquetas de anomalía (0 o 1).
        - Un array de numpy con las distancias de Mahalanobis calculadas para cada observación.
    """

    # 1. Ajustar un modelo PCA y transformar los datos
    pca = PCA(n_components=n_componentes)
    X_reducido = pca.fit_transform(datos) # Transformar los datos al espacio reducido

    # 2. Calcular el vector de medias y la matriz de covarianza de los datos proyectados
    media = np.mean(X_reducido, axis=0)
    cov = np.cov(X_reducido, rowvar=False)

    # 3. Calcular la inversa o pseudo-inversa de la matriz de covarianza
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    # 4. Calcular la distancia de Mahalanobis para cada observación
    distancias = []
    for x in X_reducido:
        diff = x - media
        # Calculo manual de la distancia de Mahalanobis: sqrt((x - mu).T @ cov_inv @ (x - mu))
        d = np.sqrt(diff.T @ cov_inv @ diff)
        distancias.append(d)

    distancias = np.array(distancias)

    # 5. Clasificar anomalías (1 si distancia > umbral, 0 en caso contrario)
    etiquetas = (distancias > umbral).astype(int)

    # 6. Devolver etiquetas y distancias
    return etiquetas, distancias
