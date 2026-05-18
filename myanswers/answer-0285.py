import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def detectar_anomalias_financieras(df, proporcion_anomalias):
    """
    Detecta anomalías en transacciones bancarias utilizando Isolation Forest.

    Argumentos:
    df (pandas.DataFrame): DataFrame de transacciones.
    proporcion_anomalias (float): Porcentaje estimado de anomalías (0 a 0.5).

    Devuelve:
    numpy.ndarray: Array con 1 (normal) o -1 (anomalía) para cada fila limpia.
    """
    # Requisito 1: Eliminar filas con valores nulos
    df_limpio = df.dropna().copy()

    # Requisito 2: Configurar y entrenar el modelo Isolation Forest
    # Se usa random_state=42 para reproducibilidad, crucial para coincidir con el generador.
    modelo = IsolationForest(
        contamination=proporcion_anomalias,
        random_state=42
    )
    modelo.fit(df_limpio) # Entrenar el modelo

    # Requisito 3: Predecir y devolver las etiquetas (1 para normal, -1 para anomalía)
    predicciones = modelo.predict(df_limpio)

    return predicciones
