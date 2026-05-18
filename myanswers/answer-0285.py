import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def limpiar_duplicados_y_estandarizar(df):
    """
    Elimina filas duplicadas, selecciona columnas numéricas
    y las estandariza usando StandardScaler.
    """

    # Eliminar duplicados basándose en todas las columnas y reiniciar índice
    # Esto coincide con el comportamiento del generador (no se puede modificar)
    df_limpio = df.drop_duplicates().reset_index(drop=True)

    # Seleccionar columnas numéricas del DataFrame ya limpio
    df_numerico = df_limpio.select_dtypes(include=[np.number]).copy()

    # Estandarizar columnas numéricas
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(df_numerico)

    # Construir DataFrame final
    df_escalado = pd.DataFrame(
        datos_escalados,
        columns=df_numerico.columns
    ).reset_index(drop=True)

    return df_escalado
