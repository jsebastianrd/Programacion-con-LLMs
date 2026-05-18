import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def prepare_survival_data(df, event_col, duration_col):
    """
    Prepara los datos para un modelo de Cox Proportional Hazards.

    Argumentos:
    df (pandas.DataFrame): DataFrame con datos clínicos y de seguimiento.
    event_col (str): Nombre de la columna de evento.
    duration_col (str): Nombre de la columna de tiempo.

    Devuelve:
    pandas.DataFrame: DataFrame transformado con características escaladas e imputadas,
                      y columnas de evento y duración intactas.
    """

    df_processed = df.copy()

    # 1. Codificación de la columna event_col a booleana
    df_processed[event_col] = df_processed[event_col].astype(bool)

    # Identificar características numéricas a transformar
    # Estas son todas las columnas numéricas EXCEPTO event_col y duration_col
    numeric_features = df_processed.select_dtypes(include=np.number).columns.tolist()
    # Remove event_col and duration_col from features to be transformed
    if event_col in numeric_features:
        numeric_features.remove(event_col)
    if duration_col in numeric_features:
        numeric_features.remove(duration_col)

    # Crear un pipeline para las características numéricas: imputación y luego escalado
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    # Crear el ColumnTransformer
    # Solo aplicamos la transformación a las 'numeric_features'.
    # Las columnas event_col y duration_col NO serán pasadas a través del ColumnTransformer
    # en este paso, sino que se reinsertarán más tarde.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ],
        remainder='drop' # Explicitly drop other columns, as we will re-add them
    )

    # Aplicar las transformaciones solo a las columnas numéricas seleccionadas
    transformed_numeric_data = preprocessor.fit_transform(df_processed[numeric_features])

    # Obtener los nombres de las columnas transformadas (sin prefijos)
    transformed_column_names = [col.replace('num__', '') for col in preprocessor.get_feature_names_out()]

    # Reconstruir el DataFrame de solo las características transformadas
    df_transformed_features = pd.DataFrame(
        transformed_numeric_data,
        columns=transformed_column_names,
        index=df_processed.index
    )

    # Combinar las características transformadas con las columnas de evento y duración originales
    # Aseguramos que event_col y duration_col mantengan sus dtypes originales (bool para event_col)
    df_final = pd.concat([
        df_transformed_features,
        df_processed[[event_col, duration_col]]
    ], axis=1)

    # Asegurar el orden final de las columnas para que coincida con el generador:
    # características transformadas, luego event_col, luego duration_col.
    final_cols_order = transformed_column_names + [event_col, duration_col]
    df_final = df_final[final_cols_order]

    return df_final
