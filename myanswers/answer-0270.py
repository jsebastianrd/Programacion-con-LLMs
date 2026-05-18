import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer

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

    df_processed = df.copy() # Trabajar en una copia para no alterar el original

    # 1. Codificación de la columna event_col a booleana
    df_processed[event_col] = df_processed[event_col].astype(bool)

    # Identificar columnas numéricas para transformación
    # Excluir las columnas de evento y duración de las características a transformar
    numeric_features = df_processed.select_dtypes(include=np.number).columns.tolist()
    if duration_col in numeric_features:
        numeric_features.remove(duration_col)
    # Asegurarse de que event_col no esté en numeric_features si fuera numérica (ej. 0/1)
    if event_col in numeric_features:
        numeric_features.remove(event_col)

    # Crear el ColumnTransformer
    # Aplicar SimpleImputer (mediana) y RobustScaler a las características numéricas
    # 'passthrough' para las columnas no transformadas (incluyendo event_col y duration_col)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numeric_features) # Aplicar RobustScaler a numéricas
        ],
        remainder='passthrough' # Mantener las columnas restantes (incluidas event_col, duration_col y otras no numéricas)
    )

    # Ajustar y transformar los datos
    # El ColumnTransformer convierte a numpy array, necesitamos reconstruir el DataFrame
    transformed_data = preprocessor.fit_transform(df_processed)

    # Obtener nombres de las columnas después de la transformación
    # Columnas numéricas escaladas
    transformed_numeric_cols = [f"num__{col}" for col in numeric_features] # Nombres generados por ColumnTransformer
    # Columnas passthrough
    passthrough_cols = [col for col in df_processed.columns if col not in numeric_features]

    # Reconstruir el DataFrame con los nombres correctos de las columnas
    # El orden de las columnas en transformed_data será: numéricas transformadas + remainder
    transformed_df = pd.DataFrame(transformed_data, columns=transformed_numeric_cols + passthrough_cols, index=df_processed.index)

    # 3. Manejo de Nulos con SimpleImputer (mediana) para las características originales
    # Realizamos la imputación *antes* del escalado, pero dentro de un pipeline conceptual
    # Ya que ColumnTransformer no maneja la imputación y escalado en la misma etapa
    # Lo hacemos de forma secuencial con un imputer separado para cada paso, 
    # o podríamos haber usado Pipeline dentro de ColumnTransformer, 
    # pero para simplicidad y cumplir la restricción, lo hacemos así.
    # El RobustScaler no maneja NaNs, por lo que la imputación debe ir antes.
    # La imputación con SimpleImputer (mediana) debe ser aplicada a las columnas originales
    # antes de ser escaladas. El ColumnTransformer para el escalado ya las procesó.
    # La consigna dice 'Impute los valores faltantes usando la mediana mediante SimpleImputer'
    # pero al tener el ColumnTransformer, la imputación de NaNs se haría primero
    # y luego el escalado.
    
    # Reconstrucción del pipeline de preprocesamiento, incluyendo imputación y escalado
    # Dado que ColumnTransformer convierte a array, la imputación se puede hacer como un paso previo
    # al fit_transform del ColumnTransformer, o dentro de un pipeline de ColumnTransformer.
    # Para mantener la simplicidad y cumplir con la restricción de ColumnTransformer
    # haremos la imputación primero.

    imputer = SimpleImputer(strategy='median')
    df_imputed = df.copy() # Imputamos sobre la copia original
    df_imputed[numeric_features] = imputer.fit_transform(df_imputed[numeric_features])

    # Ahora, aplicamos el ColumnTransformer para el escalado sobre el df_imputed
    # Tenemos que redefinir el preprocessor porque el anterior ya hizo fit_transform
    preprocessor_scaled = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numeric_features)
        ],
        remainder='passthrough'
    )

    transformed_data_final = preprocessor_scaled.fit_transform(df_imputed)

    # Reconstruir el DataFrame final con los nombres de columna correctos
    # La ColumnTransformer cambiará el prefijo de las columnas transformadas a 'num__'
    transformed_cols = []
    for t_name, t_transformer, t_cols in preprocessor_scaled.transformers_:
        if t_name == 'num': # Esto es para el RobustScaler
            transformed_cols.extend([f"{t_name}__{col}" for col in t_cols])
        else:
            # Para el remainder, necesitamos las columnas originales no transformadas
            remaining_cols = [col for col in df_imputed.columns if col not in numeric_features]
            transformed_cols.extend(remaining_cols)

    df_final = pd.DataFrame(transformed_data_final, columns=transformed_cols, index=df.index)

    # Asegurarse de que event_col y duration_col estén en su formato original y en el lugar correcto
    df_final[event_col] = df_processed[event_col] # Volvemos a asignar la columna booleana
    df_final[duration_col] = df_processed[duration_col]

    # Reorganizar las columnas para que las originales estén al final si es necesario, 
    # o mantener el orden generado por ColumnTransformer. El requisito es devolver un solo DataFrame transformado
    # que contenga las características escaladas e imputadas, y las columnas de evento y duración intactas.
    # Aseguramos que las columnas de evento y duración no sean "escaladas" o "imputadas" si no deben.

    # El ColumnTransformer pone las columnas transformadas primero, luego el `remainder`
    # Vamos a reordenar las columnas para que se parezca al output esperado del generador, 
    # es decir, las columnas de características transformadas primero, y luego las de evento y duración.
    final_columns_order = [col for col in df_final.columns if col not in [event_col, duration_col]] + [event_col, duration_col]
    df_final = df_final[final_columns_order]
    
    return df_final
