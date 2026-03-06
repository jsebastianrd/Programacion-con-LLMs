import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_calcular_promedios_signos():

    n_rows = random.randint(10,20)

    df = pd.DataFrame({
        "patient_id": np.random.randint(1,5,size=n_rows),
        "presion_sistolica": np.random.randint(100,160,size=n_rows),
        "presion_diastolica": np.random.randint(60,100,size=n_rows),
        "frecuencia_cardiaca": np.random.randint(60,110,size=n_rows)
    })

    input_data = {"df": df.copy()}

    output_data = (
        df.groupby("patient_id")
        .mean()
        .reset_index()
    )

    return input_data, output_data
