import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_limpiar_datos_glucosa():

    n = random.randint(10,20)

    df = pd.DataFrame({
        "paciente": np.arange(n),
        "glucosa": np.random.randint(-10,200,size=n).astype(float),
        "colesterol": np.random.randint(100,300,size=n).astype(float)
    })

    mask = np.random.choice([True,False], size=n, p=[0.2,0.8])
    df.loc[mask,"glucosa"] = np.nan

    input_data = {"df": df.copy()}

    output_data = df.dropna()
    output_data = output_data[output_data["glucosa"] > 0]

    return input_data, output_data
