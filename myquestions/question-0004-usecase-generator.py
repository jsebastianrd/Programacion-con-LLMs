import pandas as pd
import numpy as np
import random

from sklearn.cluster import KMeans

def generar_caso_de_uso_agrupar_pacientes():

    n = random.randint(15,30)

    df = pd.DataFrame({
        "edad": np.random.randint(20,80,size=n),
        "imc": np.random.uniform(18,35,size=n),
        "glucosa": np.random.randint(70,180,size=n)
    })

    n_clusters = random.randint(2,4)

    input_data = {
        "df": df.copy(),
        "n_clusters": n_clusters
    }

    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    labels = kmeans.fit_predict(df)

    output_data = labels

    return input_data, output_data
