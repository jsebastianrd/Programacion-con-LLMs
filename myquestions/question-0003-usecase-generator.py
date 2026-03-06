import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def generar_caso_de_uso_entrenar_modelo_riesgo():

    n = random.randint(20,40)

    df = pd.DataFrame({
        "edad": np.random.randint(30,80,size=n),
        "colesterol": np.random.randint(150,300,size=n),
        "presion": np.random.randint(90,160,size=n),
        "target": np.random.randint(0,2,size=n)
    })

    target_col = "target"

    input_data = {
        "df": df.copy(),
        "target_col": target_col
    }

    X = df.drop(columns=[target_col])
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled,y)

    preds = model.predict(X_scaled)
    acc = accuracy_score(y,preds)

    output_data = acc

    return input_data, output_data
