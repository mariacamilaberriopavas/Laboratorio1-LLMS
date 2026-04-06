def generar_caso_de_uso_detectar_correlacion_inestable():
    import pandas as pd
    import numpy as np

    n_features    = np.random.randint(4, 8)
    n_semestres   = np.random.randint(2, 5)
    filas_por_sem = np.random.randint(40, 100, size=n_semestres)

    bloques = []
    base    = pd.Timestamp("2021-01-01")

    for i, n in enumerate(filas_por_sem):
        inicio = base + pd.DateOffset(months=6 * i)
        fechas = pd.date_range(inicio, periods=n, freq="D")
        X      = np.random.randn(n, n_features)
        coefs  = np.random.uniform(-1, 1, size=n_features)
        y      = X @ coefs + np.random.randn(n) * np.random.uniform(0.5, 2.0)
        mask   = np.random.random((n, n_features)) < 0.06
        X[mask] = np.nan
        bloque  = pd.DataFrame(X, columns=[f"var_{j}" for j in range(n_features)])
        bloque["target"] = y
        bloque["fecha"]  = fechas[:n]
        bloques.append(bloque)

    df = pd.concat(bloques, ignore_index=True)
    df = df.sample(frac=1, random_state=np.random.randint(0, 999)).reset_index(drop=True)

    umbral = round(float(np.random.uniform(0.2, 0.5)), 2)
    enunciado = {
        "titulo": "Detección de Correlación Inestable",
        "descripcion": "Identificar cambios en la correlación entre variables a lo largo del tiempo",
        "funcion": "detectar_correlacion_inestable",
        "argumentos": {
            "df": "DataFrame con variables numéricas, una columna de fecha y una variable objetivo",
            "fecha_col": "Nombre de la columna de fechas",
            "target_col": "Nombre de la variable objetivo",
            "umbral": "Valor mínimo de cambio en la correlación para considerarse inestable"
        },
        "retorno": (
            "DataFrame con periodos donde la correlación cambia significativamente "
            "según el umbral definido"
        ),
        "instrucciones": [
            "Convertir fecha_col a datetime",
            "Agrupar datos por periodos (por ejemplo semestres)",
            "Calcular correlaciones entre variables y target por periodo",
            "Comparar correlaciones entre periodos consecutivos",
            "Identificar cambios mayores al umbral",
            "Reportar variables con correlación inestable"
        ]
    }

    datos = (df, "fecha", "target", umbral)

    return enunciado, datos 
