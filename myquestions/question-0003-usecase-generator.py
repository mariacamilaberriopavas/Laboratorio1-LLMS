def generar_caso_de_uso_evaluar_clusters_por_periodo():
    import pandas as pd
    import numpy as np

    n_trimestres        = np.random.randint(3, 7)
    n_features          = np.random.randint(3, 6)
    filas_por_trimestre = np.random.randint(30, 80, size=n_trimestres)

    bloques = []
    base    = pd.Timestamp("2022-01-01")

    for i, n in enumerate(filas_por_trimestre):
        inicio = base + pd.DateOffset(months=3 * i)
        fechas = pd.date_range(inicio, periods=n, freq="D")
        bloque = np.random.randn(n, n_features)
        mask   = np.random.random((n, n_features)) < 0.05
        bloque[mask] = np.nan

        df_trim = pd.DataFrame(
            bloque,
            columns=[f"feat_{j}" for j in range(n_features)]
        )
        df_trim["fecha"] = fechas[:n]
        bloques.append(df_trim)

    df = pd.concat(bloques, ignore_index=True)
    df = df.sample(frac=1, random_state=np.random.randint(0, 1000)).reset_index(drop=True)

    k_min = int(np.random.choice([2, 3]))
    k_max = int(np.random.choice([4, 5, 6]))

    
    enunciado = {
        "titulo": "Segmentación Trimestral de Clientes",
        "descripcion": "Evaluar clustering por periodos de tiempo",
        "funcion": "evaluar_clusters_por_periodo",
        "argumentos": {
            "df": "DataFrame con variables numéricas y una columna de fechas",
            "fecha_col": "Nombre de la columna de fechas",
            "k_min": "Número mínimo de clusters",
            "k_max": "Número máximo de clusters"
        },
        "retorno": (
            "DataFrame con columnas ['trimestre', 'mejor_k', 'silhouette_score'] "
            "ordenado cronológicamente"
        ),
        "instrucciones": [
            "Convertir fecha_col a datetime",
            "Extraer trimestre (YYYY-QX)",
            "Agrupar por trimestre",
            "Seleccionar variables numéricas",
            "Imputar valores faltantes con la media",
            "Escalar con StandardScaler",
            "Probar k entre k_min y k_max con KMeans",
            "Calcular silhouette_score",
            "Elegir mejor k (empate -> menor k)",
            "Omitir trimestres con pocos datos"
        ]
    }

    datos = (df, "fecha", k_min, k_max)

    return enunciado, datos
