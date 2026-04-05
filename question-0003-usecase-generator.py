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

    return df, "fecha", k_min, k_max
