import pandas as pd
import numpy as np

def generar_caso_de_uso_evaluar_importancia_temporal():
    np.random.seed(None)  # aleatoriedad real

    # 1. Parámetros aleatorios
    n_filas = np.random.randint(100, 500)
    n_features = np.random.randint(3, 8)
    n_anios = np.random.randint(2, 5)  # puede generar casos inválidos (<3 años)

    # 2. Generar años
    anios = np.random.choice(range(2018, 2025), size=n_anios, replace=False)

    # 3. Crear fechas aleatorias
    fechas = []
    for _ in range(n_filas):
        anio = np.random.choice(anios)
        mes = np.random.randint(1, 13)
        dia = np.random.randint(1, 28)
        fechas.append(f"{anio}-{mes:02d}-{dia:02d}")

    df = pd.DataFrame({
        "fecha": fechas
    })

    # 4. Crear features numéricas
    for i in range(n_features):
        df[f"feature_{i}"] = np.random.randn(n_filas) * np.random.randint(1, 10)

    # 5. Crear target con dependencia real (importante)
    pesos = np.random.uniform(0.5, 2, size=n_features)
    ruido = np.random.randn(n_filas)

    df["target"] = sum(
        df[f"feature_{i}"] * pesos[i] for i in range(n_features)
    ) + ruido

    # 6. Introducir valores faltantes aleatorios
    for col in df.columns:
        if "feature" in col:
            mask = np.random.rand(n_filas) < 0.1
            df.loc[mask, col] = np.nan

    #enunciado
    enunciado = (
        "Dado un DataFrame con una columna de fechas ('fecha'), varias variables "
        "numéricas ('feature_i') y una variable objetivo ('target'), construya una "
        "función que evalúe la importancia de las variables a lo largo del tiempo, "
        "agrupando por año."
    )

    datos = (df, "target", "fecha")
    return enunciado, datos
