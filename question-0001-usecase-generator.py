import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def generar_caso_de_uso_evaluar_deriva_mensual():
    # Componente aleatorio
    n_meses = np.random.randint(4, 9)          # entre 4 y 8 meses
    n_features = np.random.randint(2, 6)       # entre 2 y 5 features
    filas_por_mes = np.random.randint(15, 40)  # filas por mes
    nan_fraction = np.random.uniform(0.0, 0.10)
    random_seed = np.random.randint(0, 10000)
    np.random.seed(random_seed)

    # Nombres de columnas
    feature_names = [f"var_{i}" for i in range(n_features)]
    target_col = "demanda"
    fecha_col = "fecha"

    # Generar fechas mensuales
    fecha_inicio = pd.Timestamp(
        year=np.random.randint(2020, 2024),
        month=np.random.randint(1, 10),
        day=1
    )
    periodos = pd.date_range(start=fecha_inicio, periods=n_meses, freq="MS")

    filas_lista = []
    for periodo in periodos:
        n_filas = np.random.randint(filas_por_mes, filas_por_mes + 20)
        fechas_mes = [
            periodo + pd.Timedelta(days=int(d))
            for d in np.random.randint(0, 28, size=n_filas)
        ]
        X_mes = np.random.randn(n_filas, n_features)
        y_mes = np.random.randn(n_filas)

        df_mes = pd.DataFrame(X_mes, columns=feature_names)
        df_mes[target_col] = y_mes
        df_mes[fecha_col] = fechas_mes
        filas_lista.append(df_mes)

    df = pd.concat(filas_lista, ignore_index=True)

    # Inyectar NaNs
    for col in feature_names:
        mask = np.random.rand(len(df)) < nan_fraction
        df.loc[mask, col] = np.nan

    # --- INPUT ---
    input_dict = {
        "df": df.copy(),
        "target_col": target_col,
        "fecha_col": fecha_col
    }

    # --- OUTPUT esperado (replica la lógica de evaluar_deriva_mensual) ---
    df_work = df.copy()
    df_work[fecha_col] = pd.to_datetime(df_work[fecha_col])
    df_work["periodo"] = df_work[fecha_col].dt.to_period("M")

    periodos_ordenados = sorted(df_work["periodo"].unique())

    resultados = []

    for i in range(1, len(periodos_ordenados)):
        # Datos de entrenamiento acumulados
        periodos_train = periodos_ordenados[:i]
        periodos_eval = periodos_ordenados[i]

        df_train = df_work[df_work["periodo"].isin(periodos_train)]
        df_eval = df_work[df_work["periodo"] == periodos_eval]

        # Omitir si no hay suficientes datos
        if len(df_eval) < 2:
            continue

        X_train = df_train[feature_names].values
        y_train = df_train[target_col].values
        X_eval = df_eval[feature_names].values
        y_eval = df_eval[target_col].values

        # Imputar
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_eval_imp = imputer.transform(X_eval)

        # Escalar
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_imp)
        X_eval_sc = scaler.transform(X_eval_imp)

        # Modelo
        model = Ridge(alpha=1.0)
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_eval_sc)

        rmse = np.sqrt(mean_squared_error(y_eval, y_pred))

        resultados.append({
            "periodo_evaluacion": str(periodos_eval),
            "n_meses_entrenamiento": i,
            "rmse": rmse
        })

    output = pd.DataFrame(resultados).reset_index(drop=True)

    return input_dict, output


# --- Ejemplo de uso ---
if __name__ == "__main__":
    input_data, expected_output = generar_caso_de_uso_evaluar_deriva_mensual()

    print("=== INPUT ===")
    print(f"DataFrame shape:  {input_data['df'].shape}")
    print(f"target_col:       '{input_data['target_col']}'")
    print(f"fecha_col:        '{input_data['fecha_col']}'")
    print(f"NaNs totales:     {input_data['df'].isnull().sum().sum()}")
    print()
    print("=== OUTPUT ESPERADO ===")
    print(expected_output)
