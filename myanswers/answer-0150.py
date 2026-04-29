import pandas as pd

def crear_lag_features(df, columna, n_lags):
    df_result = df.copy()
    for i in range(1, n_lags + 1):
        df_result[f"{columna}_lag_{i}"] = df_result[columna].shift(i)
    df_result = df_result.dropna()
    return df_result
