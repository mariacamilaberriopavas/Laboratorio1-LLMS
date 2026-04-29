import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

def entrenar_pronostico_lasso(df, target_col, alpha_lasso):
    df = df.copy()
    df["ventas_ayer"] = df[target_col].shift(1)
    df = df.dropna()
    
    X = df[["ventas_ayer"]]
    y = df[target_col]
    
    modelo = Lasso(alpha=alpha_lasso)
    modelo.fit(X, y)
    
    y_pred = modelo.predict(X)
    r2 = r2_score(y, y_pred)
    
    return modelo, r2
