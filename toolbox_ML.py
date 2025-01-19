"""
Aquí tienen que ir todos los imports y cada una de las funciones que indica el enunciado.
Por favor, cambiad el '-' por '+' para las funciones que vayáis completando.
    + describe_df
    - tipifica_variables
    + get_features_num_regression
    - plot_features_num_regression
    - get_features_cat_regression
    - plot_features_cat_regression
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def describe_df(df):
    # Creamons un diccionario para almacenar la información
    data = {
        'DATA_TYPE': df.dtypes,
        'MISSINGS(%)': df.isnull().mean() * 100,
        'UNIQUE_VALUES': df.nunique(),
        'CARDIN(%)': df.nunique() / len(df) * 100
    }

    # Creamos un nuevo DataFrame con la información recopilada, usamos 'transpose' para cambiar
    # las filas por columnas.
    estudiantes_df = pd.DataFrame(data).transpose()

    return estudiantes_df


def get_features_num_regression(df, target_col, umbral_corr, pvalue=None):
    # Comprobaciones de los argumentos de entrada
    if not isinstance(df, pd.DataFrame):
        # comprueba que el primer argumento sea un DataFrame
        print("El primer argumento debe ser un DataFrame.")
        return None
    if target_col not in df.columns:
        # comprueba que la columna target exista en el DataFrame
        print(f"La columna '{target_col}' no existe en el DataFrame.")
        return None
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        # comprueba que la columna target sea numérica
        print(f"La columna '{target_col}' no es numérica.")
        return None
    if not (0 <= umbral_corr <= 1):
        # comprueba que el umbral de correlación esté entre 0 y 1
        print("El umbral de correlación debe estar entre 0 y 1.")
        return None
    if pvalue is not None and not (0 <= pvalue <= 1):
        # comprueba que el valor de pvalue esté entre 0 y 1
        print("El valor de pvalue debe estar entre 0 y 1.")
        return None

        # Calcular la correlación
    # 'abs' calcula el valor absoluto de las correlaciones.
    corr = df.corr()[target_col].abs()
    features = corr[corr > umbral_corr].index.tolist()
    # Eliminar la variable target de la lista de features, porque su valor de correlación es 1.
    features.remove(target_col)

    # Filtrar por pvalue si es necesario
    if pvalue is not None:
        significant_features = []
        for feature in features:
            # colocamos el guión bajo '_,' para indicar que no nos interesa el primer valor
            _, p_val = pearsonr(df[feature], df[target_col])
            if p_val < (1 - pvalue):
                significant_features.append(feature)
        features = significant_features

    return features
