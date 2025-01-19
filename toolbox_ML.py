"""
Aquí tienen que ir todos los imports y cada una de las funciones que indica el enunciado.
Por favor, cambiad el '-' por '+' para las funciones que vayáis completando.
    + describe_df
    - tipifica_variables
    + get_features_num_regression
    - plot_features_num_regression
    + get_features_cat_regression
    + plot_features_cat_regression
"""

import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

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
            _, p_val = stats.pearsonr(df[feature], df[target_col])
            if p_val < (1 - pvalue):
                significant_features.append(feature)
        features = significant_features

    return features

def get_features_cat_regression(df, target_col, pvalue=0.05): 
    """
    Función para obtener las características categóricas significativas en un modelo de regresión lineal.
    
    Params:
		df: dataframe de pandas
		target_col: columna objetivo del dataframe
		pvalue: p-valor para el test de significancia
    
    Returns:
		Lista con las características categóricas significativas
	"""

    # Verificamos si el dataframe es válido
    if not isinstance(df, pd.DataFrame):
        print("El argumento 'df' no es un dataframe válido.")
        return None
    
    # Verificamos si 'target_col' es una columna válida en el dataframe
    if target_col not in df.columns:
        print(f"La columna '{target_col}' no está en el dataframe.")
        return None
    
    # Verificamos si la columna 'target_col' es numérica
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"La columna '{target_col}' no es una columna numérica.")
        return None
    
    # Identificar las columnas categóricas del dataframe
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not cat_columns:
        print("No se encontraron características categóricas en el dataframe.")
        return None

    # Lista para almacenar las columnas categóricas que superan el pvalor
    significant_cat_features = []

    for cat_col in cat_columns:
        # Si la columna categórica tiene más de un nivel (para que sea válida para el test)
        if df[cat_col].nunique() > 1:
            # Si la columna categórica tiene dos niveles, realizar test t de Student
            if df[cat_col].nunique() == 2:
                group1 = df[target_col][df[cat_col] == df[cat_col].unique()[0]]
                group2 = df[target_col][df[cat_col] == df[cat_col].unique()[1]]
                t_stat, p_val = stats.ttest_ind(group1, group2)
            else:
                # Realizar test ANOVA si hay más de dos niveles
                f_val, p_val = stats.f_oneway(*[df[target_col][df[cat_col] == level] for level in df[cat_col].unique()])
            
            # Comprobamos si el p-valor es menor que el p-valor especificado
            if p_val < pvalue:
                significant_cat_features.append(cat_col)
    
    # Si encontramos columnas significativas, las devolvemos
    if significant_cat_features:
        return significant_cat_features
    else:
        print("No se encontraron características categóricas significativas.")
        return None
    
def plot_features_cat_regression(df, target_col="", columns=[], pvalue=0.05, with_individual_plot=False):
    """
    Función para graficar histogramas agrupados de variables categóricas significativas.
    
    Params:
        df: dataframe de pandas
        target_col: columna objetivo del dataframe (variable numérica)
        columns: lista de columnas categóricas a evaluar (si está vacía, se usan todas las columnas categóricas)
        pvalue: p-valor para el test de significancia
        with_individual_plot: si es True, genera un gráfico individual por cada categoría
    
    Returns:
        Lista de columnas que cumplen con los criterios de significancia
    """
    
    # Verificamos si el dataframe es válido
    if not isinstance(df, pd.DataFrame):
        print("El argumento 'df' no es un dataframe válido.")
        return None
    
    # Verificamos si 'target_col' es una columna válida en el dataframe
    if target_col and target_col not in df.columns:
        print(f"La columna '{target_col}' no está en el dataframe.")
        return None
    
    # Verificamos si la columna 'target_col' es numérica
    if target_col and not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"La columna '{target_col}' no es una columna numérica.")
        return None
    
    # Identificar las columnas categóricas del dataframe
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not cat_columns:
        print("No se encontraron características categóricas en el dataframe.")
        return None

    # Si 'columns' está vacío, usamos todas las columnas categóricas
    if not columns:
        columns = cat_columns
    
    # Lista para almacenar las columnas significativas
    significant_cat_features = []

    # Verificamos la significancia de las columnas categóricas con respecto al 'target_col'
    for cat_col in columns:
        if cat_col not in cat_columns:
            print(f"La columna '{cat_col}' no es categórica o no existe en el dataframe.")
            continue
        
        if df[cat_col].nunique() > 1:
            if df[cat_col].nunique() == 2:
                group1 = df[target_col][df[cat_col] == df[cat_col].unique()[0]]
                group2 = df[target_col][df[cat_col] == df[cat_col].unique()[1]]
                t_stat, p_val = stats.ttest_ind(group1, group2)
            else:
                f_val, p_val = stats.f_oneway(*[df[target_col][df[cat_col] == level] for level in df[cat_col].unique()])
            
            # Comprobamos si el p-valor es menor que el p-valor especificado
            if p_val < pvalue:
                significant_cat_features.append(cat_col)
                
    # Si no hay columnas significativas
    if not significant_cat_features:
        print("No se encontraron características categóricas significativas.")
        return None

    # Graficar histogramas agrupados para las columnas significativas
    for cat_col in significant_cat_features:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=target_col, hue=cat_col, kde=True, multiple="stack", bins=30)
        plt.title(f"Distribución de {target_col} por {cat_col}")
        plt.xlabel(target_col)
        plt.ylabel("Frecuencia")
        plt.show()

        # Si 'with_individual_plot' es True, graficar histogramas individuales por cada categoría
        if with_individual_plot:
            for level in df[cat_col].unique():
                plt.figure(figsize=(10, 6))
                sns.histplot(df[df[cat_col] == level], x=target_col, kde=True, bins=30)
                plt.title(f"Distribución de {target_col} para {cat_col} = {level}")
                plt.xlabel(target_col)
                plt.ylabel("Frecuencia")
                plt.show()

    return significant_cat_features