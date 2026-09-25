# Importar Pandas.
import pandas as pd

# Leer los datos del archivo CSV.
df = pd.read_csv("data.csv")

# Mostrar las estadsticas descriptivas
# de las columnas numericas.
print(df.describe())